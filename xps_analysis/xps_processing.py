"""
Module that reads and processes xps data from a file.
"""

import os
import numpy as np


def _find_header(lines, startswith_tuple, required_substring=None):
    """Return header line and its index from file, matching start and optional substring."""
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith_tuple):
            if required_substring is None or required_substring in line:
                return line, i
    return None, None


def _parse_data_rows(lines, header_idx, columns, skip_empty=True, break_on_empty=False):
    """Parse tab-separated data rows into dict of lists."""
    data_dict = {col: [] for col in columns}
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if break_on_empty:
                break
            if skip_empty:
                continue
        values = [v.strip() for v in line.split("\t")]
        if len(values) != len(columns):
            continue
        for col, val in zip(columns, values):
            try:
                data_dict[col].append(float(val))
            except ValueError:
                data_dict[col].append(val)
    return data_dict


def _convert_data_types(data_dict):
    """Convert lists to numpy arrays with appropriate dtype."""
    for col in data_dict:
        try:
            data_dict[col] = np.array(data_dict[col], dtype=float)
        except ValueError:
            data_dict[col] = np.array(data_dict[col], dtype=object)
    return data_dict


def read_fit_report_file(file_path):
    """
    Reads the first table in an XPS report file and parses it into a dictionary of numpy arrays.

    The dictionary keys are the table headers, and the values are 2D numpy arrays, where each column
    corresponds to a unique entry (e.g., a chemical species) and each row corresponds to a fit
    parameter.

    Parameters:
        file_path (str): Path to the report file to read.

    Returns:
        dict: Dictionary with headers as keys and 2D numpy arrays as values.
    """

    def _clean_header(header_line):
        """Remove duplicate 'Name' columns from header."""
        raw_header = [h.strip() for h in header_line.split("\t") if h.strip()]
        header = []
        name_seen = False
        for h in raw_header:
            if h == "Name":
                if name_seen:
                    continue
                name_seen = True
            header.append(h)
        return header, raw_header

    def _group_rows_by_name(table_data, name_idx):
        """Group rows by third dimension (when 'Comp Label' repeats)."""
        groups = []
        current_group = []
        unique_names = set()
        for row in table_data:
            name = row[name_idx]
            if name in unique_names:
                groups.append(current_group)
                current_group = []
                unique_names = set()
            current_group.append(row)
            unique_names.add(name)
        if current_group:
            groups.append(current_group)
        return groups

    def _table_to_dict(groups, header):
        """Convert grouped table rows to dictionary of numpy arrays."""

        def parse_value(val):
            # If value is of form 'x , y', convert to [x, y] as floats
            if "," in val:
                parts = [p.strip() for p in val.split(",")]
                try:
                    return [float(p) for p in parts]
                except ValueError:
                    return val
            try:
                return float(val)
            except ValueError:
                return val

        result = {h: [] for h in header}
        for group in groups:
            arr = np.array(group)
            for idx, h in enumerate(header):
                col = arr[:, idx]
                col_converted = [parse_value(v) for v in col]
                result[h].append(col_converted)
        for h in result:
            # Use dtype=object to allow arrays and floats
            result[h] = np.transpose(np.array(result[h], dtype=object))
        return result

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line and index
    header_line, header_idx = _find_header(
        lines, ("Name",), required_substring="Comp Label"
    )
    if header_idx is None:
        print("No table header found.")
        return None

    header, raw_header = _clean_header(header_line)
    header = ["Binding Energy (eV)" if h == "Position" else h for h in header]

    # Parse table rows (report-specific logic)
    table_data = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        row_raw = [v.strip() for v in line.split("\t") if v.strip()]
        if len(row_raw) > len(header):
            raw_name_indices = [i for i, h in enumerate(raw_header) if h == "Name"]
            if len(raw_name_indices) > 1 and len(row_raw) > raw_name_indices[1]:
                del row_raw[raw_name_indices[1]]
        if row_raw:
            table_data.append(row_raw)

    name_idx = header.index("Comp Label")
    groups = _group_rows_by_name(table_data, name_idx)
    result_dict = _table_to_dict(groups, header)
    return result_dict


def read_fit_spectrum_file(file_path):
    """
    Reads an XPS spectrum file and returns a dictionary with processed column names as keys and
    numpy arrays as values.

    Parameters:
        file_path (str): Path to the spectrum file.

    Returns:
        dict: Dictionary with processed column names as keys and numpy arrays of data as values.
    """

    def _rename_spectrum_keys(data_dict):
        """Rename keys for consistency and clarity."""
        renamed = {}
        for col, arr in data_dict.items():
            if col.startswith("Normalised_Residual"):
                new_key = "Normalised Residual"
            elif col.startswith("CPS"):
                new_key = "Measured"
            else:
                new_key = col.split("_", 1)[0] if "_" in col else col
            renamed[new_key] = arr
        return renamed

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line and index
    header_line, header_idx = _find_header(lines, ("KE_", "BE_", "CPS_"))
    if header_idx is None:
        raise ValueError("No data header found in file.")

    columns = [col.strip() for col in header_line.split("\t")]
    data_dict = _parse_data_rows(
        lines, header_idx, columns, skip_empty=True, break_on_empty=False
    )
    data_dict = _convert_data_types(data_dict)
    renamed_dict = _rename_spectrum_keys(data_dict)
    return renamed_dict
