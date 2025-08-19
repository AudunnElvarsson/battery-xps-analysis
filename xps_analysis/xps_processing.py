"""
Module that reads and processes xps data from a file.
"""

import os
import numpy as np


def _find_table_header(lines):
    """Return index of first table header line."""
    return next((i for i, line in enumerate(lines)
                if line.strip().startswith("Name") and "Comp Label" in line), None)

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

def _parse_table_rows(lines, header_idx, header, raw_header):
    """Parse table rows, removing duplicate 'Name' column data."""
    table_data = []
    for line in lines[header_idx+1:]:
        if not line.strip():
            break
        row_raw = [v.strip() for v in line.split("\t") if v.strip()]
        if len(row_raw) > len(header):
            raw_name_indices = [i for i, h in enumerate(raw_header) if h == "Name"]
            if len(raw_name_indices) > 1 and len(row_raw) > raw_name_indices[1]:
                del row_raw[raw_name_indices[1]]
        if row_raw:
            table_data.append(row_raw)
    return table_data

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

def read_report(file_path):
    """
    Reads the first table in an XPS report file and parses it into a dictionary of numpy arrays.

    The dictionary keys are the table headers, and the values are 2D numpy arrays, where each row
    corresponds to a unique entry (e.g., a chemical species) and each column corresponds to a header.
    The third dimension is split when the 'Name' field repeats, so each group represents a set of unique names.
    Duplicate 'Name' columns and their data are ignored.

    Parameters:
        file_path (str): Path to the report file to read.

    Returns:
        dict: Dictionary with headers as keys and 2D numpy arrays as values.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = _find_table_header(lines)
    if header_idx is None:
        print("No table header found.")
        return None

    header, raw_header = _clean_header(lines[header_idx])
    # Rename 'Position' header to 'Binding Energy (eV)'
    header = ["Binding Energy (eV)" if h == "Position" else h for h in header]
    table_data = _parse_table_rows(lines, header_idx, header, raw_header)
    name_idx = header.index("Comp Label")
    groups = _group_rows_by_name(table_data, name_idx)
    result_dict = _table_to_dict(groups, header)

    return result_dict
