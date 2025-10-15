"""
XPS table processing utilities.

This module contains functions for processing and structuring XPS data tables,
including header cleaning, row grouping, and data conversion.
"""

import numpy as np


def clean_header(header_line):
    """Normalize a header line into a cleaned ``header`` and the original ``raw_header``.

    Common cleaning steps:
    - split on tabs and strip whitespace,
    - remove a duplicated ``Name`` column if present (many XPS exports repeat it).

    Parameters
    ----------
    header_line : str
        The raw header line from the file.

    Returns
    -------
    tuple[list[str], list[str]]
        (cleaned_header, raw_header)
    """
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


def group_rows_by_name(table_data, name_idx):
    """Group table rows into sub-tables when a repeating name indicates a new group.

    Many XPS report tables list multiple fit parameters for a component and then
    repeat the component label for the next component. This function splits
    ``table_data`` into groups where each group belongs to a single logical
    entry.

    Parameters
    ----------
    table_data : list[list[str]]
        Parsed table rows (each row is a list of string fields).
    name_idx : int
        Column index used to detect repeated names (e.g. index of "Comp Label").

    Returns
    -------
    list[list[list[str]]]
        A list of groups; each group is a list of rows (rows are lists of strings).
    """
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


def table_to_dict(groups, header):
    """Convert grouped table rows into a dict of 2D NumPy arrays.

    Each header becomes a key mapped to a 2D object array where columns
    represent distinct grouped entries (e.g. chemical components) and rows
    correspond to fit parameters.

    The parser handles numeric fields, comma-separated numeric lists ("x, y"),
    and leaves non-numeric entries as strings.

    Parameters
    ----------
    groups : list[list[list[str]]]
        Output from ``group_rows_by_name``; groups of raw string rows.
    header : list[str]
        Cleaned header column names.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of header -> 2D NumPy array (dtype object) with shape
        (n_parameters, n_components).
    """

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


def parse_report_rows(lines, header_idx, header, raw_header):
    """Parse the rows of a report table into a list of cleaned rows.

    This helper reads lines following ``header_idx`` until the first empty
    line. If the raw header contained a duplicated ``Name`` column, the
    corresponding duplicate value is removed from parsed rows so the row length
    matches the cleaned ``header``.

    Parameters
    ----------
    lines : list[str]
        File content lines.
    header_idx : int
        Index of the header line.
    header : list[str]
        Cleaned header produced by ``clean_header``.
    raw_header : list[str]
        Original header tokens before cleaning.

    Returns
    -------
    list[list[str]]
        Parsed, cleaned rows (lists of field strings).
    """
    rows = []
    name_indices = [i for i, h in enumerate(raw_header) if h == "Name"]
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        row = [v.strip() for v in line.split("\t") if v.strip()]
        # If an extra 'Name' column exists in raw data, remove the duplicate entry
        if (
            len(name_indices) > 1
            and len(row) > len(header)
            and len(row) > name_indices[1]
        ):
            del row[name_indices[1]]
        if row:
            rows.append(row)
    return rows
