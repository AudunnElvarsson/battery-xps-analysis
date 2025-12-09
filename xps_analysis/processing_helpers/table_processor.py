"""XPS table processing utilities.

This module contains functions for processing and structuring XPS data tables,
including header cleaning, row grouping, and data conversion.

Public Functions
----------------
clean_header : Normalize a header line into cleaned and raw headers
group_rows_by_name : Group table rows by repeating name column
group_rows_by_dataset : Group table rows by Data Set column (multi-core format)
table_to_dict : Convert grouped rows to dictionary format
parse_report_rows : Parse report data rows into structured format
has_dataset_column : Check if header contains 'Data Set' column
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


def has_dataset_column(header):
    """Check if header contains 'Data Set' or 'Iteration' column indicating multi-core format.

    Parameters
    ----------
    header : list[str]
        Cleaned header column names.

    Returns
    -------
    bool
        True if 'Data Set' or 'Iteration' column is present, False otherwise.
    """
    return "Data Set" in header or "Iteration" in header


def label_spin_orbit_doublets(core_data):
    """Add spin-orbit suffixes to duplicate component names.

    For components with identical names (e.g., P-F appearing twice for P 2p),
    this adds suffixes like " (3/2)" and " (1/2)" to distinguish the
    spin-orbit components. The first occurrence is the more intense 3/2 peak.

    Parameters
    ----------
    core_data : dict
        Core level data dictionary with 'Name' field.

    Returns
    -------
    dict
        Modified core_data with updated Name field and new 'Doublet Group' field.
    """
    if "Name" not in core_data:
        return core_data

    name_array = np.array(core_data["Name"], dtype=object)
    if name_array.size == 0:
        return core_data

    # Extract component names (handle 2D format)
    if name_array.ndim == 2:
        names = [str(name_array[i, 0]) for i in range(name_array.shape[0])]
    else:
        names = [str(name_array[i]) for i in range(name_array.shape[0])]

    # Find duplicates and add suffixes
    name_counts = {}
    doublet_groups = []  # Track which components belong to same doublet

    for i, name in enumerate(names):
        if name in name_counts:
            # This is a duplicate - mark both as doublet
            first_idx = name_counts[name]["indices"][0]
            occurrence = len(name_counts[name]["indices"])

            # Add suffix to first occurrence if not already done
            if name_counts[name]["count"] == 1:
                names[first_idx] = f"{name} (3/2)"
                doublet_groups.append(name)  # Store base name

            # Add suffix to current occurrence
            if occurrence == 1:
                names[i] = f"{name} (1/2)"
            else:
                # Handle more than 2 components with same name (unlikely)
                names[i] = f"{name} ({occurrence})"

            name_counts[name]["indices"].append(i)
            name_counts[name]["count"] += 1
        else:
            name_counts[name] = {"count": 1, "indices": [i]}

    # Update the Name array with suffixes
    if name_array.ndim == 2:
        for i, new_name in enumerate(names):
            name_array[i, 0] = new_name
    else:
        name_array = np.array(names, dtype=object)

    core_data["Name"] = name_array

    # Store doublet grouping information
    if doublet_groups:
        # Create array indicating which components belong to same doublet
        doublet_group_array = np.array([""] * len(names), dtype=object)
        for base_name in set(doublet_groups):
            for i, name in enumerate(names):
                if base_name in name and ("(3/2)" in name or "(1/2)" in name):
                    doublet_group_array[i] = base_name
        core_data["Doublet Group"] = doublet_group_array

    return core_data


def group_rows_by_dataset(table_data, dataset_idx, tag_idx):
    """Group table rows by Data Set/Iteration number and then by Tag (core level).

    This function handles the multi-core-level format where measurements
    are identified by the "Data Set" or "Iteration" column and core levels
    by the "Tag" column.

    Parameters
    ----------
    table_data : list[list[str]]
        Parsed table rows (each row is a list of string fields).
    dataset_idx : int
        Column index of the "Data Set" field.
    tag_idx : int
        Column index of the "Tag" field (core level identifier).

    Returns
    -------
    dict[str, list[list[list[str]]]]
        Nested dict: {core_level: [groups]} where each group is a list of rows
        for one measurement of that core level.
    """
    # First, organize by data set and core level
    by_dataset_and_core = {}
    dataset_order = []  # Track order of datasets as they appear

    for row in table_data:
        dataset = row[dataset_idx] if row[dataset_idx] else "current"
        core_level = row[tag_idx]

        # Track the order of datasets as they first appear
        if dataset not in dataset_order:
            dataset_order.append(dataset)

        key = (dataset, core_level)
        if key not in by_dataset_and_core:
            by_dataset_and_core[key] = []
        by_dataset_and_core[key].append(row)

    # Now organize by core level with groups for each measurement
    # Preserve the original order of datasets instead of sorting
    result = {}

    for core_level in set(k[1] for k in by_dataset_and_core):
        result[core_level] = []
        for dataset in dataset_order:  # Use original order, not sorted
            key = (dataset, core_level)
            if key in by_dataset_and_core:
                result[core_level].append(by_dataset_and_core[key])

    return result


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


def table_to_dict_exclude_columns(groups, header, exclude_indices):
    """Convert grouped table rows into a dict, excluding specified column indices.

    This variant of table_to_dict removes columns at the specified indices
    before processing, useful when grouping columns should not appear in output.

    Parameters
    ----------
    groups : list[list[list[str]]]
        Output from grouping functions; groups of raw string rows.
    header : list[str]
        Cleaned header column names.
    exclude_indices : list[int]
        Column indices to exclude from the result.

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

    # Build filtered header
    filtered_header = [h for idx, h in enumerate(header) if idx not in exclude_indices]

    result = {h: [] for h in filtered_header}
    for group in groups:
        arr = np.array(group, dtype=object)
        # Ensure arr has enough columns - pad with empty strings if needed
        if arr.ndim == 1:
            # Single row - reshape to 2D
            arr = arr.reshape(1, -1)
        if arr.shape[1] < len(header):
            # Pad with empty strings if there are missing columns
            padding = np.full(
                (arr.shape[0], len(header) - arr.shape[1]), "", dtype=object
            )
            arr = np.concatenate([arr, padding], axis=1)

        # Process each column, skipping excluded indices
        for orig_idx, h in enumerate(header):
            if orig_idx in exclude_indices:
                continue
            col = arr[:, orig_idx]
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

    For multi-core format (with "Data Set" or "Iteration" column), empty leading
    columns are preserved to maintain alignment, and filled with the most recent
    data set/iteration number.

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
    has_dataset = "Data Set" in header or "Iteration" in header
    current_dataset = None

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break

        # For multi-core format, preserve empty columns
        if has_dataset:
            row = [v.strip() for v in line.split("\t")]
            # Remove trailing empty columns to match header length
            while row and not row[-1]:
                row.pop()
            # Fill empty dataset column with current dataset number
            if row and not row[0]:
                row[0] = current_dataset if current_dataset is not None else ""
            elif row and row[0]:
                current_dataset = row[0]
        else:
            # Original behavior: strip empty strings
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
