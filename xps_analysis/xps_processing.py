"""xps_processing
-----------------

Utilities for reading and converting XPS report and spectrum text files into
Python structures suitable for analysis and plotting. Functions return
dictionary-like mappings where keys are column/header names and values are
numpy arrays (typically 1D or 2D arrays with dtype float or object).

The module focuses on robust parsing of tab-separated text files produced by
XPS fitting tools, handling common quirks such as duplicate "Name" columns,
underscore-delimited headers, and comma-separated numeric fields inside cells.

Key Functions
-------------
read_fit_report_file : Parse XPS fit report files into structured dictionaries
read_fit_spectrum_file : Parse XPS spectrum data files into dictionaries
print_fit_report_averages : Print table of average values from fit report data

Example Usage
-------------
>>> from xps_analysis.xps_processing import read_fit_report_file, print_fit_report_averages
>>> fit_data = read_fit_report_file('my_fit_report.txt')
>>> print_fit_report_averages(fit_data)
"""

import os
import numpy as np


def _find_header(lines, startswith_tuple, required_substring=None):
    """Find and return the first header line that matches criteria.

    This scans ``lines`` and returns the first line that begins with any of the
    strings in ``startswith_tuple``. If ``required_substring`` is provided,
    the line must also contain that substring.

    Parameters
    ----------
    lines : list[str]
        File content split into lines.
    startswith_tuple : tuple[str]
        Tuple of possible prefixes for the header line (e.g. ("Name",)).
    required_substring : str | None
        Optional substring that must be present in the header line.

    Returns
    -------
    tuple[str | None, int | None]
        (header_line, index) if found, otherwise (None, None).
    """
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith_tuple):
            if required_substring is None or required_substring in line:
                return line, i
    return None, None


def _parse_data_rows(lines, header_idx, columns, skip_empty=True, break_on_empty=False):
    """Parse tab-separated rows following a header into a dict of column lists.

    Each value is converted to ``float`` when possible; non-convertible values
    are left as strings. Lines that do not match the expected column count are
    ignored.

    Parameters
    ----------
    lines : list[str]
        File content lines.
    header_idx : int
        Index of the header line in ``lines``.
    columns : list[str]
        Expected column names corresponding to a row's tab-separated fields.
    skip_empty : bool
        If True, skip empty lines between data rows; if False, include them
        (they will be ignored if column lengths don't match).
    break_on_empty : bool
        If True, stop parsing when the first empty line after the header is
        encountered.

    Returns
    -------
    dict[str, list]
        Mapping from column name to list of parsed values (floats or strings).
    """
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
    """Convert each list in ``data_dict`` to a NumPy array.

    Attempts to create a float array for each column and falls back to an
    object array when conversion fails. The input dict is mutated and
    returned for convenience.

    Parameters
    ----------
    data_dict : dict[str, list]
        Mapping of column names to lists of values.

    Returns
    -------
    dict[str, numpy.ndarray]
        The same mapping where each value is a NumPy array.
    """
    for col in data_dict:
        try:
            data_dict[col] = np.array(data_dict[col], dtype=float)
        except ValueError:
            data_dict[col] = np.array(data_dict[col], dtype=object)
    return data_dict


def _clean_header(header_line):
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


def _group_rows_by_name(table_data, name_idx):
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


def _table_to_dict(groups, header):
    """Convert grouped table rows into a dict of 2D NumPy arrays.

    Each header becomes a key mapped to a 2D object array where columns
    represent distinct grouped entries (e.g. chemical components) and rows
    correspond to fit parameters.

    The parser handles numeric fields, comma-separated numeric lists ("x, y"),
    and leaves non-numeric entries as strings.

    Parameters
    ----------
    groups : list[list[list[str]]]
        Output from ``_group_rows_by_name``; groups of raw string rows.
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


def _parse_report_rows(lines, header_idx, header, raw_header):
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
        Cleaned header produced by ``_clean_header``.
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


def read_fit_report_file(file_path):
    """Read the primary fit report table from an XPS report file.

    The function locates the first table header containing "Comp Label" and
    parses the subsequent rows into a dictionary of 2D NumPy arrays. Header
    names are cleaned (for example, "Position" is converted to
    "Binding Energy (eV)"). The returned dictionary will also include a
    ``"File Name"`` key containing the input file's base name for downstream
    use (e.g. when saving figures).

    Parameters
    ----------
    file_path : str
        Path to the report file to read.

    Returns
    -------
    dict | None
        Mapping from header names to 2D NumPy arrays, or ``None`` if the file
        does not exist or no table header is found.
    """

    # use module-level helpers: _clean_header, _group_rows_by_name, _table_to_dict

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

    # Parse table rows and convert to structured dict
    table_rows = _parse_report_rows(lines, header_idx, header, raw_header)
    name_idx = header.index("Comp Label")
    groups = _group_rows_by_name(table_rows, name_idx)
    result_dict = _table_to_dict(groups, header)

    try:
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        result_dict["File Name"] = file_base
    except (OSError, ValueError):
        result_dict["File Name"] = None

    return result_dict


def read_fit_spectrum_file(file_path):
    """Parse an XPS spectrum (data) file into a dictionary of NumPy arrays.

    The function locates a header line (starting with one of ``"KE_"``,
    ``"BE_"``, or ``"CPS_"``), parses the tab-separated columns below it, and
    converts values to numeric types where possible. Column names are
    simplified and renamed for consistency: keys starting with
    ``"Normalised_Residual"`` become ``"Normalised Residual"``, keys beginning
    with ``"CPS"`` are renamed ``"Measured"``, and underscores are removed
    from other keys where appropriate. A ``"File Name"`` entry with the
    input file base name is also added to the returned dict.

    Parameters
    ----------
    file_path : str
        Path to the spectrum file.

    Returns
    -------
    dict
        Mapping of processed column names to 1D NumPy arrays.

    Raises
    ------
    ValueError
        If no suitable data header is found in the file.
    """

    def _rename_spectrum_keys(data_dict):
        """Internal helper: canonicalize spectrum column names.

        It returns a new dict where certain prefixes are normalized and
        underscores are stripped from common header names.
        """
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

    try:
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        renamed_dict["File Name"] = file_base
    except (OSError, ValueError):
        renamed_dict["File Name"] = None

    return renamed_dict


def print_fit_report_averages(fit_data):
    """Print a table with average values for numeric entries in fit report data.

    This function calculates and displays averages for all numeric entries
    in the dictionary returned by ``read_fit_report_file``. Averages are calculated
    for each component (rows) across multiple measurements (columns). Constraint
    parameters (containing "Constr." in the name) and the "File Name" entry are excluded.
    String parameters like "Line Shape" and "Comp Label" are included and checked for
    consistency across measurements.

    Parameters
    ----------
    fit_data : dict
        Dictionary returned by ``read_fit_report_file`` containing 2D NumPy
        arrays with fit parameters.
    """
    if fit_data is None:
        print("No data to process.")
        return

    # Check if we have component names
    if "Name" not in fit_data:
        print("No 'Name' column found in data.")
        return

    # Get component names from the first column
    name_array = fit_data["Name"]
    if name_array.size == 0:
        print("No component data found.")
        return

    # Extract component names (first column of the Name array)
    if name_array.ndim == 2:
        component_names = [str(name_array[i, 0]) for i in range(name_array.shape[0])]
    else:
        component_names = [str(name_array[0]) if name_array.size > 0 else "Unknown"]

    # Collect parameters and their averages/values
    numeric_parameters = []
    string_parameters = []
    component_data = {comp: {} for comp in component_names}

    # Process each parameter (excluding constraints and file name)
    for key, data_array in fit_data.items():
        # Skip constraint parameters, file name, and Area/(RSF*T*MFP)
        if (
            "Constr." in key
            or key == "File Name"
            or key == "Name"
            or key == "Area/(RSF*T*MFP)"
            or not isinstance(data_array, np.ndarray)
        ):
            continue

        # Rename "Binding Energy (eV)" to "BE" and "Comp Label" to "label"
        display_key = key
        if key == "Binding Energy (eV)":
            display_key = "BE"
        elif key == "Comp Label":
            display_key = "Label"

        # Check if array contains numeric data
        is_numeric = False
        if data_array.dtype != object:
            is_numeric = True
        else:
            # Check if first element is numeric
            first_elem = data_array.flat[0] if data_array.size > 0 else None
            if isinstance(first_elem, (int, float, np.integer, np.floating)):
                is_numeric = True

        if is_numeric:
            numeric_parameters.append(display_key)
            # Calculate average for each component (average across columns for each row)
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < data_array.shape[0]:
                    comp_row = data_array[comp_idx, :]

                    # Collect numeric values from this row
                    numeric_values = []
                    for value in comp_row:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            if not np.isnan(value):
                                numeric_values.append(float(value))
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(
                                    item, (int, float, np.integer, np.floating)
                                ):
                                    if not np.isnan(item):
                                        numeric_values.append(float(item))

                    # Calculate average
                    if numeric_values:
                        avg_value = np.mean(numeric_values)
                        component_data[comp_name][display_key] = avg_value
        else:
            # Handle string parameters
            string_parameters.append(display_key)
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < data_array.shape[0]:
                    comp_row = data_array[comp_idx, :]

                    # Collect string values from this row
                    string_values = []
                    for value in comp_row:
                        if isinstance(value, str):
                            string_values.append(value)

                    # Check for consistency
                    if string_values:
                        unique_values = list(set(string_values))
                        if len(unique_values) == 1:
                            component_data[comp_name][display_key] = unique_values[0]
                        else:
                            # Multiple different values - note inconsistency
                            component_data[comp_name][
                                display_key
                            ] = f"MIXED: {', '.join(unique_values)}"

    # Combine parameters in desired order: numeric first, then strings
    all_parameters = numeric_parameters + string_parameters

    if not all_parameters:
        print("No parameters found to display.")
        return

    # Print the 2D table with improved formatting
    print("╔═══ Average values from fit report (by component) ═══╗")
    print()

    # Calculate column widths dynamically
    max_comp_width = max(len(comp) for comp in component_names)
    max_comp_width = max(max_comp_width, len("Component"))

    # Calculate column widths for parameters
    param_widths = {}
    for param in all_parameters:
        max_width = len(param)
        for comp_name in component_names:
            if param in component_data[comp_name]:
                value = component_data[comp_name][param]
                if isinstance(value, (int, float)):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                max_width = max(max_width, len(value_str))
        param_widths[param] = min(max_width + 2, 15)  # Cap at 15 chars, add padding

    # Header row with better formatting
    header = f"{'Component':<{max_comp_width}} │"
    for param in all_parameters:
        header += f" {param:^{param_widths[param]}} │"
    print(header)

    # Separator line - fix alignment to match header spacing exactly
    separator = "─" * max_comp_width + "─┼"
    for i, param in enumerate(all_parameters):
        # Each header param section is " {param:^width} │" = width + 2 chars for spaces + 1 for │
        separator += "─" * (param_widths[param] + 2) + "┼"
    print(separator)

    # Data rows with improved formatting
    for comp_name in component_names:
        row = f"{comp_name:<{max_comp_width}} │"
        for param in all_parameters:
            if param in component_data[comp_name]:
                value = component_data[comp_name][param]
                if isinstance(value, (int, float)):
                    value_str = f"{value:.2f}"
                    row += f" {value_str:>{param_widths[param]}} │"
                else:
                    # String value - truncate if too long
                    str_val = str(value)
                    if len(str_val) > param_widths[param]:
                        str_val = str_val[: param_widths[param] - 3] + "..."
                    row += f" {str_val:^{param_widths[param]}} │"
            else:
                row += f" {'N/A':^{param_widths[param]}} │"
        print(row)

    print()
    print("╚" + "═" * (len(header) - 2) + "╝")
