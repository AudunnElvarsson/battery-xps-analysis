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
read_report_file : Parse XPS report files into structured dictionaries
read_spectrum_file : Parse XPS spectrum data files into dictionaries
print_report_averages : Print table of average values from report data
"""

import os
import numpy as np

# Import parsing utilities from submodules
from .processing_helpers import (
    find_header,
    parse_data_rows,
    convert_data_types,
    clean_header,
    group_rows_by_name,
    table_to_dict,
    parse_report_rows,
    process_parameter,
    format_table_value,
    add_file_metadata,
)


def read_report_file(file_path):
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

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line and index
    header_line, header_idx = find_header(
        lines, ("Name",), required_substring="Comp Label"
    )
    if header_idx is None:
        print("No table header found.")
        return None

    header, raw_header = clean_header(header_line)
    header = ["Binding Energy (eV)" if h == "Position" else h for h in header]

    # Parse table rows and convert to structured dict
    table_rows = parse_report_rows(lines, header_idx, header, raw_header)
    name_idx = header.index("Comp Label")
    groups = group_rows_by_name(table_rows, name_idx)
    result_dict = table_to_dict(groups, header)
    add_file_metadata(result_dict, file_path)

    return result_dict


def read_spectrum_file(file_path):
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

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line and index
    header_line, header_idx = find_header(lines, ("KE_", "BE_", "CPS_"))
    if header_idx is None:
        raise ValueError("No data header found in file.")

    columns = [col.strip() for col in header_line.split("\t")]
    data_dict = parse_data_rows(
        lines, header_idx, columns, skip_empty=True, break_on_empty=False
    )
    data_dict = convert_data_types(data_dict)
    renamed_dict = _rename_spectrum_keys(data_dict)
    add_file_metadata(renamed_dict, file_path)

    return renamed_dict


def print_report_averages(fit_data):
    """Print a table with average values for numeric entries in fit report data.

    This function calculates and displays averages for all numeric entries
    in the dictionary returned by ``read_report_file``. Averages are calculated
    for each component (rows) across multiple measurements (columns). Constraint
    parameters (containing "Constr." in the name) and the "File Name" entry are excluded.
    String parameters like "Line Shape" and "Comp Label" are included and checked for
    consistency across measurements.

    Parameters
    ----------
    fit_data : dict
        Dictionary returned by ``read_report_file`` containing 2D NumPy
        arrays with fit parameters.
    """
    if not fit_data or "Name" not in fit_data or fit_data["Name"].size == 0:
        print(
            "No data to process."
            if not fit_data
            else (
                "No 'Name' column found in data."
                if "Name" not in fit_data
                else "No component data found."
            )
        )
        return

    # Extract component names
    name_array = fit_data["Name"]
    component_names = (
        [str(name_array[i, 0]) for i in range(name_array.shape[0])]
        if name_array.ndim == 2
        else [str(name_array[0]) if name_array.size > 0 else "Unknown"]
    )

    # Process parameters
    numeric_parameters, string_parameters = [], []
    component_data = {comp: {} for comp in component_names}

    skip_keys = {"Constr.", "File Name", "Name", "Area/(RSF*T*MFP)", "Core Level"}
    for key, data_array in fit_data.items():
        if any(skip in key for skip in skip_keys) or not isinstance(
            data_array, np.ndarray
        ):
            continue

        display_key, is_numeric = process_parameter(
            key, data_array, component_names, component_data
        )
        (numeric_parameters if is_numeric else string_parameters).append(display_key)

    all_parameters = numeric_parameters + string_parameters
    if not all_parameters:
        print("No parameters found to display.")
        return

    # Calculate column widths
    max_comp_width = max(len(comp) for comp in component_names + ["Component"])
    param_widths = {}
    for param in all_parameters:
        max_width = len(param)
        for comp_name in component_names:
            if param in component_data[comp_name]:
                value_str, _ = format_table_value(component_data[comp_name][param], 15)
                max_width = max(max_width, len(value_str))
        param_widths[param] = min(max_width + 2, 15)

        # Print table
        core_level = fit_data.get("Core Level") or "Unknown"
    print(f"╔═══ Average values from fit report (by component) - {core_level} ═══╗\n")

    # Header and separator
    header = f"{'Component':<{max_comp_width}} │" + "".join(
        f" {p:^{param_widths[p]}} │" for p in all_parameters
    )
    separator = (
        "─" * max_comp_width
        + "─┼"
        + "".join("─" * (param_widths[p] + 2) + "┼" for p in all_parameters)
    )
    print(header)
    print(separator)

    # Data rows
    for comp_name in component_names:
        row = f"{comp_name:<{max_comp_width}} │"
        for param in all_parameters:
            if param in component_data[comp_name]:
                value_str, is_numeric = format_table_value(
                    component_data[comp_name][param], param_widths[param]
                )
                alignment = ">" if is_numeric else "^"
                row += f" {value_str:{alignment}{param_widths[param]}} │"
            else:
                row += f" {'N/A':^{param_widths[param]}} │"
        print(row)

    print(f"\n╚{'═' * (len(header) - 2)}╝")
