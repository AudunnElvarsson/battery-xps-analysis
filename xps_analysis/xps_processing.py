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
print_report : Print table of average values from report data
get_core_levels : Get list of core levels from a report dictionary
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
    group_rows_by_dataset,
    has_dataset_column,
    table_to_dict,
    table_to_dict_exclude_columns,
    parse_report_rows,
    add_file_metadata,
    extract_component_names,
    process_all_parameters,
    calculate_column_widths,
    build_table_header,
    build_table_row,
)


def read_report_file(file_path):
    """Read the primary fit report table from an XPS report file.

    The function locates the first table header containing "Comp Label" and
    parses the subsequent rows into a dictionary of 2D NumPy arrays. Header
    names are cleaned (for example, "Position" is converted to
    "Binding Energy (eV)").

    The function automatically detects two file formats:

    1. **Single core level format**: Traditional format with one core level.
       Returns a flat dictionary with parameter arrays.

    2. **Multi-core level format**: New format with "Data Set" column and
       multiple core levels. Returns a nested dictionary where each core level
       is a key containing its own parameter arrays.

    Parameters
    ----------
    file_path : str
        Path to the report file to read.

    Returns
    -------
    dict | None
        For single core level: Mapping from header names to 2D NumPy arrays.
        For multi-core level: Nested dict with core levels as keys, each
        containing parameter arrays. Also includes "File Name" and "Core Level"
        metadata keys. Returns ``None`` if file not found or no header.
    """

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line - try both old format (starts with "Name") and new format (starts with "Data Set")
    header_line, header_idx = find_header(
        lines, ("Data Set",), required_substring="Comp Label"
    )

    if header_idx is None:
        # Fall back to old format
        header_line, header_idx = find_header(
            lines, ("Name",), required_substring="Comp Label"
        )

    if header_idx is None:
        print("No table header found.")
        return None

    header, raw_header = clean_header(header_line)
    header = ["Binding Energy (eV)" if h == "Position" else h for h in header]
    header = ["RSF" if h == "Library RSF" else h for h in header]

    # Parse table rows
    table_rows = parse_report_rows(lines, header_idx, header, raw_header)

    # Check if this is the new multi-core format
    if has_dataset_column(header):
        # Multi-core level format
        dataset_idx = header.index("Data Set")
        tag_idx = header.index("Tag")

        # Group by dataset and core level
        core_level_groups = group_rows_by_dataset(table_rows, dataset_idx, tag_idx)

        # Build result dictionary with nested structure
        # Exclude "Data Set" and "Tag" columns from the output
        result_dict = {}
        exclude_cols = [dataset_idx, tag_idx]

        for core_level, groups in core_level_groups.items():
            result_dict[core_level] = table_to_dict_exclude_columns(
                groups, header, exclude_cols
            )

        # Add metadata (use first core level as default for backwards compatibility)
        first_core = next(iter(result_dict.keys()))
        result_dict["Core Level"] = first_core
        add_file_metadata(result_dict, file_path)

    else:
        # Original single core level format
        name_idx = header.index("Comp Label")
        groups = group_rows_by_name(table_rows, name_idx)
        result_dict = table_to_dict(groups, header)
        add_file_metadata(result_dict, file_path)

    return result_dict


def get_core_levels(report_dict):
    """Get list of core levels from a report dictionary.

    For multi-core format files, this returns the list of core level names
    (e.g., ["C 1s", "F 1s", "O 1s"]). For single-core format files, this
    returns a list containing the single core level name if available, or
    an empty list if the core level cannot be determined.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned by ``read_report_file``.

    Returns
    -------
    list of str
        List of core level names found in the report. For multi-core files,
        this contains all core levels. For single-core files, this contains
        one element (the core level name) or is empty if not determinable.

    Examples
    --------
    >>> report_dict = read_report_file("multicore_report.txt")
    >>> core_levels = get_core_levels(report_dict)
    >>> print(core_levels)
    ['C 1s', 'F 1s', 'O 1s']
    >>> # Create figure with correct number of subplots
    >>> fig, axes = plt.subplots(1, len(core_levels), figsize=(6*len(core_levels), 5))
    """
    if not report_dict:
        return []

    # Check if multi-core format
    is_multicore = "Core Level" in report_dict and "Name" not in report_dict

    if is_multicore:
        # Multi-core format - extract core level keys
        return [k for k in report_dict.keys() if k not in ["Core Level", "File Name"]]
    else:
        # Single-core format - try to get core level from metadata
        if "Core Level" in report_dict:
            return [report_dict["Core Level"]]
        return []


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


def print_report(fit_data, reference="A", core_level=None):
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
        arrays with fit parameters. For multi-core format, this should be a
        nested dict with core level keys (e.g., "C 1s", "F 1s").
    reference : str, optional
        Label of the component to use as reference for relative binding energy
        calculation (default "A"). The relative BE column shows the difference
        in binding energy with respect to this reference component.
    core_level : str, optional
        For multi-core format, specify which core level to print. If not
        provided, all core levels will be printed sequentially.
    """
    # Validate input data
    if not fit_data:
        print("No data to process.")
        return

    # Check if multi-core format (nested dict with core level keys)
    is_multicore = "Core Level" in fit_data and "Name" not in fit_data

    if is_multicore:
        # Multi-core format handling
        core_levels = [
            k for k in fit_data.keys() if k not in ["Core Level", "File Name"]
        ]
        parent_file_name = fit_data.get("File Name")

        if core_level:
            # Print only the selected core level
            if core_level not in core_levels:
                print(f"Core level '{core_level}' not found. Available: {core_levels}")
                return
            _print_single_core_level(
                fit_data[core_level],
                reference,
                parent_file_name,
                core_level_override=core_level,
            )
        else:
            # Print all core levels
            for idx, cl in enumerate(core_levels):
                if idx > 0:
                    print("\n")  # Add spacing between core levels
                _print_single_core_level(
                    fit_data[cl], reference, parent_file_name, core_level_override=cl
                )
    else:
        # Single-core format (original behavior)
        _print_single_core_level(fit_data, reference)


def _print_single_core_level(
    fit_data, reference="A", file_name_override=None, core_level_override=None
):
    """Print averages table for a single core level's data.

    Parameters
    ----------
    fit_data : dict
        Single core level data dictionary with 'Name' and parameter arrays.
    reference : str, optional
        Label of the reference component for relative BE calculation.
    """
    if "Name" not in fit_data:
        print("No 'Name' column found in data.")
        return

    if fit_data["Name"].size == 0:
        print("No component data found.")
        return

    # Extract component names and process parameters
    component_names = extract_component_names(fit_data)
    numeric_params, string_params, component_data = process_all_parameters(
        fit_data, component_names
    )

    # Calculate relative binding energies if BE data is available
    if "BE" in numeric_params:
        # Find reference component
        reference_comp = None
        for comp_name in component_names:
            if component_data[comp_name].get("Label") == reference:
                reference_comp = comp_name
                break

        if reference_comp and "BE" in component_data[reference_comp]:
            reference_be = component_data[reference_comp]["BE"]
            # Add relative BE for all components
            for comp_name in component_names:
                if "BE" in component_data[comp_name]:
                    component_data[comp_name]["Rel. BE"] = (
                        component_data[comp_name]["BE"] - reference_be
                    )
            # Insert "Rel. BE" after "BE" in the numeric_params list
            be_idx = numeric_params.index("BE")
            numeric_params.insert(be_idx + 1, "Rel. BE")
        else:
            print(
                f"Warning: Reference component '{reference}' not found or has no BE data."
            )

    all_parameters = numeric_params + string_params
    if not all_parameters:
        print("No parameters found to display.")
        return

    # Calculate column widths and build table components
    max_comp_width, param_widths = calculate_column_widths(
        all_parameters, component_names, component_data
    )
    header, separator = build_table_header(max_comp_width, all_parameters, param_widths)

    # Print table
    core_level = core_level_override or fit_data.get("Core Level") or "Unknown"
    # Prefer provided file name (from parent in multi-core), else from this dict
    display_file = file_name_override or fit_data.get("File Name", "Unknown file")
    file_name = f"File: {display_file}"
    title = f"Average values from fit report - {core_level}"
    print(
        f"{'═' * int(np.floor((len(header) - len(title)) / 2 - 1))}",
        title,
        f"{'═' * int(np.ceil((len(header) - len(title)) / 2 - 1))}",
    )
    print(
        f"{'═' * int(np.floor((len(header) - len(file_name)) / 2 - 1))}",
        file_name,
        f"{'═' * int(np.ceil((len(header) - len(file_name)) / 2 - 1))}\n",
    )
    print(header)
    print(separator)

    for comp_name in component_names:
        print(
            build_table_row(
                comp_name, all_parameters, component_data, param_widths, max_comp_width
            )
        )

    print(f"{'═' * len(header)}")
