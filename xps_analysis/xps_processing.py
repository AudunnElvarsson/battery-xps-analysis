"""xps_processing
-----------------

High-level utilities for reading and processing XPS data files.

This module provides the main public API for parsing XPS report and spectrum
text files into Python dictionaries with NumPy arrays. It handles both
single and multi-core level data formats automatically.

The module focuses on robust parsing of tab-separated text files produced by
XPS fitting tools, handling common quirks such as duplicate "Name" columns,
underscore-delimited headers, and comma-separated numeric fields inside cells.

Public Functions
----------------
read_report_file : Parse XPS fit report files into structured dictionaries
read_spectrum_file : Parse XPS spectrum data files into dictionaries
print_report : Print formatted table of average values from report data
get_core_levels : Get list of core levels from a report dictionary

Notes
-----

"""

import os

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
    print_single_core_level,
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

    # Find header line - check for new format (Data Set) or old format (Name)
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

    This function reads XPS spectrum files containing measured and fitted
    spectral data. It automatically detects the data format, parses columns,
    and returns a dictionary with cleaned, standardized key names.

    The function handles:
    - Multiple data format variations (KE_, BE_, CPS_ prefixes)
    - Automatic column name normalization
    - Type conversion to numeric arrays where possible
    - Metadata extraction (file name, core level)

    Parameters
    ----------
    file_path : str
        Path to the spectrum file to read. Should be a tab-separated text
        file with a header line starting with "KE_", "BE_", or "CPS_".

    Returns
    -------
    dict or None
        Mapping of processed column names to 1D NumPy arrays. Key names are
        normalized:
        - "Normalised_Residual*" → "Normalised Residual"
        - "CPS*" → "Measured"
        - Other keys have underscores removed
        Also includes 'File Name' and 'Core Level' metadata keys.
        Returns None if file not found.

    Raises
    ------
    ValueError
        If no suitable data header (starting with KE_, BE_, or CPS_) is
        found in the file.

    See Also
    --------
    read_report_file : Parse XPS fit report files

    Examples
    --------
    Read a spectrum file:

    >>> spectrum_dict = read_spectrum_file("C1s_spectrum.txt")
    >>> print(spectrum_dict.keys())
    dict_keys(['BE', 'Measured', 'Background', 'C-C', 'C-O', 'O-C=O',
               'Residual', 'File Name', 'Core Level'])

    Access the binding energy and measured intensity:

    >>> be = spectrum_dict['BE']
    >>> intensity = spectrum_dict['Measured']
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(be, intensity)

    Notes
    -----
    Column name normalization rules:
    - Columns starting with "Normalised_Residual" → "Normalised Residual"
    - Columns starting with "CPS" → "Measured"
    - For other columns, prefix before underscore is used (e.g., "BE_1" → "BE")
    """

    def _rename_spectrum_keys(data_dict):
        """Internal helper: canonicalize spectrum column names.

        Returns a new dict where certain prefixes are normalized and
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
    """Print formatted table with average values from fit report data.

    This function calculates and displays averages for all numeric parameters
    in XPS fit report data. For multi-core format data, you can print all
    core levels or select specific ones.

    The table shows:
    - Average values for numeric parameters across measurements
    - Relative binding energy (difference from reference component)
    - String parameters (e.g., line shape, component labels)
    - Constraint parameters are automatically excluded

    Parameters
    ----------
    fit_data : dict
        Dictionary returned by ``read_report_file`` containing fit parameters.
        For single-core format: flat dict with 2D NumPy arrays.
        For multi-core format: nested dict with core level keys (e.g., "C 1s",
        "F 1s") containing data dictionaries, plus "File Name" and "Core Level"
        metadata.
    reference : str, optional
        Label of the component to use as reference for relative binding energy
        calculation (default "A"). Relative BE shows the binding energy
        difference with respect to this reference component.
    core_level : str or None, optional
        For multi-core format, specify which core level to print. If None
        (default), all core levels are printed sequentially with spacing
        between them.

    Returns
    -------
    None
        Prints formatted table(s) to stdout.

    See Also
    --------
    read_report_file : Parse XPS fit report files
    get_core_levels : Get list of available core levels

    Examples
    --------
    Print all core levels from a multi-core file:

    >>> report_dict = read_report_file("multicore_report.txt")
    >>> print_report(report_dict)

    Print only C 1s data with custom reference:

    >>> print_report(report_dict, reference="B", core_level="C 1s")

    Print single-core file:

    >>> report_dict = read_report_file("single_core_report.txt")
    >>> print_report(report_dict, reference="A")

    Notes
    -----
    - Parameters containing "Constr." in their name are excluded from display
    - String parameters are checked for consistency across measurements
    - If reference component is not found, a warning is printed and relative
      BE column is not added
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
            print_single_core_level(
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
                print_single_core_level(
                    fit_data[cl], reference, parent_file_name, core_level_override=cl
                )
    else:
        # Single-core format (original behavior)
        print_single_core_level(fit_data, reference)
