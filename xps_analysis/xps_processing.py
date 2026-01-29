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
get_available_parameters : Get list of all parameters available in report data
print_report : Print formatted table of average values from report data
get_core_levels : Get list of core levels from a report dictionary
extract_component_data : Extract data array for a specific component by name or label
extract_core_level_data : Extract and aggregate data for an entire core level

File Format Support
-------------------
The module automatically detects and handles two report file formats:

1. **Single core level format**: Traditional format containing data for one
   core level. Returns a flat dictionary with parameter arrays.

2. **Multi-core level format**: New format with "Data Set" or "Iteration"
   column containing multiple core levels. Returns a nested dictionary where
   each core level is a key containing its parameter arrays.

Typical Workflow
----------------
1. Read report file with ``read_report_file()``
2. Check available parameters with ``get_available_parameters()``
3. Print formatted tables with ``print_report()`` using custom parameters
4. Extract core level names with ``get_core_levels()`` for plotting

Examples
--------
Basic usage for report files:

>>> import xps_analysis.xps_processing as xp
>>>
>>> # Read report file (auto-detects format)
>>> report = xp.read_report_file("fit_report.txt")
>>>
>>> # See what parameters are available
>>> params = xp.get_available_parameters(report)
>>> print("Numeric:", params['numeric'])
>>> print("String:", params['string'])
>>>
>>> # Print with default parameters
>>> xp.print_report(report)
>>>
>>> # Print with custom parameters
>>> custom_params = ["Label", "BE", "FWHM", "RSF", "Raw Area"]
>>> xp.print_report(report, parameters=custom_params)
>>>
>>> # Get core levels for plotting
>>> core_levels = xp.get_core_levels(report)

Working with spectrum files:

>>> # Read spectrum file
>>> spectrum = xp.read_spectrum_file("C1s_spectrum.txt")
>>>
>>> # Access data arrays
>>> be = spectrum['BE']
>>> intensity = spectrum['Measured']

Notes
-----
All parameter names from the original data file are preserved, including:
- Binding energy, FWHM, RSF (Relative Sensitivity Factor)
- Atomic concentration, raw peak areas
- Constraint parameters (position, area, FWHM constraints)
- Line shapes and component labels

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
    print_single_core_level,
    label_spin_orbit_doublets,
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

    # Find header line - check for multi-core format (Data Set or Iteration) or single-core format (Name)
    # First try Data Set format
    header_line, header_idx = find_header(
        lines, ("Data Set",), required_substring="Comp Label"
    )

    if header_idx is None:
        # Try Iteration format
        header_line, header_idx = find_header(
            lines, ("Iteration",), required_substring="Comp Label"
        )

    if header_idx is None:
        # Fall back to old single-core format
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

    # Check if this is the multi-core format (Data Set or Iteration column)
    if has_dataset_column(header):
        # Multi-core level format
        # Get the dataset column index - could be "Data Set" or "Iteration"
        if "Data Set" in header:
            dataset_idx = header.index("Data Set")
        elif "Iteration" in header:
            dataset_idx = header.index("Iteration")
        else:
            # This shouldn't happen if has_dataset_column returned True
            print(
                "Error: Multi-core format detected but no Data Set/Iteration column found."
            )
            return None
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
            # Label spin-orbit doublets
            result_dict[core_level] = label_spin_orbit_doublets(result_dict[core_level])

        # Add metadata (use first core level as default for backwards compatibility)
        first_core = next(iter(result_dict.keys()))
        result_dict["Core Level"] = first_core
        add_file_metadata(result_dict, file_path)

    else:
        # Original single core level format
        name_idx = header.index("Comp Label")
        groups = group_rows_by_name(table_rows, name_idx)
        result_dict = table_to_dict(groups, header)
        # Label spin-orbit doublets
        result_dict = label_spin_orbit_doublets(result_dict)
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


def get_available_parameters(fit_data, core_level=None):
    """Get list of all available parameters in fit report data.

    This function returns all parameter names that can be displayed in
    print_report tables, organized by type (numeric vs string). This is
    useful for discovering what data is available before customizing
    table output.

    All parameters from the original data file are included, such as:
    - Binding energy (BE), FWHM, RSF
    - Atomic concentration (%At Conc), raw areas
    - Constraint values (Pos Constr., Area Constr., FWHM Constr.)
    - RSF-corrected areas (Area/(RSF*T*MFP))
    - Line shapes and component labels

    Parameters
    ----------
    fit_data : dict
        Dictionary returned by ``read_report_file`` containing fit parameters.
        Can be either single-core format (flat dict) or multi-core format
        (nested dict with core level keys).
    core_level : str or None, optional
        For multi-core format, specify which core level to check. If None,
        uses the first available core level. Ignored for single-core format.

    Returns
    -------
    dict
        Dictionary with two keys:

        - 'numeric' : list of str
            Names of numeric parameters (can be averaged, plotted)
        - 'string' : list of str
            Names of string parameters (text values like line shapes)

        Returns empty lists if no data found or core level doesn't exist.

    See Also
    --------
    print_report : Print tables with custom parameter selection
    read_report_file : Parse XPS fit report files

    Examples
    --------
    Discover available parameters:

    >>> report_dict = read_report_file("report.txt")
    >>> params = get_available_parameters(report_dict)
    >>> print("Numeric:", params['numeric'])
    Numeric: ['BE', 'FWHM', 'RSF', '%At Conc', 'Pos Constr.',
              'FWHM Constr.', 'Area/(RSF*T*MFP)', 'Raw Area']
    >>> print("String:", params['string'])
    String: ['Line Shape', 'Area Constr.', 'Label', 'Not Specified']

    Use with print_report to customize output:

    >>> # Show only essential parameters
    >>> essential = ["Label", "BE", "FWHM", "Raw Area"]
    >>> print_report(report_dict, parameters=essential)

    >>> # Show all numeric parameters
    >>> all_numeric = params['numeric']
    >>> print_report(report_dict, parameters=all_numeric)

    For multi-core format, check specific core level:

    >>> params_c1s = get_available_parameters(report_dict, core_level="C 1s")
    >>> params_f1s = get_available_parameters(report_dict, core_level="F 1s")

    Notes
    -----
    Metadata fields like "File Name", "Core Level", "Name", and "Doublet Group"
    are automatically excluded as they are not displayable parameters.
    """
    from .processing_helpers import extract_component_names, process_all_parameters

    # Check if multi-core format
    is_multicore = "Core Level" in fit_data and "Name" not in fit_data

    if is_multicore:
        if core_level:
            if core_level not in fit_data:
                return {"numeric": [], "string": []}
            data = fit_data[core_level]
        else:
            # Use first core level
            core_levels = [
                k for k in fit_data.keys() if k not in ["Core Level", "File Name"]
            ]
            if not core_levels:
                return {"numeric": [], "string": []}
            data = fit_data[core_levels[0]]
    else:
        data = fit_data

    if "Name" not in data or data["Name"].size == 0:
        return {"numeric": [], "string": []}

    # Extract parameters
    component_names = extract_component_names(data)
    numeric_params, string_params, _ = process_all_parameters(data, component_names)

    return {"numeric": numeric_params, "string": string_params}


def print_report(fit_data, reference="A", core_level=None, parameters=None):
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
    reference : str or dict, optional
        Component label or name to use as reference for relative binding energy
        calculation (default "A"). Can be:
        - str: Single reference applied to all core levels (e.g., "A" or "LiF")
        - dict: Maps core level names to reference components (e.g.,
          {"C 1s": "A", "F 1s": "LiF", "O 1s": "B"})

        Relative BE shows the binding energy difference with respect to this
        reference component. For multi-core format, dict allows different
        references per core level.
    core_level : str or None, optional
        For multi-core format, specify which core level to print. If None
        (default), all core levels are printed sequentially with spacing
        between them.
    parameters : list of str or None, optional
        List of parameter names to display in the table. If None (default),
        displays default parameters: ["Label", "BE", "Rel. BE", "FWHM",
        "Raw Area", "%At Conc"]. Use get_available_parameters() to see
        all available parameters.

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

    Use different references for each core level:

    >>> refs = {"C 1s": "A", "F 1s": "LiF", "O 1s": "C=O"}
    >>> print_report(report_dict, reference=refs)

    Print single-core file:

    >>> report_dict = read_report_file("single_core_report.txt")
    >>> print_report(report_dict, reference="A")

    Print with custom parameters:

    >>> params = ["Label", "BE", "FWHM", "Raw Area", "RSF"]
    >>> print_report(report_dict, parameters=params)

    See available parameters:

    >>> available = get_available_parameters(report_dict)
    >>> print(available['numeric'])  # Numeric parameters
    >>> print(available['string'])   # String parameters

    Notes
    -----
    - Parameters containing "Constr." in their name are excluded from display
    - String parameters are checked for consistency across measurements
    - If reference component is not found, a warning is printed and relative
      BE column is not added
    - Default parameters: ["Label", "BE", "Rel. BE", "FWHM", "Raw Area", "%At Conc"]
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
            # Print only the selected core level with main title
            if core_level not in core_levels:
                print(f"Core level '{core_level}' not found. Available: {core_levels}")
                return
            # Get reference for this core level
            cl_reference = (
                reference.get(core_level, "A")
                if isinstance(reference, dict)
                else reference
            )
            print_single_core_level(
                fit_data[core_level],
                cl_reference,
                parent_file_name,
                core_level_override=core_level,
                show_main_title=True,
                parameters=parameters,
            )
        else:
            # Print main title once, then all core levels with subtitles
            for idx, cl in enumerate(core_levels):
                # Get reference for this core level
                cl_reference = (
                    reference.get(cl, "A") if isinstance(reference, dict) else reference
                )
                print_single_core_level(
                    fit_data[cl],
                    cl_reference,
                    parent_file_name,
                    core_level_override=cl,
                    show_main_title=(idx == 0),
                    show_subtitle=True,
                    parameters=parameters,
                )
    else:
        # Single-core format (original behavior)
        print_single_core_level(fit_data, reference, parameters=parameters)


def extract_component_data(
    report_dict, component_identifier, parameter="%At Conc", core_level=None
):
    """Extract data array for a specific component from a report dictionary.

    This function searches for a component by its label (e.g., "A", "B") or
    name (e.g., "LiF", "C-C / C-H") and returns the corresponding data array
    for the specified parameter across all measurements.

    Parameters
    ----------
    report_dict : dict
        Report dictionary from ``read_report_file()``. Can be either single
        or multi-core level format.
    component_identifier : str
        Component label (e.g., "A", "B", "C") or component name
        (e.g., "LiF", "P-F", "C-C / C-H") to search for.
    parameter : str, optional
        Parameter to extract. Common values include:
        - "%At Conc" (default) - Atomic concentration
        - "Binding Energy (eV)" or "BE" - Binding energy
        - "Raw Area" or "Area" - Peak area
        - "FWHM" - Full width at half maximum
        Default is "%At Conc".
    core_level : str, optional
        Core level name (e.g., "C 1s", "F 1s"). Required for multi-core
        level reports; ignored for single-core level reports.

    Returns
    -------
    numpy.ndarray or None
        1D array containing the parameter values across all measurements
        for the specified component. Returns None if the component or
        parameter is not found.

    Examples
    --------
    Extract atomic concentration for a component in single-core report:

    >>> import xps_analysis.xps_processing as xp
    >>> report = xp.read_report_file("C1s_report.txt")
    >>> lif_at_conc = xp.extract_component_data(report, "LiF")
    >>> print(lif_at_conc)
    [25.3 26.1 24.8 25.9]

    Extract binding energy for a component in multi-core report:

    >>> report = xp.read_report_file("multicore_report.txt")
    >>> be_values = xp.extract_component_data(
    ...     report, "C-C / C-H", parameter="BE", core_level="C 1s"
    ... )

    Extract by component label instead of name:

    >>> at_conc = xp.extract_component_data(report, "A", core_level="F 1s")

    Notes
    -----
    - Searches both "Comp Label" and "Name" fields for the identifier
    - Parameter name shortcuts are automatically expanded (e.g., "BE" → "Binding Energy (eV)")
    - For multi-core reports, core_level must be specified
    - Returns None with a warning if component or parameter not found
    """
    from .plot_helpers.plot_rendering import get_column_name

    # Expand parameter shortcuts to full names
    param_full = get_column_name(parameter)

    # Handle multi-core level reports
    if core_level is not None:
        if core_level not in report_dict:
            available = get_core_levels(report_dict)
            print(
                f"Warning: Core level '{core_level}' not found. "
                f"Available: {available}"
            )
            return None
        data = report_dict[core_level]
    else:
        # Check if this is actually a multi-core report
        core_levels = get_core_levels(report_dict)
        if core_levels:
            print(
                f"Warning: This is a multi-core level report. "
                f"Please specify core_level parameter. Available: {core_levels}"
            )
            return None
        data = report_dict

    # Check if parameter exists
    if param_full not in data:
        available = get_available_parameters({core_level: data} if core_level else data)
        print(
            f"Warning: Parameter '{param_full}' not found. "
            f"Available numeric parameters: {available['numeric']}"
        )
        return None

    param_data = data[param_full]

    # Search by Comp Label first (typically single letters like A, B, C)
    comp_labels = data.get("Comp Label")
    if comp_labels is not None:
        for i in range(comp_labels.shape[0]):
            label = comp_labels[i, 0] if comp_labels.ndim > 1 else comp_labels[i]
            if str(label) == str(component_identifier):
                return param_data[i, :]

    # Search by Name (chemical species names)
    names = data.get("Name")
    if names is not None:
        for i in range(names.shape[0]):
            name = names[i, 0] if names.ndim > 1 else names[i]
            if str(name) == str(component_identifier):
                return param_data[i, :]

    # Component not found
    if names is not None and comp_labels is not None:
        available_labels = [
            str(comp_labels[i, 0] if comp_labels.ndim > 1 else comp_labels[i])
            for i in range(comp_labels.shape[0])
        ]
        available_names = [
            str(names[i, 0] if names.ndim > 1 else names[i])
            for i in range(names.shape[0])
        ]
        print(
            f"Warning: Component '{component_identifier}' not found.\n"
            f"Available labels: {available_labels}\n"
            f"Available names: {available_names}"
        )
    else:
        print(f"Warning: Component '{component_identifier}' not found.")

    return None


def extract_core_level_data(report_dict, core_level):
    """Extract and aggregate data for an entire core level.

    This function aggregates component data to provide core level totals and
    averages. Atomic concentrations and areas are summed across all components,
    while binding energy and FWHM are averaged across components.

    Parameters
    ----------
    report_dict : dict
        Report dictionary from ``read_report_file()``. Must be a multi-core
        level report (see Notes).
    core_level : str
        Core level name to aggregate (e.g., "C 1s", "F 1s", "O 1s").

    Returns
    -------
    dict
        Dictionary containing aggregated core level data with the following keys:
        - "Atomic Concentration": 1D array of summed atomic concentrations across
          all components in this core level, one value per measurement
        - "Area": 1D array of summed peak areas across all components
        - "Binding Energy": 1D array of average binding energies across components
        - "FWHM": 1D array of average FWHMs across components
        - "n_components": Integer count of components in this core level
        - "component_names": List of component names in this core level

    Examples
    --------
    Aggregate a core level from a multi-core report:

    >>> import xps_analysis.xps_processing as xp
    >>> report = xp.read_report_file("multicore_report.txt")
    >>> c1s_data = xp.extract_core_level_data(report, "C 1s")
    >>> print(f"Total C 1s concentration: {c1s_data['Atomic Concentration']}")
    >>> print(f"Number of components: {c1s_data['n_components']}")
    >>> print(f"Average BE: {c1s_data['Binding Energy']}")

    Extract and compare multiple core levels:

    >>> core_levels = xp.get_core_levels(report)
    >>> aggregated = {}
    >>> for core in core_levels:
    ...     aggregated[core] = xp.extract_core_level_data(report, core)
    >>>
    >>> # Compare total concentrations
    >>> for core, data in aggregated.items():
    ...     print(f"{core}: {data['Atomic Concentration'][0]:.1f}%")

    Notes
    -----
    This function is designed for multi-core level reports where you want to
    treat all components in a core level as a single entity. For single-core
    reports or to work with individual components, use ``extract_component_data()``.

    Aggregation rules:
    - **Atomic Concentration** and **Area**: Summed across components
    - **Binding Energy** and **FWHM**: Averaged across components
    - Only numeric values are included in aggregations
    """
    # Check if core_level exists
    if core_level not in report_dict:
        available = get_core_levels(report_dict)
        print(
            f"Warning: Core level '{core_level}' not found. " f"Available: {available}"
        )
        return None

    data = report_dict[core_level]

    # Extract component data
    at_conc = data.get("%At Conc")
    area = data.get("Raw Area")
    be = data.get("Binding Energy (eV)")
    fwhm = data.get("FWHM")
    names = data.get("Name")

    if at_conc is None:
        print(f"Error: No atomic concentration data found in {core_level}")
        return None

    n_components = at_conc.shape[0]
    n_measurements = at_conc.shape[1] if at_conc.ndim > 1 else 1

    # Extract component names
    component_names = []
    if names is not None:
        for i in range(n_components):
            name = names[i, 0] if names.ndim > 1 else names[i]
            component_names.append(str(name))

    # Initialize result arrays
    at_conc_sum = np.zeros(n_measurements)
    area_sum = np.zeros(n_measurements) if area is not None else None
    be_avg = np.zeros(n_measurements) if be is not None else None
    fwhm_avg = np.zeros(n_measurements) if fwhm is not None else None

    # Aggregate data
    be_count = np.zeros(n_measurements)
    fwhm_count = np.zeros(n_measurements)

    for comp_idx in range(n_components):
        # Sum atomic concentrations
        for meas_idx in range(n_measurements):
            at_conc_val = (
                at_conc[comp_idx, meas_idx] if at_conc.ndim > 1 else at_conc[comp_idx]
            )
            if isinstance(at_conc_val, (int, float, np.integer, np.floating)):
                at_conc_sum[meas_idx] += float(at_conc_val)

        # Sum areas
        if area is not None:
            for meas_idx in range(n_measurements):
                area_val = area[comp_idx, meas_idx] if area.ndim > 1 else area[comp_idx]
                if isinstance(area_val, (int, float, np.integer, np.floating)):
                    area_sum[meas_idx] += float(area_val)

        # Average binding energies
        if be is not None:
            for meas_idx in range(n_measurements):
                be_val = be[comp_idx, meas_idx] if be.ndim > 1 else be[comp_idx]
                if isinstance(be_val, (int, float, np.integer, np.floating)):
                    be_avg[meas_idx] += float(be_val)
                    be_count[meas_idx] += 1

        # Average FWHMs
        if fwhm is not None:
            for meas_idx in range(n_measurements):
                fwhm_val = fwhm[comp_idx, meas_idx] if fwhm.ndim > 1 else fwhm[comp_idx]
                if isinstance(fwhm_val, (int, float, np.integer, np.floating)):
                    fwhm_avg[meas_idx] += float(fwhm_val)
                    fwhm_count[meas_idx] += 1

    # Compute averages (avoid division by zero)
    if be is not None:
        be_avg = np.divide(
            be_avg, be_count, out=np.full_like(be_avg, np.nan), where=be_count > 0
        )

    if fwhm is not None:
        fwhm_avg = np.divide(
            fwhm_avg,
            fwhm_count,
            out=np.full_like(fwhm_avg, np.nan),
            where=fwhm_count > 0,
        )

    # Build result dictionary
    result = {
        "Atomic Concentration": at_conc_sum,
        "n_components": n_components,
        "component_names": component_names,
    }

    if area_sum is not None:
        result["Area"] = area_sum

    if be_avg is not None:
        result["Binding Energy"] = be_avg

    if fwhm_avg is not None:
        result["FWHM"] = fwhm_avg

    return result
