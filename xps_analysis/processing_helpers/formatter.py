"""XPS output formatting utilities.

This module contains functions for formatting and displaying XPS analysis results.

Public Functions
----------------
format_table_value : Format a value for table display with width constraints
calculate_column_widths : Calculate optimal column widths for table display
build_table_header : Build formatted table header and separator
build_table_row : Build a single formatted table row
print_single_core_level : Print formatted table for a single core level
"""

import numpy as np
from .data_analyzer import extract_component_names, process_all_parameters


def format_table_value(value, width):
    """Format a value for table display with width constraints."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}", True
    else:
        str_val = str(value)
        if len(str_val) > width:
            str_val = str_val[: width - 3] + "..."
        return str_val, False


def calculate_column_widths(all_parameters, component_names, component_data):
    """Calculate optimal column widths for table display.

    Parameters
    ----------
    all_parameters : list[str]
        List of parameter names to display.
    component_names : list[str]
        List of component names.
    component_data : dict
        Nested dict mapping component names to parameter values.

    Returns
    -------
    tuple[int, dict]
        (max_comp_width, param_widths) where max_comp_width is the maximum
        component name width and param_widths is a dict of parameter widths.
    """
    max_comp_width = max(len(comp) for comp in component_names + ["Component"])
    param_widths = {}
    for param in all_parameters:
        max_width = len(param)
        for comp_name in component_names:
            if param in component_data[comp_name]:
                value_str, _ = format_table_value(component_data[comp_name][param], 15)
                max_width = max(max_width, len(value_str))
        param_widths[param] = min(max_width + 2, 15)
    return max_comp_width, param_widths


def build_table_header(max_comp_width, all_parameters, param_widths):
    """Build formatted table header and separator.

    Parameters
    ----------
    max_comp_width : int
        Width for component column.
    all_parameters : list[str]
        List of parameter names.
    param_widths : dict
        Dict mapping parameter names to column widths.

    Returns
    -------
    tuple[str, str]
        (header, separator) strings for the table.
    """
    header = f"{'Component':<{max_comp_width}} │" + "".join(
        f" {p:^{param_widths[p]}} │" for p in all_parameters
    )
    separator = (
        "─" * max_comp_width
        + "─┼"
        + "".join("─" * (param_widths[p] + 2) + "┼" for p in all_parameters)
    )
    return header, separator


def build_table_row(
    comp_name, all_parameters, component_data, param_widths, max_comp_width
):
    """Build a single formatted table row.

    Parameters
    ----------
    comp_name : str
        Component name for this row.
    all_parameters : list[str]
        List of parameter names.
    component_data : dict
        Nested dict mapping component names to parameter values.
    param_widths : dict
        Dict mapping parameter names to column widths.
    max_comp_width : int
        Width for component column.

    Returns
    -------
    str
        Formatted row string.
    """
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
    return row


def print_single_core_level(
    fit_data,
    reference="A",
    file_name_override=None,
    core_level_override=None,
    show_main_title=True,
    show_subtitle=False,
    parameters=None,
):
    """Print formatted table of average values for a single core level.

    This function displays a nicely formatted table showing average values
    for all parameters in the fit data. Relative binding energies are
    calculated with respect to a reference component.

    Parameters
    ----------
    fit_data : dict
        Single core level data dictionary with 'Name' and parameter arrays.
        Must contain at least a 'Name' key with component names.
    reference : str, optional
        Label of the reference component for relative BE calculation
        (default "A"). The relative BE is calculated as the difference
        from this component's binding energy.
    file_name_override : str or None, optional
        Override for file name display. If None, uses fit_data['File Name'].
        Used when printing multi-core data where parent file name should
        be shown.
    core_level_override : str or None, optional
        Override for core level name in title. If None, uses
        fit_data['Core Level'] or 'Unknown'.
    show_main_title : bool, optional
        If True (default), print the main title with file name.
        Set to False when printing multiple core levels to avoid repetition.
    show_subtitle : bool, optional
        If True, print core level as a subtitle instead of in main title.
        Used when printing multiple core levels (default False).
    parameters : list of str or None, optional
        List of parameter names to display. If None, uses default parameters:
        ["Label", "BE", "Rel. BE", "FWHM", "Raw Area", "%At Conc"].

    Returns
    -------
    None
        Prints formatted table to stdout.

    Notes
    -----
    - Constraint parameters (containing "Constr.") are excluded
    - Relative BE column is added after BE if reference component is found
    - String parameters are checked for consistency across measurements
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

    # Filter parameters based on user selection
    if parameters is None:
        # Default parameters
        default_params = ["Label", "BE", "Rel. BE", "FWHM", "Raw Area", "%At Conc"]
        all_parameters = [
            p for p in default_params if p in numeric_params or p in string_params
        ]
    else:
        # User-specified parameters
        all_parameters = [
            p for p in parameters if p in numeric_params or p in string_params
        ]
        if not all_parameters:
            print(f"Warning: None of the specified parameters are available.")
            print(f"Available numeric: {numeric_params}")
            print(f"Available string: {string_params}")
            return

    if not all_parameters:
        print("No parameters found to display.")
        return

    # Calculate column widths and build table components
    max_comp_width, param_widths = calculate_column_widths(
        all_parameters, component_names, component_data
    )
    header, separator = build_table_header(max_comp_width, all_parameters, param_widths)

    # Print table with appropriate title/subtitle
    core_level = core_level_override or fit_data.get("Core Level") or "Unknown"
    # Prefer provided file name (from parent in multi-core), else from this dict
    display_file = file_name_override or fit_data.get("File Name", "Unknown file")

    if show_main_title and not show_subtitle:
        # Single core level or selected core level - include core level in title
        title = f"Average values from fit report - {core_level}"
        file_name_line = f"File: {display_file}"
        print(
            f"{'═' * int(np.floor((len(header) - len(title)) / 2 - 1))}",
            title,
            f"{'═' * int(np.ceil((len(header) - len(title)) / 2 - 1))}",
        )
        print(
            f"{'═' * int(np.floor((len(header) - len(file_name_line)) / 2 - 1))}",
            file_name_line,
            f"{'═' * int(np.ceil((len(header) - len(file_name_line)) / 2 - 1))}\n",
        )
    elif show_main_title and show_subtitle:
        # Multi-core format, first table - print main title only
        main_title = "Average values from fit report"
        file_name_line = f"File: {display_file}"
        print(
            f"{'═' * int(np.floor((len(header) - len(main_title)) / 2 - 1))}",
            main_title,
            f"{'═' * int(np.ceil((len(header) - len(main_title)) / 2 - 1))}",
        )
        print(
            f"{'═' * int(np.floor((len(header) - len(file_name_line)) / 2 - 1))}",
            file_name_line,
            f"{'═' * int(np.ceil((len(header) - len(file_name_line)) / 2 - 1))}",
        )

    if show_subtitle:
        # Print core level as subtitle
        subtitle = f"Core Level: {core_level}"
        print(f"\n{subtitle}")
        print(f"{'-' * len(subtitle)}")

    print(header)
    print(separator)

    for comp_name in component_names:
        print(
            build_table_row(
                comp_name, all_parameters, component_data, param_widths, max_comp_width
            )
        )

    print(f"{'═' * len(header)}")
