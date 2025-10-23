"""XPS output formatting utilities.

This module contains functions for formatting and displaying XPS analysis results.

Public Functions
----------------
format_table_value : Format a value for table display with width constraints
calculate_column_widths : Calculate optimal column widths for table display
build_table_header : Build formatted table header and separator
build_table_row : Build a single formatted table row
"""


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
