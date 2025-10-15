"""
XPS output formatting utilities.

This module contains functions for formatting and displaying XPS analysis results.
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
