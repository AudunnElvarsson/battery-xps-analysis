"""
XPS data analysis utilities.

This module contains functions for analyzing XPS fit data,
including numeric value extraction and parameter processing.
"""

import numpy as np


def _extract_numeric_values(comp_row):
    """Extract all numeric values from a component row, handling lists and scalars."""
    numeric_values = []
    for value in comp_row:
        if isinstance(value, (int, float, np.integer, np.floating)) and not np.isnan(
            value
        ):
            numeric_values.append(float(value))
        elif isinstance(value, list):
            numeric_values.extend(
                float(item)
                for item in value
                if isinstance(item, (int, float, np.integer, np.floating))
                and not np.isnan(item)
            )
    return numeric_values


def _is_numeric_array(data_array):
    """Check if array contains numeric data."""
    if data_array.dtype != object:
        return True
    first_elem = data_array.flat[0] if data_array.size > 0 else None
    return isinstance(first_elem, (int, float, np.integer, np.floating))


def process_parameter(key, data_array, component_names, component_data):
    """Process a single parameter and update component_data."""
    # Rename keys for display
    display_key = {"Binding Energy (eV)": "BE", "Comp Label": "Label"}.get(key, key)

    if _is_numeric_array(data_array):
        # Process numeric parameter
        for comp_idx, comp_name in enumerate(component_names):
            if comp_idx < data_array.shape[0]:
                numeric_values = _extract_numeric_values(data_array[comp_idx, :])
                if numeric_values:
                    component_data[comp_name][display_key] = np.mean(numeric_values)
        return display_key, True
    else:
        # Process string parameter
        for comp_idx, comp_name in enumerate(component_names):
            if comp_idx < data_array.shape[0]:
                string_values = [
                    v for v in data_array[comp_idx, :] if isinstance(v, str)
                ]
                if string_values:
                    unique_values = list(set(string_values))
                    component_data[comp_name][display_key] = (
                        unique_values[0]
                        if len(unique_values) == 1
                        else f"MIXED: {', '.join(unique_values)}"
                    )
        return display_key, False
