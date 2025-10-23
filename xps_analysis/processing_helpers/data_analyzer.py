"""XPS data analysis utilities.

This module contains functions for analyzing XPS fit data,
including numeric value extraction and parameter processing.

Public Functions
----------------
process_parameter : Process a single parameter and update component data
extract_component_names : Extract component names from fit data
process_all_parameters : Process all parameters and organize by component
"""

import numpy as np


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


def extract_component_names(fit_data):
    """Extract component names from fit data.

    Parameters
    ----------
    fit_data : dict
        Dictionary with 'Name' key containing component names.

    Returns
    -------
    list[str]
        List of component names extracted from the Name array.
    """
    name_array = fit_data["Name"]
    if name_array.ndim == 2:
        return [str(name_array[i, 0]) for i in range(name_array.shape[0])]
    else:
        return [str(name_array[0]) if name_array.size > 0 else "Unknown"]


def process_all_parameters(fit_data, component_names):
    """Process all parameters in fit data and organize by component.

    Parameters
    ----------
    fit_data : dict
        Dictionary returned by read_report_file.
    component_names : list[str]
        List of component names.

    Returns
    -------
    tuple[list, list, dict]
        (numeric_parameters, string_parameters, component_data) where
        numeric_parameters and string_parameters are lists of parameter names,
        and component_data is a dict mapping component names to their values.
    """
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

    return numeric_parameters, string_parameters, component_data


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
