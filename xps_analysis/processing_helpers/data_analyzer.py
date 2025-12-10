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
        # Handle 3D constraint arrays (min, max pairs) for Pos Constr. and FWHM Constr.
        if data_array.ndim == 3 and data_array.shape[0] == 2:
            # Shape is (2, n_components, n_measurements)
            # Extract min and max constraint values
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < data_array.shape[1]:
                    min_vals = data_array[0, comp_idx, :]
                    max_vals = data_array[1, comp_idx, :]
                    # Average the min and max across measurements
                    avg_min = np.mean(min_vals)
                    avg_max = np.mean(max_vals)
                    # Display as "min , max" or just the value if they're equal
                    if np.isclose(avg_min, avg_max):
                        component_data[comp_name][display_key] = f"{avg_min:.2f}"
                    else:
                        component_data[comp_name][
                            display_key
                        ] = f"{avg_min:.2f} , {avg_max:.2f}"
            return display_key, False  # Treat as string since we format it
        else:
            # Regular 2D numeric array
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < data_array.shape[0]:
                    numeric_values = _extract_numeric_values(data_array[comp_idx, :])
                    if numeric_values:
                        component_data[comp_name][display_key] = np.mean(numeric_values)
            return display_key, True
    else:
        # Process string/object parameter (including Area Constr. with mixed types)
        # Handle special case: 1D array containing lists of mixed constraint data
        if (
            data_array.ndim == 1
            and data_array.size > 0
            and isinstance(data_array[0], list)
        ):
            # This is a constraint array with mixed types stored as 1D array of lists
            # Each element is a list containing constraint values for all measurements
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < len(component_names):
                    # Get the constraint data for all measurements
                    constraint_list = []
                    for measurement_list in data_array:
                        if comp_idx < len(measurement_list):
                            constraint_list.append(measurement_list[comp_idx])

                    if not constraint_list:
                        continue

                    # Check if constraints are numeric ranges [min, max] or formulas
                    if (
                        isinstance(constraint_list[0], list)
                        and len(constraint_list[0]) >= 2
                    ):
                        # Numeric range constraints
                        min_vals = [
                            c[0]
                            for c in constraint_list
                            if isinstance(c, list) and len(c) >= 2
                        ]
                        max_vals = [
                            c[1]
                            for c in constraint_list
                            if isinstance(c, list) and len(c) >= 2
                        ]
                        if min_vals and max_vals:
                            avg_min = np.mean(min_vals)
                            avg_max = np.mean(max_vals)
                            if np.isclose(avg_min, avg_max):
                                component_data[comp_name][
                                    display_key
                                ] = f"{avg_min:.2f}"
                            else:
                                component_data[comp_name][
                                    display_key
                                ] = f"{avg_min:.2f} , {avg_max:.2f}"
                    elif isinstance(constraint_list[0], str):
                        # Formula constraints
                        unique_values = list(set(constraint_list))
                        component_data[comp_name][display_key] = (
                            unique_values[0]
                            if len(unique_values) == 1
                            else f"MIXED: {', '.join(unique_values)}"
                        )
        else:
            # Regular 2D string/object array
            for comp_idx, comp_name in enumerate(component_names):
                if comp_idx < data_array.shape[0]:
                    # Check if this is Area Constr. with mixed list/string values
                    values = (
                        data_array[comp_idx, :]
                        if data_array.ndim > 1
                        else [data_array[comp_idx]]
                    )
                    if len(values) > 0 and isinstance(values[0], list):
                        # Handle list-based constraints [min, max]
                        min_vals = [
                            v[0] for v in values if isinstance(v, list) and len(v) >= 2
                        ]
                        max_vals = [
                            v[1] for v in values if isinstance(v, list) and len(v) >= 2
                        ]
                        if min_vals and max_vals:
                            avg_min = np.mean(min_vals)
                            avg_max = np.mean(max_vals)
                            if np.isclose(avg_min, avg_max):
                                component_data[comp_name][
                                    display_key
                                ] = f"{avg_min:.2f}"
                            else:
                                component_data[comp_name][
                                    display_key
                                ] = f"{avg_min:.2f} , {avg_max:.2f}"
                    else:
                        # Handle string values (formulas like "A * 0.5")
                        string_values = [v for v in values if isinstance(v, str)]
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

    Handles both regular 2D arrays and NaN-containing arrays from aligned datasets
    where components may be missing in some measurements.

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
        # Extract component names, taking the first non-NaN value for each row
        component_names = []
        for i in range(name_array.shape[0]):
            # Find first non-NaN name in this row
            for j in range(name_array.shape[1]):
                name = name_array[i, j]
                # Skip NaN values (missing components in some datasets)
                if isinstance(name, str) and name:
                    component_names.append(name)
                    break
            else:
                # All values are NaN or empty - shouldn't happen but handle it
                component_names.append("Unknown")
        return component_names
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

    skip_keys = {
        "File Name",
        "Name",
        "Core Level",
        "Doublet Group",
    }
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
    """Extract all numeric values from a component row, handling lists and scalars.

    Filters out NaN values which represent missing data from datasets with
    different component sets.
    """
    numeric_values = []
    for value in comp_row:
        # Skip NaN values (missing components)
        if isinstance(value, float) and np.isnan(value):
            continue
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
