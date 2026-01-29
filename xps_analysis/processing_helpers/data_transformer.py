"""Data transformation utilities for XPS report data.

This module contains functions for transforming XPS fit report data,
including normalization, ratio calculations, and relative binding energy
conversions.

Public Functions
----------------
normalize_at_conc_per_core : Normalize atomic concentrations per core level
calculate_area_ratios : Calculate area ratios relative to reference component
convert_to_relative_be : Convert binding energies to relative values
find_component_index : Find component index by label or name
"""

import numpy as np


def find_component_index(core_data, component_label):
    """Return the row index for a given component label or name.

    Searches both 'Comp Label' and 'Name' fields to find a match.
    Component labels are typically single letters (A, B, C), while
    component names are chemical species (LiF, PFx, C-C / C-H).

    Parameters
    ----------
    core_data : dict
        Core level data dictionary.
    component_label : str
        Component label or name to search for.

    Returns
    -------
    int or None
        Index of the component, or None if not found.
    """
    if not component_label:
        return 0

    # Try Comp Label first (typically single letters like A, B, C)
    comp_labels = core_data.get("Comp Label")
    if comp_labels is not None:
        comp_labels = np.array(comp_labels, dtype=object)
        for idx in range(comp_labels.shape[0]):
            label = comp_labels[idx, 0] if comp_labels.ndim > 1 else comp_labels[idx]
            if str(label) == str(component_label):
                return idx

    # Try Name field (chemical species names)
    names = core_data.get("Name")
    if names is not None:
        names = np.array(names, dtype=object)
        for idx in range(names.shape[0]):
            name = names[idx, 0] if names.ndim > 1 else names[idx]
            if str(name) == str(component_label):
                return idx

    return None


def normalize_at_conc_per_core(core_data, col_full="%At Conc"):
    """Normalize atomic concentration values to sum to 100% per core level.

    Parameters
    ----------
    core_data : dict
        Core level data dictionary.
    col_full : str, optional
        Column name to normalize (default "%At Conc").

    Returns
    -------
    dict
        A copy of core_data with normalized atomic concentration values.
    """
    if col_full != "%At Conc" or col_full not in core_data:
        return core_data

    # Create a copy to avoid modifying the original
    normalized_data = core_data.copy()

    at_conc_data = np.array(normalized_data["%At Conc"], dtype=object)

    # Normalize each measurement to sum to 100%
    if at_conc_data.ndim > 1:
        for measurement_idx in range(at_conc_data.shape[1]):
            # Extract values for this measurement
            values = []
            indices = []
            for comp_idx in range(at_conc_data.shape[0]):
                val = at_conc_data[comp_idx, measurement_idx]
                if isinstance(val, (int, float, np.floating)):
                    values.append(val)
                    indices.append(comp_idx)

            # Normalize if we have valid numeric values
            if values:
                total = sum(values)
                if total > 0:
                    scale_factor = 100.0 / total
                    for comp_idx, val in zip(indices, values):
                        at_conc_data[comp_idx, measurement_idx] = val * scale_factor

        normalized_data["%At Conc"] = at_conc_data

    return normalized_data


def _safe_ratio(numerator, denominator):
    """Compute a safe ratio, returning NaN when division is invalid."""
    if not isinstance(numerator, (int, float, np.integer, np.floating)):
        return np.nan
    if denominator is None or not isinstance(
        denominator, (int, float, np.integer, np.floating)
    ):
        return np.nan
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def calculate_area_ratios(core_data, col_full, reference_component=None):
    """Calculate area ratios relative to a reference component.

    Creates a new dataset with area values converted to ratios relative
    to the specified reference component. The reference component is
    excluded from the output unless it's the only component.

    Parameters
    ----------
    core_data : dict
        Core level data dictionary.
    col_full : str
        Column name containing area data (e.g., "Raw Area").
    reference_component : str, optional
        Component label or name to use as reference. If None, uses first component.

    Returns
    -------
    tuple of (dict, str)
        - New dictionary with ratio values
        - New column name for the ratios
    """
    if col_full not in core_data:
        return core_data, col_full

    area_array = np.array(core_data.get(col_full), dtype=object)
    if area_array.size == 0:
        return core_data, col_full

    # Find reference component
    ref_idx = find_component_index(core_data, reference_component)
    if ref_idx is None or ref_idx >= area_array.shape[0]:
        return core_data, col_full

    # Get reference series
    if area_array.ndim == 1:
        ref_array = np.array([area_array[ref_idx]], dtype=object)
    else:
        ref_array = area_array[ref_idx]

    ref_array = np.array(ref_array, dtype=object).flatten()

    # Get the actual component name for labeling
    ref_component_name = reference_component
    if ref_idx is not None:
        names = core_data.get("Name")
        if names is not None:
            names_array = np.array(names, dtype=object)
            if names_array.ndim > 1 and names_array.shape[1] > 1:
                ref_component_name = names_array[ref_idx, 1]
            elif names_array.ndim == 1:
                ref_component_name = names_array[ref_idx]

    # Calculate ratios
    if area_array.ndim == 1:
        ratio_array = np.empty_like(area_array, dtype=object)
        for j in range(area_array.shape[0]):
            ref_val = ref_array[j] if j < ref_array.shape[0] else None
            ratio_array[j] = _safe_ratio(area_array[j], ref_val)
    else:
        ratio_array = np.empty_like(area_array, dtype=object)
        for i in range(area_array.shape[0]):
            for j in range(area_array.shape[1]):
                ref_val = ref_array[j] if j < ref_array.shape[0] else None
                ratio_array[i, j] = _safe_ratio(area_array[i, j], ref_val)

    new_col = "Area Ratio"
    if ref_component_name:
        new_col = f"Area Ratio (ref comp {ref_component_name})"

    ratio_dict = core_data.copy()
    ratio_dict[new_col] = ratio_array

    # Remove the reference component from output if there are multiple components
    num_components = area_array.shape[0]
    if ref_idx is not None and num_components > 1:
        for key in ratio_dict:
            if key in ["Name", "Comp Label", new_col, col_full]:
                arr = np.array(ratio_dict[key], dtype=object)
                if arr.ndim > 0 and arr.shape[0] > ref_idx:
                    ratio_dict[key] = np.delete(arr, ref_idx, axis=0)

    return ratio_dict, new_col


def convert_to_relative_be(core_data, col_full, reference_component):
    """Convert binding energy values to relative values with respect to a reference.

    Parameters
    ----------
    core_data : dict
        Core level data dictionary containing BE data and component labels.
    col_full : str
        Full column name for binding energy (e.g., "Binding Energy (eV)").
    reference_component : str
        Component label or name to use as reference (e.g., "A" or "PFx").

    Returns
    -------
    dict
        New dictionary with relative BE values, or original dict if conversion fails.
    """
    if col_full not in core_data:
        return core_data

    be_data = core_data.get(col_full)
    if be_data is None:
        return core_data

    # Find reference component index
    ref_idx = find_component_index(core_data, reference_component)
    if ref_idx is None:
        print(
            f"Warning: Reference component '{reference_component}' not found. Using absolute BE."
        )
        return core_data

    # Get reference BE value (first measurement)
    be_array = np.array(be_data, dtype=object)
    if be_array.ndim == 1:
        ref_be = be_array[ref_idx] if ref_idx < be_array.shape[0] else None
    else:
        ref_be = be_array[ref_idx, 0] if ref_idx < be_array.shape[0] else None

    if not isinstance(ref_be, (int, float, np.integer, np.floating)):
        print("Warning: Reference BE is not numeric. Using absolute BE.")
        return core_data

    # Convert all BE values to relative
    if be_array.ndim == 1:
        relative_be = np.array(
            [
                (
                    be_array[i] - ref_be
                    if isinstance(be_array[i], (int, float, np.integer, np.floating))
                    else be_array[i]
                )
                for i in range(be_array.shape[0])
            ],
            dtype=object,
        )
    else:
        relative_be = np.empty_like(be_array, dtype=object)
        for i in range(be_array.shape[0]):
            for j in range(be_array.shape[1]):
                val = be_array[i, j]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    relative_be[i, j] = val - ref_be
                else:
                    relative_be[i, j] = val

    # Create new dict with relative BE
    new_dict = core_data.copy()
    new_dict["Relative Binding Energy (eV)"] = relative_be
    return new_dict
