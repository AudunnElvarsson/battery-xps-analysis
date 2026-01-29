"""Report plotting utilities for XPS fit reports.

This module contains functions for plotting XPS fit report data including
single and multi-core level plotting with axis configuration.

Public Functions
----------------
configure_report_axes : Configure axis labels, title, and legend for report plot
plot_single_core_level : Plot a single core level's data
plot_all_core_levels : Plot multiple core levels in separate subplots
"""

import matplotlib.pyplot as plt
import numpy as np
from .file_operations import save_figure, derive_file_name
from .axis_management import ensure_axes_and_main
from .plot_rendering import (
    get_column_name,
    update_plot_params,
    plot_report_series,
    convert_to_relative_be,
)


def _normalize_at_conc_per_core(report_dict, core_data, col_full):
    """Normalize atomic concentration values to sum to 100% per core level.

    Parameters
    ----------
    report_dict : dict
        The data dictionary to normalize (modified in place).
    core_data : dict
        Reference to the core data being plotted.
    col_full : str
        The full column name to normalize.

    Returns
    -------
    dict
        A copy of core_data with normalized atomic concentration values.
    """
    if col_full != "%At Conc":
        return core_data

    # Create a copy to avoid modifying the original
    normalized_data = core_data.copy()

    if "%At Conc" not in normalized_data:
        return normalized_data

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


def _sum_doublet_values(core_data, col_full):
    """Sum numeric values for spin-orbit doublet components.

    For components marked as doublets (e.g., P-F (3/2) and P-F (1/2)),
    sum their values and create a new dataset with one entry per doublet pair.
    Non-doublet components are kept as-is.

    This function works for any numeric column including areas, atomic
    concentrations, and other fit parameters.

    Parameters
    ----------
    core_data : dict
        Core level data dictionary.
    col_full : str
        Column name to sum (e.g., "Raw Area", "%At Conc").

    Returns
    -------
    dict
        New dictionary with doublets summed. If no doublets exist, returns
        the original data unchanged.
    """
    if "Doublet Group" not in core_data or col_full not in core_data:
        return core_data

    doublet_groups = core_data["Doublet Group"]
    if doublet_groups is None or not hasattr(doublet_groups, "__iter__"):
        return core_data

    # Check if any doublets exist
    has_doublets = any(str(g) != "" for g in doublet_groups)
    if not has_doublets:
        return core_data

    # Get the data array for the specified column
    data_array = np.array(core_data[col_full], dtype=object)
    name_array = np.array(core_data["Name"], dtype=object)

    # Identify which rows to keep and which to sum
    processed_groups = set()
    new_rows = []
    new_names = []

    for i in range(len(doublet_groups)):
        group = str(doublet_groups[i])

        if group == "" or group in processed_groups:
            # Not a doublet or already processed
            if group == "":
                # Keep non-doublet components
                new_rows.append(i)
                if name_array.ndim > 1:
                    new_names.append(name_array[i, 0])
                else:
                    new_names.append(name_array[i])
            continue

        # Find all components in this doublet group
        doublet_indices = [j for j, g in enumerate(doublet_groups) if str(g) == group]

        if len(doublet_indices) == 2:
            # Sum the two components
            idx1, idx2 = doublet_indices
            processed_groups.add(group)

            # Create summed row
            if data_array.ndim == 1:
                summed_value = data_array[idx1] + data_array[idx2]
            else:
                summed_value = data_array[idx1] + data_array[idx2]

            new_rows.append(idx1)  # Use first component's index as template
            new_names.append(group)  # Use base name without suffix

    # Create new core_data with summed doublets
    summed_data = core_data.copy()

    # Build new data array
    if data_array.ndim == 1:
        new_data = []
        for i, orig_idx in enumerate(new_rows):
            group = str(doublet_groups[orig_idx])
            if group != "" and group in processed_groups:
                # This is a doublet - sum it
                doublet_indices = [
                    j for j, g in enumerate(doublet_groups) if str(g) == group
                ]
                summed = sum(
                    data_array[j]
                    for j in doublet_indices
                    if isinstance(data_array[j], (int, float, np.integer, np.floating))
                )
                new_data.append(summed)
            else:
                new_data.append(data_array[orig_idx])
        summed_data[col_full] = np.array(new_data, dtype=object)
    else:
        new_data = []
        for i, orig_idx in enumerate(new_rows):
            group = str(doublet_groups[orig_idx])
            if group != "" and group in processed_groups:
                # This is a doublet - sum across measurements
                doublet_indices = [
                    j for j, g in enumerate(doublet_groups) if str(g) == group
                ]
                summed_row = np.zeros_like(data_array[orig_idx], dtype=float)
                for j in doublet_indices:
                    for k in range(data_array.shape[1]):
                        val = data_array[j, k]
                        if isinstance(val, (int, float, np.integer, np.floating)):
                            summed_row[k] += val
                new_data.append(summed_row)
            else:
                new_data.append(data_array[orig_idx])
        summed_data[col_full] = np.array(new_data, dtype=object)

    # Update Name array
    if name_array.ndim > 1:
        new_name_array = np.empty((len(new_names), name_array.shape[1]), dtype=object)
        for i, name in enumerate(new_names):
            new_name_array[i, 0] = name
            # Copy other columns if they exist
            orig_idx = new_rows[i]
            for j in range(1, name_array.shape[1]):
                new_name_array[i, j] = name_array[orig_idx, j]
        summed_data["Name"] = new_name_array
    else:
        summed_data["Name"] = np.array(new_names, dtype=object)

    # Update other arrays (Comp Label, etc.)
    for key in summed_data:
        if key in [col_full, "Name", "Doublet Group", "File Name", "Core Level"]:
            continue
        if isinstance(summed_data[key], np.ndarray):
            arr = summed_data[key]
            if arr.shape[0] == len(doublet_groups):
                # This array needs to be filtered
                if arr.ndim == 1:
                    summed_data[key] = np.array(
                        [arr[i] for i in new_rows], dtype=object
                    )
                else:
                    summed_data[key] = np.array(
                        [arr[i] for i in new_rows], dtype=object
                    )

    return summed_data


def _find_component_index(core_data, component_label):
    """Return the row index for a given component label or name.

    Searches both 'Comp Label' and 'Name' fields to find a match.
    Component labels are typically single letters (A, B, C), while
    component names are chemical species (LiF, PFx, C-C / C-H).
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


def _get_reference_area_series(core_data, col_full, component_label=None):
    """Return the reference component's area series for ratio calculations."""
    if core_data is None or col_full not in core_data:
        return None

    area_array = np.array(core_data.get(col_full), dtype=object)
    if area_array.size == 0:
        return None

    ref_idx = _find_component_index(core_data, component_label)
    if ref_idx is None:
        return None

    if area_array.ndim == 1:
        return area_array[ref_idx] if ref_idx < area_array.shape[0] else None

    if ref_idx >= area_array.shape[0]:
        return None

    return area_array[ref_idx]


def _apply_area_ratio(core_data, col_full, reference_series, reference_label=None):
    """Create a copy of core_data with area values converted to ratios.

    The reference component is excluded from the output unless it's the only component,
    in which case it's kept to show a ratio of 1.
    """
    if reference_series is None or col_full not in core_data:
        return core_data, col_full

    area_array = np.array(core_data.get(col_full), dtype=object)
    ref_array = np.array(reference_series, dtype=object).flatten()

    # Find the reference component index
    ref_idx = _find_component_index(core_data, reference_label)

    # Get the actual component name (not label) for the reference
    ref_component_name = reference_label
    if ref_idx is not None:
        names = core_data.get("Name")
        if names is not None:
            names_array = np.array(names, dtype=object)
            if names_array.ndim > 1 and names_array.shape[1] > 1:
                # Use the component name (column 1) instead of label (column 0)
                ref_component_name = names_array[ref_idx, 1]
            elif names_array.ndim == 1:
                ref_component_name = names_array[ref_idx]

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

    # Check if there's only one component
    num_components = area_array.shape[0]

    # Remove the reference component from the output only if there are multiple components
    if ref_idx is not None and num_components > 1:
        for key in ratio_dict:
            if key in ["Name", "Comp Label", new_col, col_full]:
                arr = np.array(ratio_dict[key], dtype=object)
                if arr.ndim > 0 and arr.shape[0] > ref_idx:
                    # Remove the reference component row
                    ratio_dict[key] = np.delete(arr, ref_idx, axis=0)

    return ratio_dict, new_col


def configure_report_axes(ax, col_full, core_level):
    """Configure axis labels, title, and legend for report plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis to configure.
    col_full : str
        Full column name being plotted (e.g., "Binding Energy (eV)").
    core_level : str
        Core level name for title (e.g., "C 1s", "O 1s").

    Notes
    -----
    For binding energy plots, axes are swapped (BE on x-axis) and both axes
    are inverted. Legend uses monospace font for alignment.
    """
    swap_axes = "Binding Energy" in col_full

    # Simplify area ratio label to just "Area Ratio"
    y_label = col_full
    if col_full.startswith("Area Ratio (ref comp "):
        y_label = "Area Ratio"

    if swap_axes:
        ax.set_xlabel(col_full)
        ax.set_ylabel("Experimental Variable")
        ax.invert_xaxis()
        ax.invert_yaxis()
    else:
        ax.set_xlabel("Experimental Variable")
        ax.set_ylabel(y_label)

    ax.set_title(core_level)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_family("monospace")


def plot_single_core_level(
    report_dict,
    ax,
    proc_kwargs,
    plot_kwargs,
    save_kwargs,
    core_level_override=None,
    ratio_reference_core_data=None,
):
    """Plot a single core level's fit report data.

    This function handles plotting for a single core level, including
    converting to relative binding energy if requested, configuring axes,
    and saving the figure.

    Parameters
    ----------
    report_dict : dict
        Single core level data dictionary with 'Name' and parameter arrays.
    ax : matplotlib.axes.Axes or None
        Target axis or None to create new figure and axis.
    proc_kwargs : dict
        Processing options. Supported keys:
        - 'fit_param' (str): Parameter to plot (default 'BE').
        - 'calculate' (str or None): Display statistic. 'average' shows mean,
            'difference' shows last - first, 'ratio' shows mean area ratio,
            None/empty disables statistics/legend.
        - 'reference' (str or dict): Component label/name for relative BE plotting
            and area ratio calculations. Can be component label (e.g., 'A') or
            component name (e.g., 'LiF'). If str, applies to all core levels; if
            dict, maps core level -> component label/name (e.g., {"C 1s": "A",
            "O 1s": "LiF"}). For area ratios, defaults to each core's first
            component if not specified.
        - 'show_labels' (bool): If True, prepend component labels (e.g., "A", "B")
            to component names in legend (default False).
    plot_kwargs : dict or None
        Plot styling options forwarded to matplotlib plot.
    save_kwargs : dict
        Save options. Supported keys:
        - 'save_fig' (bool): Whether to save the figure.
        - 'save_folder' (str): Directory path for saving.
        - 'format' (str): File format (e.g., 'png', 'pdf').
        - 'dpi' (int): Resolution for raster formats.
    core_level_override : str or None, optional
        Override for core level name in title. If None, uses value from
        report_dict['Core Level'] or 'Unknown'.
    ratio_reference_core_data : dict or None, optional
        Core level data to use as reference when calculate='ratio' and
        fit_param='Area'. Defaults to the current core level data.

    Returns
    -------
    None
    """
    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    calculate_mode = proc_kwargs.get("calculate", "average")
    core_level_name = core_level_override or report_dict.get("Core Level") or "Unknown"
    if isinstance(core_level_name, (list, tuple, np.ndarray)):
        core_level_name = core_level_name[0] if len(core_level_name) else "Unknown"

    # Get column name and reference
    col_full = get_column_name(proc_kwargs.get("fit_param", "BE"))
    reference = proc_kwargs.get("reference", None)
    plot_dict = report_dict

    # For area and atomic concentration parameters, sum doublet components before processing
    if "Area" in col_full or col_full == "%At Conc":
        plot_dict = _sum_doublet_values(plot_dict, col_full)

    # Convert areas to ratios when requested
    if calculate_mode == "ratio" and col_full == "Raw Area":
        # Resolve reference component label (per-core mapping or global string)
        if isinstance(reference, dict):
            ref_component_label = reference.get(core_level_name, None)
        else:
            ref_component_label = reference

        ratio_base = ratio_reference_core_data or plot_dict
        ratio_series = _get_reference_area_series(
            ratio_base, col_full, ref_component_label
        )

        if ratio_series is None:
            print(
                f"Warning: Unable to compute area ratio for {core_level_name}; using raw areas instead."
            )
        else:
            # Use provided reference or default to first component
            ref_label = ref_component_label
            if not ref_label:
                names = plot_dict.get("Name") or plot_dict.get("Comp Label")
                names = np.array(names, dtype=object) if names is not None else None
                if names is not None and names.shape[0] > 0:
                    ref_label = str(names[0, 0] if names.ndim > 1 else names[0])
                else:
                    ref_label = "?"
            plot_dict, col_full = _apply_area_ratio(
                plot_dict, col_full, ratio_series, ref_label
            )

    # Convert to relative BE if requested
    if reference and "Binding Energy" in col_full:
        # Resolve per-core reference if dict provided
        ref_component = reference
        if isinstance(reference, dict):
            ref_component = reference.get(core_level_name, None)
        if ref_component:
            converted_dict = convert_to_relative_be(plot_dict, col_full, ref_component)
            if "Relative Binding Energy (eV)" in converted_dict:
                plot_dict = converted_dict
                col_full = "Relative Binding Energy (eV)"

    # Plot the data
    plot_report_series(
        plot_dict,
        main_ax,
        col_full,
        update_plot_params({"ls": "--", "lw": 1.5, "m": "o"}, plot_kwargs),
        {
            "calculate": calculate_mode,
            "swap_axes": "Binding Energy" in col_full,
            "show_labels": proc_kwargs.get("show_labels", False),
        },
    )

    # Configure axes and legend
    configure_report_axes(
        main_ax,
        col_full,
        core_level_override or report_dict.get("Core Level") or "Unknown",
    )

    # Save figure if requested
    if save_kwargs.get("save_fig", False):
        save_figure(
            fig,
            name_list=[derive_file_name(report_dict)],
            save_args=save_kwargs,
            prefix="",
        )

    if plot_here:
        plt.show()


def plot_all_core_levels(
    report_dict,
    core_levels,
    ax,
    proc_kwargs,
    plot_kwargs,
    save_kwargs,
):
    """Plot multiple core levels in separate subplots.

    This function creates a grid of subplots for multiple core levels,
    with each core level plotted in its own subplot. If axes are provided,
    they are used; otherwise new subplots are created.

    Parameters
    ----------
    report_dict : dict
        Multi-core format dictionary with core level keys (e.g., "C 1s",
        "F 1s") containing data dictionaries.
    core_levels : list of str
        List of core level names to plot.
    ax : matplotlib.axes.Axes, array of Axes, or None
        Target axes to use. If None or if the number doesn't match core_levels,
        new subplots are created. If an array of axes is provided with
        sufficient elements, they will be used.
    proc_kwargs : dict
        Processing options. Supported keys:
        - 'fit_param' (str): Parameter to plot (default 'BE').
        - 'calculate' (str or None): Display statistic. 'average' shows mean,
            'difference' shows last - first, 'ratio' shows mean area ratio,
            None/empty disables statistics/legend.
        - 'reference' (str or dict): Component label/name for relative BE plotting
            and area ratio calculations. Can be component label (e.g., 'A') or
            component name (e.g., 'LiF'). If str, applies to all core levels; if
            dict, maps core level -> component label/name (e.g., {"C 1s": "A",
            "O 1s": "LiF"}). For area ratios, defaults to each core's first
            component if not specified.
        - 'show_labels' (bool): If True, prepend component labels (e.g., "A", "B")
            to component names in legend (default False).
        - 'normalize_at_conc_per_core' (bool): When True, normalize atomic
            concentrations separately for each core level so components within
            each core sum to 100% (default False).
    plot_kwargs : dict or None
        Plot styling options forwarded to matplotlib plot.
    save_kwargs : dict
        Save options. Supported keys:
        - 'save_fig' (bool): Whether to save the figure.
        - 'save_folder' (str): Directory path for saving.
        - 'format' (str): File format (e.g., 'png', 'pdf').
        - 'dpi' (int): Resolution for raster formats.

    Returns
    -------
    None

    Notes
    -----
    Subplots are arranged in a grid with up to 3 columns. Unused subplots
    are hidden if new subplots are created by this function.
    """
    n_cores = len(core_levels)

    # Check if we have a valid array of axes provided
    axes_provided = False
    if ax is not None:
        try:
            # Try to get it as an array
            ax_array = np.atleast_1d(ax)
            # Flatten if it's a 2D array from subplots
            if ax_array.ndim > 1:
                ax_array = ax_array.flatten()
            # Check if we have the right number
            if len(ax_array) >= n_cores:
                axes = ax_array
                axes_provided = True
                fig = axes[0].get_figure()
        except (TypeError, AttributeError):
            # Not an array, might be a single axis
            if hasattr(ax, "get_figure"):
                # Single axis provided for multi-core - can't use it
                axes_provided = False

    # Create new subplots if axes not provided or incompatible
    if not axes_provided:
        # Create subplots - arrange in grid
        n_cols = min(3, n_cores)  # Max 3 columns
        n_rows = (n_cores + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        fig.set_tight_layout(True)
        axes = axes.flatten()
        plot_here = True
    else:
        # Using provided axes
        plot_here = False

    # Get column name and calculation mode
    col_full = get_column_name(proc_kwargs.get("fit_param", "BE"))
    calculate_mode = proc_kwargs.get("calculate", "average")
    reference = proc_kwargs.get("reference", None)
    normalize_at_conc = proc_kwargs.get("normalize_at_conc_per_core", False)

    # Plot each core level
    for idx, core_level in enumerate(core_levels):
        current_ax = axes[idx]

        # Get data for this core level
        core_data = report_dict[core_level]
        plot_dict = core_data
        plot_col = col_full

        # For area and atomic concentration parameters, sum doublet components before processing
        if "Area" in col_full or col_full == "%At Conc":
            plot_dict = _sum_doublet_values(plot_dict, col_full)

        # Apply area ratio conversion per core level if requested
        if calculate_mode == "ratio" and col_full == "Raw Area":
            if isinstance(reference, dict):
                ref_component_label = reference.get(core_level, None)
            else:
                ref_component_label = reference

            ratio_reference_series = _get_reference_area_series(
                plot_dict, col_full, ref_component_label
            )

            if ratio_reference_series is None:
                print(
                    f"Warning: Unable to compute area ratio for {core_level}; using raw areas instead."
                )
            else:
                # Use provided reference or default to first component
                ref_label = ref_component_label
                if not ref_label:
                    names = plot_dict.get("Name")
                    if names is None:
                        names = plot_dict.get("Comp Label")
                    names = np.array(names, dtype=object) if names is not None else None
                    if names is not None and names.shape[0] > 0:
                        ref_label = str(names[0, 0] if names.ndim > 1 else names[0])
                    else:
                        ref_label = "?"
                plot_dict, plot_col = _apply_area_ratio(
                    plot_dict, col_full, ratio_reference_series, ref_label
                )

        # Normalize atomic concentration per core level if requested
        if normalize_at_conc and col_full == "%At Conc":
            plot_dict = _normalize_at_conc_per_core(report_dict, plot_dict, col_full)

        # Convert to relative BE if needed
        if reference and "Binding Energy" in col_full:
            # Resolve per-core reference if dict provided
            ref_component = reference
            if isinstance(reference, dict):
                ref_component = reference.get(core_level, None)
            if ref_component:
                converted_dict = convert_to_relative_be(
                    plot_dict, col_full, ref_component
                )
                if "Relative Binding Energy (eV)" in converted_dict:
                    plot_dict = converted_dict
                    plot_col = "Relative Binding Energy (eV)"

        # Plot the data
        plot_report_series(
            plot_dict,
            current_ax,
            plot_col,
            update_plot_params({"ls": "--", "lw": 1.5, "m": "o"}, plot_kwargs),
            {
                "calculate": calculate_mode,
                "swap_axes": "Binding Energy" in plot_col,
                "show_labels": proc_kwargs.get("show_labels", False),
            },
        )

        # Configure this subplot
        configure_report_axes(current_ax, plot_col, core_level)

    # Hide unused subplots (only if we created them)
    if not axes_provided:
        n_cols = min(3, n_cores)
        n_rows = (n_cores + n_cols - 1) // n_cols
        for idx in range(n_cores, n_rows * n_cols):
            axes[idx].set_visible(False)

    # Save figure if requested
    if save_kwargs.get("save_fig", False):
        save_figure(
            fig,
            name_list=[derive_file_name(report_dict)],
            save_args=save_kwargs,
            prefix="",
        )

    if plot_here:
        plt.show()
