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
)
from ..processing_helpers import (
    collapse_doublet_components,
    normalize_at_conc_per_core,
    calculate_area_ratios,
    convert_to_relative_be,
    find_component_index,
)


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
        - 'normalize_at_conc_per_core' (bool): When plotting atomic
            concentration, normalize components within the current core level so
            they sum to 100% (default False).
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
        plot_dict = collapse_doublet_components(plot_dict, col_full)

    # Normalize atomic concentration values within this core level if requested
    if proc_kwargs.get("normalize_at_conc_per_core", False) and col_full == "%At Conc":
        plot_dict = normalize_at_conc_per_core(plot_dict, col_full)

    # Convert areas to ratios when requested
    if calculate_mode == "ratio" and col_full == "Raw Area":
        # Resolve reference component label (per-core mapping or global string)
        if isinstance(reference, dict):
            ref_component_label = reference.get(core_level_name, None)
        else:
            ref_component_label = reference

        # Use provided reference or default to first component
        if not ref_component_label:
            names = plot_dict.get("Name") or plot_dict.get("Comp Label")
            names = np.array(names, dtype=object) if names is not None else None
            if names is not None and names.shape[0] > 0:
                ref_component_label = str(names[0, 0] if names.ndim > 1 else names[0])

        if ref_component_label:
            plot_dict, col_full = calculate_area_ratios(
                plot_dict, col_full, ref_component_label
            )
        else:
            print(
                f"Warning: Unable to determine reference component for {core_level_name}; using raw areas instead."
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
            plot_dict = collapse_doublet_components(plot_dict, col_full)

        # Apply area ratio conversion per core level if requested
        if calculate_mode == "ratio" and col_full == "Raw Area":
            if isinstance(reference, dict):
                ref_component_label = reference.get(core_level, None)
            else:
                ref_component_label = reference

            # Use provided reference or default to first component
            if not ref_component_label:
                names = plot_dict.get("Name") or plot_dict.get("Comp Label")
                names = np.array(names, dtype=object) if names is not None else None
                if names is not None and names.shape[0] > 0:
                    ref_component_label = str(
                        names[0, 0] if names.ndim > 1 else names[0]
                    )

            if ref_component_label:
                plot_dict, plot_col = calculate_area_ratios(
                    plot_dict, col_full, ref_component_label
                )
            else:
                print(
                    f"Warning: Unable to determine reference component for {core_level}; using raw areas instead."
                )

        # Normalize atomic concentration per core level if requested
        if normalize_at_conc and col_full == "%At Conc":
            plot_dict = normalize_at_conc_per_core(plot_dict, col_full)

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
