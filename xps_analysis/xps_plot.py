"""xps_plot
-----------

High-level plotting functions for XPS spectra and fit reports.

This module provides the main public API for plotting XPS data. Functions
accept dictionaries produced by parsing utilities in ``xps_processing`` and
produce Matplotlib figures/axes with appropriate styling and configuration.

The module supports both single and multi-core level data formats, with
automatic detection and appropriate subplot creation.

Public Functions
----------------
plot_comp_report : Plot fit report parameter(s) across components and core levels
plot_region_report : Plot ratio between total values of two core levels
plot_spectrum : Plot XPS spectrum with optional residuals
"""

import matplotlib.pyplot as plt
import numpy as np

# Import plotting utilities from helper modules
from .plot_helpers import (
    save_figure,
    derive_file_name,
    get_x_axis_from_dict,
    ensure_axes_and_main,
    configure_axes,
    plot_spectrum_series,
    plot_single_core_level,
    plot_all_core_levels,
    update_plot_params,
    get_column_name,
)


def plot_comp_report(
    report_dict,
    ax=None,
    proc_kwargs=None,
    plot_kwargs=None,
    save_kwargs=None,
):
    """Plot a fit-report parameter across components and core levels.

    This is the main public function for plotting XPS fit report data. It
    automatically detects single vs multi-core format and creates appropriate
    plots. For multi-core data, you can select specific core levels or plot
    all of them.

    The function plots one fitted parameter (e.g., binding energy, area,
    atomic concentration) for each component. Average or difference values
    are shown in the legend with monospace formatting for alignment.

    **Spin-Orbit Doublets**: When plotting area or atomic concentration,
    spin-orbit split components (e.g., P 2p3/2 and P 2p1/2) are automatically
    combined to show the total value for the chemical species. This ensures
    that atomic concentrations reflect the total amount of the element in that
    chemical state, not split between the doublet components.

    Parameters
    ----------
    report_dict : dict
        Mapping produced by ``read_report_file``. For single-core format,
        this contains parameter arrays directly. For multi-core format, this
        is a nested dict where keys are core level names (e.g., "C 1s", "F 1s")
        containing data dictionaries, plus metadata keys "Core Level" and
        "File Name".
    ax : matplotlib.axes.Axes, array of Axes, or None, optional
        Target axis or axes to draw on. If ``None``, a new figure and axis
        are created. For multi-core format:
        - Single Axes: Used when plotting one core level
        - Array of Axes: Used when provided axes match the number of core
          levels to plot; otherwise new subplots are created
    proc_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'fit_param' (str): Parameter to plot (default 'BE'). Common short
          forms: "BE" (binding energy), "Area" (raw area), "At Conc" (atomic
          concentration), "Goodness" (goodness of fit).
        - 'calculate' (str or None): Statistic to display in legend. Either
            "average" (shows mean values), "difference" (shows last - first),
            "difference_max" (shows max - min), "ratio" (shows mean area ratio
            when fit_param='Area'), or None/empty to disable statistics and
            legend annotation.
        - 'reference' (str or dict): Component label or name used for both
          relative BE plotting (when fit_param='BE') and area ratio calculations
          (when calculate='ratio' and fit_param='Area'). Can be component label
          (e.g., "A") or component name (e.g., "LiF"). If str, applies to all
          core levels. If dict, maps core level -> component label/name
          (e.g., {"C 1s": "A", "O 1s": "LiF"}). For area ratios, defaults to
          each core's first component if not specified. For relative BE, plots
          binding energy with respect to the first measurement of the reference
          component.
        - 'show_labels' (bool): If True, prepend component labels (e.g., "A", "B")
          to component names in the legend (default False). Useful for quickly
          identifying component labels across different core levels.
        - 'normalize_at_conc_per_core' (bool): When plotting multiple core
          levels with atomic concentration (default False), if True the atomic
          concentrations for each core level will be normalized separately so
          components within each core level sum to 100%. If False, components
          across all core levels sum to 100%.
        - 'core_levels' (str, list of str, or None): For multi-core format,
          specify which core levels to plot:
          * Single string "C 1s" - plots only that core level
          * List ["C 1s", "O 1s"] - plots those core levels
          * None, "", [] or [""] - plots all available core levels
    plot_kwargs : dict or None, optional
        Styling arguments forwarded to ``matplotlib.axes.Axes.plot`` for the
        component series. These override module defaults. Supports both full
        and abbreviated parameter names (e.g., 'ls' or 'linestyle').
        Special handling for colors:
        - 'color' or 'colors': Can be a single color (applied to all components)
          or a list of colors (cycled through for each component).
          Example: {"colors": ["red", "blue", "green"]} or {"color": "#FF5733"}
    save_kwargs : dict or None, optional
        Options for saving the figure. Supported keys:
        - 'save_fig' (bool): Whether to save the figure (default False)
        - 'save_folder' (str): Directory path for saving
        - 'format' (str): File format (e.g., 'png', 'pdf')
        - 'dpi' (int): Resolution for raster formats

    Returns
    -------
    None

    See Also
    --------
    plot_spectrum : Plot XPS spectrum with optional residuals

    Examples
    --------
    Plot binding energy for all core levels in a multi-core file:

    >>> report_dict = xp.read_report_file("multicore_report.txt")
    >>> xplot.plot_comp_report(report_dict)

    Plot specific core levels with difference calculation:

    >>> proc_kwargs = {
    ...     "core_levels": ["C 1s", "O 1s"],
    ...     "calculate": "difference"
    ... }
    >>> xplot.plot_comp_report(report_dict, proc_kwargs=proc_kwargs)

    Plot relative binding energy with custom axes:

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> proc_kwargs = {"reference": "A", "core_levels": ["C 1s", "O 1s"]}
    >>> xplot.plot_comp_report(report_dict, ax=axes, proc_kwargs=proc_kwargs)
    """
    proc_kwargs = proc_kwargs or {}
    save_kwargs = save_kwargs or {}

    if report_dict is None:
        print("No data to plot.")
        return

    # Check if multi-core format (nested dict with core level keys)
    is_multicore = "Core Level" in report_dict and "Name" not in report_dict

    if is_multicore:
        # Multi-core format handling
        all_core_levels = [
            k for k in report_dict.keys() if k not in ["Core Level", "File Name"]
        ]

        calculate_mode = proc_kwargs.get("calculate", "average")

        # Get core_levels parameter and normalize it to a list
        core_levels_input = proc_kwargs.get("core_levels", None)

        # Normalize core_levels_input to a list
        # Handle: None, "", [], [""], "C 1s", ["C 1s"], ["C 1s", "O 1s"]
        if (
            core_levels_input is None
            or core_levels_input == ""
            or core_levels_input == []
        ):
            # Plot all core levels
            selected_cores = all_core_levels
        elif isinstance(core_levels_input, str):
            # Single core level as string
            selected_cores = [core_levels_input]
        elif isinstance(core_levels_input, list):
            # List of core levels
            if len(core_levels_input) == 0 or (
                len(core_levels_input) == 1 and core_levels_input[0] == ""
            ):
                # Empty list or list with empty string
                selected_cores = all_core_levels
            else:
                selected_cores = core_levels_input
        else:
            raise TypeError(
                f"core_levels must be str, list, or None, got {type(core_levels_input)}"
            )

        # Validate that all selected core levels exist
        invalid_cores = [c for c in selected_cores if c not in all_core_levels]
        if invalid_cores:
            print(
                f"Core level(s) {invalid_cores} not found. Available: {all_core_levels}"
            )
            return

        # If only one core level is selected, plot it as a single plot
        if len(selected_cores) == 1:
            ratio_reference_core_data = None
            if calculate_mode == "ratio":
                # Single-core path: allow explicit data injection, otherwise use same core data
                ratio_reference_core_data = report_dict.get(selected_cores[0])
            plot_single_core_level(
                report_dict[selected_cores[0]],
                ax,
                proc_kwargs,
                plot_kwargs,
                save_kwargs,
                core_level_override=selected_cores[0],
                ratio_reference_core_data=ratio_reference_core_data,
            )
        else:
            # Plot multiple core levels
            plot_all_core_levels(
                report_dict,
                selected_cores,
                ax,
                proc_kwargs,
                plot_kwargs,
                save_kwargs,
            )
    else:
        # Single-core format (original behavior)
        plot_single_core_level(
            report_dict,
            ax,
            proc_kwargs,
            plot_kwargs,
            save_kwargs,
        )


def plot_region_report(
    report_dict,
    ax=None,
    proc_kwargs=None,
    plot_kwargs=None,
    save_kwargs=None,
):
    """Plot ratios or totals between core levels across measurements.

    This function can operate in two modes:
    1. **Ratio mode**: Calculates and plots the ratio of summed parameter values
       between two core levels (e.g., F/C atomic concentration ratio).
    2. **Total mode**: Plots the total summed parameter values for specified
       core levels (e.g., total atomic concentration of F 1s and C 1s).

    For each measurement, the function sums the parameter across all components
    in each specified core level, then either calculates ratios or plots totals.

    Parameters
    ----------
    report_dict : dict
        Multi-core format dictionary from ``read_report_file``. Must contain
        at least the core levels specified.
    ax : matplotlib.axes.Axes or None, optional
        Target axes for plotting. If None (default), creates new figure.
        If provided, allows multiple series to be plotted on the same axes.
    proc_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'plot_type' (str): Type of plot (default "ratio"). Options:
          * "ratio": Plot ratio between numerator and denominator
          * "total": Plot total values for specified core levels
        - 'numerator' (str): Core level name for numerator in ratio mode
          (default "F 1s"). Example: "F 1s", "O 1s".
        - 'denominator' (str): Core level name for denominator in ratio mode
          (default "C 1s"). Example: "C 1s".
        - 'core_levels' (str or list): Core level(s) to plot in total mode.
          Single string for one core level, or list for multiple.
          Example: "F 1s" or ["F 1s", "C 1s", "O 1s"]
        - 'parameter' (str): Parameter to sum across components (default "At Conc").
          Accepts both short forms and full column names:
          * "At Conc" or "%At Conc": Atomic concentration (sum gives total atomic %)
          * "Area" or "Raw Area": Peak area (sum gives total signal intensity)
          * "BE" or "Binding Energy (eV)": Binding energy
          If a short form is provided, it will be automatically mapped to the
          full column name.
        - 'calculate' (str or None): Statistic to display in legend. Either
            "average" (shows mean values), "difference" (shows last - first),
            "difference_max" (shows max - min), or None/empty to disable
            statistics. Default is "average".
    plot_kwargs : dict or None, optional
        Styling arguments forwarded to ``matplotlib.axes.Axes.plot``.
        Supports both full and abbreviated parameter names.
        Example: {"marker": "o", "linestyle": "-", "label": "F/C ratio"}
    save_kwargs : dict or None, optional
        Options for saving the figure. Supported keys:
        - 'save_fig' (bool): Whether to save the figure (default False)
        - 'save_folder' (str): Directory path for saving
        - 'format' (str): File format (e.g., 'png', 'pdf')
        - 'dpi' (int): Resolution for raster formats

    Returns
    -------
    None

    See Also
    --------
    plot_comp_report : Plot component-level data within core levels

    Examples
    --------
    Plot F/C atomic concentration ratio evolution:

    >>> report_dict = xp.read_report_file("multicore_report.txt")
    >>> xplot.plot_region_report(report_dict,
    ...                          proc_kwargs={"plot_type": "ratio",
    ...                                      "numerator": "F 1s",
    ...                                      "denominator": "C 1s"})

    Plot total atomic concentrations for multiple core levels:

    >>> xplot.plot_region_report(report_dict,
    ...                          proc_kwargs={"plot_type": "total",
    ...                                      "core_levels": ["F 1s", "C 1s", "O 1s"]})

    Compare multiple ratios on the same plot:

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> xplot.plot_region_report(report_dict, ax=ax,
    ...                          proc_kwargs={"numerator": "F 1s", "denominator": "C 1s"},
    ...                          plot_kwargs={"marker": "o", "label": "F/C"})
    >>> xplot.plot_region_report(report_dict, ax=ax,
    ...                          proc_kwargs={"numerator": "O 1s", "denominator": "C 1s"},
    ...                          plot_kwargs={"marker": "s", "label": "O/C"})

    Plot total atomic concentration for a single core level:

    >>> xplot.plot_region_report(report_dict,
    ...                          proc_kwargs={"plot_type": "total", "core_levels": "F 1s"})
    """
    proc_kwargs = proc_kwargs or {}
    plot_kwargs = plot_kwargs or {}
    save_kwargs = save_kwargs or {}

    # Extract processing options
    plot_type = proc_kwargs.get("plot_type", "ratio")
    parameter = get_column_name(proc_kwargs.get("parameter", "At Conc"))
    calculate_mode = proc_kwargs.get("calculate", "average")

    if report_dict is None:
        print("No data to plot.")
        return

    # Verify multi-core format
    is_multicore = "Core Level" in report_dict and "Name" not in report_dict
    if not is_multicore:
        print("Error: plot_region_report requires multi-core format data.")
        return

    available_cores = [
        k for k in report_dict.keys() if k not in ["Core Level", "File Name"]
    ]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            fig.set_layout_engine("tight")
        except AttributeError:
            try:
                fig.set_tight_layout(True)
            except AttributeError:
                pass
        plot_here = True
    else:
        fig = ax.get_figure()
        plot_here = False

    if plot_type == "ratio":
        # Ratio mode: plot ratio between numerator and denominator
        numerator = proc_kwargs.get("numerator", "F 1s")
        denominator = proc_kwargs.get("denominator", "C 1s")

        # Check that both core levels exist
        if numerator not in available_cores:
            print(
                f"Error: Numerator '{numerator}' not found. Available: {available_cores}"
            )
            return
        if denominator not in available_cores:
            print(
                f"Error: Denominator '{denominator}' not found. Available: {available_cores}"
            )
            return

        # Extract data for both core levels
        num_data = report_dict[numerator].get(parameter)
        denom_data = report_dict[denominator].get(parameter)

        if num_data is None:
            print(f"Error: Parameter '{parameter}' not found in {numerator}")
            return
        if denom_data is None:
            print(f"Error: Parameter '{parameter}' not found in {denominator}")
            return

        # Convert to numpy arrays
        num_data = np.array(num_data, dtype=float)
        denom_data = np.array(denom_data, dtype=float)

        # Check dimensions
        if num_data.ndim != 2 or denom_data.ndim != 2:
            print(
                f"Error: Data must be 2D (components × measurements). Got shapes: {num_data.shape}, {denom_data.shape}"
            )
            return

        # Sum across components for each measurement (axis=0 sums over components)
        num_totals = np.nansum(num_data, axis=0)
        denom_totals = np.nansum(denom_data, axis=0)

        # Calculate ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            values = num_totals / denom_totals

        # Create measurement numbers (x-axis)
        n_measurements = len(values)
        measurement_nums = np.arange(1, n_measurements + 1)

        # Calculate statistic for legend if requested
        numeric_vals = [v for v in values if np.isfinite(v)]
        stat_label = None
        stat_value = None
        if calculate_mode and numeric_vals:
            if calculate_mode == "average":
                stat_label = "avg"
                stat_value = np.mean(numeric_vals)
            elif calculate_mode == "difference" and len(numeric_vals) >= 2:
                stat_label = "diff"
                stat_value = numeric_vals[-1] - numeric_vals[0]
            elif calculate_mode == "difference_max" and len(numeric_vals) >= 2:
                stat_label = "max-min"
                stat_value = max(numeric_vals) - min(numeric_vals)

        # Default plot styling for ratio mode
        # If user provides a label, use it; otherwise use default
        base_label = plot_kwargs.get("label", f"{numerator}/{denominator}")
        if stat_label and stat_value is not None:
            full_label = f"{base_label} ({stat_label}={stat_value:7.2f})"
        else:
            full_label = base_label
        default_kwargs = {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 8,
            "label": full_label,
        }
        # Update with plot_kwargs but exclude 'label' since we already handled it
        plot_kwargs_no_label = {k: v for k, v in plot_kwargs.items() if k != "label"}
        default_kwargs.update(plot_kwargs_no_label)

        # Plot the ratio
        line = ax.plot(measurement_nums, values, **default_kwargs)

        # Add horizontal line for average if requested
        if calculate_mode == "average" and stat_value is not None:
            ax.axhline(
                stat_value,
                color=line[0].get_color(),
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )

        # Configure axes
        ax.set_xlabel("Measurement Number", fontsize=12)
        ax.set_ylabel("Atomic Ratio", fontsize=12)
        ax.grid(True, alpha=0.3)
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_family("monospace")

        # Save figure if requested
        if save_kwargs.get("save_fig", False):
            parent_file_name = report_dict.get("File Name", "report")
            save_figure(
                fig,
                name_list=[parent_file_name],
                save_args=save_kwargs,
                prefix=f"{numerator.replace(' ', '')}_{denominator.replace(' ', '')}_ratio_",
            )

    elif plot_type == "total":
        # Total mode: plot total values for specified core levels
        core_levels_input = proc_kwargs.get("core_levels", available_cores)

        # Normalize to list
        if isinstance(core_levels_input, str):
            core_levels = [core_levels_input]
        elif isinstance(core_levels_input, list):
            core_levels = core_levels_input
        else:
            print(
                f"Error: core_levels must be str or list, got {type(core_levels_input)}"
            )
            return

        # Validate core levels exist
        invalid_cores = [c for c in core_levels if c not in available_cores]
        if invalid_cores:
            print(
                f"Error: Core level(s) {invalid_cores} not found. Available: {available_cores}"
            )
            return

        # Get color cycle for multiple core levels
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Calculate maximum label length for alignment
        max_label_len = max(
            len(plot_kwargs.get("label", core_level)) for core_level in core_levels
        )

        # Plot each core level
        for idx, core_level in enumerate(core_levels):
            # Extract data
            data = report_dict[core_level].get(parameter)

            if data is None:
                print(
                    f"Warning: Parameter '{parameter}' not found in {core_level}, skipping"
                )
                continue

            # Convert to numpy array
            data = np.array(data, dtype=float)

            # Check dimensions
            if data.ndim != 2:
                print(
                    f"Warning: Data for {core_level} must be 2D, got shape {data.shape}, skipping"
                )
                continue

            # Sum across components for each measurement
            totals = np.nansum(data, axis=0)

            # Create measurement numbers (x-axis)
            n_measurements = len(totals)
            measurement_nums = np.arange(1, n_measurements + 1)

            # Calculate statistic for legend if requested
            numeric_vals = [v for v in totals if np.isfinite(v)]
            stat_label = None
            stat_value = None
            if calculate_mode and numeric_vals:
                if calculate_mode == "average":
                    stat_label = "avg"
                    stat_value = np.mean(numeric_vals)
                elif calculate_mode == "difference" and len(numeric_vals) >= 2:
                    stat_label = "diff"
                    stat_value = numeric_vals[-1] - numeric_vals[0]
                elif calculate_mode == "difference_max" and len(numeric_vals) >= 2:
                    stat_label = "max-min"
                    stat_value = max(numeric_vals) - min(numeric_vals)

            # Default plot styling for total mode
            # If user provides a label, use it; otherwise use core level name
            base_label = plot_kwargs.get("label", core_level)
            # Pad label to max length for alignment
            padded_label = base_label.ljust(max_label_len)
            if stat_label and stat_value is not None:
                full_label = f"{padded_label} ({stat_label}={stat_value:7.2f})"
            else:
                full_label = padded_label
            default_kwargs = {
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2,
                "markersize": 8,
                "color": color_cycle[idx % len(color_cycle)],
                "label": full_label,
            }
            # Update with plot_kwargs but exclude 'label' since we already handled it
            plot_kwargs_no_label = {
                k: v for k, v in plot_kwargs.items() if k != "label"
            }
            default_kwargs.update(plot_kwargs_no_label)

            # Plot the total
            line = ax.plot(measurement_nums, totals, **default_kwargs)

            # Add horizontal line for average if requested
            if calculate_mode == "average" and stat_value is not None:
                ax.axhline(
                    stat_value,
                    color=line[0].get_color(),
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )

        # Configure axes
        ax.set_xlabel("Measurement Number", fontsize=12)
        ax.set_ylabel("Atomic Concentration (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_family("monospace")

        # Save figure if requested
        if save_kwargs.get("save_fig", False):
            parent_file_name = report_dict.get("File Name", "report")
            save_figure(
                fig,
                name_list=[parent_file_name],
                save_args=save_kwargs,
                prefix="region_totals_",
            )

    else:
        print(f"Error: Unknown plot_type '{plot_type}'. Must be 'ratio' or 'total'.")
        return

    if plot_here:
        plt.show()


def plot_spectrum(
    spectrum_dict,
    ax=None,
    proc_kwargs=None,
    plot_kwargs=None,
    save_kwargs=None,
):
    """Plot XPS spectrum with optional residuals.

    This function plots XPS spectrum data including fitted components and
    optional residuals. It automatically handles axis setup, core level
    display in the title, and figure management.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping produced by ``read_spectrum_file`` containing spectrum data.
        Expected keys include 'BE' and/or 'KE' for x-values, component names
        for fitted peaks, 'Measured' for experimental data, and optionally
        'Residual' or 'Normalised Residual' for fit residuals. Should also
        contain 'Core Level' and 'File Name' metadata.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Target axis or axes. If ``None``, a new figure with two axes (for
        residuals and main plot) is created. Can be:
        - Single Axes: Used for main plot only (no residuals)
        - Sequence [residual_ax, main_ax]: Used for residual + main plot
    proc_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'x_axis' (str): Which x-axis to use, 'BE' (binding energy, default)
          or 'KE' (kinetic energy).
        - 'normalised_residual' (bool): Whether to plot normalised residual
          instead of raw residual (default False).
        - 'plot_items' (list of str or None): Item types to include in the plot.
          Valid values: 'measured', 'background', 'components', 'envelope',
          'residual'. If None (default), plots all items.
    plot_kwargs : dict or None, optional
        Styling arguments forwarded to ``matplotlib.axes.Axes.plot`` for data
        series. These override module defaults. Supports both full and
        abbreviated parameter names (e.g., 'ls' or 'linestyle').
    save_kwargs : dict or None, optional
        Options for saving the figure. Supported keys:
        - 'save_fig' (bool): Whether to save the figure (default False)
        - 'save_folder' (str): Directory path for saving
        - 'format' (str): File format (e.g., 'png', 'pdf')
        - 'dpi' (int): Resolution for raster formats

    Returns
    -------
    None

    See Also
    --------
    plot_comp_report : Plot fit report parameter(s) across components
    plot_region_ratio : Plot ratio between core levels

    Examples
    --------
    Plot spectrum with binding energy x-axis:

    >>> spectrum_dict = xp.read_spectrum_file("spectrum.txt")
    >>> xplot.plot_spectrum(spectrum_dict)

    Plot with kinetic energy and normalized residuals:

    >>> proc_kwargs = {"x_axis": "KE", "normalised_residual": True}
    >>> xplot.plot_spectrum(spectrum_dict, proc_kwargs=proc_kwargs)

    Use custom axes layout:

    >>> fig, axes = plt.subplots(2, 1, figsize=(8, 6),
    ...                          gridspec_kw={"height_ratios": [1, 8], "hspace": 0})
    >>> xplot.plot_spectrum(spectrum_dict, ax=axes)
    """
    # Extract parameters
    proc_kwargs = proc_kwargs or {}
    save_kwargs = save_kwargs or {}
    x_axis = proc_kwargs.get("x_axis", "BE")
    normalised_residual = proc_kwargs.get("normalised_residual", False)
    plot_items = proc_kwargs.get("plot_items", None)

    if spectrum_dict is None:
        print("No data to plot.")
        return

    # Determine if residuals will be plotted
    # Default: plot all items when plot_items not specified
    if plot_items is None:
        plot_items_to_check = [
            "measured",
            "background",
            "components",
            "envelope",
            "residual",
        ]
    else:
        plot_items_to_check = [item.lower() for item in plot_items]

    will_plot_residual = "residual" in plot_items_to_check

    # Create axes if needed: 2 axes if plotting residuals, 1 if not
    if ax is None and will_plot_residual:
        # Create figure with 2 axes for residuals
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [1, 8], "hspace": 0},
        )
        try:
            fig.set_layout_engine("tight")
        except AttributeError:
            try:
                fig.set_tight_layout(True)
            except AttributeError:
                pass

    # Prepare axes
    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    # Get x-axis info and plot data
    x_values, x_label, invert = get_x_axis_from_dict(spectrum_dict, x_axis)
    plot_params = update_plot_params({"ls": "-", "lw": 1.5}, plot_kwargs)
    handles, labels = plot_spectrum_series(
        spectrum_dict,
        ax,
        x_values,
        {
            "x_axis": x_axis,
            "params": plot_params,
            "normalised_residual": proc_kwargs.get("normalised_residual", False),
            "plot_items": proc_kwargs.get("plot_items", None),
        },
    )

    # Configure labels, title and legend
    main_ax.set_xlabel(x_label)
    main_ax.set_ylabel("Intensity (a.u.)")
    fig.suptitle(spectrum_dict.get("Core Level") or "Unknown")
    if handles:
        main_ax.legend(handles, labels)

    # Configure axis inversion and residual formatting
    configure_axes(ax, main_ax, invert=invert)

    # Save figure if requested
    if save_kwargs.get("save_fig", False):
        save_figure(
            fig,
            name_list=[derive_file_name(spectrum_dict)],
            save_args=save_kwargs,
            prefix="",
        )

    if plot_here:
        plt.show()
