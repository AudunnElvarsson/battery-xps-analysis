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
plot_report : Plot fit report parameter(s) across components and core levels
plot_spectrum : Plot XPS spectrum with optional residuals

Notes
-----
All internal helper functions have been moved to the ``plot_helpers`` package
for better organization and maintainability.
"""

import matplotlib.pyplot as plt

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
)


def plot_report(
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
        - 'calculate' (str): Statistic to display in legend. Either "average"
          (default, shows mean values) or "difference" (shows last - first).
        - 'reference' (str): Component label for relative BE plotting (e.g.,
          "A", "C"). Only applies when fit_param='BE'. Plots relative binding
          energy with respect to the first value of the reference component.
        - 'core_levels' (str, list of str, or None): For multi-core format,
          specify which core levels to plot:
          * Single string "C 1s" - plots only that core level
          * List ["C 1s", "O 1s"] - plots those core levels
          * None, "", [] or [""] - plots all available core levels
    plot_kwargs : dict or None, optional
        Styling arguments forwarded to ``matplotlib.axes.Axes.plot`` for the
        component series. These override module defaults. Supports both full
        and abbreviated parameter names (e.g., 'ls' or 'linestyle').
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
    >>> xplot.plot_report(report_dict)

    Plot specific core levels with difference calculation:

    >>> proc_kwargs = {
    ...     "core_levels": ["C 1s", "O 1s"],
    ...     "calculate": "difference"
    ... }
    >>> xplot.plot_report(report_dict, proc_kwargs=proc_kwargs)

    Plot relative binding energy with custom axes:

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> proc_kwargs = {"reference": "A", "core_levels": ["C 1s", "O 1s"]}
    >>> xplot.plot_report(report_dict, ax=axes, proc_kwargs=proc_kwargs)
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
            plot_single_core_level(
                report_dict[selected_cores[0]],
                ax,
                proc_kwargs,
                plot_kwargs,
                save_kwargs,
                core_level_override=selected_cores[0],
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
    plot_report : Plot fit report parameter(s) across components

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

    if spectrum_dict is None:
        print("No data to plot.")
        return

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
