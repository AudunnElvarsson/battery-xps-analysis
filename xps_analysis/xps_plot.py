"""xps_plot
-----------

Utilities for plotting XPS spectra and fit reports.

This module provides clean functional interfaces for plotting XPS data
that accept dictionaries produced by the parsing utilities in
``xps_processing`` and produce Matplotlib figures/axes. It also contains
helpers for constructing safe filenames and saving figures with consistent
options.
"""

import matplotlib.pyplot as plt

# Import plotting utilities from helper modules
from .plot_helpers import (
    save_figure,
    derive_file_name,
    get_x_axis_from_dict,
    ensure_axes_and_main,
    configure_axes,
    get_column_name,
    update_plot_params,
    plot_report_series,
    plot_spectrum_series,
    convert_to_relative_be,
)


def plot_report(
    report_dict,
    ax=None,
    proc_kwargs=None,
    plot_kwargs=None,
    save_kwargs=None,
):
    """Plot a fit-report parameter across all components.

    The function plots one fitted parameter (for example binding energy or
    area) for each component present in ``report_dict``. A horizontal dashed
    line showing the component average is added for each series and included
    in the legend. The plot title includes the core level information.

    Parameters
    ----------
    report_dict : dict
        Mapping produced by the fit-report parser (header -> arrays). Expected
        to contain a ``'Name'`` entry and the column named by ``fit_param``.
    ax : matplotlib.axes.Axes or None, optional
        Target axis to draw on. If ``None`` a new figure and axis are created.
    proc_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'fit_param' (str): Parameter to plot (default 'BE'). Common short
          forms: "BE", "Area", "At Conc", "Goodness".
        - 'calculate' (str): Display statistic in legend, either "average"
          (default) to show mean values, or "difference" to show the
          difference between last and first values.
        - 'reference' (str): Component label to use as reference for relative
          binding energy plotting (e.g., "A", "C"). Only applies when
          fit_param='BE'. If provided, plots relative BE with respect to the
          first value of the reference component. If not provided, plots
          absolute binding energies (default behavior).
    plot_kwargs : dict or None, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot` for
        the component series (overrides module defaults).
    save_kwargs : dict or None, optional
        Options forwarded to the saving helper. Supported keys:
        - 'save_fig' (bool): Whether to save the figure (default False).
        - 'save_folder' (str): Directory path for saving.
        - 'format' (str): File format (e.g., 'png', 'pdf').
        - 'dpi' (int): Resolution for raster formats.

    Returns
    -------
    None
    """
    proc_kwargs = proc_kwargs or {}
    save_kwargs = save_kwargs or {}

    if report_dict is None:
        print("No data to plot.")
        return

    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    # Get column name and convert to relative BE if needed
    col_full = get_column_name(proc_kwargs.get("fit_param", "BE"))
    reference = proc_kwargs.get("reference", None)
    plot_dict = report_dict

    if reference and "Binding Energy" in col_full:
        plot_dict = convert_to_relative_be(report_dict, col_full, reference)
        if plot_dict is not report_dict:
            col_full = "Relative Binding Energy (eV)"

    # Plot the data
    plot_report_series(
        plot_dict,
        main_ax,
        col_full,
        update_plot_params({"ls": "--", "lw": 1.5, "m": "o"}, plot_kwargs),
        {
            "calculate": proc_kwargs.get("calculate", "average"),
            "swap_axes": "Binding Energy" in col_full,
        },
    )

    # Configure axes and legend
    _configure_report_axes(
        main_ax, col_full, report_dict.get("Core Level") or "Unknown"
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


def _configure_report_axes(ax, col_full, core_level):
    """Configure axis labels, title, and legend for report plot.

    Parameters
    ----------
    ax : Axes
        Target axis to configure.
    col_full : str
        Full column name being plotted.
    core_level : str
        Core level name for title.
    """
    swap_axes = "Binding Energy" in col_full

    if swap_axes:
        ax.set_xlabel(col_full)
        ax.set_ylabel("Experimental Variable")
        ax.invert_xaxis()
        ax.invert_yaxis()
    else:
        ax.set_xlabel("Experimental Variable")
        ax.set_ylabel(col_full)

    ax.set_title(core_level)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_family("monospace")


def plot_spectrum(
    spectrum_dict,
    ax=None,
    proc_kwargs=None,
    plot_kwargs=None,
    save_kwargs=None,
):
    """Plot a spectrum and optional residuals from a spectrum dictionary.

    This function plots XPS spectrum data with optional residuals, automatically
    handling axis setup, core level display in the title, and figure management.
    The API is purely functional without class instantiation overhead.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping with series to plot. Expected keys include ``'BE'`` and/or
        ``'KE'`` for x-values and other keys for data (components, residuals).
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Target axis or axes. If ``None``, a new figure/axes pair is created.
    proc_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'x_axis' (str): Which x-axis to use, 'BE' or 'KE' (default 'BE').
        - 'normalised_residual' (bool): Whether to plot normalised residual
          instead of raw residual (default False).
    plot_kwargs : dict or None, optional
        Keyword arguments forwarded to ``Axes.plot`` for data series.
    save_kwargs : dict or None, optional
        Options forwarded to the saving helper. Supported keys:
        - 'save_fig' (bool): Whether to save the figure (default False).
        - 'save_folder' (str): Directory path for saving.
        - 'format' (str): File format (e.g., 'png', 'pdf').
        - 'dpi' (int): Resolution for raster formats.

    Returns
    -------
    None
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
