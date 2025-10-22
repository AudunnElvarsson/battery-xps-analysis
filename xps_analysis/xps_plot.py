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
    # Extract parameters
    proc_kwargs = proc_kwargs or {}
    save_kwargs = save_kwargs or {}

    if report_dict is None:
        print("No data to plot.")
        return

    # Prepare axes
    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    # Get column name and set up plotting parameters
    fit_param = proc_kwargs.get("fit_param", "BE")
    col_full = get_column_name(fit_param)
    plot_params = update_plot_params(
        {"ls": "--", "lw": 1.5, "marker": "o"}, plot_kwargs
    )

    # Plot the data
    plot_report_series(report_dict, main_ax, col_full, plot_params)

    # Configure labels, title and legend
    main_ax.set_xlabel("Experimental Variable")
    main_ax.set_ylabel(col_full)
    main_ax.set_title(report_dict.get("Core Level") or "Unknown")
    main_ax.legend()

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
        x_axis,
        plot_params,
        proc_kwargs.get("normalised_residual", False),
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
