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
    update_plot_params,
    plot_report_series,
    plot_spectrum_series,
)


def plot_report(
    report_dict,
    ax=None,
    processing_kwargs=None,
    save_fig=False,
    save_kwargs=None,
    **kwargs,
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
    processing_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'fit_param' (str): Parameter to plot (default 'BE'). Common short
          forms: "BE", "Area", "At Conc", "Goodness".
    save_fig : bool, default False
        If True the generated figure will be saved using ``save_kwargs``.
    save_kwargs : dict or None, optional
        Options forwarded to the saving helper (may include ``save_folder``,
        ``format``, ``dpi``).
    plot_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot` for
        the component series (overrides module defaults).

    Returns
    -------
    None
    """
    # Extract parameters
    processing_kwargs = processing_kwargs or {}
    plot_kwargs = kwargs.pop("plot_kwargs", None)

    fit_param = processing_kwargs.get("fit_param", "BE")

    # Set up plotting parameters
    params = update_plot_params(
        {"linestyle": "--", "linewidth": 1.5, "marker": "o"}, plot_kwargs
    )

    if report_dict is None:
        print("No data to plot.")
        return

    # Prepare axes
    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    # Map short parameter names to full column headers
    col_full = {
        "BE": "Binding Energy (eV)",
        "Area": "Raw Area",
        "At Conc": "%At Conc",
        "Goodness": "Goodness of Fit",
    }.get(fit_param, fit_param)

    # Plot the data
    plot_report_series(report_dict, main_ax, col_full, params)

    # Configure axes and labels
    main_ax.set_xlabel("Experimental Variable")
    main_ax.set_ylabel(col_full)

    # Add core level as the title
    core_level = report_dict.get("Core Level") or "Unknown"
    main_ax.set_title(core_level)

    main_ax.legend()

    # Save figure if requested
    if save_fig:
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
    processing_kwargs=None,
    save_fig=False,
    save_kwargs=None,
    **kwargs,
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
    processing_kwargs : dict or None, optional
        Processing options for the plot. Supported keys:
        - 'x_axis' (str): Which x-axis to use, 'BE' or 'KE' (default 'BE').
        - 'normalised_residual' (bool): Whether to plot normalised residual
          instead of raw residual (default False).
    save_fig : bool, default False
        If True the generated figure will be saved using ``save_kwargs``.
    save_kwargs : dict, optional
        Save options passed to :func:`save_figure` (e.g. ``save_folder``,
        ``format``, ``dpi``).
    plot_kwargs : dict, optional
        Keyword arguments forwarded to ``Axes.plot`` for data series.

    Returns
    -------
    None
    """
    # Extract parameters
    processing_kwargs = processing_kwargs or {}
    plot_kwargs = kwargs.pop("plot_kwargs", None)

    x_axis = processing_kwargs.get("x_axis", "BE")
    normalised_residual = processing_kwargs.get("normalised_residual", False)

    # Set up plotting parameters
    params = update_plot_params({"linestyle": "-", "linewidth": 1.5}, plot_kwargs)

    if spectrum_dict is None:
        print("No data to plot.")
        return

    # Prepare axes
    fig, ax, main_ax, plot_here = ensure_axes_and_main(ax)

    # Get x-axis info and plot data
    x_values, x_label, invert = get_x_axis_from_dict(spectrum_dict, x_axis)
    handles, labels = plot_spectrum_series(
        spectrum_dict,
        ax,
        x_values,
        {
            "x_axis": x_axis,
            "params": params,
            "normalised_residual": normalised_residual,
        },
    )

    # Configure axes and labels
    main_ax.set_xlabel(x_label)
    main_ax.set_ylabel("Intensity (a.u.)")

    # Add core level as the title
    core_level = spectrum_dict.get("Core Level") or "Unknown"
    fig.suptitle(core_level)

    if handles:
        main_ax.legend(handles, labels)

    # Configure axis inversion and residual formatting
    configure_axes(ax, main_ax, invert=invert)

    # Save figure if requested
    if save_fig:
        save_figure(
            fig,
            name_list=[derive_file_name(spectrum_dict)],
            save_args=save_kwargs,
            prefix="",
        )

    if plot_here:
        plt.show()
