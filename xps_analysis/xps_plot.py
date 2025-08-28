"""xps_plot
-----------

Utilities for plotting XPS spectra and fit reports.

This module provides functions and a small plotting helper class that
accept dictionaries produced by the parsing utilities in
``xps_processing`` and produce Matplotlib figures/axes. It also contains
helpers for constructing safe filenames and saving figures with consistent
options.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np


# === Module-level helper functions ===
def _update_plot_params(defaults, user_kwargs):
    """Merge default plotting parameters with user-supplied overrides.

    Parameters
    ----------
    defaults : dict
        Default plotting parameters (e.g. ``linestyle``, ``linewidth``).
    user_kwargs : dict or None
        Optional user-supplied overrides; keys here overwrite ``defaults``.

    Returns
    -------
    dict
        A new dictionary containing the merged plotting parameters.
    """
    params = defaults.copy()
    if user_kwargs:
        params.update(user_kwargs)
    return params


def _build_save_info(save_folder, name_list, save_args=None, prefix="figure"):
    """Build a filesystem-safe filename and default save kwargs for figures.

    Parameters
    ----------
    save_folder : str
        Destination folder for the saved figure. The folder is created if it
        does not exist.
    name_list : list of str
        Sequence of parts to include in the filename (joined with
        underscores); empty or None entries are ignored.
    save_args : dict or None
        Optional overrides forwarded to :meth:`matplotlib.figure.Figure.savefig`.
        The special key ``save_folder`` is ignored here (it is handled by
        the caller).
    prefix : str
        Optional filename prefix. If empty, no prefix is used.

    Returns
    -------
    tuple
        ``(save_file, save_kwargs)`` where ``save_file`` is the absolute path
        to the file and ``save_kwargs`` is a dict of keyword arguments
        suitable for passing to ``Figure.savefig``.
    """
    os.makedirs(save_folder, exist_ok=True)
    save_kwargs = {"dpi": 300, "format": "png", "bbox_inches": "tight"}
    if save_args:
        # allow overriding format, dpi, bbox_inches etc.
        save_kwargs.update({k: v for k, v in save_args.items() if k != "save_folder"})
    # build a safe name from provided parts
    safe_parts = [
        re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_")
        for p in (name_list or [])
        if p is not None and str(p).strip() != ""
    ]
    names_str = "_".join(safe_parts) or "all"
    if prefix:
        filename = f"{prefix}_{names_str}.{save_kwargs['format']}"
    else:
        filename = f"{names_str}.{save_kwargs['format']}"
    save_file = os.path.join(save_folder, filename)
    return save_file, save_kwargs


def _save_figure(fig, name_list=None, save_args=None, prefix="figure"):
    """Save a Matplotlib figure using a generated safe filename.

    This helper constructs a safe filename from ``name_list`` and calls
    :meth:`matplotlib.figure.Figure.savefig` with the computed keyword
    arguments.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to save.
    name_list : list of str or None
        Parts used to build the filename. If ``None`` or empty, a generic
        basename is used.
    save_args : dict or None
        Options forwarded to ``Figure.savefig``. The special key
        ``save_folder`` (if present) selects the target folder.
    prefix : str
        Optional filename prefix; if empty the prefix is omitted.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    save_folder = (
        save_args.get("save_folder")
        if (save_args and "save_folder" in save_args)
        else os.getcwd()
    )
    save_file, save_kwargs = _build_save_info(
        save_folder, name_list or [], save_args, prefix=prefix
    )
    fig.savefig(save_file, **save_kwargs)
    print(f"Saved figure: {save_file}")
    return save_file


# New module-level helpers extracted from plot_fit_spectrum to reduce complexity
def _get_x_axis_from_dict(spectrum_dict, x_axis):
    """Get x-axis data and display metadata from a spectrum mapping.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping that may contain ``'BE'`` and/or ``'KE'`` arrays.
    x_axis : str
        Requested axis: ``'BE'`` (binding energy) or ``'KE'`` (kinetic energy).

    Returns
    -------
    tuple
        ``(x_values, x_label, invert)`` where ``x_values`` is the array to
        plot on x, ``x_label`` is a human-readable axis label, and ``invert``
        is a boolean indicating whether the x-axis should be inverted.
    """
    if x_axis == "KE":
        return spectrum_dict.get("KE"), "Kinetic Energy (eV)", False
    return spectrum_dict.get("BE"), "Binding Energy (eV)", True


def _should_plot_key(key, x_axis, normalised_residual):
    """Decide whether a key from a spectrum mapping should be plotted.

    Parameters
    ----------
    key : str
        Candidate key from the spectrum dictionary.
    x_axis : str
        The axis currently used for x data (``'BE'`` or ``'KE'``).
    normalised_residual : bool
        Whether normalised residuals are being plotted; affects.

    Returns
    -------
    bool
        True if the key should be plotted as a data series, False otherwise.
    """
    if key in (x_axis, "KE", "BE"):
        return False
    if key in ("File Name", "Sample"):
        return False
    if key == "Normalised Residual" and not normalised_residual:
        return False
    if key == "Residual" and normalised_residual:
        return False
    return True


def _figure_from_axes(ax_in):
    """Return the :class:`matplotlib.figure.Figure` for an axes-like input.

    Parameters
    ----------
    ax_in : Axes or sequence of Axes
        Axis-like object (single Axes or sequence) from which to obtain the
        parent figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure instance associated with the provided axes, or the
        current figure if none can be determined.
    """
    try:
        return ax_in[0].get_figure()
    except (TypeError, IndexError, AttributeError):
        try:
            return ax_in.get_figure()
        except (AttributeError, TypeError):
            return plt.gcf()


def _select_target_axis(ax, key):
    """Select the axis to plot a given series on.

    Residual series (``'Residual'`` or ``'Normalised Residual'``) are
    plotted on the top/residual axis when a two-axis layout is used
    (``ax[0]``); other series use the main plotting axis (``ax[1]`` or
    ``ax``).

    Parameters
    ----------
    ax : Axes or sequence of Axes
        Axis or axes container used for plotting.
    key : str
        Name of the series being plotted.

    Returns
    -------
    Axes
        The target axes for the given key.
    """
    try:
        if key in ("Normalised Residual", "Residual"):
            return ax[0]
        return ax[1]
    except (TypeError, IndexError, AttributeError):
        return ax


def _derive_file_name(mapping):
    """Derive a sensible filename base from a mapping.

    The function looks for keys in the order ``'File Name'``, ``'Sample'``,
    ``'Name'`` and falls back to ``'spectrum'``. If the value is a sequence,
    the first element is used.

    Parameters
    ----------
    mapping : dict
        Mapping that may contain identifying metadata for the spectrum.

    Returns
    -------
    str
        A short, filesystem-friendly name suitable for use in filenames.
    """
    file_name = (
        mapping.get("File Name")
        or mapping.get("Sample")
        or mapping.get("Name")
        or "spectrum"
    )
    if isinstance(file_name, (list, np.ndarray)):
        file_name = file_name[0] if len(file_name) else "spectrum"
    return file_name


def _ensure_axes_and_main(ax_in):
    """Prepare and return a figure/axes tuple for plotting.

    If ``ax_in`` is ``None`` a new :class:`matplotlib.figure.Figure` and
    Axes are created. When ``ax_in`` is a sequence (e.g. ``(residual_ax,
    main_ax)``), the second element is treated as the main plotting axis.

    Parameters
    ----------
    ax_in : Axes or sequence of Axes or None
        Optional target axes provided by the caller.

    Returns
    -------
    tuple
        ``(fig, ax, main_ax, plot_here)`` where ``fig`` is the Figure,
        ``ax`` is the original axis object passed or created, ``main_ax`` is
        the primary plotting axis and ``plot_here`` is True when a new
        figure was created.
    """
    plot_here = False
    if ax_in is None:
        plot_here = True
        fig, ax = plt.subplots(figsize=(8, 6))
        # prefer the newer layout engine API when available; fall back to
        # set_tight_layout for older matplotlib versions to avoid
        # PendingDeprecationWarning
        try:
            fig.set_layout_engine("tight")
        except AttributeError:
            try:
                fig.set_tight_layout(True)
            except AttributeError:
                # last-resort: ignore if neither method exists
                pass
    else:
        fig = _figure_from_axes(ax_in)
        ax = ax_in

    try:
        main_ax = ax[1]
    except (TypeError, IndexError, AttributeError):
        main_ax = ax
    return fig, ax, main_ax, plot_here


def _maybe_invert_axes(ax, main_ax, invert):
    """Backward-compatible wrapper for axis inversion (deprecated).

    Notes
    -----
    This helper delegates to :func:`_configure_axes` and is kept for
    backward compatibility; new code should call :func:`_configure_axes`
    directly.
    """
    return _configure_axes(ax, main_ax, invert)


def _format_residual_axis(ax, main_ax):
    """Format a residual axis and invert x if requested (deprecated wrapper).

    Notes
    -----
    This function forwards to :func:`_configure_axes` and exists for
    compatibility with older code paths.
    """
    # Deprecated; functionality moved to _configure_axes
    return _configure_axes(ax, main_ax, invert=True)


def _configure_axes(ax, main_ax, invert=False):
    """Configure axes: invert x-axis and apply residual-axis formatting.

    Parameters
    ----------
    ax : Axes or sequence of Axes
        Axis or axes container used for plotting. When a sequence is used the
        residual axis is expected at index 0.
    main_ax : Axes
        Primary plotting axis.
    invert : bool, optional
        If True attempt to invert the x-axes (useful for binding energy
        plots where decreasing energy is conventional).

    Notes
    -----
    The helper swallows attribute/index errors to remain robust when a
    single-Axes object is supplied.
    """
    # invert axes if requested
    if invert:
        try:
            main_ax.invert_xaxis()
        except (AttributeError, TypeError):
            pass
        try:
            ax[0].invert_xaxis()
        except (AttributeError, IndexError, TypeError):
            pass

    # residual axis formatting
    try:
        ax[0].set_ylabel("Residual")
        ax[0].spines["bottom"].set_position(("data", 0))
        ax[0].set_xticks([])
        ax[0].set_xticklabels([])
        main_ax.xaxis.set_tick_params(labelbottom=True, bottom=True)
        ax[0].spines["bottom"].set_visible(False)
        main_ax.spines["top"].set_visible(False)
    except (AttributeError, IndexError, TypeError):
        pass


def _plot_report_series(report_dict, ax, col_full, params):
    """Plot rows from a fit report mapping onto an axis.

    Each row in the report is plotted as a separate series and a horizontal
    dashed line showing the row average is added (and included in the
    legend).

    Parameters
    ----------
    report_dict : dict
        Mapping produced by the fit-report parser; expected to include a
        ``'Name'`` entry and a column with header ``col_full``.
    ax : Axes
        Target axis for plotting.
    col_full : str
        Full column name to extract from the report dictionary.
    params : dict
        Keyword arguments forwarded to ``Axes.plot``.
    """
    names = np.array(report_dict.get("Name"), dtype=object)
    y_data = np.array(report_dict.get(col_full), dtype=object)

    for i in range(y_data.shape[0]):
        y = y_data[i]
        x = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)
        label = names[i][0] if isinstance(names[i], (list, np.ndarray)) else names[i]
        avg = np.mean([v for v in y if isinstance(v, (int, float, np.floating))])
        (line,) = ax.plot(x, y, label=f"{label} (avg={avg:.2f})", **params)
        ax.plot(x, [avg] * len(x), color=line.get_color(), alpha=0.7, linestyle=":")


def _plot_spectrum_series(spectrum_dict, ax, x, opts):
    """Plot all series contained in a spectrum mapping and return legend info.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping where keys are series names and values are numeric sequences
        or arrays.
    ax : Axes or sequence of Axes
        Axis or axes used for plotting. When a sequence is supplied residual
        series are plotted on the residual axis (index 0).
    x : array-like or None
        Optional x-values to use for series that have matching length.
    opts : dict
        Options dictionary containing:
        - ``x_axis`` (str): selected axis name (``'BE'`` or ``'KE'``)
        - ``params`` (dict): plotting kwargs forwarded to ``Axes.plot``
        - ``normalised_residual`` (bool): whether to plot normalised
          residuals instead of raw residuals.

    Returns
    -------
    tuple
        ``(handles, labels)`` where ``handles`` is a list of Line2D objects
        and ``labels`` is a list of corresponding legend labels.
    """
    x_axis = opts.get("x_axis")
    params = opts.get("params") or {}
    normalised_residual = opts.get("normalised_residual", False)

    handles = []
    labels = []
    for key, values in spectrum_dict.items():
        if not _should_plot_key(key, x_axis, normalised_residual):
            continue
        target_ax = _select_target_axis(ax, key)
        try:
            cond = x is not None and len(x) == len(values)
        except TypeError:
            cond = False
        xs = x if cond else np.arange(len(values))
        try:
            (line,) = target_ax.plot(xs, values, label=key, **params)
        except (TypeError, ValueError):
            # skip series that cannot be plotted
            continue
        handles.append(line)
        labels.append(key)
    return handles, labels


class SpectrumPlotter:
    """Helper for plotting spectra and fit-report series.

    The class stores short-lived plotting state (figure and axes) so the
    instance methods can coordinate plotting and saving without threading
    many arguments through helper calls.
    """

    def __init__(self):
        # placeholder for future shared state
        self.fig = None
        self.ax = None
        self.main_ax = None
        self.plot_here = False

    # --- instance helpers to reduce parameter passing ---
    def _ensure_axes(self, ax_in):
        """Initialise or derive figure/axes state for subsequent plotting.

        Parameters
        ----------
        ax_in : Axes or sequence of Axes or None
            Optional axes provided by the caller. When ``None`` a new figure
            and axes are created.
        """
        self.fig, self.ax, self.main_ax, self.plot_here = _ensure_axes_and_main(ax_in)

    def _derive_col_full(self, fit_param):
        return {
            "BE": "Binding Energy (eV)",
            "Area": "Raw Area",
            "At Conc": "%At Conc",
            "Goodness": "Goodness of Fit",
        }.get(fit_param, fit_param)

    # Removed trivial wrapper methods to reduce indirection. Calls to
    # the corresponding module-level helpers are inlined below.

    def _save_if_requested(self, save_fig, save_args, mapping):
        if save_fig:
            _save_figure(
                self.fig,
                name_list=[_derive_file_name(mapping)],
                save_args=save_args,
                prefix="",
            )

    def plot_report(
        self, report_dict, ax=None, save_fig=False, save_args=None, **kwargs
    ):
        """Instance version of plot_fit_report."""
        fit_param = kwargs.pop("fit_param", "BE")
        plot_kwargs = kwargs.pop("plot_kwargs", None)

        params = _update_plot_params(
            {"linestyle": "--", "linewidth": 1.5, "marker": "o"}, plot_kwargs
        )

        if report_dict is None:
            print("No data to plot.")
            return

        # prepare axes and state on the instance
        self._ensure_axes(ax)

        col_full = self._derive_col_full(fit_param)
        # inline the former wrapper: plot directly using module helper
        _plot_report_series(report_dict, self.main_ax, col_full, params)

        self.main_ax.set_xlabel("Experimental Variable")
        self.main_ax.set_ylabel(col_full)
        self.main_ax.legend()

        self._save_if_requested(save_fig, save_args, report_dict)

        if self.plot_here:
            plt.show()

    def plot_spectrum(
        self, spectrum_dict, ax=None, save_fig=False, save_args=None, **kwargs
    ):
        """Instance version of plot_fit_spectrum."""
        x_axis = kwargs.pop("x_axis", "BE")
        normalised_residual = kwargs.pop("normalised_residual", False)
        plot_kwargs = kwargs.pop("plot_kwargs", None)

        params = _update_plot_params({"linestyle": "-", "linewidth": 1.5}, plot_kwargs)

        if spectrum_dict is None:
            print("No data to plot.")
            return

        # prepare axes/state
        self._ensure_axes(ax)

        # get x-axis info directly from module helper
        xinfo = _get_x_axis_from_dict(spectrum_dict, x_axis)
        handles, labels = _plot_spectrum_series(
            spectrum_dict,
            self.ax,
            xinfo[0],
            {
                "x_axis": x_axis,
                "params": params,
                "normalised_residual": normalised_residual,
            },
        )

        self.main_ax.set_xlabel(xinfo[1])
        self.main_ax.set_ylabel("Intensity (a.u.)")

        if handles:
            self.main_ax.legend(handles, labels)

        # inline axis configuration
        _configure_axes(self.ax, self.main_ax, invert=xinfo[2])

        self._save_if_requested(save_fig, save_args, spectrum_dict)

        if self.plot_here:
            plt.show()


def plot_fit_report(report_dict, ax=None, save_fig=False, save_args=None, **kwargs):
    """Plot a fit-report parameter across all components.

    The function plots one fitted parameter (for example binding energy or
    area) for each component present in ``report_dict``. A horizontal dashed
    line showing the component average is added for each series and included
    in the legend.

    Parameters
    ----------
    report_dict : dict
        Mapping produced by the fit-report parser (header -> arrays). Expected
        to contain a ``'Name'`` entry and the column named by ``fit_param``.
    ax : matplotlib.axes.Axes or None, optional
        Target axis to draw on. If ``None`` a new figure and axis are created.
    save_fig : bool, default False
        If True the generated figure will be saved using ``save_args``.
    save_args : dict or None, optional
        Options forwarded to the saving helper (may include ``save_folder``,
        ``format``, ``dpi``).
    fit_param : str, optional
        Short or full column name of the parameter to plot (default ``'BE'``).
        Common short forms ("BE", "Area", "At Conc", "Goodness") are
        mapped to full column headers internally.
    plot_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot` for
        the component series (overrides module defaults).

    Returns
    -------
    None

    Notes
    -----
    This function is a thin wrapper around :class:`SpectrumPlotter.plot_report`
    and exists for convenience when plotting a single report mapping.
    """
    # delegate to SpectrumPlotter to keep a compact module-level function
    sp = SpectrumPlotter()
    return sp.plot_report(
        report_dict, ax=ax, save_fig=save_fig, save_args=save_args, **kwargs
    )


def plot_fit_spectrum(spectrum_dict, ax=None, save_fig=False, save_args=None, **kwargs):
    """Plot a spectrum and optional residuals from a spectrum dictionary.

    Convenience wrapper that constructs a :class:`SpectrumPlotter` and calls
    its :meth:`SpectrumPlotter.plot_spectrum` method.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping with series to plot. Expected keys include ``'BE'`` and/or
        ``'KE'`` for x-values and other keys for data (components, residuals).
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Target axis or axes. If ``None``, a new figure/axes pair is created.
    save_fig : bool, default False
        If True the generated figure will be saved using ``save_args``.
    save_args : dict, optional
        Save options passed to :func:`_save_figure` (e.g. ``save_folder``,
        ``format``, ``dpi``).
    x_axis : {'BE', 'KE'}, default 'BE'
        Which x-axis data to use when plotting.
    normalised_residual : bool, default False
        Whether to plot the normalised residual series instead of raw residual.
    plot_kwargs : dict, optional
        Keyword arguments forwarded to ``Axes.plot`` for data series.

    Returns
    -------
    None

    Notes
    -----
    This function is a thin wrapper — most logic lives in
    :class:`SpectrumPlotter` and module-level helper functions.
    """
    sp = SpectrumPlotter()
    return sp.plot_spectrum(
        spectrum_dict, ax=ax, save_fig=save_fig, save_args=save_args, **kwargs
    )
