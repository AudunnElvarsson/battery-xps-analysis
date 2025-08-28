"""xps_plot
-----------

Helpers and plotting routines for visualising XPS fit reports and
spectra. Functions accept dictionaries produced by the parsing utilities in
``xps_processing`` and produce Matplotlib figures/axes. This module also
provides small helpers for saving figures with consistent filenames.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np


# === Module-level helper functions ===
def _update_plot_params(defaults, user_kwargs):
    """Merge default plot parameters with user-supplied overrides.

    Parameters
    ----------
    defaults : dict
        Default plotting parameters (e.g. linestyle, linewidth).
    user_kwargs : dict | None
        User-supplied overrides; values here overwrite ``defaults``.

    Returns
    -------
    dict
        Merged plotting parameters.
    """
    params = defaults.copy()
    if user_kwargs:
        params.update(user_kwargs)
    return params


def _build_save_info(save_folder, name_list, save_args=None, prefix="figure"):
    """Build a filesystem-safe filename and matplotlib save kwargs.

    Parameters
    ----------
    save_folder : str
        Target folder for saving; created if it does not exist.
    name_list : list[str]
        Parts to include in the filename (joined with underscores). Empty
        entries are ignored.
    save_args : dict | None
        Optional overrides forwarded to ``savefig`` (keys like ``format``,
        ``dpi``, ``bbox_inches``). The special key ``save_folder`` is ignored
        here and handled by the caller.
    prefix : str
        Optional filename prefix. If empty, no prefix and underscore are used.

    Returns
    -------
    tuple[str, dict]
        (absolute save path, keyword args for ``matplotlib.figure.Figure.savefig``)
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
    """Save a Matplotlib figure using a generated, safe filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name_list : list[str] | None
        Name parts used to construct the filename. If ``None``, defaults to
        an empty list and the filename will be ``all.<format>`` or similar.
    save_args : dict | None
        Save options. Recognized keys forwarded to ``savefig``; the special
        ``save_folder`` key selects the destination folder.
    prefix : str
        Optional prefix for the filename. When empty, the prefix is omitted so
        a bare basename can be used.

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
    """Return x data, x-axis label and whether to invert the axis."""
    if x_axis == "KE":
        return spectrum_dict.get("KE"), "Kinetic Energy (eV)", False
    return spectrum_dict.get("BE"), "Binding Energy (eV)", True


def _should_plot_key(key, x_axis, normalised_residual):
    """Decide whether a dictionary key should be plotted."""
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
    """Return a Matplotlib Figure for a provided axis-like input."""
    try:
        return ax_in[0].get_figure()
    except (TypeError, IndexError, AttributeError):
        try:
            return ax_in.get_figure()
        except (AttributeError, TypeError):
            return plt.gcf()


def _select_target_axis(ax, key):
    """Select appropriate axis for a given key (residuals on ax[0], else main)."""
    try:
        if key in ("Normalised Residual", "Residual"):
            return ax[0]
        return ax[1]
    except (TypeError, IndexError, AttributeError):
        return ax


def _derive_file_name(mapping):
    """Try to derive a sensible name from mapping (File Name / Sample / Name)."""
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
    """Return (fig, ax, main_ax, plot_here) for a given ax-like input.

    This consolidates the common pattern of creating a new figure when
    ``ax`` is None and selecting the main axis (ax[1] when available).
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
    """(Deprecated) kept for compatibility; see _configure_axes.

    This function is superseded by ``_configure_axes`` which performs both
    inversion and residual-axis formatting. New code should use
    ``_configure_axes(ax, main_ax, invert)``.
    """
    return _configure_axes(ax, main_ax, invert)


def _format_residual_axis(ax, main_ax):
    """Apply residual-axis specific formatting when present.

    This hides ticks/spines on the residual axis and positions the bottom
    spine at y=0 so the residual baseline is visible.
    """
    # Deprecated; functionality moved to _configure_axes
    return _configure_axes(ax, main_ax, invert=True)


def _configure_axes(ax, main_ax, invert=False):
    """Configure axes: optionally invert x-axes and apply residual formatting.

    - If ``invert`` is True, attempt to invert the main x-axis and the
      residual axis (ax[0]) when present.
    - Apply residual-axis formatting (ylabel, bottom spine at y=0, hide
      ticks/spines) when ax[0] exists.

    The helper swallows AttributeError/IndexError/TypeError to remain robust
    against single-Axes inputs.
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
        ax[0].spines["top"].set_visible(False)
    except (AttributeError, IndexError, TypeError):
        pass


def _plot_report_series(report_dict, ax, col_full, params):
    """Plot rows from a fit report dict on a provided axis.

    Keeps the row/average plotting logic out of plot_fit_report to reduce
    local variable pressure in the main function.
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
    """Plot all series from ``spectrum_dict`` to appropriate axes and return handles/labels.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping containing series to plot.
    ax : matplotlib.axes.Axes | sequence
        Axis or axes used for plotting.
    x : array-like | None
        X values to use for series that match length.
    opts : dict
        Options bag with keys:
          - "x_axis" : str
          - "params" : dict (plot kwargs)
          - "normalised_residual" : bool
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
    """Helper class that groups plotting helpers and state for spectra/report plotting.

    This class centralises plotting logic so instance methods can share state
    and reduce the number of parameters passed between helpers.
    """

    def __init__(self):
        # placeholder for future shared state
        self.fig = None
        self.ax = None
        self.main_ax = None
        self.plot_here = False

    # --- instance helpers to reduce parameter passing ---
    def _ensure_axes(self, ax_in):
        """Ensure self.fig, self.ax, self.main_ax, self.plot_here are set."""
        self.fig, self.ax, self.main_ax, self.plot_here = _ensure_axes_and_main(ax_in)

    def _derive_col_full(self, fit_param):
        return {
            "BE": "Binding Energy (eV)",
            "Area": "Raw Area",
            "At Conc": "%At Conc",
            "Goodness": "Goodness of Fit",
        }.get(fit_param, fit_param)

    def _plot_report_series_internal(self, report_dict, col_full, params):
        """Delegate to module helper using instance axes."""
        _plot_report_series(report_dict, self.main_ax, col_full, params)

    def _derive_xinfo(self, spectrum_dict, x_axis):
        """Return xinfo and store nothing; thin wrapper for module helper."""
        return _get_x_axis_from_dict(spectrum_dict, x_axis)

    def _plot_spectrum_series_internal(self, spectrum_dict, x, opts):
        """Delegate to module helper using instance axes."""
        return _plot_spectrum_series(spectrum_dict, self.ax, x, opts)

    def _configure_axes_instance(self, invert=False):
        _configure_axes(self.ax, self.main_ax, invert=invert)

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
        self._plot_report_series_internal(report_dict, col_full, params)

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

        xinfo = self._derive_xinfo(spectrum_dict, x_axis)
        handles, labels = self._plot_spectrum_series_internal(
            spectrum_dict,
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

        self._configure_axes_instance(invert=xinfo[2])

        self._save_if_requested(save_fig, save_args, spectrum_dict)

        if self.plot_here:
            plt.show()


def plot_fit_report(report_dict, ax=None, save_fig=False, save_args=None, **kwargs):
    """Plot a single fit parameter for all components from a report dictionary.

    The function expects ``report_dict`` to contain the cleaned table produced
    by ``read_fit_report_file``. Each component becomes a separate series; a
    horizontal line showing the component average is also drawn and included in
    the legend.

    Parameters
    ----------
    report_dict : dict
        Dictionary returned from ``read_fit_report_file`` (header -> 2D arrays).
    ax : matplotlib.axes.Axes | None
        Target axis to draw on. If ``None``, a new figure and axis are created.
    fit_param : str
        Short form or full column name of the parameter to plot (e.g. "BE",
        "Area", "At Conc"). Some shorthand values are mapped internally.
    kwargs : dict | None
        Plot styling kwargs forwarded to ``Axes.plot`` (overrides module
        defaults).
    save_fig : bool
        If True, the generated figure is saved using ``_save_figure`` and
        ``save_args``.
    save_args : dict | None
        Save options; may include ``save_folder`` and parameters for
        ``Figure.savefig`` (``format``, ``dpi``, etc.).

    Returns
    -------
    None
    """
    # delegate to SpectrumPlotter to keep a compact module-level function
    sp = SpectrumPlotter()
    return sp.plot_report(
        report_dict, ax=ax, save_fig=save_fig, save_args=save_args, **kwargs
    )


def plot_fit_spectrum(spectrum_dict, ax=None, save_fig=False, save_args=None, **kwargs):
    """Plot spectrum data and optional residuals from a spectrum dictionary.

    Supported kwargs:
      - x_axis: "BE" (default) or "KE"
      - normalised_residual: bool
      - plot_kwargs: dict forwarded to Axes.plot
    """
    sp = SpectrumPlotter()
    return sp.plot_spectrum(
        spectrum_dict, ax=ax, save_fig=save_fig, save_args=save_args, **kwargs
    )
