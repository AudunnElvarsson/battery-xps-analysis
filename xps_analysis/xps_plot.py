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


# === Main plotting functions ===
def plot_fit_report(
    report_dict, ax=None, fit_param="BE", kwargs=None, save_fig=False, save_args=None
):
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
    defaults = {"linestyle": "--", "linewidth": 1.5, "marker": "o"}
    params = _update_plot_params(defaults, kwargs)

    if report_dict is None:
        print("No data to plot.")
        return

    plot_here = False
    if ax is None:
        plot_here = True
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.set_tight_layout(True)
    else:
        # ensure we have the Figure object for potential saving
        try:
            fig = ax.get_figure()
        except (AttributeError, TypeError):
            fig = plt.gcf()

    # Map shorthand to full column names
    col_map = {
        "BE": "Binding Energy (eV)",
        "Area": "Raw Area",
        "At Conc": "%At Conc",
        "Goodness": "Goodness of Fit",
    }
    col_full = col_map.get(fit_param, fit_param)
    names = np.array(report_dict.get("Name"), dtype=object)
    y_data = np.array(report_dict.get(col_full), dtype=object)

    for i in range(y_data.shape[0]):
        y = y_data[i]
        x = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)
        label = names[i][0] if isinstance(names[i], (list, np.ndarray)) else names[i]
        avg = np.mean([v for v in y if isinstance(v, (int, float, np.floating))])
        (line,) = ax.plot(x, y, label=f"{label} (avg={avg:.2f})", **params)
        ax.plot(x, [avg] * len(x), color=line.get_color(), alpha=0.7, linestyle=":")

    ax.set_xlabel("Experimental Variable")
    ax.set_ylabel(col_full)
    ax.legend()

    if save_fig:
        # prefer 'File Name' if present, else fallback to Sample/Name
        file_name = (
            report_dict.get("File Name")
            or report_dict.get("Sample")
            or report_dict.get("Name")
            or "report"
        )
        if isinstance(file_name, (list, np.ndarray)):
            file_name = file_name[0] if len(file_name) else "report"
    _save_figure(fig, name_list=[file_name], save_args=save_args, prefix="")

    if plot_here:
        plt.show()


def plot_fit_spectrum(
    spectrum_dict,
    ax=None,
    x_axis="BE",
    kwargs=None,
    normalised_residual=False,
    save_fig=False,
    save_args=None,
):
    """Plot spectrum data and optional residuals from a spectrum dictionary.

    The function supports plotting binding-energy (BE) or kinetic-energy (KE)
    on the x-axis. Residuals may be plotted on a separate axis when provided
    in the dictionary. Non-numeric metadata keys (e.g. ``"File Name"``)
    are ignored automatically.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping produced by ``read_fit_spectrum_file``: column name -> 1D array.
    ax : matplotlib.axes.Axes | Sequence[Axes] | None
        Axis or sequence of axes to draw on. If ``None``, a new figure/axis is
        created. If a sequence is passed, residuals will be plotted on
        ``ax[0]`` and the main spectrum on ``ax[1]``.
    x_axis : {'BE', 'KE'}
        Which energy axis to use. ``'BE'`` (default) will be inverted to match
        common XPS conventions.
    kwargs : dict | None
        Plot styling kwargs forwarded to plot calls.
    normalised_residual : bool
        When True, plot the key ``'Normalised Residual'`` instead of
        ``'Residual'`` if available.
    save_fig : bool
        If True, save the figure via ``_save_figure``.
    save_args : dict | None
        Save options; may include ``save_folder`` and parameters forwarded to
        ``Figure.savefig``.

    Returns
    -------
    None
    """

    def _get_x_axis(spectrum_dict, x_axis):
        """Return x data, x-axis label and whether to invert the axis.

        Parameters
        ----------
        spectrum_dict : dict
            Spectrum mapping with possible keys ``'BE'`` or ``'KE'``.
        x_axis : str
            Requested axis identifier (``'BE'`` or ``'KE'``).

        Returns
        -------
        tuple[array | None, str, bool]
            (x array or None, x-axis label, invert_flag)
        """
        if x_axis == "KE":
            x = spectrum_dict.get("KE")
            xlabel = "Kinetic Energy (eV)"
            invert = False
        else:
            x = spectrum_dict.get("BE")
            xlabel = "Binding Energy (eV)"
            invert = True
        return x, xlabel, invert

    def _should_plot(key):
        """Decide whether a dictionary key should be plotted.

        Excludes the selected x-axis column and common metadata fields. Also
        toggles between normalised and raw residual keys based on
        ``normalised_residual``.
        """
        if key in (x_axis, "KE", "BE"):
            return False
        # skip metadata fields that are not numeric arrays
        if key in ("File Name", "Sample"):
            return False
        if key == "Normalised Residual" and not normalised_residual:
            return False
        if key == "Residual" and normalised_residual:
            return False
        return True

    def _figure_from_axes(ax_in):
        """Return a Matplotlib Figure for a provided axis-like input.

        This helper accepts a single ``Axes`` object or an array/sequence of
        axes and returns the corresponding ``Figure`` instance. Falls back to
        ``plt.gcf()`` when no figure can be resolved.
        """
        try:
            # array-like of axes
            return ax_in[0].get_figure()
        except (TypeError, IndexError, AttributeError):
            try:
                return ax_in.get_figure()
            except (AttributeError, TypeError):
                return plt.gcf()

    defaults = {"linestyle": "-", "linewidth": 1.5}
    params = _update_plot_params(defaults, kwargs)

    if spectrum_dict is None:
        print("No data to plot.")
        return

    plot_here = False
    if ax is None:
        plot_here = True
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.set_tight_layout(True)
    else:
        fig = _figure_from_axes(ax)

    x, xlabel, invert = _get_x_axis(spectrum_dict, x_axis)

    # collect handles/labels for combined legend
    handles = []
    labels = []
    for key, values in spectrum_dict.items():
        if not _should_plot(key):
            continue
        # decide which axis to use (residual on ax[0], others on ax[1])
        target_ax = (
            ax[0]
            if key in ("Normalised Residual", "Residual")
            else (ax[1] if hasattr(ax, "__len__") else ax)
        )
        if x is not None and len(x) == len(values):
            (line,) = target_ax.plot(x, values, label=key, **params)
        else:
            (line,) = target_ax.plot(
                np.arange(len(values)), values, label=key, **params
            )
        handles.append(line)
        labels.append(key)

    # set labels and legend on main axis
    main_ax = ax[1] if hasattr(ax, "__len__") else ax
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel("Intensity (a.u.)")
    # Combine legends from both axes into one on main_ax
    if handles:
        main_ax.legend(handles, labels)

        if invert:
            # invert only the main axis and the residual axis if present
            try:
                main_ax.invert_xaxis()
            except (AttributeError, TypeError):
                pass
            try:
                ax[0].invert_xaxis()
            except (AttributeError, IndexError, TypeError):
                pass

    # residual axis adjustments (if present)
    try:
        ax[0].set_ylabel("Residual")
        ax[0].spines["bottom"].set_position(("data", 0))
        ax[0].set_xticks([])
        ax[0].set_xticklabels([])
        main_ax.xaxis.set_tick_params(labelbottom=True, bottom=True)
        main_ax.spines["top"].set_visible(False)
        # remove boxes on individual axes: hide right/top/left spines
        for sp in ("top", "right", "left"):
            ax[0].spines[sp].set_visible(False)
            main_ax.spines[sp].set_visible(False)
    except (AttributeError, IndexError, TypeError):
        # single-axis case or unexpected axes shape; ignore residual-specific formatting
        pass

    if save_fig:
        # try to derive a sensible name from spectrum_dict, fallback to "spectrum"
        file_name = (
            spectrum_dict.get("File Name")
            or spectrum_dict.get("Sample")
            or spectrum_dict.get("Name")
            or "spectrum"
        )
        if isinstance(file_name, (list, np.ndarray)):
            file_name = file_name[0] if len(file_name) else "spectrum"
    _save_figure(fig, name_list=[file_name], save_args=save_args, prefix="")

    if plot_here:
        plt.show()
