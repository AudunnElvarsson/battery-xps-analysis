"""Plot rendering utilities for XPS plotting.

This module handles the core plotting logic including parameter management,
data series plotting, and plot filtering for XPS spectra and fit reports.
"""

import numpy as np
from .axis_management import select_target_axis


def get_column_name(fit_param):
    """Map short parameter names to full column headers.

    This function provides convenient shortcuts for common fit parameters
    while also allowing custom parameter names to pass through unchanged.

    Parameters
    ----------
    fit_param : str
        Parameter name, either short form (e.g., 'BE', 'Area') or full
        column header.

    Returns
    -------
    str
        Full column header name. If the input is a recognized short form,
        returns the mapped full name; otherwise returns the input unchanged.
    """
    param_mapping = {
        "BE": "Binding Energy (eV)",
        "Area": "Raw Area",
        "At Conc": "%At Conc",
        "Goodness": "Goodness of Fit",
    }
    return param_mapping.get(fit_param, fit_param)


def update_plot_params(defaults, user_kwargs):
    """Merge default plotting parameters with user-supplied overrides.

    This function accepts both full and abbreviated matplotlib parameter names
    and normalizes them to full names to avoid conflicts (e.g., 'ls' → 'linestyle').

    Parameters
    ----------
    defaults : dict
        Default plotting parameters (e.g. ``linestyle``, ``linewidth``).
    user_kwargs : dict or None
        Optional user-supplied overrides; keys here overwrite ``defaults``.

    Returns
    -------
    dict
        A new dictionary containing the merged plotting parameters with
        normalized (full) parameter names.
    """
    # Mapping of abbreviated to full parameter names
    param_aliases = {
        "ls": "linestyle",
        "lw": "linewidth",
        "c": "color",
        "m": "marker",
        "mec": "markeredgecolor",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "ms": "markersize",
    }

    # Normalize defaults by replacing abbreviated names with full names
    params = {}
    for key, value in defaults.items():
        full_key = param_aliases.get(key, key)
        params[full_key] = value

    if user_kwargs:
        # Normalize user_kwargs by replacing abbreviated names with full names
        for key, value in user_kwargs.items():
            full_key = param_aliases.get(key, key)
            params[full_key] = value

    return params


def should_plot_key(key, x_axis, normalised_residual):
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
    if key in ("File Name", "Sample", "Core Level"):
        return False
    if key == "Normalised Residual" and not normalised_residual:
        return False
    if key == "Residual" and normalised_residual:
        return False
    return True


def plot_report_series(report_dict, ax, col_full, params):
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


def plot_spectrum_series(
    spectrum_dict, ax, x, x_axis, params, normalised_residual=False
):
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
    x_axis : str
        Selected axis name (``'BE'`` or ``'KE'``).
    params : dict
        Plotting kwargs forwarded to ``Axes.plot``.
    normalised_residual : bool, default False
        Whether to plot normalised residuals instead of raw residuals.

    Returns
    -------
    tuple
        ``(handles, labels)`` where ``handles`` is a list of Line2D objects
        and ``labels`` is a list of corresponding legend labels.
    """
    handles = []
    labels = []
    for key, values in spectrum_dict.items():
        if not should_plot_key(key, x_axis, normalised_residual):
            continue
        target_ax = select_target_axis(ax, key)
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
