"""Plot rendering utilities for XPS plotting.

This module handles the core plotting logic including parameter management,
data series plotting, and plot filtering for XPS spectra and fit reports.

Public API
----------
get_column_name : Map short parameter names to full column headers
update_plot_params : Merge default plotting parameters with user overrides
should_plot_key : Decide whether a key should be plotted
convert_to_relative_be : Convert BE values to relative values vs reference
plot_report_series : Plot fit report data series
plot_spectrum_series : Plot spectrum data series

Note
----
Functions starting with underscore (_) are internal helpers and should
not be used directly.
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


def convert_to_relative_be(report_dict, col_full, reference):
    """Convert binding energy values to relative values with respect to a reference.

    Parameters
    ----------
    report_dict : dict
        Report dictionary containing BE data and component labels.
    col_full : str
        Full column name for binding energy (e.g., "Binding Energy (eV)").
    reference : str
        Component label to use as reference.

    Returns
    -------
    dict
        New dictionary with relative BE values, or original dict if conversion fails.
    """
    if col_full not in report_dict or "Comp Label" not in report_dict:
        return report_dict

    labels = report_dict.get("Comp Label")
    be_data = report_dict.get(col_full)

    if labels is None or be_data is None:
        return report_dict

    # Find reference component index
    ref_idx = _find_reference_index(labels, reference)
    if ref_idx is None:
        print(
            f"Warning: Reference component '{reference}' not found. Using absolute BE."
        )
        return report_dict

    # Get reference BE value
    ref_be = _get_reference_be(be_data, ref_idx)
    if ref_be is None:
        print("Warning: Reference BE is not numeric. Using absolute BE.")
        return report_dict

    # Create new dict with relative BE
    new_dict = report_dict.copy()
    new_dict["Relative Binding Energy (eV)"] = _convert_be_array(be_data, ref_be)
    return new_dict


def plot_report_series(report_dict, ax, col_full, params, plot_options=None):
    """Plot rows from a fit report mapping onto an axis.

    Each row in the report is plotted as a separate series. A horizontal
    dashed line and legend annotation show either the average or the
    difference (last - first) depending on the ``calculate`` parameter.

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
    plot_options : dict or None, optional
        Additional plotting options. Supported keys:
        - 'calculate' (str): Display statistic in legend, either "average"
          (default) to show mean values, or "difference" to show the
          difference between last and first values.
        - 'swap_axes' (bool): If True, swap x and y axes (default False).
    """
    plot_options = plot_options or {}
    names = np.array(report_dict.get("Name"), dtype=object)
    y_data = np.array(report_dict.get(col_full), dtype=object)

    # Find maximum label length for alignment
    max_len = max(len(_get_comp_label(names, i)) for i in range(y_data.shape[0]))

    # Plot each component
    for i in range(y_data.shape[0]):
        x_vals = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)

        # Calculate statistic and format legend label
        stat_label, stat_value = _calculate_statistic(
            [v for v in y_data[i] if isinstance(v, (int, float, np.floating))],
            plot_options.get("calculate", "average"),
        )

        # Format legend text
        comp_label = _get_comp_label(names, i).ljust(max_len)
        legend_text = f"{comp_label} ({stat_label}={stat_value:7.2f})"

        # Plot the component series with average line
        _plot_component_series(
            ax,
            {
                "x_vals": x_vals,
                "y_vals": y_data[i],
                "legend_text": legend_text,
            },
            params,
            {
                "stat_value": stat_value,
                "calculate": plot_options.get("calculate", "average"),
                "swap_axes": plot_options.get("swap_axes", False),
            },
        )


def plot_spectrum_series(spectrum_dict, ax, x_values, plot_options):
    """Plot all series contained in a spectrum mapping and return legend info.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping where keys are series names and values are numeric sequences
        or arrays.
    ax : Axes or sequence of Axes
        Axis or axes used for plotting. When a sequence is supplied residual
        series are plotted on the residual axis (index 0).
    x_values : array-like or None
        Optional x-values to use for series that have matching length.
    plot_options : dict
        Dictionary containing plotting options with keys:
        - 'x_axis' (str): Selected axis name ('BE' or 'KE').
        - 'params' (dict): Plotting kwargs forwarded to Axes.plot.
        - 'normalised_residual' (bool): Whether to plot normalised residuals.

    Returns
    -------
    tuple
        ``(handles, labels)`` where ``handles`` is a list of Line2D objects
        and ``labels`` is a list of corresponding legend labels.
    """
    x_axis = plot_options["x_axis"]
    params = plot_options["params"]
    normalised_residual = plot_options.get("normalised_residual", False)

    handles = []
    labels = []
    for key, values in spectrum_dict.items():
        if not should_plot_key(key, x_axis, normalised_residual):
            continue
        target_ax = select_target_axis(ax, key)
        try:
            cond = x_values is not None and len(x_values) == len(values)
        except TypeError:
            cond = False
        xs = x_values if cond else np.arange(len(values))
        try:
            (line,) = target_ax.plot(xs, values, label=key, **params)
        except (TypeError, ValueError):
            # skip series that cannot be plotted
            continue
        handles.append(line)
        labels.append(key)
    return handles, labels


def _find_reference_index(labels, reference):
    """Find the index of the reference component by label.

    Parameters
    ----------
    labels : np.ndarray
        Array of component labels.
    reference : str
        Reference component label to find.

    Returns
    -------
    int or None
        Index of reference component, or None if not found.
    """
    for i in range(labels.shape[0]):
        label = labels[i, 0] if labels.ndim > 1 else labels[i]
        if str(label) == reference:
            return i
    return None


def _get_reference_be(be_data, ref_idx):
    """Extract reference BE value from data array.

    Parameters
    ----------
    be_data : np.ndarray
        Binding energy data array.
    ref_idx : int
        Index of reference component.

    Returns
    -------
    float or None
        Reference BE value, or None if not numeric.
    """
    ref_be = be_data[ref_idx, 0] if be_data.ndim > 1 else be_data[ref_idx]
    if isinstance(ref_be, (int, float, np.integer, np.floating)):
        return float(ref_be)
    return None


def _convert_be_array(be_data, ref_be):
    """Convert BE array to relative values.

    Parameters
    ----------
    be_data : np.ndarray
        Original binding energy data.
    ref_be : float
        Reference BE value to subtract.

    Returns
    -------
    np.ndarray
        Array with relative BE values.
    """
    rel_be_data = np.array(be_data, dtype=object)
    for i in range(rel_be_data.shape[0]):
        if rel_be_data.ndim > 1:
            for j in range(rel_be_data.shape[1]):
                if isinstance(rel_be_data[i, j], (int, float, np.integer, np.floating)):
                    rel_be_data[i, j] = float(rel_be_data[i, j]) - ref_be
        else:
            if isinstance(rel_be_data[i], (int, float, np.integer, np.floating)):
                rel_be_data[i] = float(rel_be_data[i]) - ref_be
    return rel_be_data


def _get_comp_label(names, index):
    """Extract component label from names array at given index.

    Parameters
    ----------
    names : np.ndarray
        Array of component names.
    index : int
        Index of the component.

    Returns
    -------
    str
        Component label as string.
    """
    label = (
        names[index][0]
        if isinstance(names[index], (list, np.ndarray))
        else names[index]
    )
    return str(label)


def _calculate_statistic(numeric_vals, calculate):
    """Calculate the statistic to display in legend.

    Parameters
    ----------
    numeric_vals : list
        List of numeric values from the data.
    calculate : str
        Calculation type: "average" or "difference".

    Returns
    -------
    tuple
        (stat_label, display_value) where stat_label is the label string
        and display_value is the numeric value to display.
    """
    if calculate == "difference":
        if len(numeric_vals) >= 2:
            return "diff", numeric_vals[-1] - numeric_vals[0]
        return "val", numeric_vals[0] if numeric_vals else 0
    return "avg", np.mean(numeric_vals) if numeric_vals else 0  # calculate == "average"


def _plot_component_series(ax, data_opts, params, plot_opts):
    """Plot a single component series with optional average line.

    Parameters
    ----------
    ax : Axes
        Target axis for plotting.
    data_opts : dict
        Dictionary with keys 'x_vals', 'y_vals', and 'legend_text'.
    params : dict
        Keyword arguments forwarded to plot.
    plot_opts : dict
        Dictionary with keys 'stat_value' (float), 'calculate' (str),
        and 'swap_axes' (bool).
    """
    x_vals = data_opts["x_vals"]
    y_vals = data_opts["y_vals"]
    legend_text = data_opts["legend_text"]
    stat_value = plot_opts["stat_value"]
    calculate = plot_opts["calculate"]
    swap_axes = plot_opts["swap_axes"]

    # Plot data (swap axes if requested)
    if swap_axes:
        (line,) = ax.plot(y_vals, x_vals, label=legend_text, **params)
    else:
        (line,) = ax.plot(x_vals, y_vals, label=legend_text, **params)

    # Add horizontal/vertical average line if in average mode
    if calculate == "average":
        if swap_axes:
            ax.plot(
                [stat_value] * len(x_vals),
                x_vals,
                color=line.get_color(),
                alpha=0.7,
                linestyle=":",
            )
        else:
            ax.plot(
                x_vals,
                [stat_value] * len(x_vals),
                color=line.get_color(),
                alpha=0.7,
                linestyle=":",
            )
