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


def should_plot_key(key, x_axis, normalised_residual, plot_items=None):
    """Decide whether a key from a spectrum mapping should be plotted.

    Parameters
    ----------
    key : str
        Candidate key from the spectrum dictionary.
    x_axis : str
        The axis currently used for x data (``'BE'`` or ``'KE'``).
    normalised_residual : bool
        Whether normalised residuals are being plotted; affects.
    plot_items : list of str or None, optional
        List of item types to include in the plot. Valid values are:
        'measured', 'background', 'components', 'envelope', 'residual'.
        If None (default), all items are included.

    Returns
    -------
    bool
        True if the key should be plotted as a data series, False otherwise.
    """
    # Default: plot all items when plot_items not specified
    if plot_items is None:
        plot_items = ["measured", "background", "components", "envelope", "residual"]

    # Normalize plot_items to lowercase
    plot_items_lower = [item.lower() for item in plot_items]

    if key in (x_axis, "KE", "BE"):
        return False
    if key in ("File Name", "Sample", "Core Level"):
        return False

    # Check residual keys
    if key in ("Normalised Residual", "Residual"):
        if "residual" not in plot_items_lower:
            return False
        if key == "Normalised Residual" and not normalised_residual:
            return False
        if key == "Residual" and normalised_residual:
            return False
        return True

    # Check measured data
    if key == "Measured":
        return "measured" in plot_items_lower

    # Check background
    if key == "Background":
        return "background" in plot_items_lower

    # Check for envelope
    if key == "Envelope":
        return "envelope" in plot_items_lower

    # Everything else is treated as a component
    return "components" in plot_items_lower


def convert_to_relative_be(report_dict, col_full, reference):
    """Convert binding energy values to relative values with respect to a reference.

    Parameters
    ----------
    report_dict : dict
        Report dictionary containing BE data and component labels.
    col_full : str
        Full column name for binding energy (e.g., "Binding Energy (eV)").
    reference : str
        Component label or name to use as reference (e.g., "A" or "PFx").

    Returns
    -------
    dict
        New dictionary with relative BE values, or original dict if conversion fails.
    """
    if col_full not in report_dict:
        return report_dict

    be_data = report_dict.get(col_full)
    if be_data is None:
        return report_dict

    # Find reference component index (searches both labels and names)
    ref_idx = _find_reference_index(report_dict, reference)
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
        - 'calculate' (str or None): Display statistic in legend. "average"
            shows mean values; "difference" shows last - first; "ratio"
            shows mean ratio; None/empty disables statistics and legend
            annotation.
        - 'swap_axes' (bool): If True, swap x and y axes (default False).
        - 'show_labels' (bool): If True, prepend component labels (e.g., "A", "B")
            to component names in legend (default False).
    """
    plot_options = plot_options or {}
    names = np.array(report_dict.get("Name"), dtype=object)
    comp_labels = report_dict.get("Comp Label")
    if comp_labels is not None:
        comp_labels = np.array(comp_labels, dtype=object)
    y_data_raw = report_dict.get(col_full)
    show_labels = plot_options.get("show_labels", False)

    # Check if data exists and is valid
    if y_data_raw is None:
        print(f"Warning: Column '{col_full}' not found in data. Skipping plot.")
        return

    y_data = np.array(y_data_raw, dtype=object)

    # Check if data is empty or scalar
    if y_data.ndim == 0 or (y_data.ndim > 0 and y_data.shape[0] == 0):
        return

    # Find maximum label length for alignment
    max_len = max(
        len(_get_comp_label(names, i, show_labels, comp_labels))
        for i in range(y_data.shape[0])
    )

    # Detect doublet groups for color coordination
    doublet_info = _get_doublet_styling_info(names)

    # Track colors for doublet groups
    colors_used = {}
    color_idx = 0

    # Plot each component
    for i in range(y_data.shape[0]):
        x_vals = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)

        calc_mode = plot_options.get("calculate", "average")
        numeric_vals = [
            v
            for v in y_data[i]
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
        ]

        # Format legend text based on calculation mode
        if calc_mode == "ratio":
            legend_text = _format_ratio_legend(
                names, comp_labels, i, col_full, show_labels
            )
            stat_label, stat_value = None, None
        elif calc_mode:
            stat_label, stat_value = _calculate_statistic(numeric_vals, calc_mode)
            comp_label = _get_comp_label(names, i, show_labels, comp_labels).ljust(
                max_len
            )
            legend_text = f"{comp_label} ({stat_label}={stat_value:7.2f})"
        else:
            stat_label, stat_value = None, None
            legend_text = _get_comp_label(names, i, show_labels, comp_labels).ljust(
                max_len
            )

        # Apply doublet styling if this component is part of a doublet
        plot_params = params.copy()
        if i in doublet_info:
            base_name, is_first = doublet_info[i]

            # Get or assign color for this doublet group
            if base_name not in colors_used:
                # Use matplotlib color cycle
                colors_used[base_name] = f"C{color_idx}"
                color_idx += 1

            plot_params["color"] = colors_used[base_name]

            # Different markers/linestyles for 3/2 and 1/2 components
            if is_first:
                # 3/2 component: solid line with circle marker
                plot_params["linestyle"] = "-"
                plot_params["marker"] = "o"
            else:
                # 1/2 component: dashed line with square marker
                plot_params["linestyle"] = "--"
                plot_params["marker"] = "s"

        # Plot the component series with average line
        _plot_component_series(
            ax,
            {
                "x_vals": x_vals,
                "y_vals": y_data[i],
                "legend_text": legend_text,
            },
            plot_params,
            {
                "stat_value": stat_value,
                "calculate": calc_mode,
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
        - 'plot_items' (list of str or None): Item types to include in plot.
          Valid values: 'measured', 'background', 'components', 'envelope',
          'residual'. If None, defaults to all items.

    Returns
    -------
    tuple
        ``(handles, labels)`` where ``handles`` is a list of Line2D objects
        and ``labels`` is a list of corresponding legend labels.
    """
    x_axis = plot_options["x_axis"]
    params = plot_options["params"]
    normalised_residual = plot_options.get("normalised_residual", False)
    plot_items = plot_options.get("plot_items", None)

    handles = []
    labels = []
    for key, values in spectrum_dict.items():
        if not should_plot_key(key, x_axis, normalised_residual, plot_items):
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


def _find_reference_index(report_dict, reference):
    """Find the index of the reference component by label or name.

    Searches both 'Comp Label' (e.g., A, B, C) and 'Name' (e.g., LiF, PFx)
    fields to find a matching component.

    Parameters
    ----------
    report_dict : dict
        Report dictionary containing 'Comp Label' and/or 'Name' fields.
    reference : str
        Reference component label or name to find.

    Returns
    -------
    int or None
        Index of reference component, or None if not found.
    """
    # Try Comp Label first (typically single letters like A, B, C)
    comp_labels = report_dict.get("Comp Label")
    if comp_labels is not None:
        comp_labels = np.array(comp_labels, dtype=object)
        for i in range(comp_labels.shape[0]):
            label = comp_labels[i, 0] if comp_labels.ndim > 1 else comp_labels[i]
            if str(label) == str(reference):
                return i

    # Try Name field (chemical species names)
    names = report_dict.get("Name")
    if names is not None:
        names = np.array(names, dtype=object)
        for i in range(names.shape[0]):
            name = names[i, 0] if names.ndim > 1 else names[i]
            if str(name) == str(reference):
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


def _get_comp_label(names, index, show_label=False, comp_labels=None):
    """Extract component label from names array at given index.

    Parameters
    ----------
    names : np.ndarray
        Array of component names.
    index : int
        Index of the component.
    show_label : bool, optional
        If True and comp_labels is provided, prepend the component label
        (e.g., "A: ComponentName").
    comp_labels : np.ndarray or None, optional
        Array of component labels (e.g., "A", "B", "C").

    Returns
    -------
    str
        Component label as string, optionally with label prefix.
    """
    label = (
        names[index][0]
        if isinstance(names[index], (list, np.ndarray))
        else names[index]
    )
    label_str = str(label)

    # Prepend component label if requested
    if show_label and comp_labels is not None:
        comp_label = (
            comp_labels[index][0]
            if isinstance(comp_labels[index], (list, np.ndarray))
            else comp_labels[index]
        )
        label_str = f"{comp_label}: {label_str}"

    return label_str


def _get_doublet_styling_info(names):
    """Detect doublet pairs and return styling information.

    Returns a dict mapping component index to (base_name, is_first) tuple,
    where base_name identifies the doublet group and is_first indicates
    if it's the 3/2 (True) or 1/2 (False) component.
    """
    doublet_info = {}

    # Extract name strings
    if names.ndim > 1:
        name_strs = [str(names[i, 0]) for i in range(names.shape[0])]
    else:
        name_strs = [str(names[i]) for i in range(names.shape[0])]

    # Find doublet pairs (names ending with (3/2) and (1/2))
    doublet_groups = {}
    for i, name in enumerate(name_strs):
        if " (3/2)" in name:
            base_name = name.replace(" (3/2)", "")
            doublet_groups[base_name] = doublet_groups.get(base_name, {})
            doublet_groups[base_name]["first"] = i
        elif " (1/2)" in name:
            base_name = name.replace(" (1/2)", "")
            doublet_groups[base_name] = doublet_groups.get(base_name, {})
            doublet_groups[base_name]["second"] = i

    # Build the output dict for complete pairs only
    for base_name, indices in doublet_groups.items():
        if "first" in indices and "second" in indices:
            doublet_info[indices["first"]] = (base_name, True)
            doublet_info[indices["second"]] = (base_name, False)

    return doublet_info


def _format_ratio_legend(names, comp_labels, index, col_full, show_labels):
    """Format legend text for area ratio plots.

    Creates legend text in the format "Component / Reference" using actual
    component names and optionally prepending component labels.

    Parameters
    ----------
    names : np.ndarray
        Array of component names.
    comp_labels : np.ndarray or None
        Array of component labels.
    index : int
        Index of the numerator component.
    col_full : str
        Full column name containing reference component identifier.
    show_labels : bool
        Whether to prepend component labels to names.

    Returns
    -------
    str
        Formatted legend text.
    """
    # Extract reference name from column title
    ref_name = None
    if col_full.startswith("Area Ratio (ref comp ") and col_full.endswith(")"):
        ref_name = col_full[len("Area Ratio (ref comp ") : -1]

    # Find reference component in names array and get actual name
    ref_comp_name = ref_name
    ref_comp_label = None
    if ref_name is not None:
        for idx in range(names.shape[0]):
            label = names[idx, 0] if names.ndim > 1 else names[idx]
            if str(label) == ref_name:
                # Use actual component name if available
                ref_comp_name = (
                    names[idx, 1] if names.ndim > 1 and names.shape[1] > 1 else label
                )
                # Get label for reference if show_labels is True
                if show_labels and comp_labels is not None:
                    ref_comp_label = (
                        comp_labels[idx, 0]
                        if comp_labels.ndim > 1
                        else comp_labels[idx]
                    )
                break

    # Get the actual component name for the numerator
    num_comp_name = (
        names[index, 1] if names.ndim > 1 and names.shape[1] > 1 else names[index]
    )

    # Add label prefix if show_labels is True
    if show_labels and comp_labels is not None:
        num_comp_label = (
            comp_labels[index, 0] if comp_labels.ndim > 1 else comp_labels[index]
        )
        num_comp_name = f"{num_comp_label}: {num_comp_name}"
        if ref_comp_label:
            ref_comp_name = f"{ref_comp_label}: {ref_comp_name}"

    return f"{num_comp_name} / {ref_comp_name}" if ref_comp_name else str(num_comp_name)


def _calculate_statistic(numeric_vals, calculate):
    """Calculate the statistic to display in legend (or none if disabled).

    Parameters
    ----------
    numeric_vals : list
        List of numeric values from the data.
    calculate : str
        Calculation type: "average", "difference", "difference_max", or "ratio".

    Returns
    -------
    tuple
        (stat_label, display_value) where stat_label is the label string
        and display_value is the numeric value to display.
    """
    if calculate is None or calculate == "":
        return None, None
    if calculate == "difference":
        if len(numeric_vals) >= 2:
            return "diff", numeric_vals[-1] - numeric_vals[0]
        return "diff", numeric_vals[0] if numeric_vals else 0
    if calculate == "difference_max":
        if len(numeric_vals) >= 2:
            return "max-min", max(numeric_vals) - min(numeric_vals)
        return "max-min", 0
    if calculate == "average":
        return "avg", np.mean(numeric_vals) if numeric_vals else 0
    if calculate == "ratio":
        return "ratio", np.mean(numeric_vals) if numeric_vals else 0
    return None, None


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

    # Add guide lines for statistics
    if calculate == "average" and stat_value is not None:
        if swap_axes:
            ax.plot(
                [stat_value] * len(x_vals),
                x_vals,
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
        else:
            ax.plot(
                x_vals,
                [stat_value] * len(x_vals),
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
    elif calculate == "difference" and stat_value is not None and len(x_vals) >= 2:
        first_val = y_vals[0]
        last_val = y_vals[-1]
        if swap_axes:
            # Vertical guides at the two y-values (x is swapped)
            ax.plot(
                [first_val] * len(x_vals),
                x_vals,
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
            ax.plot(
                [last_val] * len(x_vals),
                x_vals,
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
        else:
            # Horizontal guides at the two y-values
            ax.plot(
                x_vals,
                [first_val] * len(x_vals),
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
            ax.plot(
                x_vals,
                [last_val] * len(x_vals),
                color=line.get_color(),
                alpha=0.7,
                linestyle="--",
            )
