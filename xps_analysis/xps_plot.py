"""
Module for plotting XPS data.
"""

import matplotlib.pyplot as plt
import numpy as np


# === Module-level helper functions ===
def _update_plot_params(defaults, user_kwargs):
    """
    Merge default plot parameters with user-supplied overrides.
    """
    params = defaults.copy()
    if user_kwargs:
        params.update(user_kwargs)
    return params


# === Main plotting functions ===
def plot_fit_report(report_dict, axis=None, fit_param="BE", kwargs=None):
    """
    Plot the selected fit parameter for all the core level components.

    Parameters:
        report_dict (dict): Dictionary returned from read_report containing XPS table data.
        fit_param (str): The fit parameter to plot on the y-axis. Options: "BE", "FWHM",
            "Area", "Area/(RSF*T*MFP)", "At Conc", "Goodness of Fit"
        axis: matplotlib axis to plot on (optional).
        kwargs: dict of plot style arguments (optional).
    """
    defaults = {"linestyle": "--", "linewidth": 1.5, "marker": "o"}
    params = _update_plot_params(defaults, kwargs)

    if report_dict is None:
        print("No data to plot.")
        return

    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 6))
        fig.set_tight_layout(True)

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
        (line,) = axis.plot(x, y, label=f"{label} (avg={avg:.2f})", **params)
        axis.plot(x, [avg] * len(x), color=line.get_color(), alpha=0.7, linestyle=":")

    axis.set_xlabel("Experimental Variable")
    axis.set_ylabel(col_full)
    axis.legend()
    plt.show()


def plot_fit_spectrum(
    spectrum_dict, axis=None, x_axis="BE", kwargs=None, normalised_residual=False
):
    """
    Plot the fit spectrum from the specified dictionary.

    Parameters:
        spectrum_dict (dict): Dictionary of spectrum data.
        axis: matplotlib axis or list of axes to plot on.
        x_axis (str): 'BE' (default, inverted) or 'KE' (not inverted).
        kwargs: dict of plot style arguments (optional).
        normalised_residual (bool): If True, plot 'Normalised Residual'; else plot 'Residual'.
    """

    def _get_x_axis(spectrum_dict, x_axis):
        """
        Return x data, label, and invert flag for spectrum plot.
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
        if key in (x_axis, "KE", "BE"):
            return False
        if key == "Normalised Residual" and not normalised_residual:
            return False
        if key == "Residual" and normalised_residual:
            return False
        return True

    defaults = {"linestyle": "-", "linewidth": 1.5}
    params = _update_plot_params(defaults, kwargs)

    if spectrum_dict is None:
        print("No data to plot.")
        return

    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 6))
        fig.set_tight_layout(True)

    x, xlabel, invert = _get_x_axis(spectrum_dict, x_axis)

    for key, values in spectrum_dict.items():
        if not _should_plot(key):
            continue
        target_ax = axis[0] if key in ("Normalised Residual", "Residual") else axis[1]
        if x is not None and len(x) == len(values):
            target_ax.plot(x, values, label=key, **params)
        else:
            target_ax.plot(np.arange(len(values)), values, label=key, **params)
    axis[1].set_xlabel(xlabel)
    axis[1].set_ylabel("Intensity (a.u.)")
    axis[1].legend()
    if invert:
        axis[1].invert_xaxis()
        axis[0].invert_xaxis()
    axis[0].set_ylabel("Residual")
    axis[0].legend()
    axis[0].spines["bottom"].set_position(("data", 0))
    axis[0].set_xticks([])
    axis[0].set_xticklabels([])
    axis[1].xaxis.set_tick_params(labelbottom=True, bottom=True)
    axis[1].spines["top"].set_visible(False)
    plt.show()
