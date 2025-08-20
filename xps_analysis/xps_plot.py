"""
Module that plots the xps data.
"""

import matplotlib.pyplot as plt
import numpy as np


def _update_plot_params(defaults, user_kwargs):
    """
    Merge default plot parameters with user-supplied overrides.

    Parameters:
        defaults (dict): Dictionary of default plot parameters.
        user_kwargs (dict): User-supplied parameters, similar structure to defaults.

    Returns:
        dict: Merged dictionary of updated plot parameters.
    """
    params = defaults.copy()
    if user_kwargs:
        for key, value in user_kwargs.items():
            params[key] = value
    return params


def plot_fit_report(report_dict, axis=None, fit_param="BE", kwargs=None):
    """
    Plot the selected fit_param for all entries in 'Name' from the report_dict dictionary.

    Parameters:
        report_dict (dict): Dictionary returned from read_report containing XPS table data.
        fit_param (str): The fit parameter to plot on the y-axis. Options: "Position", "FWHM", "Raw Area", "Area/(RSF*T*MFP)", "%At Conc", "Goodness of Fit"
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
        "Goodness": "Goodness of Fit"
    }
    col_full = col_map.get(fit_param, fit_param)
    names = report_dict.get("Name")
    y_data = report_dict.get(col_full)

    names = np.array(names, dtype=object)
    y_data = np.array(y_data, dtype=object)
    for i in range(y_data.shape[0]):
        y = y_data[i]
        x = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)
        label = names[i][0] if isinstance(names[i], (list, np.ndarray)) else names[i]
        axis.plot(x, y, **params, label=label)
    axis.set_xlabel("Experimental Variable")
    axis.set_ylabel(col_full)
    axis.legend()
    plt.show()
