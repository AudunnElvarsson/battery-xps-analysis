"""
Module that plots the xps data.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_fit_report(report_dict, y_column="BE"):
    """
    Plot the selected y_column for all entries in 'Name' from the report_dict dictionary.

    Parameters:
        report_dict (dict): Dictionary returned from read_report containing XPS table data.
        y_column (str): The column to plot on the y-axis. Options: "Position", "FWHM", "Raw Area", "Area/(RSF*T*MFP)", "%At Conc", "Goodness of Fit"
    """
    if report_dict is None:
        print("No data to plot.")
        return

    # Map shorthand to full column names
    col_map = {
        "BE": "Binding Energy (eV)",
        "Area": "Raw Area",
        "At Conc": "%At Conc",
        "Goodness": "Goodness of Fit"
    }
    col_full = col_map.get(y_column, y_column)
    names = report_dict.get("Name")
    y_data = report_dict.get(col_full)

    fig, ax = plt.subplots(figsize=(8, 6))
    names = np.array(names, dtype=object)
    y_data = np.array(y_data, dtype=object)
    for i in range(y_data.shape[0]):
        y = y_data[i]
        x = np.arange(y_data.shape[1]) if y_data.ndim > 1 else np.arange(1)
        label = names[i][0] if isinstance(names[i], (list, np.ndarray)) else names[i]
        ax.plot(x, y, marker='o', label=label)
    ax.set_xlabel("Experimental Variable")
    ax.set_ylabel(col_full)
    ax.legend()
    fig.set_tight_layout(True)
    plt.show()
