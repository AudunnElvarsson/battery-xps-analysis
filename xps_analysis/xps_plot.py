"""
Module that plots the xps data.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_positions(report_dict):
    """
	Plot the positions for all entries in 'Name' from the report_dict dictionary.

	Parameters:
		report_dict (dict): Dictionary returned from read_report containing XPS table data.
	"""
    if report_dict is None:
        print("No data to plot.")
        return

    names = report_dict.get("Name")
    positions = report_dict.get("Position")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(positions.shape[0]):
        y = positions[i]
        x = np.arange(positions.shape[1])
        label = names[i][0] if isinstance(names[i], (list, np.ndarray)) else names[i]
        ax.plot(x, y, marker='o', label=label)
    ax.set_xlabel("Experimental Variable")
    ax.set_ylabel("Binding Energy (eV)")
    ax.set_title("Component Peak Positions")
    ax.legend()
    fig.set_tight_layout(True)
    plt.show()
