"""Plot helpers package for XPS plotting utilities.

This package provides modular helper functions for XPS data visualization,
organized into specialized submodules for different aspects of plotting.

Submodules
----------
file_operations : Figure saving and filename generation utilities
axis_management : Axis setup, configuration, and figure preparation
plot_rendering : Core plotting logic for spectra and fit reports
report_plotting : Specialized functions for report plotting

Public API
----------
The most commonly used functions are exported at the package level for
convenient access. See individual module documentation for complete details.

File Operations
~~~~~~~~~~~~~~~
save_figure : Save a figure with auto-generated filename
derive_file_name : Extract a filename from a data mapping

Axis Management
~~~~~~~~~~~~~~~
get_x_axis_from_dict : Get x-axis data and metadata from spectrum
figure_from_axes : Extract Figure from axes-like input
select_target_axis : Select appropriate axis for plotting
ensure_axes_and_main : Prepare figure and axes for plotting
configure_axes : Apply axis labels, limits, and inversion

Plot Rendering
~~~~~~~~~~~~~~
convert_to_relative_be : Convert BE values to relative values
get_column_name : Map short parameter names to full headers
update_plot_params : Merge default and user plot parameters
should_plot_key : Decide if a spectrum key should be plotted
plot_report_series : Plot fit report data series
plot_spectrum_series : Plot spectrum data series

Report Plotting
~~~~~~~~~~~~~~~
configure_report_axes : Configure axes for report plots
plot_single_core_level : Plot single core level fit report
plot_all_core_levels : Plot multiple core levels in subplots
"""

# Make key plotting functions available at package level for easy importing
from .file_operations import save_figure, derive_file_name
from .axis_management import (
    get_x_axis_from_dict,
    figure_from_axes,
    select_target_axis,
    ensure_axes_and_main,
    configure_axes,
)
from .plot_rendering import (
    convert_to_relative_be,
    get_column_name,
    update_plot_params,
    should_plot_key,
    plot_report_series,
    plot_spectrum_series,
)
from .report_plotting import (
    configure_report_axes,
    plot_single_core_level,
    plot_all_core_levels,
)

__all__ = [
    # File operations
    "save_figure",
    "derive_file_name",
    # Axis management
    "get_x_axis_from_dict",
    "figure_from_axes",
    "select_target_axis",
    "ensure_axes_and_main",
    "configure_axes",
    # Plot rendering
    "convert_to_relative_be",
    "get_column_name",
    "update_plot_params",
    "should_plot_key",
    "plot_report_series",
    "plot_spectrum_series",
    # Report plotting
    "configure_report_axes",
    "plot_single_core_level",
    "plot_all_core_levels",
]
