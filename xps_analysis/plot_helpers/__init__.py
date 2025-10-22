"""Plot helpers package initialization."""

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
    get_column_name,
    update_plot_params,
    should_plot_key,
    plot_report_series,
    plot_spectrum_series,
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
    "get_column_name",
    "update_plot_params",
    "should_plot_key",
    "plot_report_series",
    "plot_spectrum_series",
]
