"""Processing helpers package initialization."""

# Make key functions available at package level for easy importing
from .file_parser import (
    find_header,
    parse_data_rows,
    convert_data_types,
    add_file_metadata,
)
from .table_processor import (
    clean_header,
    group_rows_by_name,
    group_rows_by_dataset,
    has_dataset_column,
    table_to_dict,
    table_to_dict_exclude_columns,
    parse_report_rows,
)
from .data_analyzer import (
    process_parameter,
    extract_component_names,
    process_all_parameters,
)
from .formatter import (
    format_table_value,
    calculate_column_widths,
    build_table_header,
    build_table_row,
)
from .core_level_extractor import extract_core_level

__all__ = [
    # File parsing
    "find_header",
    "parse_data_rows",
    "convert_data_types",
    "add_file_metadata",
    # Table processing
    "clean_header",
    "group_rows_by_name",
    "group_rows_by_dataset",
    "has_dataset_column",
    "table_to_dict",
    "table_to_dict_exclude_columns",
    "parse_report_rows",
    # Data analysis
    "process_parameter",
    "extract_component_names",
    "process_all_parameters",
    # Formatting
    "format_table_value",
    "calculate_column_widths",
    "build_table_header",
    "build_table_row",
    # Core level extraction
    "extract_core_level",
]
