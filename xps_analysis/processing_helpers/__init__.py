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
    table_to_dict,
    parse_report_rows,
)
from .data_analyzer import process_parameter
from .formatter import format_table_value
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
    "table_to_dict",
    "parse_report_rows",
    # Data analysis
    "process_parameter",
    # Formatting
    "format_table_value",
    # Core level extraction
    "extract_core_level",
]
