"""Processing helpers package for XPS data parsing and formatting.

This package provides modular helper functions for XPS data processing,
organized into specialized submodules for different aspects of data handling.

Submodules
----------
file_parser : File reading and header detection utilities
table_processor : Table parsing and row grouping functions
data_analyzer : Parameter extraction and data analysis
formatter : Output formatting and table printing
core_level_extractor : Core level name extraction from file metadata

Public API
----------
The most commonly used functions are exported at the package level for
convenient access. See individual module documentation for complete details.

File Parsing
~~~~~~~~~~~~
find_header : Locate table headers in file lines
parse_data_rows : Parse data rows into dictionary arrays
convert_data_types : Convert string data to appropriate numeric types
add_file_metadata : Add file name and core level metadata to dictionary

Table Processing
~~~~~~~~~~~~~~~~
clean_header : Normalize table header and remove duplicates
group_rows_by_name : Group table rows by repeating name column
group_rows_by_dataset : Group table rows by dataset (multi-core format)
has_dataset_column : Check if header indicates multi-core format
table_to_dict : Convert grouped rows to dictionary format
table_to_dict_exclude_columns : Convert rows excluding specific columns
parse_report_rows : Parse report rows handling multi-core format

Data Analysis
~~~~~~~~~~~~~
process_parameter : Process single parameter from fit data
extract_component_names : Extract component names from fit data
process_all_parameters : Process all parameters into organized structure

Formatting
~~~~~~~~~~
format_table_value : Format value for table display with constraints
calculate_column_widths : Calculate optimal column widths for tables
build_table_header : Build formatted table header and separator
build_table_row : Build formatted table row
print_single_core_level : Print formatted table for single core level

Core Level Extraction
~~~~~~~~~~~~~~~~~~~~~
extract_core_level : Extract core level name from file name or path
"""

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
    print_single_core_level,
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
    "print_single_core_level",
    # Core level extraction
    "extract_core_level",
]
