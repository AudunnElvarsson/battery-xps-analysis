# Multi-Core Level File Format Support - Implementation Summary

## Overview
Successfully implemented support for XPS files containing multiple core levels in a single file, while maintaining backwards compatibility with single core-level files.

## Data Structure

### Old Format (Single Core Level)
```python
{
    "Name": array(...),
    "Binding Energy (eV)": array(...),
    "FWHM": array(...),
    # ... other parameters
    "File Name": "filename",
}
```

### New Format (Multi Core Level)
```python
{
    "F 1s": {
        "Name": array(...),
        "Binding Energy (eV)": array(...),
        "FWHM": array(...),
        # ... other parameters
    },
    "C 1s": {
        "Name": array(...),
        "Binding Energy (eV)": array(...),
        # ... other parameters
    },
    "O 1s": {
        # ... parameters
    },
    "Core Level": "C 1s",  # Default core level
    "File Name": "filename",
}
```

## Key Changes

### 1. New Functions in `table_processor.py`
- `has_dataset_column(header)`: Detects multi-core format
- `group_rows_by_dataset(table_data, dataset_idx, tag_idx)`: Groups rows by dataset and core level
- `table_to_dict_exclude_columns(groups, header, exclude_indices)`: Converts groups to dict while excluding grouping columns

### 2. Updated `parse_report_rows()` in `table_processor.py`
- Preserves empty columns in multi-core format
- Fills empty "Data Set" cells with current dataset number
- Removes trailing empty columns to match header length

### 3. Updated `read_report_file()` in `xps_processing.py`
- Tries both header formats ("Data Set" and "Name")
- Detects multi-core format using `has_dataset_column()`
- Routes to appropriate grouping function
- Returns nested dictionary for multi-core format

## Usage Examples

### Single Core Level (Old Format)
```python
from xps_analysis.xps_processing import read_report_file

data = read_report_file("report_C1s.txt")
be_values = data["Binding Energy (eV)"]
component_names = data["Name"]
```

### Multi Core Level (New Format)
```python
from xps_analysis.xps_processing import read_report_file

data = read_report_file("report_F1s_O1s_C1s.txt")

# Get available core levels
core_levels = [k for k in data.keys() if k not in ['File Name', 'Core Level']]

# Access specific core level
c1s_data = data["C 1s"]
c1s_be = c1s_data["Binding Energy (eV)"]
c1s_names = c1s_data["Name"]

# Process each core level
for core in core_levels:
    core_data = data[core]
    print_report_averages(core_data, reference="A")
```

## File Format Details

### New Format Characteristics
- Has "Data Set" column (0, 1, 2, ...)
- Has "Tag" column with core level names (e.g., "F 1s", "C 1s", "O 1s")
- "Data Set" column is only filled for first row of each dataset
- Multiple core levels can be present in one file

### Detection Logic
1. Try to find header starting with "Data Set"
2. If not found, fall back to "Name" (old format)
3. Check for "Data Set" in header to determine format type

## Backwards Compatibility
- ✅ Single core-level files work exactly as before
- ✅ Existing code (like `main.py`) continues to work
- ✅ `print_report_averages()` works with both formats
- ✅ All plotting functions compatible (when passed single core data)

## Testing
- ✅ Old format: 5_e_beam_unwashed_before_after_report_C1s.txt
- ✅ New format: 5_e_beam_unwashed_before_after_report_F1s_O1s_C1s.txt
- ✅ Both formats tested successfully
- ✅ No regression in existing functionality

## Files Modified
1. `xps_analysis/processing_helpers/table_processor.py`
   - Added 3 new functions
   - Updated `parse_report_rows()`
   - Updated module docstring

2. `xps_analysis/processing_helpers/__init__.py`
   - Exported new functions

3. `xps_analysis/xps_processing.py`
   - Updated `read_report_file()` with auto-detection
   - Updated imports
   - Updated docstring

## Next Steps (Optional)
- Update `print_report_averages()` to handle multi-core format directly
- Add plotting functions for multi-core comparison
- Update documentation with new format examples
