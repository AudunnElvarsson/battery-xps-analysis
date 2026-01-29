# Battery XPS Analysis

A Python package for processing and visualizing X-ray Photoelectron Spectroscopy (XPS) data from battery materials research.

## Overview

`battery-xps-analysis` provides a comprehensive toolkit for analyzing XPS data exported from CasaXPS software. The package handles data parsing, statistical analysis, and publication-quality visualization of fit reports and spectra, with specialized features for battery material characterization.

### Key Features

- **Automatic format detection**: Handles both single and multi-core level data formats
- **Robust parsing**: Processes tab-separated text files with complex headers and nested data
- **Statistical analysis**: Calculate averages, differences, and ratios across measurements
- **Flexible visualization**: Create customizable plots with matplotlib integration
- **Spin-orbit doublets**: Automatic detection and handling of doublet peaks
- **Component tracking**: Track binding energy shifts and compositional changes over measurement series
- **Batch processing**: Convert and analyze multiple data files efficiently

## Installation

### Prerequisites

- Python >= 3.13.1
- pip or uv package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/AudunnElvarsson/battery-xps-analysis.git
cd battery-xps-analysis

# Create and activate virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

### Using uv (recommended)

```bash
# Install uv if not already installed
pip install uv

# Create environment and install dependencies
uv sync
```

## Quick Start

### 1. Import the package

```python
import xps_analysis.xps_processing as xp
import xps_analysis.xps_plot as xplot
import xps_analysis.xps_utilities as xu
```

### 2. Read and analyze fit reports

```python
# Read a fit report file (auto-detects format)
report = xp.read_report_file("fit_report.txt")

# Check available parameters
params = xp.get_available_parameters(report)
print("Available parameters:", params['numeric'])

# Print formatted table with default parameters
xp.print_report(report)

# Print custom parameters
custom_params = ["Label", "BE", "FWHM", "At Conc"]
xp.print_report(report, parameters=custom_params)
```

### 3. Visualize component data

```python
# Plot binding energy evolution across measurements
xplot.plot_comp_report(report, proc_kwargs={"fit_param": "BE"})

# Plot atomic concentrations with averages
proc_kwargs = {
    "fit_param": "At Conc",
    "calculate": "average",
    "core_levels": ["C 1s", "F 1s", "O 1s"]
}
xplot.plot_comp_report(report, proc_kwargs=proc_kwargs)

# Plot relative binding energy with reference component
proc_kwargs = {
    "fit_param": "BE",
    "reference": "A",  # Reference component label
    "show_labels": True
}
xplot.plot_comp_report(report, proc_kwargs=proc_kwargs)
```

### 4. Plot atomic ratios

```python
# Plot F/C atomic ratio
proc_kwargs = {
    "plot_type": "ratio",
    "numerator": "F 1s",
    "denominator": "C 1s",
    "parameter": "At Conc",
    "calculate": "average"
}
xplot.plot_region_report(report, proc_kwargs=proc_kwargs)

# Plot total atomic concentrations
proc_kwargs = {
    "plot_type": "total",
    "core_levels": ["F 1s", "C 1s", "O 1s"],
    "parameter": "At Conc"
}
xplot.plot_region_report(report, proc_kwargs=proc_kwargs)
```

### 5. Visualize spectra

```python
# Read and plot spectrum data
spectrum = xp.read_spectrum_file("C1s_spectrum.txt")
xplot.plot_spectrum(spectrum)

# Plot with kinetic energy axis and normalized residuals
proc_kwargs = {
    "x_axis": "KE",
    "normalised_residual": True
}
xplot.plot_spectrum(spectrum, proc_kwargs=proc_kwargs)
```

## Package Structure

```t
battery-xps-analysis/
├── xps_analysis/              # Main package
│   ├── xps_processing.py      # Data parsing and processing
│   ├── xps_plot.py            # High-level plotting functions
│   ├── xps_utilities.py       # Utility functions (file conversion, etc.)
│   ├── plot_helpers/          # Plotting utility modules
│   │   ├── file_operations.py    # Figure saving and filename generation
│   │   ├── axis_management.py    # Axis setup and configuration
│   │   ├── plot_rendering.py     # Core plotting logic
│   │   └── report_plotting.py    # Report-specific plotting
│   └── processing_helpers/    # Data processing utilities
│       ├── file_parser.py           # File reading and parsing
│       ├── table_processor.py       # Table parsing and doublets
│       ├── data_analyzer.py         # Parameter extraction
│       ├── data_transformer.py      # Normalization and transformations
│       ├── formatter.py             # Output formatting
│       └── core_level_extractor.py  # Core level name extraction
├── guides/                    # Documentation
│   └── FEATURE_SUMMARY.md    # Feature documentation
├── main.py                    # Example usage script
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Main Modules

### `xps_processing`

Data parsing and processing functions:

- `read_report_file()`: Parse XPS fit report files
- `read_spectrum_file()`: Parse spectrum data files
- `get_available_parameters()`: List available parameters in data
- `print_report()`: Print formatted tables with statistics
- `get_core_levels()`: Extract core level names

### `xps_plot`

Visualization functions:

- `plot_comp_report()`: Plot component parameters across measurements
- `plot_region_report()`: Plot ratios or totals between core levels
- `plot_spectrum()`: Plot spectra with optional residuals

### `xps_utilities`

Utility functions:

- `convert_all_vms_in_project()`: Batch convert VMS files to text
- File management and path operations

## Advanced Usage

### Multi-core level analysis

```python
# Read multi-core report
report = xp.read_report_file("multicore_report.txt")

# Get list of core levels
cores = xp.get_core_levels(report)
print("Core levels:", cores)  # ['C 1s', 'F 1s', 'O 1s', ...]

# Plot specific core levels
proc_kwargs = {
    "fit_param": "BE",
    "core_levels": ["C 1s", "F 1s"],
    "calculate": "average"
}
xplot.plot_comp_report(report, proc_kwargs=proc_kwargs)
```

### Normalize atomic concentrations per core level

```python
# Useful when comparing components within each core level
proc_kwargs = {
    "fit_param": "At Conc",
    "core_levels": ["C 1s", "O 1s", "F 1s"],
    "normalize_at_conc_per_core": True
}
xplot.plot_comp_report(report, proc_kwargs=proc_kwargs)
```

### Area ratio calculations

```python
# Calculate area ratios relative to a reference component
proc_kwargs = {
    "fit_param": "Area",
    "calculate": "ratio",
    "reference": "A",  # Reference component
    "core_levels": ["C 1s"]
}
xplot.plot_comp_report(report, proc_kwargs=proc_kwargs)
```

### Customize plot appearance

```python
# Custom styling
plot_kwargs = {
    "linestyle": "-",
    "linewidth": 2,
    "marker": "o",
    "markersize": 6,
    "alpha": 0.8
}

# Save figure
save_kwargs = {
    "save_fig": True,
    "save_folder": "./figures",
    "format": "pdf",
    "dpi": 300
}

xplot.plot_comp_report(
    report,
    proc_kwargs={"fit_param": "BE"},
    plot_kwargs=plot_kwargs,
    save_kwargs=save_kwargs
)
```

## Parameter Names

Common parameter shortcuts (automatically mapped to full names):

- `"BE"` → `"Binding Energy (eV)"`
- `"Area"` → `"Raw Area"`
- `"At Conc"` → `"%At Conc"`
- `"Goodness"` → `"Goodness of Fit"`

Full parameter names are also accepted and can be found using `get_available_parameters()`.

## File Format Support

### Fit Report Files

The package handles two report formats:

1. **Single core level**: Traditional format with one core level per file
2. **Multi-core level**: Format with "Data Set" or "Iteration" column containing multiple core levels

Both formats are automatically detected and parsed correctly.

### Spectrum Files

Tab-separated text files with columns:

- Binding Energy (BE) or Kinetic Energy (KE)
- Measured intensity
- Background
- Fitted components
- Envelope (total fit)
- Residuals (optional)

## Example Workflow

See `main.py` for a complete example workflow including:

- Reading multi-core reports
- Plotting component parameters
- Calculating atomic ratios
- Customizing visualizations
- Batch processing

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

## Contact

**Author**: Audunn Elvarsson
**Email**: <audunn.elvarsson@gmail.com>

## Acknowledgments

This package was developed for battery materials research using XPS data from CasaXPS software.
