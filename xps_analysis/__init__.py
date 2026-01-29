"""XPS Analysis Package for Battery Research.

Tools for processing and analyzing X-ray Photoelectron Spectroscopy (XPS) data
from CasaXPS software, with a focus on battery material characterization.

Main Modules
------------
xps_processing : Data processing and file parsing
    - read_report_file() : Parse XPS fit report files
    - read_spectrum_file() : Parse XPS spectrum data files
    - get_available_parameters() : List available parameters in data
    - print_report() : Print formatted tables with statistics
    - get_core_levels() : Extract core level names from data

xps_plot : Visualization and plotting
    - plot_comp_report() : Plot component parameters across measurements
    - plot_region_report() : Plot ratios or totals between core levels
    - plot_spectrum() : Plot XPS spectra with optional residuals

xps_utilities : File conversion utilities
    - convert_vms_to_txt() : Convert single .vms file to text
    - convert_all_vms_in_project() : Batch convert all .vms files

Helper Packages
---------------
processing_helpers : Modular data processing utilities
    - file_parser : File reading and header detection
    - table_processor : Table parsing and doublet handling
    - data_analyzer : Parameter extraction and analysis
    - data_transformer : Normalization, ratios, and transformations
    - formatter : Output formatting and table printing
    - core_level_extractor : Core level name extraction

plot_helpers : Modular plotting utilities
    - file_operations : Figure saving and filename generation
    - axis_management : Axis setup and configuration
    - plot_rendering : Core plotting logic
    - report_plotting : Report-specific plotting functions

Quick Start
-----------
>>> import xps_analysis.xps_processing as xp
>>> import xps_analysis.xps_plot as xplot
>>>
>>> # Read and display report
>>> report = xp.read_report_file("fit_report.txt")
>>> xp.print_report(report)
>>>
>>> # Plot binding energies
>>> xplot.plot_comp_report(report, proc_kwargs={"fit_param": "BE"})

Version
-------
Current version is accessible via the VERSION constant.
"""

import os
import sys
import tomllib
from importlib.metadata import version
from . import xps_processing
from . import xps_utilities
from . import xps_plot


def _get_version():
    """Get version from pyproject.toml file, with fallback to installed package."""
    try:
        # First try to read from pyproject.toml (source of truth)
        current_dir = os.path.dirname(__file__)
        pyproject_path = os.path.join(current_dir, "..", "pyproject.toml")

        # Try modern tomllib first (Python 3.11+)
        if sys.version_info >= (3, 11):
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            return data["project"]["version"]
    except (FileNotFoundError, IOError, KeyError, ImportError):
        # Fallback to installed package metadata
        try:
            return version("battery-xps-analysis")
        except ImportError:
            pass

    return "unknown"


VERSION = _get_version()
