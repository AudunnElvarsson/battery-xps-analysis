"""
Battery XPS Analysis Package

This package provides tools for X-ray Photoelectron Spectroscopy (XPS) analysis
of battery materials.
"""

# Import the xps_analysis modules to make them available at package level
from .xps_analysis import xps_utilities, xps_processing, xps_plot
from .xps_analysis import VERSION

# Re-export for backward compatibility
__version__ = VERSION

# Make the modules available both ways:
# 1. import battery_xps_analysis; battery_xps_analysis.xps_utilities
# 2. from battery_xps_analysis import xps_utilities
__all__ = ["xps_utilities", "xps_processing", "xps_plot"]
