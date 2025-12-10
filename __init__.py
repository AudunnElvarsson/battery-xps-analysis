"""
Battery XPS Analysis Package

This package provides tools for X-ray Photoelectron Spectroscopy (XPS) analysis
of battery materials.
"""

# Import the xps_analysis modules to make them available at package level
from .xps_analysis import xps_utilities, xps_processing, xps_plot
from .xps_analysis import VERSION

__version__ = VERSION
__all__ = ["xps_utilities", "xps_processing", "xps_plot"]
