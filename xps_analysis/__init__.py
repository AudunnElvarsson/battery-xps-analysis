"""This is the xps_analysis package."""

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
print(f"Welcome to version {VERSION} of the xps_analysis package!")
