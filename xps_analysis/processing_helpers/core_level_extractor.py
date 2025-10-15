"""Core level extraction utilities for XPS analysis.

This module provides functionality to extract core level information from
XPS filenames following the convention where the core level is the last
underscore-separated part of the filename (before the file extension).
"""

import os
import re


def extract_core_level(file_path):
    """Extract core level from XPS filename.

    The function extracts the core level from the filename by taking the last
    underscore-separated part before the file extension. Common XPS core levels
    include C1s, O1s, F1s, Li1s, P2p, etc. The returned core level has a space
    between the element and orbital (e.g., 'C 1s', 'O 1s').

    Parameters
    ----------
    file_path : str
        Path to the XPS file.

    Returns
    -------
    str or None
        The extracted core level with space (e.g., 'C 1s', 'O 1s', 'F 1s') or
        None if no valid core level pattern is found.

    Examples
    --------
    >>> extract_core_level("experiment_fit_report_C1s.txt")
    'C 1s'
    >>> extract_core_level("sample_spectrum_fit_O1s.txt")
    'O 1s'
    >>> extract_core_level("data_F1s.vms")
    'F 1s'
    >>> extract_core_level("no_core_level.txt")
    None
    """
    if not file_path:
        return None

    # Get the filename without path and extension
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
    except (OSError, ValueError):
        return None

    # Split by underscores and get the last part
    parts = base_name.split("_")
    if not parts:
        return None

    candidate = parts[-1]

    # Check if the candidate looks like a core level
    # Core levels typically follow patterns like: C1s, O1s, F1s, Li1s, P2p, etc.
    core_level_pattern = re.compile(r"^([A-Z][a-z]?)([0-9]+[spdfg])$", re.IGNORECASE)

    match = core_level_pattern.match(candidate)
    if match:
        # Extract element and orbital, add space between them
        element = match.group(1)
        orbital = match.group(2)
        return f"{element} {orbital}"

    return None
