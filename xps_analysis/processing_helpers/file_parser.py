"""
XPS file parsing utilities.

This module contains low-level functions for parsing XPS data files,
including header detection, data row parsing, and type conversion.
"""

import os
import numpy as np
from .core_level_extractor import extract_core_level


def find_header(lines, startswith_tuple, required_substring=None):
    """Find and return the first header line that matches criteria.

    This scans ``lines`` and returns the first line that begins with any of the
    strings in ``startswith_tuple``. If ``required_substring`` is provided,
    the line must also contain that substring.

    Parameters
    ----------
    lines : list[str]
        File content split into lines.
    startswith_tuple : tuple[str]
        Tuple of possible prefixes for the header line (e.g. ("Name",)).
    required_substring : str | None
        Optional substring that must be present in the header line.

    Returns
    -------
    tuple[str | None, int | None]
        (header_line, index) if found, otherwise (None, None).
    """
    for i, line in enumerate(lines):
        if line.strip().startswith(startswith_tuple):
            if required_substring is None or required_substring in line:
                return line, i
    return None, None


def parse_data_rows(lines, header_idx, columns, skip_empty=True, break_on_empty=False):
    """Parse tab-separated rows following a header into a dict of column lists.

    Each value is converted to ``float`` when possible; non-convertible values
    are left as strings. Lines that do not match the expected column count are
    ignored.

    Parameters
    ----------
    lines : list[str]
        File content lines.
    header_idx : int
        Index of the header line in ``lines``.
    columns : list[str]
        Expected column names corresponding to a row's tab-separated fields.
    skip_empty : bool
        If True, skip empty lines between data rows; if False, include them
        (they will be ignored if column lengths don't match).
    break_on_empty : bool
        If True, stop parsing when the first empty line after the header is
        encountered.

    Returns
    -------
    dict[str, list]
        Mapping from column name to list of parsed values (floats or strings).
    """
    data_dict = {col: [] for col in columns}
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if break_on_empty:
                break
            if skip_empty:
                continue
        values = [v.strip() for v in line.split("\t")]
        if len(values) != len(columns):
            continue
        for col, val in zip(columns, values):
            try:
                data_dict[col].append(float(val))
            except ValueError:
                data_dict[col].append(val)
    return data_dict


def convert_data_types(data_dict):
    """Convert each list in ``data_dict`` to a NumPy array.

    Attempts to create a float array for each column and falls back to an
    object array when conversion fails. The input dict is mutated and
    returned for convenience.

    Parameters
    ----------
    data_dict : dict[str, list]
        Mapping of column names to lists of values.

    Returns
    -------
    dict[str, numpy.ndarray]
        The same mapping where each value is a NumPy array.
    """
    for col in data_dict:
        try:
            data_dict[col] = np.array(data_dict[col], dtype=float)
        except ValueError:
            data_dict[col] = np.array(data_dict[col], dtype=object)
    return data_dict


def add_file_metadata(data_dict, file_path):
    """Add file name and core level metadata to a data dictionary.

    Extracts the base filename (without path or extension) and the core level
    from the file path, adding them as "File Name" and "Core Level" keys to
    the provided dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary to which metadata will be added.
    file_path : str
        Path to the file being processed.

    Returns
    -------
    dict
        The same dictionary with added "File Name" and "Core Level" keys.
    """
    try:
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        data_dict["File Name"] = file_base
    except (OSError, ValueError):
        data_dict["File Name"] = None

    data_dict["Core Level"] = extract_core_level(file_path)

    return data_dict
