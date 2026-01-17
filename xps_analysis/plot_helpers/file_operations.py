"""File operations utilities for plotting.

This module handles figure saving, filename generation, and file management
for XPS plotting operations.

Public Functions
----------------
save_figure : Save a figure with auto-generated filename
derive_file_name : Extract a filename from a data mapping
"""

import os
import re
import numpy as np


def save_figure(fig, name_list=None, save_args=None, prefix="figure"):
    """Save a Matplotlib figure using a generated safe filename.

    This helper constructs a safe filename from ``name_list`` and calls
    :meth:`matplotlib.figure.Figure.savefig` with the computed keyword
    arguments.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to save.
    name_list : list of str or None
        Parts used to build the filename. If ``None`` or empty, a generic
        basename is used.
    save_args : dict or None
        Options forwarded to ``Figure.savefig``. The special keys
        ``save_folder`` or ``exp_folder`` (if present) select the target folder.
    prefix : str
        Optional filename prefix; if empty the prefix is omitted.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    # Accept both 'save_folder' and 'exp_folder' for backwards compatibility
    save_folder = None
    if save_args:
        save_folder = save_args.get("save_folder") or save_args.get("exp_folder")
    if not save_folder:
        save_folder = os.getcwd()
    save_file, save_kwargs = _build_save_info(
        save_folder, name_list or [], save_args, prefix=prefix
    )
    fig.savefig(save_file, **save_kwargs)
    print(f"Saved figure: {save_file}")
    return save_file


def derive_file_name(mapping):
    """Derive a sensible filename base from a mapping.

    The function looks for keys in the order ``'File Name'``, ``'Sample'``,
    ``'Name'`` and falls back to ``'spectrum'``. If the value is a sequence,
    the first element is used.

    Parameters
    ----------
    mapping : dict
        Mapping that may contain identifying metadata for the spectrum.

    Returns
    -------
    str
        A short, filesystem-friendly name suitable for use in filenames.
    """
    file_name = (
        mapping.get("File Name")
        or mapping.get("Sample")
        or mapping.get("Name")
        or "spectrum"
    )
    if isinstance(file_name, (list, np.ndarray)):
        file_name = file_name[0] if len(file_name) else "spectrum"
    return file_name


def _build_save_info(save_folder, name_list, save_args=None, prefix="figure"):
    """Build a filesystem-safe filename and default save kwargs for figures.

    Parameters
    ----------
    save_folder : str
        Destination folder for the saved figure. The folder is created if it
        does not exist.
    name_list : list of str
        Sequence of parts to include in the filename (joined with
        underscores); empty or None entries are ignored.
    save_args : dict or None
        Optional overrides forwarded to :meth:`matplotlib.figure.Figure.savefig`.
        The special key ``save_folder`` is ignored here (it is handled by
        the caller).
    prefix : str
        Optional filename prefix. If empty, no prefix is used.

    Returns
    -------
    tuple
        ``(save_file, save_kwargs)`` where ``save_file`` is the absolute path
        to the file and ``save_kwargs`` is a dict of keyword arguments
        suitable for passing to ``Figure.savefig``.
    """
    os.makedirs(save_folder, exist_ok=True)
    save_kwargs = {"dpi": 300, "format": "png", "bbox_inches": "tight"}
    if save_args:
        # allow overriding format, dpi, bbox_inches etc.
        # Filter out custom parameters that aren't matplotlib savefig parameters
        custom_params = ("save_folder", "save_fig", "exp_folder")
        save_kwargs.update(
            {k: v for k, v in save_args.items() if k not in custom_params}
        )
    # build a safe name from provided parts
    safe_parts = [
        re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_")
        for p in (name_list or [])
        if p is not None and str(p).strip() != ""
    ]
    names_str = "_".join(safe_parts) or "all"
    if prefix:
        filename = f"{prefix}_{names_str}.{save_kwargs['format']}"
    else:
        filename = f"{names_str}.{save_kwargs['format']}"
    save_file = os.path.join(save_folder, filename)
    return save_file, save_kwargs
