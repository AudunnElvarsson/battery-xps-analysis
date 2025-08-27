"""xps_utilities
----------------

Small utilities for converting experiment-exported data files for XPS
workflows. The current helpers focus on converting vendor-specific ``.vms``
files into plain text ``.txt`` files and for batch-converting sessions in a
project folder.

The functions are intentionally lightweight and designed to be replaced or
extended with project-specific conversion logic when needed.
"""

import os


def convert_vms_to_txt(input_path, output_path):
    """Convert a single ``.vms`` file to plain-text ``.txt``.

    This function currently performs a simple read/write copy. Replace the
    body with vendor-specific parsing or sanity checks if a true conversion is
    required.

    Parameters
    ----------
    input_path : str
        Path to the source ``.vms`` file.
    output_path : str
        Destination path for the produced ``.txt`` file.

    Returns
    -------
    None
    """
    # Example placeholder conversion logic: copy file contents
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        data = infile.read()
        outfile.write(data)


def convert_all_vms_in_project(project_folder):
    """Recursively convert all ``.vms`` files in a project's sessions.

    The function looks for a ``1_data`` folder beneath ``project_folder`` and
    iterates its subfolders (sessions). For each session it expects a
    ``1_processed_data/vms_files`` directory containing ``.vms`` files. A new
    sibling directory ``txt_files`` will be created and populated with the
    converted files. If ``txt_files`` already exists the session is skipped to
    avoid accidental re-processing.

    Parameters
    ----------
    project_folder : str
        Path to the root of the project directory containing the ``1_data``
        folder.

    Returns
    -------
    None
    """
    data_folder = os.path.join(project_folder, "1_data")
    if not os.path.isdir(data_folder):
        print(f"Data folder not found: {data_folder}")
        return
    for session in os.listdir(data_folder):
        session_path = os.path.join(data_folder, session)
        if not os.path.isdir(session_path):
            continue
        processed_folder = os.path.join(session_path, "1_processed_data")
        if not os.path.isdir(processed_folder):
            print(f"No '1_processed_data' folder in session '{session}'")
            continue
        vms_folder = os.path.join(processed_folder, "vms_files")
        txt_folder = os.path.join(processed_folder, "txt_files")
        if not os.path.isdir(vms_folder):
            print(f"No 'vms_files' folder in session '{session}'")
            continue
        if os.path.exists(txt_folder):
            print(
                f"Skipping conversion, txt folder already exists in session '{session}'"
            )
            continue
        os.makedirs(txt_folder, exist_ok=True)
        print(f"Converting vms files to txt files in session '{session}'")
        for fname in os.listdir(vms_folder):
            if fname.lower().endswith(".vms"):
                vms_path = os.path.join(vms_folder, fname)
                txt_name = os.path.splitext(fname)[0] + ".txt"
                txt_path = os.path.join(txt_folder, txt_name)
                convert_vms_to_txt(vms_path, txt_path)
