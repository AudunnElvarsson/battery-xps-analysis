"""
This module provides utilities for converting XPS data files,
specifically for converting .vms files to .txt format.

Functions:
    convert_vms_to_txt(input_path, output_path):
        Converts a single .vms file to a .txt file.
    convert_all_vms_in_project(project_folder):
        Converts all .vms files in all measurement sessions in the Data folder of a project.
"""

import os


def convert_vms_to_txt(input_path, output_path):
    """
    Convert a single .vms file to .txt format.

    Parameters:
        input_path (str): Path to the input .vms file.
        output_path (str): Path to the output .txt file.
    """
    # Example placeholder conversion logic
    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        data = infile.read()
        outfile.write(data)


def convert_all_vms_in_project(project_folder):
    """
    Convert all .vms files in all measurement sessions in the Data folder of a project. For each
    session in the 'Data' subfolder of the project, this function looks for a folder named 'VMS
    files'. It converts all .vms files in that folder to .txt files, creates a new folder called
    'TXT files' in the same session folder, and writes the converted .txt files there.

    Parameters:
        project_folder (str): Path to the root of the project folder.
    """
    data_folder = os.path.join(project_folder, "Data")
    if not os.path.isdir(data_folder):
        print(f"Data folder not found: {data_folder}")
        return
    for session in os.listdir(data_folder):
        session_path = os.path.join(data_folder, session)
        if not os.path.isdir(session_path):
            continue
        vms_folder = os.path.join(session_path, "VMS files")
        txt_folder = os.path.join(session_path, "TXT files")
        if not os.path.isdir(vms_folder):
            print(f"No 'VMS files' folder in session '{session}'")
            continue
        if os.path.exists(txt_folder):
            print(f"TXT files folder already exists in session '{session}'. Skipping conversion.")
            continue
        os.makedirs(txt_folder, exist_ok=True)
        print(f"Converting VMS files in session '{session}' to TXT files.")
        for fname in os.listdir(vms_folder):
            if fname.lower().endswith(".vms"):
                vms_path = os.path.join(vms_folder, fname)
                txt_name = os.path.splitext(fname)[0] + ".txt"
                txt_path = os.path.join(txt_folder, txt_name)
                convert_vms_to_txt(vms_path, txt_path)
