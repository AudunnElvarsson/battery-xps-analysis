"""
This is to test the xps_analysis package.
"""

import xps_analysis.xps_utilities as xu


def main():
    """
    Main function to control what runs.
    """
    project_folder = r"c:/Users/audun/OneDrive - Chalmers/Documents/Research/1. Improving XPS Analysis Methods"
    xu.convert_all_vms_in_project(project_folder)


if __name__ == "__main__":
    main()
