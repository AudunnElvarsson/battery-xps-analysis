"""
This is to test the xps_analysis package.
"""

import matplotlib.pyplot as plt
import xps_analysis.xps_utilities as xu
import xps_analysis.xps_processing as xp
import xps_analysis.xps_plot as xplot


def convert_all_vms(project_folder):
    """
    Convert all VM files in the specified project folder.
    """
    xu.convert_all_vms_in_project(project_folder)


def fit_report_parameters(report_dict, fit_param):
    """
    Plot the composition report from the specified file path.
    """
    custom_kwargs = {"linestyle": "-"}
    fig, axes = plt.subplots(figsize=(8, 6))
    fig.set_tight_layout(True)
    xplot.plot_fit_report(report_dict, axis=axes, fit_param=fit_param, kwargs=custom_kwargs)


def main():
    """
    Main function to control what runs.
    """
    # Flags to control execution
    run_convert_all_vms_in_project = False
    run_plot_fit_report = True

    project_folder = r"c:/Users/audun/OneDrive - Chalmers/Documents/Research/" \
                     r"1_Improving_XPS_Analysis_Methods/"
    experiment_folder = r"1_data/250619_Gr_XPS_delithiated_and_lithiated/3_output/"
    file_path = project_folder + experiment_folder + r"line_scan_delithiated_comp_report.txt"

    if run_convert_all_vms_in_project:
        convert_all_vms(project_folder)

    if run_plot_fit_report:
        report_dict = xp.read_report(file_path)
        fit_report_parameters(report_dict, fit_param="BE")


if __name__ == "__main__":
    main()
