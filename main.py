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


def fit_report_parameters(report_dict, save_fig=False, save_args=None):
    """
    Plot the composition report from the specified file path and save figure.
    """
    custom_kwargs = {"linestyle": "-"}
    fig, axes = plt.subplots(figsize=(8, 6))
    fig.set_tight_layout(True)
    xplot.plot_fit_report(
        report_dict,
        ax=axes,
        fit_param="BE",
        plot_kwargs=custom_kwargs,
        save_fig=save_fig,
        save_args=save_args,
    )


def fit_spectrum_parameters(spectrum_dict, save_fig=False, save_args=None):
    """
    Plot the fit spectrum from the specified dictionary and save figure.
    """
    custom_kwargs = {"linestyle": "-"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0},
    )
    fig.set_tight_layout(True)
    xplot.plot_fit_spectrum(
        spectrum_dict,
        ax=axes,
        x_axis="BE",
        normalised_residual=False,
        plot_kwargs=custom_kwargs,
        save_fig=save_fig,
        save_args=save_args,
    )


def main():
    """
    Main function to control what runs.
    """
    project_folder = (
        r"c:/Users/audun/OneDrive - Chalmers/Documents/Research/"
        r"1_Improving_XPS_Analysis_Methods/1_data/"
    )
    experiment_folder = r"250619_Gr_XPS_delithiated_and_lithiated/3_output/"
    save_folder = project_folder + experiment_folder
    fit_report_file = save_folder + r"line_scan_delithiated_fit_report.txt"
    fit_spectrum_file = save_folder + r"line_scan_delithiated_spectrum_fit_0.txt"

    # Flags to control execution
    run_convert_all_vms_in_project = False
    run_plot_fit_report = True
    run_plot_fit_spectrum = True
    save_figures = False
    save_args = {"format": "png", "save_folder": save_folder}

    if run_convert_all_vms_in_project:
        convert_all_vms(project_folder)

    if run_plot_fit_report:
        report_dict = xp.read_fit_report_file(fit_report_file)
        fit_report_parameters(report_dict, save_fig=save_figures, save_args=save_args)

    if run_plot_fit_spectrum:
        spectrum_dict = xp.read_fit_spectrum_file(fit_spectrum_file)
        fit_spectrum_parameters(
            spectrum_dict, save_fig=save_figures, save_args=save_args
        )

    plt.show()


if __name__ == "__main__":
    main()
