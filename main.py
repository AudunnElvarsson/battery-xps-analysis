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


def report_parameters(report_dict, save_param=None):
    """
    Plot the composition report from the specified file path and save figure.
    """
    proc_param = {"fit_param": "BE", "calculate": "difference", "reference": ""}
    plot_param = {"linestyle": "-"}

    # Check if multi-core format
    is_multicore = "Core Level" in report_dict and "Name" not in report_dict

    if is_multicore and "core_level" not in proc_param:
        # Multi-core format plotting all core levels - create grid of axes
        core_levels = [
            k for k in report_dict.keys() if k not in ["Core Level", "File Name"]
        ]
        n_cores = len(core_levels)
        n_cols = min(3, n_cores)  # Max 3 columns
        n_rows = (n_cores + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        fig.set_tight_layout(True)
    else:
        # Single core level or specific core level selected
        fig, axes = plt.subplots(figsize=(8, 6))
        fig.set_tight_layout(True)

    xplot.plot_report(
        report_dict,
        ax=axes,
        proc_kwargs=proc_param,
        plot_kwargs=plot_param,
        save_kwargs=save_param,
    )


def spectrum_parameters(spectrum_dict, save_param=None):
    """
    Plot the spectrum from the specified dictionary and save figure.
    """
    proc_param = {"x_axis": "BE", "normalised_residual": False}
    plot_param = {"linestyle": "-"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0},
    )
    fig.set_tight_layout(True)

    xplot.plot_spectrum(
        spectrum_dict,
        ax=axes,
        proc_kwargs=proc_param,
        plot_kwargs=plot_param,
        save_kwargs=save_param,
    )


def main():
    """
    Main function to control what runs.
    """
    # Flags to control execution
    run_convert_all_vms_in_project = False
    run_print_report = True
    run_plot_report = True
    run_plot_spectrum = False

    project_folder = (
        r"c:/Users/audun/OneDrive - Chalmers/Documents/Research/"
        r"1_Improving_XPS_Analysis_Methods/1_data/"
    )
    experiment_folder = r"250221_Gr_XPS_beam_damage_and_neutralizers/3_output/"
    save_folder = project_folder + experiment_folder
    report_file = save_folder + r"5_e_beam_unwashed_before_after_report_F1s_O1s_C1s.txt"
    spectrum_file = save_folder + r"2_beam_damage_unwashed_spectrum_C1s.txt"

    save_param = {"save_fig": False, "format": "png", "save_folder": save_folder}

    if run_convert_all_vms_in_project:
        convert_all_vms(project_folder)

    if run_print_report or run_plot_report:
        report_dict = xp.read_report_file(report_file)

        if run_print_report:
            xp.print_report_averages(report_dict, reference="A")

        if run_plot_report:
            report_parameters(report_dict, save_param=save_param)

    if run_plot_spectrum:
        spectrum_dict = xp.read_spectrum_file(spectrum_file)
        spectrum_parameters(spectrum_dict, save_param=save_param)

    plt.show()


if __name__ == "__main__":
    main()
