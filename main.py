"""
This is to test the xps_analysis package.
"""

import matplotlib.pyplot as plt
import xps_analysis.xps_utilities as xu
import xps_analysis.xps_processing as xp
import xps_analysis.xps_plot as xplot


def convert_all_vms(proj_folder):
    """
    Convert all VM files in the specified project folder.
    """
    xu.convert_all_vms_in_project(proj_folder)


def comp_report_parameters(report_dict, save_param=None):
    # Select which core levels to plot
    selected_cores = xp.get_core_levels(report_dict)

    proc_param = {
        "fit_param": "At Conc",
        "calculate": "average",
        # "reference": {"F 1s": "PFx", "O 1s": "B", "C 1s": "A"},
        "core_levels": selected_cores,
        "show_labels": True,
        "normalize_at_conc_per_core": True,
    }
    plot_param = {"linestyle": "-"}

    # Create the right number of subplots based on selected cores
    n_cores = len(selected_cores)
    fig, axes = plt.subplots(1, n_cores, figsize=(6 * n_cores, 5))
    fig.set_tight_layout(True)

    xplot.plot_comp_report(
        report_dict,
        ax=axes,
        proc_kwargs=proc_param,
        plot_kwargs=plot_param,
        save_kwargs=save_param,
    )


def region_report_parameters(report_dict, save_param=None):
    # Example 1: Plot F/C and O/C atomic concentration ratios
    proc_param_1 = {
        "plot_type": "ratio",
        "parameter": "At Conc",
        "numerator": "F 1s",
        "denominator": "C 1s",
        "calculate": "average",
    }
    plot_param_1 = {"marker": "o", "color": "blue", "label": "F/C"}
    proc_param_2 = {
        "plot_type": "ratio",
        "parameter": "At Conc",
        "numerator": "P 2p",
        "denominator": "C 1s",
        "calculate": "average",
    }
    plot_param_2 = {"marker": "s", "color": "red", "label": "P/C"}
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_tight_layout(True)

    xplot.plot_region_report(
        report_dict,
        ax=ax,
        proc_kwargs=proc_param_1,
        plot_kwargs=plot_param_1,
        save_kwargs=save_param,
    )
    xplot.plot_region_report(
        report_dict,
        ax=ax,
        proc_kwargs=proc_param_2,
        plot_kwargs=plot_param_2,
        save_kwargs=save_param,
    )

    # Example 2: Plot total atomic concentrations for all core levels
    proc_param_3 = {
        "plot_type": "total",
        "parameter": "At Conc",
        "calculate": "average",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_tight_layout(True)

    xplot.plot_region_report(
        report_dict, ax=ax, proc_kwargs=proc_param_3, save_kwargs=save_param
    )


def spectrum_parameters(spectrum_dict, save_param=None):
    proc_param = {
        "x_axis": "BE",
        "normalised_residual": False,
        "plot_items": None,  # Control what to plot
    }
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
    run_print_report = False
    run_plot_comp_report = True
    run_plot_region_ratio = True
    run_plot_spectrum = False

    proj_folder = (
        r"c:/Users/audun/OneDrive - Chalmers/Documents/Research/"
        r"1_Improving_XPS_Analysis_Methods/1_data/"
    )
    exp_folder = proj_folder + r"250221_Gr_XPS_beam_damage_and_neutralizers/3_output/"
    report_file = exp_folder + r"2_beam_damage_unwashed_comp_report.txt"
    spectrum_file = exp_folder + r"2_beam_damage_unwashed_spectrum_C1s.txt"

    save_param = {
        "save_fig": False,
        "format": "png",
        "exp_folder": exp_folder,
    }

    if run_convert_all_vms_in_project:
        convert_all_vms(proj_folder)

    if run_print_report or run_plot_comp_report or run_plot_region_ratio:
        report_dict = xp.read_report_file(report_file)
        available_params = xp.get_available_parameters(report_dict)
        # print("Available parameters:", available_params)

        if run_print_report:
            xp.print_report(report_dict, reference=None, parameters=None)

        if run_plot_comp_report:
            comp_report_parameters(report_dict, save_param=save_param)

        if run_plot_region_ratio:
            region_report_parameters(report_dict, save_param=save_param)

    if run_plot_spectrum:
        spectrum_dict = xp.read_spectrum_file(spectrum_file)
        spectrum_parameters(spectrum_dict, save_param=save_param)

    plt.show()


if __name__ == "__main__":
    main()
