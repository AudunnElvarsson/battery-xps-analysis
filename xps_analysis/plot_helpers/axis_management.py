"""Axis management utilities for plotting.

This module handles axis setup, configuration, and management for XPS plots
including axis inversion, residual axis formatting, and figure/axes preparation.
"""

import matplotlib.pyplot as plt


def get_x_axis_from_dict(spectrum_dict, x_axis):
    """Get x-axis data and display metadata from a spectrum mapping.

    Parameters
    ----------
    spectrum_dict : dict
        Mapping that may contain ``'BE'`` and/or ``'KE'`` arrays.
    x_axis : str
        Requested axis: ``'BE'`` (binding energy) or ``'KE'`` (kinetic energy).

    Returns
    -------
    tuple
        ``(x_values, x_label, invert)`` where ``x_values`` is the array to
        plot on x, ``x_label`` is a human-readable axis label, and ``invert``
        is a boolean indicating whether the x-axis should be inverted.
    """
    if x_axis == "KE":
        return spectrum_dict.get("KE"), "Kinetic Energy (eV)", False
    return spectrum_dict.get("BE"), "Binding Energy (eV)", True


def figure_from_axes(ax_in):
    """Return the :class:`matplotlib.figure.Figure` for an axes-like input.

    Parameters
    ----------
    ax_in : Axes or sequence of Axes
        Axis-like object (single Axes or sequence) from which to obtain the
        parent figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure instance associated with the provided axes, or the
        current figure if none can be determined.
    """
    try:
        return ax_in[0].get_figure()
    except (TypeError, IndexError, AttributeError):
        try:
            return ax_in.get_figure()
        except (AttributeError, TypeError):
            return plt.gcf()


def select_target_axis(ax, key):
    """Select the axis to plot a given series on.

    Residual series (``'Residual'`` or ``'Normalised Residual'``) are
    plotted on the top/residual axis when a two-axis layout is used
    (``ax[0]``); other series use the main plotting axis (``ax[1]`` or
    ``ax``).

    Parameters
    ----------
    ax : Axes or sequence of Axes
        Axis or axes container used for plotting.
    key : str
        Name of the series being plotted.

    Returns
    -------
    Axes
        The target axes for the given key.
    """
    try:
        if key in ("Normalised Residual", "Residual"):
            return ax[0]
        return ax[1]
    except (TypeError, IndexError, AttributeError):
        return ax


def ensure_axes_and_main(ax_in):
    """Prepare and return a figure/axes tuple for plotting.

    If ``ax_in`` is ``None`` a new :class:`matplotlib.figure.Figure` and
    Axes are created. When ``ax_in`` is a sequence (e.g. ``(residual_ax,
    main_ax)``), the second element is treated as the main plotting axis.

    Parameters
    ----------
    ax_in : Axes or sequence of Axes or None
        Optional target axes provided by the caller.

    Returns
    -------
    tuple
        ``(fig, ax, main_ax, plot_here)`` where ``fig`` is the Figure,
        ``ax`` is the original axis object passed or created, ``main_ax`` is
        the primary plotting axis and ``plot_here`` is True when a new
        figure was created.
    """
    plot_here = False
    if ax_in is None:
        plot_here = True
        fig, ax = plt.subplots(figsize=(8, 6))
        # prefer the newer layout engine API when available; fall back to
        # set_tight_layout for older matplotlib versions to avoid
        # PendingDeprecationWarning
        try:
            fig.set_layout_engine("tight")
        except AttributeError:
            try:
                fig.set_tight_layout(True)
            except AttributeError:
                # last-resort: ignore if neither method exists
                pass
    else:
        fig = figure_from_axes(ax_in)
        ax = ax_in

    try:
        main_ax = ax[1]
    except (TypeError, IndexError, AttributeError):
        main_ax = ax
    return fig, ax, main_ax, plot_here


def configure_axes(ax, main_ax, invert=False):
    """Configure axes: invert x-axis and apply residual-axis formatting.

    Parameters
    ----------
    ax : Axes or sequence of Axes
        Axis or axes container used for plotting. When a sequence is used the
        residual axis is expected at index 0.
    main_ax : Axes
        Primary plotting axis.
    invert : bool, optional
        If True attempt to invert the x-axes (useful for binding energy
        plots where decreasing energy is conventional).

    Notes
    -----
    The helper swallows attribute/index errors to remain robust when a
    single-Axes object is supplied.
    """
    # invert axes if requested
    if invert:
        try:
            main_ax.invert_xaxis()
        except (AttributeError, TypeError):
            pass
        try:
            ax[0].invert_xaxis()
        except (AttributeError, IndexError, TypeError):
            pass

    # residual axis formatting
    try:
        ax[0].set_ylabel("Residual")
        ax[0].spines["bottom"].set_position(("data", 0))
        ax[0].set_xticks([])
        ax[0].set_xticklabels([])
        main_ax.xaxis.set_tick_params(labelbottom=True, bottom=True)
        ax[0].spines["bottom"].set_visible(False)
        main_ax.spines["top"].set_visible(False)
    except (AttributeError, IndexError, TypeError):
        pass
