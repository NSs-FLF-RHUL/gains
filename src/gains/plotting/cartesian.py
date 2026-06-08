"""Holds functions that plot curves on cartesian axes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gains.analysis.analyse_spin_up import LabeledCoordinate, get_angular_speed_vs_time
from gains.utils.misc import _get_ax_and_fig, extract_numerical_suffix


def plot_against_time(
    coord: LabeledCoordinate,
    label: str,
    path: Path,
    ek: float,
    ntheta: int,
    targets: np.ndarray | list,
    target_field: str,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[list[Path], plt.Figure]:
    """
    Plot a range of coordinate values against time.

    :param coord: The coordinate and corrsponding label you want to vary when plotting.
    :param label: The label to appear on the legend.
    :param path: The path to the output directory
    :param ek: The ekman number used in this run
    :param ntheta: The number of theta values.
    :param targets: The values of the coordinate to measure the angular speed
    against time.
    :param target_field: The group name of the target velocity field in the
    output file.
    :returns path_list: A list of only .h5 files in the specified path.
    :returns fig: Figure on which the plot was drawn.
    """
    path = Path(path)

    path_list = sorted(
        (p for p in path.iterdir() if p.suffix == ".h5"), key=extract_numerical_suffix
    )

    alphas = np.linspace(0.40, 1.0, len(targets))
    fig, ax = _get_ax_and_fig(ax, polar=False)

    colour = kwargs.get("colour", "#024cf7")
    for i in range(len(targets)):
        target = targets[i]
        omega_r, times = get_angular_speed_vs_time(
            coord,
            target,
            target_field,
            100,
            path_list,
            ntheta=ntheta,
            rotating=kwargs["rotating"],
        )
        ax.plot(
            times,
            omega_r,
            color=colour,
            alpha=alphas[i],
            label=str(label + " = " + str(round(target, 2))),
        )
    ax.legend(frameon=False, loc="lower right")
    t_ek = 1 / np.sqrt(ek)
    ax.axvline(x=t_ek, linestyle="dashed", color="black", lw=0.5)
    ax.text(t_ek + 0.5, 0.0001, r"$\tau_{Ek}$", size="large")
    ax.set_xlabel(r"Time since glitch ($\Omega_{0}^{-1}$)")
    ax.set_ylabel(r"$\Delta \Omega$")

    return path_list, fig
