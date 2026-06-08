"""Holds functions that plot onto polar slices of the star."""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

from gains.analysis.analyse_spin_up import _my_interp2d, read_angular_velocity
from gains.utils.misc import _get_ax_and_fig, select_time

def plot_stream(
    r: np.ndarray,
    theta: np.ndarray,
    vr_n: np.ndarray,
    vtheta_n: np.ndarray,
    density: float | tuple[float],
    time: float,
    ax: plt.projections.polar.PolarAxes,
    **kwargs,
) -> None:
    """
    Create streamline plots of the meridional flow.

    :param r: radial coordinate.
    :param theta: meridional coordinate.
    :param vr_n: radial speed.
    :param vtheta_n: meridional speed.
    :param density: density of streamplot.
    :param ax: Polar axis to render plots on.
    """
    rad = np.linspace(r[-1], r[0], len(r))
    theta = np.linspace(0, np.pi, len(theta))

    rr, ttheta = np.meshgrid(rad, theta)
    un = vr_n[:, ::-1]
    vn = vtheta_n[:, ::-1] / rr[:, ::-1]

    un = _my_interp2d(un, r[::-1], rad)
    vn = _my_interp2d(vn, r[::-1], rad)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_ylim(0, 1.0)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t={round(time, 2)}")
    ax.streamplot(
        ttheta.T,
        rr.T,
        vn.T,
        un.T,
        color=kwargs["colour"],
        density=density,
        broken_streamlines=True,
        linewidth=1,
    )

def plot_angular(
    ax: plt.projections.polar.PolarAxes,
    r: np.ndarray,
    theta: np.ndarray,
    omega_values: np.ndarray,
    **kwargs,
) -> plt.pcolormesh:
    """
    Plot a given angular speed on a given coordinate grid.

    :param ax: Polar axis to plot angular speed on.
    :param r: Radial coordinates.
    :param theta: Polar angles.
    :param omega_values: Angular speeds.
    :returns mesh: pcolormesh corresponding to the created plot.
    """
    r_m, theta_m = np.meshgrid(r, theta)
    mesh = ax.pcolormesh(
        theta_m,
        r_m,
        omega_values,
        clim=(0, kwargs["Delta_Omega"]),
        cmap="RdBu_r",
        edgecolors="face",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def plot_angular_velocity(
    path: str | Path,
    t: int,
    ax: plt.projections.polar.PolarAxes,
    target_field: str,
    *,
    rotating: bool,
    delta_omega: float,
) -> plt.pcolormesh:
    """
    Take an output of single_spin_up_rotating_frame.py and plots the angular velocity.

    :param path: Path to an AZ_avg_s*.h5 file.
    :param t: Integer used to select the time plotted.
    :param ax: Pre-defined matplotlib polar axis on which to plot the data. The
    axis is modified in place by this function.
    :param rotating: Set true if the simulation was done in the
    rotating reference frame.
    :returns mesh: pcolormesh for setting colourbar if this is wanted.
    """
    data = h5py.File(path, mode="r")
    r, theta, omega = read_angular_velocity(path, t, target_field, rotating=rotating)
    time = np.array(data["scales/sim_time"])
    mesh = plot_angular(ax, r, theta, omega, Delta_Omega=delta_omega)
    ax.set_ylim(r.min(), r.max())
    ax.set_title(r"$t =$" + str(time[t])[:4])
    return mesh


def plot_angular_velocity_split(
    path: Path,
    t: int,
    ax: plt.projections.polar.PolarAxes,
    core_field: str,
    crust_field: str,
    *,
    rotating: bool,
    delta_omega: float,
    crustcore_boundary: float,
) -> list:
    """
    Plot angular velocities for coupled crust/core systems.

    :param path: Path to output file.
    :param t: Index of snapshot to be plotted.
    :param ax: Polar axis to plot the angular speed on.
    :param core_field: Name of hdf5 group containing the core field.
    :param crust_field: Name of hdf5 group containing the crust field.
    :param rotating: Set true if the simulation was done in the
    rotating reference frame.
    :param delta_omega: Size of the spin up in the glitch.
    :param crustcore_boundary: Radius of crust-core interface.
    :returns meshes: pcolormesh objects for both the crust and core angular
    velocity
    """
    data = h5py.File(path, mode="r")
    meshes = []
    time = np.array(data["scales/sim_time"])

    for field in [core_field, crust_field]:
        r, theta, omega = read_angular_velocity(path, t, field, rotating=rotating)
        mesh = plot_angular(ax, r, theta, omega, Delta_Omega=delta_omega)

        meshes.append(mesh)

    ax.set_ylim(0, 1.0)
    ax.set_title(r"$t =$" + str(time[t])[:4])
    ax.plot(
        theta, np.full_like(theta, crustcore_boundary), linestyle="--", color="black"
    )

    return meshes


def plot_angular_velocity_sequence(
    target_times: list[float],
    ax: list[plt.projections.polar.PolarAxes] | plt.projections.polar.PolarAxes,
    output_dir: Path,
    target_field: str,
    **kwargs,
) -> plt.pcolormesh:
    """
    Plot a sequence of plots of the angular speed at different times.

    :param target_times: The times to plot angular velocity.
    :param ax: List of axes from matplotlib subplots.
    :param output_dir: Location of simulation outputs.
    :param target_field: The group name of the target velocity field in the
    output file.
    :param kwargs: Simulation parameters.
    :returns mesh: pcolormesh for setting colourbar if this is wanted.
    """
    for i in range(len(target_times)):
        time = target_times[i]
        path, file_index = select_time(100, time, output_dir, **kwargs)
        if isinstance(target_field, str):
            mesh = plot_angular_velocity(
                path,
                file_index,
                ax[i],
                rotating=True,
                delta_omega=kwargs["Delta_Omega"],
                target_field=target_field,
            )
        else:
            mesh = plot_angular_velocity_split(
                path,
                file_index,
                ax[i],
                target_field[0],
                target_field[1],
                rotating=True,
                delta_omega=kwargs["Delta_Omega"],
                crustcore_boundary=kwargs["Ri"],
            )
    return mesh
