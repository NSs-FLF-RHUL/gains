"""Contains functions to produce plots in scripts/plot_spin_up.py."""

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inp

from gains.utils.misc import extract_numerical_suffix, get_arg_of_nearest, select_time


class LabeledCoordinate:
    """Holds a coordinate (for example r or theta) and its name for use in plotting."""

    def __init__(self, coord: np.ndarray, label: str) -> None:
        """
        Apply a specified label to a specified coordinate.

        :param coord: The coordinate to be labelled.
        :param label: The label to give the coordinate (e.g "r" or "theta").
        """
        self.coord = coord
        self.label = label


def my_interp2d(f: np.ndarray, rad: np.ndarray, radnew: np.ndarray) -> np.ndarray:
    """
    Create a 2D interpolation of a function f.

    The interpolaion is done in 1 dimension in array slices of the other.
    :param f: The function interpolated over.
    :param rad: Initial coordinate f is defined over.
    :param radnew: New coordinate with correct shape.
    :returns fnew: Interpolation of f defined over new set ofs coords.
    """
    fnew = np.zeros_like(f)
    for i in range(f.shape[0]):
        val = f[i, :]
        spl_rep = inp.make_splrep(rad, val)
        fnew[i, :] = spl_rep(radnew)

    return fnew


def plot_stream(
    r: np.ndarray,
    theta: np.ndarray,
    vr_n: np.ndarray,
    vtheta_n: np.ndarray,
    density: float | tuple[float],
    time: float,
    ax: mpl.projections.polar.PolarAxes,
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

    un = my_interp2d(un, r[::-1], rad)
    vn = my_interp2d(vn, r[::-1], rad)
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


def get_angular_coords(path: str | Path, target_field: str) -> np.ndarray:
    """
    Return r and theta coordinates from a given dedalus output file.

    :param path: The path to the output file.
    :param target_field: The group name of the target velocity field in the
    output file.
    :returns r: The radial coordinates.
    :returns theta: The meridional coordinates.
    """
    data = h5py.File(path, mode="r")
    u_phi = data["tasks"][target_field]
    r = u_phi.dims[3][0][:].ravel()
    theta = u_phi.dims[2][0][:].ravel()
    return r, theta


def get_angular_coords_single(
    path: str | Path, r_index: int, theta_index: int, target_field: str
) -> tuple[float, float]:
    """
    Return a radial and meridional coordinate of specified independant indicies.

    :param path: path to the u_n_phi data file.
    :param r_index: Index of the desired radial coordinate.
    :param theta_index: Index of the desired meridional coordinate.
    :param target_field: The group name of the target velocity field in the
    output file.
    :returns r: The radial coordinate at r_index.
    :returns theta: The meridional coordinate at theta_index.
    """
    data = h5py.File(path, mode="r")
    u_phi = data["tasks"][target_field]
    r = u_phi.dims[3][0][r_index]
    theta = u_phi.dims[2][0][theta_index]
    return r, theta


def calculate_angular_speed(
    rs: np.ndarray, thetas: np.ndarray, u_phi: np.ndarray
) -> np.ndarray:
    """
    Calculate angular speed for a given set of azimuthal velocity components.

    Assumes the meridional coordinates are defined relative to the rotational axis.

    :param rs: The radial coordinates.
    :param thetas: The meridional coordinates.
    :param u_phi: The azimuthal speed.
    :returns omega: The angular velocity.
    """
    omega = np.zeros((len(thetas), len(rs)))
    for i in range(len(rs)):
        omega[:, i] = u_phi[:, i] / (rs[i] * np.sin(thetas)[:])
    return omega


def calculate_angular_speed_single(
    path: str | Path, r_arg: int, theta_arg: int, u_phis: np.ndarray, target_field: str
) -> float:
    """
    Calculate angular speed for a specific radius and longitude.

    The value of r and theta the angular speed is calculated for are found by providing
    the arguments they have in the full array.
    :param r_arg: Index for the desired radius.
    :param theta_arg: Index for the desired longitude.
    :param u_phis: Azimuthally averaged array of azimuthal velocities.
    :param target_field: The group name of the target velocity field in the
    output file.
    :returns: The angular speed at the specified r and theta.
    """
    r, theta = get_angular_coords_single(path, r_arg, theta_arg, target_field)
    u_phi = u_phis[theta_arg][r_arg]
    return u_phi / (r * np.sin(theta))


def read_angular_velocity(
    path: str | Path,
    t: int,
    target_field: str,
    *,
    rotating: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Caluclate the angular speed from a target velocity field.

    :param path: Path to output file.
    :param t: Index of snapshot within file.
    :param target_field: Title of target velocity components hdf5 group.
    :param rotating: Set true if simulation was done in the rotating frame.
    :returns r: Array of radial coordinates from snapshot.
    :returns theta: Array of polar angles from snapshot.
    :returns omega: Array of calculated angular speeds.
    """
    data = h5py.File(path, mode="r")
    u_phi = data["tasks"][target_field][t, -1, :, :]
    r, theta = get_angular_coords(path, target_field)
    if not rotating:
        u_background = 1.0 * np.outer(np.sin(theta), r)
    else:
        u_background = np.zeros_like(u_phi)

    du_n_phi = u_phi - u_background
    omega = calculate_angular_speed(r, theta, du_n_phi)
    return r, theta, omega


def plot_angular(
    ax: mpl.projections.polar.PolarAxes,
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
    ax: mpl.projections.polar.PolarAxes,
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
    ax: mpl.projections.polar.PolarAxes,
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
    ax: list[mpl.projections.polar.PolarAxes] | mpl.projections.polar.PolarAxes,
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


def get_angular_speed_vs_time(
    coord: LabeledCoordinate,
    target: float,
    target_field: str,
    n_writes: int,
    path_list: list[Path],
    ntheta: int,
) -> np.ndarray:
    """
    Find the angular speed at the equator at a given radius.

    :param coord: The coordinate to be varied - should be r or theta.
    :param target: Value of the coordinate we want.
    :param target_field: The group name of the target velocity field in the
    output file.
    :param n_writes: Number of writes per .h5 file.
    :param path_list: List of paths to files to analyse.
    :param ntheta: The number of theta values.
    :returns omega_rs: List of angular velocities at each time.
    :returns times: List of times data is saved at.
    """
    err_msg = "coordinate must be r or theta."

    out_size = len(path_list) * n_writes
    omega_rs = np.zeros((out_size,))
    times = np.zeros(out_size)
    theta_resolution = ntheta
    count = 0
    c_get = get_arg_of_nearest(target, coord.coord)[0]
    for path in path_list:
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        for j in range(n_writes):
            u_phi = data["tasks"][target_field][j, -1, :, :]
            if coord.label == "r":
                omega_r = calculate_angular_speed_single(
                    path, c_get, int(theta_resolution / 2), u_phi, target_field
                )  # theta arg esnures the equator is selected.
            elif coord.label == "theta":
                omega_r = calculate_angular_speed_single(
                    path, -1, c_get, u_phi, target_field
                )  # r arg ensures the surface is selected.
            else:
                raise NotImplementedError(err_msg)
            omega_rs[count] = omega_r
            times[count] = time[j]
            count += 1

    return omega_rs, times


def plot_against_time(
    coord: LabeledCoordinate,
    label: str,
    path: Path,
    ek: float,
    ntheta: int,
    targets: np.ndarray | list,
    target_field: str,
) -> tuple[list[Path], mpl.figure]:
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
    """
    path = Path(path)

    path_list = sorted(
        (p for p in path.iterdir() if p.suffix == ".h5"), key=extract_numerical_suffix
    )

    alphas = np.linspace(0.40, 1.0, len(targets))

    fig = plt.figure()
    ax = fig.gca()

    for i in range(len(targets)):
        target = targets[i]
        omega_r, times = get_angular_speed_vs_time(
            coord, target, target_field, 100, path_list, ntheta=ntheta
        )
        ax.plot(
            times,
            omega_r,
            color="#024cf7",
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
