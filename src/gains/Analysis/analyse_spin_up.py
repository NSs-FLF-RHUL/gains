"""Contains functions to produce plots in scripts/plot_spin_up.py."""

from pathlib import Path
import re

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inp

from gains.params.single_spin_up_rotating import parameters

PARAMS = parameters


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

def extract_suffix(path):
    match = re.search(r'(\d+)$', path.stem)
    return int(match.group(1))

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
) -> mpl.figure:
    """
    Create streamline plots of the meridional flow.

    :param r: radial coordinate.
    :param theta: meridional coordinate.
    :param vr_n: radial speed.
    :param vtheta_n: meridional speed.
    :param density: density of streamplot.
    """
    rad = np.linspace(r[-1], r[0], len(r))
    theta = np.linspace(0, np.pi, len(theta))

    rr, ttheta = np.meshgrid(rad, theta)
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})

    un = vr_n[:, ::-1]
    vn = vtheta_n[:, ::-1] / rr[:, ::-1]

    un = my_interp2d(un, r[::-1], rad)
    vn = my_interp2d(vn, r[::-1], rad)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_ylim(r.min(), r.max())
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.streamplot(
        ttheta.T,
        rr.T,
        vn.T,
        un.T,
        color="#d95f02",
        density=density,
        broken_streamlines=True,
        linewidth=1,
    )

    fig.tight_layout()

    return fig


def get_angular_coords(path: str | Path) -> np.ndarray:
    """
    Return r and theta coordinates from a given dedalus output file.

    :param path: The path to the output file.
    :returns r: The radial coordinates.
    :returns theta: The meridional coordinates.
    """
    data = h5py.File(path, mode="r")
    u_n_phi = data["tasks"]["u_n_phi"]
    r = u_n_phi.dims[3][0][:].ravel()
    theta = u_n_phi.dims[2][0][:].ravel()
    return r, theta


def get_angular_coords_single(
    path: str | Path, r_index: int, theta_index: int
) -> tuple[float, float]:
    """
    Return a radial and meridional coordinate of specified independant indicies.

    :param path: path to the u_n_phi data file.
    :param r_index: Index of the desired radial coordinate.
    :param theta_index: Index of the desired meridional coordinate.
    :returns r: The radial coordinate at r_index.
    :returns theta: The meridional coordinate at theta_index.
    """
    data = h5py.File(path, mode="r")
    u_n_phi = data["tasks"]["u_n_phi"]
    r = u_n_phi.dims[3][0][r_index]
    theta = u_n_phi.dims[2][0][theta_index]
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
    path: str | Path, r_arg: int, theta_arg: int, u_phis: np.ndarray
) -> float:
    """
    Calculate angular speed for a specific radius and longitude.

    The value of r and theta the angular speed is calculated for are found by providing
    the arguments they have in the full array.
    :param r_arg: Index for the desired radius.
    :param theta_arg: Index for the desired longitude.
    :param u_phis: Azimuthally averaged array of azimuthal velocities.
    :returns: The angular speed at the specified r and theta.
    """
    r, theta = get_angular_coords_single(path, r_arg, theta_arg)
    u_phi = u_phis[theta_arg][r_arg]
    return u_phi / (r * np.sin(theta))


def plot_angular_velocity(
    path: str | Path, t: int, ax: mpl.projections.polar.PolarAxes, *, rotating: bool
) -> None:
    """
    Take an output of single_spin_up_rotating_frame.py and plots the angular velocity.

    :param path: Path to an AZ_avg_s*.h5 file.
    :param t: Integer used to select the time plotted.
    :param ax: Pre-defined matplotlib polar axis on which to plot the data. The
    axis is modified in place by this function.
    :param rotating: Set true if the simulation was done in the
    rotating reference frame.
    """
    data = h5py.File(path, mode="r")
    u_n_phi = data["tasks"]["u_n_phi"][t, -1, :, :]
    r, theta = get_angular_coords(path)
    if not rotating:
        u_n_background = 1.0 * np.outer(np.sin(theta), r)
    else:
        u_n_background = np.zeros_like(u_n_phi)

    du_n_phi = u_n_phi - u_n_background
    omega = calculate_angular_speed(r, theta, du_n_phi)

    time = np.array(data["scales/sim_time"])
    r_m, theta_m = np.meshgrid(r, theta)
    ax.pcolormesh(
        theta_m,
        r_m,
        omega,
        clim=(0, PARAMS["Delta_Omega"]),
        cmap="RdBu_r",
        edgecolors="face",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.set_ylim(r.min(), r.max())
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.grid(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$t =$" + str(time[t])[:4])


def get_angular_speed_vs_time(
    coord: str, c_get: int, n_writes: int, path_list: list[Path]
) -> np.ndarray:
    """
    Find the angular speed at the equator at a given radius.

    :param coord: The coordinate to be varied - should be r or theta.
    :param c_get: Index of the coordinate we want.
    :param n_writes: Number of writes per .h5 file.
    :param path_list: List of paths to files to analyse.
    :returns omega_rs: List of angular velocities at each time.
    :returns times: List of times data is saved at.
    """
    err_msg = "coordinate must be r or theta."
    out_size = len(path_list) * n_writes
    omega_rs = np.zeros((out_size,))
    times = np.zeros(out_size)
    theta_resolution = PARAMS["Ntheta"]
    count = 0
    for path in path_list:
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        for j in range(n_writes):
            u_n_phi = data["tasks"]["u_n_phi"][j, -1, :, :]
            if coord == "r":
                omega_r = calculate_angular_speed_single(
                    path, c_get, int(theta_resolution / 2), u_n_phi
                )  # theta arg esnures the equator is selected.
            elif coord == "theta":
                omega_r = calculate_angular_speed_single(
                    path, -1, c_get, u_n_phi
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
) -> tuple[list[Path], mpl.figure]:
    """
    Plot a range of coordinate values against time.

    :param coord: The coordinate and corrsponding label you want to vary when plotting.
    :param label: The label to appear on the legend.
    :param path: The path to the output directory
    :param return_paths: Sets whether or not a list of paths to output files is
    returned.
    :param name: What to name the png file containing the figure.
    :returns path_list: A list of only .h5 files in the specified path.
    """
    path = Path(path)

    path_list = sorted((p for p in path.iterdir() if p.suffix == ".h5"), 
                       key = extract_suffix)
    print(path_list)

    coord_val = coord.coord
    coord_name = coord.label

    coord_tries = list(range(int(len(coord_val) / 2), len(coord_val), 6))
    alphas = np.linspace(0.40, 1.0, len(coord_tries))
    coord_checked = [coord_val[i] for i in coord_tries]

    fig = plt.figure()
    ax = fig.gca()

    for i in range(len(coord_tries)):
        val = coord_tries[i]
        omega_r, times = get_angular_speed_vs_time(coord_name, val, 100, path_list)
        ax.plot(
            sorted(times),
            sorted(omega_r),
            color="#024cf7",
            alpha=alphas[i],
            label=str(label + " = " + str(round(coord_checked[i], 2))),
        )
    ax.legend(frameon=False, loc="center left")
    t_ek = 1 / np.sqrt(PARAMS["Ek"])
    ax.axvline(x=t_ek, linestyle="dashed", color="black", lw=0.5)
    ax.text(t_ek + 0.5, 0.0001, r"$\tau_{Ek}$", size="large")
    ax.set_xlabel(r"Time since glitch ($\Omega_{0}^{-1}$)")
    ax.set_ylabel(r"$\Delta \Omega$")

    return path_list, fig
