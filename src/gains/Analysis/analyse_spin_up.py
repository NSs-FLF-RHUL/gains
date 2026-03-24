"""Contains functions to produce plots in scripts/plot_spin_up.py."""

import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inp

from gains.params.single_spin_up_rotating import parameters

locals().update(parameters)
PARAMS = parameters


def my_interp2d(f: np.ndarray, rad: np.ndarray, radnew: np.ndarray) -> np.ndarray:
    """Create a 2D interpolation of a function f."""
    r = rad
    rnew = radnew
    fnew = np.zeros_like(f)
    for i in range(f.shape[0]):
        val = f[i, :]
        tckp = inp.splrep(r, val)
        fnew[i, :] = inp.splev(rnew, tckp)

    return fnew


def plot_stream(
    r: np.ndarray,
    theta: np.ndarray,
    vr_n: np.ndarray,
    vtheta_n: np.ndarray,
    density: float,
) -> None:
    """Create streamline plots of the meridional flow."""
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


def get_angular_coords(path: str) -> np.ndarray:
    """Return r and theta coordinates from a given dedalus output file."""
    data = h5py.File(path, mode="r")
    u_n_phi = data["tasks"]["u_n_phi"]
    r = u_n_phi.dims[3][0][:].ravel()
    theta = u_n_phi.dims[2][0][:].ravel()
    return r, theta


def calculate_angular_speed(rs: np.ndarray, thetas: np.ndarray, u_phi: np.ndarray) -> np.ndarray:
    """Calculate angular speed for a given set of phi velocity components."""
    omega = np.zeros((len(thetas), len(rs)))
    for i in range(len(rs)):
        omega[:, i] = u_phi[:, i] / (rs[i] * np.sin(thetas)[:])
    return omega


def plot_angular_velocity(
    path: str, t: int, ax: mpl.projections.polar.PolarAxes, *, rotating: bool
) -> None:
    """
    Take an output of viscous_sphere.py and plots the angular velocity.

    :param path: Path to an AZ_avg_s*.h5 file.
    :param t: Integer used to select the time plotted.
    :param ax: Pre-defined matplotlib polar axis on which to plot the data.
    """
    data = h5py.File(path, mode="r")
    u_n_phi = data["tasks"]["u_n_phi"][t, -1, :, :]
    r, theta = coords_angular(path)
    u_n_background = np.zeros_like(u_n_phi)
    if not rotating:
        for i in range(len(r)):
            u_n_background[:, i] = parameters["Omega_Init"] * (r[i] * np.sin(theta)[:])

    du_n_phi = u_n_phi - u_n_background
    omega = get_angular(r, theta, du_n_phi)

    time = np.array(data["scales/sim_time"])
    r_m, theta_m = np.meshgrid(r, theta)
    ax.pcolormesh(
        theta_m,
        r_m,
        omega,
        clim=(0, parameters["Delta_Omega"]),
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


def get_angular_speed_vs_time(r_get: int, n_writes: int, path_list: list[str]) -> np.ndarray:
    """
    Find the angular speed at the equator at a given radius.

    :param r_get: Index of the radius we want.
    :param n_writes: Number of writes per .h5 file.
    :param path_list: List of paths to files to analyse.
    :returns omega_rs: List of angular velocities at each time.
    :returns times: List of times data is saved at.
    """
    omega_rs = []
    times = []
    theta_resolution = PARAMS["Ntheta"]
    for path in path_list:
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        r, theta = coords_angular(path)
        for j in range(n_writes):
            u_n_phi = data["tasks"]["u_n_phi"][j, -1, :, :]
            omega = get_angular(r, theta, u_n_phi)
            omega_r = omega[int(theta_resolution / 2)][r_get]
            omega_rs.append(omega_r)
            times.append(time[j])
    return omega_rs, times


def plot_against_time(
    coord: np.ndarray, name: str, label: str, path: str, *, return_paths: bool
) -> None | list[str]:
    """Plot a range of coordinate values against time."""
    file_list = sorted(os.listdir(path))
    path_list = []
    for file in file_list:
        extension = file[len(file) - 2 : len(file)]

        if extension == "h5":
            path_list.append(path + "/" + file)

    coord_tries = list(range(int(len(coord) / 2), len(coord), 4))
    alphas = np.linspace(0.40, 1.0, len(coord_tries))
    coord_checked = [coord[i] for i in range(35, len(coord), 6)]

    for i in range(len(coord_tries)):
        val = coord_tries[i]
        omega_r, times = angular_time(val, 100, path_list)
        plt.plot(
            sorted(times),
            sorted(omega_r),
            color="#024cf7",
            alpha=alphas[i],
            label=str(label + " = " + str(round(coord_checked[i], 2))),
        )

    plt.legend(frameon=False)
    t_ek = 1 / np.sqrt(parameters["Ek"])
    plt.axvline(x=t_ek, linestyle="dashed", color="black", lw=0.5)
    plt.text(t_ek + 0.5, 0.0001, r"$\tau_{Ek}$", size="large")
    plt.xlabel(r"Time since glitch ($\Omega_{0}^{-1}$)")
    plt.ylabel(r"$\Delta \Omega$")
    plt.savefig(f"outputs/su_equator/{name}.png", dpi=300)

    if return_paths:
        return path_list
    return None
