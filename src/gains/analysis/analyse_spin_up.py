"""Contains functions to produce plots in scripts/plot_spin_up.py."""

from pathlib import Path

import h5py
import numpy as np
import scipy.interpolate as inp

from gains.utils.misc import get_arg_of_nearest


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


def _my_interp2d(f: np.ndarray, rad: np.ndarray, radnew: np.ndarray) -> np.ndarray:
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
    phi = u_phi.dims[1][0][:].ravel()
    return r, theta, phi


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
    omega = np.zeros_like(u_phi)
    for i in range(len(rs)):
        omega[:, i] = u_phi[:, i] / (rs[i] * np.sin(thetas)[:])
    return omega


def calculate_angular_speed_single(
    path: str | Path,
    r_arg: int,
    theta_arg: int,
    u_phis: np.ndarray,
    target_field: str,
    *,
    rotating: bool = True,
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
    :param rotating: Sets if simulation was done in the rotating frame. True by default
    :returns: The angular speed at the specified r and theta.
    """
    r, theta = get_angular_coords_single(path, r_arg, theta_arg, target_field)
    u_phi = u_phis[theta_arg][r_arg]
    if not rotating:
        u_bg = r * np.sin(theta)
        u_phi = u_phi - u_bg
    return u_phi / (r * np.sin(theta))


def read_angular_velocity(
    path: str | Path,
    t: int,
    target_field: str,
    *,
    rotating: bool = True,
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
    u_phi = data["tasks"][target_field][t, 64, :, :]
    r, theta, phi = get_angular_coords(path, target_field)
    if not rotating:
        u_background = 1.0 * np.outer(np.sin(theta), r)
    else:
        u_background = np.zeros_like(u_phi)

    du_n_phi = u_phi - u_background
    omega = calculate_angular_speed(r, theta, du_n_phi)
    return r, theta, omega


def get_angular_speed_vs_time(
    coord: LabeledCoordinate,
    target: float,
    target_field: str,
    n_writes: int,
    path_list: list[Path],
    ntheta: int,
    *,
    rotating: bool = True,
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
    :param rotating: Sets if simulation was done in the rotating frame. True by default
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
            try:
                u_phi = data["tasks"][target_field][j, -1, :, :]
                if coord.label == "r":
                    omega_r = calculate_angular_speed_single(
                        path,
                        c_get,
                        int(theta_resolution / 2),
                        u_phi,
                        target_field,
                        rotating=rotating,
                    )  # theta arg esnures the equator is selected.
                elif coord.label == "theta":
                    omega_r = calculate_angular_speed_single(
                        path, -1, c_get, u_phi, target_field, rotating=rotating
                    )  # r arg ensures the surface is selected.
                else:
                    raise NotImplementedError(err_msg)
                omega_rs[count] = omega_r
                times[count] = time[j]
                count += 1
            except Exception:
                breakpoint()
                raise

    return omega_rs, times
