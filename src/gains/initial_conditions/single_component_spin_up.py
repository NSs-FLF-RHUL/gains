"""
Initial conditions for single_spin_up_rotating_frame.py.

Contains functions for generating initial conditions on
specific parts of the boundary of a sphere.

"""

import numpy as np

from gains.exceptions import ExpectPositiveError


def mask_angular(coord: np.ndarray, width: float, center: float) -> np.ndarray:
    """
    Create window function to enforce boundary conditions on only parts of the sphere.

    The returned function is close to 1 within the specified width, and close to 0
    everywhere else.

    :param coord: angular coordinate on which to create the window.
    :param width: Width of the window function.
    :param center: The coord value to center the mask round
    :returns mask: A smooth function equal to 1 in the window, and 0 everywhere else.
    """
    check = "width"
    if width <= 0:
        raise ExpectPositiveError(check)

    a = width / 2
    shift = center * np.ones_like(coord)
    coord = coord - shift
    mask = np.tanh((coord + a) / 0.1) - np.tanh((coord - a) / 0.1)
    return 0.5 * mask


def mask_r(rs: np.ndarray, width: float) -> np.ndarray:
    """
    Create radial window to enforce spin up torque on radial portions of the sphere.

    :param rs: Radial coordinates.
    :param width: The width to select (taken as 2 standard deviations).
    :returns mask: Gaussian window that selects a radial portion.
    """
    check = "width"
    if width <= 0:
        raise ExpectPositiveError(check)

    delta_r = width / 2
    return np.exp(-(((rs - 1.0) / delta_r) ** 2))

def circle_on_sphere(theta: np.ndarray, phi: np.ndarray, radius: float, center: tuple[float], width: float) -> np.ndarray:
    """
    Create circular mask defined on a sphere, with a smooth edge.

    Designed to use coordiantes provided by dedalus' dist.local_grids method,
    if they are instead 1D arrays they must be put through numpys meshgrid method
    first.

    :params theta: Polar angle [0,pi]
    :params phi: Azimuthal angle [0, 2pi]
    :params radius: Angular radius of the circle
    :params center: Center of circle given as (theta, phi)
    :params width: Width of smoothing function at the edge of the region
    """
    theta_0 = center[0]
    phi_0 = center[1]

    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    stheta_0 = np.sin(theta_0)
    ctheta_0 = np.cos(theta_0)
    cdiff = np.cos(phi - phi_0)

    cgamma = ctheta*ctheta_0 + stheta*stheta_0*cdiff
    gamma = np.arccos(np.clip(cgamma, -1, 1))

    f = np.zeros_like(gamma)
    inside = gamma <= radius
    transition = np.logical_and(gamma > radius, gamma < radius + width)
    x = (gamma[transition] - radius) / width

    f[inside] = 1.0
    f[transition] = np.cos(np.pi*x/2)**2

    return f
