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

    :param coord: angular coordinate on which to create the window,
    should range from 0 to pi.
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
