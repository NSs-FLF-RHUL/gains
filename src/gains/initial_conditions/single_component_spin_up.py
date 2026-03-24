"""
Initial conditions for single_spin_up_rotating_frame.py.

Contains functions for generating initial conditions on
specific parts of the boundary of a sphere.

"""

import numpy as np


def window_equator(theta: np.ndarray, width: float, dtype: type) -> np.ndarray:
    """
    Create window function to enforce boundary conditions on only parts of the sphere.

    The returned function is close to 1 within the specified width, and close to 0
    everywhere else.

    :param theta: Longitudinal coordinate on which to create the window,
    should range from 0 to pi.
    :param width: Width of the window function.
    :param dtype: Data type of coord
    :returns mask: A smooth function equal to 1 in the window, and 0 everywhere else.
    """
    a = width / 2
    shift = np.pi / 2 * np.ones_like(theta)
    theta = theta - shift
    mask = np.tanh((theta + a) / 0.1) - np.tanh((theta - a) / 0.1)
    precision = np.finfo(dtype).eps
    mask[mask < 1e3 * precision] = (
        0  # about 3 orders of magnitude above dtype precision limit
    )
    return 0.5 * mask
