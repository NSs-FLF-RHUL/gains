import numpy as np


def window(coord: np.ndarray, width: float) -> np.ndarray:
    """
    Creates window function for use in enforcing boundary conditions on only parts of the sphere.

    :param coord: Coordinate on which to create the window.
    :param width: Width of the window function.
    :returns mask: A smooth function equal to 1 in the window, and 0 everywhere else.
    """
    a = width / 2
    shift = np.pi / 2 * np.ones_like(coord)
    coord = coord - shift
    mask = np.tanh((coord + a) / 0.1) - np.tanh((coord - a) / 0.1)
    return 0.5 * mask
