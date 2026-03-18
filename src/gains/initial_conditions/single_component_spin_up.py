import numpy as np


def window(coord: np.ndarray, width: float, dtype: type) -> np.ndarray:
    """
    Create window function for use in enforcing boundary conditions on only 
    parts of the sphere.

    :param coord: Coordinate on which to create the window.
    :param width: Width of the window function.
    :param dtype
    :returns mask: A smooth function equal to 1 in the window, and 0 everywhere else.
    """
    a = width / 2
    shift = np.pi / 2 * np.ones_like(coord)
    coord = coord - shift
    mask = np.tanh((coord + a) / 0.1) - np.tanh((coord - a) / 0.1)
    precision = np.finfo(dtype).eps
    mask[mask < 1e3 * precision] = (
        0  # about 3 orders of magnitude above dtype precision limit
    )
    return 0.5 * mask
