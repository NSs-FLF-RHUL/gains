"""
Initial conditions for kelvin_helmholtz.py.

Contains functions for generating initial conditions as per
McNally et al., 2012, ApJ, 201, 18.

"""

import numpy as np

bounds = np.array([0.25, 0.5, 0.75])


def density(xs: np.ndarray, ys: np.ndarray, **parameters: float | type) -> np.ndarray:
    """
    Density function for the initial condition in McNally 2012.

    density is returned as a 2D array where;
    density[i, j] = density at coordinate (xs[j], ys[i]).

    :param ys: Y-coordinates on the boundary.
    :param xs: x-coordinates
    :param parameters: Other simulation parameters.
    :returns density: Density values on the given boundary.
    """
    out = np.zeros((len(ys),))

    region_0_mask = ys < bounds[0]
    region_1_mask = np.logical_and(ys >= bounds[0], ys < bounds[1])
    region_2_mask = np.logical_and(ys >= bounds[1], ys < bounds[2])
    region_3_mask = ys >= bounds[2]

    out[region_0_mask] = parameters["rho_1"] - parameters["rho_m"] * np.exp(
        (ys[region_0_mask] - 0.25) / parameters["L"]
    )
    out[region_1_mask] = parameters["rho_2"] + parameters["rho_m"] * np.exp(
        (-ys[region_1_mask] + 0.25) / parameters["L"]
    )
    out[region_2_mask] = parameters["rho_2"] + parameters["rho_m"] * np.exp(
        -(0.75 - ys[region_2_mask]) / parameters["L"]
    )
    out[region_3_mask] = parameters["rho_1"] - parameters["rho_m"] * np.exp(
        -(ys[region_3_mask] - 0.75) / parameters["L"]
    )

    return out[None, :] * np.ones((len(xs), 1))


def velocity_x(
    xs: np.ndarray, ys: np.ndarray, **parameters: float | type
) -> np.ndarray:
    """
    x-velocity function for the initial condition in McNally 2012.

    velocity is returned as a 2D array where
    velocity[i, j] = x-velocity component at coordinate (xs[i], ys[j]).

    :param xs: x-coordinates on the boundary.
    :param ys: y-coordinates
    :param parameters: Other simulation parameters.
    :returns vx: x velocity values on the given boundary.
    """
    out = np.zeros((len(ys),))
    region_0_mask = ys < bounds[0]
    region_1_mask = np.logical_and(ys >= bounds[0], ys < bounds[1])
    region_2_mask = np.logical_and(ys >= bounds[1], ys < bounds[2])
    region_3_mask = ys >= bounds[2]

    out[region_0_mask] = parameters["U_1"] - parameters["U_m"] * np.exp(
        (ys[region_0_mask] - 0.25) / parameters["L"]
    )
    out[region_1_mask] = parameters["U_2"] - parameters["U_m"] * np.exp(
        (-ys[region_1_mask] + 0.25) / parameters["L"]
    )
    out[region_2_mask] = parameters["U_2"] - parameters["U_m"] * np.exp(
        -(0.75 - ys[region_2_mask]) / parameters["L"]
    )
    out[region_3_mask] = parameters["U_1"] - parameters["U_m"] * np.exp(
        -(ys[region_3_mask] - 0.75) / parameters["L"]
    )
    return out[None, :] * np.ones((len(xs), 1))
