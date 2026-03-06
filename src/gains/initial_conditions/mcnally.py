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
    out = np.zeros((len(xs),))

    # Select all the x-coordinates that are in region 0
    region_0_mask = xs < bounds[0]
    region_1_mask = np.logical_and(xs >= bounds[0], xs < bounds[1])
    region_2_mask = np.logical_and(xs >= bounds[1], xs < bounds[2])
    region_3_mask = xs >= bounds[2]

    out[region_0_mask] = parameters["rho_1"] - parameters["rho_m"] * np.exp(
        (xs[region_0_mask] - 0.25) / parameters["L"]
    )
    out[region_1_mask] = parameters["rho_2"] + parameters["rho_m"] * np.exp(
        (-xs[region_1_mask] + 0.25) / parameters["L"]
    )
    out[region_2_mask] = parameters["rho_2"] + parameters["rho_m"] * np.exp(
        -(0.75 - xs[region_2_mask]) / parameters["L"]
    )
    out[region_3_mask] = parameters["rho_1"] - parameters["rho_m"] * np.exp(
        -(xs[region_3_mask] - 0.75) / parameters["L"]
    )

    return np.column_stack((out,) * len(ys)).T


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
    out = []
    for el in xs:
        if el < bounds[0]:
            out.append(
                parameters["U_1"]
                - parameters["U_m"] * np.exp((el - 0.25) / parameters["L"])
            )
        elif bounds[0] <= el < bounds[1]:
            out.append(
                parameters["U_2"]
                + parameters["U_m"] * np.exp((-el + 0.25) / parameters["L"])
            )
        elif bounds[1] <= el < bounds[2]:
            out.append(
                parameters["U_2"]
                + parameters["U_m"] * np.exp(-(0.75 - el) / parameters["L"])
            )
        else:
            out.append(
                parameters["U_1"]
                - parameters["U_m"] * np.exp(-(el - 0.75) / parameters["L"])
            )

    if np.shape(np.shape(out))[0] > 1:
        out = [out[i][0] for i in range(len(out))]

    vx_init = np.zeros((len(xs), len(ys)))

    for counter, value in enumerate(out):
        vx_init[counter] = [value for i in vx_init[counter]]

    return vx_init
