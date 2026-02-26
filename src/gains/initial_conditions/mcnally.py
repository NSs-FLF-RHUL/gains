"""
Initial conditions for kelvin_helmholtz.py.

Contains functions for generating initial conditions as per
McNally et al., 2012, ApJ, 201, 18.

"""

import numpy as np


def density(xs: np.ndarray, ys: np.ndarray, **parameters: float | type) -> np.ndarray:
    """
    Density function for the initial condition in McNally 2012.

    :param ys: Y-coordinates on the boundary.
    :param xs: x-coordinates
    :param parameters: Other simulation parameters.
    :returns density: Density values on the given boundary.
    """
    bounds = [0.25, 0.5, 0.75]
    out = []
    for el in ys:
        if el < bounds[0]:
            out.append(
                parameters["rho_1"]
                - parameters["rho_m"] * np.exp((el - 0.25) / parameters["L"])
            )
        elif bounds[0] <= el < bounds[1]:
            out.append(
                parameters["rho_2"]
                + parameters["rho_m"] * np.exp((-el + 0.25) / parameters["L"])
            )
        elif bounds[1] <= el < bounds[2]:
            out.append(
                parameters["rho_2"]
                + parameters["rho_m"] * np.exp(-(0.75 - el) / parameters["L"])
            )
        else:
            out.append(
                parameters["rho_1"]
                - parameters["rho_m"] * np.exp(-(el - 0.75) / parameters["L"])
            )

    rho_init = np.zeros((len(xs), len(ys)))

    for counter, value in enumerate(out):
        rho_init[counter] = [
            value for i in rho_init[counter]
        ]  # Flipped matrix for density

    return np.transpose(np.array(rho_init))


def velocity_x(
    xs: np.ndarray, ys: np.ndarray, **parameters: float | type
) -> np.ndarray:
    """
    x-velocity function for the initial condition in McNally 2012.

    :param xs: x-coordinates on the boundary.
    :param ys: y-coordinates
    :param parameters: Other simulation parameters.
    :returns vx: x velocity values on the given boundary.
    """
    out = []
    bounds = [0.25, 0.5, 0.75]
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
        vx_init[counter] = [
            value for i in vx_init[counter]
        ]  # Flipped matrix for density

    return vx_init
