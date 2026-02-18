"""
Contains functions for generating initial conditions as per
McNally et al., 2012, ApJ, 201, 18.
"""

import numpy as np


def density(ys: np.ndarray, **parameters: float | type) -> list[float]:
    """
    Density function for the initial condition in McNally 2012.

    :param ys: Y-coordinates on the boundary.
    :param parameters: Other simulation parameters.
    :returns density: Density values on the given boundary.
    """
    out = []
    for el in ys:
        if el < 0.25:
            out.append(
                parameters["rho_1"]
                - parameters["rho_m"] * np.exp((el - 0.25) / parameters["L"])
            )
        elif 0.25 <= el < 0.5:
            out.append(
                parameters["rho_2"]
                + parameters["rho_m"] * np.exp((-el + 0.25) / parameters["L"])
            )
        elif 0.5 <= el < 0.75:
            out.append(
                parameters["rho_2"]
                + parameters["rho_m"] * np.exp(-(0.75 - el) / parameters["L"])
            )
        else:
            out.append(
                parameters["rho_1"]
                - parameters["rho_m"] * np.exp(-(el - 0.75) / parameters["L"])
            )
    return out

def velocity_x(xs: np.ndarray, **parameters: float | type) -> list[float]:
    """
    x-velocity function for the initial condition in McNally 2012.

    :param xs: x-coordinates on the boundary.
    :param parameters: Other simulation parameters.
    :returns vx: x velocity values on the given boundary.
    """
    out = []
    for el in xs:
        if el < 0.25:
            out.append(parameters["U_1"] - parameters["U_m"] * np.exp((el - 0.25) / parameters["L"]))
        elif 0.25 <= el < 0.5:
            out.append(parameters["U_2"] + parameters["U_m"] * np.exp((-el + 0.25) / parameters["L"]))
        elif 0.5 <= el < 0.75:
            out.append(parameters["U_2"] + parameters["U_m"] * np.exp(-(0.75 - el) / parameters["L"]))
        else:
            out.append(parameters["U_1"] - parameters["U_m"] * np.exp(-(el - 0.75) / parameters["L"]))
    return out
