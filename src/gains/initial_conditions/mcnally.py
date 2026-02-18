"""
Contains functions for generating initial conditions as per
McNally et al., 2012, ApJ, 201, 18.
"""

import numpy as np


def density(xs: np.ndarray ,ys: np.ndarray, **parameters: float | type) -> np.ndarray:
    """
    Density function for the initial condition in McNally 2012.

    :param ys: Y-coordinates on the boundary.
    :param xs: x-coordinates
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
        
        rho_init = np.zeros((len(xs),len(ys)))
        for counter, value in enumerate(out):
            rho_init[counter] = [value for i in rho_init[counter]] #Flipped matrix for density

        rho_init = np.transpose(np.array(rho_init))

    return rho_init

def velocity_x(xs: np.ndarray, ys: np.ndarray, **parameters: float | type) -> np.ndarray:
    """
    x-velocity function for the initial condition in McNally 2012.

    :param xs: x-coordinates on the boundary.
    :param ys: y-coordinates
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
        
        vxs_init = np.zeros((len(xs), len(ys)))
        v_xs = [
        out[i][0] for i in range(0, len(out))
        ] #Prevents array of arrays

        for counter, value in enumerate(v_xs):
            vxs_init[counter] = [
                value for i in vxs_init[counter]
            ] #Matrix where each column is the same
    return vxs_init
