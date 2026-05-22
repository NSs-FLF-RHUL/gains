"""Shell (hollow-sphere) basis."""

import dedalus.public as d3
import numpy as np

from gains.bases.base import BaseBasis


class ShellBasis(BaseBasis):
    """A basis appropriate for solving problems on a shell=shaped domain."""

    shell: d3.ShellBasis

    @property
    def surface(self) -> d3.SphereBasis:
        """Define surface of the sphere."""
        return self.shell.outer_surface

    def __init__(
        self, mesh: list[int], dtype: type = np.float64, **params: float
    ) -> None:
        """
        Initialise spherical basis, including the surface.

        :param coords: The spherical coordinates used for the simulation.
        :param dist: The dedalus3 distributor used for the simulation.
        :param dtype: Data type used for the simulation.
        :param params: Other simulation parameters
        """
        super().__init__(mesh, dtype=dtype)
        self.shell = d3.ShellBasis(
            self.coords,
            (params["Nphi"], params["Ntheta"], params["Nr"]),
            radii=(params["Ri"], params["Ro"]),
            dealias=params["dealias"],
            dtype=dtype,
        )
