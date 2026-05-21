"""Creation of bases for use in problems."""

import dedalus.public as d3
import numpy as np


class BaseBasis:
    r"""
    Base class for Basis information, to solve spherical problems with `dedalus`.

    Assumes a spherical coordinate system $(\phi, \theta, r)$. Initialise with a
    mesh plan to instantiate the `Distributor` correctly.

    Also contains common functionality that we may require of all bases. Can also be
    used for type-hinting where necessary.
    """

    coords: d3.SphericalCoordinates
    dist: d3.Distributor

    def __init__(self, mesh: list[int], dtype: type = np.float64) -> None:
        """Create a new Basis."""
        self.coords = d3.SphericalCoordinates("phi", "theta", "r")
        self.dist = d3.Distributor(self.coords, mesh=mesh, dtype=dtype)
