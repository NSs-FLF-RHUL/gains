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

    def field(self, *args, **kwargs) -> d3.Field:
        """Create a new `Field` that is attached to this basis."""
        return self.dist.Field(*args, **kwargs)

    def vector_field(self, *args, **kwargs) -> d3.Field:
        """Create a new `VectorField` that is attached to this basis."""
        return self.dist.VectorField(self.coords, *args, **kwargs)

    def unit_vectors(self) -> tuple[d3.Field, d3.Field, d3.Field]:
        r"""Return unit vectors for the $r, \theta, \phi$ directions, in that order."""
        er = self.vector_field()
        etheta = self.vector_field()
        ephi = self.vector_field()

        er["g"][2] = 1
        etheta["g"][1] = 1
        ephi["g"][0] = 1

        return er, etheta, ephi
