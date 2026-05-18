"""Creation of bases for use in problems."""

import dedalus.public as d3


class SphericalBasis:
    """Initialise bases and distributor to solve spherical problems in dedalus."""

    coords: d3.SphericalCoordinates
    dist: d3.Distributor
    ball: d3.BallBasis

    @property
    def sphere(self) -> d3.SphereBasis:
        """Define surface of the sphere."""
        return self.ball.surface

    def __init__(self, mesh: list[int], dtype: type, **params: float) -> None:
        """
        Initialise spherical basis, including the surface.

        :param mesh: cpu mesh for dedalus distributor.
        :param dtype: data type for dedalus distributor.
        """
        self.coords = d3.SphericalCoordinates("phi", "theta", "r")
        self.dist = d3.Distributor(self.coords, dtype=dtype, mesh=mesh)
        self.ball = d3.BallBasis(
            self.coords,
            shape=(params["Nphi"], params["Ntheta"], params["Nr"]),
            radius=1,
            dealias=params["dealias"],
            dtype=dtype,
        )
