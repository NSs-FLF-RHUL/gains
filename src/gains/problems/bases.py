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

    def __init__(
        self,
        coords: d3.SphericalCoordinates,
        dist: d3.Distributor,
        dtype: type,
        radius: float,
        **params: float,
    ) -> None:
        """
        Initialise spherical basis, including the surface.

        :param mesh: cpu mesh for dedalus distributor.
        :param dtype: data type for dedalus distributor.
        :param radius: The radius of the sphere
        """
        self.coords = coords
        self.dist = dist
        self.ball = d3.BallBasis(
            self.coords,
            shape=(params["Nphi"], params["Ntheta"], params["Nr"]),
            radius=radius,
            dealias=params["dealias"],
            dtype=dtype,
        )
