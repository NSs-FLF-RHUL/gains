"""Creation of bases for use in problems."""

import dedalus.public as d3


class SphericalBasis:
    """Initialise bases and distributor to solve spherical problems in dedalus."""

    def __init__(self, radius: float, mesh: list[int], dtype: type, **params: float) -> None:
        """
        Initialise spherical basis, including the surface.

        :param mesh: cpu mesh for dedalus distributor.
        :param dtype: data type for dedalus distributor.
        :param radius: The radius of the sphere
        """
        self.coords = d3.SphericalCoordinates("phi", "theta", "r")
        self.dist = d3.Distributor(self.coords, dtype=dtype, mesh=mesh)
        self.ball = d3.BallBasis(
            self.coords,
            shape=(params["Nphi"], params["Ntheta"], params["Nr"]),
            radius=radius,
            dealias=params["dealias"],
            dtype=dtype,
        )
        self.sphere = self.ball.surface
