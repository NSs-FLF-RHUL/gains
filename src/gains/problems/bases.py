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

        :param coords: The spherical coordinates used for the simulation.
        :param dist: The dedalus3 distributor used for the simulation.
        :param dtype: Data type used for the simulation.
        :param radius: Radius of the sphere (1 in non dimensionalised units).
        :param params: Other simulation parameters
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


class ShellBasis:
    """Initialise bases and distributor to solve problems in shells in dedalus."""

    coords: d3.SphericalCoordinates
    dist: d3.Distributor
    shell: d3.ShellBasis

    @property
    def surface(self) -> d3.SphereBasis:
        """Define surface of the sphere."""
        return self.shell.outer_surface

    def __init__(
        self,
        coords: d3.SphericalCoordinates,
        dist: d3.Distributor,
        dtype: type,
        **params: float,
    ) -> None:
        """
        Initialise spherical basis, including the surface.

        :param coords: The spherical coordinates used for the simulation.
        :param dist: The dedalus3 distributor used for the simulation.
        :param dtype: Data type used for the simulation.
        :param params: Other simulation parameters
        """
        self.coords = coords
        self.dist = dist
        self.shell = d3.ShellBasis(
            coords,
            (params["Nphi"], params["Ntheta"], params["Nr"]),
            radii=(params["Ri"], params["Ro"]),
            dealias=params["dealias"],
            dtype=dtype,
        )
