"""Spherical basis."""

from collections.abc import Callable

import dedalus.public as d3
import numpy as np

from gains.bases.base import BaseBasis


class SphericalBasis(BaseBasis):
    """A spherical basis that includes the surface."""

    ball: d3.BallBasis

    @property
    def sphere(self) -> d3.SphereBasis:
        """The surface of the sphere."""
        return self.ball.surface

    def __init__(
        self, mesh: list[int], radius: float, dtype: type = np.float64, **params: float
    ) -> None:
        """
        Initialise spherical basis, including the surface.

        :param coords: The spherical coordinates used for the simulation.
        :param dist: The dedalus3 distributor used for the simulation.
        :param dtype: Data type used for the simulation.
        :param radius: Radius of the sphere (1 in non dimensionalised units).
        :param params: Other simulation parameters
        """
        super().__init__(mesh, dtype)
        self.ball = d3.BallBasis(
            self.coords,
            shape=(params["Nphi"], params["Ntheta"], params["Nr"]),
            radius=radius,
            dealias=params["dealias"],
            dtype=dtype,
        )

    def lift_operator(self) -> Callable[[d3.Field], d3.Lift]:
        """Return the Lift operator for this basis."""

        def _lift(a: d3.Field) -> d3.Lift:
            """Lift operand `a` to derivative basis."""
            return d3.Lift(a, self.ball, -1)

        return _lift

    def unit_ez(self) -> d3.Field:
        """Return the unit vector in the z-direction."""
        ez = self.dist.VectorField(self.coords, bases=self.ball)

        _, theta, _ = self.dist.local_grids(self.ball)

        ez["g"][1] = -np.sin(theta)
        ez["g"][2] = np.cos(theta)

        return ez
