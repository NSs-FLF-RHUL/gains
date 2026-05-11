"""Creation of bases for use in problems"""

import dedalus.public as d3
import dedalus.core as d3core

class spherical_basis:
    
    def __init__(self, mesh, dtype, **PARAMS):
        self.coords = d3.SphericalCoordinates("phi", "theta", "r")
        self.dist = d3.Distributor(self.coords, dtype=dtype, mesh=mesh)
        self.ball = d3.BallBasis(
        self.coords,
        shape=(PARAMS["Nphi"], PARAMS["Ntheta"], PARAMS["Nr"]),
        radius=1,
        dealias=PARAMS["dealias"],
        dtype=dtype,
    )
        self.sphere = self.ball.surface
