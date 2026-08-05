import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.params.spherical_shell import parameters as default_params
from gains.problems.bases import ShellBasis
from gains.utils.loggers import track_vorticity
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import SimulationCLI
from gains.utils.profile import profile

# Setup
logger = logging.getLogger(__name__)
parser = SimulationCLI(
    profiling_option=True,
    place_all_outputs_under="outputs",
    sim_name="two_fluid_spin_up",
)
PARAMS = parser.parse_args_and_get_params(logger, default_params=default_params)

radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek = PARAMS["Ek"]
B = PARAMS["B"]
Bprime = B / 2
Ri = PARAMS["Ri"]
Ro = PARAMS["Ro"]
mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = ShellBasis(coords, dist, dtype, **PARAMS)

#Fields

u = dist.VectorField(coords, name="u", )
