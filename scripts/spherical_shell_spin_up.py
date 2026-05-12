import datetime
import json
import logging
from pathlib import Path

import dedalus.core as d3core
import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.params.single_spin_up_rotating import parameters as default_params
from gains.problems.bases import SphericalBasis
from gains.utils.misc import mesh_cpus

# Parameters - load in from parameter file
from gains.utils.parsers import create_parser_simulation
from gains.utils.profile import add_profiling_options, profile

# Setup
logger = logging.getLogger(__name__)

parser = create_parser_simulation()
add_profiling_options(parser)
args = vars(parser.parse_args())


if args["parameter_file"] is not None:
    with Path.open(args["parameter_file"]) as param_file:
        PARAMS = json.load(param_file)

else:
    PARAMS = default_params

PARAMS["use_checkpoint"] = args["use_checkpoint"]
PARAMS["checkpoint_path"] = args["checkpoint_path"]
PARAMS["output_dir"] = (
    args["output_dir"]
    if args["output_dir"] is not None
    else "two_fluid_spin_up_"
    + datetime.datetime.now().astimezone().strftime("%Y-%m-%m-%H:%M")
)

radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek = PARAMS["Ek"]
B = PARAMS["B"]
Bprime = B / 2
Ri = 0.9
Ro = 1.0
mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis

basis_full = SphericalBasis(Ro,mesh, dtype, **PARAMS)
basis_core = SphericalBasis(Ri, mesh, dtype, **PARAMS)
basis_crust = d3.ShellBasis(basis_full.coords,
                            shape=(PARAMS["Nphi"], PARAMS["Ntheta"], PARAMS["Nr"]),
                            radii = (Ri, Ro),
                            dealias=PARAMS["dealias"],
                            dtype=dtype
                            )
sphere = basis_crust.outer_surface

#Fields
u_crust = basis_full.dist.VectorField(basis_full.coords, name="u_crust", bases=basis_crust)
p_crust = basis_full.dist.VectorField(basis_full.coords, name = "p_crust", bases=basis_crust)
tau_ucr_1 = basis_full.dist.VectorField(basis_full.coords, name = "tau_ucr_1", bases = basis_crust)
tau_ucr_2 = basis_full.dist.VectorField(basis_full.coords, name = "tau_ucr_2", bases = basis_crust)
tau_pcr = basis_full.dist.Field(name="tau_pcr", bases=basis_crust)

u_core = basis_full.dist.VectorField(basis_full.coords, name="u_core", bases=basis_core.ball)
p_core = basis_full.dist.VectorField(basis_full.coords, name = "p_core", bases=basis_core.ball)
tau_uco_1 = basis_full.dist.VectorField(basis_full.coords, name = "tau_uco_1", bases = basis_core.ball)
tau_uco_2 = basis_full.dist.VectorField(basis_full.coords, name = "tau_uco_2", bases = basis_core.ball)
tau_pco = basis_full.dist.Field(name="tau_pcr", bases=basis_core.ball)

phi, theta, r = basis_full.dist.local_grids(basis_full.ball)
breakpoint()
