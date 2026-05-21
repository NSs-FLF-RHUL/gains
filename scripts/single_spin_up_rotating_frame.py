"""Simulates the spin up of a full basis.sphere containing a viscous newtonian fluid."""

import datetime
import json
import logging
from pathlib import Path

import dedalus.core as d3core
import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.bases.spherical import SphericalBasis
from gains.params.single_spin_up_rotating import parameters as default_params
from gains.problems.single_spin_up_rotating_frame import (
    SingleSpinUpRotatingFrameProblem,
)
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import create_parser_simulation
from gains.utils.profile import add_profiling_options, profile

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
    else "single_spin_up_"
    + datetime.datetime.now().astimezone().strftime("%Y-%m-%m-%H:%M")
)
PARAMS["profile"] = args["profile"]

# Additional Parameters - not likely to change between runs
radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
comm = MPI.COMM_WORLD
ncpu = comm.size

mesh = mesh_cpus(ncpu)
basis = SphericalBasis(mesh, radius, dtype, **PARAMS)

logger.info(f"running on processor mesh={mesh}")

setup = SingleSpinUpRotatingFrameProblem(basis, **PARAMS)
problem = setup.problem
u_n = setup.fields["u_n"]
er, etheta, ephi = setup.get_spherical_units()

dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
    # Shouldn't the initial condition be solid body rotation?
else:
    # Initial condition
    u_n.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_n.low_pass_filter(scales=0.5)
    timestep = max_timestep
# Analysis

volume = (4 / 3) * np.pi * radius**3


def az_avg(a: d3.Field) -> d3.Field:
    """Average over the phi coordinate."""
    return d3.Average(a, basis.coords.coords[0])


def s2_avg(a: d3.Field) -> d3.Field:
    """Average over all angular coordinates."""
    return d3.Average(a, basis.coords.S2basis.coordsys)


def vol_avg(a: d3.Field) -> d3.Field:
    """Average over whole basis.sphere."""
    return d3.Integrate(a / volume, basis.coords)


# define every component of velocity (for output)
u_n_r, u_n_theta, u_n_phi = setup.field_projection("u_n")

save_path = Path("outputs/{}/su_equator".format(PARAMS["output_dir"]))
save_path.mkdir(parents=True, exist_ok=True)

AZ_avg = solver.evaluator.add_file_handler(
    "outputs/{}/su_equator/AZ_avg_equator".format(PARAMS["output_dir"]),
    sim_dt=0.05,
    max_writes=100,
)
AZ_avg.add_task(dot(er, u_n), name="u_n_r")
AZ_avg.add_task(dot(etheta, u_n), name="u_n_theta")
AZ_avg.add_task(az_avg(u_n_phi), name="u_n_phi")


slices = solver.evaluator.add_file_handler(
    "outputs/{}/su_equator/slices".format(PARAMS["output_dir"]),
    sim_dt=0.025,
    max_writes=100,
)

slices.add_task(
    u_n_phi(theta=np.pi / 2), scales=PARAMS["dealias"], name="u_n_phi(equator)"
)

# Checkpoint
checkpoint = solver.evaluator.add_file_handler(
    f"outputs/{PARAMS['output_dir']}/su_equator/checkpoint",
    sim_dt=50,
    max_writes=1,
    parallel="gather",
)
checkpoint.add_tasks(solver.state, layout="g")

# CFL
CFL = d3.CFL(
    solver, timestep, cadence=1, safety=0.3, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(u_n)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_n @ u_n) * PARAMS["Ek"], name="Re_n")


@profile(dirname=PARAMS["profile"], run_output_dir=PARAMS["output_dir"])
def evolve(solver: d3core.solvers.InitialValueSolver) -> None:
    """Call solver.evolve, but decorate with the profiling function."""
    return solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)


evolve(solver)
