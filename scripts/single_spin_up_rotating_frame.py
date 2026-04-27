"""Simulates the spin up of a full sphere containing a viscous newtonian fluid."""

import argparse
import cProfile
import datetime
import json
import logging
from collections.abc import Callable
from pathlib import Path

import dedalus.core as d3core
import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.exceptions import MeshError

# Parameters - load in from parameter file
from gains.initial_conditions.single_component_spin_up import window_equator
from gains.params.single_spin_up_rotating import parameters as default_params

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Simulate a localised spin up on the surface of a sphere of fluid."
)

parser.add_argument(
    "--use_checkpoint",
    type=bool,
    default=False,
    help="Boolean argument to determine if to use a checkpoint file.",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="outputs/checkpoints/checkpoints_sNumber.h5",
    help="Path to the checkpoint file you want to use.",
)

parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory to store simulation outputs"
)

parser.add_argument(
    "--parameter_file",
    type=Path,
    default=None,
    help="relative path to parameter file to use for this run, saved in json format.",
)

parser.add_argument(
    "--profile",
    type=str,
    default=None,
    help="If an arg is provided, will also generate time profiling data,"
    " stored in a directory named as the"
    " argument provided.",
)

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
log2 = np.log2(ncpu)

if log2 == int(log2):
    mesh = [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
else:
    raise MeshError

logger.info(f"running on processor mesh={mesh}")


def profile(dirname: str | None) -> Callable:
    """
    Provide a decorator to use cProfile to profile an function running in parallel.

    The stats are optionally saved in a subdirectory of the overall
    output directory for the simulation, and saved using the dump_stats
    method in a format readable by snakeviz.

    :param dirname: The name of the directory to save the profiles to.
    """
    comm = MPI.COMM_WORLD

    if dirname is None:
        return lambda f: f

    def prof_decorator(f: Callable) -> Callable:
        def wrap_f(*args: object, **kwargs: object) -> object:
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            output_dir = Path("outputs") / PARAMS["output_dir"] / dirname
            # Only rank 0 creates directory to avoid race conditions
            if comm.rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
            # All ranks wait until directory exists
            comm.Barrier()

            filename = output_dir / Path(f"time_profile.{comm.rank}")
            pr.dump_stats(filename)
            
            return result

        return wrap_f

    return prof_decorator


# Bases
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(
    coords,
    shape=(PARAMS["Nphi"], PARAMS["Ntheta"], PARAMS["Nr"]),
    radius=1,
    dealias=PARAMS["dealias"],
    dtype=dtype,
)
sphere = ball.surface

# Fields
u_n = dist.VectorField(coords, name="u_n", bases=ball)
p_n = dist.Field(name="p_n", bases=ball)
omega_n = dist.VectorField(coords, name="omega_n", bases=ball)

tau_p_n = dist.Field(name="tau_p_n")
tau_u_n = dist.VectorField(coords, name="tau_u_n", bases=sphere)
tau_omega_n = dist.VectorField(coords, name="tau_omega_n", bases=sphere)
u_n_boundary = dist.VectorField(coords, name="u_n_boundary", bases=sphere)
# Substitutions
phi, theta, r = dist.local_grids(ball)

r_vec = dist.VectorField(coords, bases=ball)
r_vec["g"][2] = r
r_vec["g"][1] = theta
r_vec["g"][0] = phi
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)
er["g"][2] = 1
etheta["g"][1] = 1
ephi["g"][0] = 1


ez = dist.VectorField(coords, bases=ball)
ez["g"][1] = -np.sin(theta)
ez["g"][2] = np.cos(theta)  # unit vector in z direction


# This field is for the Boundary Conditions
sintheta = dist.Field(name="sintheta", bases=ball)
mask = dist.Field(name="mask", bases=sphere)

sintheta["g"] = np.sin(theta)
mask["g"] = window_equator(theta, 0.5, np.float64)


uang_r1 = dist.VectorField(coords, bases=ball)(r=radius).evaluate()

uang_r1["g"][0, :] = (PARAMS["Delta_Omega"] * sintheta)(r=radius).evaluate()["g"]


def lift(a: d3.Field) -> d3.Field:
    """Lift operand to derivative basis."""
    return d3.Lift(a, ball, -1)


dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct

Ek = PARAMS["Ek"]  # Seperately defined for use in equations

problem = d3.IVP([p_n, u_n, tau_p_n, tau_u_n], namespace=locals())
problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation(
    "dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n)  = -u_n@grad(u_n) "
    "-2*cross(ez,u_n)"
)
problem.add_equation(
    "angular(u_n(r=radius)) = mask*angular(uang_r1) + (1-mask)*angular(u_n(r=radius))"
)  # spin up at outer boundary
problem.add_equation("radial(u_n(r=radius)) = 0")  # impenetrable bc
problem.add_equation("integ(p_n) = 0")  # Pressure gauge normal fluid

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
    return d3.Average(a, coords.coords[0])


def s2_avg(a: d3.Field) -> d3.Field:
    """Average over all angular coordinates."""
    return d3.Average(a, coords.S2coordsys)


def vol_avg(a: d3.Field) -> d3.Field:
    """Average over whole sphere."""
    return d3.Integrate(a / volume, coords)


# define every component of velocity (for output)
u_n_r = dot(u_n, er)
u_n_theta = dot(u_n, etheta)
u_n_phi = dot(u_n, ephi)

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
    "outputs/su_equator/checkpoint",
    wall_dt=3600,
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


@profile(dirname=PARAMS["profile"])
def evolve(solver: d3core.solvers.InitialValueSolver) -> None:
    """Call solver.evolve, but decorate with the profiling function."""
    return solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)


if PARAMS["profile"] is not None:
    evolve(solver)

else:
    solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)
