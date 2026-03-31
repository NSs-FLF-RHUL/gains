"""
Analyses the Kelvin Helmholtz instability.

Solves the incompressible Naiver Stokes equations in 2D to reproduce
the Kelvin Helmholtz instability in a shear flow.

Initialised with a real fourier basis to impose periodic boundary
conditons

Initial and boundary conditions based on those described by
McNally et al 2012, ApJ, 201, 18
"""

import argparse
import datetime
import logging

import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.initial_conditions.mcnally import density, velocity_x

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Command line interface
parser = argparse.ArgumentParser(
    description="Simulate Klevin-helmholtz instability using initial conditions "
    "from McNally et al 2012, ApJ, 201, 18"
)

parser.add_argument("--Nx", type=int, default=16, help="x-direction resolution")

parser.add_argument("--Ny", type=int, default=16, help="y-direction resolution")

parser.add_argument(
    "--stop_time", type=float, default=4.5, help="Cutoff for the simulated time"
)

parser.add_argument(
    "--viscosity", type=float, default=1e-4, help="The dynamic viscosity of the fluid"
)

parser.add_argument(
    "--snapshots_dt",
    type=float,
    default=1e-2,
    help="Gap in simulated time between snapshots",
)

parser.add_argument(
    "--logger_dt",
    type=int,
    default=100,
    help="How many timesteps between logger outputs",
)

parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="Name of the output files.",
)

args = vars(parser.parse_args())

if args["name"] is None:
    name_new = "kelvin_helmholtz" + datetime.datetime.now().astimezone().strftime(
        "%Y-%m-%m-%H:%M"
    )
    args["name"] = name_new

dtype = np.float64
PARAMS = {
    "Lx": 1,
    "Ly": 1,
    "Nx": args["Nx"],
    "Ny": args["Ny"],
    "timestepper": d3.SBDF4,
    "stop_sim_time": args["stop_time"],
    "max_timestep": 1e-4,
    "dealias": 2,
    "gamma": 5 / 3,
    "rho_1": 1.0,
    "rho_2": 2.0,
    "L": 0.025,
    "U_1": 0.5,
    "U_2": -0.5,
    "nu": args["viscosity"],
    "snap_dt": args["snapshots_dt"],
    "log_dt": args["logger_dt"],
    "name": args["name"]
    if args["name"] is not None
    else "kelvin_helmholtz_"
    + datetime.datetime.now().astimezone().strftime("%Y-%m-%m-%H:%M"),
}

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)

if log2 == int(log2):
    mesh = [ncpu]
logger.info(f"running on processor mesh={mesh}")

rho_m = (PARAMS["rho_1"] - PARAMS["rho_2"]) / 2
PARAMS["rho_m"] = rho_m
U_m = (PARAMS["U_1"] - PARAMS["U_2"]) / 2
PARAMS["U_m"] = U_m


# Bases

coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(
    coords["x"], size=PARAMS["Nx"], bounds=(0, PARAMS["Lx"]), dealias=PARAMS["dealias"]
)
ybasis = d3.RealFourier(
    coords["y"], size=PARAMS["Ny"], bounds=(0, PARAMS["Ly"]), dealias=PARAMS["dealias"]
)


# Fields

u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
p = dist.Field(name="p", bases=(xbasis, ybasis))
rho = dist.Field(name="rho", bases=(xbasis, ybasis))
tau_p = dist.Field(name="tau_p")

# Substitutions

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
x = x.squeeze()
y = y.squeeze()
# Problem
problem = d3.IVP([u, rho, p, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(rho) = - u@grad(rho)")
problem.add_equation("dt(u) + grad(p) = (PARAMS['nu']/rho)*lap(u)- u@grad(u)")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(PARAMS["timestepper"])
solver.stop_sim_time = PARAMS["stop_sim_time"]

# Initial conditions - see McNally et al., 2012, ApJ, 201, 18 for more details

# density

rho_y = density(xs=x, ys=y, **PARAMS)

rho["g"] = rho_y

# x velocity

v_xs = velocity_x(xs=x, ys=y, **PARAMS)

u["g"][0] = v_xs

# y velocity perturbations

vys = 0.01 * np.sin(4 * np.pi * x)


u["g"][1] += vys[:, None]

# Analysis
snapshots = solver.evaluator.add_file_handler(
    "outputs/{}/snapshots".format(PARAMS["name"]),
    sim_dt=PARAMS["snap_dt"],
    max_writes=10,
)
snapshots.add_task(rho, name="density")

# CFL
CFL = d3.CFL(
    solver,
    initial_dt=PARAMS["max_timestep"],
    cadence=10,
    safety=0.2,
    threshold=0.1,
    max_change=1.5,
    min_change=0.5,
    max_dt=PARAMS["max_timestep"],
)
CFL.add_velocity(u)

# Main loop
solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=PARAMS["log_dt"])
