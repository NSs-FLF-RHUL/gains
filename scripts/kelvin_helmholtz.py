"""

Solves the incompressible Naiver Stokes equations in 2D to reproduce the Kelvin
Helmholtz instability in a shear flow.

Initialised with a real fourier basis to impose periodic boundary conditons

Initial and boundary conditions based on those described by McNally et al 2012, ApJ, 201, 18


"""

import numpy as np
import dedalus.public as d3
import logging

from gains.initial_conditions.mcnally import density, velocity_x
import argparse

logger = logging.getLogger(__name__)

#Command line interface
parser = argparse.ArgumentParser(
    description="Simulate Klevin-helmholtz instability using initial conditions "
    "from McNally et al 2012, ApJ, 201, 18"
)

parser.add_argument("--Nx", 
                    type=int,
                    default=16,
                    help="x-direction resolution")

parser.add_argument("--Ny",
                    type=int,
                    default=16,
                    help="y-direction resolution")

parser.add_argument("--stop_time",
                    type=float,
                    default=4.5,
                    help = "Cutoff for the simulated time")

parser.add_argument("--viscosity",
                    type=float,
                    default=1e-4,
                    help="The dynamic viscosity of the fluid")

parser.add_argument("--snapshots_dt",
                    type=float,
                    default=1e-2,
                    help = "Gap in simulated time between snapshots")

parser.add_argument("--logger_dt",
                    type=int,
                    default=100,
                    help = "How many timesteps between logger outputs")

args = vars(parser.parse_args())
dtype = np.float64
PARAMS = {
"Lx": 1,
"Ly": 1,
"Nx": args['Nx'],
"Ny": args['Ny'],
"timestepper": d3.SBDF4,
"stop_sim_time": args['stop_time'],
"max_timestep": 1e-4,
"dealias": 2,
"gamma": 5 / 3,
"rho_1": 1.0,
"rho_2": 2.0,
"L": 0.025,
"U_1": 0.5,
"U_2": -0.5,
"nu": args['viscosity'],
"snap_dt": args['snapshots_dt'],
"log_dt": args['logger_dt']
}
rho_m = (PARAMS["rho_1"] - PARAMS["rho_2"]) / 2
PARAMS["rho_m"] = rho_m
U_m = (PARAMS["U_1"] - PARAMS["U_2"]) / 2
PARAMS["U_m"] = U_m


# Bases

coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=dtype)
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
rho_y = density(xs=x, ys=y[0], **PARAMS)

rho_init = np.zeros((len(x), len(y[0])))

#for counter, value in enumerate(rho_y):
#    rho_init[counter] = [value for i in rho_init[counter]]


rho["g"] = rho_y


# x velocity

v_xs = velocity_x(xs=x, ys=y[0], **PARAMS)

u["g"][0] = np.array(v_xs)

# y velocity perturbations

vys = 0.01 * np.sin(4 * np.pi * x)

vys = [vys[i][0] for i in range(0, len(vys))]


vys_init = np.zeros((len(x), len(y[0])))

for j in range(0, len(y[0])):
    vys_init[j] = vys


u["g"][1] += 0.01 * np.sin(4 * np.pi * x)


p_init = np.zeros((len(x), len(y[0])))

for i in range(0, len(x)):
    for j in range(0, len(y[0])):
        p_init[i][j] = 2.5


# Analysis
snapshots = solver.evaluator.add_file_handler("snapshots", sim_dt=PARAMS["snap_dt"], max_writes=10)
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
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % PARAMS['log_dt'] == 0:
            logger.info(
                "Iteration=%i, Time=%e, dt=%e"
                % (solver.iteration, solver.sim_time, timestep)
            )
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
