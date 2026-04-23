import argparse
import datetime
import json
import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.exceptions import MeshError

# Parameters - load in from parameter file
from gains.utils import create_parser_simulation
from gains.initial_conditions.single_component_spin_up import window_equator
from gains.params.single_spin_up_rotating import parameters as default_params


#Setup
logger = logging.getLogger(__name__)

parser = create_parser_simulation()
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

radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)

Ek = PARAMS["Ek"]
B = PARAMS["B"]
Bprime = PARAMS["Bprime"]

if log2 == int(log2):
    mesh = [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
else:
    raise MeshError

logger.info(f"running on processor mesh={mesh}")

#Basis

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

#Fields
u_n = dist.VectorField(coords, name='u_n', bases=ball)
u_s = dist.VectorField(coords, name = 'u_s')
p_n = dist.Field(name='p_n', bases=ball)
p_s = dist.Field(name='p_s', bases = ball)
rho_n = dist.Field(name = 'rho_m', bases = ball)
rho_s = dist.Field(name = 'rho_s', bases = ball)

tau_p_n = dist.Field(name='tau_p_n')
tau_p_s = dist.Field(name='tau_p_s')
tau_u_n = dist.VectorField(name='tau_u_n')

#Substitutions
cross = d3.CrossProduct
dot = d3.DotProduct
curl = d3.Curl
lift = lambda a: d3.Lift(a, ball, -1)

phi, theta, r = dist.local_grids(ball)
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)
er['g'][2] = 1
etheta['g'][1] = 1
ephi['g'][0] = 1

ez = dist.VectorField(coords, bases=ball)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta) # unit vector in z direction

omega_s = curl(u_s)
u_sn = u_s - u_n
omega_unit = omega_s/np.sqrt(dot(omega_s,omega_s))
F_mf = B*(cross(omega_unit, cross(omega_s,u_sn))) - Bprime*cross(omega_s,u_sn)

sintheta = dist.Field(name='sintheta',bases=ball)
uang = dist.VectorField(coords, bases = ball)(r=radius).evaluate()
uang['g'][0,:] = (PARAMS["Delta_Omega"] * sintheta)(r=radius).evaluate()['g']

#problem
problem = d3.IVP([u_n,u_s,p_n,p_s,rho_n,rho_s,tau_p_n,tau_p_s, tau_u_n], namespace = locals())

problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation("div(u_s) + tau_p_s = 0")
problem.add_equation("integ(p_n) = 0")
problem.add_equation("integ(p_s) = 0")

problem.add_equation("dt(u_n) - Ek*lap(u_n) + grad(p_n) = -u_n@grad(u_n) + rho_s/rho_n * F_mf - 2*cross(e_z,u_n)")
problem.add_equation("dt(u_s) + grad(p_s) = -u_s@grad(u_s) - F_mf - 2*cross(e_z, u_s)")

problem.add_equation("radial(u_n(r=radius)) = 0")
problem.add_equation("radial(u_s(r=radius)) = 0")
problem.add_equation("angular(u_n(r=radius)) = angular(uang)")

problem.add_equation("dt(rho_s) = - u_s@grad(rho_s)")
problem.add_equation("dt(rho_n) = - u_n@grad(rho_n)")

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
