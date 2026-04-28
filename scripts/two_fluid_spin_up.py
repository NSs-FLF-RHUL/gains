import argparse
import datetime
import json
import logging
from pathlib import Path

import dedalus.public as d3
import dedalus.core as d3core
import numpy as np
from mpi4py import MPI

from gains.exceptions import MeshError

# Parameters - load in from parameter file
from gains.utils import create_parser_simulation, profile
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
    else "two_fluid_spin_up_"
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
Bprime = B/2

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
u_s = dist.VectorField(coords, name = 'u_s', bases=ball)
p_n = dist.Field(name='p_n', bases=ball)
p_s = dist.Field(name='p_s', bases = ball)
rho_n = 0.95
rho_s = 0.05

tau_p_n = dist.Field(name='tau_p_n')
tau_p_s = dist.Field(name='tau_p_s')
tau_u_n = dist.VectorField(coords, name='tau_u_n', bases=sphere)
tau_u_s = dist.VectorField(coords, name='tau_u_s', bases=sphere)

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
u_ns = u_n - u_s
omega_s = curl(u_s) +2*ez
omega_unit = omega_s/(np.sqrt(dot(omega_s,omega_s)) + 1e-14)
F_mf = B*(cross(omega_unit, cross(omega_s,u_ns))) + Bprime*cross(omega_s,u_ns)

sintheta = dist.Field(name='sintheta',bases=ball)
sintheta["g"] = np.sin(theta)
uang = dist.VectorField(coords, bases = ball)(r=radius).evaluate()
uang['g'][0,:] = (PARAMS["Delta_Omega"] * sintheta)(r=radius).evaluate()['g']
strain_rate = d3.grad(u_s) + d3.trans(d3.grad(u_s))
shear_stress = d3.angular(d3.radial(strain_rate(r=1), index=1))

#problem - HVBK equations spin up in sphere
problem = d3.IVP([u_n,u_s,p_n,p_s,tau_p_n,tau_p_s, tau_u_n, tau_u_s], namespace = locals())

problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation("div(u_s) + tau_p_s = 0")
problem.add_equation("integ(p_n) = 0")
problem.add_equation("integ(p_s) = 0")

problem.add_equation("dt(u_n) - Ek*lap(u_n) + grad(p_n) + lift(tau_u_n)= -u_n@grad(u_n) + rho_s/rho_n * F_mf - 2*cross(ez,u_n)")
problem.add_equation("dt(u_s) + grad(p_s) + lift(tau_u_s) = -u_s@grad(u_s) - F_mf - 2*cross(ez, u_s)")

problem.add_equation("radial(u_n(r=radius)) = 0")
problem.add_equation("radial(u_s(r=radius)) = 0")
problem.add_equation("angular(u_n(r=radius)) = angular(uang)")
problem.add_equation("shear_stress = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
    # Shouldn't the initial condition be solid body rotation?
else:
    # Initial condition
    u_n.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_n.low_pass_filter(scales=0.5)
    u_s.fill_random("g", seed=42,distribution="normal", scale=1e-10)
    u_s.low_pass_filter(scales=0.5)
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
AZ_avg.add_task(az_avg(dot(ephi, u_s)), name = "u_s_phi")

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
CFL.add_velocity(u_s)


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_n @ u_n) * PARAMS["Ek"], name="Re_n")

# Main loop
@profile("profiles_4", PARAMS)
def evolve(solver: d3core.solvers.InitialValueSolver) -> None:
    return solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)


evolve(solver)
