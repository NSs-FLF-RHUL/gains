<<<<<<< HEAD
<<<<<<< HEAD
import sys
=======
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
import numpy as np
import dedalus.public as d3
=======
>>>>>>> 928e57e (preliminary pre-commit run)
import logging

import dedalus.public as d3
import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__)


# Parameters - load in from parameter file

from gains.params.single_spin_up_rotating import parameters
<<<<<<< HEAD
<<<<<<< HEAD
from gains.initial_conditions.single_component_spin_up import window
=======
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
=======

>>>>>>> 928e57e (preliminary pre-commit run)
locals().update(parameters)

# Additional Parameters
radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)

<<<<<<< HEAD

=======
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
if log2 == int(log2):
    mesh = [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
logger.info(f"running on processor mesh={mesh}")


# Bases
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(
    coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype
)
sphere = ball.surface

# Fields
u_n = dist.VectorField(coords, name="u_n", bases=ball)
p_n = dist.Field(name="p_n", bases=ball)
omega_n = dist.VectorField(coords, name="omega_n", bases=ball)

<<<<<<< HEAD
tau_p_n = dist.Field(name='tau_p_n')
tau_u_n = dist.VectorField(coords, name='tau_u_n', bases=sphere)
tau_omega_n = dist.VectorField(coords, name = 'tau_omega_n', bases = sphere)
<<<<<<< HEAD
u_n_boundary = dist.VectorField(coords, name = 'u_n_boundary', bases=sphere)
=======
=======
tau_p_n = dist.Field(name="tau_p_n")
tau_u_n = dist.VectorField(coords, name="tau_u_n", bases=sphere)
tau_omega_n = dist.VectorField(coords, name="tau_omega_n", bases=sphere)
>>>>>>> 928e57e (preliminary pre-commit run)

>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
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

# This is never actually used

ez = dist.VectorField(coords, bases=ball)
ez["g"][1] = -np.sin(theta)
ez["g"][2] = np.cos(theta)  # unit vector in z direction

<<<<<<< HEAD


# This field is for the Boundary Conditions
sintheta = dist.Field(name='sintheta', bases=ball)
mask = dist.Field(name='mask',bases=sphere)
domega = dist.Field(name = 'domega', bases=ball)
sintheta['g'] = np.sin(theta)
mask['g'] = window(theta, 0.5)

=======
# This field is for the Boundary Conditions
<<<<<<< HEAD
sintheta = dist.Field(name='sintheta', bases=ball)
domega = dist.Field(name = 'domega', bases=ball)
sintheta['g'] = np.sin(theta)
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
domega['g'] = Delta_Omega
=======
sintheta = dist.Field(name="sintheta", bases=ball)
domega = dist.Field(name="domega", bases=ball)
sintheta["g"] = np.sin(theta)
domega["g"] = Delta_Omega
>>>>>>> 928e57e (preliminary pre-commit run)

uang_R1 = dist.VectorField(coords, bases=ball)(r=radius).evaluate()

uang_R1["g"][0, :] = (Delta_Omega * sintheta)(r=radius).evaluate()["g"]

omega_n["g"][1, :] = Omega_Init * -np.sin(theta)
omega_n["g"][2, :] = Omega_Init * np.cos(theta)

<<<<<<< HEAD
lift = lambda A: d3.Lift(A, ball, -1)
boundary_product = (mask*(u_n(r=radius) - uang_R1)).evaluate()
=======

lift = lambda A: d3.Lift(A, ball, -1)
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)

dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct

<<<<<<< HEAD
problem = d3.IVP([p_n, u_n, tau_p_n, tau_u_n], namespace=locals())
problem.add_equation("div(u_n) + tau_p_n = 0")
problem.add_equation("dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n)  = -u_n@grad(u_n) -2*cross(omega_n,u_n) - cross(curl(u_n),u_n)")
problem.add_equation("angular(u_n(r=radius)) = mask*angular(uang_R1) + (1-mask)*angular(u_n(r=radius))") # spin up at outer boundary
=======

problem = d3.IVP([p_n, u_n, tau_p_n, tau_u_n], namespace=locals())
problem.add_equation("div(u_n) +tau_p_n = 0")
<<<<<<< HEAD
problem.add_equation("dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n)  = -u_n@grad(u_n) -2*cross(omega_n,u_n) - cross(curl(u_n),u_n)")
problem.add_equation("angular(u_n(r=radius)) = angular(uang_R1)") # spin up at outer boundary
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
problem.add_equation("radial(u_n(r=radius)) = 0") # impenetrable bc
=======
problem.add_equation(
    "dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n)  = -u_n@grad(u_n) -2*cross(omega_n,u_n) - cross(curl(u_n),u_n)"
)
problem.add_equation(
    "angular(u_n(r=radius)) = angular(uang_R1)"
)  # spin up at outer boundary
problem.add_equation("radial(u_n(r=radius)) = 0")  # impenetrable bc
>>>>>>> 928e57e (preliminary pre-commit run)
problem.add_equation("integ(p_n) = 0")  # Pressure gauge normal fluid

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
<<<<<<< HEAD
#write, initial_timestep  = solver.load_state('checkpoint/checkpoint_s8.h5', -1)
<<<<<<< HEAD
=======
=======
# write, initial_timestep  = solver.load_state('checkpoint/checkpoint_s8.h5', -1)
>>>>>>> 928e57e (preliminary pre-commit run)

>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
use_checkpoint = False

if use_checkpoint:
    write, timestep = solver.load_state("checkpoints/checkpoints_sNumber.h5")
    # Shouldn't the initial condition be solid body rotation?
else:
    # Initial condition
    u_n.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_n.low_pass_filter(scales=0.5)
    timestep = max_timestep
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
=======

>>>>>>> 928e57e (preliminary pre-commit run)
# Analysis

volume = (4 / 3) * np.pi * radius**3
az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg = lambda A: d3.Integrate(A / volume, coords)

# define every component of velocity (for output)
u_n_r = dot(u_n, er)
u_n_theta = dot(u_n, etheta)
u_n_phi = dot(u_n, ephi)

<<<<<<< HEAD
<<<<<<< HEAD
AZ_avg = solver.evaluator.add_file_handler('AZ_avg_equator', sim_dt=0.05, max_writes=100)
=======
AZ_avg = solver.evaluator.add_file_handler('AZ_avg', sim_dt=0.05, max_writes=100)
>>>>>>> 0230396 (Uploaded current code for modelling spin up of a single component viscous fluid.)
AZ_avg.add_task(dot(er,u_n), name='u_n_r')
AZ_avg.add_task(dot(etheta,u_n), name='u_n_theta')
AZ_avg.add_task(az_avg(u_n_phi), name='u_n_phi')
=======
AZ_avg = solver.evaluator.add_file_handler("AZ_avg", sim_dt=0.05, max_writes=100)
AZ_avg.add_task(dot(er, u_n), name="u_n_r")
AZ_avg.add_task(dot(etheta, u_n), name="u_n_theta")
AZ_avg.add_task(az_avg(u_n_phi), name="u_n_phi")
>>>>>>> 928e57e (preliminary pre-commit run)


slices = solver.evaluator.add_file_handler("slices", sim_dt=0.025, max_writes=100)

slices.add_task(u_n_phi(theta=np.pi / 2), scales=dealias, name="u_n_phi(equator)")

# Checkpoint
checkpoint = solver.evaluator.add_file_handler(
    "checkpoint", wall_dt=3600, max_writes=1, parallel="gather"
)
checkpoint.add_tasks(solver.state, layout="g")

# CFL
CFL = d3.CFL(
    solver, timestep, cadence=1, safety=0.3, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(u_n)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_n @ u_n) * Ek, name="Re_n")

# Main loop
solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)
