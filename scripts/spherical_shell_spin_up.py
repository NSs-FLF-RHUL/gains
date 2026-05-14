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
Ri = 0.5 #Add to parameter file
Ro = 1.0
mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis

coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)

basis_core = SphericalBasis(coords, dist, Ri, dtype, **PARAMS)
basis_shell = d3.ShellBasis(coords, 
                            (PARAMS['Nphi'], PARAMS['Ntheta'], PARAMS['Nr']),
                            radii=(Ri, Ro),
                            dealias=PARAMS["dealias"],
                            dtype=dtype
                            )
surface = basis_shell.outer_surface

#Crust fields

u_cr = dist.VectorField(coords, name = 'u_cr', bases=basis_shell)
p_cr = dist.Field(name='p_cr', bases=basis_shell)
tau_pcr = dist.Field(name='tau_pcr')
tau_ucr_1 = dist.VectorField(coords, name='tau_ucr_1', bases = surface)
tau_ucr_2 = dist.VectorField(coords, name='tau_ucr_2', bases=surface)

#Crust substitutions
cross = d3.CrossProduct
dot = d3.DotProduct
curl = d3.Curl

lift_basis_crust = basis_shell.derivative_basis(1)
lift_crust = lambda a: d3.Lift(a, lift_basis_crust, -1)

er_crust = dist.VectorField(coords)
etheta_crust = dist.VectorField(coords)
ephi_crust = dist.VectorField(coords)

phi_crust, theta_crust, r_crust = dist.local_grids(basis_shell)

er_crust["g"][2] = 1
etheta_crust["g"][1] = 1
ephi_crust["g"][0] = 1
ez_crust = dist.VectorField(coords, bases=basis_shell)
ez_crust["g"][1] = -np.sin(theta_crust)
ez_crust["g"][2] = np.cos(theta_crust)


rvec = dist.VectorField(coords, bases=basis_shell.radial_basis)
rvec['g'][2] = r_crust
grad_ucr = d3.grad(u_cr) + rvec*lift_crust(tau_ucr_1)

sintheta = dist.Field(name="sintheta", bases=basis_shell)
sintheta["g"] = np.sin(theta_crust)
uang = dist.VectorField(coords, bases=basis_shell)(r=radius).evaluate()
uang["g"][0, :] = (PARAMS["Delta_Omega"] * sintheta)(r=radius).evaluate()["g"]

strain_rate_cr = grad_ucr + d3.trans(grad_ucr)
shear_stress_cr = d3.angular(d3.radial(strain_rate_cr(r=Ri), index=1))

#Problem for crust (testing)



problem = d3.IVP([u_cr, p_cr, tau_pcr, tau_ucr_1, tau_ucr_2], namespace=locals())
problem.add_equation("trace(grad_ucr) + tau_pcr = 0")
problem.add_equation("integ(p_cr) = 0")

problem.add_equation("dt(u_cr) - Ek*div(grad_ucr) + grad(p_cr) + lift_crust(tau_ucr_2) = - u_cr@grad(u_cr) - 2*cross(ez_crust,u_cr)")

problem.add_equation("radial(u_cr(r=Ro)) = 0")
problem.add_equation("angular(u_cr(r=Ro)) = angular(uang)")

problem.add_equation("radial(u_cr(r=Ri)) = 0")
problem.add_equation("shear_stress_cr = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
else:
    # Initial condition
    u_cr.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_cr.low_pass_filter(scales=0.5)
    timestep = max_timestep

# Analysis
volume = (4 / 3) * np.pi * radius**3


def az_avg(a: d3.Field) -> d3.Field:
    """Average over the phi coordinate."""
    return d3.Average(a, coords.coords[0])


def s2_avg(a: d3.Field) -> d3.Field:
    """Average over all angular coordinates."""
    return d3.Average(a, coords.S2basis.coordsys)


def vol_avg(a: d3.Field) -> d3.Field:
    """Average over whole basis.sphere."""
    return d3.Integrate(a / volume, coords)


# define every component of velocity (for output)
u_n_r = dot(u_cr, er_crust)
u_n_theta = dot(u_cr, etheta_crust)
u_n_phi = dot(u_cr, ephi_crust)

save_path = Path("outputs/{}/su_equator".format(PARAMS["output_dir"]))
save_path.mkdir(parents=True, exist_ok=True)

AZ_avg = solver.evaluator.add_file_handler(
    "outputs/{}/su_equator/AZ_avg_equator".format(PARAMS["output_dir"]),
    sim_dt=0.05,
    max_writes=100,
)
AZ_avg.add_task(dot(er_crust, u_cr), name="u_n_r")
AZ_avg.add_task(dot(etheta_crust, u_cr), name="u_n_theta")
AZ_avg.add_task(az_avg(u_n_phi), name="u_n_phi")
AZ_avg.add_task(az_avg(dot(ephi_crust, u_cr)), name="u_s_phi")

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
CFL.add_velocity(u_cr)


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_cr @ u_cr) * PARAMS["Ek"], name="Re_n")


# Main loop
@profile(args["profile"], PARAMS)
def evolve(solver: d3core.solvers.InitialValueSolver) -> None:
    """Define a function to call the dedalus evolve method with profiling."""
    return solver.evolve(timestep_function=CFL.compute_timestep, log_cadence=10)


evolve(solver)
