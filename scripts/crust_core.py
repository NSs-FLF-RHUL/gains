"""
Solve Navier Stokes equations for 2 fluids coupled across a boundary.

Both fluids have 0 penetration at the boundary. The crust fluid has 0 shear stress
at the boundary, and the fluids match angular speed at the crust-core interface.
The surface of the crust is spun up.

Each region is nondimensionalized using its own characteristic length scale, leading to
distinct effective Ekman numbers in the core and shell.
"""

import datetime
import json
import logging
from collections.abc import Callable
from pathlib import Path

import dedalus.public as d3
import numpy as np
from dedalus.public import DotProduct as Dot
from mpi4py import MPI

from gains.params.single_spin_up_rotating import parameters as default_params
from gains.problems.bases import ShellBasis, SphericalBasis
from gains.utils.loggers import track_reynolds_n
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import create_parser_simulation
from gains.utils.profile import add_profiling_options, profile

# Setup
logger = logging.getLogger(__name__)

parser = create_parser_simulation()
add_profiling_options(parser)
args = vars(parser.parse_args())

if args["logfile"] is not None:
    logpath = Path(f"outputs/{args['output_dir']}/{args['logfile']}.txt")
    logpath.parent.mkdir(exist_ok=True)
    FileOutputHandler = logging.FileHandler(logpath)
    logger.addHandler(FileOutputHandler)


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

timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek_shell = PARAMS["Ek"] * (PARAMS["Ro"] - PARAMS["Ri"]) ** 2
Ek_ball = PARAMS["Ek"] * PARAMS["Ri"] ** 2
B = PARAMS["B"]
Bprime = B / 2
Ri = PARAMS["Ri"]
Ro = PARAMS["Ro"]
radius = Ro

mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis_core = SphericalBasis(coords, dist, dtype, Ri, **PARAMS)
basis_crust = ShellBasis(coords, dist, dtype, **PARAMS)

# Fields
u_b = dist.VectorField(coords, name="u_b", bases=basis_core.ball)
p_b = dist.Field(name="p_b", bases=basis_core.ball)
tau_p_b = dist.Field(name="tau_p_b")
tau_u_b_1 = dist.VectorField(coords, name="tau_u_b_1", bases=basis_core.sphere)
tau_u_b_2 = dist.VectorField(coords, name="tau_u_b_2", bases=basis_core.sphere)

u_s = dist.VectorField(coords, name="u_s", bases=basis_crust.shell)
p_s = dist.Field(name="p_s", bases=basis_crust.shell)
tau_p_s = dist.Field(name="tau_p_s")
tau_u_s_1 = dist.VectorField(coords, name="tau_u_s_1", bases=basis_crust.surface)
tau_u_s_2 = dist.VectorField(coords, name="tau_u_s_2", bases=basis_crust.surface)

# Substitutions - general
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)

er["g"][2] = 1
etheta["g"][1] = 1
ephi["g"][0] = 1

# Subsititutions for crust
lift_basis_s = basis_crust.shell.derivative_basis(1)
lift_s = lambda a: d3.Lift(a, lift_basis_s, -1)
phi_s, theta_s, r_s = dist.local_grids(basis_crust.shell)
ez_s = dist.VectorField(coords, bases=basis_crust.shell)
ez_s["g"][1] = -np.sin(theta_s)
ez_s["g"][2] = np.cos(theta_s)

rvec_s = dist.VectorField(coords, bases=basis_crust.shell.radial_basis)
rvec_s["g"][2] = r_s

grad_u_s = d3.grad(u_s) + rvec_s * lift_s(tau_u_s_1)
stheta_s = dist.Field(name="stheta", bases=basis_crust.shell)
stheta_s["g"] = np.sin(theta_s)
uang_s = dist.VectorField(coords, bases=basis_crust.shell)(r=radius).evaluate()
uang_s["g"][0, :] = (PARAMS["Delta_Omega"] * stheta_s)(r=radius).evaluate()["g"]

strain_s = grad_u_s + d3.trans(grad_u_s)
shear_stress_s_surface = d3.angular(d3.radial(strain_s(r=PARAMS["Ro"]), index=1))

shear_stress_s_interface = d3.angular(d3.radial(strain_s(r=PARAMS["Ri"]), index=1))

# Subsititutions for Core
lift_b = lambda a: d3.Lift(a, basis_core.ball, -1)
phi_b, theta_b, r_b = dist.local_grids(basis_core.ball)

ez_b = dist.VectorField(coords, bases=basis_core.ball)
ez_b["g"][1] = -np.sin(theta_b)
ez_b["g"][2] = np.cos(theta_b)

rvec_b = dist.VectorField(coords, bases=basis_core.ball.radial_basis)
rvec_b["g"][2] = r_b

grad_u_b = d3.grad(u_b) + rvec_b * lift_b(tau_u_b_1)

strain_b = grad_u_b + d3.trans(grad_u_b)
shear_stress_b_interface = d3.angular(d3.radial(strain_b(r=PARAMS["Ri"]), index=1))

# Problem
problem = d3.IVP(
    [u_s, p_s, tau_p_s, tau_u_s_1, tau_u_s_2, u_b, p_b, tau_p_b, tau_u_b_2],
    namespace=locals(),
)

problem.add_equation("trace(grad_u_s) + tau_p_s = 0")
problem.add_equation("div(u_b) + tau_p_b = 0")
problem.add_equation("integ(p_b) = 0")
problem.add_equation("integ(p_s) = 0")

problem.add_equation(
    "dt(u_b) - Ek_ball*lap(u_b) + grad(p_b) + lift_b(tau_u_b_2) = -u_b@grad(u_b) "
    "- 2*cross(ez_b, u_b)"
)
problem.add_equation(
    "dt(u_s) - Ek_shell*div(grad_u_s) + grad(p_s) + lift_s(tau_u_s_2) = -u_s@grad(u_s) "
    "- 2*cross(ez_s, u_s)"
)

problem.add_equation("radial(u_s(r=Ro)) = 0")  # Zero Penetration
problem.add_equation("angular(u_s(r=Ro)) = angular(uang_s)")  # Surface spin up

problem.add_equation("radial(u_s(r=Ri)) = 0")
problem.add_equation("shear_stress_s_interface = 0")

problem.add_equation("radial(u_b(r=Ri)) = 0")
problem.add_equation("angular(u_b(r=Ri)) = angular(u_s(r=Ri))")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
else:
    # Initial condition
    u_s.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_s.low_pass_filter(scales=0.5)
    u_b.fill_random("g", seed=67, distribution="normal", scale=1e-10)  # Random noise
    u_b.low_pass_filter(scales=0.5)
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


u_b_r = Dot(u_b, er)
u_b_theta = Dot(u_b, etheta)
u_b_phi = Dot(u_b, ephi)

u_s_r = Dot(u_s, er)
u_s_theta = Dot(u_s, etheta)
u_s_phi = Dot(u_s, ephi)

save_path = Path("outputs/{}/su_equator".format(PARAMS["output_dir"]))
save_path.mkdir(parents=True, exist_ok=True)

AZ_avg = solver.evaluator.add_file_handler(
    "outputs/{}/su_equator/AZ_avg_equator".format(PARAMS["output_dir"]),
    sim_dt=0.05,
    max_writes=100,
)
AZ_avg.add_task(az_avg(u_b_r), name="u_b_r")
AZ_avg.add_task(az_avg(u_b_theta), name="u_b_theta")
AZ_avg.add_task(az_avg(u_b_phi), name="u_b_phi")

AZ_avg.add_task(az_avg(u_s_r), name="u_s_r")
AZ_avg.add_task(az_avg(u_s_theta), name="u_s_theta")
AZ_avg.add_task(az_avg(u_s_phi), name="u_s_phi")

CFL = d3.CFL(
    solver, timestep, cadence=1, safety=0.5, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(u_b)
CFL.add_velocity(u_s)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_s @ u_s) * PARAMS["Ek"], name="Re_n")


@profile(args["profile"])
def main() -> Callable:
    """Create main loop with profiling."""
    return track_reynolds_n(logger, flow, solver, CFL)


main()
