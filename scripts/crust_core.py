"""
Solve HVBK equations for a resolved core and crust.

The superfluids both satisfy 0 shear stress and 0 penetration across
the interface, and the crust superfluid also satisfies these conditions
at the surface.

The normal fluids both also are subject to 0 penetration at the interface.
The angular velocities of the normal fluids are matched at the interface.
The normal fluid in the crust is subject to a spin up at the surface.

Each region is nondimensionalized using its own characteristic length scale, leading to
distinct effective Ekman numbers in the core and shell.

The mutual friction is in the same form as
J. R. Fuentes and Vanessa Graber 2024 ApJ 974 300.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import dedalus.public as d3
import numpy as np
from dedalus.public import CrossProduct as Cross
from dedalus.public import Curl
from dedalus.public import DotProduct as Dot
from mpi4py import MPI

from gains.params.single_spin_up_rotating import parameters as default_params
from gains.problems.bases import ShellBasis, SphericalBasis
from gains.utils.loggers import track_vorticity
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import SimulationCLI
from gains.utils.profile import profile
from gains.initial_conditions.single_component_spin_up import mask_r, circle_on_sphere

# Setup
logger = logging.getLogger(__name__)

parser = SimulationCLI(
    profiling_option=True,
    place_all_outputs_under="outputs",
    sim_name="two_fluid_spin_up",
)
PARAMS = parser.parse_args_and_get_params(logger, default_params=default_params)

timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek_shell = PARAMS["Ek_crust"]
Ek_ball = PARAMS["Ek_core"]
B = PARAMS["B"]
Bprime = PARAMS["B"] / 2
Ri = PARAMS["Ri"]
Ro = PARAMS["Ro"]
radius = Ro
x_b_n = 0.05  # Proton fraction - core
x_b_s = 0.95  # Neutron fraction - core
x_s_n = 0.05  # Electron fraction - crust
x_s_s = 0.95  # Neutron fraction - crust

PARAMS["x_b_n"] = 0.05 #Added to params for saving purposes
PARAMS["x_b_s"] = 0.95
PARAMS["x_s_n"] = 0.05
PARAMS["x_s_s"] = 0.95

mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis_core = SphericalBasis(coords, dist, dtype, Ri, **PARAMS)
basis_crust = ShellBasis(coords, dist, dtype, **PARAMS)

# Fields - naming is field_basis_fluid
# Core (ball basis)
u_b_n = dist.VectorField(coords, name="u_b_n", bases=basis_core.ball)
p_b_n = dist.Field(name="p_b_n", bases=basis_core.ball)
u_b_s = dist.VectorField(coords, name="u_b_s", bases=basis_core.ball)
p_b_s = dist.Field(name="p_b_s", bases=basis_core.ball)

tau_p_b_n = dist.Field(name="tau_p_b")
tau_u_b_n_1 = dist.VectorField(coords, name="tau_u_b_n_1", bases=basis_core.sphere)
tau_u_b_n_2 = dist.VectorField(coords, name="tau_u_b_n_2", bases=basis_core.sphere)

tau_p_b_s = dist.Field(name="tau_p_b_s")
tau_u_b_s_1 = dist.VectorField(coords, name="tau_u_b_s_1", bases=basis_core.sphere)
tau_u_b_s_2 = dist.VectorField(coords, name="tau_u_b_s_2", bases=basis_core.sphere)

# Crust (shell basis)
u_s_n = dist.VectorField(coords, name="u_s_n", bases=basis_crust.shell)
p_s_n = dist.Field(name="p_s_n", bases=basis_crust.shell)
u_s_s = dist.VectorField(coords, name="u_s_s", bases=basis_crust.shell)
p_s_s = dist.Field(name="p_s_s", bases=basis_crust.shell)

tau_p_s_n = dist.Field(name="tau_p_s_n")
tau_u_s_n_1 = dist.VectorField(coords, name="tau_u_s_n_1", bases=basis_crust.surface)
tau_u_s_n_2 = dist.VectorField(coords, name="tau_u_s_n_2", bases=basis_crust.surface)

tau_p_s_s = dist.Field(name="tau_p_s_s")
tau_u_s_s_1 = dist.VectorField(coords, name="tau_u_s_s_1", bases=basis_crust.surface)
tau_u_s_s_2 = dist.VectorField(coords, name="tau_u_s_s_2", bases=basis_crust.surface)

mask_radial = dist.Field(name="mask_radial", bases=basis_crust.shell)
mask_circ = dist.Field(name="mask_circ", bases=basis_crust.shell)

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

u_target = dist.VectorField(coords, name="u_target", bases=basis_crust.shell)
u_target["g"][0] = PARAMS["Delta_Omega"] * r_s * np.sin(theta_s)

mask_radial["g"] = mask_r(r_s, PARAMS["width_r"])
mask_circ["g"] = circle_on_sphere(theta_s, phi_s, PARAMS["radius_glitch"], (PARAMS["center_theta"], PARAMS["center_phi"]), 0.5)
rvec_s = dist.VectorField(coords, bases=basis_crust.shell.radial_basis)
rvec_s["g"][2] = r_s

grad_u_s_n = d3.grad(u_s_n) + rvec_s * lift_s(tau_u_s_n_1)
grad_u_s_s = d3.grad(u_s_s) + rvec_s * lift_s(tau_u_s_s_1)

stheta_s = dist.Field(name="stheta", bases=basis_crust.shell)
stheta_s["g"] = np.sin(theta_s)
uang_s = dist.VectorField(coords, bases=basis_crust.shell)(r=radius).evaluate()
uang_s["g"][0, :] = (PARAMS["Delta_Omega"] * stheta_s)(r=radius).evaluate()["g"]

strain_s_n = grad_u_s_n + d3.trans(grad_u_s_n)
strain_s_s = grad_u_s_s + d3.trans(grad_u_s_s)
shear_stress_s_n_surface = d3.angular(d3.radial(strain_s_n(r=PARAMS["Ro"]), index=1))

shear_stress_s_n_interface = d3.angular(d3.radial(strain_s_n(r=PARAMS["Ri"]), index=1))

shear_stress_s_s_interface = d3.angular(d3.radial(strain_s_s(r=PARAMS["Ri"]), index=1))
shear_stress_s_s_surface = d3.angular(d3.radial(strain_s_s(r=PARAMS["Ro"]), index=1))

omega_s_s = dist.VectorField(
    coords, name="omega_s_s", bases=basis_core.ball
)  # Superfluid vorticity

u_s_ns = u_s_n - u_s_s
omega_s_s = Curl(u_s_s) + 2 * ez_s
omega_unit_s = omega_s_s / 2  # Numerically unstable if fully normalised
F_mf_s = B * (Cross(omega_unit_s, Cross(omega_s_s, u_s_ns))) + Bprime * Cross(
    omega_s_s, u_s_ns
)

# Subsititutions for Core
lift_b = lambda a: d3.Lift(a, basis_core.ball, -1)
phi_b, theta_b, r_b = dist.local_grids(basis_core.ball)

ez_b = dist.VectorField(coords, bases=basis_core.ball)
ez_b["g"][1] = -np.sin(theta_b)
ez_b["g"][2] = np.cos(theta_b)


strain_b_s = d3.grad(u_b_s) + d3.trans(d3.grad(u_b_s))
strain_b_n = d3.grad(u_b_n) + d3.trans(d3.grad(u_b_n))

shear_stress_b_s_interface = d3.angular(d3.radial(strain_b_s(r=PARAMS["Ri"]), index=1))

omega_b_s = dist.VectorField(
    coords, name="omega_b_s", bases=basis_core.ball
)  # Superfluid vorticity

u_b_ns = u_b_n - u_b_s
omega_b_s = Curl(u_b_s) + 2 * ez_b
omega_unit_b = omega_b_s / 2
F_mf_b = B * (Cross(omega_unit_b, Cross(omega_b_s, u_b_ns))) + Bprime * Cross(
    omega_b_s, u_b_ns
)

# Problem
problem = d3.IVP(
    [
        u_s_n,
        p_s_n,
        tau_p_s_n,
        tau_u_s_n_1,
        tau_u_s_n_2,
        u_s_s,
        p_s_s,
        tau_p_s_s,
        tau_u_s_s_1,
        tau_u_s_s_2,
        u_b_n,
        p_b_n,
        tau_p_b_n,
        tau_u_b_n_2,
        u_b_s,
        p_b_s,
        tau_p_b_s,
        tau_u_b_s_2,
    ],
    namespace=locals(),
)

problem.add_equation("trace(grad_u_s_n) + tau_p_s_n = 0")
problem.add_equation("trace(grad_u_s_s) + tau_p_s_s = 0")
problem.add_equation("div(u_b_n) + tau_p_b_n = 0")
problem.add_equation("div(u_b_s) + tau_p_b_s = 0")
problem.add_equation("integ(p_b_n) = 0")
problem.add_equation("integ(p_s_n) = 0")
problem.add_equation("integ(p_b_s) = 0")
problem.add_equation("integ(p_s_s) = 0")

# Crust momentum equations equations
problem.add_equation(
    "dt(u_s_n) - Ek_shell*div(grad_u_s_n) + grad(p_s_n) + lift_s(tau_u_s_n_2) = -u_s_n@grad(u_s_n) - 2*cross(ez_s, u_s_n) + x_s_s/x_s_n * F_mf_s"
)
problem.add_equation(
    "dt(u_s_s) + grad(p_s_s) + lift_s(tau_u_s_s_2) = -u_s_s@grad(u_s_s) -2*cross(ez_s, u_s_s) - F_mf_s + mask_radial*mask_circ*(u_target - u_s_s)"
)

# Core momentum equations
problem.add_equation(
    "dt(u_b_n) - Ek_ball*lap(u_b_n) + grad(p_b_n) + lift_b(tau_u_b_n_2) = -u_b_n@grad(u_b_n) - 2*cross(ez_b, u_b_n) + x_b_s/x_b_n * F_mf_b"
)
problem.add_equation(
    "dt(u_b_s) + grad(p_b_s) + lift_b(tau_u_b_s_2) = - u_b_s@grad(u_b_s) - 2*cross(ez_b, u_b_s) - F_mf_b"
)

# Surface boundary conditions
problem.add_equation("radial(u_s_n(r=Ro)) = 0")  # No penetration, normal fluid
problem.add_equation("shear_stress_s_n_surface = 0")  # Stress free, normal fluid

problem.add_equation("radial(u_s_s(r=Ro)) = 0")  # No penetration, superfluid
problem.add_equation("shear_stress_s_s_surface = 0")  # Stress free, superfluid

# Iterface boundary conditions, crust side
problem.add_equation("radial(u_s_n(r=Ri)) = 0")  # No penetration, normal fluid
problem.add_equation("shear_stress_s_n_interface = 0")  # Stress free, normal fluid

problem.add_equation("radial(u_s_s(r=Ri)) = 0")  # No penetration, superfluid
problem.add_equation("shear_stress_s_s_interface = 0")  # Stress free, superfluid

# Interface boundary condition, core side
problem.add_equation("radial(u_b_n(r=Ri)) = 0")  # No penetration, normal fluid
problem.add_equation(
    "angular(u_b_n(r=Ri)) = angular(u_s_n(r=Ri))"
)  # Tangential velocity conservation, normal fluid

problem.add_equation("radial(u_b_s(r=Ri)) = 0")  # No penetration, superfluid
problem.add_equation("shear_stress_b_s_interface = 0")  # Stress free, superfluid

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
else:
    # Initial condition
    u_s_n.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_s_n.low_pass_filter(scales=0.5)
    u_b_n.fill_random("g", seed=67, distribution="normal", scale=1e-10)  # Random noise
    u_b_n.low_pass_filter(scales=0.5)
    u_b_s.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_b_s.low_pass_filter(scales=0.5)
    u_s_s.fill_random("g", seed=67, distribution="normal", scale=1e-10)  # Random noise
    u_s_s.low_pass_filter(scales=0.5)
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


u_b_n_r = Dot(u_b_n, er)
u_b_n_theta = Dot(u_b_n, etheta)
u_b_n_phi = Dot(u_b_n, ephi)

u_s_n_r = Dot(u_s_n, er)
u_s_n_theta = Dot(u_s_n, etheta)
u_s_n_phi = Dot(u_s_n, ephi)

u_b_s_r = Dot(u_b_s, er)
u_b_s_theta = Dot(u_b_s, etheta)
u_b_s_phi = Dot(u_b_s, ephi)

u_s_s_r = Dot(u_s_s, er)
u_s_s_theta = Dot(u_s_s, etheta)
u_s_s_phi = Dot(u_s_s, ephi)

save_path: Path = PARAMS["output_dir"] / "su_equator"
save_path.mkdir(parents=True, exist_ok=True)

#Save velocity fields

u_fields = solver.evaluator.add_file_handler(
    str(save_path / "AZ_avg_equator"),
    sim_dt=PARAMS["snapshot_dt"],
    max_writes=100,
    mode='append'
)
u_fields.add_task(u_b_n_r, name="u_b_n_r")
u_fields.add_task(u_b_n_theta, name="u_b_n_theta")
u_fields.add_task(u_b_n_phi, name="u_b_n_phi")

u_fields.add_task(u_s_n_r, name="u_s_n_r")
u_fields.add_task(u_s_n_theta, name="u_s_n_theta")
u_fields.add_task(u_s_n_phi, name="u_s_n_phi")

u_fields.add_task(u_b_s_r, name="u_b_s_r")
u_fields.add_task(u_b_s_theta, name="u_b_s_theta")
u_fields.add_task(u_b_s_phi, name="u_b_s_phi")

u_fields.add_task(u_s_s_r, name="u_s_s_r")
u_fields.add_task(u_s_s_theta, name="u_s_s_theta")
u_fields.add_task(u_s_s_phi, name="u_s_s_phi")

#Checkpoint

checkpoint = solver.evaluator.add_file_handler(PARAMS["output_dir"] / "checkpoint", wall_dt=PARAMS["checkpoint_cadence"], max_writes=1, parallel='gather')
checkpoint.add_tasks(solver.state, layout='g')


CFL = d3.CFL(
    solver, timestep, cadence=1, safety=0.5, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(u_b_n)
CFL.add_velocity(u_s_n)
CFL.add_velocity(u_b_s)
CFL.add_velocity(u_s_s)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_s_n @ u_s_n) * PARAMS["Ek_crust"], name="Re_n")
flow.add_property(np.sqrt(omega_b_s @ omega_b_s), name="vorticity_mag")


@profile(PARAMS["profile"], PARAMS["output_dir"])
def main() -> Callable:
    """Create main loop with profiling."""
    return track_vorticity(logger, flow, solver, CFL, PARAMS)

main()
