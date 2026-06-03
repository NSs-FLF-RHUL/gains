"""
Solve the HVBK equations for a spherical shell subject to a boundary spin up.

Equations and mutual friction are in the same form as
J. R. Fuentes and Vanessa Graber 2024 ApJ 974 300.
"""

import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np
from mpi4py import MPI

from gains.params.spherical_shell import parameters as default_params
from gains.problems.bases import ShellBasis
from gains.utils.loggers import track_vorticity
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import SimulationCLI
from gains.utils.profile import profile

# Setup
logger = logging.getLogger(__name__)
parser = SimulationCLI(
    profiling_option=True,
    place_all_outputs_under="outputs",
    sim_name="two_fluid_spin_up",
)
PARAMS = parser.parse_args(logger, default_params=default_params)

radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek = PARAMS["Ek"]
B = PARAMS["B"]
Bprime = B / 2
Ri = PARAMS["Ri"]
Ro = PARAMS["Ro"]
mesh = mesh_cpus(ncpu)
nu_hyper = 1e-6

x_s = 0.95  # Neutron fraction
x_n = 0.05  # Proton/electron fraction

logger.info(f"running on processor mesh={mesh}")

# Basis

coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = ShellBasis(coords, dist, dtype, **PARAMS)
basis_shell = basis.shell
surface = basis.surface

# Crust fields

u_n_cr = dist.VectorField(coords, name="u_n_cr", bases=basis_shell)
p_n_cr = dist.Field(name="p_n_cr", bases=basis_shell)
tau_n_pcr = dist.Field(name="tau_n_pcr")
tau_uncr_1 = dist.VectorField(coords, name="tau_nucr_1", bases=surface)
tau_uncr_2 = dist.VectorField(coords, name="tau_nucr_2", bases=surface)

u_s_cr = dist.VectorField(coords, name="u_s_cr", bases=basis_shell)
p_s_cr = dist.Field(name="p_s_cr", bases=basis_shell)
tau_s_pcr = dist.Field(name="tau_s_pcr")
tau_uscr_1 = dist.VectorField(coords, name="tau_sucr_1", bases=surface)
tau_uscr_2 = dist.VectorField(coords, name="tau_sucr_2", bases=surface)

# Crust substitutions
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
rvec["g"][2] = r_crust

grad_uncr = d3.grad(u_n_cr) + rvec * lift_crust(tau_uncr_1)
grad_uscr = d3.grad(u_s_cr) + rvec * lift_crust(tau_uscr_1)

sintheta = dist.Field(name="sintheta", bases=basis_shell)
sintheta["g"] = np.sin(theta_crust)
uang = dist.VectorField(coords, bases=basis_shell)(r=radius).evaluate()
uang["g"][0, :] = (PARAMS["Delta_Omega"] * sintheta)(r=radius).evaluate()["g"]
omega_s = dist.VectorField(coords, name="omega_s", bases=basis_shell)

omega_s_r = dot(omega_s, er_crust)
omega_s_theta = dot(omega_s, etheta_crust)
omega_s_phi = dot(omega_s, ephi_crust)

u_ns = u_n_cr - u_s_cr
omega_s = curl(u_s_cr) + 2 * ez_crust
omega_unit = omega_s / 2
F_mf = B * (cross(omega_unit, cross(omega_s, u_ns))) + Bprime * cross(omega_s, u_ns)

strain_rate_n_cr = grad_uncr + d3.trans(grad_uncr)
shear_stress_n_cr = d3.angular(d3.radial(strain_rate_n_cr(r=PARAMS["Ri"]), index=1))

strain_rate_s_cr = grad_uscr + d3.trans(grad_uscr)
shear_stress_s_cr_i = d3.angular(d3.radial(strain_rate_s_cr(r=PARAMS["Ri"]), index=1))
shear_stress_s_cr_o = d3.angular(d3.radial(strain_rate_s_cr(r=PARAMS["Ro"]), index=1))

# Problem for crust (testing)

problem = d3.IVP(
    [
        u_n_cr,
        p_n_cr,
        tau_n_pcr,
        tau_uncr_1,
        tau_uncr_2,
        u_s_cr,
        p_s_cr,
        tau_s_pcr,
        tau_uscr_1,
        tau_uscr_2,
    ],
    namespace=locals(),
)
problem.add_equation("trace(grad_uncr) + tau_n_pcr = 0")
problem.add_equation("trace(grad_uscr) + tau_s_pcr = 0")
problem.add_equation("integ(p_n_cr) = 0")
problem.add_equation("integ(p_s_cr) = 0")

problem.add_equation(
    "dt(u_n_cr) - Ek*div(grad_uncr) + grad(p_n_cr) "
    "+ lift_crust(tau_uncr_2) = - u_n_cr@grad(u_n_cr) -"
    " 2*cross(ez_crust,u_n_cr) + x_s/x_n * F_mf"
)

problem.add_equation(
    "dt(u_s_cr) + grad(p_s_cr) + lift_crust(tau_uscr_2) = "
    "-u_s_cr@grad(u_s_cr) - F_mf - 2*cross(ez_crust, u_s_cr)"
)

problem.add_equation("radial(u_n_cr(r=Ro)) = 0")
problem.add_equation("angular(u_n_cr(r=Ro)) = angular(uang)")
problem.add_equation("radial(u_s_cr(r=Ro)) = 0")
problem.add_equation("shear_stress_s_cr_o = 0")

problem.add_equation("radial(u_n_cr(r=Ri)) = 0")
problem.add_equation("shear_stress_n_cr = 0")
problem.add_equation("radial(u_s_cr(r=Ri)) = 0")
problem.add_equation("shear_stress_s_cr_i = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]

if PARAMS["use_checkpoint"]:
    write, timestep = solver.load_state(PARAMS["checkpoint_path"])
else:
    # Initial condition
    u_n_cr.fill_random("g", seed=42, distribution="normal", scale=1e-10)  # Random noise
    u_n_cr.low_pass_filter(scales=0.5)
    u_s_cr.fill_random("g", seed=67, distribution="normal", scale=1e-10)  # Random noise
    u_s_cr.low_pass_filter(scales=0.5)
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
u_n_r = dot(u_n_cr, er_crust)
u_n_theta = dot(u_n_cr, etheta_crust)
u_n_phi = dot(u_n_cr, ephi_crust)

save_path: Path = PARAMS["output_dir"] / "su_equator"
save_path.mkdir(parents=True, exist_ok=True)

AZ_avg = solver.evaluator.add_file_handler(
    str(save_path / "AZ_avg_equator"),
    sim_dt=0.05,
    max_writes=100,
)
AZ_avg.add_task(dot(er_crust, u_s_cr), name="u_n_r")
AZ_avg.add_task(dot(etheta_crust, u_s_cr), name="u_n_theta")
AZ_avg.add_task(az_avg(dot(ephi_crust, u_n_cr)), name="u_n_phi")
AZ_avg.add_task(az_avg(dot(ephi_crust, u_s_cr)), name="u_s_phi")

slices = solver.evaluator.add_file_handler(
    str(save_path / "slices"),
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
    solver, timestep, cadence=1, safety=0.5, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(u_n_cr)
CFL.add_velocity(u_s_cr)


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u_n_cr @ u_n_cr) * PARAMS["Ek"], name="Re_n")
flow.add_property(np.sqrt(omega_s @ omega_s), name="vorticity_mag")


# Main loop
@profile(PARAMS["profile"], PARAMS)
def main_loop() -> None:
    """Decorate main loop."""
    return track_vorticity(logger, flow, solver, CFL)


main_loop()
