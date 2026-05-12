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
Ri = 0.9
Ro = 1.0
mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis

coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)

basis_full = SphericalBasis(coords, dist, Ro, dtype, **PARAMS)
basis_core = SphericalBasis(coords, dist, Ri, dtype, **PARAMS)
basis_crust = d3.ShellBasis(basis_full.coords,
                            shape=(PARAMS["Nphi"], PARAMS["Ntheta"], PARAMS["Nr"]),
                            radii = (Ri, Ro),
                            dealias=PARAMS["dealias"],
                            dtype=dtype
                            )
sphere = basis_crust.outer_surface

#Fields
u_crust = dist.VectorField(basis_full.coords, name="u_crust", bases=basis_crust)
p_crust = dist.Field(name = "p_crust", bases=basis_crust)
tau_ucr_1 = dist.VectorField(basis_full.coords, name = "tau_ucr_1", bases = sphere)
tau_ucr_2 = dist.VectorField(basis_full.coords,name = "tau_ucr_2", bases = sphere)
tau_pcr = dist.Field(name="tau_pcr")

u_core = dist.VectorField(basis_full.coords, name="u_core", bases=basis_core.ball)
p_core = dist.Field(name = "p_core", bases=basis_core.ball)
tau_uco_1 = dist.VectorField(basis_full.coords, name = "tau_uco_1", bases = basis_core.sphere)
tau_uco_2 = dist.VectorField(basis_full.coords, name = "tau_uco_2", bases = basis_core.sphere)
tau_pco = dist.Field(name="tau_pco")

phi, theta, r = dist.local_grids(basis_full.ball)
er = dist.VectorField(basis_full.coords)
etheta = dist.VectorField(basis_full.coords)
ephi = dist.VectorField(basis_full.coords)

er["g"][2] = 1
etheta["g"][1] = 1
ephi["g"][0] = 1

ez_core = dist.VectorField(basis_full.coords, bases=basis_core.ball)
ez_core["g"][1] = -np.sin(theta)
ez_core["g"][2] = np.cos(theta)  # unit vector in z direction

ez_crust = dist.VectorField(basis_full.coords, bases=basis_crust)
ez_crust["g"][1] = -np.sin(theta)
ez_crust["g"][2] = np.cos(theta)

sintheta = dist.Field(name="sintheta", bases=basis_full.ball)
sintheta["g"] = np.sin(theta)
uang = dist.VectorField(basis_full.coords, bases=basis_crust)(r=Ro).evaluate()
uang["g"][0, :] = (PARAMS["Delta_Omega"] * sintheta)(r=Ro).evaluate()["g"]

strain_rate_crust = d3.grad(u_crust) + d3.trans(d3.grad(u_crust))
shear_stress_crust = d3.angular(d3.radial(strain_rate_crust(r=Ro), index=1))
r_vec_crust = dist.VectorField(basis_full.coords, bases=basis_crust.radial_basis)
r_vec_core = dist.VectorField(basis_full.coords, bases = basis_core.ball.radial_basis)
r_vec_core['g'][2] = r
r_vec_crust['g'][2] = r
lift_basis_crust = basis_crust.derivative_basis(1)
lift_basis_core = basis_core.ball.derivative_basis(1)

lift_core = lambda a: d3.Lift(a, lift_basis_core, -1)
lift_crust = lambda a: d3.Lift(a, lift_basis_crust, -1)


grad_u_crust = d3.grad(u_crust) + r_vec_crust * lift_crust(tau_ucr_1)


dot = d3.DotProduct
curl = d3.Curl
cross = d3.CrossProduct
Ek = PARAMS["Ek"]  # Seperately defined for use in equations

#Problem

problem = d3.IVP(
    [u_crust, p_crust, tau_pcr, tau_ucr_2, u_core, p_core, tau_pco, tau_uco_1, tau_uco_2],
    namespace=locals()
)

problem.add_equation("div(u_crust) + tau_pcr = 0")
problem.add_equation("div(u_core) + tau_pco = 0")
problem.add_equation("integ(p_crust) = 0")
problem.add_equation("integ(p_core) = 0")

problem.add_equation(
    "dt(u_core) + grad(p_core) - Ek*lap(u_core) +lift_core(tau_uco_2) = -u_core@grad(u_core) - 2*cross(ez_core,u_core)"
)
problem.add_equation(
    "dt(u_crust) + grad(p_crust) - Ek*lap(u_crust) +lift_crust(tau_ucr_2) = -u_crust@grad_u_crust "
    "- 2*cross(ez_crust,u_crust)"
)

problem.add_equation("radial(u_crust(r=Ro)) = 0")
problem.add_equation("shear_stress_crust = 0")
problem.add_equation("u_crust(r=Ri) = u_core(r=Ri)")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = PARAMS["stop_sim_time"]
breakpoint()
