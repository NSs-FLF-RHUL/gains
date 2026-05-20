"""
Solve the HVBK equations for a spherical star subject to a boundary spin up.

Equations and mutual friction are in the same form as
J. R. Fuentes and Vanessa Graber 2024 ApJ 974 300.
"""

import datetime
import json
import logging
from pathlib import Path

import dedalus.public as d3
import numpy as np
from dedalus.public import CrossProduct as Cross
from dedalus.public import Curl
from dedalus.public import DotProduct as Dot
from mpi4py import MPI

from gains.params.single_spin_up_rotating import parameters as default_params
from dedalus.public import CrossProduct as Cross
from dedalus.public import DotProduct as Dot
from gains.problems.bases import SphericalBasis, ShellBasis
from gains.utils.loggers import track_vorticity
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

radius = 1
timestepper = d3.SBDF2
cfl_safety = 0.2
max_timestep = 1e-2
dtype = np.float64
ncpu = MPI.COMM_WORLD.size

Ek = PARAMS["Ek"]
B = PARAMS["B"]
Bprime = B / 2

mesh = mesh_cpus(ncpu)

logger.info(f"running on processor mesh={mesh}")

# Basis
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis_core = SphericalBasis(coords, dist, dtype, radius, **PARAMS)
basis_crust = ShellBasis(coords, dist, dtype, **PARAMS)

# Fields
u_b = dist.VectorField(coords, name = "u_b", bases=basis_core.ball)
p_b = dist.Field(name="p_b", bases = basis_core.ball)
tau_p_b = dist.Field(name="tau_p_b", bases=basis_core.ball)
tau_u_b = dist.VectorField(coords, name="tau_u_b", bases=basis_core.sphere)

u_s = dist.VectorField(coords, name = "u_s", bases = basis_crust.shell)
p_s = dist.Field(name = "p_s", bases=basis_crust.shell)
tau_p_s = dist.Field(name = "tau_p_s", bases = basis_crust.shell)
tau_u_s_1 = dist.VectorField(coords, name = "tau_u_s_1", bases = basis_crust.surface)
tau_u_s_2 = dist.VectorField(coords, name = "tau_u_s_2", bases = basis_crust.surface)

#Substitutions - general
er = dist.VectorField(coords)
etheta = dist.VectorField(coords)
ephi = dist.VectorField(coords)

er['g'][2] = 1
etheta['g'][1] = 1
ephi['g'][0] = 1

#Subsititutions: crust
lift_basis_s = basis_crust.shell.derivative_basis(1)
lift_s = lambda a: d3.Lift(a, lift_s, -1)
phi_s, theta_s, r_s = dist.local_grids(basis_crust.shell)
ez_s = dist.VectorField(coords, bases=basis_crust.shell)
ez_s['g'][1] = - np.sin(theta_s)
ez_s['g'][2] = np.cos(theta_s)

rvec = dist.VectorField(coords, bases = basis_crust.shell.radial_basis)
rvec['g'][2] = r_s

grad_u_s = d3.grad(u_s) + rvec * lift_s(tau_u_s_1)
stheta_s = d3.Field(name="stheta", bases=basis_crust.shell)
stheta_s['g'] = np.sin(theta_s)
uang_s = dist.VectorField(coords, bases=basis_crust.shell)(r=radius).evaluate()
uang_s["g"][0, :] = (PARAMS["Delta_Omega"] * stheta_s)(r=radius).evaluate()["g"]

strain_s_surface = grad_u_s + d3.trans(grad_u_s)
shear_stress_s_surface = d3.angular(d3.radial(strain_s_surface(r=PARAMS["Ri"]), index=1))

#Subsititutions: Core
lift_b = lambda a: d3.Lift(a, basis_core.ball, -1)
phi_b, theta_b, r_b = dist.local_grids(basis_core.ball)

ez_b = dist.VectorField(coords, bases=basis_core.ball)
ez_b['g'][1] = -np.sin(theta_b)
ez_b['g'][2] = np.cos(theta_b)


