import logging

import dedalus.public as d3
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from gains.params.single_spin_up_rotating import parameters as default_params
from gains.problems.bases import ShellBasis, SphericalBasis
from gains.utils.misc import mesh_cpus
from gains.utils.parsers import SimulationCLI
from gains.plotting.polar import plot_angular
from pathlib import Path

ncpu = MPI.COMM_WORLD.size
mesh = mesh_cpus(ncpu)

logger = logging.getLogger(__name__)
logger.info(f"running on processor mesh={mesh}")
parser = SimulationCLI(
    profiling_option=True,
    place_all_outputs_under="outputs",
    sim_name="two_fluid_spin_up",
)
PARAMS = parser.parse_args_and_get_params(logger, default_params=default_params)

path = Path(PARAMS["output_dir"]) / "AZ_avg_equator_s1.h5"

dtype = np.float64

Ri = PARAMS["Ri"]
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis_core = SphericalBasis(coords, dist, dtype, Ri, **PARAMS)
basis_crust = ShellBasis(coords, dist, dtype, **PARAMS)

u_data_shell = dist.VectorField(coords, name="u_data_shell", bases=basis_crust.shell)
u_data_ball = dist.VectorField(coords, name="u_data_ball", bases=basis_core.ball)

with h5py.File(path, "r") as data:
    u_data_ball["g"][0] = data["tasks"]["u_b_s_phi"][-1:, :, :]
    u_data_ball["g"][1] = data["tasks"]["u_b_s_theta"][-1:, :, :]
    u_data_ball["g"][2] = data["tasks"]["u_b_s_r"][-1:, :, :]

    u_data_shell["g"][0] = data["tasks"]["u_s_s_phi"][-1:, :, :]
    u_data_shell["g"][1] = data["tasks"]["u_s_s_theta"][-1:, :, :]
    u_data_shell["g"][2] = data["tasks"]["u_s_s_r"][-1:, :, :]
    u_s_r = data["tasks"]["u_s_s_r"]
    r = u_s_r.dims[3][0][:].ravel()
    theta = u_s_r.dims[2][0][:].ravel()
    phi = u_s_r.dims[1][0][:].ravel()
vorticity_shell = d3.Curl(u_data_shell)
vorticity_ball = d3.Curl(u_data_ball)

vmag_shell = np.sqrt(vorticity_shell @ vorticity_shell).evaluate()
vmag_ball = vorticity_ball @ vorticity_ball

print(np.max(vmag_shell["g"]))
print(np.shape(vmag_shell["g"]))

flat_idx = np.argmax(vmag_shell["g"])
coords = np.unravel_index(flat_idx, np.shape(vmag_shell["g"]))
print(f"Index of blow up: {coords}")
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
vmag_shell.change_scales(1)

mesh_plot = plot_angular(ax, r, theta, vmag_shell["g"][coords[0]], omega_back=None, **PARAMS)

