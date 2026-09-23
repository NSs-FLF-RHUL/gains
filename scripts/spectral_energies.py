import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
from rich.progress import Progress, TextColumn, BarColumn
from gains.utils.parsers import create_parser_analysis
from gains.utils.misc import extract_numerical_suffix
from gains.analysis.analyse_spin_up import extract_spectra
from gains.params.single_spin_up_rotating import parameters as default_params
from pathlib import Path
import json

parser = create_parser_analysis()
args = vars(parser.parse_args())
paths = sorted(
        (p for p in Path(args["output_dir"]).iterdir() if p.suffix == ".h5"), key=extract_numerical_suffix
    )

if args["parameter_file"] is not None:
        with Path.open(args["parameter_file"]) as param_file:
            PARAMS = json.load(param_file)

else:
    PARAMS = default_params

coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.ShellBasis(coords,shape=(PARAMS["Nphi"], PARAMS["Ntheta"], PARAMS["Nr"]),
                      radii=(PARAMS["Ri"], PARAMS["Ro"]),
                      dtype=np.float64,
                      dealias=1.5)

num_data = PARAMS["stop_sim_time"] / PARAMS["snapshot_dt"] # Assumes simulation ran to completion

args["fig_dir"] = Path(args["fig_dir"])
Path.mkdir(args["fig_dir"] / "l", exist_ok=True)
Path.mkdir(args["fig_dir"] / "m", exist_ok=True)
Path.mkdir(args["fig_dir"] / "uphi", exist_ok=True)

with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green"),
        ) as p:
    task = p.add_task("Plotting: ", total=num_data)
    for path in paths:

        u_r = dist.Field(name="u_r", bases=basis)
        u_theta = dist.Field(name="u_theta", bases=basis)
        u_phi = dist.Field(name="u_phi", bases=basis)

        with h5py.File(path, 'r') as f:
            times = f['tasks/u_s_n_r'].dims[0][0][:].ravel()
            u_r_series = f['tasks/u_s_n_r']
            u_theta_series = f['tasks/u_s_n_theta']
            u_phi_series = f['tasks/u_s_n_phi']
            theta = f['tasks/u_s_n_phi'].dims[2][0][:].ravel()

            increment = 0
            for time in times:
                u_r['g'] = u_r_series[increment]
                u_theta['g'] = u_theta_series[increment]
                u_phi['g'] = u_phi_series[increment]
                time = int(time)
                
                u_r.change_scales(1)
                u_theta.change_scales(1)
                u_phi.change_scales(1)
                E_m, E_l, E_n = extract_spectra(u_r, u_theta, u_phi)

                l_axis = np.arange(len(E_l))
                m_axis = np.arange(len(E_m))
                plt.loglog(l_axis[1:], E_l[1:], 'x', markersize=3.0, c='black')
                #plt.xscale('log')
                #plt.yscale('log')
                plt.xlabel("l")
                plt.ylabel(r"$E_{l}$")
                plt.savefig(Path(args["fig_dir"]) / f"l/t={time:04d}")
                plt.close()


                plt.loglog(m_axis[1:], E_m[1:], 'x', markersize=3.0, c='black')
                plt.xlabel("m")
                plt.ylabel(r"$E_{m}$")
                plt.savefig(Path(args["fig_dir"]) / f"m/t={time:04d}")
                plt.close()

                uphi_circ = u_phi['g'][0, :, 32]
                plt.plot(theta, uphi_circ)
                plt.savefig(Path(args["fig_dir"]) / f"uphi/t={time:04d}")
                plt.close()
                p.advance(task)
                increment += 1