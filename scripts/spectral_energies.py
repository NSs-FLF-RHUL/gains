import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
from pastamarkers import markers
from rich.progress import Progress, TextColumn, BarColumn

paths = []

#paths = ["outputs/spectral_filter/su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"]

Nphi = 128
Ntheta = 64
Nr = 64
r_min, r_max = 0.8, 1.0

coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.ShellBasis(coords,shape=(Nphi, Ntheta, Nr), radii=(r_min, r_max), dtype=np.float64, dealias=1.5)

with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green"),
        ) as p:
    task = p.add_task("Plotting: ", total=201)
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
                u_r_coeff = u_r['c']
                u_theta_coeff = u_theta['c']
                u_phi_coeff = u_phi['c']

                energy_density = np.abs(u_r_coeff)**2 + np.abs(u_theta_coeff)**2 + np.abs(u_phi_coeff)**2
                energy_sum_r = np.sum(energy_density, axis=2)
                E_l = np.sum(energy_sum_r, axis=0)
                E_m = np.sum(energy_sum_r, axis=1)
                l_axis = np.arange(len(E_l))
                m_axis = np.arange(len(E_m))

                #marker = markers.farfalle

                plt.loglog(l_axis[1:], E_l[1:], 'x', markersize=3.0, c='black')
                #plt.xscale('log')
                #plt.yscale('log')
                plt.xlabel("l")
                plt.ylabel(r"$E_{l}$")
                plt.savefig(f"/Users/rebecca/Desktop/Spectra_time_series/l/t={time:04d}")
                plt.close()


                plt.loglog(m_axis[1:], E_m[1:], 'x', markersize=3.0, c='black')
                plt.xlabel("m")
                plt.ylabel(r"$E_{m}$")
                plt.savefig(f"/Users/rebecca/Desktop/Spectra_time_series/m/t={time:04d}")
                plt.close()

                uphi_circ = u_phi['g'][0, :, 32]
                plt.plot(theta, uphi_circ)
                plt.savefig(f"/Users/rebecca/Desktop/Spectra_time_series/uphi/t={time:04d}")
                plt.close()
                p.advance(task)
                increment += 1
