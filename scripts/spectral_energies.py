import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3

path = "outputs/failure_larger_Ek_larger_plate/su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"

Nphi = 128
Ntheta = 64
Nr = 64
r_min, r_max = 0.8, 1.0

coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.ShellBasis(coords,shape=(Nphi, Ntheta, Nr), radii=(r_min, r_max), dtype=np.float64)

u_r = dist.Field(name="u_r", bases=basis)
u_theta = dist.Field(name="u_theta", bases=basis)
u_phi = dist.Field(name="u_phi", bases=basis)

with h5py.File(path, 'r') as f:
    u_r['g'] = f['tasks/u_s_n_r'][-1]
    u_theta['g'] = f['tasks/u_s_n_theta'][-1]
    u_phi['g'] = f['tasks/u_s_n_phi'][-1]

u_r.change_scales(1)
u_theta.change_scales(1)
u_phi.change_scales(1)

u_r_coeff = u_r['c']
u_theta_coeff = u_theta['c']
u_phi_coeff = u_phi['c']

energy_density = np.abs(u_r_coeff)**2 + np.abs(u_theta_coeff)**2 + np.abs(u_phi_coeff)**2
energy_sum_r = np.sum(energy_density, axis=0)
E_l = np.sum(energy_sum_r, axis=1)
E_m = np.sum(energy_sum_r, axis=0)
l_axis = np.arange(len(E_l))
m_axis = np.arange(len(E_m))

plt.loglog(l_axis[1:], E_l[1:], '.-')
plt.savefig("L_spectrum.png")

plt.loglog(m_axis[1:], E_l[1:], '.-')
plt.savefig("m_spectrum.png")
