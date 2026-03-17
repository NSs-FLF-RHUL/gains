import matplotlib.pyplot as plt
import matplotlib
import h5py

import numpy as np
from matplotlib import ticker, font_manager
import warnings
import os
warnings.filterwarnings("ignore")

from gains.Analysis.Analyse_spin_up import *

'''
Plots angular velocity at different times.
'''

fig,ax = plt.subplots(1,3,figsize=(16,8),subplot_kw={'projection': 'polar'})

path_1 = './AZ_avg_equator/AZ_avg_equator_s1.h5'
p1 = plot_angular(path_1,10,ax[0],rotating=True)


path_2 = './AZ_avg_equator/AZ_avg_equator_s3.h5'
plot_angular(path_2,40,ax[1],rotating=True)

path_3 ='./AZ_avg_equator/AZ_avg_equator_s8.h5'
plot_angular(path_3,90,ax[2],rotating=True)
#plt.savefig("Angular_5e-3.png")
plt.show()

file_list = sorted(os.listdir('./AZ_avg_equator'))

path_list = []
for file in file_list:
    print(file)
    path = "./AZ_avg_equator/"+file
    path_list.append(path)

def angular_time(r_get: int, n_writes: int) -> np.ndarray | np.ndarray:
    omega_rs = []
    times = []
    for path in path_list:
        data = h5py.File(path, mode='r')
        time = np.array(data['scales/sim_time'])
        r, theta = coords_angular(path)
        for j in range(0,n_writes):
            u_n_phi = data['tasks']['u_n_phi'][j,-1,:,:]
            omega = get_angular(r, theta, u_n_phi)
            omega_r = omega[63][r_get]
            omega_rs.append(omega_r)
            times.append(time[j])
    return omega_rs, times

path = path_list[0]
r_check, theta = coords_angular(path)
print(len(r_check))

r_tries = [i for i in range(60,len(r_check),6)]
alphas = np.linspace(0.40,1.0,len(r_tries))
rs_checked = [r_check[i] for i in range(35,len(r_check),6)]
print(rs_checked)
for i in range(0,len(r_tries)):
    val = r_tries[i]
    omega_r, times = angular_time(val, 100)
    plt.plot(sorted(times), sorted(omega_r), color = '#024cf7', alpha = alphas[i], label = str(round(rs_checked[i],2)) + 'R')

plt.legend(frameon=False)
t_ek = 1/np.sqrt(Ek)
plt.axvline(x=t_ek, linestyle='dashed', color = 'black', lw = 0.5)
plt.text(15, 0.0001,r'$\tau_{Ek}$', size = 'large')
plt.xlabel('Time since glitch ($\Omega_{0}^{-1}$)')
plt.ylabel("$\Delta \Omega$")
plt.show()
#plt.savefig("spin_up_time_equator.png", dpi=300)

'''
num_files = len(path_list)
count = 0
for i in range(0,num_files):
    path = path_list[i]
    data = h5py.File(path, mode='r')
    time = np.array(data['scales/sim_time'])
    for j in range(0,len(time)):
        fig, ax = plt.subplots(1,1,figsize=(16,8),subplot_kw={'projection': 'polar'})
        plot_angular(path,j,ax,True)
        plt.savefig("frames/equator_rotating_t_%04d.png" % count)
        count = count+1
        if count % 20 == 0:
            print("saved frame %04d.png" % count)
'''