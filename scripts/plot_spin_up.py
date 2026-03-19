"""
Analysis and plotting of the results of the single fluid spin up.
"""

import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from gains.Analysis.Analyse_spin_up import angular_time, coords_angular, plot_angular
from gains.params.single_spin_up_rotating import parameters

anim_check = input("Plot frames for animation? [y/n]: ")

fig, ax = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={"projection": "polar"})

path_1 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"
p1 = plot_angular(path_1, 10, ax[0], rotating=True)

path_2 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s3.h5"
plot_angular(path_2, 40, ax[1], rotating=True)

path_3 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s4.h5"
plot_angular(path_3, 90, ax[2], rotating=True)
# plt.savefig("Angular_5e-3.png")
plt.show()

file_list = sorted(os.listdir("outputs/su_equator/AZ_avg_equator"))
path_list = []
for file in file_list:
    extension = file[len(file) - 2 : len(file)]

    if extension == "h5":
        path = "outputs/su_equator/AZ_avg_equator/" + file
        path_list.append(path)



path = path_list[0]
r_check, theta = coords_angular(path)
def plot_against_time(coord):

    coord_tries = [i for i in range(int(len(coord)/2), len(coord), 6)]
    alphas = np.linspace(0.40, 1.0, len(coord_tries))
    coord_checked = [coord[i] for i in range(35, len(coord), 6)]

    for i in range(len(coord_tries)):
        val = coord_tries[i]
        omega_r, times = angular_time(val, 100, path_list)
        plt.plot(
            sorted(times),
            sorted(omega_r),
            color="#024cf7",
            alpha=alphas[i],
            label=str(round(coord_checked[i], 2)) + "R",
        )

    plt.legend(frameon=False)
    t_ek = 1 / np.sqrt(parameters["Ek"])
    plt.axvline(x=t_ek, linestyle="dashed", color="black", lw=0.5)
    plt.text(15, 0.0001, r"$\tau_{Ek}$", size="large")
    plt.xlabel(r"Time since glitch ($\Omega_{0}^{-1}$)")
    plt.ylabel(r"$\Delta \Omega$")
    # plt.show()
    plt.savefig("outputs/su_equator/spin_up_time_equator.png", dpi=300)

plot_against_time(r_check)

if anim_check == "y":
    num_files = len(path_list)
    count = 0
    for i in range(num_files):
        path = path_list[i]
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        for j in range(len(time)):
            fig, ax = plt.subplots(
                1, 1, figsize=(16, 8), subplot_kw={"projection": "polar"}
            )
            plot_angular(path, j, ax, True)
            plt.savefig("frames/equator_rotating_t_%04d.png" % count)
            count = count + 1
            if count % 20 == 0:
                print("saved frame %04d.png" % count)
