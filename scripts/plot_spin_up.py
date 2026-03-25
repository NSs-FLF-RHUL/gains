"""Analysis and plotting of the results of the single fluid spin up."""

import logging
import pathlib
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gains.analysis.analyse_spin_up import (
    get_angular_coords,
    plot_against_time,
    plot_angular_velocity,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

anim_check = input("Plot frames for animation? [y/n]: ")

fig, ax = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={"projection": "polar"})

path_1 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"
p1 = plot_angular_velocity(path_1, 10, ax[0], rotating=True)

path_2 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s3.h5"
plot_angular_velocity(path_2, 40, ax[1], rotating=True)

path_3 = "outputs/su_equator/AZ_avg_equator/AZ_avg_equator_s4.h5"
plot_angular_velocity(path_3, 90, ax[2], rotating=True)
plt.savefig("outputs/Equator_spin_up_5e-2.png")
plt.close()
path = "outputs/su_equator/AZ_avg_equator"
r_check, theta = get_angular_coords(path + "/AZ_avg_equator_s1.h5")

return_check = False
if anim_check == "y":
    return_check = True

path_list = plot_against_time(
    "theta",theta, "surface", r"$\theta$", path, return_paths=return_check
)

if anim_check == "y":
    num_files = len(path_list)
    count = 0
    pathlib.Path.mkdir(pathlib.Path("frames"), parents=True, exist_ok=True)
    for i in range(num_files):
        path = path_list[i]
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        for j in range(len(time)):
            fig, ax = plt.subplots(
                1, 1, figsize=(16, 8), subplot_kw={"projection": "polar"}
            )
            plot_angular_velocity(path, j, ax, rotating=True)
            plt.savefig(f"frames/equator_rotating_t_{count:04d}.png")
            count = count + 1
            if count % 20 == 0:
                logger.info(f"saved frame {count:04d}.png")
