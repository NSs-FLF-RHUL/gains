"""Analysis and plotting of the results of the single fluid spin up."""

import argparse
import logging
import pathlib
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gains.analysis.analyse_spin_up import (
    LabeledCoordinate,
    get_angular_coords,
    plot_against_time,
    plot_angular_velocity,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Analyse the output of single_spin_up_rotating_frame."
)

parser.add_argument(
    "output_dir", type=str, default=None, help="Path to output directory."
)

parser.add_argument(
    "--fig_dir",
    type=str,
    default="outputs",
    help="The directory in which to save figures.",
)

parser.add_argument(
    "--frame_dir",
    type=str,
    default="frames",
    help="The directory in which to save frames.",
)

args = vars(parser.parse_args())

logger = logging.getLogger(__name__)

pathlib.Path.mkdir(pathlib.Path(args["fig_dir"]), parents=True, exist_ok=True)

anim_check = input("Plot frames for animation? [y/n]: ")

fig, ax = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={"projection": "polar"})

path_1 = "{}/su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5".format(args["output_dir"])
plot_angular_velocity(path_1, 10, ax[0], rotating=True)

path_2 = "{}/su_equator/AZ_avg_equator/AZ_avg_equator_s3.h5".format(args["output_dir"])
plot_angular_velocity(path_2, 40, ax[1], rotating=True)

path_3 = "{}/su_equator/AZ_avg_equator/AZ_avg_equator_s4.h5".format(args["output_dir"])
plot_angular_velocity(path_3, 90, ax[2], rotating=True)
plt.savefig("{}/Equator_spin_up_5e-2.png".format(args["fig_dir"]))
plt.close()

path = "{}/su_equator/AZ_avg_equator".format(args["output_dir"])
r_check, theta_check = get_angular_coords(path + "/AZ_avg_equator_s1.h5")

r = LabeledCoordinate(r_check, "r")
theta = LabeledCoordinate(theta_check, "theta")

path_list, fig = plot_against_time(r, "r", path)
fig.savefig("{}/radial_against_time.png".format(args["fig_dir"]))

if anim_check == "y":
    num_files = len(path_list)
    count = 0
    pathlib.Path.mkdir(pathlib.Path(args["frame_dir"]), parents=True, exist_ok=True)
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
