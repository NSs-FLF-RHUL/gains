"""Analysis and plotting of the results of the single fluid spin up."""

import logging
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gains.analysis.analyse_spin_up import (
    LabeledCoordinate,
    args,
    get_angular_coords,
    plot_against_time,
    plot_angular_velocity,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

args["output_dir"] = Path(args["output_dir"])
args["frame_dir"] = Path(args["frame_dir"])

logger = logging.getLogger(__name__)

Path.mkdir(Path(args["fig_dir"]), parents=True, exist_ok=True)

anim_check = input("Plot frames for animation? [y/n]: ")

fig, ax = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={"projection": "polar"})

path_1 = args["output_dir"] / "su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"
plot_angular_velocity(path_1, 10, ax[0], rotating=True)

path_2 = args["output_dir"] / "su_equator/AZ_avg_equator/AZ_avg_equator_s3.h5"
plot_angular_velocity(path_2, 40, ax[1], rotating=True)

path_3 = args["output_dir"] / "su_equator/AZ_avg_equator/AZ_avg_equator_s4.h5"
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
    Path.mkdir(args["frame_dir"], parents=True, exist_ok=True)
    for i in range(num_files):
        path = path_list[i]
        data = h5py.File(path, mode="r")
        time = np.array(data["scales/sim_time"])
        for j in range(len(time)):
            fig, ax = plt.subplots(
                1, 1, figsize=(16, 8), subplot_kw={"projection": "polar"}
            )
            plot_angular_velocity(path, j, ax, rotating=True)
            save_path = args["frame_dir"] / f"frame_spin_up_{count:04d}.png"
            plt.savefig(save_path)
            count = count + 1
            if count % 20 == 0:
                logger.info(f"saved frame {count:04d}.png")
