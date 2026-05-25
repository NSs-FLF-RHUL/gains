"""Analysis and plotting of the results of the single fluid spin up."""

import json
import logging
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gains.analysis.analyse_spin_up import (
    LabeledCoordinate,
    get_angular_coords,
    plot_against_time,
    plot_angular_velocity,
    plot_stream,
    plot_angular_velocity_sequence,
    plot_angular_velocity_split
)
from gains.params.single_spin_up_rotating import parameters as default_params
from gains.utils.parsers import create_parser_analysis
from gains.utils.misc import get_arg_of_nearest

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = create_parser_analysis()
    args = vars(parser.parse_args())

    if args["parameter_file"] is not None:
        with Path.open(args["parameter_file"]) as param_file:
            PARAMS = json.load(param_file)

    else:
        PARAMS = default_params

    args["output_dir"] = Path(args["output_dir"])
    args["frame_dir"] = Path(args["frame_dir"])

    logger = logging.getLogger(__name__)

    Path.mkdir(Path(args["fig_dir"]), parents=True, exist_ok=True)

    anim_check = input("Plot frames for animation? [y/n]: ")

    fig, ax = plt.subplots(1, len(args["times_plot"]), figsize=(16, 8), subplot_kw={"projection": "polar"})
    plot_angular_velocity_sequence(args["times_plot"],ax,args["output_dir"],("u_b_phi", "u_s_phi"), **PARAMS)
    plt.show()
    #plt.savefig("{}/Equator_spin_up_5e-2.png".format(args["fig_dir"]))
    plt.close()
    path_plot = args["output_dir"] / "su_equator/AZ_avg_equator/AZ_avg_equator_s6.h5"
    data = h5py.File(path_plot, mode="r")
    ur = data["tasks"]["u_s_r"][:, -1, :, :]
    utheta = data["tasks"]["u_s_theta"][:, -1, :, :]
    uphi = data["tasks"]["u_s_phi"]
    theta = uphi.dims[2][0][:].ravel()
    r = uphi.dims[3][0][:].ravel()

    fig = plot_stream(r[::-1], theta, ur[-1], utheta[-1], 2.0)
    plt.savefig(f"{args['fig_dir']}/meridional_streamlines.png")
    path = "{}/su_equator/AZ_avg_equator".format(args["output_dir"])
    r_check, theta_check = get_angular_coords(path + "/AZ_avg_equator_s1.h5", "u_b_phi")

    r = LabeledCoordinate(r_check, "r")
    theta = LabeledCoordinate(theta_check, "theta")

    err_msg = "Coordinate varied must be r or theta."

    if args["coordinate"] == "r":
        path_list, fig = plot_against_time(
            r, "r", path, PARAMS["Ek"], PARAMS["Ntheta"], args["targets"], "u_b_phi"
        )
        fig.savefig("{}/radial_against_time.png".format(args["fig_dir"]))

    elif args["coordinate"] == "theta":
        path_list, fig = plot_against_time(
            theta, "theta", path, PARAMS["Ek"], PARAMS["Ntheta"], args["targets"], "u_b_phi"
        )
        fig.savefig("{}/meridional_against_time.png".format(args["fig_dir"]))

    else:
        raise NotImplementedError(err_msg)

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
                plot_angular_velocity(
                    path, j, ax, rotating=True, delta_omega=PARAMS["Delta_Omega"], target_field="u_b_phi"
                )
                save_path = args["frame_dir"] / f"frame_spin_up_{count:04d}.png"
                plt.savefig(save_path)
                count = count + 1
                if count % 20 == 0:
                    logger.info(f"saved frame {count:04d}.png")
