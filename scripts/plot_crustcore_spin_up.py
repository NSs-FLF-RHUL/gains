"""Analysis and plotting of the results of simulations involving 2 bases."""

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
    plot_angular_velocity_sequence,
    plot_angular_velocity_split,
    plot_stream,
)
from gains.params.single_spin_up_rotating import parameters as default_params
from gains.utils.parsers import create_parser_analysis

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

    fig, ax = plt.subplots(
        1, len(args["times_plot"]), figsize=(16, 8), subplot_kw={"projection": "polar"}
    )
    plot_angular_velocity_sequence(
        args["times_plot"], ax, args["output_dir"], ("u_b_n_phi", "u_s_n_phi"), **PARAMS
    )
    plt.savefig("{}/angular_speed_sequence_NF.png".format(args["fig_dir"]))
    plt.close()

    path_plot = args["output_dir"] / "su_equator/AZ_avg_equator/AZ_avg_equator_s1.h5"
    data = h5py.File(path_plot, mode="r")
    time = np.array(data["scales/sim_time"])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})
    ur_b = data["tasks"]["u_b_n_r"][:, -1, :, :]
    ur_s = data["tasks"]["u_s_n_r"][:, -1, :, :]
    utheta_b = data["tasks"]["u_b_n_theta"][:, -1, :, :]
    utheta_s = data["tasks"]["u_s_n_theta"][:, -1, :, :]
    uphi_b = data["tasks"]["u_b_n_phi"]
    uphi_s = data["tasks"]["u_s_n_phi"]

    theta_b = uphi_b.dims[2][0][:].ravel()
    r_b = uphi_b.dims[3][0][:].ravel()
    theta_s = uphi_s.dims[2][0][:].ravel()
    r_s = uphi_s.dims[3][0][:].ravel()
    plot_stream(
        r_b[::-1], theta_b, ur_b[30], utheta_b[30], 2.0, time[30], ax, colour="#ff7f50"
    )
    plot_stream(
        r_s[::-1], theta_s, ur_s[30], utheta_s[30], 1.0, time[30], ax, colour="#404969"
    )

    plt.savefig(f"{args['fig_dir']}/meridional_streamlines_core.png")

    path = "{}/su_equator/AZ_avg_equator".format(args["output_dir"])
    r_check, theta_check = get_angular_coords(path + "/AZ_avg_equator_s1.h5", "u_b_n_phi")

    r = LabeledCoordinate(r_check, "r")
    theta = LabeledCoordinate(theta_check, "theta")

    err_msg = "Coordinate varied must be r or theta."

    if args["coordinate"] == "r":
        path_list, fig = plot_against_time(
            r, "r", path, PARAMS["Ek"], PARAMS["Ntheta"], args["targets"], "u_b_n_phi"
        )
        fig.savefig("{}/radial_against_time.png".format(args["fig_dir"]))

    elif args["coordinate"] == "theta":
        path_list, fig = plot_against_time(
            theta,
            "theta",
            path,
            PARAMS["Ek"],
            PARAMS["Ntheta"],
            args["targets"],
            "u_b_n_phi",
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
            ur_b = data["tasks"]["u_b_n_r"][:, -1, :, :]
            ur_s = data["tasks"]["u_s_n_r"][:, -1, :, :]
            utheta_b = data["tasks"]["u_b_n_theta"][:, -1, :, :]
            utheta_s = data["tasks"]["u_s_n_theta"][:, -1, :, :]
            uphi_b = data["tasks"]["u_b_n_phi"]
            uphi_s = data["tasks"]["u_s_n_phi"]

            theta_b = uphi_b.dims[2][0][:].ravel()
            r_b = uphi_b.dims[3][0][:].ravel()
            theta_s = uphi_s.dims[2][0][:].ravel()
            r_s = uphi_s.dims[3][0][:].ravel()
            for j in range(len(time)):
                fig, ax = plt.subplots(
                    1, 1, figsize=(16, 8), subplot_kw={"projection": "polar"}
                )
                plot_angular_velocity_split(
                    path,
                    j,
                    ax,
                    rotating=True,
                    core_field="u_s_n_phi",
                    crust_field="u_b_n_phi",
                    delta_omega=PARAMS["Delta_Omega"],
                    crustcore_boundary=PARAMS["Ri"],
                )
                save_path_angular = (
                    args["frame_dir"] / f"frame_spin_up_angular_{count:04d}.png"
                )
                save_path_stream = (
                    args["frame_dir"] / f"frame_spin_up_stream_{count:04d}.png"
                )

                fig2, ax2 = plt.subplots(
                    1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"}
                )
                plot_stream(
                    r_b[::-1],
                    theta_b,
                    ur_b[j],
                    utheta_b[j],
                    2.0,
                    time[j],
                    ax2,
                    colour="#ff7f50",
                )
                plot_stream(
                    r_s[::-1],
                    theta_s,
                    ur_s[j],
                    utheta_s[j],
                    1.0,
                    time[j],
                    ax2,
                    colour="#404969",
                )
                fig.canvas.draw()
                fig2.canvas.draw()

                fig.savefig(save_path_angular)
                fig2.savefig(save_path_stream)
                count = count + 1
                if count % 20 == 0:
                    logger.info(f"saved frame {count:04d}.png")
