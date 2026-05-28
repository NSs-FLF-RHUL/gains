"""Standardised command-line interfaces to scripts."""

import argparse
from pathlib import Path


def create_parser_simulation() -> argparse.ArgumentParser:
    """Create argument parser for simulations in a rotating spherical star."""
    parser = argparse.ArgumentParser(
        description="simulate glitch on the boundary of a spherical star"
    )

    parser.add_argument(
        "--use_checkpoint",
        type=bool,
        default=False,
        help="Boolean argument to determine if to use a checkpoint file.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/checkpoints/checkpoints_sNumber.h5",
        help="Path to the checkpoint file you want to use.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store simulation outputs",
    )

    parser.add_argument(
        "--parameter_file",
        type=Path,
        default=None,
        help="relative path to parameter file to use for this run, saved in"
        " json format.",
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Name of logfile, if you want to create one.",
    )

    return parser


def create_parser_analysis() -> argparse.ArgumentParser:
    """Create parser for command line arguments in plotting code."""
    parser = argparse.ArgumentParser(
        description="Full analysis of a single component spin up simulation"
    )
    parser.add_argument(
        "--parameter_file",
        type=str,
        default=None,
        help="relative path to parameter file to use for this run,"
        " saved in json format.",
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

    parser.add_argument(
        "--targets",
        type=float,
        nargs="*",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="The coordinate values you want to plot against time (The default "
        "assumes you are plotting different radii against time).",
    )

    parser.add_argument(
        "--coordinate",
        type=str,
        default="r",
        help="The coordinate to compare the spin up with time against "
        "(ie vary the radial or angular location)."
        " Takes r by default, pass theta to vary the meridional coordinate instead.",
    )

    parser.add_argument(
        "--times_plot",
        type=float,
        nargs="*",
        default=[0.5, 4, 20],
        help="Three values of time to plot angular velocity at.",
    )

    return parser
