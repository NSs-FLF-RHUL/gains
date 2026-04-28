"""Stores useful functions, applicable throughout the package."""

import argparse
import cProfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from mpi4py import MPI


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

    return parser


def profile(dirname: str | None, params: dict) -> Callable:
    """
    Provide a decorator to use cProfile to profile an function running in parallel.

    The stats are optionally saved in a subdirectory of the overall
    output directory for the simulation, and saved using the dump_stats
    method in a format readable by snakeviz.

    :param dirname: The name of the directory to save the profiles to.
    """
    comm = MPI.COMM_WORLD

    if dirname is None:
        return lambda f: f

    def prof_decorator(f: Callable) -> Callable:
        def wrap_f(*args: object, **kwargs: object) -> object:
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            output_dir = Path("outputs") / params["output_dir"] / dirname
            # Only rank 0 creates directory to avoid race conditions
            if comm.rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
            # All ranks wait until directory exists
            comm.Barrier()

            filename = output_dir / Path(f"time_profile.{comm.rank}")
            pr.dump_stats(filename)

            return result

        return wrap_f

    return prof_decorator


def get_arg_of_nearest(target: float, arr: np.ndarray) -> tuple[int, float]:
    """
    Return the nearest value to a target in an array, as well as its index.

    :param target: The ideal value to search for in the array.
    :param arr: The array to be searched for the target value.
    :returns index: The index of the nearest value to target in the array.
    :returns nearest: The closest value to the target in the array.
    """
    diff = np.abs(arr - target)
    index = np.argmin(diff)
    nearest = arr[index]
    return index, nearest
