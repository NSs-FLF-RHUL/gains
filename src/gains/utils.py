"""Stores useful functions, applicable throughout the package."""

import cProfile
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from mpi4py import MPI


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


def extract_numerical_suffix(path: Path) -> int | float:
    """
    Extract an integer at the end of a filename.

    Takes a path to a file saved in the form /output_dir/file_name[num].extension
    and return num. Files that don't fit this format will be assigned inf, so
    placed at the end of a list when sorting.

    :param path: path to the output file, in the form
    /output_dir/file_name[num].extension.
    :returns suffix: Integer at the end of the file name.
    """
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else float("inf")
