"""Stores useful functions, applicable throughout the package."""

import re
from pathlib import Path

import numpy as np

from gains.exceptions import MeshError


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

def read_logfile(path: Path, quantity: str) -> tuple[list[float], list[float]]:
    """
    Read a logfile from a dedalus run (e.g. an stdout file on a cluster or saved terminal output).

    Extracts a specified quantity from each log, as well as the simulation time. The quantity to extract
    must be entered exactly as it appears in the logfile.

    :param path: The path to the logfile.
    :param quantity: The quantity to extract from each log.
    :returns times: A list of the times each log was given at.
    :returns vals: A list of the values of the specified quantity from each log. 
    """
    with open(path, "r") as f:
        text = f.read()
    
    escaped_quantity = re.escape(quantity)
    regex = escaped_quantity + '=' + r'([0-9.eE+-]+)'
    vals = re.findall(regex, text)
    vals = [float(val) for val in vals]

    times = re.findall(r'Time=([0-9.eE+-]+)', text)
    times = [float(time) for time in times]
    return times, vals

def mesh_cpus(ncpu: int) -> list[int] | None:
    """
    Distribute the number of cores in a 2D mesh.

    Takes the number of cpus and distributes them in a 2D mesh to allow for
    an efficient discretisation by dedalus. Raises an error if the number
    of available cpus is not a power of 2.

    :param ncpu: The number of available cpus.
    :returns mesh: The 2D mesh to be passed to a dedalus distributor object.
    """
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        return [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
    raise MeshError
