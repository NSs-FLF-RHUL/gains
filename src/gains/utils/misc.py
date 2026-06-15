"""Stores useful functions, applicable throughout the package."""

import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gains.exceptions import MeshError


def _get_ax_and_fig(ax: plt.Axes | None, *, polar: bool) -> tuple[plt.Figure, plt.Axes]:
    """Handle optional axes arguments in plotting functions."""
    if ax is None:
        if polar:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


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
    Read a logfile from a dedalus run.

    Extracts a specified quantity from each log, as well as the simulation time.
    The quantity to extract must be entered exactly as it appears in the logfile.

    :param path: The path to the logfile.
    :param quantity: The quantity to extract from each log.
    :returns times: A list of the times each log was given at.
    :returns vals: A list of the values of the specified quantity from each log.
    """
    with Path.open(path) as f:
        text = f.read()

    escaped_quantity = re.escape(quantity)
    regex = escaped_quantity + "=" + r"([0-9.eE+-]+)"
    vals = re.findall(regex, text)
    vals = [float(val) for val in vals]

    times = re.findall(r"Time=([0-9.eE+-]+)", text)
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


def select_time(
    nwrites: int, target_time: float, output_dir: Path, **params
) -> tuple[Path, int]:
    """
    Take a simulated time and locate its position in the output files.

    :param nwrites: Number of data writes per file.
    :param target_time: Simulated time to locate.
    :param output_dir: Location of the simulation outputs.
    :param params: Simulation parameters
    :returns path: The path to the output file containing the requested time
    :returns index: The index of the time within the correct file
    """
    saved_times = np.arange(0, params["stop_sim_time"], params["snapshot_dt"])
    target_index = get_arg_of_nearest(target_time, saved_times)[0]
    file_suffix = target_index // nwrites + 1
    file_index = target_index % nwrites
    path = output_dir / f"su_equator/AZ_avg_equator/AZ_avg_equator_s{file_suffix}.h5"
    return path, file_index


def _resolve_rotating(rotating: bool | None) -> bool:  # noqa: FBT001 (All non helper functions do follow this rule)
    """Set default behaivour of functions accepting the rotating argument."""
    return True if rotating is None else rotating


def _downscale_data(src: str | Path, tmp: str | Path) -> None:
    """
    Convert output data to float32 format.

    Note that the original precision data is destroyed.
    """
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        fout.create_group("tasks")

        for name, ds in fin["tasks"].items():
            # Create new dataset with SAME layout but float32 dtype
            out = fout.create_dataset(
                f"tasks/{name}",
                shape=ds.shape,
                dtype=np.float32,
                chunks=ds.chunks,
                compression=ds.compression,
                compression_opts=ds.compression_opts,
                shuffle=ds.shuffle,
                fletcher32=ds.fletcher32,
            )

            for i in range(ds.shape[0]):
                out[i] = ds[i].astype(np.float32)

    os.replace(tmp, src)
