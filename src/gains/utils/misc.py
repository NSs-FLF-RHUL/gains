"""Stores useful functions, applicable throughout the package."""

import re
import warnings
from collections.abc import Callable, Iterable
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


def _rewrite_h5(fin: h5py.File, fout: h5py.File) -> None:
    """Create a new h5 file with same data as input, but at float32 precision."""
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


def _downscale_data(src: str | Path, tmp: str | Path) -> None:
    """
    Convert output data to float32 format.

    Note that the original precision data is destroyed.
    """
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        _rewrite_h5(fin, fout)

    Path(tmp).replace(Path(src))


def downsampling_visitor(
    destination_file: h5py.File,
    downsample_step: int = 20,
    *,
    downsample_groups: Iterable[str],
    downsample_datasets: Iterable[str],
) -> Callable[[str, h5py.Group | h5py.Dataset], None]:
    """
    Visitor factory for downsampling hdf5 data.

    The returned visitor copies the HDF5 group and dataset structure from the
    source file to destination_file. Datasets selected by
    downsample_groups or downsample_datasets are downsampled along
    their first axis by taking every downsample_step-th value. Other
    datasets are copied unchanged.

    Dataset attributes and relevant dataset metadata, such as dtype, chunks,
    compression, and shuffle settings, are preserved. Scalar (0-dimensional)
    datasets are copied without downsampling.

    :param source_path: Path to the file you want to downsample.
    :param target_path: Path to save the downsampled file to.
    :param downsample_step: Step size for downsampling (the default 20 will take every
    20th value from the original file).
    :param downsample_groups: A list of the h5 groups to be downsampled. All datasets
    in the group will be downsampled.
    :parame downsample_files: A list of specific datasets to be downsampled.
    """

    def _inner(
        name: str,
        obj: h5py.Group | h5py.Dataset,
    ) -> None:
        downsample = (
            any(name.startswith(folder) for folder in downsample_groups)
            or name in downsample_datasets
        )

        if isinstance(obj, h5py.Group):
            destination_file.create_group(name)

        elif isinstance(obj, h5py.Dataset):
            # Handle empty or 0-dimensional datasets
            if obj.shape == ():
                destination_file.create_dataset(name, dtype=obj.dtype, data=obj[()])
            else:
                # Calculate new shape assuming simulation time is on Axis 0
                old_shape = obj.shape
                new_axis_0 = int(np.ceil(old_shape[0] / downsample_step))
                new_shape = (new_axis_0, *old_shape[1:]) if downsample else old_shape

                # Create the new dataset with same metadata
                dst_dset = destination_file.create_dataset(
                    name,
                    shape=new_shape,
                    dtype=obj.dtype,
                    chunks=obj.chunks,
                    compression=obj.compression,
                    compression_opts=obj.compression_opts,
                    shuffle=obj.shuffle,
                    fletcher32=obj.fletcher32,
                )

                # Slice every 20th point along Axis 0 and stream it to the new file
                # Using [::step] prevents loading entire dataset into RAM at once
                if downsample:
                    dst_dset[...] = obj[::downsample_step, ...]
                else:
                    dst_dset[...] = obj

        # Copy over metadata/attributes (e.g., simulation units, timestamps)
        for attr_name, attr_value in obj.attrs.items():
            destination_file[name].attrs[attr_name] = attr_value

    return _inner


def downsample_h5_file(
    source_path: Path,
    target_path: Path,
    downsample_step: int = 20,
    *,
    downsample_groups: Iterable[str] = (),
    downsample_datasets: Iterable[str] = (),
) -> None:
    """
    Produce a downsampled clone of an existing h5 file.

    Clones an HDF5 structure and populates it with every Nth (default 20th)
    datapoint along the first axis of every dataset. The original file is not
    modified by calling this function.

    :param source_path: Path to the file you want to downsample.
    :param target_path: Path to save the downsampled file to.
    :param downsample_step: Step size for downsampling (the default 20 will take every
    20th value from the original file).
    :param downsample_groups: A list of the h5 groups to be downsampled. All datasets
    in the group will be downsampled.
    :parame downsample_files: A list of specific datasets to be downsampled.
    """
    if downsample_step <= 0:
        msg = "Downsample step should be greater than zero"
        raise ValueError(msg)

    if downsample_step == 1:
        warnings.warn(
            "downsample_step is 1, so selected datasets will be copied without "
            "downsampling.",
            UserWarning,
            stacklevel=2,
        )

    if not (downsample_groups or downsample_datasets):
        downsample_groups = {"tasks/"}
        downsample_datasets = {
            "scales/sim_time",
            "scales/iteration",
            "scales/write_number",
            "scales/timestep",
        }

    with h5py.File(source_path, "r") as src, h5py.File(target_path, "w") as dst:
        src.visititems(
            downsampling_visitor(
                dst,
                downsample_step,
                downsample_groups=downsample_groups,
                downsample_datasets=downsample_datasets,
            )
        )
