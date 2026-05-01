"""Wrappers and decorators for profiling runs."""

import argparse
import cProfile
from collections.abc import Callable
from pathlib import Path

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


def add_profiling_options(parser: argparse.ArgumentParser) -> None:
    """Add profiling options to existing ArgumentParser instances."""
    parser.add_argument(
        "--profile",
        default=None,
        type=str,
        help="Output directory for profiling results.",
    )
