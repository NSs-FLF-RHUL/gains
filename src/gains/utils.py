import argparse
from pathlib import Path
from collections.abc import Callable
import cProfile
from mpi4py import MPI

def create_parser_simulation() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='simulate glitch on the boundary of a spherical star'
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
        "--output_dir", type=str, default=None, help="Directory to store simulation outputs"
    )

    parser.add_argument(
        "--parameter_file",
        type=Path,
        default=None,
        help="relative path to parameter file to use for this run, saved in json format.",
    )

    return parser

def profile(dirname: str | None, PARAMS: dict) -> Callable:
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

            output_dir = Path("outputs") / PARAMS["output_dir"] / dirname
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
