"""Standardised command-line interfaces to scripts."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


class SimulationCLI(argparse.ArgumentParser):
    """Command-line interface for simulation scripts."""

    sim_name: str
    is_profiling: bool

    def __init__(
        self,
        *args,
        profiling: bool = False,
        sim_name: str = "simulation",
        description: str = "Simulate glitch on the boundary of a spherical star",
        **kwargs,
    ):
        super().__init__(*args, description=description, **kwargs)
        self._add_standard_args()

        self.sim_name = str(sim_name)

        if profiling:
            self.add_profiling_options()
        else:
            self.is_profiling = False

    def _add_standard_args(self) -> None:
        """Add arguments that all simulation CLIs accept to the instance."""
        self.add_argument(
            "--use_checkpoint",
            type=bool,
            default=False,
            help="Boolean argument to determine if to use a checkpoint file.",
        )
        self.add_argument(
            "--checkpoint_path",
            type=str,
            default="outputs/checkpoints/checkpoints_sNumber.h5",
            help="Path to the checkpoint file you want to use.",
        )
        self.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Directory to store simulation outputs",
        )
        self.add_argument(
            "--parameter_file",
            type=Path,
            default=None,
            help="relative path to parameter file to use for this run, saved in"
            " json format.",
        )
        self.add_argument(
            "--logfile",
            type=str,
            default=None,
            help="Name of logfile, if you want to create one.",
        )

    def _default_dir_name(self) -> str:
        """Generate a default name for an output directory."""
        return self.sim_name + datetime.now().astimezone().strftime("%Y-%m-%d-%H:%M")

    def add_profiling_options(self) -> None:
        """"""

    def parse_args(
        self, *args, default_params: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        args = super().parse_args(*args, **kwargs)


def create_parser_simulation() -> argparse.ArgumentParser:
    """Create argument parser for simulations in a rotating spherical star."""
    parser = argparse.ArgumentParser(
        description="simulate glitch on the boundary of a spherical star"
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
