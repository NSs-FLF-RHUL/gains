import argparse
from pathlib import Path

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
