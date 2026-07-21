import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gains.utils.parsers import SimulationCLI


@pytest.fixture
def cli_for_tests(tmp_path: Path) -> Callable[..., SimulationCLI]:
    """
    Create a CLI parser that can be used in tests.

    Importantly, the `place_all_outputs_under` variable will be set to the `tmp_path`,
    to allow for redirection to a safe place for creation / destruction of files.
    """

    def _inner(*args, **kwargs) -> SimulationCLI:
        return SimulationCLI(*args, place_all_outputs_under=tmp_path, **kwargs)

    return _inner


@pytest.fixture
def logger_for_tests() -> logging.Logger:
    """Creates a test-scoped logger."""
    return logging.getLogger("TestLogger")


def _dummy_params() -> dict[str, bool | float | int | str]:
    """
    Standalone parameters that we can use in tests.

    Create as global so we can use its values in pytest.param objects.
    """
    return {
        "string": "string",
        "int": 1,
        "float": 3.14,
        "bool": True,
    }


@pytest.fixture
def tmp_parameters(tmp_path: Path) -> Path:
    """Create a `.json` file of `_dummy_params()`."""
    parameter_file = tmp_path / "default_parameters.json"
    with parameter_file.open("w") as f:
        json.dump(_dummy_params(), f)
    return parameter_file


@pytest.mark.parametrize(
    (
        "cli_creation_opts",
        "cli_args",
        "default_params",
        "expected_output",
        "pass_parameter_file",
    ),
    [
        pytest.param(
            {},
            [],
            {},
            {},
            False,
            id="No args passed",
        ),
        pytest.param(
            {"profiling_option": True},
            [],
            {},
            {},
            False,
            id="Profiling option must still be passed to activate",
        ),
        pytest.param(
            {"profiling_option": True},
            ["--profile", "profiling/path"],
            {},
            {"profile": "profiling/path"},
            False,
            id="Pass profiling option",
        ),
        pytest.param(
            {},
            ["--profile", "profiling/path"],
            {},
            SystemExit(2),
            False,
            id="Profiling disabled but passed anyway",
        ),
        pytest.param(
            {},
            ["--logfile", "log/file"],
            {},
            {},
            False,
            id="Logging handler set",
        ),
        pytest.param(
            {},
            [],
            {},
            _dummy_params(),
            True,
            id="Load parameter file",
        ),
        pytest.param(
            {},
            [],
            {"new_key": 1.0, "int": 2},
            _dummy_params(),
            True,
            id="Load parameter file (ignore defaults)",
        ),
        pytest.param(
            {},
            [],
            {"new_key": 1.0, "int": 2},
            {"new_key": 1.0, "int": 2},
            False,
            id="Load params from defaults",
        ),
    ],
)
def test_simulation_cli_parsing(
    cli_creation_opts: dict[str, Any],
    cli_args: list[Any],
    default_params: dict[str, Any],
    expected_output: dict[str, Any] | Exception,
    cli_for_tests: Callable[..., SimulationCLI],
    logger_for_tests: logging.Logger,
    raises_context: Callable[[Exception], pytest.RaisesExc],
    tmp_parameters: Path,
    *,
    pass_parameter_file: bool,
) -> None:
    """
    Check `params` returned by `SimulationCLI.parse_args_and_get_params()`.

    Note that we trust `argparse.ArgumentParser.parse_args()` to read from the command-
    line correctly, so we are only testing our additional post-parsing logic with this
    test.
    """
    parser = cli_for_tests(**cli_creation_opts)

    # Temporary / dummy parameter file location is only known at runtime
    if pass_parameter_file:
        cli_args = [*cli_args, "--parameter_file", str(tmp_parameters)]

    if isinstance(expected_output, Exception | SystemExit):
        with raises_context(expected_output):
            parser.parse_args_and_get_params(
                logger_for_tests, cli_args, default_params=default_params
            )
    else:
        # Add defaults to the output dict if they were not explicitly set in the input,
        # which should confirm that the default values are used in the param comparison
        # below.
        expected_output.setdefault("use_checkpoint", False)
        expected_output.setdefault(
            "checkpoint_path",
            str(parser.place_all_outputs_under / parser._default_checkpoint_path),
        )
        expected_output.setdefault(
            "output_dir", parser.place_all_outputs_under / parser._default_output_dir
        )
        expected_output.setdefault("profile", None)
        expected_output.setdefault("checkpoint_cadence", 3600)

        params = parser.parse_args_and_get_params(
            logger_for_tests, cli_args, default_params=default_params
        )

        assert params == expected_output
        # Logger should have a file handler added to it, if CLI options specified a
        # logging file.
        logger_files = [
            h.baseFilename
            for h in logger_for_tests.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert (str(parser.log_path) in logger_files) == bool(parser.log_path)
