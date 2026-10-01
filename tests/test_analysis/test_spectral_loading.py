from pathlib import Path
from typing import Any

import dedalus.public as d3
import h5py
import numpy as np
import pytest

from gains.params.spherical_shell import parameters_test as default_params


@pytest.fixture
def params() -> dict[str, Any]:
    """Provides parameters as a fixture."""
    return default_params


def create_test_objects(
    params: dict[str, Any],
) -> tuple[d3.Distributor, d3.ShellBasis, d3.Field]:
    """
    Create the dedalus distributor, basis, and field needed for this test.

    :param params: Dictionary of example simlation parameters.
    """
    coords = d3.SphericalCoordinates("phi", "theta", "r")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.ShellBasis(
        coords,
        shape=(params["Nphi"], params["Ntheta"], params["Nr"]),
        radii=(params["Ri"], params["Ro"]),
        dtype=np.float64,
        dealias=params["dealias"],
    )

    u_test = dist.Field(name="u_test", bases=basis)
    random_grid_data = np.random.default_rng().random(
        (params["Nphi"], params["Ntheta"], params["Nr"]), dtype=np.float64
    )
    u_test["g"] = random_grid_data
    return dist, basis, u_test


def save_test_field(tmp_path: Path, test_field: d3.Field) -> None:
    """
    Handles dedalus boilerplate of saving the provided field.

    The actual content of the problem is not relevant for this test,
    so it is kept minimal.
    :param tmp_path: Path to save field to. Also a pytest fixture for
    a temporary directory.
    :param test_field: The field to be saved.
    """
    # Need problem and solver to create/evaluate file handlers
    problem = d3.IVP([test_field], namespace=locals())
    problem.add_equation("dt(test_field) = 0")
    solver = problem.build_solver(d3.SBDF2)

    test_data = solver.evaluator.add_file_handler(tmp_path / "test_data")
    test_data.add_task(test_field)
    solver.evaluator.evaluate_handlers(
        [test_data],
        wall_time=solver.wall_time,
        sim_time=solver.sim_time,
        iteration=solver.iteration,
    )


def test_reduction_equivalence(tmp_path: Path, params: dict[str, Any]) -> None:
    """
    Test if the coefficients of loaded data are the same as data created in scripts.

    :param tmp_path: Path to save field to. Also a pytest fixture for
    a temporary directory.
    :param params: Dictionary of example simlation parameters.
    """
    dist, basis, u_test = create_test_objects(params)
    coeffs_defined = u_test["c"]
    save_test_field(tmp_path, u_test)
    with h5py.File(tmp_path / "test_data/test_data_s1.h5") as f:
        u_data_loaded = dist.Field(name="u_data_loaded", bases=basis)
        u_data_loaded["g"] = f["tasks/u_test"]
        coeffs_loaded = u_data_loaded["c"]
        np.testing.assert_allclose(coeffs_defined, coeffs_loaded)
