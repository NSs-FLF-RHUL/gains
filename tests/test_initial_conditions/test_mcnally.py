import numpy as np
import pytest

from gains.initial_conditions.mcnally import density, velocity_x, bounds


@pytest.mark.parametrize(
    ("xs", "ys", "params", "expected_output"),
    [
        pytest.param(
            np.zeros((10,)),
            np.zeros((10,)),
            {
                "rho_1": 0.0,
                "rho_2": 0.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.zeros((10, 10)),
            id="Every (x,y) is (0,0)",
        ),
        pytest.param(
            np.zeros((4,)),
            np.array([0.125, 0.375, 0.625, 0.875]),
            {
                "rho_1": 1.0,
                "rho_2": 1.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.ones((4, 4)),
            id="Midpoint of each interval",
        ),
        pytest.param(
            np.zeros((3,)),
            np.array([0.25, 0.5, 0.75]),
            {
                "rho_1": 1.0,
                "rho_2": 10.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.transpose(np.array([[10.0] * 3, [10.0] * 3, [1.0] * 3])),
            id="On the interval boundaries",
        ),
    ],
)
def test_mcnally_density(
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict[str, float],
    expected_output: np.ndarray,
) -> None:
    """Applies unit tests for the density function in src/gains/mcnally.py."""
    computed_output = density(xs, ys, **params)

    assert np.allclose(computed_output, expected_output)


@pytest.fixture
def params_density() -> dict[str, float]:
    """Provides density parameters for keyerror test."""
    return {
        "rho_1": 1.0,
        "rho_2": 10.0,
        "L": 1.0,
        "rho_m": 0.0,
    }


@pytest.fixture
def zeros() -> np.ndarray:
    """Array of zeros for x or y axes."""
    return np.zeros((4,))


@pytest.fixture
def midpoints() -> np.ndarray:
    """Value in each bound for x or y axes."""
    return bounds


@pytest.mark.parametrize(
    "missing_key",
    [
        pytest.param("rho_1"),
        pytest.param("rho_2"),
        pytest.param("L"),
        pytest.param("rho_m"),
    ],
)
def test_density_missing_params(
    missing_key: str,
    params_density: dict[str, float],
    zeros: np.ndarray,
    bounds: np.ndarray,
) -> None:
    """
    Keyerror test for density.

    Confirms density in src/gains/mcnally.py gives correct keyerror for
    missing parameters.
    """
    del params_density[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        density(zeros, bounds, **params_density)


@pytest.mark.parametrize(
    ("xs", "ys", "params", "expected_output"),
    [
        pytest.param(
            np.zeros((10,)),
            np.zeros((10,)),
            {
                "U_1": 0.0,
                "U_2": 0.0,
                "L": 1.0,
                "U_m": 0.0,
            },
            np.zeros((10, 10)),
            id="Every (x,y) is (0,0)",
        ),
        pytest.param(
            np.zeros((4,)),
            np.array([0.125, 0.375, 0.625, 0.875]),
            {
                "U_1": 1.0,
                "U_2": 1.0,
                "L": 1.0,
                "U_m": 0.0,
            },
            np.ones((4, 4)),
            id="Midpoint of each interval",
        ),
        pytest.param(
            np.array([0.25, 0.5, 0.75]),
            np.zeros((3,)),
            {
                "U_1": 1.0,
                "U_2": 10.0,
                "L": 1.0,
                "U_m": 0.0,
            },
            np.array([[10.0] * 3, [10.0] * 3, [1.0] * 3]),
            id="On interval boudaries",
        ),
    ],
)
def test_mcnally_vx(
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict[str, float],
    expected_output: np.ndarray,
) -> None:
    """Applies unit tests for the x-velocity function in src/gains/mcnally.py."""
    computed_output = velocity_x(xs, ys, **params)

    assert np.allclose(computed_output, expected_output)


@pytest.fixture
def params_vx() -> dict[str, float]:
    """Provides vx parameters for keyerror test."""
    return {
        "U_1": 1.0,
        "U_2": 10.0,
        "L": 1.0,
        "U_m": 0.0,
    }


@pytest.mark.parametrize(
    "missing_key",
    [
        pytest.param("U_1"),
        pytest.param("U_2"),
        pytest.param("L"),
        pytest.param("U_m"),
    ],
)
def test_vx_missing_params(
    missing_key: str,
    params_vx: dict[str, float],
    bounds: np.ndarray,
    zeros: np.ndarray,
) -> None:
    """
    Keyerror test for x-velocity.

    Confirms density in src/gains/mcnally.py gives correct keyerror for
    missing parameters.
    """
    del params_vx[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        velocity_x(bounds, zeros, **params_vx)
