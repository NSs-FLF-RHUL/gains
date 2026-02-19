import numpy as np
import pytest

from gains.initial_conditions.mcnally import density


@pytest.mark.parametrize(
    ("xs", "ys", "params", "expected_output"),
    [
        pytest.param(
            np.zeros((10,)), np.zeros((10,)),
            {
                "rho_1": 0.0,
                "rho_2": 0.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.zeros((10,10)),
            id="Every y-coord is 0",
        ),
        pytest.param(
            np.zeros((4,)), np.array([0.125, 0.375, 0.625, 0.875]),
            {
                "rho_1": 1.0,
                "rho_2": 1.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.ones((4,4)),
            id="Midpoint of each interval",
        ),
        pytest.param(
            np.zeros((3,)), np.array([0.25, 0.5, 0.75]),
            {
                "rho_1": 1.0,
                "rho_2": 10.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.array([10.0, 10.0, 1.0]),
            id="On the interval boundaries",
        ),
    ],
)
def test_mcnally_density(
    xs:np.ndarray, ys: np.ndarray, params: dict[str, float], expected_output: np.ndarray
) -> None:
    computed_output = density(xs, ys, **params)

    assert np.allclose(computed_output, expected_output)


@pytest.fixture
def params() -> dict[str, float]:
    return {
        "rho_1": 1.0,
        "rho_2": 10.0,
        "L": 1.0,
        "rho_m": 0.0,
    }


@pytest.mark.parametrize(
    ("missing_key",),
    [
        pytest.param("rho_1"),
        pytest.param("rho_2"),
        pytest.param("L"),
        pytest.param("rho_m"),
    ],
)
def test_mcnally_missing_params(
    missing_key: str,
    params: dict[str, float],
    xs: np.ndarray = np.zeros((4,)),
    ys: np.ndarray = np.array([0.125, 0.375, 0.625, 0.875]),
) -> None:
    del params[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        density(xs, ys, **params)
