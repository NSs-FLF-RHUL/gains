import numpy as np
import pytest

from gains.initial_conditions.mcnally import density


@pytest.mark.parametrize(
    ("ys", "params", "expected_output"),
    [
        pytest.param(
            np.zeros((10,)),
            {
                "rho_1": 0.0,
                "rho_2": 0.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.zeros((10,)),
            id="Every y-coord is 0",
        ),
        pytest.param(
            np.array([0.125, 0.375, 0.625, 0.875]),
            {
                "rho_1": 1.0,
                "rho_2": 1.0,
                "L": 1.0,
                "rho_m": 0.0,
            },
            np.ones((4,)),
            id="Midpoint of each interval",
        ),
        pytest.param(
            np.array([0.25, 0.5, 0.75]),
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
def test_mcnally_density(ys, params, expected_output) -> None:
    computed_output = density(ys, **params)

    assert np.allclose(computed_output, expected_output)
