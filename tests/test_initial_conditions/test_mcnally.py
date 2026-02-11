import numpy as np

from gains.initial_conditions.mcnally import density


def test_mcnally_density() -> None:
    ys = np.zeros((10,), dtype=float)

    params = {
        "rho_1": 0.0,
        "rho_2": 0.0,
        "L": 1.0,
    }
    params["rho_m"] = 0.5 * (params["rho_1"] - params["rho_2"])

    computed_output = density(ys, **params)
    expected_output = np.zeros_like(ys)

    assert np.allclose(computed_output, expected_output)


def test_mcnally_density_compartments() -> None:
    ys = np.array([0.125, 0.375, 0.625, 0.875])

    params = {
        "rho_1": 1.0,
        "rho_2": 1.0,
        "L": 1.0,
    }
    params["rho_m"] = 0.5 * (params["rho_1"] - params["rho_2"])

    computed_output = density(ys, **params)
    expected_output = np.array([1.0, 1.0, 1.0, 1.0])

    assert np.allclose(computed_output, expected_output)
