import numpy as np
import pytest

from gains.initial_conditions.single_component_spin_up import window_equator, ExpectPositiveError


def thetas_full() -> np.ndarray:
    """Returns array from zero to pi."""
    return np.linspace(0, np.pi, 100)


@pytest.fixture
def zeros() -> np.ndarray:
    """Array of zeros for x or y axes."""
    return np.zeros((4,))


@pytest.fixture
def thetas_full_fix() -> np.ndarray:
    """Returns array from zero to pi."""
    return np.linspace(0, np.pi, 100)


@pytest.mark.parametrize(
    ("coords", "width", "dtype", "expected_output"),
    [
        pytest.param(
            np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]),
            1.0,
            np.float64,
            np.array(
                [
                    5.00243291e-10,
                    3.30844430e-03,
                    9.99909204e-01,
                    3.30844430e-03,
                    5.00243291e-10,
                ]
            ),
            id="Values inside and outside window",
        ),
        pytest.param(
            thetas_full(),
            10,
            np.float64,
            np.ones_like(thetas_full()),
            id="width is greater than pi",
        ),
    ],
)
def test_window(
    coords: np.ndarray, width: float, dtype: type, expected_output: np.ndarray
) -> np.ndarray:
    """Runs unit tests for window."""
    computed_output = window_equator(coords, width, dtype)
    num_negative = (computed_output < 0).sum()

    assert np.allclose(computed_output, expected_output)
    assert num_negative == 0

@pytest.mark.parametrize(
    ("coords", "width", "dtype"),
    [
        pytest.param(
            thetas_full(),
            -3.0,
            np.float64,
            id="Width is negative."
        ),
        pytest.param(
            thetas_full(),
            0.0,
            np.float64,
            id="width is 0"
        )
    ]
)
def test_error(
    coords: np.ndarray, width: float, dtype: type
) -> None:
    with pytest.raises(ExpectPositiveError):
        window_equator(coords, width, dtype)
