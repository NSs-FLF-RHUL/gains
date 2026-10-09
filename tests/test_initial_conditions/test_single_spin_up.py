import numpy as np
import pytest

from gains.initial_conditions.single_component_spin_up import (
    ExpectPositiveError,
    circle_on_sphere,
    mask_angular,
)


def thetas_full() -> np.ndarray:
    """Returns array from zero to pi."""
    return np.linspace(0, np.pi, 100)


def phis_full() -> np.ndarray:
    """Returns array from zero to pi."""
    return np.linspace(0, 2 * np.pi, 100)


@pytest.fixture
def zeros() -> np.ndarray:
    """Array of zeros for x or y axes."""
    return np.zeros((4,))


@pytest.fixture
def thetas_full_fix() -> np.ndarray:
    """Returns array from zero to pi."""
    return np.linspace(0, np.pi, 100)


@pytest.mark.parametrize(
    ("coords", "width", "center", "expected_output"),
    [
        pytest.param(
            np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]),
            1.0,
            np.pi / 2,
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
            np.pi / 2,
            np.ones_like(thetas_full()),
            id="width is greater than pi",
        ),
    ],
)
def test_window(
    coords: np.ndarray, width: float, center: float, expected_output: np.ndarray
) -> np.ndarray:
    """Runs unit tests for window."""
    computed_output = mask_angular(coords, width, center)
    num_negative = (computed_output < 0).sum()

    assert np.allclose(computed_output, expected_output)
    assert num_negative == 0


@pytest.mark.parametrize(
    ("coords", "width", "dtype"),
    [
        pytest.param(thetas_full(), -3.0, np.float64, id="Width is negative."),
        pytest.param(thetas_full(), 0.0, np.float64, id="width is 0"),
    ],
)
def test_error_mask_angular(coords: np.ndarray, width: float, dtype: type) -> None:
    """Confirms correct error is raised if width not configured correctly."""
    with pytest.raises(ExpectPositiveError):
        mask_angular(coords, width, dtype)


@pytest.mark.parametrize(
    ("theta", "phi", "radius", "center"),
    [
        pytest.param(thetas_full(), phis_full(), -3.0, (0, 0), id="Width is negative."),
        pytest.param(thetas_full(), phis_full(), 0.0, (0, 0), id="width is 0"),
    ],
)
def test_error_circle_on_sphere(
    theta: np.ndarray, phi: np.ndarray, radius: float, center: tuple[float, float]
) -> None:
    """Confirms correct error is raised if width not configured correctly."""
    with pytest.raises(ExpectPositiveError):
        circle_on_sphere(theta, phi, radius, center)


@pytest.mark.parametrize(
    ("theta", "phi", "centre", "radius", "expected_gamma"),
    [
        pytest.param(
            thetas_full(),
            np.zeros(100),
            (0.0, 0.0),
            1.0,
            thetas_full(),
            id="Check returned mask across full theta range.",
        ),
        pytest.param(
            np.pi / 2 * np.ones(100),
            phis_full(),
            (np.pi / 2, 0.0),
            1.0,
            np.concatenate((phis_full()[0:50], phis_full()[49::-1])),
            id="Check returned mask across full phi range",
        ),
    ],
)
def test_circle_on_sphere_gamma(
    theta: np.ndarray,
    phi: np.ndarray,
    centre: tuple[float, float],
    expected_gamma: np.ndarray,
    radius: float,
) -> None:
    """
    Run unit tests for circle_on_sphere.

    Centres and coordinates are selected such that the expected great circle distances
    are some variation of theta or phi.
    """
    expected_mask = np.exp(-(expected_gamma**2) / (2 * radius))

    computed_mask = circle_on_sphere(theta, phi, radius, centre)

    assert np.allclose(expected_mask, computed_mask)
