import numpy as np
import pytest

from gains.initial_conditions.single_component_spin_up import window

def thetas_full() -> np.ndarray:
    """Returns array from zero to pi"""
    return np.linspace(0,np.pi,100)


@pytest.fixture
def zeros() -> np.ndarray:
    """Array of zeros for x or y axes."""
    return np.zeros((4,))

@pytest.fixture
def thetas_full_fix() -> np.ndarray:
    """Returns array from zero to pi"""
    return np.linspace(0,np.pi,100)

@pytest.mark.parametrize(
    ("coords", "width", "expected_output"),
    [
        pytest.param(
            thetas_full(),
            0,
            np.zeros_like(thetas_full()),
            id="Width is 0",
            ),
    ],
)
def test_window(
    coords: np.ndarray ,
    width: float, 
    expected_output: np.ndarray) -> np.ndarray:
    
    computed_output = window(coords, width)
    num_negative = (computed_output<0).sum()

    assert np.allclose(computed_output, expected_output)
    assert num_negative == 0
