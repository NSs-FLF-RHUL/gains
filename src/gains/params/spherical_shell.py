"""Holds the default parameters for scripts/single_spin_up_rotating_frame.py."""

from typing import Any

parameters: dict[str, Any] = {
    "Nphi": 128,
    "Ntheta": 64,
    "Nr": 64,
    "omega": 1,
    "Delta_Omega": 1e-3,
    "Ek": 5e-2,
    "Ri": 0.5,
    "Ro": 1.0,
    "B": 0.1,
    "dealias": 3 / 2,
    "stop_sim_time": 60,
}

# Lower resolution for test_analysis/test_spectral_loading.py
parameters_test: dict[str, Any] = {
    "Nphi": 32,
    "Ntheta": 16,
    "Nr": 16,
    "omega": 1,
    "Delta_Omega": 1e-3,
    "Ek": 5e-2,
    "Ri": 0.5,
    "Ro": 1.0,
    "B": 0.1,
    "dealias": 3 / 2,
    "stop_sim_time": 60,
}
