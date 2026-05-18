"""Holds the default parameters for scripts/single_spin_up_rotating_frame.py."""

from typing import Any

parameters = dict[str, Any]()
parameters["Nphi"] = 128
parameters["Ntheta"] = 64
parameters["Nr"] = 64
parameters["omega"] = 1
parameters["Delta_Omega"] = 1e-3
parameters["Ek"] = 5e-2
parameters["Ri"] = 0.5
parameters["Ro"] = 1.0
parameters["B"] = 0.1

parameters["dealias"] = 3 / 2
parameters["stop_sim_time"] = 60