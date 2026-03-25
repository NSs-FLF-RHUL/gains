"""Holds the parameters for scripts/single_spin_up_rotating_frame.py."""

from typing import Any

parameters = dict[str, Any]()
parameters["Nphi"] = 256
parameters["Ntheta"] = 128
parameters["Nr"] = 128
parameters["omega"] = 1
parameters["Delta_Omega"] = 1e-3
parameters["Ek"] = 5e-2

parameters["dealias"] = 3 / 2
parameters["stop_sim_time"] = 20
