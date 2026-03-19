"""Holds the parameters for scripts/single_spin_up_rotating_frame.py."""

from collections import OrderedDict

parameters = OrderedDict()
parameters["Nphi"] = 256
parameters["Ntheta"] = 128
parameters["Nr"] = 128
parameters["omega"] = 1
parameters["Delta_Omega"] = 1e-3  # type: ignore[assignment]
parameters["Ek"] = 5e-2  # type: ignore[assignment]

parameters["dealias"] = 3 / 2  # type: ignore[assignment]
parameters["stop_sim_time"] = 20
parameters["Omega_Init"] = 1.0  # type: ignore[assignment]
