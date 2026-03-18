"""Holds the parameters for scripts/single_spin_up_rotating_frame.py."""

from collections import OrderedDict

parameters = OrderedDict()
parameters["Nphi"] = 256
parameters["Ntheta"] = 128
parameters["Nr"] = 128
parameters["omega"] = 1
parameters["Delta_Omega"] = 1e-3
parameters["Ek"] = 5e-3

parameters["dealias"] = 3 / 2
parameters["stop_sim_time"] = 30
parameters["Omega_Init"] = 1.0
