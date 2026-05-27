"""Holds the default parameters for scripts/single_spin_up_rotating_frame.py."""

from typing import Any

<<<<<<< HEAD
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
=======
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
>>>>>>> main
