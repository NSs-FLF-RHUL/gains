"""
Extract maximum vorticity from a logfile and plot it against time.

Configured for use with logging given by two_fluid_spin_up.py and spherical_shell.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from gains.utils.misc import read_logfile

plt.rcParams["savefig.dpi"] = 400

file = Path("outputs/Vorticity_log_long.txt")


times, vorticities = read_logfile(file, "max(omega_s)")
plt.scatter(times, vorticities, s=0.3)
plt.xlabel("time")
plt.ylabel(r"$\omega_{s, max}$")
plt.savefig("max_vorticity_agains_time.png")
