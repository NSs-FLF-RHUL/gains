"""
Extract maximum vorticity from a logfile and plot it against time.

Configured for use with logging given by two_fluid_spin_up.py and spherical_shell.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from gains.utils.misc import read_logfile

plt.rcParams["savefig.dpi"] = 400

file_approx = Path("outputs/norm_methods_comparison/Approx_norm_log.txt")
file_full = Path("outputs/norm_methods_comparison/Full_norm_log.txt")

times, vorticities_approx = read_logfile(file_approx, "max(omega_s)")
vorticities_full = read_logfile(file_full, "max(omega_s)")[1]
plt.scatter(times, vorticities_approx, s=0.3, label = "Approximation")
plt.scatter(times, vorticities_full, s=0.3, label = "Full normalisation")
plt.xlabel("time")
plt.ylabel(r"$\omega_{s, max}$")
plt.legend()
plt.savefig("outputs/norm_methods_comparison/Both_compare.png")
