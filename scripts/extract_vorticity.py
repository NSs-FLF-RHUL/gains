"""
Extract maximum vorticity from a logfile and plot it against time.

Configured for use with logging given by two_fluid_spin_up.py and spherical_shell.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from gains.utils.misc import read_logfile

plt.rcParams["savefig.dpi"] = 400

parser = argparse.ArgumentParser(
    description="Extract vorticity from logfiles using the full normalisation, and "
    "plot them both against time."
)

parser.add_argument(
    "--full_log",
    type=str,
    help="Path to the logfile using the full normalisation",
    required=True,
)

parser.add_argument(
    "--approx_log",
    type=str,
    help="Path to the logfile using the approximated normalisation",
    required=True,
)

parser.add_argument(
    "--save_dir",
    type=str,
    help="Where to save te figure, including name and file format.",
    required=True,
)

paths = vars(parser.parse_args())

file_approx = Path(paths["approx_log"])
file_full = Path(paths["full_log"])

times_approx, vorticities_approx = read_logfile(file_approx, "max(omega_s)")
times_full, vorticities_full = read_logfile(file_full, "max(omega_s)")
plt.plot(times_approx, vorticities_approx, label="dt=5e-3", color="#0072B2")
plt.plot(times_full, vorticities_full, label="dt=1e-2", color="#D55E00")
plt.xlabel("time")
plt.ylabel(r"$\omega_{s, max}$")
plt.legend()
plt.savefig(paths["save_dir"])
#plt.show()
