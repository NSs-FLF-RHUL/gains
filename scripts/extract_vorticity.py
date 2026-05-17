import matplotlib.pyplot as plt
from pathlib import Path
from gains.utils.misc import read_logfile

plt.rcParams["savefig.dpi"] = 400

file = Path("outputs/Vorticity_log_long.txt")


times, vorticities = read_logfile(file, "max(omega_s)")
print(vorticities)
plt.scatter(times, vorticities, s = 0.3)
plt.xlabel("time")
plt.ylabel(r"$\omega_{s, max}$")
plt.show()
#plt.savefig("max_vorticity_agains_time.png")
