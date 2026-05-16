import re
import matplotlib.pyplot as plt

file = "outputs/Vorticity_logging.txt"

with open(file,"r") as f:
    text = f.read()

matches_vals = re.findall(r'max\(omega_s\)=([0-9.eE+-]+)', text)
vorticities = [float(val) for val in matches_vals]
matches_time = re.findall(r'Time=([0-9.eE+-]+)', text)
times = [float(time) for time in matches_time]

plt.scatter(times, vorticities, s = 0.3)
plt.show()
