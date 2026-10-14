import numpy as np
import scipy.optimize as scpo
from scipy.signal import find_peaks
from gains.utils.parsers import SimulationCLI
from gains.params.single_spin_up_rotating import parameters as default_params
import logging
from gains.analysis.analyse_spin_up import get_angular_speed_vs_time, get_angular_coords, LabeledCoordinate
from gains.utils.misc import extract_numerical_suffix
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
parser = SimulationCLI(
    profiling_option=True,
    place_all_outputs_under="outputs",
    sim_name="two_fluid_spin_up",
)
PARAMS = parser.parse_args_and_get_params(logger, default_params=default_params)

def spin_up_exp(t, tau_su, delta_omega):
    return delta_omega*(1-np.exp(-t/tau_su))

r_check, theta_check = get_angular_coords(PARAMS["output_dir"] / "AZ_avg_equator_s1.h5", "u_n_phi")

r_label = LabeledCoordinate(r_check, "r")
path_list = sorted(
    (p for p in Path(PARAMS["output_dir"]).iterdir() if p.suffix == ".h5"),
    key=extract_numerical_suffix)

omega_r, times = get_angular_speed_vs_time(r_label, 0.8, "u_n_phi", PARAMS["nwrites"], path_list, PARAMS["Ntheta"])

popt, pcov = scpo.curve_fit(lambda t, tau_sf: spin_up_exp(t, tau_sf, PARAMS["Delta_Omega"]),
                            times,
                            omega_r)
perr = np.sqrt(np.diag(pcov))
pcorr = pcov / np.outer(perr, perr)

print(f"tau_su = {popt[0]}" + " \u00b1 " + f"{np.sqrt(pcov[0, 0])}")

fig, axs = plt.subplots(2,1, height_ratios=[3,1], sharex=True, figsize=(8,8))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(times, omega_r, label="data", color = "#3594cc")
ax1.plot(times, spin_up_exp(times, popt[0], PARAMS["Delta_Omega"]), label = "fitted", color = "#ea7f1b", linestyle = "dashed")
ax1.legend()

residuals = omega_r - spin_up_exp(times, popt[0], PARAMS["Delta_Omega"])

ax2.scatter(times, residuals, s=0.5, color="#666666")
ax2.axhline(linestyle = "dashed", color="#D3D3D3", linewidth=0.8)
#ax2.plot(times, 0.0005*np.sin(2*np.pi*0.1495*times))

plt.subplots_adjust(hspace=0.05)
fig.savefig("exponential_fit.png")
plt.close(fig)
#FFT processing for oscillatory behaiviour

fft_mags = np.abs(np.fft.rfft(residuals))
frequencies = np.fft.rfftfreq(len(residuals), d=1.0)


peaks, properties = find_peaks(fft_mags, prominence=0.0001)

plt.plot(frequencies, fft_mags, label = "FFT spectrum of residuals")


for i, peak in enumerate(peaks):
    if frequencies[peak] > 0.0005:
        print(f"Peak at f={frequencies[peak]:.4f}")
        if i == 0:
            plt.scatter(frequencies[peak], fft_mags[peak],color="red", label="identified peaks")
        else:
            plt.scatter(frequencies[peak], fft_mags[peak],color="red")

plt.legend()
plt.savefig("residual_fft.png")
