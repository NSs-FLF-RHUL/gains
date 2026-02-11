"""
Solves the incompressible euler equations in 2D to reproduce the kelvin
helmholtz instability in a shear flow.

Initialised with a real fourier basis to impose periodic boundary conditons

Initial and boundary conditions based on those described by McNally et al., 2012, ApJ, 201, 18
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

from gains.initial_conditions.mcnally import density

logger = logging.getLogger(__name__)
plt.rcParams["savefig.dpi"] = 400

# Parameters

dtype = np.float64
PARAMS = {
    "Lx": 1,
    "Ly": 1,
    "Nx": 16,
    "Ny": 16,
    "timestepper": d3.SBDF4,
    "stop_sim_time": 4.5,
    "max_timestep": 1e-4,
    "dealias": 2,
    "gamma": 5 / 3,
    "rho_1": 1.0,
    "rho_2": 2.0,
    "L": 0.025,
    "U_1": 0.5,
    "U_2": -0.5,
    "nu": 0.0001,  # Artifical viscosity to prevent shocks
}
rho_m = (PARAMS["rho_1"] - PARAMS["rho_2"]) / 2
PARAMS["rho_m"] = rho_m
U_m = (PARAMS["U_1"] - PARAMS["U_2"]) / 2
PARAMS["U_m"] = U_m
print(PARAMS)

# Bases

coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(
    coords["x"], size=PARAMS["Nx"], bounds=(0, PARAMS["Lx"]), dealias=PARAMS["dealias"]
)
ybasis = d3.RealFourier(
    coords["y"], size=PARAMS["Ny"], bounds=(0, PARAMS["Ly"]), dealias=PARAMS["dealias"]
)
print(xbasis)

# Fields

u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
p = dist.Field(name="p", bases=(xbasis, ybasis))
rho = dist.Field(name="rho", bases=(xbasis, ybasis))
e = dist.Field(name="e", bases=(xbasis, ybasis))  # Specific internal energy
s = dist.Field(name="s", bases=(xbasis, ybasis))  # Entropy
tau_p = dist.Field(name="tau_p")

# Substitutions

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
print("shape of x: {}".format(np.shape(x)))
print("shape of y: {}".format(np.shape(y)))
# Problem
problem = d3.IVP([u, rho, p, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(rho) = - u@grad(rho)")
problem.add_equation("dt(u) + grad(p) = (PARAMS['nu']/rho)*lap(u)- u@grad(u)")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(PARAMS["timestepper"])
solver.stop_sim_time = PARAMS["stop_sim_time"]

# Initial conditions - see McNally et al., 2012, ApJ, 201, 18 for more details

# density
rho_y = density(y[0], **PARAMS)

rho_init = np.zeros((len(x), len(y[0])))
# print(len(y))
for counter, value in enumerate(rho_y):
    rho_init[counter] = [value for i in rho_init[counter]]

print(np.shape(rho_init))

plt.pcolormesh(x.ravel(), y.ravel(), np.array(rho_init))
plt.title("Density distribution")
plt.colorbar()
plt.show()

rho["g"] = np.transpose(np.array(rho_init))


# x velocity
def v_x(xs):
    out = []
    for el in xs:
        if el < 0.25:
            out.append(PARAMS["U_1"] - U_m * np.exp((el - 0.25) / PARAMS["L"]))
        elif 0.25 <= el < 0.5:
            out.append(PARAMS["U_2"] + U_m * np.exp((-el + 0.25) / PARAMS["L"]))
        elif 0.5 <= el < 0.75:
            out.append(PARAMS["U_2"] + U_m * np.exp(-(0.75 - el) / PARAMS["L"]))
        else:
            out.append(PARAMS["U_1"] - U_m * np.exp(-(el - 0.75) / PARAMS["L"]))
    return out


v_xs = v_x(x)
vxs_init = np.zeros((len(x), len(y[0])))
v_xs = [
    v_xs[i][0] for i in range(0, len(v_xs))
]  # evil list comprehension to avoid an array of arrays

for counter, value in enumerate(v_xs):
    vxs_init[counter] = [
        value for i in vxs_init[counter]
    ]  # More evil list comprehension to produce a matrix where each column is the same

# plt.pcolormesh(x.ravel(), y.ravel(), vxs_init)
# plt.title("vx distribution")
# plt.show()

u["g"][0] = np.array(v_xs)

# y velocity perturbations

vys = 0.01 * np.sin(4 * np.pi * x)

vys = [vys[i][0] for i in range(0, len(vys))]


vys_init = np.zeros((len(x), len(y[0])))

for j in range(0, len(y[0])):
    vys_init[j] = vys

# plt.pcolormesh(x.ravel(), y.ravel(), vys_init)
# plt.title('vy distribution')
# plt.show()

u["g"][1] += 0.01 * np.sin(4 * np.pi * x)
print("u['g'] shape: {}".format(np.shape(u["g"])))
# Internal energy set to give a uniform pressure of 2.5. Internal energy has been eliminated from equations so just set
# p = 2.5

p_init = np.zeros((len(x), len(y[0])))

for i in range(0, len(x)):
    for j in range(0, len(y[0])):
        p_init[i][j] = 2.5

# plt.pcolormesh(x.ravel(),y.ravel(),p_init)
# plt.title('initial pressure distribution')
# plt.show()


# Entropy initialised via ideal gas law
s_init = np.array(p_init) * np.array(rho_init) ** (-PARAMS["gamma"])

print(rho_init)


""" plt.pcolormesh(x.ravel(),y.ravel(),s_init)
plt.title('entropy initial')
plt.colorbar()
plt.show() """

# Analysis
snapshots = solver.evaluator.add_file_handler("snapshots", sim_dt=1e-2, max_writes=10)
snapshots.add_task(s, name="entropy")
snapshots.add_task(rho, name="density")

# CFL
CFL = d3.CFL(
    solver,
    initial_dt=PARAMS["max_timestep"],
    cadence=10,
    safety=0.2,
    threshold=0.1,
    max_change=1.5,
    min_change=0.5,
    max_dt=PARAMS["max_timestep"],
)
CFL.add_velocity(u)

# Main loop
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10000 == 0:
            logger.info(
                "Iteration=%i, Time=%e, dt=%e"
                % (solver.iteration, solver.sim_time, timestep)
            )
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
