"""
Solves and plots a 1D problem.

Solves for the steady state solution for a fluid flowing between 2 no slip
walls subject to a constant pressure gradient.

Solves the LBVP:
    P + mu*dy(dy(u)) = 0
    u(y=h) = 0
    u(y=-h) = 0

Where P is the pressure gradient (ie -dy(p)=P) and mu is the dynamic
viscosiy.

Need 2 tau terms to impose boundary conditions, which are implemented via a
first order reduction, leading to the system of equaions:
    uy - dy(u) + tau_1 = 0
    dy(uy) + tau_2 = -P/mu


"""

import argparse
import datetime
from pathlib import Path

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.dpi"] = 400

parser = argparse.ArgumentParser(
    description="Solve for steady state flow between 2 walls subject "
    "to a constant pressure gradient"
)

parser.add_argument("--Ny", type=int, default=128, help="y resolution")

parser.add_argument(
    "--viscosity", type=float, default=1.0, help="The dynamic viscosity of the fluid"
)

parser.add_argument(
    "--Pressure_gradient",
    type=float,
    default=2.0,
    help="The constant pressure gradient.",
)

parser.add_argument(
    "--height",
    type=float,
    default=3.0,
    help="The distance between y=0 and the planes (ie one at -h and one at +h)",
)

parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="Name of the output files.",
)

args = vars(parser.parse_args())
# Parameters

PARAMS = {
    "Ly": args["height"],
    "Pgrad": args["Pressure_gradient"],
    "mu": args["viscosity"],
    "Ny": args["Ny"],
    "name": args["name"]
    if args["name"] is not None
    else "poiseuille_flow_"
    + datetime.datetime.now().astimezone().strftime("%Y-%m-%m-%H:%M")
}

path_new = Path("outputs") / PARAMS["name"]
path_new.mkdir(parents=True, exist_ok=True)


dtype = np.float64

# Bases
ycoord = d3.Coordinate("y")
dist = d3.Distributor(ycoord, dtype=dtype)
ybasis = d3.Chebyshev(ycoord, size=PARAMS["Ny"], bounds=(-PARAMS["Ly"], PARAMS["Ly"]))

# Fields
u = dist.Field(name="u", bases=ybasis)
uy = dist.Field(name="uy", bases=ybasis)
f = dist.Field(bases=ybasis)
f["g"] = -PARAMS["Pgrad"] / PARAMS["mu"]
tau_1 = dist.Field(name="tau_1")
tau_2 = dist.Field(name="tau_2")


# Substitutions
def dy(a: d3.Field) -> d3.Field:
    """Return derivative of field."""
    return d3.Differentiate(a, ycoord)


lift_basis = ybasis.derivative_basis(2)


def lift(a: d3.Field, n: int) -> d3.Field:
    """Lift operand to -nth derivative basis."""
    return d3.Lift(a, lift_basis, n)


# Problem
problem = d3.LBVP([u, tau_1, tau_2], namespace=locals())
problem.add_equation("dy(dy(u)) + lift(tau_1,-1) + lift(tau_2,-2) = -f")
problem.add_equation("u(y=-PARAMS['Ly']) = 0")
problem.add_equation("u(y=PARAMS['Ly']) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

# Analysis
y = ybasis.global_grid(dist, scale=1)
ug = -1 * u.allgather_data("g")


def u_analytic() -> np.ndarray:
    """Analytic solution for Poiseuille flow."""
    return PARAMS["Pgrad"] / (2 * PARAMS["mu"]) * (PARAMS["Ly"] ** 2 - y**2)


u_an = u_analytic()
u_err = (ug - u_an) / u_an

plt.figure(1)
plt.plot(ug, y, label="numerical solution", color="black")
plt.plot(u_an, y, linestyle="dashed", label="analytic solution", color="red")
plt.legend()
plt.xlabel("u(y)")
plt.ylabel("y")
plt.savefig(path_new / "flow_solution.png")

plt.figure(2)

plt.scatter(y, u_err, s=4, color="black")
plt.xlabel("y")
plt.ylabel("relative error")
plt.savefig(path_new / "relative_error.png")
