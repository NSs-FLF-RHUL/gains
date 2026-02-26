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

import logging

import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
plt.rcParams["savefig.dpi"] = 400

# Parameters
Ly = 3  # h=3
P = 2
mu = 1
Ny = 128
dtype = np.float64

# Bases
ycoord = d3.Coordinate("y")
dist = d3.Distributor(ycoord, dtype=dtype)
ybasis = d3.Chebyshev(ycoord, size=Ny, bounds=(-Ly, Ly))

# Fields
u = dist.Field(name="u", bases=ybasis)
uy = dist.Field(name="uy", bases=ybasis)
f = dist.Field(bases=ybasis)
f["g"] = -P / mu
tau_1 = dist.Field(name="tau_1")
tau_2 = dist.Field(name="tau_2")


# Substitutions
def dy(A):
    return d3.Differentiate(A, ycoord)


lift_basis = ybasis.derivative_basis(2)


def lift(A, n):
    return d3.Lift(A, lift_basis, n)


# Problem
problem = d3.LBVP([u, tau_1, tau_2], namespace=locals())
problem.add_equation("dy(dy(u)) + lift(tau_1,-1) + lift(tau_2,-2) = -f")
problem.add_equation("u(y=-Ly) = 0")
problem.add_equation("u(y=Ly) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

# Analysis
y = ybasis.global_grid(dist, scale=1)
ug = -1 * u.allgather_data("g")
u_an = 9 - y**2
u_err = (ug - u_an) / u_an

plt.figure(1)
plt.plot(ug, y, label="numerical solution", color="black")
plt.plot(u_an, y, linestyle="dashed", label="analytic solution", color="red")
plt.legend()
plt.xlabel("u(y)")
plt.ylabel("y")
plt.savefig("flow_solution.png")

plt.figure(2)

plt.scatter(y, u_err, s=4, color="black")
plt.xlabel("y")
plt.ylabel("relative error")
plt.savefig("relative_error.png")
