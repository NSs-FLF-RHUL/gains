"""Problem setup for single-spin up, rotating frame."""

import numpy as np

from gains.bases.spherical import SphericalBasis
from gains.initial_conditions.single_component_spin_up import window_equator
from gains.problems._base import BaseProblem


class SingleSpinUpRotatingFrameProblem(BaseProblem):
    """Single spin-up rotating frame problem."""

    basis: SphericalBasis

    @classmethod
    def _equation_constant_names(cls) -> tuple[str, ...]:
        """
        Constant values to extract from the `PARAMS` dictionary at instantiation.

        Method should be overridden by subclass when necessary.
        """
        return ("Delta_Omega", "Ek")

    def _store_derived_constants(self) -> None:
        self._equation_constants["radius"] = self.basis.radius
        self._equation_constants["lift"] = self.basis.lift_operator()
        self._equation_constants["ez"] = self.basis.unit_ez()

    def add_problem_equations(self) -> None:
        """Define the system of equations for this problem."""
        self.problem.add_equation("div(u_n) + tau_p_n = 0")
        self.problem.add_equation(
            "dt(u_n) + grad(p_n) - Ek*lap(u_n) + lift(tau_u_n) = "
            "-u_n@grad(u_n) -2*cross(ez,u_n)"
        )
        # Spin up at outer boundary
        self.problem.add_equation(
            "angular(u_n(r=radius)) = "
            "mask*angular(uang_r1) + (1-mask)*angular(u_n(r=radius))"
        )
        # Impenetrable bc
        self.problem.add_equation("radial(u_n(r=radius)) = 0")
        # Pressure gauge normal fluid
        self.problem.add_equation("integ(p_n) = 0")

    def construct_fields(self) -> None:
        """Construct named fields to solve for in this problem."""
        self.fields["u_n"] = self.new_vector_field(name="u_n", bases=self.basis.ball)
        self.fields["p_n"] = self.new_field(name="p_n", bases=self.basis.ball)

        self.fields["tau_p_n"] = self.new_field(name="tau_p_n")
        self.fields["tau_u_n"] = self.new_vector_field(
            name="tau_u_n", bases=self.basis.sphere
        )

    def construct_intermediate_fields(self) -> None:
        """Construct fields required by the system of equations."""
        self._intermediate_fields["omega_n"] = self.new_vector_field(
            name="omega_n", bases=self.basis.ball
        )
        self._intermediate_fields["tau_omega_n"] = self.new_vector_field(
            name="tau_omega_n", bases=self.basis.sphere
        )

        _, theta, _ = self.basis.dist.local_grids(self.basis.ball)

        self._intermediate_fields["sintheta"] = self.new_field(
            name="sintheta", bases=self.basis.ball
        )
        self._intermediate_fields["sintheta"]["g"] = np.sin(theta)

        self._intermediate_fields["mask"] = self.new_field(
            name="mask", bases=self.basis.sphere
        )
        self._intermediate_fields["mask"]["g"] = window_equator(theta, 0.5, np.float64)

        radius = self._equation_constants["radius"]
        delta_omega = self._equation_constants["Delta_Omega"]
        sintheta = self._intermediate_fields["sintheta"]
        self._intermediate_fields["uang_r1"] = self.new_vector_field(
            bases=self.basis.ball
        )(r=radius).evaluate()
        self._intermediate_fields["uang_r1"]["g"][0, :] = (delta_omega * sintheta)(
            r=radius
        ).evaluate()["g"]
