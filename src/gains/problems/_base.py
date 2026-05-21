from typing import TypeAlias

import dedalus.public as d3

from gains.bases.base import BaseBasis

FieldDict: TypeAlias = dict[str, d3.Field]


class BaseProblem:
    """
    Skeleton class that other problems should derive from.

    This class sets out the structure, and provides some utility functions for,
    all derived classes that describe a particular problem that we want to solve
    using `dedalus`.
    """

    _intermediate_fields: FieldDict
    _equation_constants: dict[str, float]

    @classmethod
    def _equation_constant_names(cls) -> tuple[str, ...]:
        return ()

    basis: BaseBasis
    fields: FieldDict
    problem: d3.InitialValueProblem

    def __init__(self, basis: BaseBasis, **params: float) -> None:
        """Construct a problem, to be solved using `dedalus`."""
        self.basis = basis
        self.fields = {}
        self._intermediate_fields = {}

        self._store_constants(**params)
        self.construct_fields()
        self.construct_intermediate_fields()

    def _store_constants(self, **constant_values: float) -> None:
        """
        Store any constants needed by the problem's system of equations.

        Passed keyword arguments are stored if their names match one of the designated
        constants for this class of problem, as dictated by `_equation_constant_names`.
        """
        self._equation_constants = {
            key: value
            for key, value in constant_values.items()
            if key in self._equation_constant_names()
        }

    def add_problem_equations(self) -> None:
        """
        Add equations that define this problem to the `.problem` attribute.

        Method should be explicitly overridden by subclass.
        """
        raise NotImplementedError

    def construct_fields(self) -> None:
        """
        Construct the fields that this problem solves for, adding them to `self.fields`.

        Method should be explicitly overridden by subclass.
        """
        raise NotImplementedError

    def construct_intermediate_fields(self) -> None:
        """
        Construct fields required by the equations, but that aren't to be solved for.

        Method should be explicitly overridden by subclass.
        """
        raise NotImplementedError

    def construct_problem(self) -> None:
        """Define the system of equations that this problem solves."""
        self.problem = d3.IVP(
            list(self.fields.keys()),
            namespace={
                **self.fields,
                **self._equation_constants,
                **self._intermediate_fields,
            },
        )
        self.add_problem_equations()
