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

        self._store_given_constants(**params)
        self._store_derived_constants()
        self.construct_fields()
        self.construct_intermediate_fields()

    def _store_derived_constants(self) -> None:
        """
        Construct any constants we need for the problem's system of equations.

        Unlike `_store_given_constants`, these constants should be objects, values, or
        other static instances that we can derive from the constants we have already
        been given, or any other attributes of the instance itself.

        By default, the method simply passes. But it can be overridden by subclasses if
        necessary.
        """

    def _store_given_constants(self, **constant_values: float) -> None:
        """
        Store any constants needed by the problem's system of equations.

        Constants may be any static values. This can include callable objects, or static
        class instances.

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
