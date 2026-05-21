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
    _equation_constants: dict[str, float | d3.Field]

    @classmethod
    def _equation_constant_names(cls) -> tuple[str, ...]:
        """
        Constant values to extract from the `PARAMS` dictionary at instantiation.

        Method should be overridden by subclass when necessary.
        """
        return ()

    basis: BaseBasis
    fields: FieldDict
    problem: d3.InitialValueProblem

    def __init__(self, basis: BaseBasis, **params: float) -> None:
        """Construct a problem, to be solved using `dedalus`."""
        self.basis = basis
        self.fields = {}
        self._intermediate_fields = {}

        # Note that _store_given_constants initialises _equation_constants
        self._store_given_constants(**params)
        self._store_default_constants()
        self._store_derived_constants()

        self.construct_fields()
        self.construct_intermediate_fields()

        self._construct_problem()

    def _construct_problem(self) -> None:
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

    def _store_default_constants(self) -> None:
        """
        Add frequently-used constant values to the equation namespace.

        Explicitly, this adds;

        - The `dedalus` `cross`, `curl` and `dot` operators.
        - The spherical unit vectors `er`, `etheta`, and `ephi`.
        """
        self._equation_constants["cross"] = d3.CrossProduct
        self._equation_constants["curl"] = d3.Curl
        self._equation_constants["dot"] = d3.DotProduct

        er, etheta, ephi = self.basis.unit_vectors()
        self._equation_constants["er"] = er
        self._equation_constants["etheta"] = etheta
        self._equation_constants["ephi"] = ephi

    def _store_derived_constants(self) -> None:
        """
        Construct any constants we need for the problem's system of equations.

        Unlike `_store_given_constants`, these constants should be objects, values, or
        other static instances that we can derive from the constants we have already
        been given, or any other attributes of the instance itself.

        By default, the `dedalus` `cross`, `curl`, and `dot` operations are added to the
        equation namespace at creation. But it can be overridden or extended by
        subclasses if necessary.
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

    def field_projection(self, field_name: str) -> tuple[d3.Field, d3.Field, d3.Field]:
        r"""
        Return the projection of the field onto the spherical unit vectors.

        Projections are returned in the order: $r, \theta, \phi$.
        """
        er, etheta, ephi = self.get_spherical_units()
        target_field = self.fields[field_name]

        return (
            d3.DotProduct(target_field, er),
            d3.DotProduct(target_field, etheta),
            d3.DotProduct(target_field, ephi),
        )

    def get_spherical_units(self) -> tuple[d3.Field, d3.Field, d3.Field]:
        r"""
        Get the spherical unit vectors being used in this problem.

        Unit vectors are returned in the order: $r, \theta, \phi$.
        """
        return (
            self._equation_constants["er"],
            self._equation_constants["etheta"],
            self._equation_constants["ephi"],
        )

    def new_field(self, *args, **kwargs) -> d3.Field:
        """
        Create a scalar field attached to the basis of this problem.

        Thin wrapper around `BaseBasis.field`.
        """
        return self.basis.field(*args, **kwargs)

    def new_vector_field(self, *args, **kwargs) -> d3.Field:
        """
        Create a vector field attached to the basis of this problem.

        Thin wrapper around `BaseBasis.vector_field`.
        """
        return self.basis.vector_field(*args, **kwargs)
