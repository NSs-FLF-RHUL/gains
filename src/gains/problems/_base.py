from gains.bases.base import BaseBasis


class _BaseProblem:
    """
    Skeleton class that other problems should derive from.

    This class sets out the structure, and provides some utility functions for,
    all derived classes that describe a particular problem that we want to solve
    using `dedalus`.
    """

    basis: BaseBasis
    fields: dict
