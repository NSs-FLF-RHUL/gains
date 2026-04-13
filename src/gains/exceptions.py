"""Holds custom exceptions used in the package."""

class MeshError(Exception):
    """Exception for negative values in instances they should be positive."""

    def __init__(self) -> None:
        """Error message."""
        super().__init__("Number of cpus should be a power of 2")


class ExpectPositiveError(Exception):
    """Exception for negative values in instances they should be positive."""

    def __init__(self, var: str | float) -> None:
        """:param var: The variable that should be positive."""
        super().__init__(f"{var} should be positive.")
