class NumericalMethodError(Exception):
    """Base exception for Numerical Methods Toolkit."""


class InvalidInputError(NumericalMethodError):
    """Raised when solver input is invalid."""


class InvalidBracketError(InvalidInputError):
    """Raised when a root-finding bracket is invalid."""


class DerivativeZeroError(NumericalMethodError):
    """Raised when a Newton derivative becomes zero."""

class SecantDenominatorZeroError(NumericalMethodError):
    """Raised when the Secant denominator becomes zero."""

class NonFiniteValueError(NumericalMethodError):
    """Raised when an algorithm encounters NaN or infinity."""


class ConvergenceError(NumericalMethodError):
    """Raised for explicit convergence failures."""