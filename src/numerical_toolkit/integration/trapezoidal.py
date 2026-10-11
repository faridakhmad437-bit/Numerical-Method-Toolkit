
import math
from collections.abc import Callable

from numerical_toolkit.core.exceptions import InvalidInputError, NonFiniteValueError
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
)


def trapezoidal(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """Approximate a definite integral using the composite trapezoidal rule."""
    validate_callable(f, "f")
    validate_finite_scalar(a, "a")
    validate_finite_scalar(b, "b")

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise InvalidInputError("n must be a positive integer.")

    h = (b - a) / n

    fa = float(f(a))
    validate_finite_scalar(fa, "f(a)")

    fb = float(f(b))
    validate_finite_scalar(fb, "f(b)")

    total = 0.5 * (fa + fb)

    for i in range(1, n):
        x = a + i * h
        fx = float(f(x))
        validate_finite_scalar(fx, "f(x)")
        total += fx

    result = h * total

    if not math.isfinite(result):
        raise NonFiniteValueError("Integration result is non-finite.")

    return result
