
import math
from collections.abc import Callable
from numbers import Integral

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
)


def simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """Approximate a definite integral using composite Simpson's 1/3 rule."""
    validate_callable(f, "f")
    a = validate_finite_scalar(a, "a")
    b = validate_finite_scalar(b, "b")

    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 or n % 2 != 0:
        raise InvalidInputError("n must be a positive even integer.")

    n = int(n)

    h = (b - a) / n

    fa = float(f(a))
    validate_finite_scalar(fa, "f(a)")

    fb = float(f(b))
    validate_finite_scalar(fb, "f(b)")

    total = fa + fb

    for i in range(1, n):
        x = a + i * h
        fx = float(f(x))
        validate_finite_scalar(fx, "f(x)")

        weight = 4 if i % 2 != 0 else 2
        total += weight * fx

    result = h * total / 3

    if not math.isfinite(result):
        raise NonFiniteValueError("Integration result is non-finite.")

    return result
