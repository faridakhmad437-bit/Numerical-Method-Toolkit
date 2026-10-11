
from collections.abc import Callable

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
)


def euler(
    f: Callable,
    t0: float,
    y0,
    t_end: float,
    h: float,
):
    """Solve an initial-value problem using the explicit Euler method."""
    validate_callable(f, "f")
    validate_finite_scalar(t0, "t0")
    validate_finite_scalar(t_end, "t_end")
    validate_finite_scalar(h, "h")

    if isinstance(h, bool) or h <= 0:
        raise InvalidInputError("h must be a positive finite number.")

    if t_end < t0:
        raise InvalidInputError("t_end must be >= t0.")

    try:
        raw_y0 = np.asarray(y0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError("y0 must contain valid numeric values.") from exc

    if raw_y0.dtype.kind == "b":
        raise InvalidInputError("y0 must contain numeric, non-boolean values.")

    try:
        y = np.asarray(raw_y0, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError("y0 must contain valid numeric values.") from exc

    if not np.all(np.isfinite(y)):
        raise NonFiniteValueError("Initial state y0 must be finite.")

    t = float(t0)
    t_end = float(t_end)
    h = float(h)

    times = [t]
    values = [y.copy()]

    while t < t_end:
        dt = min(h, t_end - t)

        try:
            derivative = np.asarray(f(t, y), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(
                "The derivative function must return numeric values."
            ) from exc

        if not np.all(np.isfinite(derivative)):
            raise NonFiniteValueError(
                "The derivative function produced a non-finite value."
            )

        if derivative.shape != y.shape:
            raise InvalidInputError(
                "The derivative must have the same shape as the state y."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            y_next = y + dt * derivative

        if not np.all(np.isfinite(y_next)):
            raise NonFiniteValueError(
                "Euler integration produced a non-finite state."
            )

        t_next = t + dt

        # Avoid an infinite loop if floating-point precision prevents progress.
        if t_next <= t:
            raise InvalidInputError(
                "The step size is too small to advance time at this scale."
            )

        # Snap the final time to t_end when the remaining interval is consumed.
        if dt >= t_end - t:
            t_next = t_end

        t = t_next
        y = y_next

        times.append(t)
        values.append(y.copy())

    return np.asarray(times), np.asarray(values)
