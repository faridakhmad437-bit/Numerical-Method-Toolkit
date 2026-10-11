
import numpy as np
from collections.abc import Callable

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
)


def rk4(
    f: Callable,
    t0: float,
    y0,
    t_end: float,
    h: float,
):
    """Solve an initial-value problem using the classical RK4 method."""
    validate_callable(f, "f")
    validate_finite_scalar(t0, "t0")
    validate_finite_scalar(t_end, "t_end")
    validate_finite_scalar(h, "h")

    if isinstance(h, bool) or h <= 0:
        raise InvalidInputError("h must be a positive finite number.")

    if t_end < t0:
        raise InvalidInputError("t_end must be >= t0.")

    try:
        y = np.asarray(y0, dtype=float)
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

        def evaluate_derivative(time, state):
            try:
                derivative = np.asarray(f(time, state), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidInputError(
                    "The derivative function must return numeric values."
                ) from exc

            if derivative.shape != y.shape:
                raise InvalidInputError(
                    "The derivative must have the same shape as the state y."
                )

            if not np.all(np.isfinite(derivative)):
                raise NonFiniteValueError(
                    "The derivative function produced a non-finite value."
                )

            return derivative

        with np.errstate(over="ignore", invalid="ignore"):
            k1 = evaluate_derivative(t, y)
            k2 = evaluate_derivative(t + dt / 2, y + dt * k1 / 2)
            k3 = evaluate_derivative(t + dt / 2, y + dt * k2 / 2)
            k4 = evaluate_derivative(t + dt, y + dt * k3)

            y_next = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if not np.all(np.isfinite(y_next)):
            raise NonFiniteValueError(
                "RK4 integration produced a non-finite state."
            )

        t_next = t + dt

        if t_next <= t:
            raise InvalidInputError(
                "The step size is too small to advance time at this scale."
            )

        if dt >= t_end - t:
            t_next = t_end

        t = t_next
        y = y_next

        times.append(t)
        values.append(y.copy())

    return np.asarray(times), np.asarray(values)
