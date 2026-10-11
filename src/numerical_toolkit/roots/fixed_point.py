
import math
from collections.abc import Callable

from numerical_toolkit.core.exceptions import NonFiniteValueError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
)


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """Solve x = g(x) using fixed-point iteration."""
    validate_callable(g, "g")
    validate_finite_scalar(x0, "x0")

    o = options or SolverOptions()
    x = float(x0)
    history = []
    calls = 0
    previous_step = None

    gx = float(g(x))
    calls += 1

    if not math.isfinite(gx):
        raise NonFiniteValueError("g(x) produced a non-finite value.")

    residual = abs(gx - x)

    if residual <= o.ftol:
        return SolverResult(
            x, True, 0, calls, residual, 0.0,
            "Initial guess satisfies residual tolerance.",
            tuple(history) if o.store_history else None,
        )

    for k in range(1, o.maxiter + 1):
        xn = gx
        step = abs(xn - x)
        rel = step / max(1.0, abs(xn))

        if o.store_history:
            history.append(xn)

        x = xn
        gx = float(g(x))
        calls += 1

        if not math.isfinite(gx):
            raise NonFiniteValueError("g(x) produced a non-finite value.")

        residual = abs(gx - x)

        if residual <= o.ftol:
            return SolverResult(
                x, True, k, calls, residual, step,
                "Converged successfully: residual tolerance satisfied.",
                tuple(history) if o.store_history else None,
            )

        if step <= o.xtol or rel <= o.rtol:
            return SolverResult(
                x, True, k, calls, residual, step,
                "Converged successfully: step tolerance satisfied.",
                tuple(history) if o.store_history else None,
            )

        previous_step = step

    return SolverResult(
        x, False, o.maxiter, calls, residual, previous_step,
        "Maximum number of iterations reached.",
        tuple(history) if o.store_history else None,
    )
