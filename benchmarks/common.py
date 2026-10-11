
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from numerical_toolkit.core.results import SolverResult


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured performance and accuracy information for one solver."""

    method: str
    converged: bool
    solution: Any
    error: float | None
    residual: float | None
    iterations: int
    function_calls: int
    runtime_seconds: float


def _calculate_error(
    solution: Any,
    exact_solution: Any,
) -> float | None:
    """Calculate absolute scalar error or Euclidean array error."""
    try:
        computed = np.asarray(solution, dtype=float)
        exact = np.asarray(exact_solution, dtype=float)

        if computed.shape != exact.shape:
            return None

        if not np.all(np.isfinite(computed)):
            return None

        if not np.all(np.isfinite(exact)):
            return None

        error = float(np.linalg.norm(computed - exact))

        if not np.isfinite(error):
            return None

        return error

    except (TypeError, ValueError, OverflowError):
        return None


def measure_solver(
    method: str,
    solver: Callable[[], SolverResult],
    *,
    exact_solution: Any | None = None,
) -> BenchmarkResult:
    """
    Execute a solver and collect standardized benchmark metrics.

    Parameters
    ----------
    method:
        Human-readable solver name.

    solver:
        Zero-argument callable returning a SolverResult.

    exact_solution:
        Optional exact solution used to calculate error.
        Scalars use absolute error; arrays use the Euclidean norm.
        If omitted or incompatible, error is reported as None.
    """
    start = perf_counter()
    result = solver()
    runtime = perf_counter() - start

    error = None

    if exact_solution is not None:
        error = _calculate_error(result.solution, exact_solution)

    return BenchmarkResult(
        method=method,
        converged=result.converged,
        solution=result.solution,
        error=error,
        residual=result.residual,
        iterations=result.iterations,
        function_calls=result.function_calls,
        runtime_seconds=runtime,
    )
