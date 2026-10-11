from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

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
        Optional exact solution used to calculate absolute error.
        If omitted, error is reported as None.
    """
    start = perf_counter()
    result = solver()
    runtime = perf_counter() - start

    error: float | None = None

    if exact_solution is not None:
        try:
            error = abs(float(result.solution) - float(exact_solution))
        except (TypeError, ValueError):
            error = None

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