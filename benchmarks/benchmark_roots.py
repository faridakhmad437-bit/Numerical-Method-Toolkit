from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


@dataclass(frozen=True)
class RootBenchmarkResult:
    """Benchmark metrics for one root-finding method."""

    method: str
    converged: bool
    solution: float
    error: float
    residual: float
    iterations: int
    function_calls: int
    runtime_seconds: float


def run_root_benchmark(
    *,
    options: SolverOptions | None = None,
) -> list[RootBenchmarkResult]:
    """
    Benchmark the root-finding methods on f(x) = x^2 - 2.

    The exact solution is sqrt(2).

    Returns
    -------
    list[RootBenchmarkResult]
        Benchmark metrics for Bisection, Newton, Secant,
        and Fixed Point.
    """
    if options is None:
        options = SolverOptions()

    def f(x: float) -> float:
        return x**2 - 2.0

    def df(x: float) -> float:
        return 2.0 * x

    def g(x: float) -> float:
        return 0.5 * (x + 2.0 / x)

    exact_solution = np.sqrt(2.0)

    methods = [
        (
            "Bisection",
            lambda: bisection(
                f,
                1.0,
                2.0,
                options=options,
            ),
        ),
        (
            "Newton",
            lambda: newton(
                f,
                df,
                1.5,
                options=options,
            ),
        ),
        (
            "Secant",
            lambda: secant(
                f,
                1.0,
                2.0,
                options=options,
            ),
        ),
        (
            "Fixed Point",
            lambda: fixed_point(
                g,
                1.5,
                options=options,
            ),
        ),
    ]

    results: list[RootBenchmarkResult] = []

    for method_name, solver in methods:
        start = perf_counter()
        result = solver()
        runtime = perf_counter() - start

        solution = float(result.solution)
        error = abs(solution - exact_solution)

        results.append(
            RootBenchmarkResult(
                method=method_name,
                converged=result.converged,
                solution=solution,
                error=error,
                residual=float(result.residual or 0.0),
                iterations=result.iterations,
                function_calls=result.function_calls,
                runtime_seconds=runtime,
            )
        )

    return results


def print_root_benchmark(
    results: list[RootBenchmarkResult],
) -> None:
    """Print benchmark results as a compact table."""

    header = (
        f"{'Method':<15}"
        f"{'Converged':<12}"
        f"{'Error':<14}"
        f"{'Residual':<14}"
        f"{'Iterations':<12}"
        f"{'Calls':<8}"
        f"{'Runtime (s)':<14}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.method:<15}"
            f"{str(result.converged):<12}"
            f"{result.error:<14.3e}"
            f"{result.residual:<14.3e}"
            f"{result.iterations:<12}"
            f"{result.function_calls:<8}"
            f"{result.runtime_seconds:<14.6e}"
        )


if __name__ == "__main__":
    benchmark_results = run_root_benchmark()
    print_root_benchmark(benchmark_results)