from __future__ import annotations

import numpy as np

from benchmarks.common import BenchmarkResult, measure_solver
from numerical_toolkit.core.results import SolverResult


def test_measure_solver_returns_benchmark_result():
    def solver():
        return SolverResult(
            solution=2.0,
            converged=True,
            iterations=5,
            function_calls=6,
            residual=1e-12,
            step_error=1e-10,
            message="Converged.",
            history=None,
        )

    result = measure_solver(
        "Test Solver",
        solver,
        exact_solution=2.0,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.method == "Test Solver"
    assert result.converged is True
    assert result.solution == 2.0
    assert result.error == 0.0
    assert result.residual == 1e-12
    assert result.iterations == 5
    assert result.function_calls == 6
    assert result.runtime_seconds >= 0.0


def test_measure_solver_without_exact_solution():
    def solver():
        return SolverResult(
            solution=3.0,
            converged=True,
            iterations=2,
            function_calls=3,
            residual=1e-8,
            step_error=1e-6,
            message="Converged.",
            history=None,
        )

    result = measure_solver(
        "Test Solver",
        solver,
    )

    assert result.error is None


def test_measure_solver_supports_array_solution():
    def solver():
        return SolverResult(
            solution=np.array([1.0, 2.0]),
            converged=True,
            iterations=3,
            function_calls=4,
            residual=1e-10,
            step_error=1e-8,
            message="Converged.",
            history=None,
        )

    result = measure_solver(
        "Vector Solver",
        solver,
        exact_solution=np.array([1.0, 2.0]),
    )

    # Scalar error calculation is intentionally unavailable
    # for vector-valued solutions in this first abstraction.
    assert result.error is None


def test_measure_solver_preserves_solver_solution():
    solution = np.array([1.0, 2.0])

    def solver():
        return SolverResult(
            solution=solution,
            converged=True,
            iterations=1,
            function_calls=2,
            residual=0.0,
            step_error=0.0,
            message="Already converged.",
            history=None,
        )

    result = measure_solver(
        "Vector Solver",
        solver,
    )

    assert np.array_equal(result.solution, solution)