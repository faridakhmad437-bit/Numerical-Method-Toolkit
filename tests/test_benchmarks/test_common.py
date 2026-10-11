
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.common import BenchmarkResult, measure_solver
from numerical_toolkit.core.results import SolverResult


def make_solver(solution):
    """Create a minimal solver returning a controlled result."""

    def solver():
        return SolverResult(
            solution=solution,
            converged=True,
            iterations=3,
            function_calls=4,
            residual=1e-10,
            step_error=1e-8,
            message="Converged.",
            history=None,
        )

    return solver


def test_measure_solver_returns_benchmark_result():
    result = measure_solver(
        "Test Solver",
        make_solver(2.0),
        exact_solution=2.0,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.method == "Test Solver"
    assert result.converged is True
    assert result.solution == 2.0
    assert result.error == 0.0
    assert result.residual == 1e-10
    assert result.iterations == 3
    assert result.function_calls == 4
    assert result.runtime_seconds >= 0.0


def test_measure_solver_without_exact_solution():
    result = measure_solver(
        "Test Solver",
        make_solver(3.0),
    )

    assert result.error is None


def test_measure_solver_calculates_vector_error():
    solution = np.array([1.0, 2.0])
    exact_solution = np.array([1.0, 2.5])

    result = measure_solver(
        "Vector Solver",
        make_solver(solution),
        exact_solution=exact_solution,
    )

    assert result.error == pytest.approx(0.5)


def test_measure_solver_calculates_multidimensional_error():
    solution = np.array([[1.0, 2.0], [3.0, 4.0]])
    exact_solution = np.array([[2.0, 2.0], [3.0, 6.0]])

    result = measure_solver(
        "Matrix Solver",
        make_solver(solution),
        exact_solution=exact_solution,
    )

    assert result.error == pytest.approx(np.sqrt(5.0))


def test_measure_solver_returns_none_for_shape_mismatch():
    result = measure_solver(
        "Vector Solver",
        make_solver(np.array([1.0, 2.0])),
        exact_solution=np.array([1.0, 2.0, 3.0]),
    )

    assert result.error is None


@pytest.mark.parametrize(
    "solution, exact_solution",
    [
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),
        (np.array([1.0, np.inf]), np.array([1.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([1.0, np.nan])),
    ],
)
def test_measure_solver_returns_none_for_non_finite_values(
    solution,
    exact_solution,
):
    result = measure_solver(
        "Vector Solver",
        make_solver(solution),
        exact_solution=exact_solution,
    )

    assert result.error is None


def test_measure_solver_returns_none_for_non_numeric_values():
    result = measure_solver(
        "Invalid Solver",
        make_solver(["invalid"]),
        exact_solution=[1.0],
    )

    assert result.error is None


def test_measure_solver_preserves_solver_solution():
    solution = np.array([1.0, 2.0])

    result = measure_solver(
        "Vector Solver",
        make_solver(solution),
    )

    assert result.solution is solution
    assert np.array_equal(result.solution, solution)
