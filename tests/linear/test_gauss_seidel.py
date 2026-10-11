import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.linear.gauss_seidel import gauss_seidel


def test_gauss_seidel_solves_diagonally_dominant_system():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    result = gauss_seidel(A, b)

    expected = np.linalg.solve(A, b)

    assert isinstance(result, SolverResult)
    assert result.converged
    assert np.allclose(result.solution, expected, atol=1e-8)


def test_gauss_seidel_residual_is_small():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])

    b = np.array([12.0, 13.0, 14.0])

    result = gauss_seidel(A, b)

    assert result.converged
    assert result.residual is not None
    assert result.residual < 1e-8


def test_gauss_seidel_custom_initial_guess():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    # Deliberately choose a non-solution initial guess.
    x0 = np.array([0.0, 0.0])

    result = gauss_seidel(A, b, x0=x0)

    expected = np.linalg.solve(A, b)

    assert result.converged
    assert result.iterations > 0
    assert np.allclose(result.solution, expected, atol=1e-8)


def test_gauss_seidel_zero_initial_guess():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    result = gauss_seidel(A, b)

    assert result.converged
    assert result.iterations > 0


def test_gauss_seidel_initial_solution_converges_immediately():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    x_exact = np.linalg.solve(A, b)

    result = gauss_seidel(A, b, x0=x_exact)

    assert result.converged
    assert result.iterations == 0
    assert result.residual is not None
    assert result.residual <= 1e-10


def test_gauss_seidel_rejects_zero_diagonal():
    A = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b)


def test_gauss_seidel_rejects_non_square_matrix():
    A = np.array([
        [4.0, 1.0, 0.0],
        [2.0, 3.0, 1.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b)


def test_gauss_seidel_rejects_wrong_b_shape():
    A = np.eye(2)
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b)


def test_gauss_seidel_rejects_wrong_x0_shape():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    x0 = np.array([0.0, 0.0, 0.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b, x0=x0)


def test_gauss_seidel_rejects_invalid_options():
    A = np.eye(2)
    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b, options="invalid")


def test_gauss_seidel_respects_maxiter():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    options = SolverOptions(maxiter=1)

    result = gauss_seidel(A, b, options=options)

    assert result.converged is False
    assert result.iterations == 1


def test_gauss_seidel_reports_nonconvergence():
    # A system for which the Gauss-Seidel iteration does not converge.
    A = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    b = np.array([1.0, 1.0])

    options = SolverOptions(maxiter=10)

    result = gauss_seidel(A, b, options=options)

    assert result.converged is False
    assert result.iterations == 10
    assert np.all(np.isfinite(result.solution))


def test_gauss_seidel_detects_nonfinite_matrix():
    A = np.array([
        [np.nan, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(NonFiniteValueError):
        gauss_seidel(A, b)


def test_gauss_seidel_detects_nonfinite_rhs():
    A = np.eye(2)
    b = np.array([1.0, np.inf])

    with pytest.raises(NonFiniteValueError):
        gauss_seidel(A, b)


def test_gauss_seidel_detects_nonfinite_initial_guess():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    x0 = np.array([0.0, np.nan])

    with pytest.raises(NonFiniteValueError):
        gauss_seidel(A, b, x0=x0)


def test_gauss_seidel_stores_history():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    result = gauss_seidel(A, b)

    assert result.history is not None
    assert len(result.history) == result.iterations + 1

    first = result.history[0]

    assert first["iteration"] == 0
    assert "solution" in first
    assert "residual" in first
    assert "step_error" in first


def test_gauss_seidel_can_disable_history():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    options = SolverOptions(store_history=False)

    result = gauss_seidel(A, b, options=options)

    assert result.converged
    assert result.history is None


def test_gauss_seidel_does_not_mutate_inputs():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])
    x0 = np.array([0.0, 0.0])

    A_original = A.copy()
    b_original = b.copy()
    x0_original = x0.copy()

    gauss_seidel(A, b, x0=x0)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)
    assert np.array_equal(x0, x0_original)


def test_gauss_seidel_handles_small_scaled_system():
    A = np.array([
        [4e-20, 1e-20],
        [2e-20, 3e-20],
    ])
    b = np.array([1e-20, 2e-20])

    options = SolverOptions(
        xtol=1e-12,
        ftol=1e-32,
        maxiter=100,
    )

    result = gauss_seidel(A, b, options=options)

    assert result.converged
    assert np.allclose(result.solution, [0.1, 0.6], atol=1e-10)


def test_gauss_seidel_rejects_boolean_matrix_input():
    A = np.array([
        [True, False],
        [False, True],
    ])
    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b)


def test_gauss_seidel_rejects_boolean_initial_guess():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    x0 = np.array([True, False])

    with pytest.raises(InvalidInputError):
        gauss_seidel(A, b, x0=x0)


def test_gauss_seidel_relative_tolerance_changes_stopping_criterion():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])
    b = np.array([1.0, 2.0])

    strict = gauss_seidel(
        A,
        b,
        options=SolverOptions(
            xtol=1e-14,
            rtol=1e-14,
            ftol=1e-2,
            maxiter=100,
        ),
    )

    relaxed = gauss_seidel(
        A,
        b,
        options=SolverOptions(
            xtol=1e-14,
            rtol=1e-2,
            ftol=1e-2,
            maxiter=100,
        ),
    )

    assert strict.converged
    assert relaxed.converged
    assert strict.iterations > relaxed.iterations
    assert strict.residual <= 1e-2
    assert relaxed.residual <= 1e-2
