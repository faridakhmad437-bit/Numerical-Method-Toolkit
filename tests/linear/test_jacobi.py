import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.linear.jacobi import jacobi


def test_jacobi_solves_diagonally_dominant_system():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=100,
        ),
    )

    expected = np.array([
        0.1,
        0.6,
    ])

    assert isinstance(result, SolverResult)
    assert result.converged is True
    assert result.iterations < 100

    np.testing.assert_allclose(
        result.solution,
        expected,
        atol=1e-10,
    )


def test_jacobi_residual_is_small():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])

    b = np.array([
        12.0,
        13.0,
        14.0,
    ])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=100,
        ),
    )

    assert result.converged is True
    assert result.residual is not None
    assert result.residual < 1e-10


def test_jacobi_uses_custom_initial_guess():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    x0 = np.array([
        0.1,
        0.6,
    ])

    result = jacobi(
        A,
        b,
        x0=x0,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=100,
        ),
    )

    assert result.converged is True
    assert result.iterations == 0

    np.testing.assert_allclose(
        result.solution,
        x0,
        atol=1e-12,
    )


def test_jacobi_default_initial_guess_is_zero():
    A = np.eye(2)
    b = np.array([3.0, 4.0])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=10,
        ),
    )

    assert result.converged is True
    assert result.iterations == 2

    np.testing.assert_allclose(
        result.solution,
        b,
        atol=1e-12,
    )


def test_jacobi_rejects_zero_diagonal():
    A = np.array([
        [0.0, 1.0],
        [1.0, 2.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(InvalidInputError):
        jacobi(A, b)


@pytest.mark.parametrize(
    "A,b",
    [
        (
            np.array([
                [4.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
            ]),
            np.array([1.0, 2.0]),
        ),
        (
            np.array([
                [4.0, 1.0],
                [2.0, 3.0],
            ]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        ),
    ],
)
def test_jacobi_rejects_invalid_shapes(A, b):
    with pytest.raises(InvalidInputError):
        jacobi(A, b)


def test_jacobi_rejects_invalid_x0_shape():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    x0 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(InvalidInputError):
        jacobi(A, b, x0=x0)


def test_jacobi_rejects_invalid_options_type():
    A = np.eye(2)
    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        jacobi(A, b, options=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        SolverOptions(ftol=1e-12, xtol=1e-12, maxiter=1),
        SolverOptions(ftol=1e-12, xtol=1e-12, maxiter=2),
    ],
)
def test_jacobi_respects_maxiter(options):
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    result = jacobi(
        A,
        b,
        options=options,
    )

    assert result.iterations <= options.maxiter


def test_jacobi_rejects_non_finite_input():
    A = np.array([
        [4.0, np.nan],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(NonFiniteValueError):
        jacobi(A, b)


def test_jacobi_rejects_non_finite_x0():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    x0 = np.array([0.0, np.nan])

    with pytest.raises(NonFiniteValueError):
        jacobi(A, b, x0=x0)


def test_jacobi_reports_non_convergence():
    # Jacobi iteration matrix for this system has spectral radius > 1.
    A = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    b = np.array([
        1.0,
        1.0,
    ])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=10,
        ),
    )

    assert result.converged is False
    assert result.iterations == 10
    assert result.residual is not None
    assert result.residual > 1e-12


def test_jacobi_history_is_stored():
    A = np.eye(2)
    b = np.array([3.0, 4.0])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=10,
            store_history=True,
        ),
    )

    assert result.history is not None
    assert len(result.history) == result.iterations + 1

    assert result.history[0]["iteration"] == 0
    assert result.history[-1]["iteration"] == result.iterations


def test_jacobi_can_disable_history():
    A = np.eye(2)
    b = np.array([3.0, 4.0])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=10,
            store_history=False,
        ),
    )

    assert result.history is None


def test_jacobi_does_not_modify_inputs():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    x0 = np.array([
        0.0,
        0.0,
    ])

    A_original = A.copy()
    b_original = b.copy()
    x0_original = x0.copy()

    jacobi(
        A,
        b,
        x0=x0,
        options=SolverOptions(
            ftol=1e-10,
            xtol=1e-10,
            maxiter=100,
        ),
    )

    np.testing.assert_array_equal(A, A_original)
    np.testing.assert_array_equal(b, b_original)
    np.testing.assert_array_equal(x0, x0_original)


def test_jacobi_handles_small_scaled_system():
    A = np.array([
        [4e-20, 1e-20],
        [2e-20, 3e-20],
    ])
    b = np.array([1e-20, 2e-20])

    result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-32,
            xtol=1e-12,
            maxiter=100,
        ),
    )

    assert result.converged is True

    np.testing.assert_allclose(
        result.solution,
        np.array([0.1, 0.6]),
        atol=1e-10,
    )


def test_jacobi_rejects_boolean_matrix_input():
    A = np.array([
        [True, False],
        [False, True],
    ])
    b = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        jacobi(A, b)
