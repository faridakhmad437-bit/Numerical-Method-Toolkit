import math

import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.linear.gaussian import gaussian_elimination


def test_gaussian_solves_2x2_system():
    A = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    b = np.array([5.0, 6.0])

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        x,
        np.array([1.8, 1.4]),
        atol=1e-12,
    )


def test_gaussian_solves_3x3_system():
    A = np.array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 4.0],
        [-1.0, 0.5, -1.0],
    ])

    b = np.array([
        1.0,
        -2.0,
        0.0,
    ])

    x = gaussian_elimination(A, b)

    expected = np.array([
        1.0,
        -2.0,
        -2.0,
    ])

    np.testing.assert_allclose(
        x,
        expected,
        atol=1e-12,
    )


def test_gaussian_handles_zero_leading_pivot_with_partial_pivoting():
    # The first diagonal element is zero.
    # A solver without row pivoting would fail immediately.

    A = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        x,
        np.array([1.0, 1.0]),
        atol=1e-12,
    )


def test_gaussian_handles_row_swap_with_larger_pivot():
    A = np.array([
        [1e-20, 1.0],
        [1.0, 1.0],
    ])

    b = np.array([
        1.0,
        2.0,
    ])

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        A @ x,
        b,
        atol=1e-10,
    )


def test_gaussian_solves_diagonal_system():
    A = np.diag([
        2.0,
        4.0,
        5.0,
    ])

    b = np.array([
        6.0,
        8.0,
        10.0,
    ])

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        x,
        np.array([3.0, 2.0, 2.0]),
        atol=1e-12,
    )


def test_gaussian_solves_1x1_system():
    A = np.array([[5.0]])
    b = np.array([15.0])

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        x,
        np.array([3.0]),
        atol=1e-12,
    )


def test_gaussian_zero_rhs_returns_zero_solution():
    A = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    b = np.zeros(2)

    x = gaussian_elimination(A, b)

    np.testing.assert_allclose(
        x,
        np.zeros(2),
        atol=1e-12,
    )


def test_gaussian_residual_is_small():
    A = np.array([
        [4.0, 2.0, 1.0],
        [1.0, 5.0, 2.0],
        [2.0, 1.0, 4.0],
    ])

    b = np.array([
        7.0,
        8.0,
        9.0,
    ])

    x = gaussian_elimination(A, b)

    residual = np.linalg.norm(A @ x - b)

    assert residual < 1e-12


@pytest.mark.parametrize(
    "A,b",
    [
        (
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
            np.array([1.0, 2.0]),
        ),
        (
            np.array([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ),
    ],
)
def test_gaussian_rejects_invalid_shapes(A, b):
    with pytest.raises(InvalidInputError):
        gaussian_elimination(A, b)


def test_gaussian_rejects_singular_matrix():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    b = np.array([
        3.0,
        6.0,
    ])

    with pytest.raises(InvalidInputError):
        gaussian_elimination(A, b)


def test_gaussian_rejects_zero_matrix():
    A = np.zeros((3, 3))
    b = np.zeros(3)

    with pytest.raises(InvalidInputError):
        gaussian_elimination(A, b)


def test_gaussian_rejects_nearly_singular_matrix():
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-16],
    ])

    b = np.array([
        2.0,
        2.0,
    ])

    with pytest.raises(InvalidInputError):
        gaussian_elimination(A, b)


def test_gaussian_rejects_empty_system():
    A = np.empty((0, 0))
    b = np.empty(0)

    with pytest.raises(InvalidInputError):
        gaussian_elimination(A, b)


def test_gaussian_rejects_non_finite_input():
    A = np.array([
        [1.0, math.nan],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(NonFiniteValueError):
        gaussian_elimination(A, b)


def test_gaussian_does_not_modify_inputs():
    A = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    b = np.array([
        5.0,
        6.0,
    ])

    A_original = A.copy()
    b_original = b.copy()

    gaussian_elimination(A, b)

    np.testing.assert_array_equal(A, A_original)
    np.testing.assert_array_equal(b, b_original)