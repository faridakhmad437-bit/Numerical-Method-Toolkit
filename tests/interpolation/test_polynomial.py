import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.interpolation.polynomial import (
    lagrange,
    newton_interpolation,
)


# ============================================================================
# LAGRANGE — BASIC CORRECTNESS
# ============================================================================


def test_lagrange_recovers_linear_function():
    x = np.array([0.0, 1.0, 2.0])
    y = 2.0 * x + 3.0

    result = lagrange(x, y, 1.5)

    assert result == pytest.approx(6.0)


def test_lagrange_recovers_quadratic_function():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = x**2 + 2.0 * x + 1.0

    result = lagrange(x, y, 1.5)

    expected = 1.5**2 + 2.0 * 1.5 + 1.0

    assert result == pytest.approx(expected)


def test_lagrange_interpolates_original_nodes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([2.0, 5.0, 10.0, 17.0])

    for xi, yi in zip(x, y):
        result = lagrange(x, y, xi)

        assert result == pytest.approx(yi)


def test_lagrange_constant_function():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.full_like(x, 7.0)

    result = lagrange(x, y, 1.7)

    assert result == pytest.approx(7.0)


# ============================================================================
# NEWTON — BASIC CORRECTNESS
# ============================================================================


def test_newton_recovers_linear_function():
    x = np.array([0.0, 1.0, 2.0])
    y = 2.0 * x + 3.0

    result = newton_interpolation(x, y, 1.5)

    assert result == pytest.approx(6.0)


def test_newton_recovers_quadratic_function():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = x**2 + 2.0 * x + 1.0

    result = newton_interpolation(x, y, 1.5)

    expected = 1.5**2 + 2.0 * 1.5 + 1.0

    assert result == pytest.approx(expected)


def test_newton_interpolates_original_nodes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([2.0, 5.0, 10.0, 17.0])

    for xi, yi in zip(x, y):
        result = newton_interpolation(x, y, xi)

        assert result == pytest.approx(yi)


def test_newton_constant_function():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.full_like(x, 7.0)

    result = newton_interpolation(x, y, 1.7)

    assert result == pytest.approx(7.0)


# ============================================================================
# LAGRANGE VS NEWTON
# ============================================================================


def test_lagrange_and_newton_agree():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = x**3 - 2.0 * x**2 + x + 5.0

    query_points = [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ]

    for xq in query_points:
        lagrange_result = lagrange(
            x,
            y,
            xq,
        )

        newton_result = newton_interpolation(
            x,
            y,
            xq,
        )

        assert lagrange_result == pytest.approx(
            newton_result,
            rel=1e-12,
            abs=1e-12,
        )


# ============================================================================
# INPUT VALIDATION — SHAPE
# ============================================================================


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_non_1d_x(function):
    x = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )

    y = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(InvalidInputError):
        function(x, y, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_non_1d_y(function):
    x = np.array([0.0, 1.0])

    y = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(InvalidInputError):
        function(x, y, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_mismatched_lengths(function):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0])

    with pytest.raises(InvalidInputError):
        function(x, y, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_duplicate_x_values(function):
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(InvalidInputError):
        function(x, y, 1.5)


# ============================================================================
# EMPTY INPUT
# ============================================================================


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_empty_input(function):
    x = np.array([])
    y = np.array([])

    with pytest.raises(InvalidInputError):
        function(x, y, 1.0)


# ============================================================================
# NON-FINITE INPUT
# ============================================================================


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_x(function, bad_value):
    x = np.array([0.0, bad_value, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(NonFiniteValueError):
        function(x, y, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_y(function, bad_value):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, bad_value, 3.0])

    with pytest.raises(NonFiniteValueError):
        function(x, y, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_query(function, bad_value):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(NonFiniteValueError):
        function(x, y, bad_value)


# ============================================================================
# QUERY VALIDATION
# ============================================================================


@pytest.mark.parametrize(
    "function",
    [
        lagrange,
        newton_interpolation,
    ],
)
def test_rejects_non_scalar_query(function):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    xq = np.array([0.5, 1.5])

    with pytest.raises(InvalidInputError):
        function(x, y, xq)


# ============================================================================
# INPUT IMMUTABILITY
# ============================================================================


def test_lagrange_does_not_modify_inputs():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 7.0])

    x_original = x.copy()
    y_original = y.copy()

    lagrange(x, y, 1.5)

    assert np.array_equal(x, x_original)
    assert np.array_equal(y, y_original)


def test_newton_does_not_modify_inputs():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 7.0])

    x_original = x.copy()
    y_original = y.copy()

    newton_interpolation(x, y, 1.5)

    assert np.array_equal(x, x_original)
    assert np.array_equal(y, y_original)


# ============================================================================
# EXACT POLYNOMIAL RECOVERY
# ============================================================================


def test_lagrange_exact_polynomial_recovery():
    x = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    def polynomial(x):
        return (
            2.0 * x**4
            - 3.0 * x**3
            + x**2
            + 4.0 * x
            - 7.0
        )

    y = polynomial(x)

    query_points = np.linspace(
        -2.0,
        2.0,
        21,
    )

    for xq in query_points:
        result = lagrange(
            x,
            y,
            xq,
        )

        assert result == pytest.approx(
            polynomial(xq),
            rel=1e-10,
            abs=1e-10,
        )


def test_newton_exact_polynomial_recovery():
    x = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    def polynomial(x):
        return (
            2.0 * x**4
            - 3.0 * x**3
            + x**2
            + 4.0 * x
            - 7.0
        )

    y = polynomial(x)

    query_points = np.linspace(
        -2.0,
        2.0,
        21,
    )

    for xq in query_points:
        result = newton_interpolation(
            x,
            y,
            xq,
        )

        assert result == pytest.approx(
            polynomial(xq),
            rel=1e-10,
            abs=1e-10,
        )