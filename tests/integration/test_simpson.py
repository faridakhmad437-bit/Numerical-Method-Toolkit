import math

import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.integration.simpson import simpson


def test_simpson_integrates_linear_function_exactly():
    result = simpson(lambda x: 2.0 * x + 1.0, 0.0, 3.0, 10)

    assert result == pytest.approx(12.0, abs=1e-12)


def test_simpson_integrates_quadratic_function_exactly():
    result = simpson(lambda x: x**2, 0.0, 1.0, 10)

    assert result == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_simpson_integrates_cubic_function_exactly():
    result = simpson(lambda x: x**3, 0.0, 1.0, 10)

    assert result == pytest.approx(1.0 / 4.0, abs=1e-12)


def test_simpson_integrates_constant_function():
    result = simpson(lambda x: 5.0, 2.0, 7.0, 10)

    assert result == pytest.approx(25.0, abs=1e-12)


def test_simpson_supports_reversed_interval():
    forward = simpson(lambda x: x**2, 0.0, 1.0, 100)
    backward = simpson(lambda x: x**2, 1.0, 0.0, 100)

    assert backward == pytest.approx(-forward, abs=1e-12)


def test_simpson_zero_width_interval():
    result = simpson(lambda x: math.sin(x), 2.0, 2.0, 10)

    assert result == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [0, -2, -10])
def test_simpson_rejects_non_positive_n(n):
    with pytest.raises(InvalidInputError):
        simpson(lambda x: x, 0.0, 1.0, n)


@pytest.mark.parametrize("n", [1, 3, 5, 7, 9])
def test_simpson_rejects_odd_n(n):
    with pytest.raises(InvalidInputError):
        simpson(lambda x: x, 0.0, 1.0, n)


@pytest.mark.parametrize("n", [1.5, 2.0, "10", None])
def test_simpson_rejects_non_integer_n(n):
    with pytest.raises(InvalidInputError):
        simpson(lambda x: x, 0.0, 1.0, n)


def test_simpson_rejects_non_callable_function():
    with pytest.raises(InvalidInputError):
        simpson(123, 0.0, 1.0, 10)


@pytest.mark.parametrize(
    "a,b",
    [
        (math.inf, 1.0),
        (-math.inf, 1.0),
        (0.0, math.inf),
        (0.0, math.nan),
    ],
)
def test_simpson_rejects_non_finite_bounds(a, b):
    with pytest.raises(NonFiniteValueError):
        simpson(lambda x: x, a, b, 10)


def test_simpson_rejects_non_finite_function_values():
    def bad_function(x):
        return math.inf

    with pytest.raises(NonFiniteValueError):
        simpson(bad_function, 0.0, 1.0, 10)


def test_simpson_rejects_nan_function_values():
    def bad_function(x):
        return math.nan

    with pytest.raises(NonFiniteValueError):
        simpson(bad_function, 0.0, 1.0, 10)


def test_simpson_convergence_is_fourth_order():
    f = lambda x: x**4

    exact = 1.0 / 5.0

    error_n = abs(simpson(f, 0.0, 1.0, 10) - exact)
    error_2n = abs(simpson(f, 0.0, 1.0, 20) - exact)

    ratio = error_n / error_2n

    # Composite Simpson's 1/3 rule has O(h^4) error,
    # so doubling n should reduce the error by approximately 16.
    assert ratio == pytest.approx(16.0, rel=0.05)

@pytest.mark.parametrize("n", [True, False])
def test_simpson_rejects_boolean_n(n):
    with pytest.raises(InvalidInputError):
        simpson(lambda x: x, 0.0, 1.0, n)