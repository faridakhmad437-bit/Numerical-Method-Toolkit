import math

import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.integration.trapezoidal import trapezoidal


def test_trapezoidal_integrates_linear_function_exactly():
    result = trapezoidal(lambda x: 2.0 * x + 1.0, 0.0, 3.0, 10)

    assert result == pytest.approx(12.0, abs=1e-12)


def test_trapezoidal_integrates_quadratic_function():
    result = trapezoidal(lambda x: x**2, 0.0, 1.0, 100)

    assert result == pytest.approx(1.0 / 3.0, abs=2e-5)


def test_trapezoidal_integrates_constant_function():
    result = trapezoidal(lambda x: 5.0, 2.0, 7.0, 10)

    assert result == pytest.approx(25.0, abs=1e-12)


def test_trapezoidal_supports_reversed_interval():
    forward = trapezoidal(lambda x: x**2, 0.0, 1.0, 100)
    backward = trapezoidal(lambda x: x**2, 1.0, 0.0, 100)

    assert backward == pytest.approx(-forward, abs=1e-12)


def test_trapezoidal_zero_width_interval():
    result = trapezoidal(lambda x: math.sin(x), 2.0, 2.0, 10)

    assert result == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [0, -1, -10])
def test_trapezoidal_rejects_non_positive_n(n):
    with pytest.raises(InvalidInputError):
        trapezoidal(lambda x: x, 0.0, 1.0, n)


@pytest.mark.parametrize("n", [1.5, 2.0, "10", None])
def test_trapezoidal_rejects_non_integer_n(n):
    with pytest.raises(InvalidInputError):
        trapezoidal(lambda x: x, 0.0, 1.0, n)


def test_trapezoidal_rejects_non_callable_function():
    with pytest.raises(InvalidInputError):
        trapezoidal(123, 0.0, 1.0, 10)


@pytest.mark.parametrize(
    "a,b",
    [
        (math.inf, 1.0),
        (-math.inf, 1.0),
        (0.0, math.inf),
        (0.0, math.nan),
    ],
)
def test_trapezoidal_rejects_non_finite_bounds(a, b):
    with pytest.raises(NonFiniteValueError):
        trapezoidal(lambda x: x, a, b, 10)


def test_trapezoidal_rejects_non_finite_function_values():
    def bad_function(x):
        return math.inf

    with pytest.raises(NonFiniteValueError):
        trapezoidal(bad_function, 0.0, 1.0, 10)


def test_trapezoidal_rejects_nan_function_values():
    def bad_function(x):
        return math.nan

    with pytest.raises(NonFiniteValueError):
        trapezoidal(bad_function, 0.0, 1.0, 10)


def test_trapezoidal_convergence_is_second_order():
    f = lambda x: x**2

    exact = 1.0 / 3.0

    error_n = abs(trapezoidal(f, 0.0, 1.0, 10) - exact)
    error_2n = abs(trapezoidal(f, 0.0, 1.0, 20) - exact)

    ratio = error_n / error_2n

    # Composite trapezoidal rule has O(h^2) error,
    # so doubling n should reduce the error by approximately 4.
    assert ratio == pytest.approx(4.0, rel=0.05)

def test_trapezoidal_rejects_boolean_n():
    with pytest.raises(InvalidInputError):
        trapezoidal(lambda x: x, 0.0, 1.0, True)

def test_trapezoidal_accepts_numpy_integer_n():
    import numpy as np

    result = trapezoidal(lambda x: x, 0.0, 1.0, np.int64(4))

    assert result == pytest.approx(0.5)
