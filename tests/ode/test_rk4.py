import math

import numpy as np
import pytest

from numerical_toolkit.core.exceptions import InvalidInputError
from numerical_toolkit.ode.rk4 import rk4


def test_rk4_solves_exponential_growth_accurately():
    # y' = y
    # y(0) = 1
    # Exact solution: y(t) = e^t

    f = lambda t, y: y

    times, values = rk4(
        f,
        t0=0.0,
        y0=1.0,
        t_end=1.0,
        h=0.1,
    )

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)

    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(math.e, rel=1e-5)


def test_rk4_solves_linear_function_exactly():
    # y' = 2
    # y(0) = 1
    # Exact solution: y = 1 + 2t

    f = lambda t, y: 2.0

    times, values = rk4(
        f,
        t0=0.0,
        y0=1.0,
        t_end=2.0,
        h=0.1,
    )

    expected = 1.0 + 2.0 * times

    np.testing.assert_allclose(
        values,
        expected,
        atol=1e-12,
    )


def test_rk4_solves_cubic_polynomial_ode_accurately():
    # y' = 3t^2
    # y(0) = 1
    # Exact solution: y = t^3 + 1

    f = lambda t, y: 3.0 * t**2

    times, values = rk4(
        f,
        t0=0.0,
        y0=1.0,
        t_end=1.0,
        h=0.1,
    )

    expected = times**3 + 1.0

    np.testing.assert_allclose(
        values,
        expected,
        atol=1e-12,
    )


def test_rk4_supports_system_ode():
    # x' = v
    # v' = -x
    #
    # Initial state:
    # x(0) = 1
    # v(0) = 0

    def f(t, y):
        x, v = y
        return np.array([v, -x])

    times, values = rk4(
        f,
        t0=0.0,
        y0=np.array([1.0, 0.0]),
        t_end=1.0,
        h=0.01,
    )

    assert times.shape == (101,)
    assert values.shape == (101, 2)

    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.0)

    # Exact solution:
    # x(t) = cos(t)
    # v(t) = -sin(t)

    assert values[-1, 0] == pytest.approx(math.cos(1.0), abs=1e-8)
    assert values[-1, 1] == pytest.approx(-math.sin(1.0), abs=1e-8)


def test_rk4_reaches_t_end_exactly():
    f = lambda t, y: 1.0

    times, values = rk4(
        f,
        t0=0.0,
        y0=0.0,
        t_end=1.0,
        h=0.3,
    )

    assert times[-1] == pytest.approx(1.0)

    assert len(times) == 5
    assert len(values) == 5


def test_rk4_zero_length_interval():
    f = lambda t, y: y

    times, values = rk4(
        f,
        t0=1.0,
        y0=2.0,
        t_end=1.0,
        h=0.1,
    )

    assert len(times) == 1
    assert times[0] == pytest.approx(1.0)
    assert values[0] == pytest.approx(2.0)


@pytest.mark.parametrize("h", [0.0, -0.1, -1.0])
def test_rk4_rejects_non_positive_step(h):
    with pytest.raises(InvalidInputError):
        rk4(
            lambda t, y: y,
            0.0,
            1.0,
            1.0,
            h,
        )


def test_rk4_rejects_reversed_interval():
    with pytest.raises(InvalidInputError):
        rk4(
            lambda t, y: y,
            1.0,
            1.0,
            0.0,
            0.1,
        )


def test_rk4_rejects_non_callable_function():
    with pytest.raises((InvalidInputError, TypeError)):
        rk4(
            123,
            0.0,
            1.0,
            1.0,
            0.1,
        )


def test_rk4_preserves_initial_state():
    y0 = np.array([1.0, 2.0])

    times, values = rk4(
        lambda t, y: np.array([0.0, 0.0]),
        0.0,
        y0,
        1.0,
        0.1,
    )

    assert np.array_equal(
        y0,
        np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(
        values[0],
        y0,
    )