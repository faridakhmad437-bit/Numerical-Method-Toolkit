import math

import numpy as np
import pytest

from numerical_toolkit.core.exceptions import InvalidInputError
from numerical_toolkit.ode.euler import euler


def test_euler_solves_exponential_growth():
    # y' = y, y(0) = 1
    # Exact solution: y(t) = e^t

    f = lambda t, y: y

    times, values = euler(
        f,
        t0=0.0,
        y0=1.0,
        t_end=1.0,
        h=0.01,
    )

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)

    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(math.e, rel=0.01)


def test_euler_reaches_t_end_exactly():
    f = lambda t, y: 1.0

    times, values = euler(
        f,
        t0=0.0,
        y0=0.0,
        t_end=1.0,
        h=0.3,
    )

    assert times[-1] == pytest.approx(1.0)

    assert len(times) == 5
    assert len(values) == 5


def test_euler_solves_linear_function_exactly():
    # y' = 2
    # y(0) = 1
    # Exact solution: y = 1 + 2t

    f = lambda t, y: 2.0

    times, values = euler(
        f,
        t0=0.0,
        y0=1.0,
        t_end=2.0,
        h=0.1,
    )

    expected = 1.0 + 2.0 * times

    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_euler_supports_system_ode():
    # x' = v
    # v' = -x
    #
    # Initial state:
    # x(0) = 1
    # v(0) = 0

    def f(t, y):
        x, v = y
        return np.array([v, -x])

    times, values = euler(
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


def test_euler_zero_length_interval():
    f = lambda t, y: y

    times, values = euler(
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
def test_euler_rejects_non_positive_step(h):
    with pytest.raises(InvalidInputError):
        euler(
            lambda t, y: y,
            0.0,
            1.0,
            1.0,
            h,
        )


def test_euler_rejects_reversed_interval():
    with pytest.raises(InvalidInputError):
        euler(
            lambda t, y: y,
            1.0,
            1.0,
            0.0,
            0.1,
        )


def test_euler_rejects_non_callable_function():
    with pytest.raises((InvalidInputError, TypeError)):
        euler(
            123,
            0.0,
            1.0,
            1.0,
            0.1,
        )


def test_euler_preserves_initial_state():
    y0 = np.array([1.0, 2.0])

    times, values = euler(
        lambda t, y: np.array([0.0, 0.0]),
        0.0,
        y0,
        1.0,
        0.1,
    )

    assert np.array_equal(y0, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(values[0], y0)


def test_euler_handles_very_short_interval():
    times, values = euler(
        lambda t, y: 1.0,
        0.0,
        0.0,
        1e-16,
        1e-17,
    )

    assert times[-1] == pytest.approx(1e-16, abs=1e-30)
    assert len(times) > 1
    assert values[-1] == pytest.approx(1e-16, abs=1e-30)


@pytest.mark.parametrize(
    "t0,t_end,h",
    [
        (math.nan, 1.0, 0.1),
        (0.0, math.nan, 0.1),
        (0.0, 1.0, math.nan),
        (0.0, 1.0, math.inf),
    ],
)
def test_euler_rejects_non_finite_time_parameters(t0, t_end, h):
    from numerical_toolkit.core.exceptions import NonFiniteValueError

    with pytest.raises(NonFiniteValueError):
        euler(lambda t, y: y, t0, 1.0, t_end, h)


def test_euler_rejects_boolean_step():
    with pytest.raises(InvalidInputError):
        euler(lambda t, y: y, 0.0, 1.0, 1.0, True)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_euler_rejects_non_finite_initial_state(bad_value):
    from numerical_toolkit.core.exceptions import NonFiniteValueError

    with pytest.raises(NonFiniteValueError):
        euler(lambda t, y: y, 0.0, bad_value, 1.0, 0.1)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_euler_rejects_non_finite_derivative(bad_value):
    from numerical_toolkit.core.exceptions import NonFiniteValueError

    with pytest.raises(NonFiniteValueError):
        euler(lambda t, y: bad_value, 0.0, 1.0, 1.0, 0.1)
