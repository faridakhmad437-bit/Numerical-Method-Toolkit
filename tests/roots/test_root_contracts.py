import math

import numpy as np
import pytest

import numerical_toolkit
from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
    SecantDenominatorZeroError,
)
from numerical_toolkit.roots import bisection, fixed_point, newton, secant


def f(x):
    return x * x - 2.0


def df(x):
    return 2.0 * x


def g(x):
    return 0.5 * (x + 2.0 / x)


SOLVERS = {
    "bisection": lambda *, options=None, a=1.0, b=2.0: bisection(
        f, a, b, options=options
    ),
    "newton": lambda *, options=None, a=1.5, b=None: newton(
        f, df, a, options=options
    ),
    "secant": lambda *, options=None, a=1.0, b=2.0: secant(
        f, a, b, options=options
    ),
    "fixed_point": lambda *, options=None, a=1.5, b=None: fixed_point(
        g, a, options=options
    ),
}


@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("options", [{"maxiter": 5}, 5, "fast"])
def test_root_solvers_reject_invalid_options(name, options):
    with pytest.raises(InvalidInputError, match="options"):
        SOLVERS[name](options=options)


@pytest.mark.parametrize("name", SOLVERS)
def test_root_solvers_reject_boolean_initial_values(name):
    with pytest.raises(InvalidInputError):
        SOLVERS[name](a=True)


@pytest.mark.parametrize("name", SOLVERS)
def test_root_solvers_accept_numpy_scalars(name):
    result = SOLVERS[name](a=np.float32(1.0) if name in ("bisection", "secant") else np.float64(1.5))

    assert result.converged
    assert abs(result.solution - math.sqrt(2.0)) < 1e-8


def test_bisection_accepts_numpy_integer_bracket():
    result = bisection(f, np.int64(1), np.int64(2))

    assert result.converged
    assert abs(result.solution - math.sqrt(2.0)) < 1e-8


def test_secant_reports_non_finite_update_as_non_finite_error():
    # x1 - x0 overflows to infinity, so the secant update is non-finite.
    with pytest.raises(NonFiniteValueError):
        secant(lambda x: x / 1e300, -1e308, 1e308)


def test_secant_denominator_error_is_exported():
    assert numerical_toolkit.SecantDenominatorZeroError is SecantDenominatorZeroError
