import math

import pytest

from numerical_toolkit.core.exceptions import (
    DerivativeZeroError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.newton import newton


def test_newton_finds_sqrt_2():
    result = newton(
        lambda x: x**2 - 2,
        lambda x: 2 * x,
        1.5,
    )

    assert result.converged
    assert abs(result.solution - math.sqrt(2)) < 1e-10
    assert result.residual is not None
    assert result.residual < 1e-10


def test_newton_is_fast_for_quadratic_root():
    result = newton(
        lambda x: x**2 - 2,
        lambda x: 2 * x,
        1.5,
    )

    assert result.converged
    assert result.iterations <= 6


def test_newton_accepts_exact_initial_root():
    result = newton(
        lambda x: x**2 - 1,
        lambda x: 2 * x,
        1.0,
    )

    assert result.converged
    assert result.solution == 1.0
    assert result.iterations == 0


def test_newton_rejects_zero_derivative():
    with pytest.raises(DerivativeZeroError):
        newton(
            lambda x: x**2 + 1,
            lambda x: 2 * x,
            0.0,
        )


def test_newton_rejects_nonfinite_function_value():
    with pytest.raises(NonFiniteValueError):
        newton(
            lambda x: math.inf,
            lambda x: 1.0,
            1.0,
        )


def test_newton_rejects_nonfinite_derivative():
    with pytest.raises(NonFiniteValueError):
        newton(
            lambda x: x**2 - 2,
            lambda x: math.inf,
            1.0,
        )


def test_newton_can_disable_history():
    result = newton(
        lambda x: x**2 - 2,
        lambda x: 2 * x,
        1.5,
        options=SolverOptions(store_history=False),
    )

    assert result.converged
    assert result.history is None


def test_newton_stops_after_max_iterations():
    result = newton(
        lambda x: x**2 - 2,
        lambda x: 2 * x,
        1.5,
        options=SolverOptions(
            maxiter=1,
            xtol=1e-20,
            rtol=1e-20,
            ftol=1e-20,
        ),
    )

    assert result.converged is False
    assert result.iterations == 1


def test_newton_history_length_matches_iterations():
    result = newton(
        lambda x: x**2 - 2,
        lambda x: 2 * x,
        1.5,
    )

    assert result.history is not None
    assert len(result.history) == result.iterations