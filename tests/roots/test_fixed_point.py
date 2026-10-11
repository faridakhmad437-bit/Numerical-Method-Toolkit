import math

import pytest

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.exceptions import NonFiniteValueError
from numerical_toolkit.roots.fixed_point import fixed_point


def test_fixed_point_finds_sqrt_2():
    result = fixed_point(
        lambda x: 0.5 * (x + 2 / x),
        1.0,
    )

    assert result.converged
    assert abs(result.solution - math.sqrt(2)) < 1e-8


def test_fixed_point_finds_simple_fixed_point():
    result = fixed_point(
        lambda x: 0.5 * x + 1.0,
        0.0,
    )

    assert result.converged
    assert abs(result.solution - 2.0) < 2e-8


def test_fixed_point_detects_nonfinite_value():
    with pytest.raises(NonFiniteValueError):
        fixed_point(
            lambda x: math.inf,
            1.0,
        )


def test_fixed_point_can_disable_history():
    result = fixed_point(
        lambda x: 0.5 * x + 1.0,
        0.0,
        options=SolverOptions(store_history=False),
    )

    assert result.converged
    assert result.history is None


def test_fixed_point_stops_after_max_iterations():
    result = fixed_point(
        lambda x: 0.5 * x + 1.0,
        0.0,
        options=SolverOptions(
            maxiter=2,
            xtol=1e-20,
            rtol=1e-20,
            ftol=1e-20,
        ),
    )

    assert result.converged is False
    assert result.iterations == 2


def test_fixed_point_history_length_matches_iterations():
    result = fixed_point(
        lambda x: 0.5 * x + 1.0,
        0.0,
    )

    assert result.history is not None
    assert len(result.history) == result.iterations

def test_fixed_point_respects_function_tolerance():
    result = fixed_point(
        lambda x: 0.5 * x + 1.0,
        0.0,
        options=SolverOptions(
            maxiter=10,
            xtol=1e-20,
            rtol=1e-20,
            ftol=1.1,
        ),
    )

    assert result.converged
    assert result.solution == 0.0
    assert result.iterations == 0
    assert result.residual == 1.0

def test_fixed_point_residual_matches_returned_solution():
    g = lambda x: 0.5 * x + 1.0

    result = fixed_point(g, 0.0)

    assert result.converged
    assert result.residual == pytest.approx(
        abs(g(result.solution) - result.solution)
    )