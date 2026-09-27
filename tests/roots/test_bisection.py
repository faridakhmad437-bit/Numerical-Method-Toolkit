import math

import pytest

from numerical_toolkit.core.exceptions import InvalidBracketError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection


def test_bisection_finds_sqrt_2():
    result = bisection(
        lambda x: x**2 - 2,
        1.0,
        2.0,
    )

    assert result.converged
    assert abs(result.solution - math.sqrt(2)) < 1e-8
    assert result.residual is not None
    assert result.residual < 1e-8


def test_bisection_preserves_bracket_problem():
    result = bisection(
        lambda x: x**3 - 1,
        0.0,
        2.0,
    )

    assert result.converged
    assert abs(result.solution - 1.0) < 1e-8


def test_bisection_accepts_left_endpoint_root():
    result = bisection(
        lambda x: x**2 - 1,
        1.0,
        2.0,
    )

    assert result.converged
    assert result.solution == 1.0
    assert result.iterations == 0


def test_bisection_accepts_right_endpoint_root():
    result = bisection(
        lambda x: x**2 - 1,
        0.0,
        1.0,
    )

    assert result.converged
    assert result.solution == 1.0
    assert result.iterations == 0


def test_bisection_rejects_invalid_interval():
    with pytest.raises(InvalidBracketError):
        bisection(
            lambda x: x**2 + 1,
            -1.0,
            1.0,
        )


def test_bisection_rejects_reversed_interval():
    with pytest.raises(InvalidBracketError):
        bisection(
            lambda x: x**2 - 2,
            2.0,
            1.0,
        )


def test_bisection_can_disable_history():
    result = bisection(
        lambda x: x**2 - 2,
        1.0,
        2.0,
        options=SolverOptions(store_history=False),
    )

    assert result.converged
    assert result.history is None


def test_bisection_stops_after_max_iterations():
    result = bisection(
        lambda x: x**2 - 2,
        1.0,
        2.0,
        options=SolverOptions(
            maxiter=2,
            xtol=1e-20,
            rtol=1e-20,
            ftol=1e-20,
        ),
    )

    assert result.converged is False
    assert result.iterations == 2


def test_bisection_history_contains_midpoints():
    result = bisection(
        lambda x: x**2 - 2,
        1.0,
        2.0,
    )

    assert result.history is not None
    assert len(result.history) == result.iterations