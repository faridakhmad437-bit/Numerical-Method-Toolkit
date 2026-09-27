import math

import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    SecantDenominatorZeroError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.secant import secant


def test_secant_finds_sqrt_2():
    result = secant(
        lambda x: x**2 - 2,
        1.0,
        2.0,
    )

    assert result.converged
    assert abs(result.solution - math.sqrt(2)) < 1e-8


def test_secant_finds_linear_root():
    result = secant(
        lambda x: 3 * x - 6,
        0.0,
        3.0,
    )

    assert result.converged
    assert abs(result.solution - 2.0) < 1e-10


def test_secant_accepts_exact_first_initial_root():
    result = secant(
        lambda x: x - 2,
        2.0,
        5.0,
    )

    assert result.converged
    assert result.solution == 2.0
    assert result.iterations == 0


def test_secant_accepts_exact_second_initial_root():
    result = secant(
        lambda x: x - 2,
        1.0,
        2.0,
    )

    assert result.converged
    assert result.solution == 2.0
    assert result.iterations == 0


def test_secant_rejects_identical_initial_guesses():
    with pytest.raises(InvalidInputError):
        secant(
            lambda x: x**2 - 2,
            1.0,
            1.0,
        )


def test_secant_rejects_zero_denominator():
    with pytest.raises(SecantDenominatorZeroError):
        secant(
            lambda x: 1.0,
            0.0,
            1.0,
        )


def test_secant_can_disable_history():
    result = secant(
        lambda x: x**2 - 2,
        1.0,
        2.0,
        options=SolverOptions(store_history=False),
    )

    assert result.converged
    assert result.history is None


def test_secant_stops_after_max_iterations():
    result = secant(
        lambda x: x**2 - 2,
        1.0,
        2.0,
        options=SolverOptions(
            maxiter=1,
            xtol=1e-20,
            rtol=1e-20,
            ftol=1e-20,
        ),
    )

    assert result.converged is False
    assert result.iterations == 1


def test_secant_history_is_recorded():
    result = secant(
        lambda x: x**2 - 2,
        1.0,
        2.0,
    )

    assert result.history is not None
    assert len(result.history) == result.iterations + 2