import pytest

from numerical_toolkit.core.results import SolverResult


def test_solver_result():
    result = SolverResult(
        solution=2.0,
        converged=True,
        iterations=5,
        function_calls=7,
        residual=1e-10,
        step_error=1e-8,
        message="Converged.",
        history=(),
    )

    assert result.solution == 2.0
    assert result.converged is True
    assert result.iterations == 5
    assert result.function_calls == 7


def test_negative_iterations_rejected():
    with pytest.raises(ValueError):
        SolverResult(
            solution=2.0,
            converged=False,
            iterations=-1,
            function_calls=1,
            residual=None,
            step_error=None,
            message="Failed.",
        )


def test_negative_function_calls_rejected():
    with pytest.raises(ValueError):
        SolverResult(
            solution=2.0,
            converged=False,
            iterations=1,
            function_calls=-1,
            residual=None,
            step_error=None,
            message="Failed.",
        )


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        SolverResult(
            solution=2.0,
            converged=False,
            iterations=1,
            function_calls=1,
            residual=None,
            step_error=None,
            message="",
        )