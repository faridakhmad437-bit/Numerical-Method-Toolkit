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


@pytest.mark.parametrize(
    "field",
    ["iterations", "function_calls"],
)
@pytest.mark.parametrize(
    "value",
    [1.5, 2.0, True, False, float("nan"), float("inf")],
)
def test_count_fields_must_be_integers(field, value):
    kwargs = {
        "solution": 2.0,
        "converged": True,
        "iterations": 1,
        "function_calls": 1,
        "residual": 0.0,
        "step_error": 0.0,
        "message": "Converged.",
    }
    kwargs[field] = value

    with pytest.raises(TypeError):
        SolverResult(**kwargs)


@pytest.mark.parametrize("field", ["residual", "step_error"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_diagnostics_must_be_finite(field, value):
    kwargs = {
        "solution": 2.0,
        "converged": True,
        "iterations": 1,
        "function_calls": 1,
        "residual": 0.0,
        "step_error": 0.0,
        "message": "Converged.",
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        SolverResult(**kwargs)


@pytest.mark.parametrize("field", ["residual", "step_error"])
@pytest.mark.parametrize("value", [True, False, "0.1"])
def test_diagnostics_reject_invalid_types(field, value):
    kwargs = {
        "solution": 2.0,
        "converged": True,
        "iterations": 1,
        "function_calls": 1,
        "residual": 0.0,
        "step_error": 0.0,
        "message": "Converged.",
    }
    kwargs[field] = value

    with pytest.raises(TypeError):
        SolverResult(**kwargs)


@pytest.mark.parametrize("message", [None, 123])
def test_message_must_be_a_string(message):
    with pytest.raises(TypeError):
        SolverResult(
            solution=2.0,
            converged=True,
            iterations=1,
            function_calls=1,
            residual=0.0,
            step_error=0.0,
            message=message,
        )
