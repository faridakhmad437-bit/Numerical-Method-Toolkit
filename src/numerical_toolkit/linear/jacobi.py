
from __future__ import annotations

from typing import Any

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import as_float_array


def jacobi(
    A,
    b,
    x0=None,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """Solve A @ x = b using the Jacobi iterative method."""
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    A = as_float_array(A, "A")
    b = as_float_array(b, "b")

    if A.ndim != 2:
        raise InvalidInputError("A must be a two-dimensional matrix.")

    if A.shape[0] != A.shape[1]:
        raise InvalidInputError("A must be square.")

    n = A.shape[0]

    if n == 0:
        raise InvalidInputError(
            "A and b must define a non-empty linear system."
        )

    if b.shape != (n,):
        raise InvalidInputError(
            "b must be a one-dimensional vector with the same length as A."
        )

    if not np.all(np.isfinite(A)):
        raise NonFiniteValueError("A must contain only finite values.")

    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError("b must contain only finite values.")

    diagonal = np.diag(A)

    if np.any(diagonal == 0.0):
        raise InvalidInputError(
            "Jacobi requires nonzero diagonal entries."
        )

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = as_float_array(x0, "x0").copy()

        if x.shape != b.shape:
            raise InvalidInputError("x0 must have the same shape as b.")

        if not np.all(np.isfinite(x)):
            raise NonFiniteValueError(
                "x0 must contain only finite values."
            )

    # Normalize the system by its largest coefficient to reduce
    # intermediate scaling problems without changing the solution.
    matrix_scale = float(np.max(np.abs(A)))

    if matrix_scale == 0.0:
        raise InvalidInputError(
            "Jacobi cannot solve a system with a zero coefficient matrix."
        )

    A_scaled = A / matrix_scale
    b_scaled = b / matrix_scale
    diagonal_scaled = np.diag(A_scaled)

    R = A_scaled - np.diagflat(diagonal_scaled)

    history: list[Any] = []

    def calculate_residual(solution):
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.linalg.norm(A @ solution - b, ord=np.inf)

        if not np.isfinite(value):
            raise NonFiniteValueError(
                "Jacobi residual calculation produced a non-finite value."
            )

        return float(value)

    residual = calculate_residual(x)
    step_error: float | None = None

    if options.store_history:
        history.append({
            "iteration": 0,
            "solution": x.copy(),
            "residual": residual,
            "step_error": None,
        })

    if residual <= options.ftol:
        return SolverResult(
            solution=x,
            converged=True,
            iterations=0,
            function_calls=0,
            residual=residual,
            step_error=None,
            message="Jacobi converged at the initial guess.",
            history=tuple(history) if options.store_history else None,
        )

    for k in range(1, options.maxiter + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x_new = (b_scaled - R @ x) / diagonal_scaled

        if not np.all(np.isfinite(x_new)):
            raise NonFiniteValueError(
                "Jacobi iteration produced a non-finite value."
            )

        step_error = float(np.linalg.norm(x_new - x, ord=np.inf))
        residual = calculate_residual(x_new)

        if options.store_history:
            history.append({
                "iteration": k,
                "solution": x_new.copy(),
                "residual": residual,
                "step_error": step_error,
            })

        step_tolerance = (
            options.xtol
            + options.rtol * np.linalg.norm(x, ord=np.inf)
        )

        if step_error <= step_tolerance and residual <= options.ftol:
            return SolverResult(
                solution=x_new,
                converged=True,
                iterations=k,
                function_calls=0,
                residual=residual,
                step_error=step_error,
                message="Jacobi converged successfully.",
                history=tuple(history) if options.store_history else None,
            )

        x = x_new

    return SolverResult(
        solution=x,
        converged=False,
        iterations=options.maxiter,
        function_calls=0,
        residual=residual,
        step_error=step_error,
        message="Jacobi reached the maximum number of iterations.",
        history=tuple(history) if options.store_history else None,
    )
