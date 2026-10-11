
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult


def gauss_seidel(
    A,
    b,
    x0=None,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """
    Solve a linear system Ax = b using Gauss-Seidel iteration.

    Parameters
    ----------
    A : array-like
        Square coefficient matrix.
    b : array-like
        Right-hand-side vector.
    x0 : array-like, optional
        Initial guess. Defaults to the zero vector.
    options : SolverOptions, optional
        Convergence and iteration settings.

    Returns
    -------
    SolverResult
        Approximate solution and convergence diagnostics.
    """
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    # Validate and convert inputs without silently accepting booleans.
    try:
        raw_A = np.asarray(A)
        raw_b = np.asarray(b)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(
            "A and b must be valid numeric arrays."
        ) from exc

    if raw_A.dtype.kind == "b" or raw_b.dtype.kind == "b":
        raise InvalidInputError("Boolean inputs are not supported.")

    try:
        A = np.asarray(raw_A, dtype=float)
        b = np.asarray(raw_b, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(
            "A and b must contain numeric values."
        ) from exc

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

    # Reject exact zeros, not valid small diagonal entries.
    if np.any(diagonal == 0.0):
        raise InvalidInputError(
            "Gauss-Seidel requires nonzero diagonal entries."
        )

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        try:
            raw_x0 = np.asarray(x0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(
                "x0 must be a valid numeric array."
            ) from exc

        if raw_x0.dtype.kind == "b":
            raise InvalidInputError(
                "Boolean initial guesses are not supported."
            )

        try:
            x = np.asarray(raw_x0, dtype=float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(
                "x0 must contain numeric values."
            ) from exc

        if x.shape != (n,):
            raise InvalidInputError(
                "x0 must have the same shape as b."
            )

        if not np.all(np.isfinite(x)):
            raise NonFiniteValueError(
                "x0 must contain only finite values."
            )

    # Scale the system to improve numerical behavior when coefficients
    # are uniformly very small or very large.
    matrix_scale = float(np.max(np.abs(A)))

    if matrix_scale == 0.0:
        raise InvalidInputError(
            "Gauss-Seidel cannot solve a zero coefficient matrix."
        )

    A_scaled = A / matrix_scale
    b_scaled = b / matrix_scale
    diagonal_scaled = np.diag(A_scaled)

    if np.any(diagonal_scaled == 0.0):
        raise InvalidInputError(
            "A diagonal entry is too small to scale safely."
        )

    history: list[Any] = []

    def calculate_residual(solution: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            residual_vector = A @ solution - b

        if not np.all(np.isfinite(residual_vector)):
            raise NonFiniteValueError(
                "Gauss-Seidel residual became non-finite."
            )

        residual = float(np.linalg.norm(residual_vector, ord=np.inf))

        if not math.isfinite(residual):
            raise NonFiniteValueError(
                "Gauss-Seidel residual became non-finite."
            )

        return residual

    residual = calculate_residual(x)
    step_error: float | None = None

    if options.store_history:
        history.append(
            {
                "iteration": 0,
                "solution": x.copy(),
                "residual": residual,
                "step_error": None,
            }
        )

    if residual <= options.ftol:
        return SolverResult(
            solution=x.copy(),
            converged=True,
            iterations=0,
            function_calls=0,
            residual=residual,
            step_error=None,
            message="Gauss-Seidel converged at the initial guess.",
            history=tuple(history) if options.store_history else None,
        )

    for k in range(1, options.maxiter + 1):
        x_old = x.copy()

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i in range(n):
                lower = np.dot(A_scaled[i, :i], x[:i])
                upper = np.dot(A_scaled[i, i + 1 :], x_old[i + 1 :])

                x[i] = (
                    b_scaled[i] - lower - upper
                ) / diagonal_scaled[i]

        if not np.all(np.isfinite(x)):
            raise NonFiniteValueError(
                "Gauss-Seidel iteration produced a non-finite value."
            )

        step_error = float(np.linalg.norm(x - x_old, ord=np.inf))

        if not math.isfinite(step_error):
            raise NonFiniteValueError(
                "Gauss-Seidel step error became non-finite."
            )

        residual = calculate_residual(x)

        if options.store_history:
            history.append(
                {
                    "iteration": k,
                    "solution": x.copy(),
                    "residual": residual,
                    "step_error": step_error,
                }
            )

        if step_error <= options.xtol and residual <= options.ftol:
            return SolverResult(
                solution=x.copy(),
                converged=True,
                iterations=k,
                function_calls=0,
                residual=residual,
                step_error=step_error,
                message="Gauss-Seidel converged successfully.",
                history=tuple(history) if options.store_history else None,
            )

    return SolverResult(
        solution=x.copy(),
        converged=False,
        iterations=options.maxiter,
        function_calls=0,
        residual=residual,
        step_error=step_error,
        message="Gauss-Seidel reached the maximum number of iterations.",
        history=tuple(history) if options.store_history else None,
    )
