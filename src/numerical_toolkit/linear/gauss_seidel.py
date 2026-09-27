from __future__ import annotations

from typing import Any

import numpy as np

from numerical_toolkit.core.exceptions import InvalidInputError, NonFiniteValueError
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
    Solve a linear system Ax = b using the Gauss-Seidel iteration.

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
        Result containing the approximate solution, convergence status,
        iteration count, residual, step error, and optional history.

    Notes
    -----
    Gauss-Seidel updates the solution components in-place within each
    iteration, so newly computed values are immediately used.
    """

    # ---------------------------------------------------------
    # Options
    # ---------------------------------------------------------
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    # ---------------------------------------------------------
    # Convert inputs
    # ---------------------------------------------------------
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # ---------------------------------------------------------
    # Validate matrix
    # ---------------------------------------------------------
    if A.ndim != 2:
        raise InvalidInputError(
            "A must be a two-dimensional matrix."
        )

    if A.shape[0] != A.shape[1]:
        raise InvalidInputError(
            "A must be square."
        )

    n = A.shape[0]

    if n == 0:
        raise InvalidInputError(
            "A and b must define a non-empty linear system."
        )

    # ---------------------------------------------------------
    # Validate vector
    # ---------------------------------------------------------
    if b.shape != (n,):
        raise InvalidInputError(
            "b must be a one-dimensional vector with the same length as A."
        )

    # ---------------------------------------------------------
    # Validate finite values
    # ---------------------------------------------------------
    if not np.all(np.isfinite(A)):
        raise NonFiniteValueError(
            "A must contain only finite values."
        )

    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError(
            "b must contain only finite values."
        )

    # ---------------------------------------------------------
    # Validate diagonal
    # ---------------------------------------------------------
    diagonal = np.diag(A)

    if np.any(np.abs(diagonal) < 1e-15):
        raise InvalidInputError(
            "Gauss-Seidel requires nonzero diagonal entries."
        )

    # ---------------------------------------------------------
    # Initial guess
    # ---------------------------------------------------------
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x0, dtype=float).copy()

        if x.shape != b.shape:
            raise InvalidInputError(
                "x0 must have the same shape as b."
            )

        if not np.all(np.isfinite(x)):
            raise NonFiniteValueError(
                "x0 must contain only finite values."
            )

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------
    history: list[Any] = []

    residual = float(np.linalg.norm(A @ x - b, ord=np.inf))
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

    # ---------------------------------------------------------
    # Initial convergence check
    # ---------------------------------------------------------
    if residual <= options.ftol:
        return SolverResult(
            solution=x,
            converged=True,
            iterations=0,
            function_calls=0,
            residual=residual,
            step_error=None,
            message="Gauss-Seidel converged at the initial guess.",
            history=tuple(history) if options.store_history else None,
        )

    # ---------------------------------------------------------
    # Gauss-Seidel iteration
    # ---------------------------------------------------------
    for k in range(1, options.maxiter + 1):

        x_old = x.copy()

        for i in range(n):
            # Contribution from already updated values.
            lower = np.dot(A[i, :i], x[:i])

            # Contribution from values that have not yet been updated.
            upper = np.dot(A[i, i + 1 :], x_old[i + 1 :])

            x[i] = (b[i] - lower - upper) / A[i, i]

        # -----------------------------------------------------
        # Numerical safety
        # -----------------------------------------------------
        if not np.all(np.isfinite(x)):
            raise NonFiniteValueError(
                "Gauss-Seidel iteration produced a non-finite value."
            )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------
        step_error = float(
            np.linalg.norm(x - x_old, ord=np.inf)
        )

        residual = float(
            np.linalg.norm(A @ x - b, ord=np.inf)
        )

        if options.store_history:
            history.append(
                {
                    "iteration": k,
                    "solution": x.copy(),
                    "residual": residual,
                    "step_error": step_error,
                }
            )

        # -----------------------------------------------------
        # Convergence
        # -----------------------------------------------------
        if (
            step_error <= options.xtol
            and residual <= options.ftol
        ):
            return SolverResult(
                solution=x.copy(),
                converged=True,
                iterations=k,
                function_calls=0,
                residual=residual,
                step_error=step_error,
                message="Gauss-Seidel converged successfully.",
                history=(
                    tuple(history)
                    if options.store_history
                    else None
                ),
            )

    # ---------------------------------------------------------
    # Maximum iterations reached
    # ---------------------------------------------------------
    return SolverResult(
        solution=x.copy(),
        converged=False,
        iterations=options.maxiter,
        function_calls=0,
        residual=residual,
        step_error=step_error,
        message="Gauss-Seidel reached the maximum number of iterations.",
        history=(
            tuple(history)
            if options.store_history
            else None
        ),
    )