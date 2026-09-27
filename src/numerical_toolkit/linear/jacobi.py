from __future__ import annotations

from typing import Any

import numpy as np

from numerical_toolkit.core.exceptions import InvalidInputError, NonFiniteValueError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult


def jacobi(
    A,
    b,
    x0=None,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """
    Solve a linear system A @ x = b using the Jacobi iterative method.

    Parameters
    ----------
    A : array-like
        Square coefficient matrix.
    b : array-like
        Right-hand-side vector.
    x0 : array-like, optional
        Initial guess. Defaults to the zero vector.
    options : SolverOptions, optional
        Solver configuration.

    Returns
    -------
    SolverResult
        Numerical solution and convergence diagnostics.

    Notes
    -----
    Jacobi convergence is not guaranteed for arbitrary matrices.
    In practice, convergence is commonly associated with conditions
    such as strict diagonal dominance or spectral radius rho(B_J) < 1.
    """
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # ------------------------------------------------------------------
    # Validate system dimensions
    # ------------------------------------------------------------------

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
            "b must be a one-dimensional vector with the same "
            "length as A."
        )

    # ------------------------------------------------------------------
    # Validate finite values
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(A)):
        raise NonFiniteValueError(
            "A must contain only finite values."
        )

    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError(
            "b must contain only finite values."
        )

    # ------------------------------------------------------------------
    # Validate diagonal
    # ------------------------------------------------------------------

    diagonal = np.diag(A)

    if np.any(np.abs(diagonal) < 1e-15):
        raise InvalidInputError(
            "Jacobi requires nonzero diagonal entries."
        )

    # ------------------------------------------------------------------
    # Validate initial guess
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Jacobi decomposition
    # ------------------------------------------------------------------

    D = diagonal
    R = A - np.diagflat(D)

    history: list[Any] = []

    residual = float(
        np.linalg.norm(A @ x - b, ord=np.inf)
    )

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

    # Initial convergence check
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

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    for k in range(1, options.maxiter + 1):
        x_new = (b - R @ x) / D

        if not np.all(np.isfinite(x_new)):
            raise NonFiniteValueError(
                "Jacobi iteration produced a non-finite value."
            )

        step_error = float(
            np.linalg.norm(x_new - x, ord=np.inf)
        )

        residual = float(
            np.linalg.norm(A @ x_new - b, ord=np.inf)
        )

        if options.store_history:
            history.append(
                {
                    "iteration": k,
                    "solution": x_new.copy(),
                    "residual": residual,
                    "step_error": step_error,
                }
            )

        # Both the update and the equation residual should be small.
        if (
            step_error <= options.xtol
            and residual <= options.ftol
        ):
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

    # ------------------------------------------------------------------
    # Maximum iterations reached
    # ------------------------------------------------------------------

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