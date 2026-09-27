from __future__ import annotations

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult


def _validate_initial_guess(x0) -> np.ndarray:
    """Validate Newton system initial guess."""

    x = np.asarray(x0, dtype=float)

    if x.ndim != 1:
        raise InvalidInputError(
            "x0 must be a one-dimensional vector."
        )

    if len(x) == 0:
        raise InvalidInputError(
            "x0 must not be empty."
        )

    if not np.all(np.isfinite(x)):
        raise NonFiniteValueError(
            "x0 must contain only finite values."
        )

    return x.copy()


def newton_system(
    F,
    J,
    x0,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """
    Newton's method for a nonlinear system.

    Solves

        F(x) = 0

    using

        J(x_k) * delta = -F(x_k)

        x_(k+1) = x_k + delta

    Parameters
    ----------
    F : callable
        Vector-valued nonlinear function.

    J : callable
        Jacobian matrix of F.

    x0 : array-like
        Initial guess.

    options : SolverOptions, optional
        Convergence and iteration settings.

    Returns
    -------
    SolverResult
        Result containing the approximate solution, convergence status,
        iteration count, function calls, residual, step error,
        message, and optional history.
    """

    # ---------------------------------------------------------
    # Validate callables
    # ---------------------------------------------------------
    if not callable(F):
        raise InvalidInputError(
            "F must be callable."
        )

    if not callable(J):
        raise InvalidInputError(
            "J must be callable."
        )

    # ---------------------------------------------------------
    # Validate options
    # ---------------------------------------------------------
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    # ---------------------------------------------------------
    # Validate initial guess
    # ---------------------------------------------------------
    x = _validate_initial_guess(x0)
    n = len(x)

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------
    history = []

    # ---------------------------------------------------------
    # Initial function evaluation
    # ---------------------------------------------------------
    fx = np.asarray(
        F(x),
        dtype=float,
    )

    function_calls = 1

    # ---------------------------------------------------------
    # Validate F(x)
    # ---------------------------------------------------------
    if fx.ndim != 1 or fx.shape != (n,):
        raise InvalidInputError(
            "F must return a one-dimensional vector "
            "with the same length as x0."
        )

    if not np.all(np.isfinite(fx)):
        raise NonFiniteValueError(
            "F(x) must contain only finite values."
        )

    residual = float(
        np.linalg.norm(
            fx,
            ord=np.inf,
        )
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

    # ---------------------------------------------------------
    # Initial convergence
    # ---------------------------------------------------------
    if residual <= options.ftol:
        return SolverResult(
            solution=x.copy(),
            converged=True,
            iterations=0,
            function_calls=function_calls,
            residual=residual,
            step_error=None,
            message="Newton system converged at the initial guess.",
            history=(
                tuple(history)
                if options.store_history
                else None
            ),
        )

    # ---------------------------------------------------------
    # Newton iteration
    # ---------------------------------------------------------
    for k in range(1, options.maxiter + 1):

        # -----------------------------------------------------
        # Evaluate Jacobian
        # -----------------------------------------------------
        jac = np.asarray(
            J(x),
            dtype=float,
        )

        function_calls += 1

        # -----------------------------------------------------
        # Validate Jacobian dimensions
        # -----------------------------------------------------
        if jac.shape != (n, n):
            raise InvalidInputError(
                "J must return a square matrix "
                "with dimensions matching x0."
            )

        # -----------------------------------------------------
        # Validate Jacobian values
        # -----------------------------------------------------
        if not np.all(np.isfinite(jac)):
            raise NonFiniteValueError(
                "J(x) must contain only finite values."
            )

        # -----------------------------------------------------
        # Newton step
        # -----------------------------------------------------
        try:
            delta = np.linalg.solve(
                jac,
                -fx,
            )
        except np.linalg.LinAlgError as exc:
            raise InvalidInputError(
                "Jacobian is singular."
            ) from exc

        # -----------------------------------------------------
        # Validate Newton step
        # -----------------------------------------------------
        if not np.all(np.isfinite(delta)):
            raise NonFiniteValueError(
                "Newton step must contain only finite values."
            )

        step_error = float(
            np.linalg.norm(
                delta,
                ord=np.inf,
            )
        )

        # -----------------------------------------------------
        # Update solution
        # -----------------------------------------------------
        x_new = x + delta

        if not np.all(np.isfinite(x_new)):
            raise NonFiniteValueError(
                "Newton iteration produced a non-finite value."
            )

        # -----------------------------------------------------
        # Evaluate new residual
        # -----------------------------------------------------
        fx_new = np.asarray(
            F(x_new),
            dtype=float,
        )

        function_calls += 1

        # -----------------------------------------------------
        # Validate new F(x)
        # -----------------------------------------------------
        if fx_new.ndim != 1 or fx_new.shape != (n,):
            raise InvalidInputError(
                "F must return a one-dimensional vector "
                "with the same length as x0."
            )

        if not np.all(np.isfinite(fx_new)):
            raise NonFiniteValueError(
                "F(x) must contain only finite values."
            )

        new_residual = float(
            np.linalg.norm(
                fx_new,
                ord=np.inf,
            )
        )

        # -----------------------------------------------------
        # Store history
        # -----------------------------------------------------
        if options.store_history:
            history.append(
                {
                    "iteration": k,
                    "solution": x_new.copy(),
                    "residual": new_residual,
                    "step_error": step_error,
                }
            )

        # -----------------------------------------------------
        # Convergence
        #
        # A small step alone is insufficient. Both the step and
        # residual must satisfy their respective tolerances.
        # -----------------------------------------------------
        if (
            step_error <= options.xtol
            and new_residual <= options.ftol
        ):
            return SolverResult(
                solution=x_new.copy(),
                converged=True,
                iterations=k,
                function_calls=function_calls,
                residual=new_residual,
                step_error=step_error,
                message="Newton system converged successfully.",
                history=(
                    tuple(history)
                    if options.store_history
                    else None
                ),
            )

        # -----------------------------------------------------
        # Continue iteration
        # -----------------------------------------------------
        x = x_new
        fx = fx_new
        residual = new_residual

    # ---------------------------------------------------------
    # Maximum iterations reached
    # ---------------------------------------------------------
    return SolverResult(
        solution=x.copy(),
        converged=False,
        iterations=options.maxiter,
        function_calls=function_calls,
        residual=residual,
        step_error=step_error,
        message=(
            "Newton system reached the maximum number "
            "of iterations."
        ),
        history=(
            tuple(history)
            if options.store_history
            else None
        ),
    )


def _validate_target(target) -> np.ndarray:
    """Validate a 2D planar target."""

    target_array = np.asarray(
        target,
        dtype=float,
    )

    if target_array.ndim != 1:
        raise InvalidInputError(
            "target must be a one-dimensional vector."
        )

    if target_array.shape != (2,):
        raise InvalidInputError(
            "target must contain exactly two values: (x, y)."
        )

    if not np.all(
        np.isfinite(target_array)
    ):
        raise NonFiniteValueError(
            "target must contain only finite values."
        )

    return target_array


def _validate_link_length(
    length: float,
    name: str,
) -> float:
    """Validate a planar robot link length."""

    if isinstance(length, bool):
        raise InvalidInputError(
            f"{name} must be a positive finite number."
        )

    try:
        length = float(length)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"{name} must be a positive finite number."
        ) from exc

    if not np.isfinite(length):
        raise NonFiniteValueError(
            f"{name} must be finite."
        )

    if length <= 0:
        raise InvalidInputError(
            f"{name} must be positive."
        )

    return length


def planar_2link_ik(
    target,
    l1,
    l2,
    initial=(0.0, 0.0),
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """
    Solve inverse kinematics for a planar 2-link manipulator.

    Parameters
    ----------
    target : array-like
        Desired end-effector position (x, y).

    l1 : float
        Length of the first link.

    l2 : float
        Length of the second link.

    initial : array-like, optional
        Initial joint-angle guess.

    options : SolverOptions, optional
        Convergence and iteration settings.

    Returns
    -------
    SolverResult
        Result containing joint angles as ``solution``.

    Notes
    -----
    If the target lies outside the robot workspace, the solver
    returns a non-converged SolverResult with zero iterations.
    """

    # ---------------------------------------------------------
    # Validate target and link lengths
    # ---------------------------------------------------------
    target = _validate_target(target)

    l1 = _validate_link_length(
        l1,
        "l1",
    )

    l2 = _validate_link_length(
        l2,
        "l2",
    )

    # ---------------------------------------------------------
    # Validate options
    # ---------------------------------------------------------
    if options is None:
        options = SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    # ---------------------------------------------------------
    # Validate initial guess
    # ---------------------------------------------------------
    initial = _validate_initial_guess(initial)

    if initial.shape != (2,):
        raise InvalidInputError(
            "initial must contain exactly two joint angles."
        )

    # ---------------------------------------------------------
    # Workspace validation
    # ---------------------------------------------------------
    distance = float(
        np.linalg.norm(target)
    )

    min_reach = abs(l1 - l2)
    max_reach = l1 + l2

    if distance < min_reach or distance > max_reach:
        return SolverResult(
            solution=initial.copy(),
            converged=False,
            iterations=0,
            function_calls=0,
            residual=None,
            step_error=None,
            message=(
                "Target lies outside the reachable workspace "
                "of the planar 2-link manipulator."
            ),
            history=(
                (
                    {
                        "iteration": 0,
                        "solution": initial.copy(),
                        "residual": None,
                        "step_error": None,
                    },
                )
                if options.store_history
                else None
            ),
        )

    tx, ty = target

    # ---------------------------------------------------------
    # Nonlinear system F(q) = 0
    # ---------------------------------------------------------
    def F(q):
        q1, q2 = q

        return np.array(
            [
                (
                    l1 * np.cos(q1)
                    + l2 * np.cos(q1 + q2)
                    - tx
                ),
                (
                    l1 * np.sin(q1)
                    + l2 * np.sin(q1 + q2)
                    - ty
                ),
            ]
        )

    # ---------------------------------------------------------
    # Jacobian
    # ---------------------------------------------------------
    def J(q):
        q1, q2 = q

        return np.array(
            [
                [
                    (
                        -l1 * np.sin(q1)
                        -l2 * np.sin(q1 + q2)
                    ),
                    (
                        -l2 * np.sin(q1 + q2)
                    ),
                ],
                [
                    (
                        l1 * np.cos(q1)
                        + l2 * np.cos(q1 + q2)
                    ),
                    (
                        l2 * np.cos(q1 + q2)
                    ),
                ],
            ]
        )

    # ---------------------------------------------------------
    # Solve using Newton System
    # ---------------------------------------------------------
    return newton_system(
        F,
        J,
        initial,
        options=options,
    )