
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from math import isfinite
from numbers import Integral, Real

@dataclass(frozen=True)
class SolverResult:
    """
    Standard result returned by iterative numerical solvers.

    Attributes
    ----------
    solution:
        Final numerical solution.
    converged:
        Whether the solver satisfied its convergence criterion.
    iterations:
        Number of iterations performed.
    function_calls:
        Number of callable evaluations counted by the solver.

        Depending on the algorithm, this may include evaluations
        of the objective or residual function and evaluations of
        its derivative or Jacobian. Consult the individual solver's
        documentation for the exact counting convention.

        This field does not represent the total number of numerical
        operations performed internally by the solver.
    residual:
        Final residual or error of the governing equation.
    step_error:
        Size of the final solution update, when available.
    message:
        Human-readable solver status.
    history:
        Optional iteration history.
    """

    solution: Any
    converged: bool
    iterations: int
    function_calls: int
    residual: float | None
    step_error: float | None
    message: str
    history: tuple[Any, ...] | None = None
        
    def __post_init__(self) -> None:
        # Validate iteration and evaluation counts.
        for name in ("iterations", "function_calls"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if not isinstance(self.converged, bool):
            raise TypeError(
                "converged must be a boolean."
            )

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

        if not self.message.strip():
            raise ValueError("message must not be empty.")

        # Validate optional numerical diagnostics.
        for name in ("residual", "step_error"):
            value = getattr(self, name)

            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number or None.")

            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

