from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SolverResult:
    """
    Standard result returned by iterative numerical solvers.

    Attributes
    ----------
    solution:
        Final numerical solution.
    converged:
        Whether the solver satisfied a convergence criterion.
    iterations:
        Number of iterations performed.
    function_calls:
        Number of function evaluations.
    residual:
        Final residual/error of the governing equation.
    step_error:
        Size of the final update.
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
        if self.iterations < 0:
            raise ValueError(
                "iterations must be non-negative."
            )

        if self.function_calls < 0:
            raise ValueError(
                "function_calls must be non-negative."
            )

        if not isinstance(self.converged, bool):
            raise TypeError(
                "converged must be a boolean."
            )

        if not self.message.strip():
            raise ValueError(
                "message must not be empty."
            )

        if self.residual is not None:
            if self.residual < 0:
                raise ValueError(
                    "residual must be non-negative."
                )

        if self.step_error is not None:
            if self.step_error < 0:
                raise ValueError(
                    "step_error must be non-negative."
                )