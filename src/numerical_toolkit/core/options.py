from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverOptions:
    """
    Common configuration for iterative numerical solvers.

    Parameters
    ----------
    xtol:
        Absolute tolerance for the solution/step.
    rtol:
        Relative tolerance for the solution/step.
    ftol:
        Absolute tolerance for the residual.
    maxiter:
        Maximum number of iterations.
    store_history:
        Whether iteration history should be stored.
    """

    xtol: float = 1e-10
    rtol: float = 1e-8
    ftol: float = 1e-10
    maxiter: int = 100
    store_history: bool = True

    def __post_init__(self) -> None:
        if self.xtol <= 0:
            raise ValueError("xtol must be positive.")

        if self.rtol <= 0:
            raise ValueError("rtol must be positive.")

        if self.ftol <= 0:
            raise ValueError("ftol must be positive.")

        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        if not isinstance(self.store_history, bool):
            raise TypeError("store_history must be a boolean.")