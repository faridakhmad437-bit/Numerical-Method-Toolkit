
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


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
        # Validate tolerances before checking their numerical values.
        for name in ("xtol", "rtol", "ftol"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")

            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(self.maxiter, bool) or not isinstance(
            self.maxiter, Integral
        ):
            raise TypeError("maxiter must be an integer.")

        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive.")

        if not isinstance(self.store_history, bool):
            raise TypeError("store_history must be a boolean.")
