from __future__ import annotations

import math

from numerical_toolkit.core.exceptions import (
    DerivativeZeroError,
    InvalidBracketError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_case(name: str, solver, *args, **kwargs) -> None:
    """
    Run one solver case and print the result or expected failure.
    """

    print(f"\n{name}")
    print("-" * 70)

    try:
        result = solver(*args, **kwargs)

        print(f"Converged : {result.converged}")
        print(f"Solution  : {result.solution}")
        print(f"Iterations: {result.iterations}")
        print(f"Message   : {result.message}")

        if result.residual is not None:
            print(f"Residual  : {result.residual:.6e}")

        if result.step_error is not None:
            print(f"Step error: {result.step_error:.6e}")

    except (
        InvalidBracketError,
        DerivativeZeroError,
        NonFiniteValueError,
    ) as exc:
        print(f"Expected solver failure: {type(exc).__name__}")
        print(f"Message: {exc}")

    except Exception as exc:
        print(f"Unexpected failure: {type(exc).__name__}")
        print(f"Message: {exc}")


def main() -> None:

    options = SolverOptions(
        xtol=1e-12,
        rtol=1e-12,
        ftol=1e-12,
        maxiter=50,
        store_history=True,
    )

    # ==========================================================
    # 1. BISECTION — invalid bracket
    # ==========================================================

    print_header(
        "1. BISECTION — INVALID BRACKET"
    )

    print(
        "Problem: f(x) = x² + 1 has no real root."
    )

    run_case(
        "Bisection with [-1, 1]",
        bisection,
        lambda x: x**2 + 1,
        -1.0,
        1.0,
        options=options,
    )

    # ==========================================================
    # 2. BISECTION — valid bracket
    # ==========================================================

    print_header(
        "2. BISECTION — VALID BRACKET"
    )

    print(
        "Problem: f(x) = x² - 2."
    )

    run_case(
        "Bisection with [1, 2]",
        bisection,
        lambda x: x**2 - 2,
        1.0,
        2.0,
        options=options,
    )

    # ==========================================================
    # 3. NEWTON — derivative becomes zero
    # ==========================================================

    print_header(
        "3. NEWTON — ZERO DERIVATIVE"
    )

    print(
        "Problem: f(x) = x² + 1."
    )

    run_case(
        "Newton starting at x₀ = 0",
        newton,
        lambda x: x**2 + 1,
        lambda x: 2 * x,
        0.0,
        options=options,
    )

    # ==========================================================
    # 4. NEWTON — poor initial guess
    # ==========================================================

    print_header(
        "4. NEWTON — POOR INITIAL GUESS"
    )

    print(
        "Problem: f(x) = x³ - 2x + 2."
    )

    print(
        "This function has a real root, but Newton can behave "
        "poorly for certain initial guesses."
    )

    run_case(
        "Newton starting at x₀ = 0",
        newton,
        lambda x: x**3 - 2 * x + 2,
        lambda x: 3 * x**2 - 2,
        0.0,
        options=options,
    )

    # ==========================================================
    # 5. NEWTON — non-finite function
    # ==========================================================

    print_header(
        "5. NEWTON — NON-FINITE FUNCTION"
    )

    print(
        "Problem: f(x) deliberately returns infinity."
    )

    run_case(
        "Newton with non-finite f(x)",
        newton,
        lambda x: math.inf,
        lambda x: 1.0,
        1.0,
        options=options,
    )

    # ==========================================================
    # 6. SECANT — identical function values
    # ==========================================================

    print_header(
        "6. SECANT — ZERO SECANT DENOMINATOR"
    )

    print(
        "Problem: f(x) = 1 has identical function values."
    )

    run_case(
        "Secant with f(x) = 1",
        secant,
        lambda x: 1.0,
        0.0,
        1.0,
        options=options,
    )

    # ==========================================================
    # 7. FIXED POINT — convergent mapping
    # ==========================================================

    print_header(
        "7. FIXED POINT — CONVERGENT MAPPING"
    )

    print(
        "g(x) = 0.5x + 1"
    )

    run_case(
        "Fixed Point starting at x₀ = 0",
        fixed_point,
        lambda x: 0.5 * x + 1,
        0.0,
        options=options,
    )

    # ==========================================================
    # 8. FIXED POINT — divergent mapping
    # ==========================================================

    print_header(
        "8. FIXED POINT — DIVERGENT MAPPING"
    )

    print(
        "g(x) = 2x"
    )

    print(
        "The fixed point x = 0 exists, but |g'(x)| = 2 > 1."
    )

    divergent_options = SolverOptions(
        xtol=1e-12,
        rtol=1e-12,
        ftol=1e-12,
        maxiter=10,
        store_history=True,
    )

    run_case(
        "Fixed Point starting at x₀ = 1",
        fixed_point,
        lambda x: 2 * x,
        1.0,
        options=divergent_options,
    )

    # ==========================================================
    # Summary
    # ==========================================================

    print_header(
        "FAILURE-MODE STUDY SUMMARY"
    )

    print(
        """
Bisection:
    Requires a valid sign-changing bracket.

Newton:
    Requires a usable derivative and a suitable initial guess.

Secant:
    Does not require an explicit derivative, but can fail
    when consecutive function values become equal.

Fixed Point:
    Convergence depends strongly on the mapping g(x).
    A sufficient local condition is |g'(x*)| < 1.
"""
    )


if __name__ == "__main__":
    main()