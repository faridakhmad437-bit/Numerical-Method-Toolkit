from __future__ import annotations

import math

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


ROOT = math.sqrt(2.0)


def extract_errors(result) -> list[float]:
    """Extract usable absolute errors from solver history."""

    if result.history is None:
        return []

    errors = []

    for value in result.history:

        if isinstance(value, dict):

            if "x" in value:
                x = value["x"]

            elif "midpoint" in value:
                x = value["midpoint"]

            else:
                continue

        else:
            x = value

        error = abs(float(x) - ROOT)

        if error > 1e-14:
            errors.append(error)

    return errors


def estimate_order(errors: list[float]) -> float | None:
    """
    Estimate convergence order from the final stable errors.

    Uses the last usable triple of errors that remains
    sufficiently far from machine precision.
    """

    if len(errors) < 3:
        return None

    for i in range(len(errors) - 3, -1, -1):

        e0 = errors[i]
        e1 = errors[i + 1]
        e2 = errors[i + 2]

        if min(e0, e1, e2) <= 1e-14:
            continue

        denominator = math.log(e1 / e0)

        if abs(denominator) < 1e-12:
            continue

        numerator = math.log(e2 / e1)

        return numerator / denominator

    return None


def bisection_bound_order(
    initial_width: float,
    iterations: int,
) -> float:
    """
    Theoretical convergence order of the Bisection error bound.

    The interval width is divided by 2 every iteration.
    """

    if iterations < 2:
        return float("nan")

    return 1.0


def main() -> None:

    f = lambda x: x**2 - 2.0
    df = lambda x: 2.0 * x

    options = SolverOptions(
        xtol=1e-14,
        rtol=1e-14,
        ftol=1e-14,
        maxiter=100,
        store_history=True,
    )

    results = {
        "Bisection": bisection(
            f,
            1.0,
            2.0,
            options=options,
        ),

        "Newton": newton(
            f,
            df,
            1.5,
            options=options,
        ),

        "Secant": secant(
            f,
            1.0,
            2.0,
            options=options,
        ),

        "Fixed Point": fixed_point(
            lambda x: 0.5 * (x + 2.0 / x),
            1.0,
            options=options,
        ),
    }

    print("=" * 70)
    print("REFINED ROOT-FINDING CONVERGENCE STUDY")
    print("=" * 70)
    print(f"Reference root: {ROOT:.16f}")
    print()

    # ---------------------------------------------------------
    # Bisection
    # ---------------------------------------------------------

    bisection_result = results["Bisection"]

    print("Bisection")
    print("-" * 70)

    if bisection_result.history is not None:

        initial_width = 1.0

        print(
            "Theoretical convergence of interval bound: p = 1 "
            "(linear convergence)"
        )

        print(
            "Interval width after n iterations:"
        )

        for n in [1, 2, 5, 10, 20, 30, 40]:

            width = initial_width / (2**n)

            print(
                f"  n={n:2d}: width = {width:.6e}"
            )

    print()

    # ---------------------------------------------------------
    # Newton
    # ---------------------------------------------------------

    newton_errors = extract_errors(
        results["Newton"]
    )

    newton_order = estimate_order(
        newton_errors
    )

    print("Newton")
    print("-" * 70)
    print(f"Estimated order: {newton_order:.6f}")
    print("Theoretical order: 2.0")
    print()

    # ---------------------------------------------------------
    # Secant
    # ---------------------------------------------------------

    secant_errors = extract_errors(
        results["Secant"]
    )

    secant_order = estimate_order(
        secant_errors
    )

    print("Secant")
    print("-" * 70)

    if secant_order is not None:
        print(
            f"Estimated order: {secant_order:.6f}"
        )

    print(
        f"Theoretical order: "
        f"{(1 + math.sqrt(5)) / 2:.6f}"
    )
    print()

    # ---------------------------------------------------------
    # Fixed Point
    # ---------------------------------------------------------

    fixed_point_errors = extract_errors(
        results["Fixed Point"]
    )

    fixed_point_order = estimate_order(
        fixed_point_errors
    )

    print("Fixed Point")
    print("-" * 70)

    if fixed_point_order is not None:
        print(
            f"Estimated order: "
            f"{fixed_point_order:.6f}"
        )

    print(
        "For this specific g(x), the iteration is "
        "algebraically equivalent to Newton-Raphson."
    )

    print(
        "Therefore the observed order is expected to "
        "be approximately 2."
    )

    print()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Bisection   : linear, guaranteed under valid bracket"
    )

    print(
        "Newton      : quadratic near a simple root"
    )

    print(
        "Secant      : superlinear, theoretical order ≈ 1.618"
    )

    print(
        "Fixed Point : depends on g(x); this specific mapping "
        "is equivalent to Newton"
    )


if __name__ == "__main__":
    main()