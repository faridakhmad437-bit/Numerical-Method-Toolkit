from __future__ import annotations

import math

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


ROOT = math.sqrt(2.0)


def estimate_orders(errors: list[float]) -> list[float]:
    """
    Estimate convergence order from a sequence of errors.

    p_n =
        log(e_{n+1}/e_n)
        -----------------
        log(e_n/e_{n-1})
    """

    orders: list[float] = []

    for i in range(2, len(errors)):

        e_previous = errors[i - 2]
        e_current = errors[i - 1]
        e_next = errors[i]

        if (
            e_previous <= 0
            or e_current <= 0
            or e_next <= 0
        ):
            continue

        numerator = math.log(e_next / e_current)
        denominator = math.log(e_current / e_previous)

        if abs(denominator) < 1e-15:
            continue

        order = numerator / denominator

        orders.append(order)

    return orders


def extract_errors(result) -> list[float]:
    """
    Extract absolute errors from solver history.
    """

    if result.history is None:
        return []

    errors: list[float] = []

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

        # Ignore floating-point zero.
        if error > 1e-15:
            errors.append(error)

    return errors


def print_order(name: str, errors: list[float]) -> None:

    orders = estimate_orders(errors)

    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Number of usable errors: {len(errors)}")

    if not orders:
        print("Not enough data to estimate convergence order.")
        return

    print("Estimated orders:")

    for i, order in enumerate(orders, start=3):
        print(
            f"  iteration {i:2d}: "
            f"p ≈ {order:.6f}"
        )

    print()

    # Use the last few estimates as an asymptotic indicator.
    tail = orders[-3:]

    if tail:
        average = sum(tail) / len(tail)

        print(
            f"Average of final estimates: "
            f"{average:.6f}"
        )


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

    print()
    print("ROOT-FINDING CONVERGENCE ORDER")
    print(f"Reference root: {ROOT:.16f}")
    print()

    for name, result in results.items():

        errors = extract_errors(result)

        print_order(
            name,
            errors,
        )


if __name__ == "__main__":
    main()