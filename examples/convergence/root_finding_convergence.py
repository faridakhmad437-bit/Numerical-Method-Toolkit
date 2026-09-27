from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


ROOT = math.sqrt(2.0)


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

    # ---------------------------------------------------------
    # Solve using all four methods
    # ---------------------------------------------------------

    bisection_result = bisection(
        f,
        1.0,
        2.0,
        options=options,
    )

    newton_result = newton(
        f,
        df,
        1.5,
        options=options,
    )

    secant_result = secant(
        f,
        1.0,
        2.0,
        options=options,
    )

    # Fixed-point formulation for sqrt(2):
    #
    #     x = 0.5 * (x + 2/x)
    #
    fixed_point_function = lambda x: 0.5 * (x + 2.0 / x)

    fixed_point_result = fixed_point(
        fixed_point_function,
        1.0,
        options=options,
    )

    results = {
        "Bisection": bisection_result,
        "Newton": newton_result,
        "Secant": secant_result,
        "Fixed Point": fixed_point_result,
    }

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("=" * 70)
    print("ROOT-FINDING CONVERGENCE STUDY")
    print("=" * 70)
    print(f"True root: {ROOT:.16f}")
    print()

    for name, result in results.items():
        print(f"{name:12s}")
        print(f"  solution     = {result.solution:.16f}")
        print(f"  converged    = {result.converged}")
        print(f"  iterations   = {result.iterations}")
        print(f"  function calls = {result.function_calls}")
        print(f"  final error  = {abs(result.solution - ROOT):.6e}")
        print()

    # ---------------------------------------------------------
    # Extract errors
    # ---------------------------------------------------------

    errors: dict[str, list[float]] = {}

    for name, result in results.items():

        if result.history is None:
            continue

        method_errors: list[float] = []

        for value in result.history:

            # -------------------------------------------------
            # History format differs between implementations.
            # -------------------------------------------------

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

            # Avoid log(0) in plotting.
            error = max(error, 1e-16)

            method_errors.append(error)

        if method_errors:
            errors[name] = method_errors

    # ---------------------------------------------------------
    # Plot convergence
    # ---------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 6))

    for name, method_errors in errors.items():

        iterations = range(1, len(method_errors) + 1)

        plt.semilogy(
            iterations,
            method_errors,
            marker="o",
            markersize=3,
            label=name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.title("Root-Finding Convergence")
    plt.grid(True, which="both")
    plt.legend()

    output_path = (
        output_dir / "root_finding_convergence.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("=" * 70)
    print(f"Plot saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()