from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.roots.bisection import bisection
from numerical_toolkit.roots.fixed_point import fixed_point
from numerical_toolkit.roots.newton import newton
from numerical_toolkit.roots.secant import secant


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def empirical_orders(errors: np.ndarray) -> np.ndarray:
    """
    Estimate empirical convergence order from an error sequence.

    p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1})

    Only finite, positive errors are used. Values at or below
    the floating-point noise floor are excluded.
    """
    errors = np.asarray(errors, dtype=float)

    machine_epsilon = np.finfo(float).eps
    valid = np.isfinite(errors) & (errors > 10.0 * machine_epsilon)

    filtered = errors[valid]

    if len(filtered) < 3:
        return np.array([], dtype=float)

    orders = []

    for k in range(1, len(filtered) - 1):
        e_prev = filtered[k - 1]
        e_curr = filtered[k]
        e_next = filtered[k + 1]

        denominator = np.log(e_curr / e_prev)

        if denominator == 0.0:
            continue

        order = np.log(e_next / e_curr) / denominator

        if np.isfinite(order):
            orders.append(order)

    return np.asarray(orders, dtype=float)


def run_convergence_study() -> None:
    """Compare root-finding convergence for f(x) = x^2 - 2."""

    exact = np.sqrt(2.0)

    def f(x: float) -> float:
        return x**2 - 2.0

    def df(x: float) -> float:
        return 2.0 * x

    def g(x: float) -> float:
        return 0.5 * (x + 2.0 / x)

    options = SolverOptions(
        xtol=1e-12,
        rtol=1e-10,
        ftol=1e-12,
        maxiter=100,
        store_history=True,
    )

    solvers = {
        "Bisection": lambda: bisection(
            f,
            1.0,
            2.0,
            options=options,
        ),
        "Newton": lambda: newton(
            f,
            df,
            1.5,
            options=options,
        ),
        "Secant": lambda: secant(
            f,
            1.0,
            2.0,
            options=options,
        ),
        "Fixed Point": lambda: fixed_point(
            g,
            1.5,
            options=options,
        ),
    }

    machine_epsilon = np.finfo(float).eps

    print()
    print("=" * 78)
    print("ROOT-FINDING CONVERGENCE STUDY")
    print("Problem: f(x) = x^2 - 2")
    print(f"Exact solution: sqrt(2) = {exact:.16f}")
    print(f"float64 machine epsilon: {machine_epsilon:.3e}")
    print("=" * 78)

    plt.figure(figsize=(10, 6))

    for method_name, solver in solvers.items():
        result = solver()

        if result.history is None:
            continue

        errors = np.asarray(
            [abs(float(x) - exact) for x in result.history],
            dtype=float,
        )

        plot_errors = np.maximum(
            errors,
            machine_epsilon,
        )

        iterations = np.arange(len(plot_errors))

        plt.semilogy(
            iterations,
            plot_errors,
            marker="o",
            markersize=4,
            label=method_name,
        )

        orders = empirical_orders(errors)

        if len(orders) > 0:
            # Use the final few valid order estimates because
            # they are usually the most representative of
            # the asymptotic convergence regime.
            tail = orders[-3:]
            empirical_order = float(np.median(tail))

            print(
                f"{method_name:<15}"
                f"iterations={result.iterations:<4d}"
                f"final_error={errors[-1]:.3e}    "
                f"empirical_order≈{empirical_order:.4f}"
            )
        else:
            print(
                f"{method_name:<15}"
                f"iterations={result.iterations:<4d}"
                f"final_error={errors[-1]:.3e}    "
                f"empirical_order=N/A"
            )

    plt.axhline(
        machine_epsilon,
        linestyle="--",
        linewidth=1.0,
        label="float64 machine epsilon",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.title("Root-Finding Convergence: f(x) = x² - 2")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "root_convergence.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Saved convergence plot to: {output_path}")
    print("=" * 78)


if __name__ == "__main__":
    run_convergence_study()