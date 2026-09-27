import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

from numerical_toolkit.ode.euler import euler
from numerical_toolkit.ode.rk4 import rk4


def estimate_order(error_h: float, error_h2: float) -> float:
    """Estimate convergence order from two consecutive errors."""
    return math.log(error_h / error_h2, 2)


def count_steps(t0: float, t_end: float, h: float) -> int:
    """Count the number of integration steps."""
    steps = 0
    t = t0

    while t < t_end - 1e-15:
        dt = min(h, t_end - t)
        t += dt
        steps += 1

    return steps


def main() -> None:
    # ---------------------------------------------------------------
    # Problem definition
    # ---------------------------------------------------------------
    #
    #     y' = y
    #     y(0) = 1
    #
    # Exact:
    #
    #     y(t) = exp(t)
    #
    # Therefore:
    #
    #     y(1) = e
    #
    # ---------------------------------------------------------------

    f = lambda t, y: y

    t0 = 0.0
    y0 = 1.0
    t_end = 1.0
    exact = math.e

    h_values = [
        0.2,
        0.1,
        0.05,
        0.025,
        0.0125,
        0.00625,
    ]

    euler_errors = []
    rk4_errors = []

    euler_evals = []
    rk4_evals = []

    euler_times = []
    rk4_times = []

    previous_euler_error = None
    previous_rk4_error = None

    print("=" * 100)
    print("EULER vs RK4: ACCURACY vs COMPUTATIONAL COST")
    print("=" * 100)
    print(f"Exact solution y(1) = {exact:.16f}")
    print()

    print(
        f"{'h':>10}"
        f"{'Euler Error':>18}"
        f"{'Euler Order':>14}"
        f"{'Euler Eval':>14}"
        f"{'Euler Time':>14}"
        f"{'RK4 Error':>18}"
        f"{'RK4 Order':>14}"
        f"{'RK4 Eval':>14}"
        f"{'RK4 Time':>14}"
    )

    print("-" * 100)

    for h in h_values:
        steps = count_steps(t0, t_end, h)

        # -----------------------------------------------------------
        # Euler
        # -----------------------------------------------------------

        start = time.perf_counter()

        _, euler_values = euler(
            f,
            t0=t0,
            y0=y0,
            t_end=t_end,
            h=h,
        )

        euler_runtime = time.perf_counter() - start

        euler_solution = float(euler_values[-1])
        euler_error = abs(euler_solution - exact)

        # Euler evaluates f once per step.
        euler_function_evaluations = steps

        # -----------------------------------------------------------
        # RK4
        # -----------------------------------------------------------

        start = time.perf_counter()

        _, rk4_values = rk4(
            f,
            t0=t0,
            y0=y0,
            t_end=t_end,
            h=h,
        )

        rk4_runtime = time.perf_counter() - start

        rk4_solution = float(rk4_values[-1])
        rk4_error = abs(rk4_solution - exact)

        # RK4 evaluates f four times per step.
        rk4_function_evaluations = 4 * steps

        # -----------------------------------------------------------
        # Store results
        # -----------------------------------------------------------

        euler_errors.append(euler_error)
        rk4_errors.append(rk4_error)

        euler_evals.append(euler_function_evaluations)
        rk4_evals.append(rk4_function_evaluations)

        euler_times.append(euler_runtime)
        rk4_times.append(rk4_runtime)

        # -----------------------------------------------------------
        # Estimate convergence order
        # -----------------------------------------------------------

        if previous_euler_error is None:
            euler_order_text = "-"
            rk4_order_text = "-"
        else:
            euler_order = estimate_order(
                previous_euler_error,
                euler_error,
            )

            rk4_order = estimate_order(
                previous_rk4_error,
                rk4_error,
            )

            euler_order_text = f"{euler_order:.4f}"
            rk4_order_text = f"{rk4_order:.4f}"

        print(
            f"{h:>10.6f}"
            f"{euler_error:>18.6e}"
            f"{euler_order_text:>14}"
            f"{euler_function_evaluations:>14}"
            f"{euler_runtime:>14.6e}"
            f"{rk4_error:>18.6e}"
            f"{rk4_order_text:>14}"
            f"{rk4_function_evaluations:>14}"
            f"{rk4_runtime:>14.6e}"
        )

        previous_euler_error = euler_error
        previous_rk4_error = rk4_error

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("Theoretical convergence:")
    print("  Euler ≈ O(h^1)")
    print("  RK4   ≈ O(h^4)")

    print()
    print("Function evaluation cost per step:")
    print("  Euler = 1 evaluation")
    print("  RK4   = 4 evaluations")

    # ---------------------------------------------------------------
    # Plot 1: Error vs step size
    # ---------------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.loglog(
        h_values,
        euler_errors,
        marker="o",
        label="Euler",
    )

    plt.loglog(
        h_values,
        rk4_errors,
        marker="s",
        label="RK4",
    )

    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute error at t = 1")
    plt.title("Euler vs RK4: Accuracy")
    plt.grid(True, which="both")
    plt.legend()

    error_plot = output_dir / "euler_vs_rk4_error.png"

    plt.tight_layout()
    plt.savefig(error_plot, dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: Error vs function evaluations
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.loglog(
        euler_evals,
        euler_errors,
        marker="o",
        label="Euler",
    )

    plt.loglog(
        rk4_evals,
        rk4_errors,
        marker="s",
        label="RK4",
    )

    plt.xlabel("Function evaluations")
    plt.ylabel("Absolute error at t = 1")
    plt.title("Euler vs RK4: Accuracy vs Computational Cost")
    plt.grid(True, which="both")
    plt.legend()

    cost_plot = output_dir / "euler_vs_rk4_cost.png"

    plt.tight_layout()
    plt.savefig(cost_plot, dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Plot 3: Runtime vs error
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.loglog(
        euler_times,
        euler_errors,
        marker="o",
        label="Euler",
    )

    plt.loglog(
        rk4_times,
        rk4_errors,
        marker="s",
        label="RK4",
    )

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Absolute error at t = 1")
    plt.title("Euler vs RK4: Runtime vs Accuracy")
    plt.grid(True, which="both")
    plt.legend()

    runtime_plot = output_dir / "euler_vs_rk4_runtime.png"

    plt.tight_layout()
    plt.savefig(runtime_plot, dpi=150)
    plt.close()

    print()
    print(f"Error plot saved to: {error_plot}")
    print(f"Cost plot saved to: {cost_plot}")
    print(f"Runtime plot saved to: {runtime_plot}")


if __name__ == "__main__":
    main()