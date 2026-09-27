import math
from pathlib import Path

import matplotlib.pyplot as plt

from numerical_toolkit.ode.rk4 import rk4


def estimate_order(error_h: float, error_h2: float) -> float:
    """Estimate convergence order from two consecutive step sizes."""
    return math.log(error_h / error_h2, 2)


def main() -> None:
    # ODE:
    #
    #     y' = y
    #     y(0) = 1
    #
    # Exact solution:
    #
    #     y(t) = exp(t)
    #
    # At t = 1:
    #
    #     y(1) = e

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

    errors = []

    print("=" * 72)
    print("RK4 CONVERGENCE STUDY")
    print("=" * 72)
    print(f"Exact solution y(1) = {exact:.16f}")
    print()

    print(
        f"{'h':>12}"
        f"{'Numerical y(1)':>20}"
        f"{'Absolute Error':>20}"
        f"{'Estimated Order':>20}"
    )

    print("-" * 76)

    previous_error = None

    for h in h_values:
        _, values = rk4(
            f,
            t0=t0,
            y0=y0,
            t_end=t_end,
            h=h,
        )

        numerical = float(values[-1])
        error = abs(numerical - exact)

        errors.append(error)

        if previous_error is None:
            order_text = "-"
        else:
            order = estimate_order(previous_error, error)
            order_text = f"{order:.6f}"

        print(
            f"{h:>12.6f}"
            f"{numerical:>20.12f}"
            f"{error:>20.6e}"
            f"{order_text:>20}"
        )

        previous_error = error

    print()
    print("Expected theoretical order:")
    print("  RK4 ≈ 4")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.loglog(
        h_values,
        errors,
        marker="o",
        label="RK4",
    )

    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute error at t = 1")
    plt.title("RK4 Convergence")
    plt.grid(True, which="both")
    plt.legend()

    output_path = output_dir / "rk4_convergence.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()