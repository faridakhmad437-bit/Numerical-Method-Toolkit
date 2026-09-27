import math
from pathlib import Path

import matplotlib.pyplot as plt

from numerical_toolkit.integration.simpson import simpson
from numerical_toolkit.integration.trapezoidal import trapezoidal


def estimate_order(error_n: float, error_2n: float) -> float:
    """Estimate convergence order from two consecutive errors."""
    return math.log(error_n / error_2n, 2)


def main() -> None:
    # Integral:
    #
    #     ∫_0^1 x^4 dx = 1/5
    #
    # x^4 is intentionally chosen because:
    # - Trapezoidal has O(h^2) error
    # - Simpson 1/3 has O(h^4) error
    # - Simpson is not exact for x^4

    f = lambda x: x**4
    exact = 1.0 / 5.0

    n_values = [4, 8, 16, 32, 64, 128, 256]

    trapezoidal_errors = []
    simpson_errors = []

    print("=" * 72)
    print("NUMERICAL INTEGRATION CONVERGENCE STUDY")
    print("=" * 72)
    print(f"Exact integral = {exact:.16f}")
    print()

    print(
        f"{'n':>6}"
        f"{'Trap. Error':>18}"
        f"{'Trap. Order':>16}"
        f"{'Simpson Error':>18}"
        f"{'Simpson Order':>16}"
    )
    print("-" * 74)

    previous_trap_error = None
    previous_simpson_error = None

    for n in n_values:
        trap_result = trapezoidal(f, 0.0, 1.0, n)
        simpson_result = simpson(f, 0.0, 1.0, n)

        trap_error = abs(trap_result - exact)
        simpson_error = abs(simpson_result - exact)

        trapezoidal_errors.append(trap_error)
        simpson_errors.append(simpson_error)

        if previous_trap_error is None:
            trap_order_text = "-"
            simpson_order_text = "-"
        else:
            trap_order = estimate_order(previous_trap_error, trap_error)
            simpson_order = estimate_order(
                previous_simpson_error,
                simpson_error,
            )

            trap_order_text = f"{trap_order:.6f}"
            simpson_order_text = f"{simpson_order:.6f}"

        print(
            f"{n:>6}"
            f"{trap_error:>18.6e}"
            f"{trap_order_text:>16}"
            f"{simpson_error:>18.6e}"
            f"{simpson_order_text:>16}"
        )

        previous_trap_error = trap_error
        previous_simpson_error = simpson_error

    print()
    print("Expected theoretical orders:")
    print("  Trapezoidal ≈ 2")
    print("  Simpson     ≈ 4")

    # ------------------------------------------------------------------
    # Plot convergence
    # ------------------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.loglog(
        n_values,
        trapezoidal_errors,
        marker="o",
        label="Trapezoidal",
    )

    plt.loglog(
        n_values,
        simpson_errors,
        marker="s",
        label="Simpson 1/3",
    )

    plt.xlabel("Number of subintervals (n)")
    plt.ylabel("Absolute error")
    plt.title("Numerical Integration Convergence")
    plt.grid(True, which="both")
    plt.legend()

    output_path = output_dir / "integration_convergence.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()