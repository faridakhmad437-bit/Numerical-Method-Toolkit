from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from numerical_toolkit.interpolation.polynomial import (
    lagrange,
    newton_interpolation,
)


OUTPUT_DIR = "outputs"

NODE_COUNTS = [5, 10, 15, 20, 25, 30]


def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def chebyshev_nodes(n):
    k = np.arange(n)

    return np.cos(
        (2.0 * k + 1.0)
        * np.pi
        / (2.0 * n)
    )


def max_absolute_error(reference, approximation):
    return float(
        np.max(
            np.abs(
                reference - approximation
            )
        )
    )


def rmse(reference, approximation):
    return float(
        np.sqrt(
            np.mean(
                (reference - approximation) ** 2
            )
        )
    )


def evaluate_interpolation(
    x_nodes,
    y_nodes,
    x_test,
):
    y_lagrange = np.array(
        [
            lagrange(
                x_nodes,
                y_nodes,
                xq,
            )
            for xq in x_test
        ]
    )

    y_newton = np.array(
        [
            newton_interpolation(
                x_nodes,
                y_nodes,
                xq,
            )
            for xq in x_test
        ]
    )

    return y_lagrange, y_newton


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    x_test = np.linspace(
        -1.0,
        1.0,
        1001,
    )

    y_exact = runge_function(
        x_test
    )

    results = []

    print("=" * 100)
    print("EQUALLY SPACED VS CHEBYSHEV INTERPOLATION")
    print("=" * 100)

    for n in NODE_COUNTS:

        # ================================================================
        # EQUALLY SPACED NODES
        # ================================================================

        x_equal = np.linspace(
            -1.0,
            1.0,
            n,
        )

        y_equal = runge_function(
            x_equal
        )

        equal_lagrange, equal_newton = (
            evaluate_interpolation(
                x_equal,
                y_equal,
                x_test,
            )
        )

        # ================================================================
        # CHEBYSHEV NODES
        # ================================================================

        x_cheb = chebyshev_nodes(n)

        # Sort nodes so plotting is visually ordered.
        x_cheb = np.sort(x_cheb)

        y_cheb = runge_function(
            x_cheb
        )

        cheb_lagrange, cheb_newton = (
            evaluate_interpolation(
                x_cheb,
                y_cheb,
                x_test,
            )
        )

        # ================================================================
        # ERROR METRICS
        # ================================================================

        equal_max = max_absolute_error(
            y_exact,
            equal_lagrange,
        )

        cheb_max = max_absolute_error(
            y_exact,
            cheb_lagrange,
        )

        equal_rmse = rmse(
            y_exact,
            equal_lagrange,
        )

        cheb_rmse = rmse(
            y_exact,
            cheb_lagrange,
        )

        equal_agreement = max_absolute_error(
            equal_lagrange,
            equal_newton,
        )

        cheb_agreement = max_absolute_error(
            cheb_lagrange,
            cheb_newton,
        )

        # ================================================================
        # NEAR-ENDPOINT REGION
        # ================================================================

        endpoint_region = (
            np.abs(x_test) >= 0.9
        )

        equal_endpoint_error = float(
            np.max(
                np.abs(
                    y_exact[endpoint_region]
                    - equal_lagrange[
                        endpoint_region
                    ]
                )
            )
        )

        cheb_endpoint_error = float(
            np.max(
                np.abs(
                    y_exact[endpoint_region]
                    - cheb_lagrange[
                        endpoint_region
                    ]
                )
            )
        )

        # ================================================================
        # STORE RESULTS
        # ================================================================

        results.append(
            {
                "n": n,
                "equal_max": equal_max,
                "cheb_max": cheb_max,
                "equal_rmse": equal_rmse,
                "cheb_rmse": cheb_rmse,
                "equal_endpoint": equal_endpoint_error,
                "cheb_endpoint": cheb_endpoint_error,
                "equal_agreement": equal_agreement,
                "cheb_agreement": cheb_agreement,
            }
        )

        # ================================================================
        # PRINT
        # ================================================================

        print()
        print("-" * 100)
        print(f"Nodes: {n}")
        print("-" * 100)

        print(
            f"Equally spaced max error : "
            f"{equal_max:.6e}"
        )

        print(
            f"Chebyshev max error      : "
            f"{cheb_max:.6e}"
        )

        print(
            f"Equally spaced RMSE      : "
            f"{equal_rmse:.6e}"
        )

        print(
            f"Chebyshev RMSE           : "
            f"{cheb_rmse:.6e}"
        )

        print(
            f"Equally spaced near-endpoint error : "
            f"{equal_endpoint_error:.6e}"
        )

        print(
            f"Chebyshev near-endpoint error      : "
            f"{cheb_endpoint_error:.6e}"
        )

        print(
            f"Lagrange/Newton agreement "
            f"(equally spaced): "
            f"{equal_agreement:.6e}"
        )

        print(
            f"Lagrange/Newton agreement "
            f"(Chebyshev): "
            f"{cheb_agreement:.6e}"
        )

        # ================================================================
        # VISUAL COMPARISON
        # ================================================================

        if n in [10, 20, 30]:

            plt.figure(
                figsize=(10, 6)
            )

            plt.plot(
                x_test,
                y_exact,
                label="Exact Runge function",
                linewidth=2,
            )

            plt.plot(
                x_test,
                equal_lagrange,
                linestyle="--",
                label="Equally spaced",
            )

            plt.plot(
                x_test,
                cheb_lagrange,
                linestyle=":",
                label="Chebyshev nodes",
            )

            plt.scatter(
                x_equal,
                y_equal,
                label="Equally spaced nodes",
            )

            plt.scatter(
                x_cheb,
                y_cheb,
                marker="x",
                label="Chebyshev nodes",
            )

            plt.xlabel("x")
            plt.ylabel("f(x)")

            plt.title(
                f"Node Distribution Comparison — n={n}"
            )

            plt.grid(
                alpha=0.3
            )

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"node_comparison_n{n}.png",
                ),
                dpi=150,
            )

            plt.close()

    # ====================================================================
    # CONVERT RESULTS
    # ====================================================================

    n_values = np.array(
        [
            r["n"]
            for r in results
        ]
    )

    equal_max = np.array(
        [
            r["equal_max"]
            for r in results
        ]
    )

    cheb_max = np.array(
        [
            r["cheb_max"]
            for r in results
        ]
    )

    equal_rmse = np.array(
        [
            r["equal_rmse"]
            for r in results
        ]
    )

    cheb_rmse = np.array(
        [
            r["cheb_rmse"]
            for r in results
        ]
    )

    equal_endpoint = np.array(
        [
            r["equal_endpoint"]
            for r in results
        ]
    )

    cheb_endpoint = np.array(
        [
            r["cheb_endpoint"]
            for r in results
        ]
    )

    # ====================================================================
    # MAX ERROR
    # ====================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        equal_max,
        marker="o",
        label="Equally spaced",
    )

    plt.semilogy(
        n_values,
        cheb_max,
        marker="s",
        label="Chebyshev",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "Maximum absolute error"
    )

    plt.title(
        "Equally Spaced vs Chebyshev Interpolation"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "equal_vs_chebyshev_max_error.png",
        ),
        dpi=150,
    )

    plt.close()

    # ====================================================================
    # RMSE
    # ====================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        equal_rmse,
        marker="o",
        label="Equally spaced",
    )

    plt.semilogy(
        n_values,
        cheb_rmse,
        marker="s",
        label="Chebyshev",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "RMSE: Equally Spaced vs Chebyshev"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "equal_vs_chebyshev_rmse.png",
        ),
        dpi=150,
    )

    plt.close()

    # ====================================================================
    # NEAR-ENDPOINT ERROR
    # ====================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        equal_endpoint,
        marker="o",
        label="Equally spaced",
    )

    plt.semilogy(
        n_values,
        cheb_endpoint,
        marker="s",
        label="Chebyshev",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "Maximum error for |x| >= 0.9"
    )

    plt.title(
        "Near-Endpoint Error Comparison"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "equal_vs_chebyshev_endpoint_error.png",
        ),
        dpi=150,
    )

    plt.close()

    # ====================================================================
    # SUMMARY
    # ====================================================================

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"{'Nodes':>8}"
        f"{'Equal Max':>18}"
        f"{'Cheb Max':>18}"
        f"{'Equal RMSE':>18}"
        f"{'Cheb RMSE':>18}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['n']:>8}"
            f"{r['equal_max']:>18.6e}"
            f"{r['cheb_max']:>18.6e}"
            f"{r['equal_rmse']:>18.6e}"
            f"{r['cheb_rmse']:>18.6e}"
        )

    print()
    print("=" * 100)
    print("Experiment completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()