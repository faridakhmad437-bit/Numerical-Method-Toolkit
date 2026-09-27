from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from numerical_toolkit.interpolation.polynomial import (
    lagrange,
    newton_interpolation,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "outputs"

NODE_COUNTS = [
    5,
    10,
    15,
    20,
    25,
    30,
]


# ============================================================================
# TEST FUNCTION
# ============================================================================


def runge_function(x):
    """
    Classical Runge function.

    f(x) = 1 / (1 + 25 x^2)
    """

    return 1.0 / (1.0 + 25.0 * x**2)


# ============================================================================
# ERROR METRICS
# ============================================================================


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


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # Dense grid for evaluating interpolation quality.
    x_test = np.linspace(
        -1.0,
        1.0,
        1001,
    )

    y_exact = runge_function(x_test)

    results = []

    print("=" * 90)
    print("INTERPOLATION CONVERGENCE STUDY")
    print("=" * 90)

    for n in NODE_COUNTS:

        # ---------------------------------------------------------------
        # Equally spaced interpolation nodes
        # ---------------------------------------------------------------

        x_nodes = np.linspace(
            -1.0,
            1.0,
            n,
        )

        y_nodes = runge_function(
            x_nodes
        )

        # ---------------------------------------------------------------
        # Evaluate both interpolation methods
        # ---------------------------------------------------------------

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

        # ---------------------------------------------------------------
        # Error metrics
        # ---------------------------------------------------------------

        lagrange_max_error = max_absolute_error(
            y_exact,
            y_lagrange,
        )

        newton_max_error = max_absolute_error(
            y_exact,
            y_newton,
        )

        lagrange_rmse = rmse(
            y_exact,
            y_lagrange,
        )

        newton_rmse = rmse(
            y_exact,
            y_newton,
        )

        agreement_error = max_absolute_error(
            y_lagrange,
            y_newton,
        )

        # ---------------------------------------------------------------
        # Endpoint error
        # ---------------------------------------------------------------

        endpoint_indices = [
            0,
            1,
            -2,
            -1,
        ]

        endpoint_error_lagrange = float(
            np.max(
                np.abs(
                    y_exact[endpoint_indices]
                    - y_lagrange[endpoint_indices]
                )
            )
        )

        endpoint_error_newton = float(
            np.max(
                np.abs(
                    y_exact[endpoint_indices]
                    - y_newton[endpoint_indices]
                )
            )
        )

        # ---------------------------------------------------------------
        # Store results
        # ---------------------------------------------------------------

        results.append(
            {
                "n": n,
                "lagrange_max_error": lagrange_max_error,
                "newton_max_error": newton_max_error,
                "lagrange_rmse": lagrange_rmse,
                "newton_rmse": newton_rmse,
                "agreement_error": agreement_error,
                "endpoint_error_lagrange": endpoint_error_lagrange,
                "endpoint_error_newton": endpoint_error_newton,
            }
        )

        # ---------------------------------------------------------------
        # Console output
        # ---------------------------------------------------------------

        print()
        print("-" * 90)
        print(f"Number of nodes: {n}")
        print("-" * 90)

        print(
            f"Lagrange max error : "
            f"{lagrange_max_error:.6e}"
        )

        print(
            f"Newton max error   : "
            f"{newton_max_error:.6e}"
        )

        print(
            f"Lagrange RMSE      : "
            f"{lagrange_rmse:.6e}"
        )

        print(
            f"Newton RMSE        : "
            f"{newton_rmse:.6e}"
        )

        print(
            f"Lagrange/Newton agreement error : "
            f"{agreement_error:.6e}"
        )

        print(
            f"Endpoint error (Lagrange) : "
            f"{endpoint_error_lagrange:.6e}"
        )

        print(
            f"Endpoint error (Newton)   : "
            f"{endpoint_error_newton:.6e}"
        )

        # ---------------------------------------------------------------
        # Plot interpolants for selected node counts
        # ---------------------------------------------------------------

        if n in [5, 10, 20, 30]:

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
                y_lagrange,
                label="Lagrange interpolation",
                linestyle="--",
            )

            plt.plot(
                x_test,
                y_newton,
                label="Newton interpolation",
                linestyle=":",
            )

            plt.scatter(
                x_nodes,
                y_nodes,
                label="Interpolation nodes",
            )

            plt.xlabel("x")
            plt.ylabel("f(x)")

            plt.title(
                f"Runge Phenomenon — "
                f"{n} Equally Spaced Nodes"
            )

            plt.grid(
                alpha=0.3
            )

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"runge_interpolation_n{n}.png",
                ),
                dpi=150,
            )

            plt.close()

    # =========================================================================
    # CONVERT RESULTS
    # =========================================================================

    n_values = np.array(
        [
            result["n"]
            for result in results
        ]
    )

    lagrange_max_errors = np.array(
        [
            result["lagrange_max_error"]
            for result in results
        ]
    )

    newton_max_errors = np.array(
        [
            result["newton_max_error"]
            for result in results
        ]
    )

    lagrange_rmse_values = np.array(
        [
            result["lagrange_rmse"]
            for result in results
        ]
    )

    newton_rmse_values = np.array(
        [
            result["newton_rmse"]
            for result in results
        ]
    )

    agreement_errors = np.array(
        [
            result["agreement_error"]
            for result in results
        ]
    )

    endpoint_lagrange = np.array(
        [
            result["endpoint_error_lagrange"]
            for result in results
        ]
    )

    endpoint_newton = np.array(
        [
            result["endpoint_error_newton"]
            for result in results
        ]
    )

    # =========================================================================
    # ERROR VS NUMBER OF NODES
    # =========================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        lagrange_max_errors,
        marker="o",
        label="Lagrange max error",
    )

    plt.semilogy(
        n_values,
        newton_max_errors,
        marker="s",
        label="Newton max error",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "Maximum absolute error"
    )

    plt.title(
        "Interpolation Error vs Number of Nodes"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "interpolation_error_vs_nodes.png",
        ),
        dpi=150,
    )

    plt.close()

    # =========================================================================
    # RMSE VS NUMBER OF NODES
    # =========================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        lagrange_rmse_values,
        marker="o",
        label="Lagrange RMSE",
    )

    plt.semilogy(
        n_values,
        newton_rmse_values,
        marker="s",
        label="Newton RMSE",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Interpolation RMSE vs Number of Nodes"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "interpolation_rmse_vs_nodes.png",
        ),
        dpi=150,
    )

    plt.close()

    # =========================================================================
    # LAGRANGE VS NEWTON AGREEMENT
    # =========================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        agreement_errors,
        marker="o",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "Maximum |P_L(x) - P_N(x)|"
    )

    plt.title(
        "Lagrange vs Newton Agreement"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lagrange_vs_newton_agreement.png",
        ),
        dpi=150,
    )

    plt.close()

    # =========================================================================
    # ENDPOINT ERROR
    # =========================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.semilogy(
        n_values,
        endpoint_lagrange,
        marker="o",
        label="Lagrange",
    )

    plt.semilogy(
        n_values,
        endpoint_newton,
        marker="s",
        label="Newton",
    )

    plt.xlabel(
        "Number of interpolation nodes"
    )

    plt.ylabel(
        "Endpoint absolute error"
    )

    plt.title(
        "Runge Phenomenon Near Interval Endpoints"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "runge_endpoint_error.png",
        ),
        dpi=150,
    )

    plt.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"{'Nodes':>8} "
        f"{'Lagrange Max':>18} "
        f"{'Newton Max':>18} "
        f"{'L/N Agreement':>18}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['n']:>8} "
            f"{result['lagrange_max_error']:>18.6e} "
            f"{result['newton_max_error']:>18.6e} "
            f"{result['agreement_error']:>18.6e}"
        )

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print(
        "Error vs nodes : "
        f"{os.path.join(OUTPUT_DIR, 'interpolation_error_vs_nodes.png')}"
    )

    print(
        "RMSE vs nodes  : "
        f"{os.path.join(OUTPUT_DIR, 'interpolation_rmse_vs_nodes.png')}"
    )

    print(
        "Lagrange/Newton agreement : "
        f"{os.path.join(OUTPUT_DIR, 'lagrange_vs_newton_agreement.png')}"
    )

    print(
        "Endpoint error : "
        f"{os.path.join(OUTPUT_DIR, 'runge_endpoint_error.png')}"
    )

    print()
    print(
        "Runge plots:"
    )

    for n in [5, 10, 20, 30]:
        print(
            f"  n={n}: "
            f"{os.path.join(OUTPUT_DIR, f'runge_interpolation_n{n}.png')}"
        )

    print()
    print("Experiment completed.")


if __name__ == "__main__":
    main()