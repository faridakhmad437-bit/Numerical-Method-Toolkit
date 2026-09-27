from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Gauss-Seidel theory helpers
# ---------------------------------------------------------------------------

def gauss_seidel_iteration_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct the Gauss-Seidel iteration matrix.

    For

        A = D + L + U

    the Gauss-Seidel iteration matrix is

        B_GS = -(D + L)^(-1) U
    """
    A = np.asarray(A, dtype=float)

    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    return -np.linalg.solve(
        D + L,
        U,
    )


def spectral_radius(B: np.ndarray) -> float:
    """
    Compute spectral radius:

        rho(B) = max |lambda_i|
    """
    eigenvalues = np.linalg.eigvals(B)

    return float(
        np.max(np.abs(eigenvalues))
    )


# ---------------------------------------------------------------------------
# Gauss-Seidel experiment with history
# ---------------------------------------------------------------------------

def gauss_seidel_history(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
):
    """
    Run Gauss-Seidel while storing convergence history.

    Returns
    -------
    x :
        Final solution estimate.

    converged :
        Whether the step-based stopping criterion was satisfied.

    iterations :
        Number of iterations.

    step_history :
        Infinity norm of x_(k+1) - x_k.

    residual_history :
        Infinity norm of b - A*x_k.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x0, dtype=float).copy()

    step_history = []

    residual_history = [
        float(
            np.linalg.norm(
                b - A @ x,
                ord=np.inf,
            )
        )
    ]

    for iteration in range(1, maxiter + 1):

        old = x.copy()

        for i in range(len(b)):

            lower = np.dot(
                A[i, :i],
                x[:i],
            )

            upper = np.dot(
                A[i, i + 1:],
                x[i + 1:],
            )

            x[i] = (
                b[i] - lower - upper
            ) / A[i, i]

        step_error = float(
            np.linalg.norm(
                x - old,
                ord=np.inf,
            )
        )

        residual = float(
            np.linalg.norm(
                b - A @ x,
                ord=np.inf,
            )
        )

        step_history.append(step_error)
        residual_history.append(residual)

        if step_error <= tol:
            return (
                x,
                True,
                iteration,
                step_history,
                residual_history,
            )

    return (
        x,
        False,
        maxiter,
        step_history,
        residual_history,
    )


# ---------------------------------------------------------------------------
# Observed convergence factor
# ---------------------------------------------------------------------------

def observed_convergence_factors(
    history: list[float],
) -> np.ndarray:
    """
    Compute consecutive convergence ratios:

        q_k = e_(k+1) / e_k
    """

    values = np.asarray(
        history,
        dtype=float,
    )

    if len(values) < 2:
        return np.array([], dtype=float)

    previous = values[:-1]
    current = values[1:]

    valid = (
        np.isfinite(previous)
        & np.isfinite(current)
        & (previous > 0)
        & (current > 0)
    )

    return current[valid] / previous[valid]


def estimate_asymptotic_factor(
    history: list[float],
    tail_size: int = 10,
) -> float:
    """
    Estimate the asymptotic convergence factor using
    the median of the final valid ratios.
    """

    factors = observed_convergence_factors(
        history
    )

    if len(factors) == 0:
        return float("nan")

    tail = factors[-tail_size:]

    return float(
        np.median(tail)
    )


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------

systems = {
    "A: Strongly Dominant": {
        "A": np.array(
            [
                [10.0, 1.0, 1.0],
                [2.0, 10.0, 1.0],
                [1.0, 2.0, 10.0],
            ]
        ),
        "b": np.array(
            [12.0, 13.0, 13.0]
        ),
    },

    "B: Weakly Dominant": {
        "A": np.array(
            [
                [4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0],
                [1.0, 1.0, 4.0],
            ]
        ),
        "b": np.array(
            [6.0, 6.0, 6.0]
        ),
    },

    "C: Slow Convergence": {
        "A": np.array(
            [
                [4.0, 3.0],
                [3.0, 4.0],
            ]
        ),
        "b": np.array(
            [7.0, 7.0]
        ),
    },

    "D: Divergent": {
        "A": np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        ),
        "b": np.array(
            [3.0, 3.0]
        ),
    },
}


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

results = {}

print("=" * 90)
print("GAUSS-SEIDEL CONVERGENCE STUDY")
print("=" * 90)

for name, system in systems.items():

    A = system["A"]
    b = system["b"]

    # ---------------------------------------------------------------
    # Theoretical analysis
    # ---------------------------------------------------------------

    B = gauss_seidel_iteration_matrix(A)

    rho = spectral_radius(B)

    # ---------------------------------------------------------------
    # Numerical experiment
    # ---------------------------------------------------------------

    (
        solution,
        converged,
        iterations,
        step_history,
        residual_history,
    ) = gauss_seidel_history(
        A,
        b,
        tol=1e-10,
        maxiter=100,
    )

    # ---------------------------------------------------------------
    # Reference solution
    # ---------------------------------------------------------------

    exact_solution = np.linalg.solve(
        A,
        b,
    )

    solution_error = float(
        np.linalg.norm(
            solution - exact_solution,
            ord=np.inf,
        )
    )

    # ---------------------------------------------------------------
    # Observed convergence factors
    # ---------------------------------------------------------------

    residual_factor = estimate_asymptotic_factor(
        residual_history
    )

    step_factor = estimate_asymptotic_factor(
        step_history
    )

    residual_factors = observed_convergence_factors(
        residual_history
    )

    step_factors = observed_convergence_factors(
        step_history
    )

    results[name] = {
        "rho": rho,
        "converged": converged,
        "iterations": iterations,
        "solution_error": solution_error,
        "step_history": step_history,
        "residual_history": residual_history,
        "residual_factors": residual_factors,
        "step_factors": step_factors,
        "residual_factor": residual_factor,
        "step_factor": step_factor,
    }

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------

    print()
    print("-" * 90)
    print(name)
    print("-" * 90)

    print("A =")
    print(A)

    print("\nb =")
    print(b)

    print("\nGauss-Seidel iteration matrix B_GS =")
    print(B)

    print(
        f"\nSpectral radius:"
        f"\n  rho(B_GS) = {rho:.6f}"
    )

    if rho < 1:
        print("  Theory   = CONVERGENT")
    elif rho == 1:
        print("  Theory   = BORDERLINE")
    else:
        print("  Theory   = DIVERGENT")

    print("\nNumerical result:")

    print(
        f"  Converged       = {converged}"
    )

    print(
        f"  Iterations      = {iterations}"
    )

    print(
        f"  Solution error  = {solution_error:.6e}"
    )

    print("\nObserved convergence factors:")

    print(
        f"  Residual factor = {residual_factor:.6f}"
    )

    print(
        f"  Step factor     = {step_factor:.6f}"
    )

    print("\nComparison:")

    print(
        f"  rho(B_GS)       = {rho:.6f}"
    )

    print(
        f"  residual q      = {residual_factor:.6f}"
    )

    print(
        f"  step q          = {step_factor:.6f}"
    )

    print("\nComputed solution:")
    print(solution)

    print("\nReference solution:")
    print(exact_solution)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 105)
print("SUMMARY")
print("=" * 105)

print(
    f"{'System':<25}"
    f"{'rho(B_GS)':>13}"
    f"{'Residual q':>15}"
    f"{'Step q':>15}"
    f"{'Converged':>14}"
    f"{'Iterations':>14}"
)

print("-" * 105)

for name, result in results.items():

    print(
        f"{name:<25}"
        f"{result['rho']:>13.6f}"
        f"{result['residual_factor']:>15.6f}"
        f"{result['step_factor']:>15.6f}"
        f"{str(result['converged']):>14}"
        f"{result['iterations']:>14}"
    )


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

output_dir = (
    PROJECT_ROOT / "outputs"
)

output_dir.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------------------------
# Plot 1: Residual convergence
# ---------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

for name, result in results.items():

    residuals = np.asarray(
        result["residual_history"]
    )

    plt.semilogy(
        range(len(residuals)),
        residuals,
        marker="o",
        markersize=3,
        label=name,
    )

plt.xlabel("Iteration")

plt.ylabel(
    r"$\|b - Ax^{(k)}\|_\infty$"
)

plt.title(
    "Gauss-Seidel Residual Convergence"
)

plt.grid(
    True,
    which="both",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

residual_path = (
    output_dir
    / "gauss_seidel_residual_convergence.png"
)

plt.savefig(
    residual_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Step error convergence
# ---------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

for name, result in results.items():

    step_errors = np.asarray(
        result["step_history"]
    )

    plt.semilogy(
        range(1, len(step_errors) + 1),
        step_errors,
        marker="o",
        markersize=3,
        label=name,
    )

plt.xlabel("Iteration")

plt.ylabel(
    r"$\|x^{(k+1)} - x^{(k)}\|_\infty$"
)

plt.title(
    "Gauss-Seidel Step Error Convergence"
)

plt.grid(
    True,
    which="both",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

step_path = (
    output_dir
    / "gauss_seidel_step_convergence.png"
)

plt.savefig(
    step_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Spectral radius
# ---------------------------------------------------------------------------

names = list(results.keys())

rho_values = [
    results[name]["rho"]
    for name in names
]

plt.figure(figsize=(9, 6))

plt.bar(
    names,
    rho_values,
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label=r"$\rho(B_{GS})=1$",
)

plt.ylabel(
    r"Spectral Radius $\rho(B_{GS})$"
)

plt.title(
    "Gauss-Seidel Spectral Radius"
)

plt.xticks(
    rotation=20,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

rho_path = (
    output_dir
    / "gauss_seidel_spectral_radius.png"
)

plt.savefig(
    rho_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Theory vs observed convergence factors
# ---------------------------------------------------------------------------

x = np.arange(
    len(names)
)

width = 0.25

rho_values = [
    results[name]["rho"]
    for name in names
]

residual_factors = [
    results[name]["residual_factor"]
    for name in names
]

step_factors = [
    results[name]["step_factor"]
    for name in names
]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    rho_values,
    width,
    label=r"Theoretical $\rho(B_{GS})$",
)

plt.bar(
    x,
    residual_factors,
    width,
    label="Observed residual factor",
)

plt.bar(
    x + width,
    step_factors,
    width,
    label="Observed step factor",
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label=r"Convergence boundary $\rho=1$",
)

plt.xticks(
    x,
    names,
    rotation=20,
    ha="right",
)

plt.ylabel(
    "Convergence Factor"
)

plt.title(
    "Theoretical vs Observed "
    "Gauss-Seidel Convergence Factors"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

factor_path = (
    output_dir
    / "gauss_seidel_convergence_factor_comparison.png"
)

plt.savefig(
    factor_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(
    f"Residual convergence : {residual_path}"
)

print(
    f"Step convergence     : {step_path}"
)

print(
    f"Spectral radius      : {rho_path}"
)

print(
    f"Factor comparison    : {factor_path}"
)

print()
print("Experiment completed.")