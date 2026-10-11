from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Make the project root importable when this file is run directly.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Jacobi theory helpers
# ---------------------------------------------------------------------------

def jacobi_iteration_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct the Jacobi iteration matrix:

        B_J = -D^{-1}(L + U)

    where:

        A = D + L + U
    """
    A = np.asarray(A, dtype=float)

    D = np.diag(np.diag(A))
    R = A - D

    return -np.linalg.inv(D) @ R


def spectral_radius(B: np.ndarray) -> float:
    """
    Compute the spectral radius:

        rho(B) = max |lambda_i|

    where lambda_i are the eigenvalues of B.
    """
    eigenvalues = np.linalg.eigvals(B)

    return float(np.max(np.abs(eigenvalues)))


# ---------------------------------------------------------------------------
# Jacobi experiment with history
# ---------------------------------------------------------------------------

def jacobi_history(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> tuple[
    np.ndarray,
    bool,
    int,
    list[float],
    list[float],
]:
    """
    Run Jacobi while explicitly storing convergence history.

    Returns
    -------
    x :
        Final solution estimate.

    converged :
        Whether the stopping criterion was satisfied.

    iterations :
        Number of iterations performed.

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

    D = np.diag(A)
    R = A - np.diagflat(D)

    step_history: list[float] = []

    residual_history: list[float] = [
        float(np.linalg.norm(b - A @ x, ord=np.inf))
    ]

    for iteration in range(1, maxiter + 1):

        x_new = (b - R @ x) / D

        step_error = float(
            np.linalg.norm(x_new - x, ord=np.inf)
        )

        residual = float(
            np.linalg.norm(b - A @ x_new, ord=np.inf)
        )

        step_history.append(step_error)
        residual_history.append(residual)

        if step_error <= tol:
            return (
                x_new,
                True,
                iteration,
                step_history,
                residual_history,
            )

        x = x_new

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
    Estimate convergence factors from consecutive history values.

    For a quantity e_k:

        q_k = e_(k+1) / e_k

    For a convergent stationary iteration, the asymptotic factor
    is related to the dominant eigenvalue magnitude.

    The returned array contains finite, positive ratios only.
    """

    values = np.asarray(history, dtype=float)

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
    Estimate the asymptotic convergence factor using the median
    of the final valid convergence ratios.

    Median is used to reduce sensitivity to small floating-point
    fluctuations.
    """

    factors = observed_convergence_factors(history)

    if len(factors) == 0:
        return float("nan")

    tail = factors[-tail_size:]

    return float(np.median(tail))


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
        "b": np.array([12.0, 13.0, 13.0]),
    },

    "B: Weakly Dominant": {
        "A": np.array(
            [
                [4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0],
                [1.0, 1.0, 4.0],
            ]
        ),
        "b": np.array([6.0, 6.0, 6.0]),
    },

    "C: Slow Convergence": {
        "A": np.array(
            [
                [4.0, 3.0],
                [3.0, 4.0],
            ]
        ),
        "b": np.array([7.0, 7.0]),
    },

    "D: Divergent": {
        "A": np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        ),
        "b": np.array([3.0, 3.0]),
    },
}


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

results = {}

print("=" * 86)
print("JACOBI CONVERGENCE STUDY")
print("=" * 86)

for name, system in systems.items():

    A = system["A"]
    b = system["b"]

    # ---------------------------------------------------------------
    # Theoretical convergence analysis
    # ---------------------------------------------------------------

    B = jacobi_iteration_matrix(A)
    rho = spectral_radius(B)

    # ---------------------------------------------------------------
    # Numerical Jacobi experiment
    # ---------------------------------------------------------------

    (
        solution,
        converged,
        iterations,
        step_history,
        residual_history,
    ) = jacobi_history(
        A,
        b,
        tol=1e-10,
        maxiter=100,
    )

    # ---------------------------------------------------------------
    # Reference solution
    # ---------------------------------------------------------------

    exact_solution = np.linalg.solve(A, b)

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
    print("-" * 86)
    print(name)
    print("-" * 86)

    print("A =")
    print(A)

    print("\nb =")
    print(b)

    print("\nJacobi iteration matrix B_J =")
    print(B)

    print("\nSpectral radius:")
    print(f"  rho(B_J) = {rho:.6f}")

    if rho < 1:
        print("  Theory   = CONVERGENT")
    elif rho == 1:
        print("  Theory   = BORDERLINE")
    else:
        print("  Theory   = DIVERGENT")

    print("\nNumerical result:")
    print(f"  Converged       = {converged}")
    print(f"  Iterations      = {iterations}")
    print(f"  Solution error  = {solution_error:.6e}")

    print("\nObserved convergence factors:")
    print(f"  Residual factor = {residual_factor:.6f}")
    print(f"  Step factor     = {step_factor:.6f}")

    print("\nComparison:")
    print(f"  rho(B_J)        = {rho:.6f}")
    print(f"  residual q      = {residual_factor:.6f}")
    print(f"  step q           = {step_factor:.6f}")

    print("\nComputed solution:")
    print(solution)

    print("\nReference solution:")
    print(exact_solution)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

header = (
    f"{'System':<25}"
    f"{'rho(B_J)':>12}"
    f"{'Residual q':>15}"
    f"{'Step q':>15}"
    f"{'Converged':>14}"
    f"{'Iterations':>14}"
)

print(header)
print("-" * 100)

for name, result in results.items():

    print(
        f"{name:<25}"
        f"{result['rho']:>12.6f}"
        f"{result['residual_factor']:>15.6f}"
        f"{result['step_factor']:>15.6f}"
        f"{str(result['converged']):>14}"
        f"{result['iterations']:>14}"
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
plt.ylabel(r"$\|b - Ax^{(k)}\|_\infty$")
plt.title("Jacobi Residual Convergence")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

residual_path = (
    output_dir / "jacobi_residual_convergence.png"
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
plt.title("Jacobi Step Error Convergence")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

step_path = (
    output_dir / "jacobi_step_convergence.png"
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
    label=r"$\rho(B_J)=1$",
)

plt.ylabel(
    r"Spectral Radius $\rho(B_J)$"
)

plt.title("Jacobi Spectral Radius")

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
    output_dir / "jacobi_spectral_radius.png"
)

plt.savefig(
    rho_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Observed convergence factor vs spectral radius
# ---------------------------------------------------------------------------

x = np.arange(len(names))
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
    label=r"Theoretical $\rho(B_J)$",
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

plt.ylabel("Convergence Factor")

plt.title(
    "Theoretical vs Observed Jacobi Convergence Factors"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

factor_path = (
    output_dir
    / "jacobi_convergence_factor_comparison.png"
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
print("=" * 86)
print("OUTPUT FILES")
print("=" * 86)

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