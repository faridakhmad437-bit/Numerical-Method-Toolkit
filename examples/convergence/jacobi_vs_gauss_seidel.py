from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Iteration matrices
# ---------------------------------------------------------------------------

def jacobi_iteration_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct the Jacobi iteration matrix:

        B_J = -D^(-1)(L + U)

    where:

        A = D + L + U
    """
    A = np.asarray(A, dtype=float)

    D = np.diag(np.diag(A))
    R = A - D

    return -np.linalg.inv(D) @ R


def gauss_seidel_iteration_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct the Gauss-Seidel iteration matrix:

        B_GS = -(D + L)^(-1) U

    where:

        A = D + L + U
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
# Solver experiment runners
# ---------------------------------------------------------------------------

def run_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
):
    """
    Run Jacobi and store step/residual history.
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

    D = np.diag(A)
    R = A - np.diagflat(D)

    for iteration in range(1, maxiter + 1):

        x_new = (b - R @ x) / D

        step_error = float(
            np.linalg.norm(
                x_new - x,
                ord=np.inf,
            )
        )

        residual = float(
            np.linalg.norm(
                b - A @ x_new,
                ord=np.inf,
            )
        )

        step_history.append(step_error)
        residual_history.append(residual)

        if step_error <= tol:
            return {
                "solution": x_new,
                "converged": True,
                "iterations": iteration,
                "step_history": step_history,
                "residual_history": residual_history,
            }

        x = x_new

    return {
        "solution": x,
        "converged": False,
        "iterations": maxiter,
        "step_history": step_history,
        "residual_history": residual_history,
    }


def run_gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
):
    """
    Run Gauss-Seidel and store step/residual history.
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
            return {
                "solution": x,
                "converged": True,
                "iterations": iteration,
                "step_history": step_history,
                "residual_history": residual_history,
            }

    return {
        "solution": x,
        "converged": False,
        "iterations": maxiter,
        "step_history": step_history,
        "residual_history": residual_history,
    }


# ---------------------------------------------------------------------------
# Convergence factor
# ---------------------------------------------------------------------------

def observed_convergence_factors(
    history: list[float],
) -> np.ndarray:
    """
    Compute consecutive ratios:

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
    Estimate the asymptotic convergence factor from
    the median of the final valid ratios.
    """

    factors = observed_convergence_factors(
        history
    )

    if len(factors) == 0:
        return float("nan")

    return float(
        np.median(
            factors[-tail_size:]
        )
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
# Experiment configuration
# ---------------------------------------------------------------------------

TOL = 1e-10
MAXITER = 100


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

results = {}

print("=" * 100)
print("JACOBI VS GAUSS-SEIDEL")
print("=" * 100)

for name, system in systems.items():

    A = system["A"]
    b = system["b"]

    # ---------------------------------------------------------------
    # Theoretical iteration matrices
    # ---------------------------------------------------------------

    B_jacobi = jacobi_iteration_matrix(A)
    B_gs = gauss_seidel_iteration_matrix(A)

    rho_jacobi = spectral_radius(
        B_jacobi
    )

    rho_gs = spectral_radius(
        B_gs
    )

    # ---------------------------------------------------------------
    # Numerical experiments
    # ---------------------------------------------------------------

    jacobi_result = run_jacobi(
        A,
        b,
        tol=TOL,
        maxiter=MAXITER,
    )

    gs_result = run_gauss_seidel(
        A,
        b,
        tol=TOL,
        maxiter=MAXITER,
    )

    # ---------------------------------------------------------------
    # Reference solution
    # ---------------------------------------------------------------

    exact_solution = np.linalg.solve(
        A,
        b,
    )

    jacobi_error = float(
        np.linalg.norm(
            jacobi_result["solution"]
            - exact_solution,
            ord=np.inf,
        )
    )

    gs_error = float(
        np.linalg.norm(
            gs_result["solution"]
            - exact_solution,
            ord=np.inf,
        )
    )

    # ---------------------------------------------------------------
    # Observed convergence factors
    # ---------------------------------------------------------------

    jacobi_residual_factor = (
        estimate_asymptotic_factor(
            jacobi_result["residual_history"]
        )
    )

    jacobi_step_factor = (
        estimate_asymptotic_factor(
            jacobi_result["step_history"]
        )
    )

    gs_residual_factor = (
        estimate_asymptotic_factor(
            gs_result["residual_history"]
        )
    )

    gs_step_factor = (
        estimate_asymptotic_factor(
            gs_result["step_history"]
        )
    )

    # ---------------------------------------------------------------
    # Iteration reduction
    # ---------------------------------------------------------------

    jacobi_iterations = (
        jacobi_result["iterations"]
    )

    gs_iterations = (
        gs_result["iterations"]
    )

    if gs_iterations > 0:
        iteration_ratio = (
            jacobi_iterations
            / gs_iterations
        )
    else:
        iteration_ratio = float("nan")

    # ---------------------------------------------------------------
    # Store
    # ---------------------------------------------------------------

    results[name] = {
        "rho_jacobi": rho_jacobi,
        "rho_gs": rho_gs,

        "jacobi": jacobi_result,
        "gauss_seidel": gs_result,

        "jacobi_error": jacobi_error,
        "gs_error": gs_error,

        "jacobi_residual_factor": (
            jacobi_residual_factor
        ),

        "jacobi_step_factor": (
            jacobi_step_factor
        ),

        "gs_residual_factor": (
            gs_residual_factor
        ),

        "gs_step_factor": (
            gs_step_factor
        ),

        "iteration_ratio": iteration_ratio,
    }

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------

    print()
    print("-" * 100)
    print(name)
    print("-" * 100)

    print("\nJacobi:")
    print(
        f"  rho(B_J)       = {rho_jacobi:.6f}"
    )
    print(
        f"  converged       = "
        f"{jacobi_result['converged']}"
    )
    print(
        f"  iterations      = "
        f"{jacobi_iterations}"
    )
    print(
        f"  residual q      = "
        f"{jacobi_residual_factor:.6f}"
    )
    print(
        f"  step q          = "
        f"{jacobi_step_factor:.6f}"
    )
    print(
        f"  solution error  = "
        f"{jacobi_error:.6e}"
    )

    print("\nGauss-Seidel:")
    print(
        f"  rho(B_GS)       = {rho_gs:.6f}"
    )
    print(
        f"  converged       = "
        f"{gs_result['converged']}"
    )
    print(
        f"  iterations      = "
        f"{gs_iterations}"
    )
    print(
        f"  residual q      = "
        f"{gs_residual_factor:.6f}"
    )
    print(
        f"  step q          = "
        f"{gs_step_factor:.6f}"
    )
    print(
        f"  solution error  = "
        f"{gs_error:.6e}"
    )

    print("\nComparison:")

    print(
        f"  spectral radius ratio "
        f"(Jacobi / GS) = "
        f"{rho_jacobi / rho_gs:.6f}"
    )

    print(
        f"  iteration ratio "
        f"(Jacobi / GS) = "
        f"{iteration_ratio:.6f}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    f"{'System':<25}"
    f"{'rho J':>10}"
    f"{'rho GS':>10}"
    f"{'Iter J':>10}"
    f"{'Iter GS':>10}"
    f"{'J/GS Iter':>12}"
    f"{'J Conv.':>12}"
    f"{'GS Conv.':>12}"
)

print("-" * 120)

for name, result in results.items():

    jacobi_result = result["jacobi"]
    gs_result = result["gauss_seidel"]

    print(
        f"{name:<25}"
        f"{result['rho_jacobi']:>10.6f}"
        f"{result['rho_gs']:>10.6f}"
        f"{jacobi_result['iterations']:>10}"
        f"{gs_result['iterations']:>10}"
        f"{result['iteration_ratio']:>12.3f}"
        f"{str(jacobi_result['converged']):>12}"
        f"{str(gs_result['converged']):>12}"
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
# Plot 1: Spectral radius comparison
# ---------------------------------------------------------------------------

names = list(results.keys())

x = np.arange(
    len(names)
)

width = 0.35

rho_j = [
    results[name]["rho_jacobi"]
    for name in names
]

rho_gs = [
    results[name]["rho_gs"]
    for name in names
]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    rho_j,
    width,
    label=r"Jacobi $\rho(B_J)$",
)

plt.bar(
    x + width / 2,
    rho_gs,
    width,
    label=r"Gauss-Seidel $\rho(B_{GS})$",
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
    "Spectral Radius"
)

plt.title(
    "Jacobi vs Gauss-Seidel Spectral Radius"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

spectral_path = (
    output_dir
    / "jacobi_vs_gauss_seidel_spectral_radius.png"
)

plt.savefig(
    spectral_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Iteration count comparison
# ---------------------------------------------------------------------------

iterations_j = [
    results[name]["jacobi"]["iterations"]
    for name in names
]

iterations_gs = [
    results[name]["gauss_seidel"]["iterations"]
    for name in names
]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    iterations_j,
    width,
    label="Jacobi",
)

plt.bar(
    x + width / 2,
    iterations_gs,
    width,
    label="Gauss-Seidel",
)

plt.xticks(
    x,
    names,
    rotation=20,
    ha="right",
)

plt.ylabel(
    "Iterations"
)

plt.title(
    "Jacobi vs Gauss-Seidel Iteration Count"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

iteration_path = (
    output_dir
    / "jacobi_vs_gauss_seidel_iterations.png"
)

plt.savefig(
    iteration_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Residual convergence
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)

for ax, name in zip(
    axes.flat,
    names,
):

    result = results[name]

    jacobi_residuals = np.asarray(
        result["jacobi"]["residual_history"]
    )

    gs_residuals = np.asarray(
        result["gauss_seidel"]["residual_history"]
    )

    ax.semilogy(
        range(len(jacobi_residuals)),
        jacobi_residuals,
        marker="o",
        markersize=3,
        label="Jacobi",
    )

    ax.semilogy(
        range(len(gs_residuals)),
        gs_residuals,
        marker="s",
        markersize=3,
        label="Gauss-Seidel",
    )

    ax.set_title(name)

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        r"$\|b-Ax^{(k)}\|_\infty$"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

fig.suptitle(
    "Jacobi vs Gauss-Seidel Residual Convergence",
    fontsize=15,
)

fig.tight_layout()

residual_path = (
    output_dir
    / "jacobi_vs_gauss_seidel_residual.png"
)

fig.savefig(
    residual_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Step error convergence
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)

for ax, name in zip(
    axes.flat,
    names,
):

    result = results[name]

    jacobi_steps = np.asarray(
        result["jacobi"]["step_history"]
    )

    gs_steps = np.asarray(
        result["gauss_seidel"]["step_history"]
    )

    ax.semilogy(
        range(1, len(jacobi_steps) + 1),
        jacobi_steps,
        marker="o",
        markersize=3,
        label="Jacobi",
    )

    ax.semilogy(
        range(1, len(gs_steps) + 1),
        gs_steps,
        marker="s",
        markersize=3,
        label="Gauss-Seidel",
    )

    ax.set_title(name)

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        r"$\|x^{(k+1)}-x^{(k)}\|_\infty$"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

fig.suptitle(
    "Jacobi vs Gauss-Seidel Step Error",
    fontsize=15,
)

fig.tight_layout()

step_path = (
    output_dir
    / "jacobi_vs_gauss_seidel_step_error.png"
)

fig.savefig(
    step_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print(
    f"Spectral radius : {spectral_path}"
)

print(
    f"Iterations      : {iteration_path}"
)

print(
    f"Residual        : {residual_path}"
)

print(
    f"Step error      : {step_path}"
)

print()
print("Experiment completed.")