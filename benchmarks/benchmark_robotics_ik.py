
from __future__ import annotations

from time import perf_counter

import numpy as np

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.robotics.ik import planar_2link_ik


L1 = 1.0
L2 = 0.8


def forward_kinematics(
    q: np.ndarray,
    l1: float,
    l2: float,
) -> np.ndarray:
    """Calculate the end-effector position of a planar 2-link robot."""
    q1, q2 = q

    return np.array(
        [
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
            l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
        ],
        dtype=float,
    )


def run_benchmark() -> None:
    """Benchmark IK convergence and end-effector position accuracy."""

    # Exact joint angles generate reachable target positions.
    cases = [
        ("Target A", (0.5, 0.8), (0.55, 0.75)),
        ("Target B", (-0.7, 1.1), (-0.65, 1.05)),
        ("Target C", (1.0, -0.9), (0.95, -0.85)),
    ]

    options = SolverOptions(
        xtol=1e-10,
        ftol=1e-10,
        maxiter=100,
        store_history=False,
    )

    print("\nPLANAR 2-LINK INVERSE KINEMATICS BENCHMARK")
    print(f"Link lengths: L1={L1}, L2={L2}")
    print("-" * 105)
    print(
        f"{'Case':<12}"
        f"{'Converged':<12}"
        f"{'Iterations':>11}"
        f"{'F/J Calls':>12}"
        f"{'Residual':>15}"
        f"{'Position Error':>17}"
        f"{'Runtime (s)':>14}"
    )
    print("-" * 105)

    for name, exact_q, initial in cases:
        exact_q_array = np.asarray(exact_q, dtype=float)
        initial_array = np.asarray(initial, dtype=float)

        target = forward_kinematics(
            exact_q_array,
            L1,
            L2,
        )

        start = perf_counter()

        result = planar_2link_ik(
            target,
            L1,
            L2,
            initial=initial_array,
            options=options,
        )

        runtime = perf_counter() - start

        actual_position = forward_kinematics(
            np.asarray(result.solution, dtype=float),
            L1,
            L2,
        )

        position_error = float(
            np.linalg.norm(actual_position - target, ord=2)
        )

        residual_text = (
            f"{result.residual:.3e}"
            if result.residual is not None
            else "N/A"
        )

        print(
            f"{name:<12}"
            f"{str(result.converged):<12}"
            f"{result.iterations:>11}"
            f"{result.function_calls:>12}"
            f"{residual_text:>15}"
            f"{position_error:>17.3e}"
            f"{runtime:>14.6e}"
        )

    print("-" * 105)
    print("F/J Calls includes evaluations of both F and Jacobian J.")
    print("Position Error is the Euclidean distance to the target.")


if __name__ == "__main__":
    run_benchmark()
