
from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.robotics.ik import planar_2link_ik


L1 = 1.0
L2 = 0.8

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 100
POSITION_TOLERANCE = 1e-10


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


def percentile_95(values: list[float]) -> float:
    """Calculate the 95th percentile of measured runtimes."""
    return float(np.percentile(values, 95))


def run_benchmark() -> None:
    """Benchmark IK accuracy and repeated execution time."""

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

    print("\nPLANAR 2-LINK IK REPEATED BENCHMARK")
    print(f"Link lengths: L1={L1}, L2={L2}")
    print(f"Warm-up runs per case: {WARMUP_RUNS}")
    print(f"Measured runs per case: {MEASUREMENT_RUNS}")
    print("-" * 125)
    print(
        f"{'Case':<12}"
        f"{'Converged':<12}"
        f"{'Iterations':>11}"
        f"{'F/J Calls':>12}"
        f"{'Residual':>15}"
        f"{'Position Error':>17}"
        f"{'Median (ms)':>14}"
        f"{'P95 (ms)':>12}"
    )
    print("-" * 125)

    for name, exact_q, initial in cases:
        exact_q_array = np.asarray(exact_q, dtype=float)
        initial_array = np.asarray(initial, dtype=float)

        target = forward_kinematics(
            exact_q_array,
            L1,
            L2,
        )

        def solve():
            return planar_2link_ik(
                target,
                L1,
                L2,
                initial=initial_array,
                options=options,
            )

        # Warm-up is intentionally excluded from measured runtimes.
        for _ in range(WARMUP_RUNS):
            warmup_result = solve()

            if not warmup_result.converged:
                raise RuntimeError(
                    f"{name}: solver failed during warm-up."
                )

        runtimes: list[float] = []
        result = None

        for _ in range(MEASUREMENT_RUNS):
            start = perf_counter()
            current_result = solve()
            elapsed = perf_counter() - start

            # Do not silently benchmark an incorrect solution.
            if not current_result.converged:
                raise RuntimeError(
                    f"{name}: solver did not converge."
                )

            actual_position = forward_kinematics(
                np.asarray(current_result.solution, dtype=float),
                L1,
                L2,
            )

            position_error = float(
                np.linalg.norm(actual_position - target, ord=2)
            )

            if (
                current_result.residual is None
                or current_result.residual > options.ftol
                or position_error > POSITION_TOLERANCE
            ):
                raise RuntimeError(
                    f"{name}: numerical accuracy check failed. "
                    f"Residual={current_result.residual}, "
                    f"position_error={position_error:.3e}"
                )

            runtimes.append(elapsed)
            result = current_result

        assert result is not None

        final_position = forward_kinematics(
            np.asarray(result.solution, dtype=float),
            L1,
            L2,
        )
        final_position_error = float(
            np.linalg.norm(final_position - target, ord=2)
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
            f"{final_position_error:>17.3e}"
            f"{median(runtimes) * 1000:>14.6f}"
            f"{percentile_95(runtimes) * 1000:>12.6f}"
        )

    print("-" * 125)
    print("F/J Calls counts both F and Jacobian J evaluations.")
    print("Median and P95 are measured runtimes in milliseconds.")
    print("Warm-up executions are excluded from timing statistics.")
    print("Every measured result must pass convergence and accuracy checks.")


if __name__ == "__main__":
    run_benchmark()
