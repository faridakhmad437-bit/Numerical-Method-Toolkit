
from __future__ import annotations

import numpy as np
import pytest

from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.robotics.ik import planar_2link_ik


L1 = 1.0
L2 = 0.8


def forward_kinematics(q, l1=L1, l2=L2):
    q1, q2 = q

    return np.array(
        [
            l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
            l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    ("exact_q", "initial"),
    [
        ((0.5, 0.8), (0.55, 0.75)),
        ((-0.7, 1.1), (-0.65, 1.05)),
        ((1.0, -0.9), (0.95, -0.85)),
    ],
)
def test_planar_ik_converges_to_target(exact_q, initial):
    target = forward_kinematics(exact_q)

    options = SolverOptions(
        xtol=1e-10,
        ftol=1e-10,
        maxiter=100,
        store_history=False,
    )

    result = planar_2link_ik(
        target,
        L1,
        L2,
        initial=initial,
        options=options,
    )

    actual_position = forward_kinematics(result.solution)
    position_error = np.linalg.norm(actual_position - target)

    assert result.converged
    assert result.residual is not None
    assert result.residual <= options.ftol
    assert position_error <= 1e-10

    # One initial F evaluation, then J and F per iteration.
    assert result.function_calls == 1 + 2 * result.iterations


def test_planar_ik_rejects_unreachable_target():
    result = planar_2link_ik(
        target=(2.0, 0.0),
        l1=L1,
        l2=L2,
    )

    assert not result.converged
    assert result.iterations == 0
    assert result.function_calls == 0
    assert result.residual is None
