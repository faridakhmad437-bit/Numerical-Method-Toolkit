import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.robotics.ik import (
    newton_system,
    planar_2link_ik,
)


# ============================================================================
# Helpers
# ============================================================================

def linear_F(x):
    return np.array([
        x[0] - 2.0,
        x[1] - 3.0,
    ])


def linear_J(x):
    return np.eye(2)


def forward_2link(q, l1, l2):
    q1, q2 = q

    return np.array([
        l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
        l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
    ])


# ============================================================================
# Newton System — Basic Correctness
# ============================================================================

def test_newton_system_solves_linear_system():
    result = newton_system(
        linear_F,
        linear_J,
        x0=np.array([0.0, 0.0]),
    )

    assert isinstance(result, SolverResult)
    assert result.converged
    assert np.allclose(
        result.solution,
        np.array([2.0, 3.0]),
        atol=1e-10,
    )
    assert result.iterations >= 1


def test_newton_system_accepts_list_initial_guess():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
    )

    assert result.converged
    assert np.allclose(
        result.solution,
        [2.0, 3.0],
    )


def test_newton_system_does_not_mutate_initial_guess():
    x0 = np.array([0.0, 0.0])
    original = x0.copy()

    newton_system(
        linear_F,
        linear_J,
        x0,
    )

    assert np.array_equal(x0, original)


def test_newton_system_converges_at_initial_guess():
    x0 = np.array([2.0, 3.0])

    result = newton_system(
        linear_F,
        linear_J,
        x0,
    )

    assert result.converged
    assert result.iterations == 0
    assert result.residual is not None
    assert result.residual <= 1e-10


# ============================================================================
# Newton System — Result Diagnostics
# ============================================================================

def test_newton_system_reports_small_residual():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
    )

    assert result.converged
    assert result.residual is not None
    assert result.residual <= 1e-10


def test_newton_system_reports_step_error():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
    )

    assert result.converged
    assert result.step_error is not None
    assert result.step_error >= 0.0


def test_newton_system_stores_history():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
    )

    assert result.history is not None
    assert len(result.history) == result.iterations + 1

    first = result.history[0]

    assert first["iteration"] == 0
    assert "solution" in first
    assert "residual" in first
    assert "step_error" in first


def test_newton_system_can_disable_history():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
        options=SolverOptions(
            store_history=False,
        ),
    )

    assert result.converged
    assert result.history is None


def test_newton_system_respects_maxiter():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
        options=SolverOptions(
            maxiter=1,
        ),
    )

    assert result.iterations == 1


def test_newton_system_counts_function_evaluations():
    result = newton_system(
        linear_F,
        linear_J,
        x0=[0.0, 0.0],
    )

    assert result.function_calls > 0


# ============================================================================
# Newton System — Dimension Validation
# ============================================================================

def test_newton_system_rejects_non_vector_F():
    def F(x):
        return np.array([
            [x[0]],
            [x[1]],
        ])

    with pytest.raises(InvalidInputError):
        newton_system(
            F,
            linear_J,
            [0.0, 0.0],
        )


def test_newton_system_rejects_wrong_J_shape():
    def J(x):
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

    with pytest.raises(InvalidInputError):
        newton_system(
            linear_F,
            J,
            [0.0, 0.0],
        )


def test_newton_system_rejects_wrong_F_dimension():
    def F(x):
        return np.array([
            x[0] - 2.0,
        ])

    with pytest.raises(InvalidInputError):
        newton_system(
            F,
            linear_J,
            [0.0, 0.0],
        )


def test_newton_system_rejects_wrong_x0_shape():
    with pytest.raises(InvalidInputError):
        newton_system(
            linear_F,
            linear_J,
            [0.0, 0.0, 0.0],
        )


# ============================================================================
# Newton System — Callable Validation
# ============================================================================

def test_newton_system_rejects_non_callable_F():
    with pytest.raises(InvalidInputError):
        newton_system(
            "not callable",
            linear_J,
            [0.0, 0.0],
        )


def test_newton_system_rejects_non_callable_J():
    with pytest.raises(InvalidInputError):
        newton_system(
            linear_F,
            "not callable",
            [0.0, 0.0],
        )


def test_newton_system_rejects_invalid_options():
    with pytest.raises(InvalidInputError):
        newton_system(
            linear_F,
            linear_J,
            [0.0, 0.0],
            options="invalid",
        )


# ============================================================================
# Newton System — Jacobian Failure
# ============================================================================

def test_newton_system_rejects_singular_jacobian():
    def F(x):
        return np.array([
            x[0] - 1.0,
            x[1] - 1.0,
        ])

    def J(x):
        return np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

    with pytest.raises(InvalidInputError):
        newton_system(
            F,
            J,
            [0.0, 0.0],
        )


# ============================================================================
# Newton System — Parameter Validation
# ============================================================================

@pytest.mark.parametrize("ftol", [0.0, -1.0])
def test_newton_system_rejects_non_positive_ftol(ftol):
    with pytest.raises(ValueError):
        SolverOptions(
            ftol=ftol,
        )


@pytest.mark.parametrize("xtol", [0.0, -1.0])
def test_newton_system_rejects_non_positive_xtol(xtol):
    with pytest.raises(ValueError):
        SolverOptions(
            xtol=xtol,
        )


@pytest.mark.parametrize("maxiter", [0, -1])
def test_newton_system_rejects_non_positive_maxiter(maxiter):
    with pytest.raises(ValueError):
        SolverOptions(
            maxiter=maxiter,
        )


# ============================================================================
# Newton System — Non-finite Values
# ============================================================================

@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_newton_system_rejects_non_finite_x0(bad_value):
    x0 = np.array([
        bad_value,
        0.0,
    ])

    with pytest.raises(NonFiniteValueError):
        newton_system(
            linear_F,
            linear_J,
            x0,
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_newton_system_rejects_non_finite_F(bad_value):
    def F(x):
        return np.array([
            bad_value,
            1.0,
        ])

    with pytest.raises(NonFiniteValueError):
        newton_system(
            F,
            linear_J,
            [0.0, 0.0],
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_newton_system_rejects_non_finite_J(bad_value):
    def J(x):
        return np.array([
            [bad_value, 0.0],
            [0.0, 1.0],
        ])

    with pytest.raises(NonFiniteValueError):
        newton_system(
            linear_F,
            J,
            [0.0, 0.0],
        )


# ============================================================================
# Newton System — Nonconvergence
# ============================================================================

def test_newton_system_reports_nonconvergence():
    def F(x):
        return np.array([
            1.0,
            1.0,
        ])

    def J(x):
        return np.eye(2)

    result = newton_system(
        F=F,
        J=J,
        x0=np.array([0.0, 0.0]),
        options=SolverOptions(
            ftol=1e-12,
            xtol=1e-12,
            maxiter=10,
        ),
    )

    assert result.converged is False
    assert result.iterations == 10
    assert np.all(np.isfinite(result.solution))


# ============================================================================
# Planar 2-Link IK — Basic Correctness
# ============================================================================

def test_planar_2link_ik_reaches_known_target():
    l1 = 1.0
    l2 = 1.0

    target = forward_2link(
        np.array([
            np.pi / 4,
            np.pi / 4,
        ]),
        l1,
        l2,
    )

    result = planar_2link_ik(
        target,
        l1,
        l2,
        initial=(0.5, 0.5),
    )

    assert isinstance(result, SolverResult)
    assert result.converged

    q = result.solution

    reconstructed = forward_2link(
        q,
        l1,
        l2,
    )

    assert np.allclose(
        reconstructed,
        target,
        atol=1e-8,
    )


def test_planar_2link_ik_solution_has_small_position_error():
    l1 = 1.0
    l2 = 0.75

    target = np.array([
        1.0,
        0.5,
    ])

    result = planar_2link_ik(
        target,
        l1,
        l2,
        initial=(0.2, 0.4),
    )

    assert result.converged

    q = result.solution

    reconstructed = forward_2link(
        q,
        l1,
        l2,
    )

    error = np.linalg.norm(
        reconstructed - target,
        2,
    )

    assert error < 1e-8


def test_planar_2link_ik_reports_residual():
    result = planar_2link_ik(
        [1.0, 0.5],
        1.0,
        0.75,
        initial=(0.2, 0.4),
    )

    assert result.converged
    assert result.residual is not None
    assert result.residual < 1e-8


def test_planar_2link_ik_supports_solver_options():
    result = planar_2link_ik(
        [1.0, 0.5],
        1.0,
        0.75,
        initial=(0.2, 0.4),
        options=SolverOptions(
            xtol=1e-10,
            ftol=1e-10,
            maxiter=50,
        ),
    )

    assert result.converged


def test_planar_2link_ik_stores_history():
    result = planar_2link_ik(
        [1.0, 0.5],
        1.0,
        0.75,
        initial=(0.2, 0.4),
    )

    assert result.history is not None
    assert len(result.history) == result.iterations + 1


def test_planar_2link_ik_can_disable_history():
    result = planar_2link_ik(
        [1.0, 0.5],
        1.0,
        0.75,
        initial=(0.2, 0.4),
        options=SolverOptions(
            store_history=False,
        ),
    )

    assert result.converged
    assert result.history is None


# ============================================================================
# Planar 2-Link IK — Link Validation
# ============================================================================

@pytest.mark.parametrize(
    "l1,l2",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_planar_2link_ik_rejects_non_positive_link_lengths(l1, l2):
    with pytest.raises(InvalidInputError):
        planar_2link_ik(
            [1.0, 0.0],
            l1,
            l2,
        )


# ============================================================================
# Planar 2-Link IK — Reachability
# ============================================================================

def test_planar_2link_ik_unreachable_target_does_not_claim_success():
    result = planar_2link_ik(
        [3.0, 0.0],
        1.0,
        1.0,
        initial=(0.0, 0.0),
    )

    assert isinstance(result, SolverResult)
    assert result.converged is False
    assert result.iterations == 0
    assert result.function_calls == 0
    assert result.residual is None

    assert np.allclose(
        result.solution,
        [0.0, 0.0],
    )


# ============================================================================
# Planar 2-Link IK — Non-finite Inputs
# ============================================================================

@pytest.mark.parametrize(
    "target",
    [
        [np.nan, 0.0],
        [np.inf, 0.0],
        [-np.inf, 0.0],
    ],
)
def test_planar_2link_ik_rejects_non_finite_target(target):
    with pytest.raises(NonFiniteValueError):
        planar_2link_ik(
            target,
            1.0,
            1.0,
        )


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf],
)
def test_planar_2link_ik_rejects_non_finite_link_length(value):
    with pytest.raises(NonFiniteValueError):
        planar_2link_ik(
            [1.0, 0.0],
            value,
            1.0,
        )


# ============================================================================
# Planar 2-Link IK — Target Shape
# ============================================================================

def test_planar_2link_ik_rejects_wrong_target_shape():
    with pytest.raises(InvalidInputError):
        planar_2link_ik(
            [1.0, 2.0, 3.0],
            1.0,
            1.0,
        )


def test_planar_2link_ik_rejects_scalar_target():
    with pytest.raises(InvalidInputError):
        planar_2link_ik(
            1.0,
            1.0,
            1.0,
        )