"""Malformed array inputs must raise InvalidInputError, never a raw NumPy error."""

import numpy as np
import pytest

from numerical_toolkit.core.exceptions import InvalidInputError
from numerical_toolkit.interpolation import lagrange, newton_interpolation
from numerical_toolkit.linear import gauss_seidel, gaussian_elimination, jacobi
from numerical_toolkit.robotics import newton_system, planar_2link_ik

RAGGED = [[1.0, 2.0], [3.0]]
TEXT = [["a", "b"], ["c", "d"]]


@pytest.mark.parametrize("solver", [gaussian_elimination, jacobi, gauss_seidel])
@pytest.mark.parametrize("A", [RAGGED, TEXT])
def test_linear_solvers_reject_malformed_matrix(solver, A):
    with pytest.raises(InvalidInputError):
        solver(A, [1.0, 2.0])


@pytest.mark.parametrize("solver", [gaussian_elimination, jacobi, gauss_seidel])
def test_linear_solvers_reject_text_rhs(solver):
    with pytest.raises(InvalidInputError):
        solver([[4.0, 1.0], [1.0, 3.0]], ["a", "b"])


@pytest.mark.parametrize("solver", [jacobi, gauss_seidel])
@pytest.mark.parametrize("x0", [[[1.0], [2.0, 3.0]], ["a", "b"]])
def test_iterative_solvers_reject_malformed_initial_guess(solver, x0):
    with pytest.raises(InvalidInputError):
        solver([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0], x0)


@pytest.mark.parametrize("method", [lagrange, newton_interpolation])
@pytest.mark.parametrize(
    "x, y, xq",
    [
        (["a", "b"], [1.0, 2.0], 0.5),
        ([0.0, 1.0], ["a", "b"], 0.5),
        ([[0.0], [1.0, 2.0]], [1.0, 2.0], 0.5),
        ([0.0, 1.0], [1.0, 2.0], "a"),
    ],
)
def test_interpolation_rejects_malformed_inputs(method, x, y, xq):
    with pytest.raises(InvalidInputError):
        method(x, y, xq)


@pytest.mark.parametrize("method", [lagrange, newton_interpolation])
@pytest.mark.parametrize(
    "x, y, xq",
    [
        ([False, True], [1.0, 2.0], 0.5),
        ([0.0, 1.0], [False, True], 0.5),
        ([0.0, 1.0], [1.0, 2.0], True),
    ],
)
def test_interpolation_rejects_boolean_inputs(method, x, y, xq):
    with pytest.raises(InvalidInputError):
        method(x, y, xq)


@pytest.mark.parametrize("target", [["a", "b"], [True, False]])
def test_planar_ik_rejects_malformed_target(target):
    with pytest.raises(InvalidInputError):
        planar_2link_ik(target, 1.0, 1.0)


@pytest.mark.parametrize("initial", [["a", "b"], [True, False]])
def test_planar_ik_rejects_malformed_initial_guess(initial):
    with pytest.raises(InvalidInputError):
        planar_2link_ik([1.0, 1.0], 1.0, 1.0, initial)


def test_newton_system_rejects_non_numeric_function_value():
    with pytest.raises(InvalidInputError):
        newton_system(lambda x: ["a"], lambda x: [[1.0]], [1.0])


def test_newton_system_rejects_non_numeric_jacobian():
    with pytest.raises(InvalidInputError):
        newton_system(lambda x: x - 2.0, lambda x: [["a"]], [1.0])


def test_newton_system_rejects_boolean_initial_guess():
    with pytest.raises(InvalidInputError):
        newton_system(lambda x: x - 2.0, lambda x: np.eye(1), [True])
