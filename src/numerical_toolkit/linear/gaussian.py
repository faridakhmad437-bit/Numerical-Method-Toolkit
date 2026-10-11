
import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)


def gaussian_elimination(A, b):
    """Solve a square linear system using Gaussian elimination.

    Partial pivoting is used to improve numerical stability.
    """
    # Reject boolean inputs before NumPy converts them to floats.
    raw_A = np.asarray(A)
    raw_b = np.asarray(b)

    if raw_A.dtype.kind == "b" or raw_b.dtype.kind == "b":
        raise InvalidInputError("A and b must contain numeric, non-boolean values.")

    try:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError("A and b must contain valid numeric values.") from exc

    if (
        A.ndim != 2
        or A.shape[0] != A.shape[1]
        or b.shape != (A.shape[0],)
    ):
        raise InvalidInputError("A must be square and b must match its size.")

    if A.shape[0] == 0:
        raise InvalidInputError("A and b must define a non-empty linear system.")

    if not np.all(np.isfinite(A)):
        raise NonFiniteValueError("A must contain only finite values.")

    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError("b must contain only finite values.")

    # Scale the coefficient matrix to avoid using an absolute pivot threshold.
    matrix_scale = np.max(np.abs(A))
    if matrix_scale == 0.0:
        raise InvalidInputError("Matrix is singular.")

    M = np.column_stack([A / matrix_scale, b / matrix_scale])
    n = len(b)

    # Use a scale-aware tolerance on the normalized coefficient matrix.
    pivot_tol = np.finfo(float).eps * n

    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(M[k:, k])))

        if abs(M[p, k]) <= pivot_tol:
            raise InvalidInputError(
                "Matrix is singular or numerically rank-deficient."
            )

        if p != k:
            M[[k, p]] = M[[p, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k] = 0.0
            M[i, k + 1:] -= factor * M[k, k + 1:]

    if abs(M[-1, -2]) <= pivot_tol:
        raise InvalidInputError(
            "Matrix is singular or numerically rank-deficient."
        )

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])
        ) / M[i, i]

    if not np.all(np.isfinite(x)):
        raise NonFiniteValueError("The computed solution is non-finite.")

    return x
