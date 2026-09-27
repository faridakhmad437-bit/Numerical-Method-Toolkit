import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)


def gaussian_elimination(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Validate matrix/vector dimensions.
    if (
        A.ndim != 2
        or A.shape[0] != A.shape[1]
        or b.shape != (A.shape[0],)
    ):
        raise InvalidInputError(
            "A must be square and b must match its size."
        )

    # Empty linear systems are not meaningful for this solver.
    if A.shape[0] == 0:
        raise InvalidInputError(
            "A and b must define a non-empty linear system."
        )

    # Reject NaN and infinity before numerical operations.
    if not np.all(np.isfinite(A)):
        raise NonFiniteValueError(
            "A must contain only finite values."
        )

    if not np.all(np.isfinite(b)):
        raise NonFiniteValueError(
            "b must contain only finite values."
        )

    M = np.column_stack([A, b]).copy()
    n = len(b)

    # Forward elimination with partial pivoting.
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(M[k:, k])))

        if abs(M[p, k]) < 1e-14:
            raise InvalidInputError(
                "Matrix is singular or nearly singular."
            )

        if p != k:
            M[[k, p]] = M[[p, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

    # Validate the final pivot.
    if abs(M[-1, -2]) < 1e-14:
        raise InvalidInputError(
            "Matrix is singular or nearly singular."
        )

    # Back substitution.
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            M[i, -1]
            - np.dot(M[i, i + 1:n], x[i + 1:n])
        ) / M[i, i]

    return x