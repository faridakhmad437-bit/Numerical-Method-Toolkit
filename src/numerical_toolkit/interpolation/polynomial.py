from __future__ import annotations

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.validation import as_float_array


def _validate_interpolation_inputs(
    x,
    y,
    xq,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Validate and normalize interpolation inputs.

    Parameters
    ----------
    x : array-like
        Interpolation nodes.
    y : array-like
        Function values at interpolation nodes.
    xq : scalar
        Query point where the interpolating polynomial is evaluated.

    Returns
    -------
    x : numpy.ndarray
        Validated 1D interpolation nodes.
    y : numpy.ndarray
        Validated 1D function values.
    xq : float
        Validated scalar query point.

    Raises
    ------
    InvalidInputError
        If dimensions, lengths, uniqueness, or query shape are invalid.
    NonFiniteValueError
        If x, y, or xq contains a non-finite value.
    """

    # ------------------------------------------------------------------
    # Convert x and y to NumPy arrays
    # ------------------------------------------------------------------

    x = as_float_array(x, "x")
    y = as_float_array(y, "y")

    # ------------------------------------------------------------------
    # Validate dimensions
    # ------------------------------------------------------------------

    if x.ndim != 1:
        raise InvalidInputError(
            "x must be a one-dimensional array."
        )

    if y.ndim != 1:
        raise InvalidInputError(
            "y must be a one-dimensional array."
        )

    # ------------------------------------------------------------------
    # Validate non-empty input
    # ------------------------------------------------------------------

    if len(x) == 0:
        raise InvalidInputError(
            "x and y must contain at least one interpolation point."
        )

    # ------------------------------------------------------------------
    # Validate matching lengths
    # ------------------------------------------------------------------

    if len(x) != len(y):
        raise InvalidInputError(
            "x and y must have the same length."
        )

    # ------------------------------------------------------------------
    # Validate finite x and y
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(x)):
        raise NonFiniteValueError(
            "x must contain only finite values."
        )

    if not np.all(np.isfinite(y)):
        raise NonFiniteValueError(
            "y must contain only finite values."
        )

    # ------------------------------------------------------------------
    # Validate unique interpolation nodes
    # ------------------------------------------------------------------

    if len(np.unique(x)) != len(x):
        raise InvalidInputError(
            "x values must be unique."
        )

    # ------------------------------------------------------------------
    # Validate query point
    # ------------------------------------------------------------------

    xq_array = np.asarray(xq)

    if xq_array.ndim != 0:
        raise InvalidInputError(
            "xq must be a scalar query point."
        )

    if xq_array.dtype.kind in "bc":
        raise InvalidInputError(
            "xq must be a real, non-boolean scalar."
        )

    try:
        xq = float(xq_array)
    except (TypeError, ValueError):
        raise InvalidInputError(
            "xq must be a real scalar."
        ) from None

    if not np.isfinite(xq):
        raise NonFiniteValueError(
            "xq must be finite."
        )

    return x, y, xq


def lagrange(x, y, xq):
    """
    Evaluate the Lagrange interpolating polynomial.

    Parameters
    ----------
    x : array-like
        Interpolation nodes.
    y : array-like
        Function values corresponding to x.
    xq : scalar
        Query point.

    Returns
    -------
    float
        Interpolated value at xq.
    """

    x, y, xq = _validate_interpolation_inputs(
        x,
        y,
        xq,
    )

    total = 0.0

    for i in range(len(x)):
        term = y[i]

        for j in range(len(x)):
            if i != j:
                term *= (
                    (xq - x[j])
                    / (x[i] - x[j])
                )

        total += term

    if not np.isfinite(total):
        raise NonFiniteValueError(
            "Lagrange interpolation produced a non-finite result."
        )

    return float(total)


def newton_interpolation(x, y, xq):
    """
    Evaluate the Newton interpolating polynomial
    using divided differences.

    Parameters
    ----------
    x : array-like
        Interpolation nodes.
    y : array-like
        Function values corresponding to x.
    xq : scalar
        Query point.

    Returns
    -------
    float
        Interpolated value at xq.
    """

    x, y, xq = _validate_interpolation_inputs(
        x,
        y,
        xq,
    )

    # Copy y so the caller's input is never modified.
    coef = y.copy()

    n = len(x)

    # ------------------------------------------------------------------
    # Compute divided-difference coefficients.
    #
    # After this loop:
    #
    # coef =
    # [f[x0],
    #  f[x0,x1],
    #  f[x0,x1,x2],
    #  ...]
    # ------------------------------------------------------------------

    for j in range(1, n):
        denominator = (
            x[j:n] - x[:n-j]
        )

        coef[j:n] = (
            coef[j:n] - coef[j-1:n-1]
        ) / denominator

    # ------------------------------------------------------------------
    # Evaluate Newton polynomial using nested multiplication
    # (Horner-like form).
    # ------------------------------------------------------------------

    result = coef[-1]

    for i in range(n - 2, -1, -1):
        result = (
            result * (xq - x[i])
            + coef[i]
        )

    if not np.isfinite(result):
        raise NonFiniteValueError(
            "Newton interpolation produced a non-finite result."
        )

    return float(result)