from __future__ import annotations

import math
from numbers import Real

import numpy as np

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions


def validate_callable(
    value: object,
    name: str = "function",
) -> None:
    """
    Validate that an object is callable.

    Parameters
    ----------
    value:
        Object to validate.
    name:
        Human-readable name used in the error message.
    """

    if not callable(value):
        raise InvalidInputError(
            f"{name} must be callable."
        )


def validate_finite_scalar(
    value: float,
    name: str = "value",
) -> float:
    """
    Validate and return a finite real scalar.

    Parameters
    ----------
    value:
        Value to validate.
    name:
        Human-readable name used in the error message.
    """

    # bool is a subclass of int in Python, so reject it explicitly.
    # numbers.Real also admits NumPy integer and floating scalars.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidInputError(
            f"{name} must be a real scalar."
        )

    value = float(value)

    if not math.isfinite(value):
        raise NonFiniteValueError(
            f"{name} must be finite, got {value!r}."
        )

    return value


def validate_positive(
    value: float,
    name: str = "value",
) -> float:
    """
    Validate and return a positive finite scalar.
    """

    value = validate_finite_scalar(value, name)

    if value <= 0:
        raise InvalidInputError(
            f"{name} must be positive, got {value!r}."
        )

    return value


def validate_non_negative(
    value: float,
    name: str = "value",
) -> float:
    """
    Validate and return a non-negative finite scalar.
    """

    value = validate_finite_scalar(value, name)

    if value < 0:
        raise InvalidInputError(
            f"{name} must be non-negative, got {value!r}."
        )

    return value


def validate_options(
    options: SolverOptions | None,
) -> SolverOptions:
    """
    Return solver options, using the defaults when ``options`` is None.
    """

    if options is None:
        return SolverOptions()

    if not isinstance(options, SolverOptions):
        raise InvalidInputError(
            "options must be a SolverOptions instance or None."
        )

    return options


def as_float_array(
    value: object,
    name: str = "value",
) -> np.ndarray:
    """
    Convert array-like numeric input to a floating-point NumPy array.

    Ragged, non-numeric, boolean, and complex inputs raise
    InvalidInputError instead of a raw NumPy exception. Shape and
    finiteness are left to the caller.
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(
            f"{name} must be a valid numeric array."
        ) from exc

    if raw.dtype.kind in "bc":
        raise InvalidInputError(
            f"{name} must contain real, non-boolean values."
        )

    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(
            f"{name} must contain numeric values."
        ) from exc
