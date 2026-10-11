import math

import numpy as np
import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
    validate_non_negative,
    validate_options,
    validate_positive,
)


def test_validate_callable():
    validate_callable(lambda x: x)


def test_validate_callable_rejects_non_callable():
    with pytest.raises(InvalidInputError):
        validate_callable(42)


def test_validate_finite_scalar():
    assert validate_finite_scalar(2) == 2.0


@pytest.mark.parametrize(
    "value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_validate_finite_scalar_rejects_non_finite(value):
    with pytest.raises(NonFiniteValueError):
        validate_finite_scalar(value)


def test_validate_positive():
    assert validate_positive(2) == 2.0


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_validate_positive_rejects_invalid_values(value):
    with pytest.raises(InvalidInputError):
        validate_positive(value)


def test_validate_non_negative():
    assert validate_non_negative(0) == 0.0
    assert validate_non_negative(2) == 2.0


def test_validate_non_negative_rejects_negative():
    with pytest.raises(InvalidInputError):
        validate_non_negative(-1)

@pytest.mark.parametrize("value", [True, False])
def test_validate_finite_scalar_rejects_booleans(value):
    with pytest.raises(InvalidInputError):
        validate_finite_scalar(value)


@pytest.mark.parametrize("value", ["1.0", None, [1.0], 1 + 2j])
def test_validate_finite_scalar_rejects_non_real_values(value):
    with pytest.raises(InvalidInputError):
        validate_finite_scalar(value)


@pytest.mark.parametrize(
    "value",
    [np.int64(3), np.float32(3.0), np.float64(3.0)],
)
def test_validate_finite_scalar_accepts_numpy_scalars(value):
    result = validate_finite_scalar(value)

    assert result == 3.0
    assert type(result) is float


def test_validate_options_returns_defaults_for_none():
    assert validate_options(None) == SolverOptions()


def test_validate_options_returns_given_options():
    options = SolverOptions(maxiter=5)

    assert validate_options(options) is options


@pytest.mark.parametrize("value", [{"maxiter": 5}, 5, "fast"])
def test_validate_options_rejects_other_types(value):
    with pytest.raises(InvalidInputError):
        validate_options(value)
