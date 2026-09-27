import math

import pytest

from numerical_toolkit.core.exceptions import (
    InvalidInputError,
    NonFiniteValueError,
)
from numerical_toolkit.core.validation import (
    validate_callable,
    validate_finite_scalar,
    validate_non_negative,
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