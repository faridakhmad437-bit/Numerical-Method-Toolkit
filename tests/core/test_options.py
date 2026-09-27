import pytest

from numerical_toolkit.core.options import SolverOptions


def test_default_options():
    options = SolverOptions()

    assert options.xtol == 1e-10
    assert options.rtol == 1e-8
    assert options.ftol == 1e-10
    assert options.maxiter == 100
    assert options.store_history is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("xtol", 0),
        ("xtol", -1),
        ("rtol", 0),
        ("rtol", -1),
        ("ftol", 0),
        ("ftol", -1),
        ("maxiter", 0),
        ("maxiter", -1),
    ],
)
def test_invalid_options(field, value):
    with pytest.raises(ValueError):
        SolverOptions(**{field: value})


def test_invalid_history_type():
    with pytest.raises(TypeError):
        SolverOptions(store_history=1)