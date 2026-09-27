import math
from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidInputError, NonFiniteValueError
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar

def trapezoidal(f:Callable[[float],float], a:float, b:float, n:int)->float:
    validate_callable(f,"f"); validate_finite_scalar(a,"a"); validate_finite_scalar(b,"b")
    if not isinstance(n,int) or n<=0: raise InvalidInputError("n must be a positive integer.")
    h=(b-a)/n; total=0.5*(float(f(a))+float(f(b)))
    for i in range(1,n): total+=float(f(a+i*h))
    result=h*total
    if not math.isfinite(result): raise NonFiniteValueError("Integration result is non-finite.")
    return result
