import math
from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidInputError, NonFiniteValueError
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar

def simpson(f:Callable[[float],float], a:float, b:float, n:int)->float:
    validate_callable(f,"f"); validate_finite_scalar(a,"a"); validate_finite_scalar(b,"b")
    if not isinstance(n,int) or n<=0 or n%2: raise InvalidInputError("n must be a positive even integer.")
    h=(b-a)/n; total=float(f(a))+float(f(b))
    for i in range(1,n): total+=(4 if i%2 else 2)*float(f(a+i*h))
    result=h*total/3
    if not math.isfinite(result): raise NonFiniteValueError("Integration result is non-finite.")
    return result
