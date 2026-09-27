import math
from collections.abc import Callable
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar
from numerical_toolkit.core.exceptions import NonFiniteValueError

def fixed_point(g:Callable[[float],float], x0:float, *, options:SolverOptions|None=None)->SolverResult:
    validate_callable(g,"g"); validate_finite_scalar(x0,"x0"); o=options or SolverOptions(); x=float(x0); hist=[]; calls=0; prev=None
    for k in range(1,o.maxiter+1):
        xn=float(g(x)); calls+=1
        if not math.isfinite(xn): raise NonFiniteValueError("g(x) produced a non-finite value.")
        step=abs(xn-x); rel=step/max(1.0,abs(xn))
        if o.store_history: hist.append(xn)
        if step<=o.xtol or rel<=o.rtol:
            return SolverResult(xn,True,k,calls,step,step,"Converged successfully.",tuple(hist) if o.store_history else None)
        prev=step; x=xn
    return SolverResult(x,False,o.maxiter,calls,prev,prev,"Maximum number of iterations reached.",tuple(hist) if o.store_history else None)
