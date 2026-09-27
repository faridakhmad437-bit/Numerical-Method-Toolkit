import math
from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidInputError, SecantDenominatorZeroError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar

def secant(f:Callable[[float],float], x0:float, x1:float, *, options:SolverOptions|None=None)->SolverResult:
    validate_callable(f,"f"); validate_finite_scalar(x0,"x0"); validate_finite_scalar(x1,"x1")
    if x0==x1: raise InvalidInputError("x0 and x1 must be different.")
    o=options or SolverOptions(); f0=float(f(x0)); f1=float(f(x1)); calls=2
    validate_finite_scalar(f0,"f(x0)"); validate_finite_scalar(f1,"f(x1)")
    hist=[x0,x1] if o.store_history else []
    if f0==0: return SolverResult(x0,True,0,calls,0.0,0.0,"First initial guess is an exact root.",tuple(hist) if o.store_history else None)
    if f1==0: return SolverResult(x1,True,0,calls,0.0,0.0,"Second initial guess is an exact root.",tuple(hist) if o.store_history else None)
    prev_step=None
    for k in range(1,o.maxiter+1):
        den=f1-f0
        if den==0: raise SecantDenominatorZeroError("Secant denominator became zero.")
        x2=x1-f1*(x1-x0)/den
        if not math.isfinite(x2): raise InvalidInputError("Secant update became non-finite.")
        f2=float(f(x2)); calls+=1; validate_finite_scalar(f2,"f(x2)")
        step=abs(x2-x1); rel=step/max(1.0,abs(x2))
        if o.store_history: hist.append(x2)
        if abs(f2)<=o.ftol or step<=o.xtol or rel<=o.rtol:
            return SolverResult(x2,True,k,calls,abs(f2),step,"Converged successfully.",tuple(hist) if o.store_history else None)
        x0,f0,x1,f1=x1,f1,x2,f2; prev_step=step
    return SolverResult(x1,False,o.maxiter,calls,abs(f1),prev_step,"Maximum number of iterations reached.",tuple(hist) if o.store_history else None)
