import math
from collections.abc import Callable
from matplotlib.pyplot import step
from numerical_toolkit.core.exceptions import DerivativeZeroError, NonFiniteValueError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar

def newton(f: Callable[[float],float], df: Callable[[float],float], x0: float, *, options: SolverOptions|None=None)->SolverResult:
    validate_callable(f,"f"); validate_callable(df,"df"); validate_finite_scalar(x0,"x0")
    o=options or SolverOptions(); x=float(x0); fx=float(f(x)); calls=1
    if not math.isfinite(fx): raise NonFiniteValueError("f(x0) must be finite.")
    if fx==0: return SolverResult(x,True,0,calls,0.0,0.0,"Initial guess is an exact root.", (x,) if o.store_history else None)
    hist=[]; prev_step=None
    for k in range(1,o.maxiter+1):
        dfx=float(df(x)); calls+=1
        if not math.isfinite(dfx): raise NonFiniteValueError("df(x) must be finite.")
        if dfx==0: raise DerivativeZeroError(f"Derivative is zero at x={x!r}.")
        xn=x-fx/dfx
        if not math.isfinite(xn): raise NonFiniteValueError("Newton update is non-finite.")
        fn=float(f(xn)); calls+=1
        if not math.isfinite(fn): raise NonFiniteValueError("f(x_new) must be finite.")
        step=abs(xn-x); rel=step/max(1.0,abs(xn))
        if o.store_history: hist.append(xn)
        if abs(fn) <= o.ftol:
            return SolverResult(
                xn,
                True,
                k,
                calls,
                abs(fn),
                step,
                "Converged successfully: residual tolerance satisfied.",
                tuple(hist) if o.store_history else None,
            )

        if step <= o.xtol or rel <= o.rtol:
            return SolverResult(
                xn,
                False,
                k,
                calls,
                abs(fn),
                step,
                "Step tolerance reached before residual tolerance.",
                tuple(hist) if o.store_history else None,
            )
        x,fx,prev_step=xn,fn,step
    return SolverResult(x,False,o.maxiter,calls,abs(fx),prev_step,"Maximum number of iterations reached.",tuple(hist) if o.store_history else None)
