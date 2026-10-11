from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidBracketError
from numerical_toolkit.core.options import SolverOptions
from numerical_toolkit.core.results import SolverResult
from numerical_toolkit.core.validation import validate_callable, validate_finite_scalar, validate_options

def bisection(f: Callable[[float], float], a: float, b: float, *, options: SolverOptions|None=None) -> SolverResult:
    validate_callable(f,"f"); a=validate_finite_scalar(a,"a"); b=validate_finite_scalar(b,"b")
    if a>=b: raise InvalidBracketError("Require a < b.")
    o=validate_options(options); fa=float(f(a)); fb=float(f(b)); calls=2
    validate_finite_scalar(fa,"f(a)"); validate_finite_scalar(fb,"f(b)")
    if fa==0: return SolverResult(a,True,0,calls,0.0,0.0,"Left endpoint is an exact root.",(a,) if o.store_history else None)
    if fb==0: return SolverResult(b,True,0,calls,0.0,0.0,"Right endpoint is an exact root.",(b,) if o.store_history else None)
    if (fa > 0 and fb > 0) or (fa < 0 and fb < 0):
        raise InvalidBracketError(
            "f(a) and f(b) must have opposite signs."
        )
    hist=[]; prev=None
    for k in range(1,o.maxiter+1):
        m=a+0.5*(b-a); fm=float(f(m)); calls+=1; validate_finite_scalar(fm,"f(midpoint)")
        if o.store_history: hist.append(m)
        step=None if prev is None else abs(m-prev)
        rel=None if step is None else step/max(1.0,abs(m))
        if abs(fm)<=o.ftol or abs(b-a)<=2*o.xtol or (rel is not None and rel<=o.rtol):
            return SolverResult(m,True,k,calls,abs(fm),step,"Converged successfully.",tuple(hist) if o.store_history else None)
        if (fa < 0 < fm) or (fm < 0 < fa):
            b, fb = m, fm
        else:
            a, fa = m, fm
        prev=m
    m=a+0.5*(b-a); fm=float(f(m)); calls+=1; validate_finite_scalar(fm,"f(midpoint)")
    return SolverResult(m,False,o.maxiter,calls,abs(fm),None if prev is None else abs(m-prev),"Maximum number of iterations reached.",tuple(hist) if o.store_history else None)
