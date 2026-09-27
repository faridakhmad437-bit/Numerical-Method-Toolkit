import numpy as np
from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidInputError

def euler(f:Callable, t0:float, y0, t_end:float, h:float):
    if h<=0: raise InvalidInputError("h must be positive.")
    if t_end<t0: raise InvalidInputError("t_end must be >= t0.")
    y=np.asarray(y0,dtype=float); times=[float(t0)]; values=[y.copy()]; t=float(t0)
    while t<t_end-1e-15:
        dt=min(h,t_end-t); y=y+dt*np.asarray(f(t,y),dtype=float); t+=dt; times.append(t); values.append(y.copy())
    return np.asarray(times),np.asarray(values)
