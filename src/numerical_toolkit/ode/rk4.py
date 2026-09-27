import numpy as np
from collections.abc import Callable
from numerical_toolkit.core.exceptions import InvalidInputError

def rk4(f:Callable, t0:float, y0, t_end:float, h:float):
    if h<=0: raise InvalidInputError("h must be positive.")
    if t_end<t0: raise InvalidInputError("t_end must be >= t0.")
    y=np.asarray(y0,dtype=float); times=[float(t0)]; values=[y.copy()]; t=float(t0)
    while t<t_end-1e-15:
        dt=min(h,t_end-t); k1=np.asarray(f(t,y),float); k2=np.asarray(f(t+dt/2,y+dt*k1/2),float); k3=np.asarray(f(t+dt/2,y+dt*k2/2),float); k4=np.asarray(f(t+dt,y+dt*k3),float); y=y+dt*(k1+2*k2+2*k3+k4)/6; t+=dt; times.append(t); values.append(y.copy())
    return np.asarray(times),np.asarray(values)
