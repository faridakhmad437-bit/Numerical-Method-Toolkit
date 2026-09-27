import math
from numerical_toolkit import bisection,newton,secant,fixed_point,simpson,rk4,gaussian_elimination,planar_2link_ik

f=lambda x:x*x-2
print("Bisection:",bisection(f,1,2).solution)
print("Newton:",newton(f,lambda x:2*x,1.5).solution)
print("Secant:",secant(f,1,2).solution)
print("Fixed point:",fixed_point(lambda x:0.5*(x+2/x),1.5).solution)
print("Simpson:",simpson(lambda x:x*x,0,1,100))
t,y=rk4(lambda t,y:y,0,[1],1,0.1); print("RK4:",y[-1])
print("Gaussian:",gaussian_elimination([[4,1],[2,3]],[1,2]))
print("IK:",planar_2link_ik((1.5,0.5),1,1,initial=(0.3,0.2))[0])
