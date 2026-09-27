import math
import numpy as np
from numerical_toolkit.roots import bisection,newton,secant,fixed_point
from numerical_toolkit.integration import trapezoidal,simpson
from numerical_toolkit.ode import euler,rk4
from numerical_toolkit.linear import gaussian_elimination,jacobi,gauss_seidel
from numerical_toolkit.interpolation import lagrange,newton_interpolation
from numerical_toolkit.robotics import planar_2link_ik
from numerical_toolkit.core.options import SolverOptions

def test_roots():
    f=lambda x:x*x-2
    assert abs(bisection(f,1,2).solution-math.sqrt(2))<1e-8
    assert abs(newton(f,lambda x:2*x,1.5).solution-math.sqrt(2))<1e-8
    assert abs(secant(f,1,2).solution-math.sqrt(2))<1e-8
    assert abs(fixed_point(lambda x:0.5*(x+2/x),1.5).solution-math.sqrt(2))<1e-8

def test_integration():
    assert abs(trapezoidal(lambda x:x*x,0,1,100)-1/3)<1e-4
    assert abs(simpson(lambda x:x*x,0,1,100)-1/3)<1e-10

def test_ode():
    f=lambda t,y:y
    t,y=euler(f,0,[1],1,0.01); assert abs(y[-1,0]-math.e)<0.02
    t,y=rk4(f,0,[1],1,0.1); assert abs(y[-1,0]-math.e)<1e-4

def test_linear():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    b = np.array([1.0, 2.0])

    expected = np.linalg.solve(A, b)

    assert np.allclose(
        gaussian_elimination(A, b),
        expected,
    )

    jacobi_result = jacobi(
        A,
        b,
        options=SolverOptions(
            ftol=1e-9,
            xtol=1e-9,
        ),
    )

    assert jacobi_result.converged
    assert np.allclose(
        jacobi_result.solution,
        expected,
        atol=1e-7,
    )

    gauss_seidel_result = gauss_seidel(
        A,
        b,
        options=SolverOptions(
            ftol=1e-9,
            xtol=1e-9,
        ),
    )

    assert gauss_seidel_result.converged
    assert np.allclose(
        gauss_seidel_result.solution,
        expected,
        atol=1e-7,
    )
def test_interpolation():
    x=[0,1,2]; y=[1,3,7]
    assert abs(lagrange(x,y,1.5)-4.75)<1e-12
    assert abs(newton_interpolation(x,y,1.5)-4.75)<1e-12

def test_robotics():
    result = planar_2link_ik(
        target=(1.5, 0.5),
        l1=1.0,
        l2=1.0,
        initial=(0.3, 0.2),
    )

    assert result.converged

    q = result.solution
    px = np.cos(q[0]) + np.cos(q[0] + q[1])
    py = np.sin(q[0]) + np.sin(q[0] + q[1])
    assert np.allclose([px, py], [1.5, 0.5], atol=1e-7)
