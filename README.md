# Numerical Methods Toolkit

An educational Python library that implements classic numerical algorithms
from first principles, with explicit API contracts, validated inputs,
convergence diagnostics, tests, and reproducible benchmarks.

The goal is to show not only *what* the answer is, but *how* it was obtained,
how fast the method converged, how many function evaluations it needed, and
when the result should not be trusted.

> Theory → Implement → Test → Verify → Measure → Document → Apply

This project is for learning and experimentation. It does not try to replace
mature libraries such as SciPy.

## Algorithms

| Area | Algorithms |
|---|---|
| Root finding | Bisection, Newton–Raphson, Secant, Fixed-point iteration |
| Integration | Composite trapezoidal rule, composite Simpson 1/3 rule |
| ODEs | Explicit Euler, classical Runge–Kutta (RK4) |
| Linear systems | Gaussian elimination with partial pivoting, Jacobi, Gauss–Seidel |
| Interpolation | Lagrange polynomial, Newton divided differences |
| Robotics | Newton's method for nonlinear systems, planar 2-link inverse kinematics |

See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for the mathematics, stopping
rules, evaluation counts, history formats, and failure modes of each method.

## Installation

Requires Python 3.11 or newer. The only runtime dependency is NumPy.

```bash
git clone <repository-url> numerical-methods-toolkit
cd numerical-methods-toolkit
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The `dev` extra installs pytest, SciPy, Matplotlib, Ruff, and build, which are
used by the tests, examples, and release checks. For runtime use only,
`python -m pip install .` is enough.

## Quick start

```python
from numerical_toolkit import SolverOptions, newton, planar_2link_ik, rk4

result = newton(lambda x: x**2 - 2, lambda x: 2 * x, x0=1.5)
print(result.solution)        # ≈ 1.41421356237 (stops once |f(x)| <= ftol)
print(result.converged)       # True
print(result.iterations)      # 3
print(result.function_calls)  # 7  (1 + 2 per iteration: f and df)

# Tighter tolerances, no history
strict = SolverOptions(ftol=1e-14, maxiter=50, store_history=False)
result = newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.5, options=strict)

# ODE: y' = y, y(0) = 1
times, values = rk4(lambda t, y: y, 0.0, [1.0], 1.0, 0.1)
print(values[-1])             # ≈ [2.71828]

# Inverse kinematics: joint angles in radians
ik = planar_2link_ik((1.5, 0.5), l1=1.0, l2=1.0, initial=(0.3, 0.2))
print(ik.converged, ik.solution)
```

A runnable version is in [examples/quickstart.py](examples/quickstart.py).

## API at a glance

- **Iterative solvers** (`bisection`, `newton`, `secant`, `fixed_point`,
  `jacobi`, `gauss_seidel`, `newton_system`, `planar_2link_ik`) take a
  keyword-only `options: SolverOptions | None` and return a `SolverResult`
  with `solution`, `converged`, `iterations`, `function_calls`, `residual`,
  `step_error`, `message`, and `history`.
- **Direct methods** return plain values: `gaussian_elimination` returns a
  NumPy array, `trapezoidal`/`simpson`/`lagrange`/`newton_interpolation`
  return a float, and `euler`/`rk4` return `(times, values)` arrays.
- **Reaching `maxiter` is not an exception.** Check `result.converged`.
- **Invalid input fails loudly.** Library errors derive from
  `NumericalMethodError`: `InvalidInputError` for bad arguments,
  `NonFiniteValueError` for NaN/infinity, plus `InvalidBracketError`,
  `DerivativeZeroError`, and `SecantDenominatorZeroError`. Booleans are never
  accepted where numbers are expected.

## Development

```bash
python -m pytest -q          # full test suite
python -m ruff check .       # lint (pyflakes + syntax errors)
python -m build              # build sdist and wheel into dist/
```

Benchmarks and convergence studies (Matplotlib plots are written to
`outputs/`, which is git-ignored):

```bash
python benchmarks/benchmark_roots.py
python benchmarks/root_convergence.py
python benchmarks/benchmark_robotics_ik.py
python examples/convergence/euler_vs_rk4.py
```

Run them from the repository root with the package installed (or with
`PYTHONPATH=src`). Benchmark timings depend on the machine and are meant for
comparing methods on the same problem, not as absolute performance figures.

On Windows, some scripts print Unicode symbols such as `≈`. If you redirect
their output to a file and see `UnicodeEncodeError`, set `PYTHONUTF8=1`.

## Project layout

```text
src/numerical_toolkit/
  core/            SolverOptions, SolverResult, exceptions, validation helpers
  roots/           bisection, newton, secant, fixed_point
  integration/     trapezoidal, simpson
  ode/             euler, rk4
  linear/          gaussian_elimination, jacobi, gauss_seidel
  interpolation/   lagrange, newton_interpolation
  robotics/        newton_system, planar_2link_ik
tests/             pytest suite, mirroring the package layout
benchmarks/        timing and convergence benchmarks
examples/          quickstart and convergence studies
docs/              algorithm reference
```

## Limitations

- Scalar root finders work on real scalar functions only; there is no complex
  arithmetic.
- ODE solvers use a fixed step and are explicit, so they are not suitable for
  stiff problems.
- Interpolation evaluates at one scalar query point per call.
- `newton_system` uses a dense `numpy.linalg.solve` for each step and has no
  line search or trust region; convergence is local.
- Implementations favour clarity over speed and are not vectorised beyond what
  NumPy provides naturally.

See [CHANGELOG.md](CHANGELOG.md) for release notes.
