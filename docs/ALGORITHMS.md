# Algorithm reference

This document describes every public algorithm in `numerical_toolkit`: the
mathematical idea, the exact API contract, the stopping rule, how
`iterations` and `function_calls` are counted, what `history` contains, and the
known failure modes.

| Area | Function | Returns | Order / cost |
|---|---|---|---|
| Roots | `bisection(f, a, b, *, options)` | `SolverResult` | Linear, 1 `f` call per iteration |
| Roots | `newton(f, df, x0, *, options)` | `SolverResult` | Quadratic near a simple root, 2 calls per iteration |
| Roots | `secant(f, x0, x1, *, options)` | `SolverResult` | Superlinear (≈1.618), 1 call per iteration |
| Roots | `fixed_point(g, x0, *, options)` | `SolverResult` | Linear with rate ≈ \|g'(x*)\| |
| Integration | `trapezoidal(f, a, b, n)` | `float` | Error O(h²), n + 1 calls |
| Integration | `simpson(f, a, b, n)` | `float` | Error O(h⁴), n + 1 calls |
| ODE | `euler(f, t0, y0, t_end, h)` | `(times, values)` | Global error O(h), 1 call per step |
| ODE | `rk4(f, t0, y0, t_end, h)` | `(times, values)` | Global error O(h⁴), 4 calls per step |
| Linear | `gaussian_elimination(A, b)` | `numpy.ndarray` | O(n³) direct solve |
| Linear | `jacobi(A, b, x0=None, *, options)` | `SolverResult` | O(n²) per iteration |
| Linear | `gauss_seidel(A, b, x0=None, *, options)` | `SolverResult` | O(n²) per iteration |
| Interpolation | `lagrange(x, y, xq)` | `float` | O(n²) per query |
| Interpolation | `newton_interpolation(x, y, xq)` | `float` | O(n²) per query |
| Robotics | `newton_system(F, J, x0, *, options)` | `SolverResult` | Quadratic near a regular root |
| Robotics | `planar_2link_ik(target, l1, l2, initial=(0, 0), *, options)` | `SolverResult` | Uses `newton_system` |

All functions are importable from the top-level package, for example
`from numerical_toolkit import newton, SolverOptions`.

---

## 1. Shared contracts

### 1.1 `SolverOptions`

Frozen dataclass accepted by every iterative solver through the keyword-only
`options` argument. Passing `None` (the default) uses the defaults below;
passing any other non-`SolverOptions` value raises `InvalidInputError`.

| Field | Default | Meaning |
|---|---:|---|
| `xtol` | `1e-10` | Absolute tolerance on the step (change in the solution) |
| `rtol` | `1e-8` | Relative tolerance on the step |
| `ftol` | `1e-10` | Absolute tolerance on the residual |
| `maxiter` | `100` | Maximum number of iterations |
| `store_history` | `True` | Whether `SolverResult.history` is populated |

Validation: tolerances must be real, finite and positive (booleans rejected,
`TypeError` for wrong type, `ValueError` for wrong value); `maxiter` must be a
positive integer (booleans rejected); `store_history` must be a boolean.

How `xtol` and `rtol` combine depends on the solver family:

- **Scalar root solvers** treat them as alternatives:
  `step <= xtol` **or** `step / max(1, |x|) <= rtol`.
- **Vector solvers** (`jacobi`, `gauss_seidel`, `newton_system`) use the
  combined bound `step <= xtol + rtol * ||x||_inf`.

### 1.2 `SolverResult`

Frozen dataclass returned by all iterative solvers.

| Field | Meaning |
|---|---|
| `solution` | Final approximation (float for scalar solvers, `ndarray` for vector solvers) |
| `converged` | `True` only when the solver's convergence criterion was met |
| `iterations` | Iterations performed (`0` when the initial guess is accepted) |
| `function_calls` | Evaluations of user callables, see each algorithm below |
| `residual` | Final residual, or `None` when unavailable |
| `step_error` | Size of the final update, or `None` when unavailable |
| `message` | Human-readable status |
| `history` | Tuple of per-iteration records, or `None` when `store_history=False` |

Validation: counts are non-negative integers (not booleans), `converged` is a
boolean, `message` is a non-empty string, and `residual`/`step_error` are
`None` or finite non-negative real numbers.

`function_calls` counts calls to user-supplied callables only. It is not a
count of floating-point operations. The linear solvers take matrices, not
callables, so they always report `function_calls == 0`.

### 1.3 Exceptions

All library exceptions derive from `NumericalMethodError`.

```text
NumericalMethodError
├── InvalidInputError            invalid argument type, shape, or value
│   └── InvalidBracketError      bisection bracket is not valid
├── NonFiniteValueError          NaN or infinity in an input or during iteration
├── DerivativeZeroError          Newton derivative is exactly zero
├── SecantDenominatorZeroError   f(x1) == f(x0) during a secant step
└── ConvergenceError             reserved; no solver currently raises it
```

`SolverOptions` and `SolverResult` raise the built-in `TypeError` and
`ValueError`, because they validate configuration objects rather than
numerical input.

Reaching `maxiter` is **not** an exception: solvers return a `SolverResult`
with `converged=False` so that the diagnostics remain inspectable.

### 1.4 Input validation

- Scalars must be finite real numbers. Python and NumPy integer and floating
  scalars are accepted and converted to `float`; booleans, strings, complex
  numbers and `None` raise `InvalidInputError`; NaN and infinity raise
  `NonFiniteValueError`.
- Arrays must be rectangular, real and non-boolean. Ragged, non-numeric,
  boolean or complex arrays raise `InvalidInputError`; arrays containing NaN or
  infinity raise `NonFiniteValueError`.
- Values returned by user callables are validated on every evaluation. A
  non-finite value raises `NonFiniteValueError` instead of being returned as a
  result.

---

## 2. Root finding

All four solvers return a `SolverResult` whose `history` (when stored) is a
tuple of floats, as described per method.

### 2.1 Bisection

**Idea.** If `f` is continuous on `[a, b]` and `f(a)` and `f(b)` have opposite
signs, a root lies in the interval. Halve the interval each iteration and keep
the half where the sign change persists.

**Assumptions.** `a < b`, both finite, `f(a)` and `f(b)` finite with opposite
signs (otherwise `InvalidBracketError`). Sign comparison is done without
multiplying `f(a) * f(b)`, so it is safe against underflow.

**Stopping rule** (any one of these is convergence):
`|f(m)| <= ftol`, or `b - a <= 2 * xtol`, or
`|m_k - m_{k-1}| / max(1, |m_k|) <= rtol`.

**Counts.** `function_calls` = 2 endpoint evaluations + 1 per iteration
(+1 for the final midpoint when `maxiter` is reached).

**History.** The midpoint of every iteration. If an endpoint is an exact root,
the result has `iterations == 0` and `history == (endpoint,)`.

**Failure modes.** Only finds one root of odd multiplicity in the bracket;
cannot detect roots of even multiplicity (no sign change); converges to a
discontinuity with a sign change just as happily as to a root.

### 2.2 Newton

**Idea.** `x_{k+1} = x_k - f(x_k) / f'(x_k)`.

**Stopping rule.** `|f(x_{k+1})| <= ftol` gives `converged=True`. If the step
satisfies `xtol` or `rtol` while the residual is still larger than `ftol`, the
solver stops with `converged=False` and the message
`"Step tolerance reached before residual tolerance."`. A small step alone is
never reported as success.

**Counts.** One `f` evaluation for `x0`, then one `df` and one `f` evaluation
per iteration.

**History.** The new iterate of each iteration. If `f(x0) == 0` the result has
`iterations == 0` and `history == (x0,)`.

**Failure modes.** `DerivativeZeroError` when `f'(x) == 0`;
`NonFiniteValueError` when `f`, `f'`, or the update becomes non-finite;
divergence or cycling from a poor initial guess; only linear convergence at
multiple roots.

### 2.3 Secant

**Idea.** Newton with the derivative replaced by the slope through the last two
iterates: `x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))`.

**Assumptions.** `x0 != x1` (otherwise `InvalidInputError`).

**Stopping rule.** Same as Newton: success only through `|f| <= ftol`; a small
step with a large residual stops with `converged=False`.

**Counts.** Two evaluations for the initial guesses, then one per iteration.

**History.** `(x0, x1, x2, ...)`, so `len(history) == iterations + 2`.

**Failure modes.** `SecantDenominatorZeroError` when `f(x_k) == f(x_{k-1})`;
`NonFiniteValueError` when the update or `f` becomes non-finite; divergence
from poor starting points.

### 2.4 Fixed-point iteration

**Idea.** Rewrite `f(x) = 0` as `x = g(x)` and iterate `x_{k+1} = g(x_k)`.
Locally convergent when `|g'(x*)| < 1`.

**Stopping rule** (both are reported as `converged=True`, with distinct
messages): residual `|g(x) - x| <= ftol`, or step `<= xtol` / relative step
`<= rtol`. Unlike Newton and Secant, a small step counts as convergence here,
because for a contraction the step bounds the error.

**Counts.** One `g` evaluation for `x0`, then one per iteration.

**History.** The iterates `x_1, x_2, ...`. If `x0` already satisfies the
residual tolerance the history is empty and `iterations == 0`.

**Failure modes.** Divergence or oscillation when `|g'| >= 1` near the fixed
point; slow convergence when `|g'|` is close to 1 (a step-based stop can then
be premature); `NonFiniteValueError` when `g` overflows.

---

## 3. Numerical integration

Both rules take a callable `f`, finite bounds `a` and `b`, and the number of
subintervals `n`, and return a `float`. `a > b` returns the signed integral and
`a == b` returns `0.0`. `n` must be a positive integer (Python or NumPy
integer, not a boolean); Simpson additionally requires `n` to be even.

| Rule | Formula on `n` panels of width `h = (b - a) / n` | Exact for | Error |
|---|---|---|---|
| Trapezoidal | `h [f0/2 + f1 + ... + f_{n-1} + fn/2]` | Polynomials of degree ≤ 1 | `-(b - a) h² f''(ξ) / 12` |
| Simpson 1/3 | `h/3 [f0 + 4 f1 + 2 f2 + ... + 4 f_{n-1} + fn]` | Polynomials of degree ≤ 3 | `-(b - a) h⁴ f⁗(ξ) / 180` |

**Failure modes.** Every evaluation is checked; a non-finite `f(x)` or a
non-finite total raises `NonFiniteValueError`. The error bounds assume a smooth
integrand; singularities or discontinuities inside `[a, b]` degrade the rate.

---

## 4. Ordinary differential equations

`euler(f, t0, y0, t_end, h)` and `rk4(f, t0, y0, t_end, h)` integrate
`y' = f(t, y)` from `t0` to `t_end` with fixed step `h`.

**Contract.**

- `f(t, y)` must return values with the same shape as `y0`.
- `y0` may be a scalar or an array; booleans are rejected.
- `h` must be positive and finite; `t_end >= t0`.
- The last step is shortened so that `times[-1] == t_end` exactly.
- Returns `(times, values)` where `times.shape == (m,)` and
  `values.shape == (m, *y0.shape)`. When `t_end == t0`, only the initial
  point is returned.

| Method | Update | Calls per step | Global error |
|---|---|---:|---|
| Euler | `y + h f(t, y)` | 1 | O(h) |
| RK4 | `y + h (k1 + 2 k2 + 2 k3 + k4) / 6` | 4 | O(h⁴) |

**Failure modes.** Both are explicit methods: stiff problems need a very small
`h` for stability. A non-finite derivative or state raises
`NonFiniteValueError`; a step too small to advance `t` in floating point raises
`InvalidInputError`.

---

## 5. Linear systems

All three take a square, finite, non-empty matrix `A` and a vector `b` of
matching length.

### 5.1 Gaussian elimination

Forward elimination with partial pivoting, then back substitution. The system
is first divided by `max|A|`, and a pivot is treated as zero when it is at most
`n · machine epsilon` on that normalized scale. Singular or numerically
rank-deficient matrices raise `InvalidInputError`. Returns a plain
`numpy.ndarray` (not a `SolverResult`), because a direct method has no
iteration diagnostics.

### 5.2 Jacobi and Gauss–Seidel

**Idea.** Split `A = D + R` and iterate. Jacobi updates every component from
the previous iterate; Gauss–Seidel uses each new component immediately, which
usually converges faster.

**Convergence.** Guaranteed when `A` is strictly diagonally dominant; in
general the iteration converges if and only if the spectral radius of the
iteration matrix is below 1. Both require non-zero diagonal entries
(`InvalidInputError` otherwise). The iteration runs on `A / max|A|` to reduce
scaling problems; residuals are reported for the original system.

**Stopping rule.** Both conditions must hold:
`||A x - b||_inf <= ftol` **and** `step <= xtol + rtol * ||x||_inf`
(Jacobi scales `rtol` by the previous iterate, Gauss–Seidel by the new one).
If the initial guess already satisfies `ftol`, the solver returns with
`iterations == 0`.

**Counts.** `function_calls` is always `0`.

**History.** A tuple of dictionaries, starting with iteration 0 (the initial
guess):

```python
{"iteration": k, "solution": ndarray, "residual": float, "step_error": float | None}
```

**Failure modes.** Divergence for matrices that are not diagonally dominant
(`NonFiniteValueError` if it overflows, otherwise `converged=False` at
`maxiter`).

---

## 6. Polynomial interpolation

`lagrange(x, y, xq)` and `newton_interpolation(x, y, xq)` evaluate the unique
polynomial of degree ≤ n−1 through the `n` points `(x_i, y_i)` at a single
scalar `xq`, returning a `float`. Both give the same polynomial; Newton uses
divided differences and nested evaluation.

**Contract.** `x` and `y` are non-empty one-dimensional arrays of equal length
with finite values; nodes in `x` must be distinct; `xq` is a finite real scalar.

**Failure modes.** High-degree interpolation on equally spaced nodes suffers
from the Runge phenomenon (large oscillations near the ends of the interval);
Chebyshev-type nodes reduce it. See
`examples/convergence/interpolation_node_comparison.py`.

---

## 7. Robotics

### 7.1 `newton_system`

Solves `F(x) = 0` for `x ∈ ℝⁿ` via `J(x_k) Δ = -F(x_k)`, `x_{k+1} = x_k + Δ`.

**Contract.** `x0` is a non-empty one-dimensional vector; `F` returns a vector
of the same length; `J` returns an `n × n` matrix.

**Stopping rule.** Both `||F(x)||_inf <= ftol` and
`||Δ||_inf <= xtol + rtol * ||x_{k+1}||_inf`. If `F(x0)` already satisfies
`ftol`, the solver returns with `iterations == 0`.

**Counts.** One `F` call for `x0`, then one `J` and one `F` call per iteration.

**History.** Same dictionary records as Jacobi/Gauss–Seidel, starting with
iteration 0.

**Failure modes.** An exactly singular Jacobian raises `InvalidInputError`; a
nearly singular one can produce large steps. Convergence is local and depends
on `x0`.

### 7.2 `planar_2link_ik`

Finds joint angles `(q1, q2)` in radians such that the end effector of a planar
arm with link lengths `l1`, `l2` reaches `target = (x, y)`:

```text
x = l1 cos q1 + l2 cos(q1 + q2)
y = l1 sin q1 + l2 sin(q1 + q2)
```

The system is solved with `newton_system` and the analytical Jacobian, so the
result and counts follow §7.1.

**Workspace.** Reachable targets satisfy `|l1 - l2| <= ||target|| <= l1 + l2`.
For a target outside this annulus the function does not iterate: it returns
`converged=False`, `iterations=0`, `function_calls=0`, `residual=None`, and
`solution` equal to the initial guess.

**Failure modes.** Two solutions (elbow-up/elbow-down) exist inside the
workspace; the one found depends on `initial`. On the workspace boundary the
Jacobian is singular and convergence slows or fails.
