# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — Unreleased

First stable release of the public API.

### Algorithms

- Root finding: `bisection`, `newton`, `secant`, `fixed_point`.
- Integration: `trapezoidal`, `simpson`.
- ODEs: `euler`, `rk4`.
- Linear systems: `gaussian_elimination`, `jacobi`, `gauss_seidel`.
- Interpolation: `lagrange`, `newton_interpolation`.
- Robotics: `newton_system`, `planar_2link_ik`.

### API contracts

- `SolverOptions` validates its fields: finite positive real tolerances,
  positive integer `maxiter`, boolean `store_history`; booleans are rejected
  where numbers are expected.
- `SolverResult` validates its fields: non-negative integer counts, boolean
  `converged`, non-empty `message`, and finite non-negative (or `None`)
  `residual` and `step_error`.
- Every iterative solver rejects an `options` value that is not a
  `SolverOptions` instance or `None` with `InvalidInputError`.
- Scalar inputs accept Python and NumPy real scalars and reject booleans.
- Malformed array inputs (ragged, non-numeric, boolean, complex) raise
  `InvalidInputError` instead of raw NumPy errors.
- `SecantDenominatorZeroError` is exported from `numerical_toolkit` and
  `numerical_toolkit.core`.

### Fixed

- Importing the package no longer requires Matplotlib (a stray import in the
  Newton solver).
- `newton_system` (and therefore `planar_2link_ik`) applies `rtol` to its step
  criterion; previously solutions whose floating-point spacing exceeded `xtol`
  could never be reported as converged.
- Root solvers no longer iterate in `float32` when given NumPy `float32`
  arguments.
- Secant reports a non-finite update as `NonFiniteValueError` (was
  `InvalidInputError`).
- Bisection validates the final midpoint evaluation when `maxiter` is reached.
- Jacobi and Gauss–Seidel apply `rtol` to their step criterion.
- Newton and Secant no longer report convergence from a small step while the
  residual is still large.
- Bisection sign checks are safe against underflow.
- Fixed-point iteration distinguishes residual-based and step-based
  convergence in its message.
- Trapezoidal, Simpson, Euler, RK4, Gaussian elimination, Jacobi, and
  Gauss–Seidel validate inputs and reject non-finite intermediate values;
  the linear solvers scale the system before elimination or iteration.
- The quickstart example reads `SolverResult.solution` instead of indexing
  the result.

### Tooling and documentation

- Benchmarks for root solvers and robotics IK with repeated measurements, and
  convergence studies under `examples/convergence/`.
- Algorithm reference in `docs/ALGORITHMS.md`; README with installation,
  quick start, API summary, and limitations.
- Tests that import the package without dev dependencies, run the quickstart,
  and execute the README examples.
- Ruff configuration and a GitHub Actions workflow for lint, tests on
  Python 3.11–3.14, and a clean wheel install.
