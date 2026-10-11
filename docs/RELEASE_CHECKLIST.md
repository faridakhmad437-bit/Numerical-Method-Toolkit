# Release checklist

Run from the repository root with the dev environment active.

## Before tagging

- [ ] Working tree is clean: `git status --short` prints nothing.
- [ ] Version is the same in `pyproject.toml` and
      `src/numerical_toolkit/__init__.py` (checked by `tests/test_packaging.py`).
- [ ] `CHANGELOG.md` has a dated section for the version, not "Unreleased".
- [ ] A license is chosen, a `LICENSE` file is committed, and `pyproject.toml`
      declares it.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m pytest -q` passes on the newest supported Python.
- [ ] The test suite passes on the oldest supported Python (3.11).
- [ ] `python -m build` produces an sdist and a wheel.
- [ ] The wheel installs into a fresh virtual environment and
      `import numerical_toolkit` works with only NumPy installed.
- [ ] `examples/quickstart.py` and the benchmarks in `benchmarks/` run.
- [ ] CI is green on the release commit (once the repository has a remote).

## Tagging

```bash
git tag -a v1.0.0 -m "Numerical Methods Toolkit 1.0.0"
git push --follow-tags
```

## After tagging

- [ ] Start a new "Unreleased" section in `CHANGELOG.md`.
