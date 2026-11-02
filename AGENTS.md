# AGENTS.md

Guidance for AI coding agents and reviewers working on scikit-rf (`skrf`).

## Before you finish

- Run the tests: `pytest` (whole suite) or `pytest -v -c "" skrf/tests/test_network.py` (one file).
- Run the linter: `ruff check .` (or `prek install` once, to run it on every commit).
- Add or update tests for every behavior change or bug fix.
- Warnings fail the test suite (`filterwarnings = error`). Fix the cause rather than silencing it.

## Correctness

- Keep the public API backward compatible. Deprecate with a warning before removing or renaming.
- S-parameter arrays have shape `(nfreq, nports, nports)`. Keep that shape through every operation.
- Frequencies are stored in Hz. Port impedances (`z0`) can be complex and differ per port.
- Prefer vectorized numpy over Python loops over frequency points.
- Check numerical results against a reference: a published formula, another simulator's output, or an existing test file.

## Code style

- Python >= 3.10. Runtime dependencies are only numpy, scipy, pandas and typing-extensions.
- Import optional dependencies (matplotlib, networkx, openpyxl, pyvisa) inside the functions that need them.
- Line length is 120. Ruff rules live in `pyproject.toml`.
- Docstrings follow the numpydoc format.

## Tests

- Tests live next to the code they test, in a `tests/` folder (for example `skrf/media/tests/`).
- Test data files (`.s2p`, `.csv` and so on) go in that same `tests/` folder.
- Shared fixtures live in `conftest.py`.
- Virtual instrument tests (`skrf/vi`) are excluded by default.

## Docs

- Sources are in `doc/source/`, written in reStructuredText and built with Sphinx (`cd doc && make html`).
- Examples and tutorials are Jupyter notebooks. Clear all outputs before committing.
- Notebooks run as tests in CI, so keep them working.

## Layout

- `skrf/network.py`: `Network`, the core class.
- `skrf/frequency.py`, `skrf/networkSet.py`, `skrf/circuit.py`: frequency axis, sets of networks, circuit building.
- `skrf/calibration/`: VNA calibration algorithms.
- `skrf/media/`: transmission line models.
- `skrf/io/`: file formats (Touchstone, CSV, MDIF and others).
- `skrf/vi/`: instrument control.

## Pull requests

- Work on a fork and open the PR against `master`.
- Keep each PR to one change, and say in the description what it fixes and how it was tested.
