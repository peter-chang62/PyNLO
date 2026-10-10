# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyNLO is a Python package for modeling the nonlinear interaction of light with matter. This is a fork of the original PyNLO, rewritten to add 2nd-order (χ²) nonlinearities to the pulse propagation model. The repo contains three importable packages plus data:

- **pynlo/**: core nonlinear optics (pulses, modes, propagation models, materials)
- **edf/**: Er-doped fiber amplifier (5-level rate equations coupled to the NLSE)
- **eydf/**: Er/Yb co-doped fiber amplifier (7-level rate equations coupled to the NLSE)
- **ydf/**: Yb cross-section data and a plotting script only (not yet a package)

## Setup and Commands

Not pip-installed in normal use: put the repo root on `sys.path` (examples do `sys.path.append("../")`) or symlink `pynlo/` into site-packages. Dependencies: numpy, scipy, mkl_fft, numba; matplotlib for plotting/examples. Many examples also `import clipboard` / `import blit` as top-level modules, so `pynlo/utility/` must be on `sys.path` for them.

- All tests: `python -m pytest tests/`
- Single test: `python -m pytest tests/test_light.py::test_pulse_shapes`
- Sphinx docs: `cd docs && make html` (output in `docs/build/html`, `docs/index.html` redirects there)
- Type checking: `pyrightconfig.json` (venv `env`, most strict reports disabled)

Test coverage is thin (`tests/test_light.py`, `tests/test_utility.py`); the scripts in `examples/` are the de-facto integration checks. The two `test_light.py` failures are a known open issue (see `dev_docs/OPEN_ISSUES.md`).

## Architecture

### Core simulation pipeline (pynlo/)

`Pulse` + `Mode` → `Model` → `model.simulate(z_grid, ...)`:

- **`pynlo.utility.TFGrid`** (`pynlo/utility/__init__.py`): the time/frequency grid. `light.Pulse` subclasses it and stores the field; all spectral quantities live on its `v_grid`.
- **`media.Mode`**: linear (`beta`, `alpha`) and nonlinear (`g2`, `g3`, `r3`, poling) properties of a waveguide mode, sampled on the pulse's frequency grid; parameters may be z-dependent callables.
- **`model.Model`**: adaptive-step ERK4(3)-IP integrator (Balac & Mahé 2013). `simulate` → `propagate` → `step`; `update_linearity/update_nonlinearity/update_poling` refresh z-dependent operators. Subclasses:
  - `NLSE`: Kerr + Raman (χ³) only; no χ³ sum/difference-frequency generation.
  - `UPE`: unidirectional propagation equation with simultaneous χ² + χ³ (plus quasi-phase-matching poling). Nonlinear products alias unless the `Pulse` is built with `alias > 1` (see the `UPE` docstring).
- **`materials`**: convenience classes (`SilicaFiber`, `MgLN`, `cLN`) that build a `Mode` and return a model via `generate_model(pulse, ...)`. This is the usual entry point in examples.
- **`utility/chi1.py, chi2.py, chi3.py`**: conversions from material parameters (n, D, γ, Raman, poling domains) to the `beta`/`g2`/`g3`/`r3` inputs `Mode` expects. `fft.py` wraps mkl_fft (falls back to scipy); `blit.py`/`clipboard.py` are plotting/clipboard helpers.
- `device.py` is currently a stub (TODOs only).

### Active fiber packages (edf/, eydf/)

Both follow the same pattern of extending PyNLO classes:

- `Mode` subclasses `pynlo.media.Mode`, adding ion populations (`n1..n5`), pump power ODE (`_dPp_dz`, stepped with scipy `RK45`), and a `gain(z, p_v)` built from the cross sections, populations, and passive loss.
- `Model_EDF`/`Model_EYDF` and `NLSE` subclasses override `propagate`/`simulate` to step the rate equations alongside the pulse and record `Pp`, `sum_a`, `sum_e`.
- `EDF`/`EYDF` subclass `pynlo.materials.SilicaFiber`; `generate_model(pulse, Pp_fwd, ...)` is the entry point.
- `edfa.py`/`eydfa.py`: `amplify(p_fwd, p_bck, fiber, length, Pp_fwd, Pp_bck, ...)` handles bidirectional pumping/seeding by iterating forward and backward passes until the populations converge (`tolerance`).
- Steady-state population equations are in `five_level_ss_eqns.py` / `seven_level_ss_eqns.py` (derived in `edf/notebooks/RE.nb`).
- `*_wsplice.py` variants in edf/ add splice-loss modeling.
- `edf/utility/`: `crossSection` and fiber dispersion (`ER*_betas`) loaders backed by nLIGHT xlsx files in `edf/utility/NLight_provided/`.

See `edf/README.md` and `eydf/README.md` for the physics and full API.

## Conventions

- All quantities in base SI units (Hz, s, J, m, W).
- Public methods/properties use monotonically ordered grids; private `_`-prefixed ones often use FFT-ordered (ifftshifted) arrays.
- Performance-critical kernels use numba JIT.
- Branching: work on `main`. README.rst's "branch off `develop`" rule came from the upstream repo and doesn't apply to this fork (open issue, see `dev_docs/OPEN_ISSUES.md`).
- `.github/workflows/claude.yml` runs Claude Code on `@claude` mentions in issues/PRs.

## Documentation layout

- `docs/` is the **Sphinx user/API documentation** only. Do not put session notes there.
- `dev_docs/` is the **agent session history** described below.

## Code session documentation (required)

Every code session is documented in `dev_docs/`. Sessions run as **plan → implementation → review**, and may loop (review can send work back to a new plan).

```
dev_docs/
├── README.md                          # index of all sessions — read this first
├── OPEN_ISSUES.md                     # living list of known problems / deferred decisions
├── _template/                         # session_README.md, plan.md, implementation.md, review.md
└── sessions/
    └── YYYY-MM-DD_<short-slug>/
        ├── README.md                  # session goal, status, iteration list, key files
        ├── 01_plan.md
        ├── 01_implementation.md
        ├── 01_review.md
        ├── 02_plan.md                 # next iteration, if the review requires it
        └── ...
```

- One folder per session, named `YYYY-MM-DD_<short-slug>`; the `NN_` prefix is the iteration number, so files sort chronologically.
- At the start of a session: read `dev_docs/README.md` and `dev_docs/OPEN_ISSUES.md` (and any related session folders) for prior context, create the session folder from `_template/`, and add a row to the index.
- At each phase transition, write that phase's file. At the end, update the session README and the index row (status, summary, key files touched), add any newly found but unresolved problems to `OPEN_ISSUES.md`, and remove entries this session resolved.
- Write for the next agent: record decisions and their reasons, dead ends, commands run, test results, and open issues, not just a list of changes.

## Agent delegation

When the main agent is running on Opus:

- Delegate menial subtasks (broad file searches, bulk reading/summarizing, running tests or example scripts, mechanical edits, data gathering) to subagents with `model: "sonnet"` (Sonnet 5.5). Give each subagent self-contained instructions and the exact output you need back.
- Opus stays the orchestrator: it plans, decomposes work, verifies subagent results, makes the non-trivial design and physics decisions, and **writes all `dev_docs/` markdown itself** so the docs stay readable and pass full detail to the next session.
