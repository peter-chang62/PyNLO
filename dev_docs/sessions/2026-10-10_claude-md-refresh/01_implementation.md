# 01 Implementation — CLAUDE.md refresh and session-docs setup

## What changed
- **`CLAUDE.md`** (rewritten in place):
  - Project overview now lists `ydf/` (data and plot script only).
  - New Setup and Commands section:
    - the `sys.path` install approach;
    - `pynlo/utility/` must be on the path because examples run `import clipboard` / `import blit`;
    - how to run a single test, build the Sphinx docs, and use the pyright config.
  - Architecture is now organized around the pipeline `TFGrid` → `Pulse` / `Mode` → `Model` (`NLSE`, `UPE`) → `simulate`. The text was checked against the docstrings:
    - `NLSE` covers Kerr and Raman only;
    - `UPE` covers χ² + χ³ and has an aliasing caveat;
    - `materials.*.generate_model` is the usual entry point.
  - The edf/eydf section describes the shared subclassing pattern:
    - `Mode` gets populations, a pump ODE, and `gain()`;
    - the `Model` and `NLSE` subclasses record pump power and populations;
    - `EDF`/`EYDF` subclass `SilicaFiber`;
    - `amplify()` iterates forward and backward passes until they converge;
    - it also covers the `*_wsplice` variants and the data loaders in `edf/utility`.
  - New sections: Documentation layout (`docs/` vs `dev_docs/`), Code session documentation, and Agent delegation.
- **`dev_docs/`** (new):
  - `README.md`: index of sessions;
  - `_template/`: `session_README.md`, `plan.md`, `implementation.md`, `review.md`;
  - this session's folder.

## Deviations from the plan
- Added a `session_README.md` template, which the plan did not list.
- Added a note to CLAUDE.md that `tests/test_light.py` is stale. This was found during verification.

## Commands run
- Checked that every path named in CLAUDE.md exists: all present.
- `git status --short`: only `CLAUDE.md` is modified, plus the untracked `dev_docs/`. Nothing under `docs/` changed.
- `python -m pytest tests/`: failed because pytest is not installed in the `env` conda environment. Running the test functions directly instead gave:
  - `test_utility.test_TFGrid`: PASS
  - `test_light.test_pulse_shapes` and `test_light.test_pulse_properties`: FAIL. The cause is a `TypeError` because `Pulse.Gaussian()` is missing the required `min_time_window` argument. This failure existed before this session.

## Notes for the reviewer
- The delegation rule says `model: "sonnet"` because that is the value the Agent tool expects for Sonnet 5.5.
- Upstream README says new work should branch off `develop`, but this repo works on `main`. CLAUDE.md quotes the upstream guidance without enforcing it.
