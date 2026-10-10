# Open Issues

This is a living list of known problems and deferred decisions. Check it at the start of every session.

- **Add** an entry when a session finds something it doesn't fix. Link the session where it was found.
- **Remove** an entry when a session resolves it. Note the fix in that session's review, and link the review in the commit or index row.

---

## Pulse classmethod docstrings omit `min_time_window` and `alias`
- **Found:** [2026-10-10_branch-cleanup-and-tests](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md)
- **Problem:** `Pulse.Gaussian/Sech/Parabolic/Lorentzian/CW/FromPowerSpectrum` in `pynlo/light.py` take a required `min_time_window` (s) and `alias=2`, but none of their docstrings document either. The classmethod default `alias=2` also differs from `Pulse.__init__`/`TFGrid` (`alias=1`). The `FromPowerSpectrum` docstring calls `n`, `v_min` and `v_max` "optional" although they are required.
- **Fix direction:** add both parameters to each docstring. `min_time_window` is the minimum time window; `n` is raised to `next_fast_len(ceil(Δv·min_time_window))` if needed.

## `mkl_fft` listed as a hard dependency but not installed
- **Found:** [2026-10-10_branch-cleanup-and-tests](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md)
- **Problem:** `pyproject.toml` requires `mkl_fft`, but the dev env (conda `env`, Python 3.13, macOS) doesn't have it. `pynlo/utility/fft.py` silently falls back to scipy/numpy and prints `NOT USING MKL FOR FFT'S IN PYNLO`.
- **Decision needed (user):** make `mkl_fft` an optional dependency, or install it where it's available.
