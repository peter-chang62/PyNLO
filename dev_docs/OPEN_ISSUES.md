# Open Issues

This is a living list of known problems and deferred decisions. Check it at the start of every session.

- **Add** an entry when a session finds something it doesn't fix. Link the session where it was found.
- **Remove** an entry when a session resolves it. Note the fix in that session's review, and link the review in the commit or index row.

---

## Stale tests in `tests/test_light.py`
- **Found:** [2026-10-10_claude-md-refresh](sessions/2026-10-10_claude-md-refresh/01_review.md)
- **Problem:** `test_pulse_shapes` and `test_pulse_properties` call `Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)` without the `min_time_window` argument, which is now required. Both tests fail with a `TypeError`. `tests/test_utility.py` passes.
- **Fix direction:** update the calls to match the current `Pulse` classmethod signatures in `pynlo/light.py`. Check every `Pulse.*` constructor the tests use, not just `Gaussian`.
- **Update (2026-10-10):** there is a second failure behind the `TypeError`. `test_pulse_properties` line 96 assumes `v_width().rms` is σ, but since `84452fd` it is the full 2σ width. The `atol=test.dv` checks on lines 37/44/51/58 are also vacuous because the tolerance is in Hz.
- **Plan:** [2026-10-10_branch-cleanup-and-tests/01_plan.md](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md) (Step 1). Remove this entry once that is implemented.

## Branching policy
- **Found:** [2026-10-10_claude-md-refresh](sessions/2026-10-10_claude-md-refresh/01_review.md)
- **Problem:** `README.rst` says new work should be based on `develop`. That rule came over from the upstream PyNLO repo when this was forked. This fork has no `develop` branch; it has `main`, `peter`, and two stale remote branches named `add-claude-github-actions-*`.
- **Current rule:** work on `main` until the user decides a policy.
- **Decision needed (user):** choose a branching model, then update `README.rst` and CLAUDE.md and clean up the stale branches.
- **Plan:** [2026-10-10_branch-cleanup-and-tests/01_plan.md](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md) (Steps 4, 6-8). Trunk-based on `main`. All non-main branches are fully merged (zero unique commits) and are slated for deletion, with user confirmation for each remote deletion. Remove this entry once that is implemented.

## `min_points` off-by-one: resized grids fall just short of `min_time_window`
- **Found:** [2026-10-10_branch-cleanup-and-tests](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md)
- **Problem:** `t_window = (n-1)/(v_max - v_min)`, but `min_points` (`pynlo/light.py:48`) uses `ceil(V*T)`. As a result, `Pulse.Gaussian(2, 100e12, 500e12, ..., min_time_window=1e-12)` gives `t_window = 0.9975 ps`.
- **Plan:** [01_plan.md Step 0b](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md). Add `+ 1` and the `test_min_time_window` regression test; the user approved this fix. Remove this entry once that is implemented.

## Pulse classmethod docstrings omit `min_time_window` and `alias`
- **Found:** [2026-10-10_branch-cleanup-and-tests](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md)
- **Problem:** `Pulse.Gaussian/Sech/Parabolic/Lorentzian/CW/FromPowerSpectrum` in `pynlo/light.py` take a required `min_time_window` (s) and `alias=2`, but none of their docstrings document either. The classmethod default `alias=2` also differs from `Pulse.__init__`/`TFGrid` (`alias=1`). The `FromPowerSpectrum` docstring calls `n`, `v_min` and `v_max` "optional" although they are required.
- **Fix direction:** add both parameters to each docstring. `min_time_window` is the minimum time window; `n` is raised to `next_fast_len(ceil(Δv·min_time_window))` if needed.

## `mkl_fft` listed as a hard dependency but not installed
- **Found:** [2026-10-10_branch-cleanup-and-tests](sessions/2026-10-10_branch-cleanup-and-tests/01_plan.md)
- **Problem:** `pyproject.toml` requires `mkl_fft`, but the dev env (conda `env`, Python 3.13, macOS) doesn't have it. `pynlo/utility/fft.py` silently falls back to scipy/numpy and prints `NOT USING MKL FOR FFT'S IN PYNLO`.
- **Decision needed (user):** make `mkl_fft` an optional dependency, or install it where it's available.
