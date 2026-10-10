# Open Issues

This is a living list of known problems and deferred decisions. Check it at the start of every session.

- **Add** an entry when a session finds something it doesn't fix. Link the session where it was found.
- **Remove** an entry when a session resolves it. Note the fix in that session's review, and link the review in the commit or index row.

---

## Stale tests in `tests/test_light.py`
- **Found:** [2026-10-10_claude-md-refresh](sessions/2026-10-10_claude-md-refresh/01_review.md)
- **Problem:** `test_pulse_shapes` and `test_pulse_properties` call `Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)` without the `min_time_window` argument, which is now required. Both tests fail with a `TypeError`. `tests/test_utility.py` passes.
- **Fix direction:** update the calls to match the current `Pulse` classmethod signatures in `pynlo/light.py`. Check every `Pulse.*` constructor the tests use, not just `Gaussian`.

## Branching policy
- **Found:** [2026-10-10_claude-md-refresh](sessions/2026-10-10_claude-md-refresh/01_review.md)
- **Problem:** `README.rst` says new work should be based on `develop`. That rule came over from the upstream PyNLO repo when this was forked. This fork has no `develop` branch; it has `main`, `peter`, and two stale remote branches named `add-claude-github-actions-*`.
- **Current rule:** work on `main` until the user decides a policy.
- **Decision needed (user):** choose a branching model, then update `README.rst` and CLAUDE.md and clean up the stale branches.
