# 01 Review — CLAUDE.md refresh and session-docs setup

## Verification results
- Every path in CLAUDE.md exists.
- `docs/` (Sphinx) is untouched.
- `test_utility` passes. Both `test_light` tests fail; this failure existed before this session (see implementation notes). This session made no Python changes.

## Issues found
- `tests/test_light.py` does not match the current `Pulse.Gaussian` signature (`min_time_window` is now required).
- pytest was not installed in the `env` conda environment (the user has since installed it).

## Outcome
Done. No further iteration is needed.

## Follow-up (same day)
- The user installed pytest. `python -m pytest tests/` gives 11 passed, 2 failed (the known `test_light` failures).
- The user chose to defer two items, the stale `test_light.py` tests and the branching policy. Both moved to [OPEN_ISSUES.md](../../OPEN_ISSUES.md).
- Added `dev_docs/OPEN_ISSUES.md` as the single living list of unresolved issues. CLAUDE.md now tells every session to read it at the start and update it at the end.

## Open items for future sessions
- See [OPEN_ISSUES.md](../../OPEN_ISSUES.md).
