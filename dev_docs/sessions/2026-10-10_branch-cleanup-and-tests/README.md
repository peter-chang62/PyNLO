# Stale branch cleanup and outdated tests

- **Date:** 2026-10-10
- **Status:** planned. The plan is written; implementation is pending in a separate session.
- **Summary:** This was a planning-only session.
  - **Branches:** every non-`main` branch (`peter`, plus the two `add-claude-github-actions-*` remotes) is fully merged into `main` with zero unique commits. `develop` does not exist.
  - **Tests:** both `test_light.py` failures are test drift, not library bugs. One is the `min_time_window` argument added in `f2ef918`. The other, previously hidden, is the full-width (2σ) rms convention from `84452fd`.
  - **Plan:** delete the stale branches (remote deletions need user confirmation), base contributions on `main` instead of `develop` in README.rst, and fix the tests.

## Iterations
| # | Plan | Implementation | Review | Outcome |
|---|------|----------------|--------|---------|
| 01 | [plan](01_plan.md) | — | — | plan approved; awaiting implementation |

## Key files touched
- `dev_docs/` only: this session folder, the index row, and `OPEN_ISSUES.md`.
- Planned for the implementation session: `tests/test_light.py`, `tests/test_utility.py`, `README.rst`, `CLAUDE.md`, plus remote branch deletions.

## Notes for the implementing agent
- Follow [01_plan.md](01_plan.md) step by step.
- Steps 6 and 8 (push, remote deletions) each need explicit user confirmation.
- Open question 1 in the plan (whether to keep the `peter` branch) must be answered by the user before Step 7/8.
- I verified the test fixes against a scratch copy of `test_light.py` (outside the repo). The exact values used are in the plan's Findings.

## Related sessions
- [2026-10-10_claude-md-refresh](../2026-10-10_claude-md-refresh/README.md) logged both issues this session plans to resolve.
