# dev_docs — Code Session History

This folder records every code-agent session on this repo so future sessions can pick up context quickly. It is separate from `docs/`, which holds the Sphinx user/API documentation.

**Start here:** read [OPEN_ISSUES.md](OPEN_ISSUES.md), scan the index below, then open the session folders related to your task.

## Conventions

- One folder per session: `sessions/YYYY-MM-DD_<short-slug>/`.
- Each session runs **plan → implementation → review**, and may loop. Files are prefixed by iteration number: `01_plan.md`, `01_implementation.md`, `01_review.md`, `02_plan.md`, ...
- Each session folder has a `README.md` with goal, status, iteration table, and key files touched.
- Copy starting files from `_template/`.
- Add a row to the index when a session starts; update its status and summary when it ends.

Full rules live in the root `CLAUDE.md` under "Code session documentation".

## Session index

| Date | Session | Status | Summary | Key files |
|------|---------|--------|---------|-----------|
| 2026-10-10 | [claude-md-refresh](sessions/2026-10-10_claude-md-refresh/README.md) | done | Refreshed CLAUDE.md for current repo layout; set up `dev_docs/` session logging and the Opus→Sonnet delegation rule. | `CLAUDE.md`, `dev_docs/` |
