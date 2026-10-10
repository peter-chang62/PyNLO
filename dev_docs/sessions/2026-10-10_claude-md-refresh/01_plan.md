# 01 Plan — CLAUDE.md refresh and session-docs setup

## Goal
1. CLAUDE.md was last committed in July 2025 and no longer matched the repo. It left out `ydf/`, the edf splice-loss variants, `edf/utility/` data loaders, `pyrightconfig.json`, and the `@claude` GitHub workflow. Re-run `/init` and correct it.
2. Adopt the rules from the user's `ai_prompts.md`:
   - Document every code session as plan → implementation → review. Iterations can repeat, and an index README should help future agents find past sessions.
   - When running on Opus, delegate menial subtasks to Sonnet 5.5 subagents. Opus orchestrates and writes the final markdown docs.

## Approach
- **Folder location:** `ai_prompts.md` asked for a `docs/` folder, but `docs/` already holds the Sphinx build (`docs/source`, `docs/build`, Makefile, an `index.html` redirect). Three options were weighed:
  - New top-level `dev_docs/`: chosen by the user.
  - `docs/sessions/`: rejected because it mixes agent notes with the published-docs tooling.
  - Move Sphinx to `docs/sphinx/`: rejected because it means rewriting the build scripts and the redirect.
- **Layout:** one folder per session, named `YYYY-MM-DD_slug`. Files carry an iteration prefix (`NN_plan.md`, `NN_implementation.md`, `NN_review.md`). A plan → impl → review loop then sorts chronologically without nesting folders.
- **CLAUDE.md:** keep the content that is still accurate and tighten the architecture section around the `Pulse`/`Mode` → `Model` → `simulate` pipeline. Document how edf/eydf subclass the core classes. Add sections for documentation layout, session docs, and agent delegation.

## Files to touch
- `CLAUDE.md`
- `dev_docs/README.md`, `dev_docs/_template/*`, `dev_docs/sessions/2026-10-10_claude-md-refresh/*`
- `docs/` must remain untouched.

## Verification
- Check that every path named in CLAUDE.md exists.
- `git status` shows no changes under `docs/`.
- `python -m pytest tests/` passes.
