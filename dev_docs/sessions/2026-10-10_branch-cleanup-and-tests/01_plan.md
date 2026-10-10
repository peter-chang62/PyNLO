# 01 Plan — Stale branch cleanup and outdated tests

## Goal
1. Remove stale branches and the inherited "base work on `develop`" rule, so `main` is the only branch.
2. Make `python -m pytest tests/` pass by fixing the outdated `tests/test_light.py`, after confirming the failures are test drift, not library bugs.

This session is planning only. A separate implementation session executes the steps below.

## Findings

### Environment
- Interpreter: `/Users/peterchang/miniconda3/envs/env/bin/python` (conda env `env`, Python 3.13.11). There is no `env/` folder in the repo; `pyrightconfig.json`'s `venv: env` refers to this conda env.
- Installed: numpy 2.3.5, scipy 1.16.3, numba 0.63.1, pytest 9.1.1, matplotlib 3.11.1. **`mkl_fft` is not installed.** `pynlo/utility/fft.py:14-48` falls back to scipy/numpy and prints `NOT USING MKL FOR FFT'S IN PYNLO`. Tests don't need mkl_fft.
- `site-packages/{pynlo,edf,eydf,blit.py,clipboard.py}` are symlinks into this repo, so `import pynlo` always resolves to the working tree. Both `python -m pytest tests/` and bare `pytest tests/` work.

### Git state (commands: `git branch -a -vv`, `git remote -v`, `git ls-remote origin`, `git rev-list --left-right --count main...<b>`, `git merge-base --is-ancestor <b> main`, `git branch -a --merged main`, `git worktree list`, `git worktree prune --dry-run -v`, `git remote prune origin --dry-run`, `git tag -l`, `git stash list`, `gh pr list --state all`)

| Branch | Ahead / behind `main` | Last commit | Merged into `main` | Recommendation |
|---|---|---|---|---|
| `main` (local) | — | 2026-10-10, peter chang, `03c3429` | — | Keep. **1 commit ahead of `origin/main`** (unpushed `03c3429`), so push it. |
| `origin/main` | 0 / 1 | 2026-10-10, `63f2888` | yes | Keep (updated by push). |
| `peter` (local, tracks `origin/peter`) | 0 / 107 | 2023-06-28, peter chang, `fa7eecd` "changed cmd back to ctrl for clipboard.py" | yes (ancestor of `main`) | Delete. |
| `origin/peter` | 0 / 107 | same `fa7eecd` | yes | Delete (remote, needs confirmation). |
| `origin/add-claude-github-actions-1753769404818` | 0 / 18 | 2025-07-28, peter chang, `a314b4d` "CLAUDE.md" | yes | Delete (remote, needs confirmation). |
| `origin/add-claude-github-actions-1753769543858` | 0 / 16 | 2025-07-29, peter-chang62, `5c493bb` "Claude Code Review workflow" | yes, via PR peter-chang62/PyNLO#1 (MERGED 2025-07-29) | Delete (remote, needs confirmation). |
| `develop` | — | — | — | **Does not exist** locally or on `origin` (`git cat-file -t develop` → not a valid object; absent from `ls-remote`). Nothing to delete. |

- No tags, no stashes, one worktree (the main checkout), nothing for `git worktree prune` or `git remote prune origin` to remove.
- Every non-`main` branch has **zero unique commits**: each tip is an ancestor of `main`. Deleting them loses no work, so a backup tag isn't required. One is offered as optional below.
- `refs/pull/1/head` on GitHub points at `5c493bb`. GitHub owns that ref and it can't be deleted. It is harmless.
- No CI config: `.github/` is gone since `03c3429`.
- References to `develop`, found with `git grep -n -iE "develop|branch"`:
  - `README.rst:32`: "New additions should be based off the `develop` branch."
  - `CLAUDE.md:59`: the Branching bullet, which points at the open issue.
  - `dev_docs/OPEN_ISSUES.md` "Branching policy" entry.
  - Historical mentions in `dev_docs/sessions/2026-10-10_claude-md-refresh/`. Leave those alone; they are history.
  - No Sphinx source under `docs/` mentions `develop`.

### Tests (command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ -p no:cacheprovider -v -rA`)
Result: **2 failed, 11 passed in 0.77s.**
```
FAILED tests/test_light.py::test_pulse_shapes - TypeError: Pulse.Gaussian() missing 1 required positional argument: 'min_time_window'
FAILED tests/test_light.py::test_pulse_properties - TypeError: Pulse.Gaussian() missing 1 required positional argument: 'min_time_window'
```
All 11 tests in `tests/test_utility.py` pass.

#### Failure 1 (both tests): missing `min_time_window`. The test is outdated.
- Commit `f2ef918` (2023-06-22, Peter, "...made changes to Pulse class to automatically adjust the number of points if needed") made `min_time_window` a required positional argument of every `Pulse` classmethod. If `ceil((v_max - v_min) * min_time_window)` rounded up to a fast FFT length is bigger than `n`, the method raises `n` and prints "changing n from ...". The same commit changed the classmethod default `alias` from 1 (through `**kwargs` → `__init__`) to 2.
- Current signatures in `pynlo/light.py`:
  - `Gaussian(cls, n, v_min, v_max, v0, e_p, t_fwhm, min_time_window, m=1, alias=2)` at line 201.
  - `Sech` at 249, `Parabolic` at 296, `Lorentzian` at 343: `(..., t_fwhm, min_time_window, alias=2)`.
  - `CW(cls, n, v_min, v_max, v0, p_avg, min_time_window, alias=2)` at 389.
  - `FromPowerSpectrum(..., min_time_window, ...)` at 129.
  - `min_points()` is at line 48.
- `tests/test_light.py` was last changed in `515e57f` (2023-04-06), before `f2ef918`. The change to the library was intentional, so the tests are outdated. This is not a library bug.

#### Failure 2 (hidden behind failure 1): `test_pulse_properties` line 96 uses an old RMS definition. The test is outdated.
I found this by running a scratch copy of the test with `min_time_window` added:
```
assert np.isclose(np.log(4)**0.5 / (2*pi*v_w.rms), t_w.fwhm, atol=test.dt)
AssertionError: lhs 5.0e-14 vs fwhm 1.02e-13
```
- Commit `84452fd` (2023-05-26, cdfredrick, "NLSE + Refactor") changed `v_width`, `t_width` and `autocorrelation` from `rms = sqrt(var)` to `rms = 2*sqrt(var)`. Their docstrings now say "The **full** root-mean-square width" (`pynlo/light.py:632, 665, 886, 917, 999, 1034`).
- The same commit updated the only library consumer, `spectrogram`, to `v_sigma = 0.5 * self.v_width().rms` (line 1118). That shows the change was deliberate. The test still assumes `rms = σ`. For a transform-limited Gaussian, FWHM_t = sqrt(ln 4) / (2π σ_v), so the test should use `σ_v = 0.5 * v_w.rms`.
- Verified in scratch: with the fix, lhs = 1.000e-13 and fwhm = 1.021e-13. That is within `atol = dt = 2.49e-15`, so it passes.
- The `ac.rms == sqrt(2) * t_w.rms` check (line 98) holds under either definition, because both sides scale by 2. Verified: ratio 0.9999998 at n=256, which is within default rtol.

#### Stale-but-passing issues
- `tests/test_light.py:37, 44, 51, 58`: `np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)` compares a time against a tolerance in Hz (`dv` is about 1.6e12). The check always passes, so it tests nothing. It should be `atol=test.dt`. Verified that all four shapes pass with `atol=test.dt`: error is 2.10e-15 and dt is 2.49e-15.
- `tests/test_utility.py:25` `TestResampleT.test_all` and `:122` `TestResampleV.test_all` only run code under `if __name__ == "__main__":`. Under pytest they are empty tests that always pass. pytest already runs `test_r1/r2/c1/c2` individually.
- `tests/test_utility.py:159`: a debug `print(a_v[-1])` in `TestResampleV.test_r2`.
- `tests/test_utility.py:300, 302`: TODOs say `TFGrid.rtf_grids` and `TFGrid.FromFreqRange` are untested.
- No test uses a removed API. Every attribute the tests use exists on `Pulse` and `TFGrid` (checked with `hasattr`).

## Decisions
- **`develop`:** it doesn't exist in this clone or on `origin`, so the user's question about whether it is out of sync with `main` doesn't apply. The only remaining work is removing the README.rst rule. The upstream repo's `develop` is not a remote of this fork and is out of scope.
- **Branching policy:** trunk-based on `main`, which CLAUDE.md already does in practice. Short-lived feature branches are allowed but get deleted after merging.
- **Test fix rather than library fix** for both failures. The git history shows both library changes were intentional API/semantics changes with matching docstring and consumer updates.
- **`min_time_window` value:** use `0.5e-12`, passed as a keyword. With `v_max - v_min = 400 THz` this needs 200 points, which is ≤ 256, so `n` stays 256. That keeps the original test grid, avoids the "changing n" print, and leaves `t_window` at about 0.64 ps (6× the 100 fs pulse). A value of `1e-12` also passes, but it silently resizes to n=400.
- **No backup tags needed:** no branch has unique commits. An optional tag is offered for `peter` in case the user wants the name kept as a historical marker.

## Implementation steps

Run everything from `/Users/peterchang/Github/PyNLO` on `main`, in the conda env `env`. Steps marked ⚠️ **CONFIRM** are remote or destructive. Ask the user for an explicit yes before each one, separately.

### Step 0: Preconditions
```bash
git status --short          # expect clean
git fetch origin --prune
git branch -a -vv
```
Verify that the branch table above still holds: `git rev-list --left-right --count main...origin/peter` gives `N 0`, and the same for both `origin/add-claude-github-actions-*` branches. If any branch now shows a non-zero right-hand (ahead) count, **stop** and report it to the user.

### Step 1: Fix `tests/test_light.py`
Add a shared variable and pass it by keyword in every constructor call:
- In `test_pulse_shapes` (after line 30) and `test_pulse_properties` (after line 75), add `min_time_window = 0.5e-12`.
- Lines 33, 40, 47, 54: append `, min_time_window=min_time_window` to the `Gaussian`, `Sech`, `Parabolic` and `Lorentzian` calls.
- Line 62: `light.Pulse.CW(n, v_min, v_max, v_0, p_p, min_time_window=min_time_window)`.
- Line 76: same as the `Gaussian` call.
- Lines 37, 44, 51, 58: change `atol=test.dv` to `atol=test.dt`, which fixes the unit bug.
- Line 96: change it to
  ```python
  v_sigma = 0.5 * v_w.rms  # v_width().rms is the full (2σ) rms width
  assert np.isclose(np.log(4)**0.5 / (2*pi*v_sigma), t_w.fwhm, atol=test.dt)
  ```
- Check the `if __name__ == "__main__":` exploratory block (lines 101-145) for `Pulse.*` calls and add `min_time_window` the same way, so the file still runs as a script.

Verify:
```bash
python -m pytest tests/test_light.py -v
python tests/test_light.py   # script block still runs; close plots
```
Expected: 2 passed.

### Step 2: Clean up stale bits in `tests/test_utility.py`
- Delete the two no-op `test_all` methods at lines 25-32 and 122-129. Also update the `__main__` block at the bottom if it calls them; currently it only calls `test_TFGrid()`.
- Remove the debug `print(a_v[-1])` at line 159.

Verify: `python -m pytest tests/test_utility.py -v` gives 9 passed (11 minus the 2 `test_all`).

### Step 3 (optional, high-value regression tests)
Add these to `tests/test_light.py` only if the user wants them:
1. `test_min_time_window_resizes_grid`: `Pulse.Gaussian(256, 100e12, 500e12, 300e12, 1, 100e-15, min_time_window=1e-12)` should give `p.n == 400` and `p.t_window >= 1e-12`. With `min_time_window=0.5e-12`, `p.n == 256`. This pins the `f2ef918` behavior.
2. `test_width_rms_is_full_width`: for a Gaussian, `t_w.rms ≈ 2*sigma_t`, where `sigma_t = t_fwhm / (2*sqrt(2*ln 2))`. Use `rtol=1e-3`. This pins the `84452fd` convention so any future change is deliberate.
3. Add to `tests/test_utility.py`: `test_TFGrid_FromFreqRange`, which checks that `v_grid[0] ≈ v_min`, `v_grid[-1] ≈ v_max`, and `n` is preserved. This resolves the TODO at line 302.

Verify: `python -m pytest tests/ -v`, all pass.

### Step 4: Fix README.rst and CLAUDE.md
- `README.rst:32`: replace "New additions should be based off the `develop` branch." with "New additions should be based off the `main` branch."
- `CLAUDE.md:59`: replace the Branching bullet with: "- Branching: trunk-based on `main`. Short-lived feature branches are fine, but delete them once merged. There is no `develop` branch."

Verify: `git grep -n "develop\b" -- README.rst CLAUDE.md` returns nothing.

### Step 5: Commit the local changes (suggested breakdown below)
Verify: `python -m pytest tests/` passes, and `git status` is clean.

### Step 6: ⚠️ CONFIRM, push `main`
```bash
git push origin main
```
This pushes `03c3429` plus the commits from Step 5. Verify: `git status -sb` shows `main...origin/main` with no ahead count.

### Step 7: Delete the local stale branch (safe, no confirmation needed)
Optional backup first, only if the user wants the name kept as a marker:
```bash
git tag archive/peter fa7eecd
```
Then delete:
```bash
git branch -d peter
```
`-d` refuses to delete unmerged branches, which is a safety check; `peter` is merged. Verify with `git branch` (only `main` remains).

### Step 8: ⚠️ CONFIRM EACH, delete remote branches
Ask before each command separately:
```bash
git push origin --delete peter
```
```bash
git push origin --delete add-claude-github-actions-1753769404818
```
```bash
git push origin --delete add-claude-github-actions-1753769543858
```
If the Step 7 tag was created and the user wants it on GitHub, that needs its own ⚠️ CONFIRM:
```bash
git push origin archive/peter
```
Verify:
```bash
git fetch origin --prune
git branch -a -vv
git ls-remote origin
```
Expected: only `main`/`origin/main`/`origin/HEAD` (+ GitHub's `refs/pull/1/head`).

### Step 9: Session docs
Write `01_implementation.md` and `01_review.md` in the session folder.
- Remove the "Stale tests in `tests/test_light.py`" and "Branching policy" entries from `OPEN_ISSUES.md`.
- Update the session README and the index row to `done`.
- Update CLAUDE.md's "Test coverage" paragraph, which mentions "two `test_light.py` failures", so it no longer claims failing tests.

## Suggested commits
1. `Fix outdated test_light.py for min_time_window and full-width rms` (Step 1)
2. `Remove no-op test_all methods and debug print from test_utility.py` (Step 2)
3. `Add regression tests for grid resizing, rms width, FromFreqRange` (Step 3, optional)
4. `Drop inherited develop-branch rule; document trunk-based main` (Step 4: README.rst, CLAUDE.md)
5. `dev_docs: branch-cleanup-and-tests implementation and review` (Step 9)

Branch deletions are not commits.

## Needs user confirmation
- Step 6: push `main`.
- Step 8: each remote branch deletion, and the optional tag push.
- Step 3: whether to add the optional tests.

## Open questions
1. **`peter` branch:** it is fully merged and was last touched 2023-06-28. Is it used as a personal working branch on another machine, or by any external link? If not, delete it. A backup tag (`archive/peter`) is optional, since no commits would be lost either way.
2. **mkl_fft:** keep it as a hard dependency in `pyproject.toml`, or make it optional (`[project.optional-dependencies]`), given it isn't installed in the dev env and the code falls back? This is out of scope here and logged in OPEN_ISSUES.

## Dead ends
- I looked for `develop` to compare against `main`. It doesn't exist locally or remotely, so there was nothing to diff.
- `min_time_window=1e-12` in the test passes too, but it silently resizes the grid. I rejected it in favor of a window that preserves the original `n`.

## Acceptance criteria
- `git branch -a` shows only `main`, `origin/main`, `origin/HEAD`. `git ls-remote origin` shows only `main` (+ `refs/pull/1/head`).
- `main` is in sync with `origin/main`.
- `python -m pytest tests/` passes with 0 failures.
- `README.rst` and `CLAUDE.md` don't mention `develop` as a working branch.
- `OPEN_ISSUES.md` has no "Stale tests" or "Branching policy" entries. Session docs and the index are updated.
