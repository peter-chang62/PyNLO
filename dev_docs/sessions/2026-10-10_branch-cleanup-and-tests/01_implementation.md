# 01 Implementation — Stale branch cleanup and outdated tests

Implements [01_plan.md](01_plan.md) in full, including the optional Step 3. Environment: conda env `env` (Python 3.13, no mkl_fft). Every pytest run used `PYTHONDONTWRITEBYTECODE=1 ... -p no:cacheprovider`.

User decisions made at the start of implementation:
- **Step 3:** add both optional regression tests.
- **`peter`:** delete it with no `archive/peter` tag (plan open question 1).
- **Steps 6/8:** the user explicitly confirmed the push and each of the three remote deletions, one by one.

## What changed

### `pynlo/light.py`: `min_points` (commit `f434b8d`)
- Added `+ 1` to `n_min` with a comment: n points span the bandwidth with n − 1 intervals, so `t_window = (n − 1)/Δv`.
- Resized grids now reach `min_time_window`:
  - Δv = 400 THz, T = 0.5 ps: n 200 → 216, t_window 0.4975 → 0.5375 ps.
  - Δv = 400 THz, T = 1 ps: n 400 → 405, t_window 0.9975 → 1.010 ps.
- Grids where the user's `n` is already large enough don't change.

### `tests/test_light.py`
- **`test_min_time_window`** (`f434b8d`): the plan's version, verbatim. I confirmed it fails before the fix (`assert 4.975e-13 >= 5e-13`) and passes after.
- **`test_pulse_shapes` / `test_pulse_properties`** (`32400db`):
  - `min_time_window = 0.5e-12` is passed by keyword to all six constructors, and to the one in the `__main__` block. n = 256 ≥ 216, so the grid isn't resized and nothing prints.
  - The four FWHM checks use `atol=test.dt` instead of `test.dv`. The old tolerance was in Hz, so those checks could never fail.
  - The `v_width` check now uses `v_sigma = 0.5 * v_w.rms`, because `rms` is the full 2σ width since `84452fd`.
- **`test_width_rms_is_full_width`** (`477d4b5`): asserts `t_width().rms ≈ 2σ_t` with rtol 1e-3. The measured ratio is 1.0000000. If someone changes the convention back to σ, the ratio becomes 0.5 and the test fails.
- **Encoding:** the file is LF and is now UTF-8 instead of ASCII, because of the `σ` characters in two comments.

### `tests/test_utility.py`
- **Cleanup** (`3d14979`): removed both no-op `TestResampleT/V.test_all` methods and the debug `print(a_v[-1])` in `TestResampleV.test_r2`.
- **`test_TFGrid_FromFreqRange`** (`477d4b5`) replaces the `#TODO` about `FromFreqRange`. For n ∈ {256, 257, 1000} it checks:
  - `n` is preserved.
  - The span `v_grid[-1] − v_grid[0]` equals `v_max − v_min` exactly.
  - Both endpoints are within `dv` of their targets. Measured offsets were 0.75·dv for n = 256 and 1000, and 0 for n = 257, where `dv` happens to divide `v_min`.
- I left the `rtf_grids` TODO alone; it was out of scope.
- **Line endings:** this file uses **CRLF**. My first scripted edit read it with Python's universal newlines and wrote LF back, which turned a 17-line diff into a 595-line one. I caught that in `git diff --stat` and restored CRLF before committing. Future edits should open the file with `newline=''` or use the Edit tool.

### `README.rst`, `CLAUDE.md` (`057d374`)
- `README.rst`: contributions are now based off `main`, not `develop`.
- `CLAUDE.md`:
  - The Branching bullet now reads "trunk-based on `main`. Short-lived feature branches are fine, but delete them once merged. There is no `develop` branch."
  - Deleted the "two `test_light.py` failures" sentence from the Test coverage paragraph (plan Step 9). I made this edit in the same commit, not the docs commit.

### Git / remote
- Pushed `main` (`63f2888..057d374`).
- Deleted local `peter` with `git branch -d`, so git checked that it was merged.
- Deleted the remote branches `peter`, `add-claude-github-actions-1753769404818` and `add-claude-github-actions-1753769543858`.
- No tags were created.

## Deviations from the plan
- **Step 4 verify command:** `git grep -n "develop\b" -- README.rst CLAUDE.md` is not empty. It matches the new CLAUDE.md bullet's own sentence "There is no `develop` branch.", which is the wording the plan itself specified. The acceptance criterion ("don't mention `develop` as a working branch") is met. The verify command and the specified wording conflict with each other; I kept the wording.
- **Commit grouping:** the CLAUDE.md test-coverage sentence went into commit 5 (README/CLAUDE.md), not commit 6 (dev_docs).
- **Step 0 ahead counts:** they were 109 / 20 / 18 rather than the plan's 107 / 18 / 16, because the two plan commits were made after the plan's counts were taken. The right-hand (ahead) count was 0 for every branch, as required.
- **No `01_review.md`:** at the user's request, the formal review runs in a separate session.

## Commands run
- `git fetch origin --prune`, then `git rev-list --left-right --count main...<b>` for every branch: right-hand count 0 for all.
- `python -m pytest tests/test_light.py::test_min_time_window`: 1 failed before the fix, 1 passed after. Output included "changing n from 2 to 216" and "changing n from 2 to 405".
- `python -m pytest tests/test_light.py`: 3 passed after Step 1.
- `MPLBACKEND=Agg python tests/test_light.py`: the script block runs. Its only output is an expected `log10` divide-by-zero warning from the spectrogram plot.
- `python -m pytest tests/test_utility.py`: 9 passed after Step 2.
- **Final:** `python -m pytest tests/ -v` gave **14 passed** (test_light 4, test_utility 10).
- **Example scripts** (run by a Sonnet subagent with `MPLBACKEND=Agg`; both scripts exercise the resize):
  - `examples/edfa_examples/re_only_5level.py`: "changing n from 256 to 384". t_window = 10.135 ps ≥ 10 ps. No traceback.
  - `examples/pynlo_examples/optical-solitons.py`: "changing n from 2048 to 4050". t_window = 10.1225 ps ≥ 10 ps. All three simulations finished with exit 0.
- **Final git state:**
  - `git branch -a -vv`: only `main`, `origin/main` and `origin/HEAD`.
  - `git ls-remote origin`: `HEAD`, `refs/heads/main` and `refs/pull/1/head` (owned by GitHub, can't be deleted).
  - `git status -sb`: `main...origin/main` with no ahead count. The commit for this document needs one more push.

## Notes for the reviewer
- **Behavior change:** the `min_points` fix makes every resized grid slightly larger than before, e.g. 200 → 216 or 384 vs. the old count. Results from examples that relied on resizing will differ a little from earlier runs. `edf/`/`eydf/` don't call `min_points` directly; they go through the `Pulse` classmethods.
- The FWHM tolerance in `test_pulse_shapes` is now strict: `atol = dt ≈ 2.5 fs`, and the measured error was about 2.1 fs in planning. That is a tight margin, so a change to the grid or the FWHM interpolation could trip it.
- **Still open in OPEN_ISSUES:** the classmethod docstrings don't document `min_time_window`/`alias`, and `mkl_fft` is listed as a hard dependency. Both are out of scope here.
