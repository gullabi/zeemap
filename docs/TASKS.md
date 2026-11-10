# Build order (tasks)

**Note:** this list is the record of the initial build (2026-10-17 to 2026-11); new work is tracked in GitHub issues.
It is kept because the plan check (`docs/review/lint_plan.py`) uses it.
**Status:** M0-M5 and M7 (the changes of the design review, decisions #22-#29) are built. Open: T2.3 (phase smearing,
absolute-flux mode), and T6.4 and T7.12 (fits of real data, done outside this repository).
**Who this is for:** whoever builds zeemap, person or agent.
**What you will know afterwards:** the tasks in order, what each needs, and the command that shows it is done.

## How to use this list

- Do the milestones in order (M0, M1, ...). Inside a milestone, tasks whose "needs" are done can run in parallel.
- Each task should fit in one working session. If it does not, split it and update this list.
- Before writing code, write the tests for the check items the task names, and see them fail. Then write the code
  until they pass. Test names start with the item label, for example `test_v2c_teff_halfway`.
- A task is done when its "done when" command passes **and** `tools/check.sh 1` still passes. Then ask for review
  ([GATES.md](GATES.md)).
- A task that "needs" a decision (Dn) waits until it is decided (DECISIONS.md, "Open questions"). So does any part of
  a requirement that carries a `[NEEDS DECISION]` marker.
- Out of scope for every task: anything not named in its row. Note other problems in a GitHub issue, instead of fixing
  them on the side.
- Items of a check that belong to a later task do not exist yet; running the whole check runs only the items that
  exist. That is why every "done when" names its items.

## The check runner (built in T0.2)

`tools/check.sh <tier> [--fast] [V<k>[<letter>] ...] [-- pytest arguments]`
- With only a tier, it runs every check of that tier and the tiers below (tier 1: V1-V8 and the V14 tests; tier 2:
  V10-V13; tier 3: V15-V19). V9 is retired.
- With items, it runs only those, selected by test name (`test_v2c_...`). Asking for an item of a higher tier than
  `<tier>` is an error, and so is an item that selects no test, also next to items that do (`tests/conftest.py`
  names each such item).
- `--fast` leaves out the tier-1 tests marked `slow` (the end-to-end fits; pytest `-m "tier1 and not slow"`) and
  prints `SUMMARY tier 1 (fast): ...` instead of `SUMMARY tier 1: ...`; the other tiers are not affected. The full
  tier 1 takes about 10 min on 4 idle cores, `--fast` at most 2 min (VERIFICATION.md section 2; decision #38).
- It prints one line per tier it ran: `SUMMARY tier <n>: <p> passed, <f> failed, <e> errors, <s> skipped`, and exits
  with 1 if anything failed or errored, if a named item selected no test, or (in tier 2 and above) if anything was
  skipped. With no item named and no tests in the repository, it prints zeros and exits with 0 (pytest's "no tests"
  exit code 5 is mapped to 0 only in that case).
- Every test carries a marker `tier1`, `tier2` or `tier3`, which must match its check's tier; a test whose marker and
  name disagree fails. The marker `slow` comes in addition, on tier-1 tests only.
- **Check items not built:** V2h, V3j (decision #38). T2.3's absolute-flux mode and phase smearing
  were never built. `python3 docs/review/lint_plan.py .` fails when a "done when" command names
  an item that has no test by name (`test_v2h_...`; for an item without a letter, any test of that check). It reports
  such an item only as a notice in three cases: the item is listed on this line, its check is waived by decision
  #35, or it is in tier 3.

## M0. Skeleton

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T0.1 | package layout: `src/zeemap/`, `tests/`, `examples/`, `tools/`; `pyproject.toml` (build backend hatchling; optional-dependencies group `dev`); `CHANGELOG.md`; lock file `requirements.lock` made by `uv pip compile --python-version 3.12 --generate-hashes --extra dev pyproject.toml` (Linux x86-64). Runtime: numpy, scipy, numba, emcee. Group `dev`: pytest, hypothesis, tomli-w, and optionally mutmut (all pre-approved) | ENG-1 | none | `uv venv -p 3.12 .venv && uv pip sync --python .venv/bin/python requirements.lock && uv pip install --python .venv/bin/python --no-deps -e . && .venv/bin/python -c "import zeemap"` succeeds on a clean machine |
| T0.2 | `tools/check.sh` as described above; markers registered in `pyproject.toml` with `--strict-markers` | gates | T0.1 | `tools/check.sh 1` prints `SUMMARY tier 1: 0 passed, 0 failed, 0 errors, 0 skipped` and exits with 0; `tools/check.sh 1 V1a` exits with 1 (no test selected) |
| T0.3 | synthetic library generator `tests/synth.py` (VERIFICATION.md section 4), and a reader and writer for the library format (FORMATS.md section 1), with the size check | LIB-1 | T0.2 | `tools/check.sh 1 V2e` passes |
| T0.4 | `tools/scan.py`, `tools/scan_allow.txt`, `tools/forbidden_identifiers.txt` (empty at first; the maintainers fill it) | ENG-4, ENG-5 | T0.2 | `tools/check.sh 1 V8` passes |
| T0.5 | CI: `.github/workflows/tier1.yml` (action `astral-sh/setup-uv`, then the T0.1 commands, then `tools/check.sh 1`) on every push; also `tools/hooks/pre-push`, an optional git hook that ran `tools/check.sh 1` (removed at the public release, where CI and branch protection do this) | ENG-1 | T0.2 | a pushed test branch shows the CI result on GitHub |

## M1. Core model

**Order:** M0, then T1.0. It was decided not to use the prototype (decision #14, 2026-10-18), so T1.1
is skipped and T1.2-T1.7 are written from this plan alone. T1.1 stays in the table as a record.

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T1.0 | configuration reader: the key table of FORMATS.md section 3 in `src/zeemap/defaults.toml`, with comments; strict about unknown keys; paths resolved as FORMATS.md says | CFG-1, CFG-2, CFG-3 | M0 | `tools/check.sh 1 V1k V4c` passes |
| T1.1 (skipped, decision #14) | copy the prototype in (DESIGN.md section 8). Replace absolute paths by `$ZEEMAP_DATA` and relative paths; sort its tests (those that read real data become tier 2; build tier-1 versions on synthetic data). Re-read every function against SPEC.md and DESIGN.md and record the result per module in `docs/REVIEW_PROTOTYPE.md` (module, checked against, differences found) | ENG-4, ENG-5 | T1.0, D11 | `tools/check.sh 1` passes; `docs/REVIEW_PROTOTYPE.md` lists every module |
| T1.2 | library reader in full: Teff interpolation, slice caching, range errors | LIB-1, LIB-2, LIB-3, LIB-7 | T0.3, T1.0 | `tools/check.sh 1 V2c V2f V4a` passes |
| T1.3 | surfaces (Fibonacci) | SURF-1 | T0.2 | `tools/check.sh 1 V1b V1c` passes |
| T1.4 | field models "dqo" and "dqo_aligned"; vector parameters; canonical form | FIELD-1, FIELD-2, FIELD-3, PAR-1, PAR-2 | T1.0 | `tools/check.sh 1 V1a V1f V3a V3b V3c V3f` passes (their field-level tests, `test_v3a_field_...` and so on) |
| T1.5 | forward model: visibility, limb darkening, lookup, B outside the library, convolution, resampling, V/I, phase offset, batches; and the plain-numpy reference | FWD-1 to FWD-5, FWD-10, FWD-11, LIB-5, LIB-6 | T1.2, T1.3, T1.4 | `tools/check.sh 1 V1d V1e V1n V2a V2b V2d V3a V3c V3d V3f V5a` passes (V3a, V3c and V3f now at the spectrum level) |
| T1.6 | data reader, errors, windows, excluded ranges, weights, per-phase switches | DATA-1 to DATA-4, DATA-6 | T1.0 | `tools/check.sh 1 V1i V1j V4b` passes |
| T1.7 | `eval` (`zeemap eval <config> -o <dir>` evaluates the parameters' `value`s and writes `eval.json` and `spectra/`); the compatibility features of the first build were removed 2026-11-08 | CLI-2 | T1.5, T1.6 | `tools/check.sh 1 V7a` passes |

## M2. Smooth model and likelihood

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T2.1 | plain chi2 and the full likelihood (DESIGN.md section 5), each nuisance term switchable; invalid models | LIKE-1 to LIKE-4, LIKE-6 to LIKE-9 | M1 | `tools/check.sh 1 V1g V1h V1m V3e V3g V3h V3i V4c V4d V5b` passes |
| T2.2 | measure the memory at start-up; replace whatever dominates | ENG-6, FWD-12 | T2.1 | `tools/check.sh 2 V12` passes |
| T2.3 (open: phase smearing and the absolute-flux mode were not built; the shift was built in T7.11; decision #38) | wavelength shift, phase smearing, absolute-flux mode, multipoles up to l = 5 | FWD-13, FWD-14, DATA-7, FIELD-4 | T2.1 | `tools/check.sh 1 V1a V2g V2h V3j` passes, with V1a covering the multipoles |
| T2.4 | interpolation in B with a continuous first derivative (cubic in B), as an option; planned now because the research expects linear interpolation to fail V11a | FWD-15 | T2.1 | `tools/check.sh 1 V1e` passes for both interpolations, and the option `model.interp_b` (`"linear"` or `"pchip"`) is in `defaults.toml` |

## M3. Search

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T3.1 | start distribution, polish, restart rounds, fingerprints and grouping into modes, the mode band (DESIGN.md section 3); `fit` (search part) | SEARCH-1, SEARCH-2, SEARCH-3, SEARCH-7, PAR-4, CLI-3 | M2, T3.2 | `tools/check.sh 1 V3k V7c` passes |
| T3.2 | `inject` | CLI-4 | T2.1 | `tools/check.sh 1 V7b` passes |
| T3.3 | process pool with one seed stream per start | SEARCH-5, ENG-2, ENG-3 | T3.1 | `tools/check.sh 1 V6` passes |
| T3.4 | search statistics, the warning, re-ranking at the error-bar patch count | SEARCH-4, SEARCH-8 | T3.1 | `tools/check.sh 1 V4g` passes |
| T3.5 | planted test problem (hit rate 0.5-2 %, measured with 500 starts); set the default budget so a run succeeds with probability at least 0.99 | SEARCH-6 | T3.1-T3.4 | `tools/check.sh 3 V17a V17b V17d` passes; the pytest option `--seeds N`, used by gate B, exists |
| T3.6 | smoothness and patch-count study on RE J0317-853 | SURF-3, FWD-6, FWD-15 | T3.1 | `tools/check.sh 2 V11` passes, with the PCHIP option of T2.4 as the default if linear interpolation fails V11a |
| T3.7 | a toy field model registered from a test file, run through `fit` | FIELD-4 | T3.1 | `tools/check.sh 1 V7e` passes |

## M4. Error bars

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T4.1 | prior (DESIGN.md section 4, step 0), 4 MCMC ensembles per mode, run-time convergence checks, hand-back to the search | UNC-1, UNC-2, UNC-3 | M3 (D8 only for extra priors) | `tools/check.sh 1 V14` and `tools/check.sh 2 V13` pass |
| T4.2 | coverage campaign | UNC-1, UNC-4 | T4.1, T3.2 | `tools/check.sh 3 V15` passes |
| T4.3 | simulation-based calibration on the reduced problem | UNC-1, LIKE-9 | T4.1 | `tools/check.sh 3 V16` passes |
| T4.4 | tempered widths, profiles, the "not constrained" flag (the tempered widths are withdrawn by decision #24 and the profiles redone per family in T7.10) | UNC-5, UNC-6, REP-4 | T4.1 | `tools/check.sh 1 V1l V4f` passes |

## M5. Systematics and report

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T5.1 | swap refits, jackknife, "mode change" | SYS-1, SYS-2, SYS-4, UNC-7 | M4 (default list: decision #21) | `tools/check.sh 3 V18a V18b` passes |
| T5.2 | report and every output file, with the report rules; floor chi2; residuals and histograms | SYS-3, SYS-5, REP-1, REP-2, REP-3, DIAG-1, DIAG-2 | T5.1 | `tools/check.sh 1 V2i V4e V4f V7f` passes |
| T5.3 | planted faults | all | T5.2 | `tools/check.sh 3 V19` passes |

## M7. Changes from the design review (decisions #22-#29, 2026-10-22)

**Order.** T7.1 first. Pass A (T7.2-T7.6) unblocks the rest; its tasks can run in parallel once T7.1 is done, except
that T7.6 needs T7.4. Pass B (T7.7-T7.11) follows. T7.12 needs real data and is done outside this repository.

| task | what | requirements | needs | done when |
|---|---|---|---|---|
| T7.1 | the plan: SPEC, DESIGN, VERIFICATION, GATES, TASKS, FORMATS, GLOSSARY, DECISIONS for decisions #22-#29 | all changed | decisions #22-#29 | `python3 docs/review/lint_plan.py .` prints `SUMMARY: 0 failures`, and one review pass is done |
| T7.2 | review fixes (2026-10-19): beta in the "incl" fingerprint (polarity); swaps compared without the components they set; normalisation nodes checked against the library range; atomic output files; relative paths in `run.json` | SEARCH-2, SYS-4, LIB-7, ENG-4, ENG-7 | T7.1 | `tools/check.sh 1 V3k V4a V7g V7h` passes |
| T7.3 | make V14a and V7c independent of the BLAS build (their failures on the workstation, 2026-10-19, reproduced 2026-10-22 with `OPENBLAS_CORETYPE=Haswell tools/check.sh 1 V14a V7c`: `test_v14a_observer_chart_samples_the_prior_with_the_data_off`, KS p = 0.005 < 0.01 at a fixed seed; `test_v7c_modes_consistent_neg_log_l_is_the_cost_of_the_written_vector`, an exact == of costs from batches of different size, 4e-13 apart) | UNC-2, SEARCH-1 | T7.1 | `tools/check.sh 1 V14a V7c` passes, and passes again with `OPENBLAS_NUM_THREADS=4` and with `OPENBLAS_CORETYPE=Haswell` |
| T7.4 | continuum: the normalised basis; the setting in every table header and in `run.json`; default swaps `continuum_deg0` and `continuum_off`; weights with every likelihood; the tempered widths removed (`temper_T` gone from `defaults.toml`) | LIKE-3, SYS-1, DATA-4, UNC-1, UNC-6 | T7.1 | `tools/check.sh 1 V3g V3l V4f V7f V14a` passes |
| T7.5 | Teff interpolation with continuum-matched ("planck") weights (`model.interp_teff`), and the Planck variant of the synthetic library | LIB-3 | T7.1 | `tools/check.sh 1 V2c` passes |
| T7.6 | kernel fit per spectrum (with its white part) and its place in the search (DESIGN.md section 3, step 2b); the whiteness check; the kernel in `run.json` and in the report header; `objective.kernel_fit`, `[[objective.kernel]]`; the floor chi2 of DIAG-1 (DESIGN.md section 5b2; not built before) and the per-window residual table DIAG-3 | LIKE-2, LIKE-4, LIKE-10, UNC-4, DIAG-1, DIAG-3 | T7.4 | `tools/check.sh 1 V1o V1p V2i V4h V7i` passes |
| T7.7 | polish: `fd_step` 1e-4, `search.lbfgs`; the LM finish's Jacobian by central differences at a half-step of `fd_step`, in one batched forward call (decision #31, replacing the analytic Jacobian of #25); measure the LM shortfall on the toy star (and on a real object) and set `search.lm_maxiter` so the median shortfall is below 1 nat; the LM stops `search.lm_stop_nat` and `search.lm_max_rejections` (defaults 1e-3 and 5 by decision #37, provisional; a damped step's small gain is no stop); the value tolerance measured at the error-bar patch count | SEARCH-8 | T7.6 | `tools/check.sh 1 V1q V3n V7c` passes; the measured shortfall is in DESIGN.md section 3 |
| T7.8 | MCMC batching (no calls below one half-step of all ensembles) and post-processing off the critical path | UNC-1 | T7.1 | `tools/check.sh 1 V5c V14 V7f` passes |
| T7.9 | families: link test, components, band components, `families.json`, `family` in `modes.json`, statistics per family | SEARCH-4, SEARCH-9 | T7.7 | `tools/check.sh 1 V3m V4e V4g V7c V7j` passes (V7j: its `families.json` part) |
| T7.10 | profiles and envelopes per family, the side-by-side table and the two sentences per view; `profiles.json` | UNC-5, UNC-8, REP-1 | T7.9 | `tools/check.sh 1 V1l V4i V7j` passes |
| T7.11 | scope: mu-grid libraries; the wavelength shift (FWD-13; not built before: T2.3 left it out), velocity and constant, fixed and free, on the model only (DESIGN.md section 5b) | LIB-8, FWD-13 | T7.1 | `tools/check.sh 1 V2g V2k` passes |
| T7.12 | (outside this repository) the real-object comparison of decision #29: a low-field object with three phases, then RE J0317-853 on two library versions, as recorded runs; the side-by-side tables and the two written interpretations per object, as input for D12 | decision #29 | T7.2-T7.11, T6.1 | the runs are recorded, and the two write-ups exist |
| T7.13 | (2026-11-01) checkpoints of every `fit` stage in `<out>/checkpoint/`, `fit --resume` and `fit --from <dir>`, the guard on the search's inputs | ENG-8 | T7.10 | `tools/check.sh 1 V7k` passes |

## M6. Use in a research project

These tasks were done outside this repository, in the research project that uses zeemap for its fits. They stay here
as a record, because T7.12 and V18c refer to them.

| task | what | needs | done when |
|---|---|---|---|
| T6.1 | a locked environment for zeemap in that project | M0 | the environment prints its package versions |
| T6.2 | that project's checks run zeemap's tier 1 on every change, and tier 2 before it moves to a new zeemap commit | T6.1, M1 | the project's checks show zeemap's SUMMARY line, passing |
| T6.3 | a comparison of two fit outputs at tolerance: `modes.json` (mode tables matched by fingerprint) and `spectra/` | M5 | two runs of the same recorded fit compare equal |
| T6.4 | first recorded RE J0317-853 fit | T6.1-T6.3 | the run is recorded, and `tools/check.sh 3 V17c V18c` passes |
