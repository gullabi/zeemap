# tests/

The tests are organised by **check**, not by module. Each check V1-V19 is described in
[docs/VERIFICATION.md](../docs/VERIFICATION.md) section 5, and each belongs to a **tier** (when it runs, section 2).

| check | what it shows | tier |
|---|---|---|
| V1 | known answers of the small parts (field, surface, data, likelihood, noise, profiles ...) | 1 |
| V2 | known answers of the whole forward model on a synthetic library | 1 |
| V3 | symmetries and consistency (metamorphic and property tests) | 1 |
| V4 | error handling and report rules | 1 |
| V5 | the fast code agrees with the slow reference (`src/zeemap/*_reference.py`) | 1 |
| V6 | determinism: same seed, same result | 1 |
| V7 | the commands run end to end | 1 |
| V8 | repository hygiene (absolute paths, data files) | 1 |
| V14 | run-time checks of convergence and consistency (MCMC) | 1 |
| V10-V13 | regression on real data, convergence, speed, prior recovery | 2 (real data; only V12 is built) |
| V15-V19 | coverage, calibration, search completeness, systematics, planted faults | 3 (days; not built yet) |

**Names.** A test of check item V2c is named `test_v2c_...` and lives in a file `test_v2_<topic>.py`. Every test
carries exactly one marker `tier1`, `tier2` or `tier3`, matching its check (`conftest.py` enforces this); end-to-end
fits also carry `slow`.

**Running.** `tools/check.sh 1 --fast` while working, `tools/check.sh 1` before a pull request, `tools/check.sh 1 V2c`
for one item. Plain pytest works too: `.venv/bin/python -m pytest tests/test_v2_forward.py -k v2c`.

**Data.** Tier 1 builds small synthetic libraries in temporary directories (`synth.py`, `tools/zeeman_library.py`);
no test reads files from the repository except the stored numbers in `regression/`. Tiers 2 and 3 read real libraries
from `$ZEEMAP_DATA`.
