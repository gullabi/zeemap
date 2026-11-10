# tests/regression

Stored reference numbers of the checks.

- `speed_baselines.json`: V12 (VERIFICATION.md V12; `tests/test_v12_speed.py`). One entry per machine, keyed by
  `"<CPU model> | <cores>"`, with the best-of-repeats times in ms per evaluation of `Problem.cost` on the
  V12 problem (5000 patches, 8 phases, 900 pixels, synthetic library): `full_single_ms`, `full_batch_ms` (per row of
  a batch of 32), `chi2_single_ms`, `chi2_batch_ms`, measured single-threaded. V12 fails when a time is more than 2 x
  its entry. The first run on a machine without an entry writes it and passes with a note; commit the new entry only
  for machines that run tier 2 regularly. To renew an entry after an intended speed change, delete it, run
  `tools/check.sh 2 V12`, and note the before/after numbers in the pull request.
