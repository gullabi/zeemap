# How a change gets into main

**Who this is for:** anyone who wants to contribute a change, and the maintainers who review and merge it.
**What you will know afterwards:** the steps from a fork to a merge, and the three gates. The checks themselves are in
[VERIFICATION.md](VERIFICATION.md).

**Summary.** `main` must always work. A change reaches `main` through a pull request: its checks pass, and a maintainer
who did not write it reviews and merges it. A change that can move numerical results also passes the checks on real
model libraries. If `main` breaks anyway, the change is reverted first and fixed afterwards.

## Steps

1. Fork the repository and make a branch for the change.
2. While working, run `tools/check.sh 1 --fast` (about 4 minutes; it leaves out the slow end-to-end fits).
3. Before opening the pull request, run the full `tools/check.sh 1` (about 10 minutes on 4 cores) and paste its
   SUMMARY line (the one-line result that `tools/check.sh` prints per tier) into the pull request.
4. Say in one sentence what the change does. If it changes results on purpose, regenerate the references in the same
   change and give the before and after values; note user-visible changes in `CHANGELOG.md`.
5. CI runs tier 1 on the pull request (gate A). Someone other than the author reviews it (gate B): a maintainer, or,
   for a maintainer's own change, another maintainer or an independent review session. A maintainer then merges it
   with a merge commit.

Changing a tolerance, a default budget or a pass rule needs agreement in an issue first, before the pull request.

## Clean room

zeemap is written from scratch. No code may be copied or translated from other fitting programs, Numerical Recipes or
MAGWD, and contributors do not read their source while writing zeemap code. The repository holds no library files, observations or outputs of real objects (ENG-5).

## Gate A: every pull request
- Tier 1 passes in CI, from the lock file on a clean machine.
- The scans of V8 find no absolute paths, data files or forbidden identifiers.

## Gate B: merge into main
- Gate A passes, and the pull request shows the SUMMARY line of the author's full tier-1 run.
- Someone other than the author has reviewed it (step 5 above), using the checklist below.
- **Changes that can move numerical results** (the forward model, library reading, likelihood, search, uncertainty,
  defaults or tolerances): a maintainer with real model libraries also runs tier 2, `ZEEMAP_DATA=... tools/check.sh 2`,
  before merging, with no skipped checks, and adds its SUMMARY line to the pull request. Depending on what changed,
  also:

  | if the change touches ... | also run | pass rule |
  |---|---|---|
  | the likelihood, the prior or the MCMC | `tools/check.sh 3 V16` (a few hours) | V16's rule |
  | the search | `tools/check.sh 3 V17a -- --seeds 10` (about a day) | at least 9 of 10 seeds succeed |
  | only documentation | `python3 docs/review/lint_plan.py .` (for changes to `docs/`, also the review of [review/README.md](review/README.md)) | `SUMMARY: 0 failures` |

  **Not built yet** (decision #35; "Open questions" in DECISIONS.md): of the real-data checks, only V12 exists today, so
  `tools/check.sh 2` runs V12 alone and the V16 and V17a items above select no test. Until they are built, the
  pull request names the checks that would apply and says that they could not be run.

  The 9-of-10 rule passes a search that succeeds 99 % of the time with probability 0.996, and one that succeeds 80 % of
  the time with probability 0.38: it catches large regressions before the merge; tier 3 (28 of 30) catches smaller
  ones before a release.
- Every new or changed requirement in SPEC.md has its "Checked by" line, and every check its "Covers" line.
- A decision, default or pass rule that rests on a fit of real data cites a recorded run, or is marked "provisional"
  in DECISIONS.md (decision #28).

**Review checklist.** Count as findings only problems of correctness, of a stated requirement, or of understanding.
1. Does the author's one-sentence description match the diff?
2. Does every changed behaviour have a check that fails without the change?
3. Was a tolerance or pass rule loosened? If so, is the reason written down, and was it agreed in an issue?
4. For each changed rule in the documentation, the author restates the rule in their own words in the pull request.
   Does that restatement match the text? If it does not, the text is unclear and is rewritten.

## Gate C: release, or fits for a publication

Version 0.1.0 was released on gates A and B only, because tiers 2 and 3 are not built yet (decision #35); the release
rule below applies once they exist. The tag and the GitHub release are made by CI (CONTRIBUTING.md, "Releasing").

- **Release.** Gate B passes, tier 2 passes on the commit being tagged, and tier 3 passes on it (for V15 its part
  V15a; V15b is run and its result reported, decision #24). A tier-3 check may be skipped only when nothing it depends
  on changed since its last pass, and that pass is cited: V15 and V16 depend on the likelihood, prior, MCMC and forward
  model; V17 on the search and forward model; V18 on the systematics code and the search. V19 always runs.
- **Fits for a publication.** Run them from a tagged release or a recorded commit, and keep the run directory:
  `run.json` records the zeemap version, the configuration, the input checksums and the seed, so the fit can be rerun
  and compared at tolerance. A recorded commit that is not a release needs tiers 2 and 3 on that commit first, as for
  a release.

## When main breaks
If a check fails on `main` after a merge, the merge is reverted first. The fix comes as a new pull request through
gate B.
