# How this plan was reviewed, and how to review changes to it

**Who this is for:** anyone who changes the documents in `docs/`, or reviews such a change.

## The checklists
- [PRACTICES_SPEC.md](PRACTICES_SPEC.md): 40 yes/no checks for specifications and plans (requirement quality,
  document structure, specs an agent can carry out, readability for first-time readers), with 18 common mistakes. Each
  check names the published guidance it comes from (ISO/IEC/IEEE 29148, RFC 2119/8174, EARS, Google design docs, PEPs
  and Rust RFCs, architecture decision records, Diataxis, GitHub Spec Kit, plain-language and GOV.UK guidance).
- [PRACTICES_VERIFY.md](PRACTICES_VERIFY.md): 33 yes/no checks for verifying scientific and statistical software and
  for merge gates (known answers, metamorphic tests, simulation-based calibration, injection and recovery, MCMC
  diagnostics, search completeness, seeds, tiers).

The checks marked [TOP] block acceptance. Both files were written by research agents on 2026-10-17 from sources they
opened; claims they could not open are marked [unverified].

## The mechanical check
`python3 docs/review/lint_plan.py .` (from the repository root) checks that every requirement names an existing check,
that "Checked by" and "Covers" agree both ways, that every requirement has a task, that every `[NEEDS DECISION]`
marker is listed in SPEC.md section 8, that links resolve, and that no absolute path appears. It also checks that
every check item a task's "done when" command names has a test by name in `tests/` (TASKS.md, "The check runner":
items listed as not built, items of checks waived by decision #35 and tier-3 items are reported as a notice, any other
item without a test is a failure). It also lists sentences over 30 words for a human to judge. It must print
`SUMMARY: 0 failures`.

## How the plan was reviewed (2026-10-17)
Four review turns, at most six planned. In each turn, new reviewers who had not seen the plan before worked in
parallel:
- first-time readers with different backgrounds (astronomy student, software engineer, physics student, AI coding
  agent, astronomer collaborator, postdoc) explained the plan back in their own words and listed every place they
  stumbled (a paraphrase test);
- an auditor applied both checklists;
- a fact checker compared every number, formula and convention with the sources (the research notes and a research
  prototype, which are not part of this repository).

After each turn the plan was revised, and the mechanical check was run. The loop stopped when the readers could
explain the plan correctly with no blocking stumbles and the auditor accepted it (turn 4). The change history is in
[../DECISIONS.md](../DECISIONS.md).

## Reviewing a change to the documents
1. Run the mechanical check.
2. A reviewer who did not write the change applies the [TOP] checks of both checklists to the changed parts.
3. For a large change, one first-time reader explains the changed parts back in their own words (GATES.md, review
   checklist item 4).
