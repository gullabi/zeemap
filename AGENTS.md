# AGENTS.md: instructions for coding agents working in zeemap

Read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) first; this file repeats their rules as commands.
The number after `check.sh` is the tier of checks ([docs/VERIFICATION.md](docs/VERIFICATION.md) section 2).

## Commands
| when | command |
|---|---|
| set up | `uv venv -p 3.12 .venv && uv pip sync --python .venv/bin/python requirements.lock && uv pip install --python .venv/bin/python --no-deps -e .` |
| while working (leaves out the end-to-end fits marked `slow`; about 4 min) | `tools/check.sh 1 --fast` |
| one check only | `tools/check.sh 1 V3` (any tier and check) |
| before opening a pull request (about 10 min on 4 idle cores) | `tools/check.sh 1` |
| maintainers, before merging a change that can move results (only V12 is built; later checks read `$ZEEMAP_DATA`) | `tools/check.sh 2` |

## Always
- Write the check first and see it fail, then make it pass.
- Report the SUMMARY lines of the checks you ran, with any failures, exactly as printed.
- Keep `docs/` in step with the code: a new requirement or check goes into SPEC.md or VERIFICATION.md in the same change.
- Make the smallest change that solves the problem, in the style of the surrounding code.

## Ask a maintainer first (in the issue)
- Changing a tolerance, a default budget (the number of starts, rounds and evaluations) or a pass rule.
- Anything listed under "Open questions" in `docs/DECISIONS.md`.
- Adding a dependency.

## Never
- Copy or translate code from other fitting programs, Numerical Recipes or MAGWD, or read their source while writing
  zeemap code (clean room, [docs/GATES.md](docs/GATES.md)).
- Commit library files, observations, outputs for real objects, or any file from `$ZEEMAP_DATA`.
- Write absolute paths into code, tests or examples.
- Merge your own change, or review a change written in the same session.
- Loosen or delete a check to make it pass.
