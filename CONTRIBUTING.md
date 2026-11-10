# Contributing to zeemap

Thank you for helping. Bug reports, questions, documentation fixes and code are all welcome. This page takes you from a
fresh clone to a merged pull request.

## Ask first, code second

- **Found a bug or have a question?** Open an [issue](https://github.com/gullabi/zeemap/issues). For a bug, give the
  zeemap version (`zeemap --version`), the command, the configuration file and what you expected.
- **Want to change behaviour?** Open an issue first, so we can agree on the change before you spend time on it. This is
  required for changes to a tolerance, a default budget (starts, rounds, evaluations) or a pass rule of a check.
- **Small fixes** (typos, docs, an obvious bug with a test) can go straight to a pull request.

## Set up

```bash
git clone https://github.com/<you>/zeemap.git && cd zeemap
uv venv -p 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock     # the versions CI uses
uv pip install --python .venv/bin/python --no-deps -e .
tools/check.sh 1 --fast                                     # about 4 minutes on 4 cores
```

Without uv: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install --no-deps -e .`

## The checks

The checks are grouped into **tiers** by cost and by whether they need real data
([docs/VERIFICATION.md](docs/VERIFICATION.md) section 2):

| command | when | time |
|---|---|---|
| `tools/check.sh 1 --fast` | while working (leaves out the end-to-end fits) | ~4 min |
| `tools/check.sh 1` | before you open a pull request; CI runs it too | ~10 min |
| `tools/check.sh 1 V3` or `.venv/bin/python -m pytest tests/test_v3_search.py` | one check or one file | seconds to minutes |
| `tools/check.sh 2` | maintainers, before merging changes that can move results (only V12, a speed check on a synthetic library, is built so far; later checks read real libraries from `$ZEEMAP_DATA`) | minutes |

Each run prints one `SUMMARY tier <n>: ... passed, ... failed` line per tier. Tier 1 uses only synthetic data, so
everyone can run it. How the tests are named and organised: [tests/README.md](tests/README.md).

## Make the change

1. Fork the repository and create a branch.
2. If you fix a bug, first write a test that fails because of it, then fix it. If you add behaviour, add the test that
   shows it works. Put the test in the file of the matching check and name it `test_v<k><letter>_...`
   ([tests/README.md](tests/README.md)).
3. Keep it simple: the smallest change that solves the problem, in the style of the surrounding code. No new
   abstraction for a single use, no option nobody needs yet.
4. Keep the docs in step. A new requirement or check goes into [docs/SPEC.md](docs/SPEC.md) or
   [docs/VERIFICATION.md](docs/VERIFICATION.md) in the same pull request; a user-visible change gets a line in
   [CHANGELOG.md](CHANGELOG.md) under "Unreleased".
5. Comments explain *why*, in plain words. They may cite requirement ids (`SEARCH-9`), check ids (`V7e`), sections of
   `docs/` and decision numbers (`decision #27`); they do not tell the history of the change (that is what git log is
   for).
6. Run `tools/check.sh 1` and open a pull request. Paste its SUMMARY line into the description, and say in one sentence
   what the change does. If a check failed first, say so and why.

Someone other than you reviews the change, and a maintainer merges it. For changes that can move numerical results (the
forward model, library reading, likelihood, search, uncertainty, defaults), a maintainer also runs tier 2 on real data
before merging. The full rules are in [docs/GATES.md](docs/GATES.md).

## Never

- **Copy or translate code from other fitting programs, Numerical Recipes or MAGWD, or read their source while writing
  zeemap code.** zeemap is written from scratch, and its licence depends on that.
- Commit model libraries, observations, or results of real stars. They live outside the repository, in the
  directory that `$ZEEMAP_DATA` names (the hygiene check V8 looks for data files and absolute paths).
- Loosen or delete a check to make it pass.

## Notes for AI coding agents

Agents follow [AGENTS.md](AGENTS.md), which repeats the rules above as commands.

## Code of conduct

Everyone taking part follows our [code of conduct](CODE_OF_CONDUCT.md).

By contributing, you agree that your contribution is licensed under the [BSD 3-Clause licence](LICENSE) of this project.
