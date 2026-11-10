# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through GitHub's
[security advisory form](https://github.com/gullabi/zeemap/security/advisories/new). We aim to answer within two weeks.

## What to be careful about

- **Only resume runs you trust.** `zeemap fit --resume` and `--from DIR` read the checkpoints in `checkpoint/`, which
  are Python pickles. zeemap loads them with a restricted unpickler that accepts only zeemap's checkpoint classes and
  numpy arrays, but treat a run directory from someone else like any other program input: share configurations and
  results, not `checkpoint/`.
- Configurations, libraries and observations are read as data only; a configuration cannot make zeemap import or run
  code.

## Supported versions

Only the latest release gets security fixes.
