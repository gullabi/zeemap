**What this changes, in one sentence:**

Fixes # <!-- the issue, if there is one -->

**If results change on purpose:** the before and after values.

**Checks** (paste the SUMMARY line of `tools/check.sh 1`; say if a check failed first and why):

```
SUMMARY tier 1: ...
```

- [ ] A test fails without this change and passes with it (or: docs only)
- [ ] `docs/` and `CHANGELOG.md` are updated where behaviour changed
- [ ] Can this move numerical results (forward model, library reading, likelihood, search, uncertainty, defaults)?
      If yes, a maintainer runs tier 2 before merging ([docs/GATES.md](../docs/GATES.md))
- [ ] No code copied from or written while reading other fitting programs, Numerical Recipes or MAGWD
