# Changelog

All user-visible changes to zeemap. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [semantic versioning](https://semver.org/) (before 1.0, a minor version can change configuration keys
or output formats).

## Unreleased

- Releases are tagged and published by CI when a new version reaches `main` (CONTRIBUTING.md, "Releasing").

## 0.1.0 - 2026-11-08

The first public version.

- **Forward model:** Fibonacci surface grids; field models `dqo` (offset dipole + quadrupole + octupole)
  and `harm4` (centred spherical harmonics up to l = 4); library lookup with interpolation in B, psi and Teff; mu-grid
  libraries; instrumental broadening, wavelength shifts, limb darkening.
- **Likelihood:** plain chi2, or the full likelihood with a correlated-noise kernel fitted per object and a
  marginalised Legendre continuum on I.
- **Search:** many starts, local optimisation, restart rounds, modes grouped into families joined by low-cost paths;
  per-stage checkpoints with `zeemap fit --resume` and `--from DIR`.
- **Uncertainty:** profile likelihoods per family, MCMC inside each mode (`--mcmc`), run-time convergence checks,
  refits with changed assumptions (systematics) reported apart from the error bars.
- **Commands:** `fit`, `eval`, `inject`.
- **Tests:** more than 450 tier-1 checks on synthetic data, run on every push and pull request. Of the checks on real
  libraries (tiers 2 and 3) only the speed check V12 is built so far.
- Checkpoints are loaded with a restricted unpickler that accepts only zeemap's checkpoint classes and numpy arrays
  (SECURITY.md).
