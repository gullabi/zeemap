# How zeemap is built (design)

**Status:** describes zeemap 0.1.0; the history of its decisions is in [DECISIONS.md](DECISIONS.md).
**Who this is for:** whoever writes the code, person or agent, and reviewers.
**What you will know afterwards:** the parts of the program, how they talk to each other, the formulas and
algorithms with their default settings, and why each choice was made. *What* the program must do is in
[SPEC.md](SPEC.md). The design may change without changing SPEC.md, as long as the checks still pass.

All research numbers below come from the research of 2026-10-17 and the design review of 2026-10-19/20, done before
zeemap was published (not part of this repository). They are exploratory: they come from agent scripts, not from
recorded runs.

## 1. Parts and how data flows through them

```
config.toml ──> config       read, check against defaults.toml, resolve paths
                   │
                   v
problem ──── library      reads info.dat + .bin (FORMATS.md section 1), keeps the Teff slices in memory
   │     ──── data         observed I, V/I and errors; windows, masks, weights
   │     ──── surface      patch positions and areas (Fibonacci sphere)
   │     ──── field        field models (registry); parameters <-> field vectors
   │     ──── forward      patches -> visible weights -> library lookup -> I, V per phase
   │     ──── likelihood   full | chi2
   │   cost(X): a batch of parameter vectors (K x dim) -> K values of -ln L
   v
search       start distribution, local optimisation, kernel fit, restart rounds, fingerprints, modes, families
uncertainty  MCMC ensembles per mode, convergence checks, profiles and envelopes per family
systematics  refits under named swaps, jackknife
report       JSON files, report.txt
cli          eval | fit | inject
```

**Modules.** Every file in `src/zeemap/`:

| file | what it holds |
|---|---|
| `__init__.py` | the package and its version |
| `defaults.toml` | every configuration default; the only place defaults are written (CFG-1 to CFG-3) |
| `config.py` | reads the TOML configuration, merges it over `defaults.toml`, stops on unknown or missing keys, resolves paths |
| `types.py` | data types shared by `data`, `forward` and `likelihood` (DEV_INTERFACES.md) |
| `library.py` | reader and writer of the library format (FORMATS.md section 1) |
| `data.py` | observations, errors, fit windows, excluded ranges, weights |
| `surface.py` | the patch surface (Fibonacci sphere), the observer direction, limb darkening |
| `field.py` | the field models "dqo", "dqo_aligned" and "harm4" (section 2) |
| `kernels.py` | numba kernels of the forward model: the scatter of visible patches onto the library nodes, field sums |
| `forward.py` | the forward model: parameters to model I and V/I on the data grid, for a batch |
| `forward_reference.py` | a plain numpy reference of the forward model, written for clarity (check V5a) |
| `likelihood.py` | the plain chi2 and the full likelihood (section 5) |
| `likelihood_reference.py` | a slow, direct reference of the likelihood (check V5b) |
| `noise.py` | the noise kernel fitted per object and the whiteness check (LIKE-2, LIKE-4, LIKE-10) |
| `problem.py` | the fitting problem: configuration to library, surface, field, data, forward model and objective; the parameter vector; the cost function the search calls |
| `search.py` | the search for modes (section 3): starts, polish, kernel fit, restart rounds, fingerprints |
| `table.py` | the table stage between the search and the error bars: re-polishing at the error-bar patch count, the mode band, families (section 3, steps 6 and 6b) |
| `families.py` | families: end points joined by paths below a level (section 3b) |
| `uncertainty.py` | error bars inside a mode: prior, MCMC, hand-back, intervals (section 4) |
| `convergence.py` | the run-time convergence rules of the MCMC (R-hat, ESS, autocorrelation time; check V14) |
| `profiles.py` | profiles and envelopes per family, and the side-by-side table (section 4b) |
| `systematics.py` | systematic swaps and the leave-one-phase-out jackknife (SYS-1 to SYS-5) |
| `diagnostics.py` | the floor chi2 and the per-window residual table (DIAG-1, DIAG-3; section 5b2) |
| `report.py` | the output files of `eval` and `fit` (FORMATS.md section 6) |
| `atomic.py` | atomic writes of output files (temporary file, then rename) |
| `inject.py` | `zeemap inject`: simulated observations from a known geometry |
| `cli.py` | the command line: `inject`, `eval`, `fit` |

Rules for the interfaces:
- Every evaluation takes a batch. `problem.cost(X)` with X of shape (K, dim) returns K values of -ln L. A gradient by
  central differences (2 dim + 1 points), an MCMC step of all ensembles and a Hessian are each one call.
  `problem.residuals(x)` returns the whitened residuals of one point, and `problem.residuals_batch(X, scales)` those
  of a batch, from which the polish takes its Jacobian by central differences (section 3, polish; decision #31).
- Parts below `problem` know nothing about the search or the MCMC. Parts above `problem` know nothing about the library
  format or the physics. This is what lets new field models be added without touching the search (FIELD-4).
- Only `config` and `cli` read environment variables (`$ZEEMAP_DATA`), except that each worker process sets its own
  thread variables to 1 at start-up (ENG-2), and tests read `$ZEEMAP_DATA` to decide whether to skip.

## 2. Field model and parameters

**Field of one centred component.** In the component's own frame (z along its axis), at a point r = (x, y, z) relative
to the field centre, with polar strength a and r = |r|:

| component | B_x and B_y | B_z |
|---|---|---|
| dipole (l = 1) | (3 a z / (2 r^5)) x, same with y | a (3 z^2 - r^2) / (2 r^5) |
| quadrupole (l = 2) | (a (5 z^2 - r^2) / (2 r^7)) x, same with y | a z (5 z^2 - 3 r^2) / (2 r^7) |
| octupole (l = 3) | (5 a z (7 z^2 - 3 r^2) / (8 r^9)) x, same with y | a (35 z^4 - 30 z^2 r^2 + 3 r^4) / (8 r^9) |

Each is the gradient of a potential proportional to P_l(cos theta) / r^(l+1), scaled so that B_z = a at the pole of
the centred component (r = 1, z = 1). The field at a surface point is the sum of the three components, each rotated
from its frame into the star's frame. The star's frame has z along the rotation axis.

**Parameters.** The search works with these 14 numbers (13 that matter, see the gauge below):

| parameter | meaning |
|---|---|
| m_d (3 numbers) | dipole vector: direction = dipole axis, length = polar strength in MG |
| b_q (1) | quadrupole polar strength in MG, with sign |
| a_q (3) | quadrupole axis; only its direction matters |
| m_o (3) | octupole vector, as for the dipole |
| c (3) | field centre in the star's frame, in stellar radii |
| u (1) | cos i, the cosine of the inclination, in [-1, 1] |
| teff, shift (optional) | Teff in K; wavelength shift in A or km/s |

Why vectors: axis angles live in a box of +-90 degrees. The box edges cut through the space of geometries, and
searches stop on them. Vectors have no edges. The quadrupole is different: an even multipole is unchanged when its axis
is reversed, so its sign cannot be put into the axis, and it needs a separate signed strength.

**Symmetries** (different parameters, identical spectra):
- the length of a_q is arbitrary (a small penalty keeps it near 1);
- reversing a_q, with b_q unchanged;
- the mirror through the equator: negate the z components of m_d, a_q, m_o and c, and negate u;
- for a field that is symmetric about one axis (a centred dipole, or "dqo_aligned" with the centre on the axis),
  swapping the inclination and the axis's angle from the rotation axis, with the matching azimuth, gives identical
  spectra.

The search runs in the full space, without cutting it in half, so no new edges appear. For grouping and reporting each
point is mapped to one **canonical form**: |a_q| = 1, a_q pointing into z >= 0 (ties broken by x, then y), and u >= 0.

In the star's frame the observer direction at phase phi is n = (sin i cos 2 pi phi, -sin i sin 2 pi phi, -cos i), with
u = cos i; at i = 0 the observer sees the southern hemisphere. A patch is visible when mu = n . (its normal) > 0.

**"harm4": the full expansion up to l = 4** (FIELD-4; the published l <= 4 fits of PG 1015+014, Euchner et al. 2006,
Table 4, and of Grw+70 8247, Jordan 2003). Centred; B = -grad V, V = sum over l = 1..4, m = 0..l of
(g_lm cos m phi + h_lm sin m phi) P_l^m(cos theta) / r^(l+1) in the star's frame, P_l^m Schmidt semi-normalised
without the Condon-Shortley phase (Langel 1987). Parameters: m_d = 2 (g11, h11, g10) (the l = 1 part is the centred
dipole, so the dipole quantities and the fingerprint work unchanged), the 21 coefficients g20, g21, h21, ..., h44 in MG,
and u: 25 numbers. Each l is closed under rotations, so no axis angles are needed: a set published in a tilted frame
is turned into the star's frame by `Harm4.rotated` (an exact least-squares re-expansion). The field is linear in the
24 field numbers; the model keeps their basis fields at the patch points. Its equator mirror multiplies g_lm and h_lm
by (-1)^(l+m) and negates u; canonical form u >= 0. With one phase every coefficient is a declared nuisance
(decision #18). The papers do not state every convention (normalisation, Condon-Shortley phase, sign, observer side).
Provisional, until a recorded fit decides it: a published set enters by negating its odd-m coefficients (a
Condon-Shortley phase), `rotated` and the mirror with u kept (an observer on the +z side). The evidence is an unrecorded
forward-model study: with this convention the l <= 4 sets of PG 1015+014 (Euchner et al. 2006, Table 4) and Grw+70
8247 (Jordan 2003) match their published spectra.
Jordan's printed sign of the field (B_r = -(l+1) ...) gives the mirror image of his spectrum, so a sign read from a paper
alone is not enough.

## 3. Search for modes

**Start distribution** (configurable in `[search.starts]`; the defaults are those of the research, for RE J0317-853):
- |m_d| uniform in [200, 600] MG, direction uniform on the sphere;
- b_q uniform in [-250, 250] MG, axis uniform on the sphere;
- |m_o| uniform in [0, 200] MG, direction uniform on the sphere;
- the centre c uniform in a ball of radius 0.25 stellar radii;
- u = cos i uniform in [0.03, 1] (the mirror symmetry covers negative u).

**Prior cut.** At most 5 % of the whole surface may have |B| outside the library range. Starts that break it are
redrawn. During polishing it acts as a penalty, 2e4 x max(0, f - 0.05), where f is the outside fraction smoothed with a
logistic step 2 MG wide, so that gradients exist. End points that break the cut are dropped from the mode table. The
MCMC treats it as part of the prior (density 0 outside).

**Data-informed rules.** The research budget below was measured with two more acceptance rules: the whole surface inside
the library range, and at least 10 % of the surface between 170 and 210 MG with the smallest |B| at most 195 MG (a rule
made from the RE J0317-853 data). Such rules may be added in the configuration and are written to `run.json`. Without
them, more starts may be needed.

**Algorithm:**
1. Draw the configured number of starts (research: 250, the default until task T3.5 sets it; the runs so far used 40).
2. Improve each by polishing (below): L-BFGS-B (a standard quasi-Newton optimiser with range limits) in coordinates
   scaled by each parameter's step, then Levenberg-Marquardt steps (a least-squares optimiser) with a Jacobian from
   central differences in one batch (decision #31). About 5000 evaluations per start.
2b. **Kernel fit** (LIKE-2, decision #22; section 5): (i) fit the kernel of every spectrum at the best end point of
   step 2 (searched with the initial kernel); (ii) re-polish, with that fitted kernel, the end points that lie in the
   band (SEARCH-7) as measured under the initial kernel; (iii) refit the kernel at the new best end point and freeze
   it; (iv) re-evaluate (without polishing) every other end point of step 2 under the frozen kernel. Every later step
   (restart rounds, SEARCH-8, families, profiles, MCMC, swaps) uses the frozen kernel. Then the whiteness check
   (LIKE-10) runs at the best end point. With `objective.kernel_fit = false` the step is skipped.
3. Group the end points into modes by **fingerprint**: an angle, the field centre, B50 and B90 (FORMATS.md section 7)
   at a reference phase, the rms of B50 over the phases, and Teff when it is free. Two end points are one mode when,
   after mapping to the canonical form, they agree within 8 degrees in the angle, 0.06 stellar radii in the centre and
   1000 K in Teff. The B thresholds are fractions of B50 (the larger of the two at the reference phase): 4 % in B50,
   10 % in B90 and 5 % in the B50 rms, with floors of 0.5, 1 and 1 MG (decision #16). With more than one phase the angle is the inclination
   (folded to 0-90 degrees), and beta, the angle between m_d and the line of sight at the reference phase (0-180
   degrees, with the same 8-degree threshold), is compared as well, so that a geometry and its field-reversed copy
   (same inclination, V of opposite sign) are two modes; the centre is compared in the star's frame. With one phase only the angle beta between
   the dipole and the line of sight is determined: the inclination and the turn of the whole geometry about the line
   of sight are not, and the centre turns with the dipole. So the angle is beta, and the centre is compared in the
   frame of the dipole axis and the observer (along the axis, towards the observer, and the size of the third
   component). On a real single-phase fit this grouped band modes that differed only in that turn.
   Grouping is greedy, from the best end point down. The research's absolute 5 MG in B50, 25 MG in B90 and 10 MG rms
   are the relative thresholds at B50 of 125, 250 and 200 MG; on a low-field star they merged end points far apart in
   -ln L.
   These thresholds group search end points; they are narrower than the posterior of one mode (in the research, 944 of
   1500 MCMC samples of one mode failed them). So whether a point has **left a mode** (SYS-4, UNC-3, V15) is decided
   against that mode's MCMC samples: a point is in another mode when its fingerprint distance to the mode, d = the
   largest of |difference| / threshold over the fingerprint components above, exceeds the 99th percentile of d over the
   mode's MCMC samples. For a swap refit (SYS-4) the components that the swap sets (Teff, for a swap that fixes Teff)
   are left out of d.
4. Restart round: perturb each of the 10 best modes 12 times (`search.restarts`) (5 % in field strengths, 8 degrees in axes, 0.04 radii in
   the centre, 4 degrees in inclination) and re-optimise (about 3200 evaluations each).
5. Repeat step 4 until a round improves the best mode by less than 5 nats, at most 3 rounds.
6. Re-polish and re-rank every mode within the band at 20 000 patches (SEARCH-8): per mode up to
   `search.tol_polishes` (default 5) of its end points, best first; the value tolerance is measured from these
   (below).
6b. Group the re-polished end points into families (section 3b, SEARCH-9).
7. Search statistics: starts, distinct modes, hits on the best mode, and the fraction of modes reached only once. When
   most modes are reached only once, the starts cannot tell how many modes remain unseen; the report says so
   (SEARCH-4).

**Value tolerance.** End points that belong to one mode do not reach exactly the same -ln L: in the research,
L-BFGS-B end points scattered by 5-25 nats, and the Levenberg-Marquardt finish reduced this; in the design review the
scatter was about 3 nats at 5000 patches, set by the roughness of the 5000-patch surface, not by the optimiser. The
**optimiser scatter** of a run is the standard deviation of -ln L over the end points of each mode with at least 3
re-polished end points at the error-bar patch count (step 6), pooled over those modes (5 nats if no mode has 3). The **value tolerance** used by UNC-3, V14f, V17 and the
full-rerun row of VERIFICATION.md section 3 is max(2 nats, min(2 x scatter, 10 nats)). The cap keeps it well inside
the 45-nat band and the 20-nat gap of the planted problem.

**Polish.** "Polishing" a point means: L-BFGS-B from it, then the Levenberg-Marquardt finish (decision #25).
- L-BFGS-B uses central differences with a half-step of `search.fd_step` = 1e-4 parameter steps (the old 0.5 gave a
  gradient whose cosine with the exact one was -0.12 and 0.07 at two test points; 1e-4 gave 0.99999), or the analytic
  gradient when the problem provides one. `search.lbfgs = false` skips the stage. A coordinate whose +h or -h point
  lies beyond a polish limit (that point is not taken: its row of the batch repeats the current point) or gives an
  invalid model takes the one-sided difference from the other side. The L-BFGS-B gradient, the LM Jacobian below and
  the family midpoint (section 3b) share this one stencil (`search.fd_rows`, `search.fd_derivative`).
- The Teff interpolation weights are continuous: a Teff counts as a slice only to rounding (1e-9 x Teff,
  `library.SLICE_SNAP_REL`), not within 0.5 K as before. With the 0.5 K window and a Teff half-step of 0.05 K (a step
  of 500 K), the Teff derivative was exactly 0 within about 0.4 K of every slice and at the polish limits, which are
  the edge slices. Up to 0.5 K beyond an edge slice the model is that slice (`check_teff`), so
  `parameters.teff` limits written there are clamped to the edge slice at load (`Problem`): the polish limits, the
  starts and the MCMC prior range stay inside the slice range, where the Teff derivative is not 0.
  The value is clamped with the limits, so a range wholly beyond an edge (lo 40 000.2, hi 40 000.4 at an edge slice
  of 40 000 K) becomes lo = hi = value = the edge slice; clamping the limits on their own gave lo > hi there, and the
  first polish stopped with an error. Each clamp is noted in run.json (`limit_notes`).
  Every test of whether a Teff is a slice uses the same rule, `library.on_slice`.
- Levenberg-Marquardt needs residuals: it uses the whitened residuals of each spectrum divided by that spectrum's
  current noise scale, after the best continuum, and accepts a step only if the full -ln L goes down. Its Jacobian
  (decision #25, as built; #31): central differences with the half-step `search.fd_step` = 1e-4 parameter steps, all
  2 n rows in one batched forward call, with the noise scales held at the current point (taken from the batch's row
  0, the current point itself, so the call holds 2 n + 1 rows and nothing else is evaluated). This measures the local
  derivative of the piecewise-linear interpolation as a closed-form chain rule would (the review's gradient at 1e-4
  had cosine 0.99999 with the exact one), at the cost of one batch of 2 n rows per Jacobian instead of n separate
  calls at a half step of 0.5. A closed-form chain rule through the field formulas, the visibility and the
  interpolation weights would cost less per Jacobian; it is not built (decision #31). A coordinate whose +h or -h
  point lies outside the polish limits or gives an invalid model takes the one-sided difference from the other side
  (as the L-BFGS-B gradient does), and a coordinate on a limit whose descent direction points outward is held there
  for that LM step (for example a free Teff at the edge of the slices, or u = 1). V1q checks the Jacobian.
- LM stops (`search.LM_STOPS`; each end point records which one in `lm_stop`): converged when an accepted step gains
  at most 1e-12 |-ln L| or the clipped step does not move the point or every coordinate is held on a limit; not
  converged when the damping lam passes 1e12 (16 rejected steps in a row from lam = 1e-3: the Gauss-Newton model
  found no descent, which is not evidence that none is left, because the residuals leave out the 0.5 ln det continuum
  term, so J^T r can disagree with the true gradient; in the initial build this exit counted as converged) or at
  `lm_maxiter`. Two further stops, set by decision #37 (provisional until a run on real data, open; 0 turns each off):
  `search.lm_stop_nat` (nats; converged when the undamped Gauss-Newton step 0.5 g^T (J^T J)^+ g at a fresh Jacobian
  predicts less; default 1e-3) and `search.lm_max_rejections` (not converged after that many rejected steps in a
  row; default 5). An accepted step that gains less than `search.lm_stop_nat` is no stop: after rejections the step
  is damped and can gain little while the undamped step still predicts much more (a Rosenbrock LM
  stopped 0.148 nat above its minimum that way); the next Jacobian's prediction decides. Stopping on such a gain only
  at undamped steps (lam at most its start, 1e-3) was measured too and also stopped early on Rosenbrock (3 of 200
  starts, up to 0.25 nat). On the synthetic toy (`tools/measure_lm_shortfall.py`, 20 polishes) the prediction never
  falls below 1e-3 nat (0.4-0.6 nat at end points where -ln L no longer falls: the residuals, divided by the fitted
  noise scales, keep 0.5 |r|^2 = N / 2), so there the saving of decision #37 comes from `search.lm_max_rejections`
  alone: LM evaluations 1160 -> 952, end points at most 4.8e-7 nat above those with both stops off (with the
  small-gain stop: 352, 2.1e-4 nat). A polish counts as converged when the stage that made its end point stopped on a
  convergence rule: the LM finish when its total gain over L-BFGS-B's end point exceeds `search.lm_stop_nat` (with
  that stop off, 0: when it accepted at least one step), or when `search.lbfgs = false`; else L-BFGS-B. So when the
  LM gains no more than `search.lm_stop_nat`, `lm_stop` says why it stopped and `converged` is L-BFGS-B's flag: after
  a converged L-BFGS-B, 5 rejections leave the polish converged, also after a few tiny accepted steps (on the real
  objective the "predicted" stop never fires, so an LM that accepts any step ends on "rejections"; with "the LM
  accepted a step" as the rule, 8 of 20 polishes on synth_single read not converged, exactly those the LM improved,
  by at most 5e-4 nat), and after an L-BFGS-B that ran out of iterations, a "predicted" stop at once
  leaves it not converged, since the Gauss-Newton model of the residuals leaves out the continuum log-det and is no
  convergence test for -ln L by itself. The flag is written per end point (`diagnostics.json`) and
  gates nothing.
- The LM iteration cap `search.lm_maxiter` is 100 (provisional; was 20, which 16 of 20 polishes hit, stopping a median
  3.7 nats short). The **shortfall** of a polish is f(its LM end point at `lm_maxiter`) minus f(the same LM continued
  until a step gains less than 1e-6 nat or the Gauss-Newton model predicts less, i.e. `lm_stop_nat` = 1e-6, or 1000
  iterations; `tools/measure_lm_shortfall.py`). Task T7.7 measures its median over 20 polishes from
  starts perturbed around the best mode (the restart perturbation of step 4), at the error-bar patch count, on the
  toy star and on a real object, and sets the default to the smallest cap of 50, 100,
  200, 400 with a median shortfall below 1 nat.

**Range limits in the vector space.** L-BFGS-B gets wide safety limits that are not meant to be reached: each component
of m_d, m_o and b_q within +-2 x the library's largest B, each component of c within +-0.4, u within [-1, 1]. A mode
whose end point touches a limit is flagged (REP-3).

**Measured cost on RE J0317-853.** 250 starts x 5055 evaluations = 1.3e6, plus one restart round (120 x 3240 =
0.4e6): about 1.6e6 evaluations, 40 minutes to 1.5 hours on 12 cores. This reached the near-best set (within 5 nats of
the best) with an estimated probability of 90 %, but the estimate rests on 3-4 successes, so it is uncertain by a
factor 2-3. Task T3.5 sets the default budget with the planted problem of V17, for a success probability of at least 0.99. That
is expected to need about twice the research budget (about 3e6 evaluations, 1.5-3 hours on 12 cores).

## 3b. Families (SEARCH-9, decision #29)

A fingerprint groups end points by geometry; a family groups them by the likelihood surface between them. On a real
low-field object the fingerprint cut one connected valley into several modes, through the B50, |m_d| and B90
thresholds, while the straight path from the best point to some of those "modes" climbed monotonically: they were
polish end points on the slope of one basin (design review).

- **Nodes.** The end points re-polished at the error-bar patch count in step 6 of section 3, in their canonical form.
  After the MCMC, when a hand-back (UNC-3) has moved a mode more than the value tolerance below its best node, the
  families are recomputed with the new point as an extra node of that mode (`families.json` names the moved modes in
  `recomputed_after_handbacks`), so the level, the families and each family's best node, where its profiles start,
  are not stale.
- **Level.** The family level is f_best + h, with h = max(value tolerance, `search.family.min_level`) and f_best the
  best node's -ln L. The default `min_level` is 5 nats: below about 5 nats the roughness of the 5000-patch surface made
  levels undecidable in the review on one real library (on another, the 5000-patch roughness was 1-3 nats, and
  20 000 against 5000 patches moved the profile by about 1 nat).
- **Link test.** Two nodes i and j, both at or below the level, are linked when one of these paths stays at or below
  the level at every evaluated point:
  1. the straight segment from x_i to x_j in the scaled search coordinates, at 15 equally spaced interior points (one
     batch call); for each pair the symmetric copies (section 2) closest in scaled distance are used; at every path
     point a_q is rescaled to unit length before the evaluation (its length is a gauge), so the gauge penalty is 0 on
     paths, while the prior-cut penalty stays;
  2. if the straight segment fails: a re-polished midpoint m, found by polishing from the segment's midpoint with
     the step constrained to the hyperplane through it perpendicular to the segment (L-BFGS-B on the projected
     gradient, at most `search.family.mid_iter` = 50 iterations; the gradient is the polish's, with the half-step
     `search.fd_step` and one-sided at a polish limit or next to an invalid model, and m stays within the polish
     limits; before, plain central differences next to an invalid model gave a gradient of about
     5e33 and the midpoint never moved), then the segments i-m and m-j at 7 interior points each. No further
     subdivision.
- **Components.** Families are the connected components of the linked nodes (union-find). A mode belongs to the family
  of its best node; a mode whose nodes fall into several families lists them all (`families_all` in `modes.json`). Pairs are tested in order
  of increasing scaled distance, and a pair whose nodes are already in one component is skipped. At most
  `search.family.max_pairs` (default 3000) pairs are tested; when that cap stops the grouping, `families.json` says so.
  With `--workers W > 1` the next W pairs not yet joined are tested at once in the search's pool and their results
  applied in that order, a pair joined meanwhile skipped and not counted, so the components, links and pairs tested
  are the serial ones; only the evaluation count grows by the pairs tested in vain.
- **Band level.** The same grouping at the level f_best + band (SEARCH-7) gives the band components, reported as a
  second column; a mode whose nodes all lie above the family level has no family at that level and is listed with its
  band component only.
- **Cost.** About 15 evaluations per straight pair and a few hundred for a pair that needs a midpoint; in the review,
  the link tests took about 3 % of the search's evaluations.
- Statistics per family (SEARCH-4): the number of modes, end points and starts it holds, and whether the best family
  was reached by one start only.

**Alternatives that were tried, and what happened:**
| alternative | result |
|---|---|
| nested sampling (dynesty, nautilus) | two dynesty seeds ended in two different wrong modes, and their evidence differed by 110 nats; nautilus did not converge in its budget |
| CMA-ES (62 000 evaluations), differential evolution (78 000) | did not reach the best mode; but these ran at about 1/20 of the multi-start budget, so they are not ruled out at a larger budget |
| random starts without restart rounds | never came within 10 nats of the best |
| a simplex optimiser with hand restarts | 1 success in 102 starts |

A global MCMC over the whole parameter space was not tried; emcee's own documentation warns against it for strongly
multimodal problems.

## 4. Error bars inside a mode

0. **Prior.** The MCMC samples exp(-(-ln L)) x p(theta), where p is the density of the start distribution (section 3):
   proportional to |m_d|^-2 on its shell and |m_o|^-2 in its ball (a uniform strength with a uniform direction),
   uniform in b_q, in the direction of a_q, in c and in u (|u| in [0.03, 1]: the start range and its mirror image);
   a_q's length has a Gaussian penalty on ln |a_q| with width 0.1; the prior cut of section 3, and any data-informed
   rule that is configured, set the density to 0 where they fail. V13 checks that the sampler returns exactly this
   when the data are switched off. `[NEEDS DECISION D8]`: whether other priors are allowed.
1. **Sampling coordinates.** When the field is in vectors with m_d free, the sampler works in the observer frame of
   the first phase. Its coordinates are |m_d|; cos beta, beta being the angle between m_d and the line of sight; the
   azimuth of m_d about the line of sight; u; the other vectors' components in that frame; and the scalars. The
   azimuth is held as a 2-d vector whose length has a standard normal factor, so the circle has no edge. The density
   carries the Jacobian |m_d|^2. One phase fixes
   |m_d| and beta but not the azimuth or u, so in Cartesian m_d and u the posterior is a thin curved shell that
   straight steps leave; in these coordinates it is a product of narrow and flat directions. Samples are stored in the
   original parameters. Other problems sample their free parameters directly. The start covariance is the
   Gauss-Newton estimate at the mode, pulled back to these coordinates.
2. Start 4 ensembles of walkers with their own seeds, spread over 3 times the Gauss-Newton widths (each width capped
   at a quarter of its range), max(2 d + 2, 16) walkers each. Moves: 90 % differential evolution, reflected at the
   prior's hard bounds (a symmetric move), and 10 % snooker moves (Ter Braak & Vrugt 2008, written here). The ensembles are stepped in lock-step: every half-step's proposals of all ensembles go to one batched cost
   call (23 rows per call instead of 7 on the toy star). Use at least 20 000 patches. Decision #19. That call is split
   into chunks of `uncertainty.COST_CHUNK` = 8 rows, so one mode keeps at most c = `uncertainty.pool_width` workers
   busy (ensembles x walkers / 2 / 8, rounded up). With n modes and W workers: one process per mode when
   n >= min(W, c), else the modes in turn and the chunks of each call go to up to c workers. The chunks
   are the same in every layout, so the chains do not depend on the number of workers. The fixed chunks cost the
   serial and one-process-per-mode layouts part of the batching gain (+10 s on about 104 s of
   sampling, 479 s against 469 s, in one measurement).
   **Batching (decision #25).** An ensemble whose step ends before another's (its move has fewer sub-steps: the
   differential-evolution move splits the walkers in two, the snooker move in four) starts its next step at once, so
   its proposals join the next call instead of leaving calls of a few rows. In the review a third of the calls carried
   1-10 rows, and a one-row call cost 37-43 ms against about 10 ms per additional row; on the toy star the calls of
   1-10 rows went from 223 of 865 to 34 of 734 (V5c). The remaining small calls are the walker redraws at the start
   and the ends of the convergence-check chunks.
3. Run until the V14 rules pass (VERIFICATION.md): at least 50 autocorrelation times, R-hat below 1.01 across the 4
   ensembles, effective sample size at least 400, no drift, no stuck walkers. If the rules still fail after
   `uncertainty.max_evals` evaluations (default 600 000, decision #20), flag the mode "not converged". The rules are
   also applied to each quantity on its own; a quantity that passes keeps its interval when the mode fails (UNC-2,
   decision #17).
4. If a polished sample is better than the mode by more than the value tolerance (section 3), hand it to the search
   (UNC-3), at most 3 times per mode.
5. Samples are mapped to the canonical form only for reporting. Intervals are computed in the canonical form (V15 lists the
   gated quantities), and a chain that reaches u = 0 is flagged. Intervals: 16/84 % and 2.5/97.5 % points of the
   samples. (The tempered half-widths were dropped, decision #24.)
   **Post-processing off the critical path (decision #25).** Derived quantities are computed for the thinned samples
   only (thinned to a quarter of the shortest integrated autocorrelation time), in vectorised batches; the intervals,
   KL values and convergence numbers of a mode are computed after its chains stop, in the worker pool, while the next
   mode's chains run. In the review this post-processing took a quarter to a half of the MCMC wall time.
6. "Not constrained by the data" (REP-4): for each sampled parameter, KL(posterior || prior) is estimated from
   histograms with 20 bins over the prior's range. The posterior samples are thinned to their effective sample size
   n_eff; the prior is 10 000 direct draws. This plug-in estimate is biased upwards by about (filled bins - 1) /
   (2 n_eff) nats: 0.08 nat at n_eff = 124, where a quarter of pure prior draws exceeded 0.1. So the flag is set when
   KL - (filled bins - 1) / (2 n_eff) < 0.1 nat (decision #19). Both values are written. Derived quantities get no
   KL value.
7. Profiles and envelopes are per family, not per mode (section 4b).

**Single phase: the orientation is a declared nuisance** (decision #18). With one phase the turn of the whole
geometry about the line of sight and the inclination change nothing in the spectra. Their coordinates are the azimuth,
u, and the components of the other vectors across the line of sight. They are still sampled, but left out of the V14
verdict, and their R-hat and ESS are reported as "nuisance, not required". The inclination and the raw components of
m_d and c get the flag "orientation not determined by one phase" instead of intervals. The reported quantities are
|m_d|, beta, the offset along and across the dipole axis, |c|, B10, B50, B90 and Teff. On a real single-phase fit the
orientation never converged (its autocorrelation time kept growing with the chain length), while those quantities
passed every rule. Any problem may declare other nuisance coordinates (`nuisance_coordinates`).

Cost: a real single-phase fit needed several times 1e5 evaluations per mode for the determined quantities, hence the
default of 6e5 (decision #20); the toy star converged at 1e5. The research's estimate, 1.5-3e5 evaluations per
ensemble at 20 autocorrelation times, i.e. 1.5-3e6 per mode at 50 times and 4 ensembles, was made before the
sampling coordinates of step 1; multi-phase problems will show which is closer. At 1e6 evaluations and 2-5 ms each,
a mode costs 0.5-1.5 core-hours; the configuration can limit MCMC to the best modes.
Why not the Laplace (curvature) approximation alone: in the research it was off by 2-6 sigma and 1.1-5.7 times too
narrow. It stays as a quick preview, labelled as such.

## 4b. Profiles and envelopes per family (UNC-5, UNC-8, decision #29)

- **Quantities.** `uncertainty.profile_quantities` (FORMATS.md section 3). Default: `incl_deg`, `dipole_MG`
  (|m_d|), `c_along_axis`, `B10_MG`, `B50_MG`, `B90_MG` (at the reference phase `search.fingerprint.ref_phase` only:
  decision #40; other phases when listed), and `teff_K` when Teff is free (the names of
  FORMATS.md section 7). With one phase, `incl_deg` is left out (decision #18) and `beta_deg` is added. A default
  quantity that no free parameter moves (q unchanged when each free parameter in turn moves by +-its step from the
  family's best node, e.g. `c_along_axis` with c fixed at 0) is left out: it cannot be profiled.
  A listed name is profiled as listed.
- **Holding a quantity fixed.** A quantity that is a coordinate of the search (Teff; the inclination through u) is
  fixed directly and the other coordinates are polished. |m_d| and the B quantiles (B10, B50, B90 per phase) are held
  exactly by scaling the field strengths (decision #39; `profiles.strength_indices`, `_Scaled`). B of
  "dqo" is linear in md, bq and mo, B of "dqo_aligned" in md, bq and bo (its axis is md's direction, which a positive
  factor keeps); aq is a direction only, and c and u are geometry. The
  patch weights (area, mu, limb darkening) and the visibility depend only on the geometry, and a factor s > 0 keeps the
  order of the |B|, so the weighted quantile and |m_d| scale by s (V1l, to 1e-12 relative, with a quadrupole, an
  octupole and an offset). Each row x of the polish stands for x with every free strength times q0 / q(x), so
  q = q0 exactly. The strength largest in |x| at the start is fixed, which removes the scale this makes redundant; the
  other parameters are polished. This needs every strength parameter free or fixed at 0 (a fixed non-zero strength
  would not scale), and a "dqo" or "dqo_aligned" field (a plugin field is not assumed linear); otherwise, and for
  every other quantity, the penalty holds q. A row whose scaled strengths leave their polish limits is invalid. When
  the start of a grid value, the previous point scaled to q0, would take a strength beyond its polish limit, the walk
  polishes at the q where that strength reaches the limit and ends there, a closed end "constraint not met". It does
  not look for a rotation of m_d that would pass the limit; the penalty walk could (on V1l's case with md limited to
  +-420 MG per component it reached |m_d| = 518 MG, past the configured length 420; the scaled walk closes at
  459 MG). Residual roughness: q(x) still jumps with the geometry (the quantile's step between neighbouring patches),
  so the factor q0 / q(x), and with it -ln L, jumps along the geometry directions. On the synthetic problem of V7j at
  1 % below the best B50 of phase 0, -ln L deviates from a quadratic over +-0.01 parameter steps by 0.086 nats at
  1000 patches, 0.002-0.05 at 5000 and 0.007-0.01 at 20 000. Polishes still stop at such jumps: from the truth the
  walk point lies 0.07 nats above a direct minimisation at 1000 patches, 0.26 at 5000 and 0.18 at 20 000 (0 to
  0.2 nats at 2 % below). So the B-quantile profiles can lie above the true profile and their intervals be too narrow;
  every B10/B50/B90 entry carries the flag "B quantile is a step function: provisional" until the quantile is made
  continuous in the parameters (open; decision #39). |m_d| is continuous and is not flagged. For a quantity held by the penalty, q is held at q0 by adding the penalty
  kappa ((q(x) - q0) / dq)^2 to -ln L, with kappa = 1000 nats and dq the quantity's grid step; the profile value is -ln L
  without the penalty, and a point with |q(x) - q0| > 0.1 dq has missed the constraint (see "Walks" for what follows).
  Polishes here use L-BFGS-B on the penalised objective, then the Levenberg-Marquardt finish with the penalty as one
  extra residual.
- **Grid steps** (`uncertainty.profile_steps`): 2.5 degrees in inclination and beta, 1 % of the family's best value for
  |m_d| and the B quantities, 0.01 stellar radii for offsets, 100 K for Teff. The grid is q_best + k dq (k integer),
  q_best the quantity at the family's best node; a walk from another member starts at the grid value nearest its own.
  A step in percent, p %, is taken of max(|q_best|, s_q). Here s_q is the quantity's own scale: the largest change of
  q over one configured `step` of any free parameter from the best node. It costs n_free + 1 evaluations of q, once
  per family and quantity. Reason: p % of a best value near 0 (|m_d| or B10 near 0, the inclination near 0)
  gave a step of 1e-9, and the walk never left its start. The rule changes nothing while |q_best| >= s_q, which holds
  for the default steps at any field the search resolves; it uses only the configured parameter steps, so it adds no
  new number. When no free parameter moves q (s_q = 0 and q_best = 0) the step is p / 100 in the quantity's unit.
- **Walks.** A walk starts at a member and steps outwards in both directions, each grid point polished from the
  previous one (continuation), until the profile exceeds the family's best by max(2 h, 10 nats) (h of section 3b), or
  the range of the quantity ends, or after `uncertainty.profile_max_steps` (default 40) steps per direction. The range
  ends (a closed end): for a quantity fixed directly, when the next grid value lies beyond
  its range (the inclination 0-90 degrees, Teff the polish limits), that range end itself is polished and the walk
  stops there; for a quantity held by the penalty, at a point that misses the constraint where q cannot reach q0
  (|m_d| at 0, a polish limit, the library's field range), which stops the walk at the q it reached. Before this rule
  the walk went on to `profile_max_steps` through values the model cannot take, and those points entered the
  intervals (on a toy with q >= 0, best 0.3, delta_0.5 came out [-9.7, 3.3] instead of [0, 3.3]).
- **A missed constraint** (for the quantities held by the penalty: the scaled hold always meets q0). With kappa = 1000 nats a well-converged polish misses a reachable q0 only
  where the profile rises by more than about 200 nats per grid step, far above the stop level. But a polish can stall
  short of q0: under the LM stops of decision #37, and wherever q is not smooth on the scale of the finite-difference
  step. The B quantiles are such a quantity: a weighted quantile over the patches jumps when two neighbouring patches
  swap places (about 0.3 MG at 1000 patches, on the synthetic problem of V7j), so central differences of the penalty
  over 1e-4 steps come out up to 250 times too large and the polish stops where it started. After a miss, one search
  from the polish's end point towards q0 tells a limit from a stall: each free parameter in turn moves by +-its `step`
  within the polish limits, those that move q towards q0 give a direction, and a secant search along it (at most 8
  evaluations of q, no likelihood evaluations) looks for q = q0. The end is closed ("constraint not met") only where
  this gives evidence of a limit: no free parameter moves q towards q0, or the search reaches the polish limits of
  every parameter that does while still more than 0.1 dq short. Otherwise the miss is polished once more, with both
  LM stops off (`search.lm_stop_nat` = 0, `search.lm_max_rejections` = 0), from the point of the search nearest q0;
  at a limit there is no retry, since no polish can pass it (earlier the retry ran there too, and a second
  search from its end point decided; on V1l's toy q = p >= 0 the profile's evaluations fall from 10 503 to 10 475).
  If the retry meets the constraint, the point is an ordinary point of the walk. If it misses too, the point is a
  stalled polish: it is kept in
  profiles.json with `met` false and `stalled` true, it is not a value of the profile, nothing is closed, the quantity
  carries the flag "polish stalled", and the walk goes on from its last point that met the constraint to the next
  grid value, so that values beyond the stall are still profiled. After 2 stalled points in a row the direction ends,
  open. A walk back meets a stall with the same retry, so it cannot close the side there either. The flag
  "constraint not met" means a closed end in the entry's `ends` and nothing else. Cost: one more polish per stall;
  on V1l's rough toy 2050 evaluations before the rule (both sides wrongly closed within 2 grid steps) and about 7e4
  after, against 3.8e4 for the same toy without the roughness.
- **Walk starts.** Each quantity is
  walked from at least 2 members: the family's best end point and the end point whose value of the quantity is
  farthest from it. If the family has one end point, the second walk starts from each far end of the first and walks
  back; it stops above the stop level only after it has been below it, so it crosses the whole profile. The reported
  profile is the lower envelope of all walks at each grid value: the lowest point that met the constraint. A grid
  value where no walk's point met it is not a value of the profile and is left out of the envelope, the intervals and
  the minima (a gap between the grid values, open on both sides; a closed end beyond it bounds the intervals). Reason
  for several walks: in the review a single walk from the best point reported the wrong branch by 10-15 nats
  (hysteresis); walks from three members agreed to 0.5 nat at the far end.
- **Intervals.** From the lower envelope, linearly interpolated between grid values: the ranges where it stays within
  0.5 nat and within 2 nats of the family's own best, and the range where it stays below the family level f_best + h
  (section 3b; one level for all families, the one that defines membership). A range that reaches the end of a walk,
  or a gap between the grid values of the walks (walks cut short by `profile_max_steps`, a grid value where every
  walk's polish stalled), is open on that side and
  flagged; nothing is interpolated across a gap. A range that reaches a closed end of the quantity's range (beyond the
  last grid value on that side) is closed: it ends at the end's q when the end's -ln L lies at or below the threshold,
  else at the linear crossing between the last grid value and the end.
- **Minima** (the family sentence): points of the envelope, or the middle of a run of equal values, from which it rises
  by at least 0.5 nat on each side before falling below them; a side that first reaches an open grid end or a gap does
  not count (a closed end of the range does).
- **Envelope.** The range of the quantity over the family's nodes, the envelope's grid values and the closed ends (at
  the q they reached) whose -ln L lies below the family level.
- **Side by side** (UNC-8): per quantity, the envelope, the three profile intervals and the per-mode MCMC 68 % interval
  of the family's best mode (UNC-1), and two ratios: R1 = the envelope's width / that interval's width, printed and
  not flagged; R2 = R1 / sqrt(2 h), flagged above 1.5 (decision #30; D13 settled 2026-10-23). When the family's best mode has no
  interval for the quantity (no MCMC run, "not converged", or a declared nuisance), both ratios are null and the
  table says why. For a Gaussian basin with standard
  deviation s, -ln L rises by z^2 / 2 at z s, so the envelope below h spans 2 sqrt(2 h) s against 2 s for the 68 %
  interval: R1 = sqrt(2 h) and R2 = 1. The report also writes two sentences per family and quantity, one per view, from
  templates (FORMATS.md section 6), so that the difference in what a reader takes away is visible; the mode sentence is
  about the family's best mode.
- **Cost.** About 1.7e4 evaluations per walk in the review (380 s on a loaded machine at 5000 patches); here at the
  error-bar patch count. A list of 3 + 3 x (phases) quantities, plus Teff, has 13 entries for 3 phases and 28 for 8 phases; with 2 walks
  each, about 4e5 and 1e6 evaluations per family (the default list is shorter, decision #40). Walks are independent
  and run in the worker pool: each direction of each walk (family, quantity, start member, direction) is one task of
  the search's pool (`--workers`; single-threaded workers, ENG-2), mapped in a fixed order, so the
  profiles do not depend on the number of workers (V6d); the walks back of a one-member family follow as a second
  batch on the same pool. Holding |m_d| and the B quantities by scaling instead of the penalty cut the profile stage
  on the synthetic problem of V7j (8 default quantities, 1000 patches) from 168 228 to 69 831 evaluations, 2.1-4.0
  times fewer for each of the 7 quantities concerned (355 s -> 273 s on a loaded machine: each evaluation of the
  scaled view also computes q, as the penalty did).

## 5. Likelihood "full"

For each spectrum s (one phase and one Stokes quantity, n_s pixels, about 900), with r_s = data - model:
- **Covariance.** C_s = sigma_s^2 x D K_s D, where D is the diagonal matrix of the pixel errors divided by the square
  root of their weights (DATA-4; a pixel of weight 0 is removed) and
  K_s = M(rho_short) + a^2 M(rho_long) + j^2 x identity. M(rho) is the Matern-3/2 kernel,
  M_ij = (1 + sqrt(3) d_ij / rho) exp(-sqrt(3) d_ij / rho), with d_ij the wavelength distance of pixels i and j. j is
  the white part: 0.05 in the initial kernel (a floor that keeps K well conditioned), fitted in [0.05, 10] by the
  kernel fit, because a kernel without a fitted white part cannot describe white noise (the whiteness check then
  failed on simulated white data in the plan review). The share of the short term in the correlated variance is
  1 / (1 + a^2). The long term is the model-discrepancy term of LIKE-4.
- **Kernel fit** (LIKE-2, decision #22). Per spectrum, (rho_short, a, rho_long, j) maximise the marginal likelihood of
  that spectrum at a fixed model (the best end point of the search, evaluated at the error-bar patch count), with the
  noise scale and the continuum integrated out as below; ln det K_s, a constant while the kernel is fixed, is part of
  this objective. Bounds: rho_short in [0.5 pixel, FWHM], rho_long in [FWHM, a quarter of the spectrum's longest fit
  window] (rho_long = FWHM when that quarter is shorter), a in [1e-3, 10], j in [0.05, 10]. Optimiser: L-BFGS-B in
  (ln rho_short, ln a, ln rho_long, ln j) from 3 starts (the initial kernel, and both lengths at the geometric
  mid-points of their bounds with a = 1, j = 0.3 and with a = 3, j = 0.1), the best kept. The initial kernel, used by the search before the fit, is `objective.corr` and
  `objective.disc` (4 A; a = 0.71, 85 A for I; a = 0.42, 205 A for V/I: the research values for RE J0317-853). With
  `objective.kernel_fit = false` the initial kernel is used throughout (tests, and re-runs with a kernel copied from
  an earlier `run.json`). In the review, on a real low-field object, two fitted Materns (a short length of a few A and
  a long one of tens of A) were much better per spectrum than the default kernel, one Matern less so, and Gaussian
  kernels were worse than the Matern.
- **Whiteness check** (LIKE-10). With L_s the Cholesky factor of K_s, the whitened residual is
  e = L_s^-1 D^-1 (data - model - best continuum) / sigma_hat_s. Its autocorrelation at lag k counts the pairs of
  pixels k data-grid steps apart that are both fitted with every pixel between them fitted (no pair spans a gap):
  rho_k = sum e_i e_(i+k) / sum e_i^2. Pass when max over k = 1..10 of |rho_k|
  < 3 / sqrt(n_s) + 0.05. For white noise each rho_k has standard deviation about 1 / sqrt(n_s), so a white spectrum
  of 900 pixels (threshold 0.15) fails with a probability of about 7e-5; an AR(1) residual with coefficient 0.3 fails
  at n_s = 900 almost surely. In the review, on real spectra, the default kernel failed this check, one Matern sat at the
  threshold, and two Materns passed it clearly.
- **Noise scale.** sigma_s is unknown. A Jeffreys prior (uniform in log sigma) is integrated out in closed form. Because
  sigma_s multiplies the whole covariance, the integral is exact.
- **Continuum of I.** The model I is multiplied by a Legendre series of degree 3 by default (`objective.continuum.degree`;
  P_0 to P_3, 4 coefficients) in wavelength (the fitted pixels of the spectrum mapped linearly onto [-1, 1]), and a
  constant is added where the zero point of I is unknown. These k_s coefficients (4, or 5 with the constant) get a flat
  prior and are integrated out in closed form. k_s = 0 for V/I. It replaces a normalisation through continuum node
  windows, which biases parameters. **Normalised basis** (decision #26): the whitened columns W_s (P_k(x) m) are
  divided by nu_s = |W_s m| / |W_s 1|, the size of the model measured with the whitening (it scales with the model,
  does not change when all errors are scaled, and needs no unwhitened model, which the pre-whitened path never
  forms), so the flat prior is on coefficients of a model of unit size; then
  -ln L does not change when the model I is multiplied by a constant c (before, ln det(A^T A) added (degree + 1) ln c
  per I spectrum, several nats between neighbouring Teff slices at a fixed geometry). The constant
  column is not normalised. `continuum_off` is one refit with `objective.continuum = false` for every I spectrum (the
  free constants dropped too); it always runs, and the report prints each I spectrum's `scale_ratio` next to its
  shifts, because "off" means something only when the I scale is known (absolute flux, or data scaled to the model).
- **Result.** With whitened quantities (multiplied by the inverse Cholesky factor of K_s): A_s = the design matrix of the
  continuum terms, chi2_s = the whitened residual after the best continuum. Then
  -ln L = (1/2) sum over s of [(n_s - k_s) ln chi2_s + ln det(A_s^T A_s)] + a constant.
  A_s depends on the model through the model I, so the ln det term changes with the parameters and must be kept.
- **Invalid models** get -ln L = 1e30, a finite value; optimisers treat it as "step rejected", and the run counts it.

Calibration in the research: on simulated data from the same model, the 68 % intervals covered the truth 90 % of the
time (too wide). Those were Laplace intervals started at the truth, and the injected noise had no discrepancy part, so
they do not yet say how the MCMC intervals will do (V15 measures that). With data from another library or temperature,
coverage fell to 0.27-0.35, and the temperature T needed to fix it ranged from 0.4 to 170 between parameters (13-17 on
average). This is why systematic shifts are separate (SPEC SYS-1), why no tempering factor is applied (decision #24),
and why "calibrated" needs injections from another library (UNC-4, V15b).

## 5b. Convolution and resampling

- Convolution: a Gaussian with sigma = FWHM / (2 sqrt(2 ln 2)) = FWHM / 2.3548, cut at 4 sigma, normalised to a sum of 1,
  applied once to each 1 A library record. The fit windows MUST lie at least 4 sigma inside the library's wavelength
  range; otherwise the run stops (LIB-7), so the ends of the spectrum never matter.
- Resampling: each output pixel is the mean of the input over the output pixel's width, with partial input pixels
  weighted by their overlap (the exact bin average).
- Wavelength shift (FWD-13, decision #27; not built before M7: T2.3 left it out). A fixed shift (a velocity v, or a
  constant in A) is folded into the start-up matrix: the library wavelengths are multiplied by (1 + v / c), or the
  constant is added, before the convolution and resampling matrix is built; a positive v is a redshift. A free shift
  (parameter `shift`, global; `model.shift_unit` = "kms" or "A") gets records at `model.shift_nodes` (default 5)
  shift values evenly over the parameter's [lo, hi], each built exactly as a fixed shift; a row's model combines them
  with cubic Lagrange weights in the shift (the 4 nearest nodes; exact at a node), the same way as the Teff slices
  (each pair of Teff slice and shift node is one record set). An exact per-row shift would need the records on the library's 1 A grid, about 190
  MB per Teff slice for a real library, against ENG-6; the nodes cost `shift_nodes` times the records. The data are
  never interpolated. A shift per phase (`shift_<k>`) is not built. V12 is re-measured with a free shift.

## 5b2. Floor chi2 (DIAG-1, decision #23)

Per phase, the floor chi2 is the smallest plain chi2 (the errors and weights of LIKE-6; no kernel, no continuum, no
noise scale) that any non-negative mixture of library records reaches: min over c >= 0 of
|D^-1 ([I_obs; (V/I)_obs x I_obs] - sum_k c_k [I_k; V_k])|^2 over the phase's fitted pixels, where the columns k are
every convolved and resampled record (each node in B and cos psi) of the bracketing Teff slices, and the errors of the V
rows are sigma_VI x I_obs. V/I is replaced by V because V/I of a mixture is not linear in c. Solved by
`scipy.optimize.nnls` on the error-scaled system. It is written per phase and per Stokes part (the I rows and the V rows
of the minimum), next to the best mode's plain chi2 at the same pixels for the model I written to `spectra/` (its
continuum as fitted inside the likelihood). The two are not ordered: the floor has no continuum, and under a
fitted kernel the mode's continuum minimises the whitened, not the plain, chi2, so with a large long kernel (a near its
cap) the gap also holds continuum freedom that the kernel allowed. Without a kernel the gap shows how much any field
model could still gain on these records. The columns are the
records of the Teff slices that bracket the best mode's Teff: all selected slices would make the system too wide for
NNLS on real libraries (thousands of nodes per slice). They are taken at the best mode's wavelength shift (FWD-13,
section 5b): a fixed shift is built into the records; with a free shift each slice's records are the record sets of
the shift nodes combined with the cubic Lagrange weights of the mode's shift, as the forward model combines them.

## 5c. Teff interpolation (LIB-3, decision #26): continuum-matched ("planck") weights

For a Teff between two slices T1 < T2, the model is (1 - w) S1 + w S2, where S1 and S2 are the disc sums (I and V)
of the two slices on the data grid and w is a weight per data pixel:
- **"planck"** (default): w(lambda) = (P(T, lambda) - P(T1, lambda)) / (P(T2, lambda) - P(T1, lambda)), with P the
  Planck function B_lambda. This is exact for any spectrum of the form g(lambda) x P(T, lambda), so the continuum
  follows Teff the way a Planck continuum does, and the lines are averaged with the same weight. It costs nothing: the
  matrix products per slice are unchanged, only the weights per pixel differ. In the review, continuum-matched
  weights cut the I chord error 3.7 times against weights linear in ln Teff; the V/I error (1.5-3 % at 4 kK spacing)
  comes from the lines, and only finer slices remove it (1000 K slices: decision #26).
- **"log"**: w = (ln T - ln T1) / (ln T2 - ln T1) at every pixel (the rule before decision #26).
The polish's Jacobian (central differences, section 3) sees the weights' dependence on T through the forward model.

## 5d. Libraries with a mu grid (LIB-8, decision #27)

With NMU > 1 (FORMATS.md section 1), each patch's spectrum is interpolated linearly in mu between the two bracketing mu
values, together with the interpolation in B and psi, and the limb-darkening factor is 1: the patch weight is area x
mu only. A patch with mu below the smallest library mu takes the smallest-mu spectrum; their brightness-weighted area
fraction is written to `diagnostics.json`. Memory grows with NMU (about 54 MB per mu value per Teff slice); ENG-6 still
applies, and a library that would break it stops the run with a message.

## 6. Data layout outside the repository

`$ZEEMAP_DATA` points to a directory outside the repository:
```
$ZEEMAP_DATA/
  MANIFEST.sha256                       checksums of every file below; tier-2 checks verify it first
  libraries/<name>/info.dat, <Teff>.bin
  observations/<object>/<dataset>/      FORMATS.md section 2
  references/                           regression references on real data (V10)
```
Libraries and observations are not distributed with zeemap; each user supplies their own.

## 7. Speed, memory and parallelism

- Reference machine: a 14-core workstation of the maintainers. Its CPU model is recorded in `run.json`, and V12's
  10 ms rule applies only there.
- One process per core, each single-threaded (numba, BLAS and OpenMP threads set to 1).
- Each start keeps a fixed batch layout, and gradients are computed in float64, so results do not depend on the number
  of workers (V6).
- Seeds: one master seed; `numpy.random.SeedSequence(seed).spawn(n)` gives one independent stream per start, ensemble
  and injection. Random numbers are drawn in numpy and passed into numba code, because numba has its own random state.
- Measured: 4-7 ms per evaluation at 5000 patches with one process; 2-3 times slower with 8 processes at once (memory
  bandwidth); 400-500 evaluations per second on 4 cores.
- Records as two factors (decision #41): the disc sum `W @ R` of a record set (nodes x 2M) is 60 % of an
  evaluation on a library like RE J0317's (8500 nodes, 878 pixels), and at one row it is bound by streaming R from
  memory. At load each set is factored, R = A B (A orthonormal, nodes x r, by a truncated SVD with each Stokes part
  scaled to unit norm), with the smallest r that holds each part to `forward.COMPRESS_TOL` = 3e-8 relative, below the
  float32 rounding of the stored library; when the two thin products would cost more than half the full one, the
  set stays as it is. The records are replaced by A B, so all paths and checks use one model; the pre-whitened
  matrices of `likelihood.bind_forward` are built from B (r rows instead of the node count). Load cost: the full SVD
  runs before r is known, about 14 s per record set at the RE J0317 size (8500 x 1756,
  single-threaded), so n_teff x n_shift SVDs per process start (fit, eval, inject, every library swap), paid also when no r
  passes; a fit of many hours does not notice it. The factors are cached per process by the records' hash
  (`forward._FACTORS`) and never released: tens of MB per record set, for every library the process loads.
- Memory: about 54 MB per Teff slice. The prototype used 1.4 GB per process at start-up; the cause is not yet measured
  (dense whitening matrices account for only about 0.1 GB; a dense convolution matrix for about 0.3 GB). Task T2.2
  measures it first.

## 8. The research prototype

**Where:** not part of this repository; it was written during the research before zeemap and is kept only as a record.

**What it has:** 14 modules matching section 1 (with `normalise` and `problem` split out), about 1950 lines, 13 passing
tests.
**What it lacks:** a search run at scale, MCMC and swap runs, multipoles, phase smearing, relative errors and
derived V/I errors, the wavelength shift and banded whitening.

It was decided on 2026-10-18 to rewrite from this plan alone (decision #14); the prototype is not copied in, and task T1.1 is skipped. It
stays a research record only.

## 9. Using zeemap for fits that a paper cites

A project that cites zeemap fits pins zeemap at a tagged release or a recorded commit, builds the environment from
`requirements.lock`, records the library, data, configuration and seed of each run, and keeps the run directory
(`run.json` records the version, the inputs' checksums and the seed). A rerun is compared with the first at tolerance
(`modes.json`, `spectra/`), not byte for byte (GATES.md, gate C).
