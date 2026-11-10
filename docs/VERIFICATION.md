# How we know zeemap works (verification loops)

**Status:** describes zeemap 0.1.0; the history of its decisions is in [DECISIONS.md](DECISIONS.md).
**Who this is for:** anyone who changes the code, reviews a change, or wants to trust a result.
**What you will know afterwards:** which checks exist, what each one proves, its pass rule, and when it runs.
**Words:** see [GLOSSARY.md](GLOSSARY.md). The requirements are in [SPEC.md](SPEC.md). "Check" and "loop" mean the
same thing here: a test or test campaign with a pass rule, which is run again and again.

## 1. Three ways a fitting code can be wrong

1. **The model is computed wrongly** (a sign, a unit, an off-by-one). Caught by comparing with answers known exactly:
   formulas, symmetries, a synthetic library, and a second slow implementation.
2. **The search misses the best mode.** Caught by planting a known answer in simulated data and checking that the
   search finds it at the stated budget.
3. **The error bars are too narrow or too wide.** Caught by simulating many data sets from known geometries and
   counting how often the true value falls inside the interval.

## 2. When each check runs

| tier | when | time budget | needs real data? | checks |
|---|---|---|---|---|
| 1 | while working (`--fast`), before opening a pull request, and in CI on every pull request | about 10 min on 4 idle cores; `--fast` (without the tests marked `slow`) about 4 min | no (synthetic data only) | V1-V8, and the tests of V14 |
| 2 | before merging a change that can move numerical results, and before a release (GATES.md) | 20 min | V10 (real-data part) and V11 do; V12 and V13 do not | V10-V13, plus the extra checks for the kind of change (GATES.md) |
| every fit | inside every `zeemap fit` run with MCMC (`--mcmc`, decision #42) | part of the run | the run's own data | V14 |
| 3 | before a release, and before a result goes into a paper | days on 12 cores | V15, V17c, V18b and V18c do | V15-V19 |

**Running checks.** `tools/check.sh N` runs every tier up to N, so `tools/check.sh 2` runs tiers 1 and 2.
`tools/check.sh N V<k>` runs one check, and `tools/check.sh N V<k><letter>` one item of it (for example `V2c`). Each
item below has a label such as **V2c**; the tests for it are named `test_v2c_...`. The runner fails when a selection
finds no test (naming each item that selected none), and when the item belongs to a higher tier than N.
`tools/check.sh 1 --fast` leaves out the tier-1 tests marked `slow` (the end-to-end fits) and prints
`SUMMARY tier 1 (fast): ...`; use it while working, and the full tier 1 before opening a pull request (decision #38, which replaced the earlier
budget of 2 min for the whole tier).

## 3. Tolerances: "agrees" means "agrees within a stated tolerance"

Results are compared with a tolerance, never byte for byte. The order of floating-point sums changes with compilers and
batch layout, and the library is stored in single precision (float32, about 7 digits).

| comparison | relative tolerance | absolute floor (for V and V/I, which cross zero) |
|---|---|---|
| pure arithmetic in float64 (formulas, identities) | 1e-12 | 1e-15 |
| the same computation in two ways (numba against plain numpy) | 1e-10 | 1e-13 |
| anything that passes through float32 library values | 1e-5 | 1e-6 |
| a quantity that depends on the patch grid (quadrature) | 1e-3 at 5000 patches | 1e-4 |
| a full fit rerun, search included | the same modes (by fingerprint, DESIGN.md section 3), each within the value tolerance of DESIGN.md section 3 | |

zeemap's own tests use the levels above. A project that compares zeemap outputs between its runs may use coarser
levels of its own (for example 1e-6 for the same code, 1e-5 through float32, 1e-3 for a physics bound).

## 4. Synthetic test data

Tier 1 must run anywhere, so it cannot use a real library. `tests/synth.py` (task T0.3) writes a small
**synthetic library** in the library format (FORMATS.md section 1), fully fixed by these numbers:
- Teff: 20 000, 30 000, 40 000, 50 000 K; files `20000.bin` ... `50000.bin`;
- B: 31 values, 50 to 800 MG in steps of 25 MG;
- psi: 9 values, 0 to 90 degrees in steps of 11.25 degrees;
- wavelength: 4000 to 7000 A in steps of 1 A, both ends included (3001 points); the wavelength file is `-`;
- I(lambda) = (Teff / 30 000 K) x [1 - sum over k = 0..4 of 0.3 exp(-(lambda - lambda_k)^2 / (2 x 4^2))], with
  lambda_k = 4500 + 500 k + 0.02 (k - 2) B (lambda in A, B in MG); so line 2 does not move with B;
- V(lambda) = 0.01 x cos psi x dI/dlambda x (1 A), with the derivative taken analytically.

Because the formulas are known, many results can be computed by hand. Nothing in it comes from MAGWD. A second
synthetic library, `tests/synth.py --variant b`, uses 0.022 instead of 0.02 for the line shifts; V18a uses it as a
stand-in for "another library version".

## 5. The checks

Every pass rule uses the tolerances of section 3 unless it says otherwise.

### Tier 1: before every commit (synthetic data)

**V1. Known answers of the small parts.**
- What: each formula is implemented right.
- Items:
  - V1a: field of each component against the formulas of DESIGN.md section 2, and against a numerical gradient of its
    potential (central differences, step 1e-5, agreement 1e-7); for "harm4", each of its 24 field numbers against the
    gradient of a Schmidt potential written with scipy's lpmv, its value on the z axis against the limit next to it,
    and its l = 1 part against the centred dipole;
  - V1b: Fibonacci patch areas sum to 4 pi;
  - V1c: the visible projected area (sum of area x mu over visible patches) is pi within 1e-4 at 5000 patches, for each
    of 100 fixed observer directions; and the rms relative error over those directions falls by at least 2 when the
    count is quadrupled from 5000 to 20 000 (the prototype's Fibonacci grid gives a factor of about 5.6);
  - V1d: each limb-darkening law integrates over the disc to its closed form;
  - V1e: interpolation at a library node returns that node; linear interpolation halfway between two nodes returns
    their mean; the PCHIP option is exact for data linear in B and has a continuous first derivative at the nodes
    (checked numerically, 1e-8);
  - V1f: for "harm4", the equator mirror (B'(M p) = M B(p)), `rotated` (B'(p) = R B(R^T p)),
    `Problem.mirror` and, with one phase, `nuisance_coordinates`;
  - V1g: closed-form integration over the noise scale and the continuum equals brute-force numerical integration on a
    small problem, for differences of -ln L between parameter sets (to 1e-6);
  - V1h: chi2 of a model against itself is 0; chi2 of a known residual vector r is sum (r / sigma)^2;
  - V1i: errors: absolute and relative errors, floors, the V/I error derived from the I error;
  - V1j: fit windows and excluded ranges select exactly the expected pixels;
  - V1k: the configuration is written and read back unchanged; every key in `defaults.toml` has a comment;
  - V1l: profiles (DESIGN.md section 4b): a toy -ln L with a known profile gives that profile at the grid values
    (1e-6); on a toy with two branches, where a walk from one member follows its own branch past the crossing (the
    hysteresis of the review), the lower envelope of the walks from two members equals the true profile, and the
    0.5-nat, 2-nat and family-level intervals equal their closed forms to one grid step; the envelope of UNC-8 equals
    the range of the toy's points below the level. All of it through the real walks (`profiles.family_profiles` on toy
    problems with `cost`, `residuals_batch`, `residual_scales` and `derived_batch`), for a coordinate fixed directly and
    for a quantity held by the penalty (closed form P(q0) / (1 + dq^2 / (2 kappa V))^2), with one family member (the
    walks back cross the whole profile) and with two. Also: an interval stops at a gap between the walks' grid values
    and is open there; the envelope takes the lowest point that met the constraint and leaves out grid values where none
    did; a local minimum needs a rise of 0.5 nat on both sides and is not an open grid end or a point of a plateau.
    Closed ends: on the toy q = p >= 0 held by the penalty, best at 0.3, the walk down
    stops at its first point that misses the constraint (with no retry, since the search towards q0 finds the limit, and
    with no stalled point), no grid value below 0
    enters the envelope, the 0.5-nat and family-level intervals and the UNC-8 envelope are closed at 0 and the flag
    "constraint not met" is set, as on every toy exactly when `ends` holds such an end; an
    inclination whose profile stays low down to 0 degrees gives intervals closed at 0 (the walk polishes at 0 and
    stops); with c fixed at 0 the default list leaves out c_along_axis (kept with c free), and `profile_quantities =
    "default"` gives that list; each profiles.json walk names its start (role, node, mode, grid index) and lists per
    point k, q0, q, -ln L, met and stalled. Missed constraints: on a toy with q = p + a square wave
    (jumps of 0.01 every 1.5e-4, as a B quantile jumps between patches) and no limit, where the first polish misses
    most reachable grid values, the retries reach them: no closed end, no flag, the walks reach the stop level on both
    sides, the envelope is the toy's profile to 0.03 nats and the family-level interval its closed form to one grid
    step (before: both sides closed within 2 grid steps); on the quadratic toy with every polish at one grid value made
    to stall while an LM stop is on, the retry with the stops off reaches it (closed form to 1e-6 there) and nothing is
    flagged; made to stall always, that point is written with met and stalled set, the flag "polish stalled" is set and
    "constraint not met" is not, no end is closed, the first walk and both walks back go on past it, and the intervals
    that reach it are open there ('<' in the side-by-side table, whose legend explains the flag); made to stall at
    two grid values in a row (k = -2, -3), the walks down end there (`MAX_STALLS` = 2: no walk reaches k = -4) and the
    lower side is open at k = -1; a polish planted just short of the bound of q = p >= 0 (p = 0.01 at q0 <= -0.2),
    where p still moves q, closes the side at q = 0.01 with "constraint not met" and no stalled point, by the second
    limit test (the search reaches the polish limits of every parameter that moves q);
    a step "<p>%" at a best value of 0 is p % of the quantity's own scale s_q (DESIGN.md section 4b), not 1e-9, is
    unchanged when |q_best| >= s_q, and a malformed percentage raises. Held by scaling (decision #39,
    on the synthetic library): scaling every field strength by 0.7 or 1.3 scales dipole_MG and the B quantiles of
    both phases by the same factor to 1e-12 relative, with a quadrupole, an octupole and an offset present, and a strength fixed at a non-zero value turns the scaled hold off; B50_MG[0] walked from the truth
    meets q0 to 1e-9 and lies within 0.1 nat (the roughness the quantile's jumps leave) of a direct minimisation with
    |m_d| scaled to q0 at 1 % and 2 % below its best (held by the penalty: 0.73 against 0.09 nats); with md limited to
    +-420 MG per component, dipole_MG walked upwards closes at the q where md_x reaches the limit;
  - V1m: noise model: kernel entries equal the Matern-3/2 formula; for a 20-pixel spectrum and 10 000 whitened noise
    draws with a fixed seed, every entry of the sample covariance is within 5 standard errors of the identity (a correct
    implementation then fails with probability about 1.2e-4);
  - V1n: resampling a linear spectrum onto a grid whose pixel centres lie on whole or half angstroms (as in our data)
    gives its value at each pixel centre; the convolution of a constant spectrum away from the ends is that constant;
  - V1o: kernel fit (LIKE-2): the objective equals the exact Gaussian marginal likelihood with ln det K included, against
    a brute-force evaluation on a 30-pixel spectrum (1e-8); for 8 noise draws (fixed seed, 300 pixels) from a known
    two-Matern kernel (2.5 A and 40 A, a = 2, 2.5 A pixels), the fitted maximum is never below the value at the true
    kernel (it is a maximisation, so a correct optimiser passes always, up to 1e-6 nat), and the median fitted
    rho_long over the 8 draws lies within a factor 1.5 of 40 A, the median rho_short within a factor 1.5 of 2.5 A and
    the median a within a factor 1.5 of 2; for 10 white Gaussian series of 400 pixels (fixed seed), every fitted
    kernel's whitened residuals pass LIKE-10; the fitted values are written to `run.json`; a fit of white noise ends
    with j at its upper bound and one of a 5-sigma smooth misfit with a at its upper bound (rho_long and j free), each
    named with its side; the plain chi2 per spectrum (DIAG-1's chi2 of the mode) equals that of the model
    I with the continuum fitted under a given kernel (a = 10), to 1e-8;
  - V1p: whiteness check (LIKE-10): 200 white Gaussian series of 900 pixels (fixed seed) all pass (each lag's
    autocorrelation has standard deviation 1/30 against the threshold 0.15, so a correct implementation fails one of
    the 200 with probability about 0.014 over seeds); 20 AR(1) series with coefficient 0.3 all fail; at n = 400 the
    threshold is 0.2 (the rule depends on n, not a fixed number);
    pixel pairs across an excluded range or a gap between windows are not counted (also for isolated pixels whose
    array neighbours are equal, which fail the check if the contiguity rule is removed);
  - V1q: the Jacobian of the whitened residuals in the LM finish (DESIGN.md section 3, polish; central differences at
    `search.fd_step` in one batched call) equals the same differences taken one coordinate at a time (relative 1e-9;
    absolute floor the larger of 1e-9 of the largest element and 4 d / (2h), with d the batch-vs-single rounding of
    the residuals measured in the test and itself at most 1e-11 of the largest residual: multithreaded BLAS rounds
    batches of different sizes differently), and a
    Richardson extrapolation from half-steps 1e-3 and 5e-4 to 1e-3 relative in the Frobenius norm (the reference's
    larger steps cross kinks of the piecewise-linear interpolation: 1.7e-4 measured), on the synthetic library at 3
    random geometries, with Teff free and both Teff weight rules; on a toy whose minimum lies on the polish limits
    (invalid models beyond them), the Jacobian at a corner of the limits takes one-sided differences (no invalid row
    in any column; equal to the exact Jacobian, 1e-8) and the LM finish alone reaches the bound-constrained minimum
    (1e-9 nats); on a synthetic library whose line depth grows with Teff, with both Teff weight rules, the Teff column
    of the Jacobian and the Teff component of the L-BFGS-B gradient of `polish` are nonzero, and more than 0.2 times
    their size 2 K away, at a slice (30 000 K), 0.3 K from it, at both polish limits (the edge slices 20 000 and
    40 000 K) and 0.3-0.4 K inside them (before, they were exactly 0 within 0.4 K of a slice); Teff limits
    written up to 0.5 K beyond the edge slices (19 999.6 and 40 000.4 K) give polish limits on the edge slices, and
    there the same two quantities pass the same test (at the unclamped limits both were 0); a Teff
    range wholly beyond an edge slice, free (lo 40 000.2, hi 40 000.4, value 40 000.3 K; also 19 999.6-19 999.8 K) or
    fixed (40 000.3 K), becomes lo = hi = value = the edge slice with one note in `limit_notes`, and a polish from it
    runs (before, free, it raised "An upper bound is less than the corresponding lower bound"; fixed, lo > hi);
    a value alone beyond an edge (19 999.7 K) is clamped to it, and limits inside the range give no note;
- Covers: SURF-1, FIELD-1, FIELD-3, FIELD-4, PAR-1, FWD-2, DATA-2, DATA-3, LIKE-1, LIKE-2, LIKE-3, LIKE-4, LIKE-6, LIKE-9, LIKE-10, UNC-5, CFG-1, CFG-3

**V2. Known answers of the whole forward model (synthetic library).**
- What: the chain from parameters to spectra is right, end to end.
- Items:
  - V2a: a star with the same B and psi everywhere, divided by the sum of weights, equals that library spectrum after
    convolution;
  - V2b: a centred dipole seen pole-on: the weight that each library node receives equals an independent plain-numpy
    sum over the same patches (1e-10), and the distribution of B over the visible disc matches its closed form
    (quadrature tolerance);
  - V2c: Teff interpolation: at a library Teff both rules return that slice; with `interp_teff = "log"`, halfway in
    log Teff gives the mean of the two slices; with `"planck"`, a synthetic library whose spectra are a fixed shape
    times the Planck function (`tests/synth.py --variant planck`: the I and V of the standard synthetic library at
    30 000 K, times P(T, lambda) / P(30 000 K, lambda)) is reproduced at 5 Teff values between its slices to 1e-6
    relative with a data grid equal to the library grid and no convolution (the per-pixel weight is then exact; 1e-6
    because the records are float32), and at FWHM 7 A to 1e-5 in I and 1e-4 of max |V| in V (the weight varies across
    the kernel, and V is steep in the lines), where `"log"` is off by more than 1e-3 in I;
  - V2d: B outside the library: "clamp" gives the edge spectrum, "reject" marks the model invalid; the reported area
    fraction outside the range matches the closed form for a pole-on centred dipole (quadrature tolerance);
  - V2e: a library written and read back is unchanged; a file of the wrong size (NB x NP x 2 x NL values, or NMU x NB
    x NP x 2 x NL for a mu-grid library) is refused with a message;
  - V2f: each library slice is read from disk once per process (the reader counts reads);
  - V2g: data made by the model with a shift of 0.7 A (constant) or 23 km/s (velocity, between the shift nodes):
    fitting only the shift returns it within 0.01 A or 0.5 km/s on noise-free data; the model with the shift fixed at
    the truth gives a cost below 1e-8; the data arrays the likelihood receives are the input arrays, unchanged (the
    shift acts on the model only, FWD-13). Because data and fit share the model, the sign and unit are anchored
    outside it. The synthetic line at 5500 A is the same for every B. With a fixed shift v, its centroid in the model
    spectrum lies at 5500 (1 + v/c) A for v = +-300 km/s; for v = +-5 A, at 5500 + v A. The tolerance is 0.01 A
    (0.5 % of a 2 A pixel); flipping the sign of the shift fails this;
  - V2h: absolute-flux mode: doubling the distance divides the model flux by 4;
  - V2i: the floor chi2 (DESIGN.md section 5b2) of data made from a non-negative mixture of library records is below
    1e-8 x the chi2 of the zero model; for the same truth with noise added, it is above 0; the same with a free shift
    (5 shift nodes over [-60, 60] km/s) for a true shift of 0 with one Teff slice and of +-30 km/s with three slices
    (the truth on a slice, and between two with the "log" Teff weights), and at a wrong shift (-60 km/s) the floor is
    above 1e-6 x the chi2 of the zero model;
  - V2k: mu-grid library (LIB-8): a synthetic library with 3 mu values (`tests/synth.py --nmu 3`) is read and checked
    for size; at a library node in (B, psi, mu) the patch spectrum is that record; halfway between two mu values it is
    their mean; the disc sum applies no limb-darkening factor; a patch below the smallest mu takes the edge record and
    is counted.
- Covers: LIB-1, LIB-2, LIB-3, LIB-5, LIB-6, LIB-8, FWD-1, FWD-3, FWD-4, FWD-5, FWD-13, DATA-7, DIAG-1

**V3. Symmetries and consistency (metamorphic and property tests).**
These need no known answer. They change the input in a way that must not matter, or must matter in a known way, and
compare. They use Hypothesis (a library that generates many random inputs) with a fixed seed, 200 cases each.
- V3a: mirror through the equator (z components of all field vectors and of the centre negated, cos i negated):
  identical fields at mirrored points (1e-12); identical spectra on a mirror-symmetric patch set (1e-12) and on the
  Fibonacci grid within the quadrature tolerance;
- V3b: reverse the quadrupole axis with its strength unchanged: identical fields (1e-12);
- V3c: reverse the whole field (every component's sign): V and V/I change sign, I is unchanged (1e-12);
- V3d: add 1 to every phase, or add a phase offset and subtract it from the data phases: identical spectra;
- V3e: list the phases in another order: identical likelihood;
- V3f: "dqo_aligned" equals "dqo" with all three axes equal; zero offset and zero higher components equal a centred
  dipole; for a centred dipole, swapping the inclination and the dipole axis's angle from the rotation axis (with the
  matching azimuth) gives identical spectra (1e-10 at beta = 180 - i on a mirror-symmetric patch set; else within the
  quadrature tolerance);
- V3g: plain chi2: multiplying all errors by k divides chi2 by k^2; a weight w gives the same chi2 as errors divided by
  sqrt(w); a pixel with its error multiplied by 1e6 gives the chi2 of the data with that pixel left out, to 1e-9
  relative; full likelihood: a weight w gives the same -ln L as errors divided by sqrt(w), and a weight 0 the same as
  the pixel left out (1e-10);
- V3h: full likelihood: multiplying all errors by k changes -ln L by the same constant for every parameter set;
- V3i: switching I or V/I off for a phase removes exactly that spectrum's terms;
- V3j: phase smearing over a zero-width range equals no smearing;
- V3k: symmetric copies of a geometry (V3a, V3b) get the same fingerprint and land in one mode; with one phase, one
    geometry seen at two inclinations, or turned about the line of sight, is one mode with the default thresholds
    (decision #16); with several phases, a geometry and its field-reversed copy (V3c) are two modes;
- V3l: continuum scale (LIKE-3): multiplying the model I by c (0.01 to 100) leaves the full -ln L unchanged (1e-10),
    with and without the free constant, at degrees 0, 1 and 3;
- V3m: families (SEARCH-9): on a toy surface with two basins joined by a saddle of known height s above the better
    minimum, the two minima are one family when the family level is above s and two when it is below; end points
    placed on the slope of one basin join that basin's family; a pair whose straight segment crosses a ridge but which
    a re-polished midpoint joins below the level is one family; the grouping does not depend on the order of the end
    points; symmetric copies of one end point are one family; two nodes on the edge of an invalid region (or on a
    polish limit beyond which -ln L falls away), whose straight path crosses a ridge and whose path around it stays
    low, are linked through a midpoint that stays valid and within the limits; with 3 workers the
    families, band components, links, pairs tested, cap and mode assignments equal the serial ones, with and without
    the `max_pairs` cap stopping the grouping;
- V3n: the table stage (SEARCH-8, table.py): on a toy surface whose wells change depth between the
    search's and the error-bar patch count (the re-polish made the identity, so every re-polished -ln L is known), two
    modes swap order, a mode invalid on the error-bar patches and a mode that leaves the band are dropped, each mode
    keeps its own re-polished nodes after the drops, `table_repolished` is true, the optimiser scatter is pooled over
    the modes with at least 3 re-polished points only and `value_tolerance_table` is the value tolerance from it
    (1e-9); after a hand-back that moves a mode by more than the value tolerance, the families are recomputed with
    its new point as a node, which becomes the family's best node (the profiles' start), and without one nothing is
    recomputed.
- Pass: 1e-12 for identities, 1e-10 where sums are reordered, unless stated.
- Covers: FIELD-2, PAR-2, FWD-10, FWD-14, DATA-1, DATA-4, LIKE-3, SEARCH-2, SEARCH-8, SEARCH-9

**V4. Error handling and report rules.**
- What: wrong input stops the program with a clear message, and the report never says more than the checks allow.
- Items:
  - V4a: stops before fitting, with a message naming the range: Teff or wavelength range outside the library;
  - V4b: stops with a message naming both ranges: irregular or non-overlapping data grid;
  - V4c: stops with a message naming the key: unknown configuration key; at load, before any search: an unknown profiled quantity or phase index, a profile step that is not a
    positive number or `"<p>%"` with p > 0 (`"two%"`, `"-5%"`, `"0%"`), `profile_max_steps` < 1, `profile_families`
    < 0, `search.family` min_level <= 0 or an integer below 1, `model.shift_nodes` < 2, and an `[[objective.kernel]]`
    label that matches no spectrum (also when the likelihood is built from a configuration that skipped the reader);
  - V4d: an invalid model gives the defined worst value (DESIGN.md section 5), never NaN, and is counted;
  - V4e: the report has no single best fit without the mode table and no evidence weights; once families exist (T7.9),
    not without the family table either;
  - V4f: labels: no tempered widths anywhere in the outputs; per-mode intervals labelled "per mode: conditional on this
    basin"; intervals "uncalibrated" when no coverage run is named, and when the named run has no passed V15b part
    (with the message "computation tested (V15a), not calibrated against model error"); intervals within 1 % of a
    range limit flagged; "limited by model systematics" above 3 sigma; "not constrained by the data" (DESIGN.md
    section 4); shifts listed apart from error bars; a named coverage run whose recorded likelihood settings (the
    kernel included), library checksum or data-set checksums differ from this run's gives "uncalibrated", with a
    message naming the difference; every table header states the continuum setting;
  - V4h: the kernel in the report (T7.6): every table header states the kernel ("fitted, frozen" or "given"); a failed
    whiteness check puts "noise model fails the whiteness check" on every -ln L difference and interval, and makes a
    named coverage run unable to give "calibrated". End to end: correlated noise injected (kind full), fitted with
    `objective.kernel_fit = false` and an identity kernel, once with `--no-mcmc` (systematics and a profile on) and once
    with a capped MCMC and a named coverage run: modes.json, families.json, profiles.json, systematics.json,
    diagnostics.json and intervals.json (those written) carry the flag and the failing spectra, intervals.json says
    "uncalibrated", and each table of report.txt carries the flag line; a run that passes writes no flag;
  - V4i: the side-by-side table (UNC-8, T7.10): on a toy family with a known envelope and per-mode interval, R1 and R2
    equal their values; R1 never flags (decision #30); R2 flags at 1.6 and not at 1.4; for a Gaussian toy basin R2 is
    1 within one grid step; the table prints a quantity's flag "constraint not met" in its row;
  - V4g: the search warning of SEARCH-4 appears when more than half of the modes in the band were reached once.
- Covers: LIB-7, DATA-6, LIKE-2, LIKE-3, LIKE-7, LIKE-8, LIKE-10, SEARCH-4, UNC-4, UNC-6, UNC-8, SYS-3, SYS-5, REP-1, REP-2, REP-3, REP-4, CFG-2

**V5. Fast code against a slow reference.**
- What: the numba code computes the same as a plain numpy version written for clarity.
- Items: V5a forward model, single and batched calls; V5b likelihood. numba's `fastmath` is off. V5c: the MCMC's
  batching (DESIGN.md section 4, step 2): on the toy star with 4 ensembles of 16 walkers, at most 10 % of the
  sampler's calls carry fewer proposals than a quarter of all walkers (proposals that break the prior cut are counted,
  though not evaluated; the snooker move's sub-step is a quarter of an ensemble, and the walker redraws at the start
  and the ends of the convergence-check chunks leave small calls), and the median call carries at least 3/8 of all
  walkers (24 of 64; 28 measured); the chains are the same as separate emcee samplers' for a row-independent cost (the V14 lock-step test).
- Pass: 1e-10 relative, unless stated.
- Covers: FWD-11, UNC-1

**V6. Determinism.**
- What: the same seed gives the same answer whatever the number of worker processes.
- Items: V6a: a search with 4 starts at 1000 patches, run with 1, 2 and 4 workers: identical mode tables (1e-12); a
  different seed gives different starting points. V6b: numba, BLAS and OpenMP threads are 1 in each worker. V6c:
  `fit` twice with the same configuration, seed and 2 workers (search, a capped MCMC and systematics on the best
  mode): the same output files; the JSON files equal at 1e-12 outside the top-level `_meta` (`run.json` excepted);
  `report.txt`, `spectra/`, `histograms/` and `samples/` identical (FORMATS.md section 6). V6d: profiles (DESIGN.md
  section 4b) with 1 and 2 workers: the same result on a toy (two families, one of them with one node) and the same
  `profiles.json` from `fit --no-mcmc` (outside `_meta`), which holds exactly the two quantities of its explicit
  `profile_quantities` list, and the same `families.json` (outside `_meta` and `nevals`).
- Covers: SEARCH-5, ENG-2, ENG-3

**V7. The commands run end to end.**
- What: `eval`, `inject` and `fit` work on a small synthetic problem (1000 patches, at most 4 starts and
  200 optimiser iterations each, so tier 1 stays within its budget).
- Items:
  - V7a: `eval` on a synthetic configuration writes `eval.json` (FORMATS.md section 6);
  - V7b: `inject` makes noise-free data from a known geometry;
  - V7c: `fit` (search only) on a small problem (a centred dipole and the inclination, 4 free parameters, noise-free):
    every start's end point is written, each mode lists its number of starts and its family, every mode within the
    band is listed, the start distribution is in `run.json`, and the best mode is within
    0.1 Gauss-Newton sigma of the truth in every parameter (with the plain chi2, because the full likelihood is
    unbounded at a perfect fit); the result must not depend on the BLAS build: values from different batch layouts are
    compared at 1e-10 relative (section 3), never with ==, and statistical assertions in tier-1 tests use thresholds
    that a correct build fails with a stated small probability; the LM finish: on a toy
    whose Gauss-Newton model points uphill, the exit after the damping passes 1e12 (16 rejections) is not converged
    (`lm_stop` "lambda"); `search.lm_stop_nat` and `search.lm_max_rejections` set to 0 (off) change nothing (bit-equal to the LM before
    the keys existed); a cap of 5 rejections ends that toy's LM after 5
    iterations (not converged), and a stop of 1e-3 nat ends a Rosenbrock LM in fewer iterations (converged), less than
    1e-3 nat above the end point without it; an accepted step that gains less than `search.lm_stop_nat` after
    rejections (a damped step) is no stop: from the start (-0.0803, -0.0712) and from 200 seeded starts every
    converged LM-only Rosenbrock end point lies within `search.lm_stop_nat` of the one with the stops off (before,
    0.148 nat above from that start); `converged` is the LM's verdict only when the LM's total gain over
    L-BFGS-B's end point exceeds `search.lm_stop_nat` (or L-BFGS-B did not run), else L-BFGS-B's: an LM that only
    rejects steps after a converged L-BFGS-B leaves the polish converged, and an LM "predicted" stop without a step
    after an L-BFGS-B that ran out of iterations leaves it not converged; after a converged L-BFGS-B
    an LM that accepts a step of 5e-4 nat and then hits the rejection cap leaves it converged, one that gains 1e-2 nat
    that way not converged, and with `search.lm_stop_nat` = 0 any accepted step makes the LM's verdict count; the LM trace has one entry per iteration, also for an iteration that ends in
    LinAlgError;
  - V7e: `fit` with a toy field model registered from the test file (not in `src/`) runs without any change to the
    search or error code (`tests/test_v7_field_plugin.py`): a dipole plus an l = 4 component on its axis, whose
    parameters carry the configuration's names (md, u), recovers its noise-free truth with `fit --no-mcmc`; a toy
    with a parameter name of its own (a dipole on a fixed axis with strength `bd`, plus u) is an expected failure
    (xfail, strict) until `config.py` accepts the parameter names of the registered model and `problem.py` stops
    taking the dipole from the first three field numbers or from md_x, md_y, md_z (found 2026-10-25; the test names
    the lines); the built-in "harm4" (truth: a dipole plus g20 and h32; fit: md held at its true value, g20, h32 and u
    free) fits its noise-free truth to -ln L < 1e-2 and returns g20 and h32 within 1 MG;
  - V7f: `fit` with MCMC (capped at 2 ensembles of 2000 steps, so the mode may be flagged "not converged") and
    systematics: every output file of FORMATS.md section 6 is written and valid (`families.json` and `profiles.json`
    from tasks T7.9 and T7.10 on: V7j); the default swaps of SYS-1 that the
    configuration allows (V/I x 0.9 and x 1.1, Teff at each slice when Teff is a parameter, one per entry of
    `systematics.library_alternatives`, `continuum_deg0` and `continuum_off`, with the scale ratio of each I spectrum
    next to `continuum_off`) appear in `systematics.json`; one phase: the orientation quantities carry "orientation
    not determined by one phase" and no numbers; a swap that removes spectra (the jackknife's left-out phase,
    `use_VI = false`) drops their inherited `[[objective.kernel]]` blocks and keeps the others, and the jackknife drops
    the profile names and steps of the left-out phase and moves those of later phases down one index (three phases,
    each left out in turn); the swap `teff_<T>` is skipped only when Teff is fixed at the slice T to
    the library's snap rule (`library.on_slice`), not when it is fixed 0.3 K from it;
  - V7g: outputs: a run interrupted while writing (a writer made to fail half-way, in the test) leaves no file under a
    final name that is incomplete (ENG-7); `run.json` contains no absolute path, and its input paths start with
    `$ZEEMAP_DATA/` or are relative to the configuration's directory (ENG-4); a second `fit` into the same directory
    with other settings leaves no file of the first that it does not write itself (planted files of a run with more
    modes, MCMC and systematics included), and keeps files of other names; a failure in the profile stage (made to
    fail in the test) still writes every other output, the MCMC's included, and names the failure in
    `diagnostics.json` and `report.txt`;
  - V7i: the per-window residual table of DIAG-3 (T7.6; kind full, injected noise): one row per family's best mode
    (tagged with the family and mode ids of families.json), library, spectrum and window, with the pixel count and the whitened chi2 under the run's continuum setting
    and at degree 0; for noise-free synthetic data from the run's library its chi2 under the run's setting is 0
    (1e-8 relative to the data's sum of squares);
  - V7j: `families.json` (T7.9) and `profiles.json` (T7.10) are written and valid (FORMATS.md section 6), and every mode
    in `modes.json` names its family or null; with the shipped `profile_quantities = "default"` on the synthetic
    problem, where no limit lies near any profile, no quantity has a closed end "constraint not met", the flags agree
    with `ends` and the stalled points, every point that missed is a stalled one, the table has no "constraint not
    met" row, and B50_MG[0] reaches below the 251.4 MG at which a stalled polish closed it before;
    the profile intervals of incl_deg and dipole_MG contain the truth (noise-free data); both profile checks share one
    fit. On the same problem with bq fixed at 30 MG (a strength fixed non-zero does not scale, so the
    penalty holds B50_MG[0]), a walk of 2 grid steps down under the shipped LM stops retries a missed point at least
    once, and every point met the constraint or stalled (on the default problem every quantity is held
    directly or by scaling, so no real-Problem walk enters the retry);
  - V7h: a default Teff swap whose refit keeps the geometry (noise-free synthetic data, Teff at the neighbouring
    slice) is reported as a shift, not as "mode change" (SYS-4); a swap whose refit moves the geometry beyond the
    fingerprint thresholds is "mode change".
  - V7k: checkpoints and resume (ENG-8; 3 starts, a capped MCMC and systematics on the best mode): a run
    made to crash right after the search's round-0 checkpoint and resumed with `--resume` writes the same
    `modes.json`, `families.json`, `intervals.json` and `systematics.json` as an uninterrupted run (the V6a
    tolerance), profiles included, and writes a null time per evaluation; resumed again with another MCMC budget, the
    table stage, the profile walks and every refit come from their checkpoints (run.json `resumed`) and only the MCMC
    runs; `--from` a search-only run (`--no-mcmc --no-systematics`) with the MCMC and the systematics on draws no
    starts, walks no profile again and gives the same outputs; a resume with a changed data path, or with changed
    bytes of a data file at the same path, stops with one line naming `data` and exit status 2; `run_search` resumed
    from each state it saved (round 0, the kernel stage with the full likelihood, a restart round that met the stop
    rule with `rounds_max` = 3) gives the uninterrupted search's result.
- Covers: FIELD-4, PAR-4, SEARCH-1, SEARCH-3, SEARCH-7, SEARCH-9, SYS-1, SYS-4, DIAG-2, DIAG-3, ENG-4, ENG-7, ENG-8, CLI-2, CLI-3, CLI-4

**V8. Repository hygiene.**
- What: nothing forbidden is committed. (That the environment builds from the lock file is checked by CI itself,
  task T0.5.)
- Items, all run by `tools/scan.py` over the files tracked by git:
  - V8a: no absolute paths: the patterns `/home/`, `/Users/`, `/tmp/` and `C:\`, except the lines listed in
    `tools/scan_allow.txt` (one `path:pattern` per line; it lists `tools/scan.py`, `docs/review/lint_plan.py` and
    `docs/VERIFICATION.md`, which name the patterns);
  - V8b: no files with the extensions `.bin`, `.dat`, `.fits` or `.npz` outside `tests/fixtures/` and
    `tests/regression/`, and no file over 1 MB;
  - V8c: no identifier from `tools/forbidden_identifiers.txt` (one name per line, whole words, case-sensitive; names of
    routines of Numerical Recipes and other fitting programs, kept by the maintainers);
  - V8d: the scan fails on a planted absolute path in a temporary copy of one file;
  - V8e: the check runner: a named item that selects no test fails the run and is named, also next to items that do;
    `tools/check.sh 1 --fast` passes pytest `-m "tier1 and not slow"`, prints `SUMMARY tier 1 (fast): ...`, and
    leaves out exactly the tests marked `slow`.
- Covers: ENG-1, ENG-4, ENG-5

**V9. Retired (2026-11-08).**

### Tier 2: before merging a change that can move results, and before a release

Checks that need real data first verify `$ZEEMAP_DATA/MANIFEST.sha256`; a checksum mismatch fails them.

**V10. Regression against stored references.**
- What: a change did not alter results by accident.
- How: a fixed list of evaluations and short fits (`tests/regression/cases.toml`), compared with stored references.
  References for synthetic data are in the repository (`tests/regression/`); references for real data are kept outside the
  repository, in `$ZEEMAP_DATA/references/`.
- Pass: every case within the section 3 tolerances. If a change is meant to alter results, the references are
  regenerated in the same change, and `CHANGELOG.md` gives the before and after values.
- Covers: every requirement whose output is a number.

**V11. Smoothness and patch-count convergence.**
- What: the smooth model is smooth enough to sample, and the patch counts are high enough.
- Items:
  - V11a: smoothness: at 20 000 patches, around one mode of RE J0317-853, along each of the 13 principal directions of
    the Gauss-Newton covariance (the curvature estimate at the mode), 61 points over +-3 sigma.
    Roughness is the residual of -ln L from a cubic smoothing spline with knots every 0.5 sigma (so that a smooth but
    non-quadratic shape does not count as roughness). Pass: residual rms at most 0.2 nat and maximum at most 0.5 nat along every direction, and no extra local
    minimum deeper than 0.5 nat within +-2 sigma. (The research measured up to 0.40 and 0.86 nat with linear
    interpolation, as residuals from a quartic, which bounds the spline residual from above; this may still fail, and
    then the PCHIP option of FWD-15 becomes the default.)
  - V11b: patches, at fixed parameters: 20 000 against 80 000 patches. Pass: the rms difference of the model spectra
    below 0.05 of the error per pixel, and |delta -ln L| below 0.5.
- The posterior's convergence with the patch count is part of V15 (it needs MCMC, which is too slow for tier 2).
- Covers: SURF-3, FWD-6, FWD-15

**V12. Speed and memory.**
- What: speed and memory stay within the requirements and do not get worse.
- How: time one evaluation (5000 patches, 8 phases, 900 pixels, synthetic library) and a batch of 32; measure the peak
  memory of each worker in a 4-worker search.
- Pass: on the reference machine, below 10 ms per evaluation; on any machine, at most 2 times slower than its stored
  baseline (`tests/regression/speed_baselines.json`, keyed by host name, CPU model and core count; the
  first run on a new machine writes the baseline and passes with a note); peak memory below 1 GB per worker.
- Covers: FWD-12, ENG-6

**V13. Prior recovery.**
- What: the MCMC samples the prior (DESIGN.md section 4) when the data say nothing.
- How: on a synthetic problem, multiply -ln L by 0, run the MCMC, map samples to the canonical form, thin them to their
  effective sample size, and compare them with direct draws from the prior.
- Pass: a two-sample Kolmogorov-Smirnov test per parameter with p > 0.01 / (number of parameters).
- Covers: PAR-4

### Inside every fit

**V14. Run-time checks of convergence and consistency.**
- What: the tool refuses or flags results whose MCMC did not converge, and hands better points back to the search.
- Rules, applied in every fit:
  - V14a: per mode, at least 4 ensembles started independently (own seeds, starting points spread over at least 3
    times the Gauss-Newton widths); the first 5 autocorrelation times discarded as burn-in;
  - V14b: chain length at least 50 times the longest integrated autocorrelation time, and that time changes by less
    than 5 % over the last quarter of the chain (per quantity: the same with the quantity's own autocorrelation time,
    the longest over the ensembles);
  - V14c: rank-normalised split R-hat below 1.01 across the ensembles (one chain per ensemble, split in halves), for
    every parameter and derived quantity (walkers of one ensemble are not independent, so R-hat is never computed
    between them);
  - V14d: bulk and tail effective sample size at least 400 for each quantity;
  - V14e: no drift: the median -ln L of the second half of the kept chain differs from the first half by less than
    1 nat; no stuck walkers: each walker's median -ln L lies within max(3 nats, 5 x the median absolute deviation of the
    walkers' medians) of
    the median of the other walkers' medians (in the research, 3 of 32 walkers stayed behind); the acceptance fraction
    is reported, with a warning outside 0.2-0.5;
  - V14f: the best 3 samples by -ln L are polished; none may end better than the mode by more than the value tolerance of DESIGN.md section 3;
    if one is, it goes back to the search (UNC-3), at most 3 times per mode.
- Verdicts (decisions #17 and #18): the mode passes when every rule passes; when it fails, each quantity that passes
  V14b-d on its own, and the run's V14e, still gets its interval, and only the failing quantities are flagged "not
  converged" (UNC-2). Coordinates declared nuisance (the orientation of a single-phase fit) are left out of the
  burn-in, the longest autocorrelation time and the verdict; their R-hat and ESS are reported as "nuisance, not
  required".
- Tests of these rules (tier 1, on a toy target with a known answer): a deliberately short chain and a deliberately
  drifting chain are flagged "not converged"; a planted better point is handed back once and the mode table updated;
  a toy where one direction is flat and unconverged (R-hat far above 1.01) and another is converged yields an interval
  for the latter only, and none at all when the drift rule fails; the same toy with the flat direction declared
  nuisance passes the mode-level verdict and reports no interval for that direction.
- Covers: UNC-2, UNC-3

### Tier 3: before a release or a paper

**V15. Coverage of the error bars (injection and recovery at fixed truths).**
- What: V15a tests the computation: a 68 % interval contains the truth about 68 % of the time when the data come from
  the model that is fitted. V15b measures what happens when they do not: the data come from another library version.
  Only a passed V15b makes intervals "calibrated" (UNC-4, decision #24).
- How (V15a): at the best mode of the star being calibrated (RE J0317-853 for the first release), 100 data sets made by `inject`, with noise drawn from the full likelihood's
  noise model including the discrepancy term, with the frozen kernel of the star's fit; the injections are fitted
  with that kernel too (no kernel fit per injection). Each is fitted by a local search from the truth plus one MCMC ensemble
  run to at least 50 autocorrelation times (one ensemble, not four, to keep the cost to days; it must pass the V14
  rules except R-hat). For 14 gated quantities, computed in the canonical form (DESIGN.md section 2): |m_d|, the
  colatitude and azimuth of the dipole axis, b_q, the quadrupole axis, |m_o|, the colatitude and azimuth of the octupole
  axis, c_x, c_y, c_z, |c|, the inclination, and B50 at phase 0.25, count how often the truth is inside the 68 % and the
  95 % interval. Azimuths are compared on the circle. For the quadrupole axis, the truth counts as inside the p interval
  when its angle to the samples' principal axis (the leading eigenvector of the mean of a_q a_q^T, folded to 0-90
  degrees) is below the p quantile of the samples' angles to that axis.
- Pass, for each of the 14 quantities: 53-82 of 100 inside the 68 % interval, and at least 87 of 100 inside the 95 %
  interval. These bands are binomial and corrected for the 28 tests (14 quantities at two levels), so a correct method
  passes all of them with probability about 0.98. Below the band: intervals too narrow (the serious failure). Above:
  too wide. Also: the mean coverage over the 14 quantities within 3 bootstrap standard errors of 0.68 and of 0.95. With
  100 data sets, intervals about 30 % too narrow are caught reliably; smaller errors may pass.
- Reported, not gated: the coverage of the other derived quantities; a second geometry (the best mode of the other
  inclination branch); a set with realistic noise (white noise resampled like the real data); and the posterior medians
  of 10 of the fits rerun at 80 000 patches, which must shift by less than 0.2 posterior standard deviations.
- Injections whose fit lands in another mode, or fails V14, are reported, and counted as "not covered" in a second,
  conservative figure. They are never dropped.
- V15b: the same, with 100 data sets made with the library in `systematics.library_alternatives` (the other physics
  version) at the same truth, fitted with the run's library; the truth is the alternative library's best fit to the
  real data (the injected geometry is that one, not V15a's), so it is one that library can produce. Each injection is
  fitted with the kernel frozen at the real-data fit's (as in V15a). Reported, not gated: its coverage per quantity
  against the bands of V15a. A result inside the bands is what UNC-4 needs for "calibrated"; outside, the intervals
  stay "uncalibrated" and the report names V15b's numbers. The research measured 0.27-0.35 coverage in this situation,
  so V15b is expected to fall outside the bands until the noise model or the libraries improve.
- Note: the research likelihood gave 0.90 at 68 %, which is too wide; the injected noise there had no discrepancy part.
- Covers: LIKE-1, LIKE-2, LIKE-3, LIKE-4, UNC-1, CLI-4

**V16. Simulation-based calibration (SBC).**
- What: a test of the computation (decision #24): the sampler, the prior and the forward model together give the
  right posterior when averaged over the prior, not only the right width at one point. It says nothing about
  misspecification.
- How: a reduced problem with one mode: dipole vector and inclination (cos i >= 0), with the field centre fixed off the
  axis at c = (0.10, 0, 0.05) stellar radii, 3 phases, 1000 patches, synthetic library. The offset breaks the symmetry
  between inclination and axis angle of a centred dipole (V3f), which would otherwise give two exact modes. The noise scale and the continuum are fixed at their true values in both the simulation and the fit
  (configuration switches of LIKE-9), because the closed-form integrals of the full likelihood use priors that cannot be
  drawn from; V1g tests those integrals. Repeat 200 times: draw a truth from the prior, simulate data, sample, keep 99
  draws thinned to be nearly independent, and record the rank of the truth among them. Do this for each parameter and
  for -ln L.
- Pass: for every quantity, the empirical distribution of ranks stays inside its 99 % simultaneous band, and a
  chi-squared test of the 10-bin rank histogram gives p > 0.01 / (number of quantities). A U-shaped histogram means
  posteriors too narrow; a hump means too wide; a slope means bias.
- Covers: LIKE-9, UNC-1

**V17. Search completeness.**
- What: at the default budget the search finds the best mode, and its own statistics say so honestly.
- Items:
  - V17a: planted problem (built in task T3.5): synthetic data from a geometry whose best mode is reached by 0.5-2 % of
    random starts (measured with 500 starts; the real RE J0317-853 rate was about 1 %), with a second mode at least 20
    nats worse. Run `fit` with 30 seeds. Success = the mode table contains the planted best mode (same fingerprint)
    within the value tolerance of DESIGN.md section 3 of its reference value (the best of 8 polishes started from the
    truth at 20 000 patches). Pass: at least 28 of 30 runs succeed. The default budget is set so that a run succeeds
    with probability at least 0.99; then this rule passes with probability 0.997, while a search that succeeds only
    80 % of the time passes with probability 0.04.
  - V17b: the per-start hit rate measured over all 30 runs predicts the run success rate: the observed number of
    successes lies in its 95 % predictive interval.
  - V17c: RE J0317-853: the run reaches within 5 nats of the best mode found in the research, after polishing at 20 000
    patches with the same likelihood settings, in either inclination branch.
  - V17d: budget doubling: twice the starts with a new seed finds no mode better than the best by more than the value
    tolerance, and the same modes within the band.
- Covers: SEARCH-1, SEARCH-4, SEARCH-6, SEARCH-8

**V18. Systematics and reruns.**
- What: systematic shifts are measured right and reproducibly, and recorded fits can be rerun.
- Items:
  - V18a: synthetic data made with the synthetic library and fitted with variant b (section 4): over 20 injections, the
    mean measured shift equals the shift found by fitting noise-free data, within 3 standard errors; a swap that moves
    the fit to another mode is reported as "mode change";
  - V18b: the swaps of SYS-1 on one RE J0317-853 mode, twice with different seeds: the shifts of the derived quantities
    agree within 1 statistical sigma, or within the spread of the 4 swap starts if that is larger; raw angles are
    reported, not gated; the jackknife sigma is written;
  - V18c: a recorded fit, rerun from its run directory (the configuration, inputs and seed in its `run.json`), gives
    the same modes within the full-rerun tolerance of section 3 (task T6.4).
- Covers: UNC-7, SYS-1, SYS-2, SYS-4

**V19. Planted faults (do the checks catch real mistakes?).**
- What: the checks above are sharp enough to notice bugs.
- How: deliberate faults, kept as small patch files in `tests/faults/` and applied one at a time to a temporary copy:
  the sign of V; the limb-darkening law; an off-by-one in resampling; degrees used as radians; the area weight; a seed
  reused across workers; the quadrupole axis reversed together with its strength. Optional: mutation testing of the
  kernel modules with `mutmut`, reporting the share of killed mutants.
- Pass: every planted fault makes at least one tier-1 or tier-2 check fail. A fault that passes means a check is
  missing: add one.
- Covers: the checks themselves.

**Cost of tier 3.** V15a needs 100 fits with MCMC (about 5e5 evaluations each at 10-26 ms): 2-4 days on 12 cores;
V15b as much again. V17a
needs 30 full searches at the default budget (expected about 3e6 evaluations each, so about 1e8): 1-3 days on the
synthetic library, which is faster than the real one. V16 runs
on a small problem in hours, V19 in about 3 hours.

## 6. The everyday development loop

For every task, by a person or an agent:
1. Take the issue and the requirements it names in SPEC.md section 6.
2. Write the test for each check item the task lists, and see it fail.
3. Write the code until the tests pass.
4. Run `tools/check.sh 1 --fast` while working and `tools/check.sh 1` before opening a pull request. For a change that
   can move numerical results, a maintainer also runs `tools/check.sh 2` and the extra checks for the kind of change
   (GATES.md).
5. Update the requirement's "Checked by" line and the check's "Covers" line if they changed, and `CHANGELOG.md`.
6. Open a pull request with the SUMMARY lines (GATES.md).

## 7. When a check fails

- Do not loosen a tolerance or delete a check to make it pass. A tolerance that is wrong is changed in its own commit,
  with the reason, after agreement in an issue (GATES.md).
- Treat a failure first as a possible bug in the code or a problem in the model, not as a reason to run longer.
- Find out whether the code or the reference is wrong. If the reference is wrong, regenerate it in the same commit as
  the fix, with before and after values in `CHANGELOG.md`.
- Every fixed bug gets a check that would have caught it.
