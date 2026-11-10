# Verification practices and merge gates: reviewer rubric for the fitter development plan

Written 2026-10-17 (research agent). Scope: a new Python + numba fitter for MWD spectropolarimetry (a field-geometry model with about 13 parameters, fitted against a precomputed library of local Stokes spectra), It has a multimodal objective, multi-start search with perturbation, per-mode emcee, SBC and injection-recovery calibration, and systematics from refits under swaps.

How to use: apply each check to the development plan as a yes/no question. A "no" on a [TOP] check should block the plan. A "no" on any other check needs a written reason.

Source rule. Every source in section 7 was opened in this session, unless it is marked [unverified]. A tag of the form [derived] marks a statement that is this agent's own reasoning or a domain example, not a claim from a source.

---

## 1. Test oracles and known answers

**A1 [TOP] Known-answer (analytic) oracle for every computational component.**
- *Practice:* test each kernel (library interpolation, geometry and field map, disk integration, Stokes summation, chi2) against cases whose answer is known in closed form.
- *Why:* analytic solutions are preferred over pseudo-oracles, "since they can identify common algorithmic errors among the implementations". Comparing a theoretical convergence rate with the rate the code achieves is a standard fault check (S3, Results RQ3, "Techniques used to overcome oracle problems", item 2). Wilson et al. 2014 also recommend assertions, an off-the-shelf unit-test library, and turning bugs into test cases (S1, Box 1, 5a-5c).
- *Examples for this tool* [derived]:
  - interpolation at a library node returns that node exactly;
  - a centred dipole gives |B|(pole) = Bp and |B|(equator) = Bp/2;
  - a constant-intensity disk integrates to the analytic flux, and a limb-darkening law integrates to its known closed form;
  - chi2(model, model) = 0;
  - chi2 with a known residual vector r equals sum (r/sigma)^2;
  - integration error falls at the expected order as the surface grid is refined.
- *Reviewer check:* does every forward-model and objective component have at least one known-answer test with a stated tolerance? Is there at least one convergence-order test for the surface or disk quadrature?

**A3 [TOP] Metamorphic relations, listed explicitly and tested.**
- *Practice:* when no exact expected output exists, check relations between outputs under known input transformations.
- *Why:* metamorphic testing addresses the oracle problem (S4, abstract). In scientific code it uncovered faults that special-value tests missed. Its main weakness is that the relations are listed by hand and can miss important ones (S3, RQ3 item 9).
- *Candidate relations for this tool* [derived; physics to be confirmed by the user]:
  - chi2 is invariant under reordering phases or wavelengths;
  - scaling all sigma by k scales chi2 by 1/k^2;
  - adding a data point with sigma -> infinity leaves chi2 unchanged;
  - a rotation phase shift of 2*pi gives an identical model;
  - reversing the field direction (B -> -B) leaves I unchanged and flips the sign of V (and of any circular-polarization quantity);
  - for a centred dipole the light curve is symmetric under swapping inclination and obliquity (cos theta = cos i cos beta + sin i sin beta cos phi);
  - zero offsets and zero higher multipoles reduce the general model to the centred-dipole code path.
- *Reviewer check:* does the plan list its metamorphic relations, say which component each one covers, and run them per commit?

**A4 Property-based tests for kernels over the whole valid input range.**
- *Practice:* generate inputs across declared ranges, including edge cases, rather than hand-picked values. Shrink failures, replay them, and pin regression examples.
- *Why:* Hypothesis tests "should pass for all inputs in whatever range you describe". It saves failures to a database and replays them; `@seed` and `derandomize` make the generated cases reproducible; `@example` adds explicit inputs that are always tried (S6, index and API reference).
- *Reviewer check:* are the invariants (A3) and input-domain assertions written as property-based tests? Is CI derandomized (or seeded), so that a failure in CI reproduces locally?

**A5 Independent slow reference implementation of the forward model.**
- *Practice:* keep a plain numpy (non-numba) implementation of the forward model and chi2 for small grids, and compare the numba kernels against it.
- *Why:* this is a pseudo-oracle internal to the project (S3, RQ3 item 1). Numba "may use a different algorithm than Python or Numpy. The results may not be bit-by-bit compatible", and small accumulated differences can grow (S20, fpsemantics).
- *Reviewer check:* is there a reference implementation, with a tolerance tier for numba vs reference (same algorithm, float64)?

**A6 Regression tests at tolerance, not bitwise, and with a justified tolerance.**
- *Practice:* compare current output with recorded output within `atol + rtol*|desired|`. Use a mixed absolute/relative tolerance near zero, and choose the tolerance per quantity.
- *Why:*
  - Regression testing against earlier outputs, with tolerances per variable, is common in scientific software (S3, RQ3 "Regression testing").
  - Choosing tolerances is hard, and Kelly et al. found that tighter oracle tolerances find more faults (S3, RQ3, final paragraph).
  - `assert_allclose` uses `|actual - desired| <= atol + rtol*|desired|`, with defaults rtol=1e-7, atol=0 (S21).
  - Near zero, a relative tolerance fails, and an absolute epsilon scaled to the magnitude of the inputs is needed (S22, "Infernal zero").
- *Reviewer check:* does each regression check state rtol and atol and the reason for them (same code, different compiler or threads, float32, physics bound)? Do quantities that can be near zero (V/I, residuals) have an atol?

**A7 Gradient and Jacobian check, if any local optimiser uses derivatives** [derived].
- *Practice:* compare analytic or automatic derivatives with central finite differences at random points.
- *Reviewer check:* if a gradient-based local optimiser is used, is there a finite-difference check with a stated tolerance?

**A8 Test-suite quality check: would the tests fail if the code were wrong?**
- *Practice:* run mutation testing on the core numerical modules from time to time, and review the surviving mutants.
- *Why:* mutation testing inserts small faults to measure whether the tests detect them. At Google, surfacing surviving mutants during code review led developers to add tests (S7). Google's review guide asks: "Will the tests actually fail when the code is broken?" (S26, "What to look for", Tests). For Python, mutmut is incremental and remembers earlier work (S31).
- *Reviewer check:* is there an occasional mutation run (for example mutmut on the interp/geometry/chi2 modules) with a target, or at least a review of surviving mutants?

**A9 Small end-to-end example data set shipped with the code.**
- *Practice:* include a small test set that checks the software "is actually working", which also serves as a usage example (S24, Rule 9; S2, Software practice 9, "Provide a simple example or test data set").
- *Reviewer check:* does the plan ship a minute-scale end-to-end example (a synthetic star plus one real phase), with the expected result recorded?

---

## 2. Statistical calibration

**B1 [TOP] Simulation-based calibration (SBC) with a quantitative pass threshold and data-dependent test quantities.**
- *Practice:*
  - draw theta from the prior, simulate data, and run the full inference pipeline;
  - compute the rank of each true theta among L posterior draws;
  - test whether the ranks are uniform.
- *Why:*
  - The ranks are uniform when the computation is correct (S8, Section 4.1, Theorem 1).
  - The histogram gets a 99% band from Binomial(N, 1/(L+1)); about N/B ≈ 20 replications per bin is a good trade-off (S8, Section 4.1).
  - A U shape means underdispersed posteriors, a central hump means overdispersed, and an asymmetric histogram means bias (S8, Section 4.2, Figures 5-6).
  - Autocorrelated MCMC draws must be thinned to about the effective sample size first, or the histogram shows artefacts (S8, Section 5.1).
  - SBC on parameters alone "could never detect" large classes of problems, including posterior = prior. Adding data-dependent test quantities such as the likelihood fixes this (S9, abstract).
  - SBC is the full version of the fake-data check; it is costly, and a single truth point "does not guarantee anything" (S11, Section 4.2; S10 via S8, Section 3).
- *Caveat for this tool* [derived]: SBC tests the whole pipeline, including the multi-start search, mode selection and mode weighting, not only emcee. Per-mode MCMC that drops minor modes will show up as miscalibration, which is a feature of the test, not a flaw.
- *Reviewer check:* does each posterior claim the tool makes (intervals, mode probabilities) have an SBC test that states all of the following?
  - N, L and the binning;
  - the thinning rule;
  - the test quantities (parameters plus log-likelihood);
  - the pass criterion (counts within the 99% binomial band, or an ECDF test at a stated level);
  - what is done on failure.

**B2 [TOP] Injection-recovery coverage at astrophysically relevant parameters.**
- *Practice:*
  - inject synthetic signals into realistic noise, at parameters drawn from the prior and also on a designed grid of hard cases;
  - record the credible level at which each truth falls;
  - show a P-P plot with binomial bands and per-parameter p-values.
- *Why:*
  - This is the standard validation of gravitational-wave inference codes (S17, Section 3.2): 100 injections from the prior, a P-P plot with 1-3 sigma bands, and a combined p-value of the per-parameter p-values.
  - A single fake-data run will "often reveal blatant errors"; fake-data checks should probe where the data are or are not informative, and one should "break our method" deliberately (S11, Section 4.1).
  - Coverage at a fixed truth is not guaranteed even for a correct method. Bayesian calibration holds averaged over the prior, and a truth at the prior centre can show overcoverage (S11, Section 4.2).
- *Reviewer check:* does the plan separate prior-averaged coverage (an SBC-type test with a pass/fail threshold) from fixed-truth injection studies (diagnostic, reported per region of parameter space)? Does it name the realistic noise model (real error bars, phase sampling of the target stars)?

**B3 [TOP] Runtime MCMC diagnostics as a hard gate on every reported fit.**
- *Practice:* the fit refuses to report uncertainties, or flags them, unless all of the following hold:
  - the chain is long relative to tau;
  - the tau estimate is stable;
  - the acceptance fraction is in range;
  - independent runs agree.
- *Why:*
  - emcee recommends chains longer than about 50 tau, which is "often sufficient". Its monitoring example stops when N > 100 tau and tau has changed by less than 1% (S13, autocorrelation and monitor tutorials). Tau differs for each function f (S13).
  - An acceptance fraction of 0.2-0.5 is a quick indicator. A low fraction usually signals multimodality; the advice is then to split the space into single-mode regions and sample each independently (S14, Section 4). This is the per-mode design chosen here.
  - Rank-normalised R-hat < 1.01, at least four chains, and rank-normalised bulk and tail ESS > 400 (S12, Section 1). With well-separated modes, an ESS from a single chain "can be drastically overestimated" (S12).
- *Caveat* [derived]: emcee walkers in one ensemble are not independent chains. Compute R-hat across ≥4 independently initialised ensembles per mode, not across walkers.
- *Reviewer check:* are the convergence thresholds (N/tau, tau stability, acceptance range, R-hat, bulk/tail ESS) fixed in the plan, enforced in code, and written to the run record?

**B4 Prior-recovery check of the sampler wrapper.**
- *Practice:* scale the log-likelihood by a very small factor (or set it to zero) and check that the sampler returns the prior.
- *Why:* "a great functional test of your sampler and your data analysis setup", which needs a proper prior (S15, Section 4).
- *Reviewer check:* is there a per-merge prior-recovery test with a quantitative pass criterion (for example a KS or SBC-style comparison of marginals with the prior)?

**B5 Multimodal uncertainty: mode weights and combination stated and tested.**
- *Practice:* say how per-mode posteriors are combined, or reported separately, and test that combination.
- *Why:* mode-based approximations "understate posterior uncertainty", and the relative weights of modes are hard to estimate. Stacking parallel runs is one approach, but "not necessarily equivalent ... to fully Bayesian inference" (S16, abstract). Gelman et al. distinguish negligible-mass minor modes, symmetric modes (restrict the model) and distinct high-mass modes (S11, Section 5.6).
- *Reviewer check:* does the plan say how mode weights are computed (or that modes are reported without weights), and does B1 include a case with two comparable modes?

**B6 Statistical tests in CI have a known false-alarm rate.**
- *Practice:* tests with random outcomes use fixed seeds in the per-commit tier, and stated significance levels in the slow tier.
- *Why:* a statistical oracle "cannot decide whether a single test case has passed or failed" (S3, RQ3 item 7). Under uniformity, about one bin in a hundred falls outside the 99% band (S8, Section 4.1). With many parameters some failures are expected [derived].
- *Reviewer check:* for each statistical gate, does the plan state the expected false-alarm rate and the rule for a marginal failure (rerun with a new seed once? multiple-testing correction across 13 parameters)?

**B7 Computational trouble is treated as a possible model or code problem first.**
- *Why:* the "folk theorem": poor convergence often means a model problem or "bugs in code", and the first instinct "should not be to throw more computational resources" at it (S11, Section 5.1).
- *Reviewer check:* does the plan route failed diagnostics to a diagnosis step (model, parameterisation, code), rather than only to longer chains?

---

## 3. Search completeness (global mode)

**C1 [TOP] Multi-start statistics with a stopping rule.**
- *Practice:* record which basin each start converges to. Report the number of distinct minima found and how many times the best one was hit, and stop on a quantitative rule.
- *Why:* observed minima can be treated as a multinomial sample over the basins of attraction, which gives a posterior on the number of local optima and on the relative size of their basins. That posterior defines Bayesian stopping rules (S18, abstract). With multimodality "knowing that you have a complete and representative sampling is effectively impossible" (S15, Section 5), so the stopping rule has to be explicit.
- *Reviewer check:* does each fit report the number of starts, the number of distinct basins (with the clustering rule), the hit count of the best basin, and the stopping rule (for example the best basin hit ≥ k times and no new basin in the last M starts)?

**C2 [TOP] Known-truth recovery of the global minimum, including deliberately hard cases.**
- *Practice:* on synthetic data, noiseless and noisy, from known geometries, check that the search finds a chi2 at or below the truth's chi2.
- *Why:* fake data are "the only point where we can directly check that our inference on latent variables is reliable" (S11, Section 4.1).
- *Hard cases for this tool* [derived]: near-degenerate geometries (the i/beta swap), large offsets, a weak octupole, poor phase coverage.
- *Reviewer check:* is there a test set of hard synthetic cases with a pass criterion such as "chi2_best ≤ chi2_truth + delta in ≥ X% of cases", run per merge (small) and occasionally (full)?

**C3 Budget-doubling test.**
- *Practice* [derived from S18 and S14]: rerun selected fits with twice the starts and a different seed. Pass if no better basin appears (beyond a stated delta chi2) and the set of reported modes is unchanged.
- *Reviewer check:* does the plan include a budget-doubling check on representative stars before results are published?

**C4 Perturbation stage tested separately.**
- *Practice* [derived]: test that perturbation around the best clusters moves between nearby basins. Plant a better basin near a found one on synthetic data and check that it is found.
- *Reviewer check:* is the perturbation step's benefit measured (the hit rate with and without it)?

**C5 Search and MCMC agree.**
- *Practice* [derived; motivated by S14, Section 4: within-mode tau can be short while mode-to-mode migration is long]: check that no MCMC chain finds a log-probability above the best optimum. If one does, the search missed something.
- *Reviewer check:* is "MCMC max log-prob ≤ optimiser best + tolerance" an automatic per-fit assertion?

---

## 4. Reproducibility

**D1 [TOP] Seeds: a single recorded root seed, independent child streams, and rerun at tolerance.**
- *Practice:* derive all random streams (starts, perturbations, emcee ensembles, SBC draws) from one recorded root via `SeedSequence.spawn`.
- *Why:* NumPy documents spawning as the way to get independent streams and calls `root_seed + worker_id` "UNSAFE", because overlapping seeds across runs bias ensembles of results (S19). Programs should produce identical results for identical inputs and echo all parameters and software versions (S24, Rule 10).
- *Numba specifics:*
  - numba's RNG state is independent of NumPy's, so seeding from interpreted code "will seed the NumPy random generator, not the Numba random generator";
  - numba's Generator support "is not thread-safe" in parallel code (S20, numpysupported).
- *Reviewer check:* does the plan specify how seeds are spawned and recorded, how numba-side randomness is seeded (or avoided, by drawing in NumPy and passing arrays in), and the rerun tolerance for the same seed?

**D2 Floating-point determinism under parallelism and fastmath stated.**
- *Why:*
  - Results with `fastmath=True` relax IEEE rigour, and SVML intrinsics go from 1 ULP to 4 ULP (S20, performance tips).
  - Numba results may differ from NumPy at the bit level (S20, fpsemantics).
  - Parallel reductions sum in thread-dependent order, so results can change with the thread count [derived; S20 lists the supported prange reductions but does not state this outright].
- *Reviewer check:* does the plan fix fastmath on or off per kernel, and give rerun tolerances for (a) same seed and threads, (b) a different thread count, (c) a different machine or compiler, mapped to the host tiers 1e-6, 1e-5 and 1e-3?

**D3 Locked environment, with CI installing from the lock.**
- *Why:* pin versions (`==`) and optionally hashes for repeatable installs (S23, pip "Repeatable installs"). uv keeps a lockfile and syncs the environment from it (S23, uv sync). S2 Software practice 7: "Make dependencies and requirements explicit".
- *Reviewer check:* is there a lock file (uv.lock, or a pinned and hashed requirements file), does CI install from it, and does every run record the numba, llvmlite, numpy and Python versions and the CPU?

**D4 Every published number traces to a recorded run that can be rerun.**
- *Why:* release the software at the same time as the paper so that the results can be reproduced (S24, Rule 4).
- *Reviewer check:* does the plan say that every fit in a paper comes from a recorded run (commit, lock, seed, inputs, library hash) that `replay` reproduces at a stated tolerance?

**D5 Library and input data identified by content hash** [derived].
- *Reviewer check:* is the Stokes library (and any variant used in systematics swaps) identified by a hash in each run record, and does the fitter refuse a library whose hash does not match?

---

## 5. Gates and CI

**E1 [TOP] Tiered gates mapped to cost, with main always green.**
- *Practice:* run a fast commit tier on every commit, a slower tier before merging, and expensive statistical tiers occasionally.
- *Why:*
  - CI needs a self-testing build, a fast build (the XP guideline is ten minutes) and a staged deployment pipeline whose fast "commit build" runs first (S28).
  - When the mainline breaks, "nobody has a higher priority task than fixing the build", and reverting is usually best (S28).
  - Google's small/medium/large tests have time limits of 60, 300 and 900+ s (S29).
  - Test pyramid: write lots of small, fast unit tests and very few high-level ones (S27).
- *Reviewer check:* does the plan assign every loop in section 6 to a tier with a time budget? Does it state that main is reverted, not patched forward, when the per-merge gate fails after a merge?

**E2 Required checks enforced by the platform, not by memory.**
- *Why:* protected branches can require status checks and approving reviews before a merge. In "strict" mode the branch must be up to date with the base branch (S25).
  - Caveat: per S25, protected branches on private repositories need GitHub Pro, Team or Enterprise. For a private repo on a free plan the gate script has to be enforced locally (a pre-push hook) [derived].
- *Reviewer check:* does the plan say how gates are enforced for the (private) repo: branch protection with required checks if the plan allows it, otherwise a hook plus the documented merge routine?

**E3 Definition of done for a change.**
- *Why:* the Definition of Done is "a formal description of the state of the Increment when it meets the quality measures required" (S30).
- *Reviewer check:* does the plan define done for each change type? For example:
  - kernel change: tests plus a changelog entry with before/after numbers;
  - statistics change: SBC rerun;
  - search change: C2 and C3 rerun.

**E4 Code review focused on design, tests and correctness.**
- *Why:* approve when a change "definitely improves the overall code health", not when it is perfect (S26, "The standard of code review"). Check design first, then complexity and over-engineering, and confirm the tests are valid (S26, "What to look for"). Wilson et al. recommend pre-merge code reviews (S1, Box 1, 8a).
- *Reviewer check:* is there a short review checklist (design, a test for every bug fixed, tolerances justified, provenance comment for any physics change)?

**E5 Bugs become tests.**
- *Why:* "Turn bugs into test cases" (S1, Box 1, 5c).
- *Reviewer check:* does each fixed bug get a test that fails before the fix?

**E6 Optimise only after correctness is gated.**
- *Why:* "Optimize software only after it works correctly" and use a profiler (S1, Box 1, 6, 6a). The numba speed-up path should be checked against A5 at every step [derived].
- *Reviewer check:* does each performance change (numba, parallelisation, fastmath) have to pass A5 and D2 before merging?

**E7 Build bottom-up from tested modules.**
- *Why:* development should go "bottom up from well-tested foundational functions" (S11, Section 9.2).
- *Reviewer check:* do the plan's milestones put the kernel tests (A1, A3, A5) before the search and the statistics?

---

## 6. Recommended verification loops for this fitter

| # | Loop | What it checks | Pass criterion | Tier |
|---|---|---|---|---|
| V1 | Kernel known answers | Interpolation at nodes, dipole field values, analytic disk integrals, quadrature convergence order, chi2 identities (A1) | Exact at nodes (≤1e-12 rel); analytic cases within the quadrature-error bound; observed order within ±0.2 of the formal order | per-commit |
| V2 | Metamorphic and property tests | The A3 relations via Hypothesis, derandomized (A4) | All relations hold at 1e-10 rel (float64) | per-commit |
| V3 | numba vs numpy reference | Forward model and chi2 on small grids (A5) | ≤1e-10 rel, fastmath off; a stated wider bound for fastmath kernels | per-commit |
| V5 | Recorded-run regression | Replays recorded fits and compares them at tolerance (A6, D4) | 1e-6 same code and machine; 1e-3 on a different machine or thread count for chi2 and parameters | per-merge |
| V6 | Determinism | Same seed and threads gives the same output; different thread count; a different seed gives statistically consistent quantiles (D1, D2) | Same-seed byte-identical or ≤1e-12; thread change ≤1e-10 on chi2; seed change: quantiles agree within 3 MCSE | per-merge |
| V7 | Prior recovery | Log-likelihood × 1e-6 makes the emcee wrapper return the prior (B4) | Marginal KS p > 0.01 for all parameters (seeded) | per-merge |
| V8 | Synthetic global-mode recovery (small) | About 20 hard synthetic stars, noiseless and noisy (C2, C4) | chi2_best ≤ chi2_truth + 1e-6 (noiseless) in 100%; noisy: chi2_best ≤ chi2_truth in ≥95% | per-merge |
| V9 | Per-fit runtime assertions | Convergence and consistency inside every production fit (B3, C1, C5) | N > 50 tau and tau change < 1%; acceptance 0.2-0.5; R-hat < 1.01 across ≥4 independent ensembles per mode; bulk/tail ESS > 400; MCMC max logp ≤ best optimum + tol; best basin hit ≥ k times. A failure flags the result | every fit |
| V10 | Mini-SBC | SBC on a reduced model (centred dipole, 3-5 parameters), N ≈ 100-200 (B1) | Rank counts within the 99% binomial band (≤ expected exceedances), parameters plus log-likelihood | per-merge (if ≤ ~15 min), else nightly or occasional |
| V11 | Full SBC | The whole pipeline (search, per-mode emcee, mode weighting) on the 13-parameter model, N ≈ 500-1000, L ≈ 100 after thinning to ESS (B1, B5) | Uniformity at the 99% band or an ECDF test; no U, hump or skew pattern; includes a two-mode case | occasional (before a paper or release) |
| V12 | Injection-recovery coverage | Realistic noise and phase sampling of the target stars; a designed grid of geometries (B2) | P-P curves within the binomial band; coverage reported per region; failures documented | occasional |
| V13 | Search budget doubling | Twice the starts with a new seed on representative real stars (C3) | No better basin beyond the stated delta chi2; same reported mode set | occasional (before a paper) |
| V14 | Systematics machinery | Generate with library variant A and fit with B, on synthetic data where the shift is known; leave-one-phase-out on a synthetic case | The swap pipeline runs and reproduces the known shift within MC error | occasional |
| V15 | Mutation testing | mutmut on the interp, geometry, chi2 and seeding modules (A8) | Surviving mutants reviewed; a kill-rate target agreed (for example ≥80% on kernels) | occasional |
| V16 | Environment lock | CI installs from the lock; the run record contains the versions (D3) | Install from the lock succeeds; the version record is present | per-commit |

---

## 7. Sources (opened in this session unless marked)

- **S1.** Wilson G. et al. 2014, "Best Practices for Scientific Computing", PLoS Biol 12(1): e1001745, doi:10.1371/journal.pbio.1001745. Box 1 (practices 5a-5c, 6, 8a). https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745
- **S2.** Wilson G. et al. 2017, "Good enough practices in scientific computing", PLoS Comput Biol, doi:10.1371/journal.pcbi.1005510. Software practices 5, 7, 9. Unit testing, coverage and CI are deliberately left out for newcomers ("What we left out").
- **S3.** Kanewala U., Bieman J. M. 2014, "Testing scientific software: A systematic literature review", Inf. Softw. Technol. 56(10), doi:10.1016/j.infsof.2014.05.006. Opened as arXiv:1804.01954, Results RQ3: testing levels; "Techniques used to overcome oracle problems", items 1-9; tolerance paragraph.
- **S4.** Segura S., Fraser G., Sanchez A. B., Ruiz-Cortés A. 2016, "A Survey on Metamorphic Testing", IEEE TSE 42(9): 805-824, doi:10.1109/TSE.2016.2532875 (abstract opened; White Rose eprint 110335).
- **S5.** Salari K., Knupp P. 2000, "Code verification by the method of manufactured solutions", SAND2000-1444 [unverified: the page failed to load]. Roy C. J. 2005, J. Comput. Phys., doi:10.1016/j.jcp.2004.10.036 [unverified: 403]. The convergence-order idea is cited here through S3, which was opened.
- **S6.** Hypothesis documentation: https://hypothesis.readthedocs.io/en/latest/ and the API reference (database, `@seed`, `derandomize`, `@example`).
- **S7.** Petrović G., Ivanković M. 2018, "State of Mutation Testing at Google", ICSE-SEIP, doi:10.1145/3183519.3183521 (PDF opened; read through a summary, so its finer claims are not quoted).
- **S8.** Talts S., Betancourt M., Simpson D., Vehtari A., Gelman A. 2018, "Validating Bayesian Inference Algorithms with Simulation-Based Calibration", arXiv:1804.06788. Sections 3, 4.1, 4.2 and 5.1 opened.
- **S9.** Modrák M. et al., "Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity", Bayesian Analysis, doi:10.1214/23-BA1404 (advance publication 2023), arXiv:2211.02383 (abstract opened). The issue details (20(2): 461-488, 2025) come from a search snippet [unverified].
- **S10.** Cook S. R., Gelman A., Rubin D. B. 2006, JCGS 15: 675-692, doi:10.1198/106186006X136976 [the paper itself was not opened]. Its method and the 2017 correction are described in S8, Section 3, which was opened.
- **S11.** Gelman A. et al. 2020, "Bayesian Workflow", arXiv:2011.01808. Sections 4.1, 4.2, 5.1, 5.6 and 9.2 opened.
- **S12.** Vehtari A., Gelman A., Simpson D., Carpenter B., Bürkner P.-C. 2021, "Rank-normalization, folding, and localization: an improved R-hat", Bayesian Analysis, doi:10.1214/20-BA1221, arXiv:1903.08008. Section 1 recommendations (R-hat < 1.01, ≥4 chains, ESS > 400) opened.
- **S13.** emcee documentation, "Autocorrelation analysis & convergence" (chains of about 50 tau often sufficient) and "Saving & monitoring progress" (N > 100 tau and Δtau < 1%). https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
- **S14.** Foreman-Mackey D., Hogg D. W., Lang D., Goodman J. 2013, "emcee: The MCMC Hammer", arXiv:1202.3665 (PASP). Section 4 "Discussion & Tips" opened.
- **S15.** Hogg D. W., Foreman-Mackey D. 2018, "Data analysis recipes: Using Markov Chain Monte Carlo", arXiv:1710.06068 (ApJS). Sections 4 and 5 opened.
- **S16.** Yao Y., Vehtari A., Gelman A., "Stacking for Non-mixing Bayesian Computations: The Curse and Blessing of Multimodal Posteriors", arXiv:2006.12335 (abstract opened; journal version JMLR 2022 [unverified]).
- **S17.** Romero-Shaw I. M. et al. 2020, "Bayesian inference for compact binary coalescences with BILBY: validation and application to ... GWTC-1", arXiv:2006.00714 (MNRAS). Section 3.2 and Figure 4 opened.
- **S18.** Boender C. G. E., Rinnooy Kan A. H. G. 1987, "Bayesian stopping rules for multistart global optimization methods", Math. Program. 37: 59-80, doi:10.1007/BF02591684. The abstract was opened through the Semantic Scholar and Crossref APIs; the full text was not.
- **S19.** NumPy documentation, "Parallel random number generation" (SeedSequence spawning; `root_seed + worker_id` is unsafe). https://numpy.org/doc/stable/reference/random/parallel.html
- **S20.** Numba documentation:
  - "Supported NumPy features", random section (independent RNG state; Generator not thread-safe);
  - "Floating-point pitfalls" (not bit-compatible with NumPy);
  - "Performance tips" (fastmath; SVML 1 vs 4 ULP);
  - "Automatic parallelization" (prange reductions).
- **S21.** NumPy documentation, `numpy.testing.assert_allclose` (the formula; defaults rtol=1e-7, atol=0).
- **S22.** Dawson B. 2012, "Comparing Floating Point Numbers, 2012 Edition". https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
- **S23.** pip documentation, "Repeatable Installs" (pinning, hash-checking): https://pip.pypa.io/en/stable/topics/repeatable-installs/. uv documentation, "Locking and syncing": https://docs.astral.sh/uv/concepts/projects/sync/
- **S24.** Taschuk M., Wilson G. 2017, "Ten simple rules for making research software more robust", PLoS Comput Biol, doi:10.1371/journal.pcbi.1005412. Rules 4, 9 and 10 opened.
- **S25.** GitHub Docs, "About protected branches" (required status checks, strict vs loose, required reviews, plan availability).
- **S26.** Google Engineering Practices, "The Standard of Code Review" and "What to look for in a code review". https://google.github.io/eng-practices/review/reviewer/
- **S27.** Vocke H. 2018, "The Practical Test Pyramid", martinfowler.com.
- **S28.** Fowler M., "Continuous Integration" (self-testing build, fix broken builds immediately, keep the build fast, deployment pipeline). https://martinfowler.com/articles/continuousIntegration.html
- **S29.** Google Testing Blog 2010, "Test Sizes" (small/medium/large; 60/300/900+ s). https://testing.googleblog.com/2010/12/test-sizes.html
- **S30.** Schwaber K., Sutherland J. 2020, The Scrum Guide, "Commitment: Definition of Done". https://scrumguides.org/scrum-guide.html
- **S31.** mutmut documentation. https://mutmut.readthedocs.io/

## 8. The 9 most important checks (summary)

1. **A1** Known-answer tests for every kernel, plus a convergence-order test.
2. **A3** Metamorphic relations listed and run per commit.
3. **B1** SBC with data-dependent test quantities and a binomial-band pass threshold, covering the whole pipeline.
4. **B2** Injection-recovery coverage (P-P with bands), with prior-averaged and fixed-truth studies kept apart.
5. **B3** Hard runtime MCMC gates (N/tau, tau stability, acceptance, R-hat < 1.01 across independent ensembles, ESS > 400).
6. **C1** Multi-start statistics and an explicit stopping rule.
7. **C2** Known-truth global-minimum recovery on hard synthetic cases.
8. **D1** SeedSequence-spawned, recorded seeds (numba RNG handled) and rerun at tolerance.
9. **E1** Tiered gates with time budgets, main always green, and revert on failure.
