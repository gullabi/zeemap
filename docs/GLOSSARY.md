# Glossary

Plain-language meanings of the words used in this repository, in alphabetical order.

| term | meaning |
|---|---|
| autocorrelation time | how many MCMC steps it takes before a chain "forgets" where it was. A chain of length 50 x this time holds about 50 nearly independent samples per walker. |
| B | the strength of the magnetic field at a point on the star's surface, in megagauss (MG). The Earth's field is about 0.5 gauss; magnetic white dwarfs have from about 1 kG to about 1000 MG. |
| B10, B50, B90 | the field strength below which 10, 50 or 90 % of the visible, brightness-weighted area lies, at one phase |
| band (mode band) | the range of fit quality within which every mode is reported: by default 45 nats from the best mode |
| batch | several parameter sets evaluated in one call, which is much faster than one at a time |
| Bonferroni correction | when many tests are made at once, each is made stricter (its error rate divided by the number of tests), so that the chance of any false alarm stays small |
| bootstrap standard error | the spread of a statistic over resamples of the data drawn with replacement |
| budget | how much computing a search may use: the number of starts, restart rounds and evaluations |
| calibrated (interval) | an interval whose coverage was measured, for this object and data set, with injections that include data made with another library version (V15b), and passed (UNC-4); every other interval is "uncalibrated" |
| canonical form | one chosen representative of a group of parameter sets that give identical spectra (DESIGN.md section 2) |
| chi2 (chi-squared) | the sum of squared differences between model and data, each divided by the squared error. Smaller is better. |
| CI | continuous integration: GitHub runs tier-1 checks automatically on every push |
| clean room | a way of writing code from a description of another program's behaviour, without looking at that program's code |
| continuum-matched weights ("planck") | the default Teff interpolation: per-pixel weights chosen so that a Planck continuum is reproduced exactly between two Teff slices (DESIGN.md section 5c) |
| cos psi | the cosine of the angle between the field at a patch and the direction to the observer. The polarisation signal changes sign with it. |
| coverage | how often an interval contains the true value, measured on simulated data. A correct 68 % interval contains it 68 % of the time. |
| derived quantity | a number computed from the parameters, such as inclination, B50 or the centre offset (FORMATS.md section 7) |
| dipole, quadrupole, octupole | the simplest field shapes: a bar magnet (dipole, two poles) and more complex ones with four (quadrupole) and eight (octupole) poles. zeemap adds the three, each with its own axis. "dqo" is the model with all three. |
| discrepancy term | a part of the likelihood that allows for smooth differences between model and data that no field geometry can produce |
| effective sample size (ESS) | the number of independent samples an MCMC chain is worth. "Bulk" ESS concerns the middle of the distribution, "tail" ESS its ends. |
| ensemble, walker | emcee moves many points ("walkers") through parameter space together; one such group is an ensemble |
| evaluation | one computation of the model spectra and -ln L for one parameter set |
| evidence (Bayesian) | the probability of the data under a whole model, used to weight models or modes against each other; zeemap does not use it |
| family envelope | the range of a quantity over a family's end points and profile points that lie below the family level (UNC-8) |
| family level | f_best + h, with f_best the best end point's -ln L and h = max(value tolerance, `search.family.min_level`, default 5 nats): the one -ln L below which paths join end points into families (SEARCH-9) |
| family | a set of search end points joined by paths along which -ln L stays below the family level (SPEC SEARCH-9, DESIGN.md section 3b); one family can hold several modes |
| fingerprint | a short summary of a solution (an angle, the field centre, B50, B90 and a free Teff) used to decide whether two solutions are the same mode; with one phase the angle is beta, the angle between the dipole and the line of sight (DESIGN.md section 3, step 3) |
| fit window | a wavelength interval that counts in the fit |
| floor chi2 | the best chi2 that any mixture of library spectra could reach. It shows how much of the misfit is the field model's fault. |
| forward model | the computation from parameters to predicted spectra |
| gate | a set of checks that must pass before a step (a push, a merge, a release) is allowed |
| gauge | a freedom in the parameters that changes nothing physical, here the length of the quadrupole axis vector a_q |
| Gauss-Newton | a quick estimate of the curvature at a best fit, used to start the MCMC |
| hits | how many search starts ended in a given mode |
| inclination (i) | the angle between the star's rotation axis and the direction to the observer |
| injection and recovery | simulating data from a known answer ("injecting" it) and checking whether the fit gets it back |
| invalid model | a parameter set the forward model cannot evaluate (for example a field centre outside the star) |
| jackknife | refitting with one phase left out at a time; the spread of the results (the jackknife sigma) shows how much one phase drives a result |
| Jeffreys prior | a standard "know nothing" prior for a scale: uniform in its logarithm |
| kernel (noise kernel), kernel fit | the covariance of the noise between pixels, here two Matern-3/2 terms with a short and a long length; the kernel fit chooses their lengths and shares for each spectrum from the residuals of the best fit, once per run, and then keeps them fixed ("frozen") (LIKE-2) |
| Kolmogorov-Smirnov test | a test of whether two samples come from one distribution, from the largest gap between their cumulative distributions |
| Kullback-Leibler divergence | how much one distribution differs from another, in nats; 0 when they are equal |
| Laplace approximation | treating the region around a best fit as a Gaussian, computed from the curvature there. Fast, often too narrow here. |
| library node | one grid point of the library (one Teff, one B, one psi) |
| library | the precomputed model spectra, one for each combination of temperature, B and psi on a grid, made by a magnetic model-atmosphere code; not part of this repository. |
| likelihood, -ln L | the probability of the observed data given a set of parameters; -ln L is its negative logarithm. Smaller -ln L is better. |
| limb darkening, mu | the edge (limb) of the star looks darker than its centre; mu is the cosine of the viewing angle at a patch (1 at the centre, 0 at the edge) |
| lock file | a file listing the exact version of every software package, so the environment can be rebuilt identically |
| MAGWD | S. Jordan's program that computes the library spectra. Its code is not part of this repository. |
| maintainers | the people with write access to the zeemap repository; they review pull requests and merge them into main (GATES.md) |
| marginalise, integrate out | remove a parameter we do not care about by averaging over all its values instead of fitting it |
| Matern-3/2 kernel | a standard formula for how correlated two pixels are, as a function of their distance in wavelength |
| MCMC | Markov-chain Monte Carlo: a random walk through parameter space that spends time in each region in proportion to its probability. Used here for error bars inside a mode. |
| median absolute deviation | the median of the distances of the values from their median: a spread measure that ignores outliers |
| mode | a group of search end points with nearly the same geometry, judged by their fingerprint (DESIGN.md section 3); a mode need not be a separate basin of the likelihood: several modes can lie in one family |
| MWD | magnetic white dwarf |
| nat | the unit of -ln L. A difference of 1 nat is a factor e (about 2.7) in likelihood, or 2 in chi2 for Gaussian errors. |
| nested sampling | a method that computes the Bayesian evidence by shrinking the prior volume step by step; tried in the research and dropped |
| nuisance parameter | a parameter needed to model the data that we do not care about, such as the continuum level. A "declared nuisance" coordinate of the MCMC is sampled but not required to converge and gets no interval: the orientation of a single-phase fit (decision #18) |
| Numerical Recipes | a book of numerical routines whose code may not be redistributed |
| optimiser scatter | the standard deviation of -ln L over the end points of one mode after polishing (DESIGN.md section 3); before the Levenberg-Marquardt finish it was 5-25 nats in the research |
| patch | one small piece of the star's surface in the model |
| phase | where the star is in its rotation, from 0 to 1 |
| planted problem | synthetic data made from a geometry chosen so that its best mode is hard to find; used to test the search |
| polish | improve a point with the local optimiser: L-BFGS-B, then Levenberg-Marquardt (DESIGN.md section 3) |
| posterior, prior | the probability of the parameters after (posterior) and before (prior) seeing the data. Here the start distribution plays the role of the prior. |
| profile likelihood | the best fit quality as a function of one quantity, with all other parameters re-optimised; per family, the lower envelope of several profile walks (UNC-5) |
| profile walk | computing a profile by stepping a quantity along a grid and re-polishing at each step from the previous point (DESIGN.md section 4b) |
| psi | the angle between the field and the direction to the observer (see cos psi) |
| R-hat | a number that compares independent MCMC runs; close to 1 means they agree. "Split": each run is cut in halves, which also detects drift; "rank-normalised": computed on ranks, so heavy tails do not distort it. |
| RE J0317-853 | the magnetic white dwarf used as the main test case (8 phases of spectra and polarisation) |
| recorded run | a run whose code version, configuration, inputs and seed are stored with its outputs, so it can be rerun and compared at tolerance (GATES.md, gate C) |
| seed | the starting number of a random-number generator; the same seed gives the same random numbers |
| simplex, evolution strategy, CMA-ES, differential evolution | optimisers that need no gradients; the simplex moves dim + 1 points, the others evolve a population of points |
| simulation-based calibration (SBC) | a check of a whole inference method: repeatedly draw a truth, simulate data, fit, and test that the truth's rank among the posterior samples is uniform |
| simultaneous band | a band that a whole curve (here the cumulative distribution of ranks) stays inside with a stated probability |
| spectrum (in fit quality) | one phase and one Stokes quantity (I or V/I) |
| start distribution | the distribution from which the search draws its starting points (DESIGN.md section 3) |
| Stokes I, V, V/I | I is the total light; V is the circularly polarised part; V/I is the fraction that is circularly polarised |
| swap | replacing one input we do not fully trust (library version, Teff, the set of phases) by an alternative and refitting |
| synthetic library | a small fake library made by the test code from simple formulas, so tests can run without real data |
| systematic shift | how far a result moves under a swap |
| Teff, Teff slice | effective temperature of the star's surface, in kelvin; a Teff slice is the part of the library for one temperature (one `.bin` file) |
| tempered width | an error bar multiplied by sqrt(T) to allow for model errors; dropped by decision #24, because no single T was right |
| tier | a group of checks with similar cost: tier 1 runs on every pull request, tier 2 before merging a change that can move numerical results and before a release, tier 3 before a release or a paper |
| tolerance | how much two numbers may differ and still count as equal |
| value tolerance | how close two -ln L values must be to count as the same: max(2 nats, min(2 x optimiser scatter, 10 nats)) (DESIGN.md section 3) |
| whiteness check | a test that the residuals, after whitening with the fitted kernel, are uncorrelated from pixel to pixel (LIKE-10) |
| whitening | transforming residuals with correlated noise into residuals with independent noise, so they can be summed as in chi2 |
