# What zeemap must do (specification)

**Status:** describes zeemap 0.1.0; the history of its decisions is in [DECISIONS.md](DECISIONS.md).
**Who this is for:** anyone who will build, review or use zeemap, including AI coding agents.
**You need to know:** what a spectrum, a chi-squared fit and a magnetic dipole are. Every other term is explained where
it first appears, and in [GLOSSARY.md](GLOSSARY.md).
**What you will know afterwards:** the problem, what goes in and comes out, and every requirement with the check that
proves it. *How* it is built: [DESIGN.md](DESIGN.md). *In which order*: [TASKS.md](TASKS.md). File layouts:
[FORMATS.md](FORMATS.md).

**Keywords.** MUST, MUST NOT, SHOULD and MAY in capitals mean what RFC 2119 and RFC 8174 say: MUST is an absolute
requirement; SHOULD is a requirement that may be skipped only for a reason that is written down; MAY is optional.
Words in lower case have their everyday meaning.

**Units.** Fit quality is given as -ln L, the negative logarithm of the likelihood. A difference of 1 in -ln L is
called 1 nat. For independent Gaussian errors, a difference of 1 nat equals a difference of 2 in chi-squared. Field
strengths are in megagauss (MG), wavelengths in angstrom (A).

## 1. Summary

A magnetic white dwarf rotates, so we see different parts of its magnetic field at different times. Its spectrum (the
total light, called Stokes I) and its circular polarisation (the polarised fraction, V/I) change with the rotation
phase. zeemap finds the field geometries that reproduce spectra observed at several phases, and says how well each one
is pinned down.

It works by comparison with a **library**: model spectra computed beforehand by another program (MAGWD, by S. Jordan).
Each library spectrum belongs to a small piece of stellar surface with a given temperature (Teff), field strength (B)
and angle between the field and the line of sight (psi). zeemap covers the star with small patches and looks up each
patch in the library. It adds up what the observer sees, and compares the sum with the data.

Fitting a field geometry this way has two general problems this tool must solve:
1. **One best fit hides the others.** A local optimiser started by hand stops in the first minimum it reaches, and
   many quite different field geometries fit almost equally well. A single best fit says nothing about the other
   solutions, nor about whether a better one was missed.
2. **chi2 is not a likelihood.** Error files often underestimate the noise, by several times, and neighbouring pixels
   are correlated. Intervals read off chi2 with such errors are neither too wide nor too narrow in any predictable way.

So zeemap finds the good solutions and reports them in two ways, side by side (decision #29). A **mode** is a group
of search end points with nearly the same geometry (its **fingerprint**); each mode gets error bars from MCMC inside it.
A **family** is a set of end points joined by paths along which the fit stays good; each family gets profiles and an
envelope: how far each quantity can move while the fit stays within a stated distance of the best. Which of the two
is the unit of uncertainty for published results is open (D12). Error bars are **tested on simulated data**, and they count as
**calibrated** only when that test included data made with another library version (UNC-4). zeemap lists separately
how much each result moves when an input we do not trust is changed (the **systematic shifts**). The noise model is
fitted to each object's residuals (LIKE-2), and the treatment of the I continuum is stated in every table (LIKE-3),
because both change the numbers.

## 2. (removed 2026-11-08)

## 3. Goals and non-goals

**Goals**
- G1. Find the modes from random starts. Report how often each mode was reached, and warn when the search cannot bound
  the chance that a better mode was missed.
- G2. Per mode, statistical intervals whose coverage has been measured on simulated data, including data made with
  another library version; per family, profiles and envelopes next to them.
- G3. Per mode, systematic shifts under named changes of the inputs, reported separately from G2.
- G4. Offer the field models, data handling and settings that published fits of magnetic white dwarfs use, so their
  set-ups can be refitted.
- G5. Every result can be rerun later and agrees within stated tolerances.

**Non-goals** (things a reader might expect, which zeemap deliberately does not do)
- **It does not compute model spectra or atmospheres.** The library comes from MAGWD.
- **It does not weight modes by Bayesian evidence.** In the research, the order of the best modes changed sign under
  every library and temperature swap and between likelihood variants, and nested-sampling evidence differed by 110 nats
  between two seeds. Such weights would be decided by systematics, not by the data.
- **It does not fold systematic shifts into the statistical error bars.**
- **It has no graphical interface and is not a web service.** It is a command-line program that writes files.
- **It does not ship libraries or observations.** Users supply their own, under `$ZEEMAP_DATA`.

## 4. A first fit

This walk-through shows a fit with the commands and file names of zeemap. The numbers are illustrative, not results:
they show the shape of the output.

```
$ export ZEEMAP_DATA=/path/to/your/data          # library and observations, outside the repository
$ zeemap fit star.toml -o out/ --workers 12 --seed 1 --mcmc
search: 400 starts + 2 restart rounds, 3.1e6 evaluations, 1.5 h
noise kernel fitted per spectrum (two Matern lengths); whitened residuals: white in 16 of 16 spectra
modes within 45 nats of the best: 11 (7 reached by one start only: the search cannot rule out missed modes)
families at the 6.2-nat level: 2 (9 of the 11 modes; 2 modes lie above the level: band components only)
mcmc: 11 modes x 4 ensembles; all converged
profiles: 1 family x 6 quantities (B10, B50, B90 at the reference phase), 2 walks each
systematics: 14 swaps x 11 modes; 2 refits changed mode
wrote out/run.json out/modes.json out/intervals.json out/systematics.json out/diagnostics.json out/spectra/ ...
```

`out/report.txt` then starts like this:

```
likelihood: full; kernel fitted, frozen; continuum: Legendre degree 3 on I, marginalised; value tolerance 6.2 nats
mode  family  delta -lnL  starts  incl         |dipole| MG   B50 at phase 0.25 (MG)
1     A       0           3       60 +3 -4     52 +2 -2      31.4 +0.2 -0.2
2     A       4           2       57 +4 -3     54 +2 -3      31.3 +0.3 -0.2
3     B       4.5         1       30 +5 -5     61 +3 -3      31.3 +0.3 -0.2
...
family  modes  delta -lnL  incl: profile (delta < 0.5)  envelope (below level)  mode 1 68 %  ratio  ratio / sqrt(2h)
A       2      0           56-63                        44-68                   56-63        3.4    0.97
...
intervals: statistical (68 %), uncalibrated: no coverage run with other-library injections named in the configuration
systematic shifts of mode 1: library B instead of A: incl +8 ...; continuum degree 0: Teff -600 K ...
limited by model systematics (a swap moves it by more than 3 sigma): incl, B90, centre offset
```

Here "delta -lnL" is the distance in nats from the best mode, known to within the value tolerance of DESIGN.md section
3 (between 2 and 10 nats, printed in the report); modes are told apart by their geometry (fingerprint), not by -ln L,
so two modes may lie closer than the value tolerance; a family (A, B) holds the modes whose end points are joined by
paths that stay within the family level of the best (SEARCH-9); its "profile" interval is where the best fit with the
quantity held fixed stays within 0.5 nat of the family's best, and its "envelope" where it stays below the family
level; "ratio" compares the envelope's width with the best mode's 68 % interval, "ratio / sqrt(2h)" is 1 for a single Gaussian basin, and "!" marks it above 1.5 (UNC-8);
"ensembles" are the independent MCMC runs per mode (DESIGN.md section 4); a "swap" is a refit with one untrusted
input replaced (SYS-1); the "kernel" is the noise covariance fitted to this object (LIKE-2). B50 and B90 are the field strengths below which half and
90 % of the visible, brightness-weighted area lie (FORMATS.md section 7). Libraries A and B are two versions of the model library, computed
with different physics settings.

## 5. Inputs and outputs

Inputs: a library and observations (not part of zeemap, under the directory named by the environment variable `$ZEEMAP_DATA`), a
TOML configuration. Outputs: one directory per run with `run.json`, `modes.json`,
`families.json`, `intervals.json`, `profiles.json`, `systematics.json`, `diagnostics.json`, model spectra, visible
(B, psi) histograms and `report.txt`.
Every layout is in [FORMATS.md](FORMATS.md).

## 6. Requirements

**How to read a requirement.** Each has an ID, a keyword, the goal it serves, and the check that proves it. "Checked
by" names a loop in [VERIFICATION.md](VERIFICATION.md), and the loop lists the requirement under "Covers". A
requirement without a check is not finished. `[NEEDS DECISION Dn]` marks a point that is not decided yet (section 8);
do not build that point while the marker is there.

### 6.1 Library (LIB)
- **LIB-1** MUST read the library format of FORMATS.md section 1. (G4) Checked by: V2.
- **LIB-2** MUST load each selected Teff slice (the library part for one temperature) once per process and keep it for
  the whole run. (G1) Checked by: V2.
- **LIB-3** When the configuration lists several Teff slices, zeemap MUST interpolate between the two slices that
  bracket Teff. By default it MUST use continuum-matched per-pixel weights (DESIGN.md section 5c), which are exact
  for a spectrum that is a fixed shape times the Planck function ("planck"); interpolation linear in ln Teff MUST stay available
  as an option (`model.interp_teff`). Reason: with linear-in-ln-Teff weights and 2000 K slices, fits of a real low-field object
  had cusps and a false Teff minimum (decision #26). (G1) Checked by: V2. `[NEEDS DECISION D7]`: whether Teff is free
  by default.
- **LIB-5** If a patch's B lies outside the library range, zeemap MUST either use the edge value or mark the model
  invalid, as the configuration says. (G4) Checked by: V2.
- **LIB-6** The outputs MUST state, per mode and phase, the fraction of the brightness-weighted area (FORMATS.md section 7) whose B lies
  outside the library range. (G3) Checked by: V2.
- **LIB-7** If the configuration asks for a Teff (a slice in `library.teffs`, or a `teff` range) or a wavelength range
  that the library does not cover, or a fit window closer than 4 sigma of the convolution kernel (FWD-3) to an end of the library range, zeemap MUST stop before
  fitting, with an error message that names the covered range. (Field strengths outside the library are handled by LIB-5 and the prior
  cut of DESIGN.md section 3.) (G5) Checked by: V4.
- **LIB-8** MUST accept a library with more than one mu value (FORMATS.md section 1): it then interpolates each
  patch's spectrum linearly in mu between the two bracketing mu values, and applies no limb-darkening law, because
  the library holds the angular dependence. A mu outside the library's mu range takes the nearest edge value and is
  counted in `diagnostics.json`. Reason: the research plan needs mu-grid libraries; the limb treatment biases V/I by
  7-15 % (decision #27). (G2) Checked by: V2.

### 6.2 Surface and field (SURF, FIELD)
- **SURF-1** MUST offer an evenly spread grid of patches (a Fibonacci sphere) with a configurable number of patches.
  (G1) Checked by: V1.
- **SURF-3** The default number of patches for error bars MUST be at least 20 000, and for the search at least 5000.
  Reason: below these counts, the research found the error bars and the order of modes still changing. (G2) Checked by:
  V11.
- **FIELD-1** MUST provide the field model "dqo": a dipole, a quadrupole and an octupole, each with its own axis,
  sharing one centre offset. (G4) Checked by: V1.
- **FIELD-2** MUST provide "dqo_aligned": the same with all three axes on the dipole axis. (G4)
  Checked by: V3.
- **FIELD-3** Each field component MUST be the gradient of its magnetic potential. (G1) Checked by: V1.
- **FIELD-4** SHOULD provide general multipoles up to l = 5, added without changes to the search or error code.
  Reason for SHOULD: no fit compared with zeemap so far used them.
  Built: "harm4", the full centred expansion up to l = 4 with all orders m (Euchner et al. 2006, Table 4; Jordan 2003),
  DESIGN.md section 2. It is built in, not a plug-in: its coefficient names are in `config.HARM_NAMES` and
  `Problem.nuisance_coordinates` names them. A model with parameter names of its own still needs changes in `src/`
  (V7e, the xfail toy model).
  (G1) Checked by: V1, V7.

Example (FIELD-3), Given-When-Then:
*Given* a centred dipole with polar field 100 MG along z, *when* the field is evaluated at the pole and at the equator
of the star, *then* |B| is 100 MG at the pole and 50 MG at the equator, to 1e-12.

### 6.3 Parameters (PAR)
- **PAR-1** Every parameter MUST have a value, a starting step, a free-or-fixed flag and a range in the configuration.
  (G4) Checked by: V1.
- **PAR-2** By default the search MUST use vector parameters (DESIGN.md section 2): they have no angle box, whose edges
  created false minima (40 of 150 minima of earlier fits in angle parameters sat on an edge). (G1) Checked by: V3.
- **PAR-4** Starting points MUST be drawn from a start distribution given in the configuration, with documented
  defaults (DESIGN.md section 3), and the distribution used MUST be written to `run.json`. The same distribution is the
  prior of the MCMC (DESIGN.md section 4). (G1, G5) Checked by: V7, V13. `[NEEDS DECISION D8]`: whether the
  configuration may also set other priors for the MCMC.

### 6.4 Forward model (FWD)
- **FWD-1** For each phase, zeemap MUST sum the library spectra of all visible patches (mu > 0), each weighted by its
  area, by mu and by limb darkening. (G1) Checked by: V2.
- **FWD-2** MUST offer the limb-darkening laws 1, 0.4 + 0.6 mu and 0.7 + 0.3 mu (`model.ld_mode` 0, 1, 3). (G4) Checked by:
  V1.
- **FWD-3** MUST convolve the model with a Gaussian of the configured full width at half maximum (FWHM). (G4) Checked
  by: V2.
- **FWD-4** MUST resample the model onto the data wavelengths. (G4) Checked by: V2.
- **FWD-5** MUST form model V/I as V divided by I after summing over the visible disc. (G4) Checked by: V2.
- **FWD-6** By default, zeemap MUST interpolate the library linearly in B and in cos psi. (G1) Checked by: V11.
- **FWD-10** MUST accept a phase offset added to all phases. (G4) Checked by: V3.
- **FWD-11** MUST evaluate a batch of parameter sets in one call. (G1) Checked by: V5.
- **FWD-12** One evaluation at 5000 patches, 8 phases and 900 pixels MUST take less than 10 ms on one core of the
  reference machine (DESIGN.md section 7). (G1) Checked by: V12.
- **FWD-13** MUST offer a velocity shift parameter (a shift proportional to wavelength, in km/s), global, SHOULD offer
  it per phase, and SHOULD offer a constant shift in A. A positive shift MUST move the model's lines to longer
  wavelengths (a redshift): lambda (1 + v / c) for a velocity v, lambda + v for a constant. The shift MUST be applied
  to the model on the library's wavelength grid, before convolution and resampling; the data MUST NOT be interpolated.
  Reason: model line lists can be slightly off in wavelength, so a shift is needed; shifting the data by interpolation
  instead smooths them, which lowers -ln L whatever the direction of the shift (decision #27). (G1) Checked by: V2.
- **FWD-14** SHOULD offer phase smearing: averaging the model over the exposure's phase range. (G4) Checked by: V3.
- **FWD-15** SHOULD offer an interpolation in B with a continuous first derivative (monotone piecewise-cubic Hermite,
  "PCHIP"). It becomes a MUST if the linear
  interpolation fails the smoothness check V11. (G2) Checked by: V11.

Example (FWD-1, FWD-3), Given-When-Then:
*Given* a synthetic library and a field model with the same B and psi on every patch, *when* the model spectrum is
computed and divided by the sum of the patch weights, *then* it equals that one library spectrum after convolution, to
1e-5 (the float32 tolerance).

### 6.5 Data, errors and weights (DATA)
- **DATA-1** MUST allow I and V/I to be switched on or off separately for each phase. (G4) Checked by: V3.
- **DATA-2** MUST read errors as absolute values or relative to I, raise them to configurable floors, and, when no V/I
  error file is given, derive the V/I error from the I error as sigma_VI = sigma_I,rel x sqrt(1 + (V/I)^2). (G4) Checked
  by: V1.
- **DATA-3** MUST restrict the fit to configured wavelength windows and drop configured excluded ranges, separately for
  I and V/I. (G4) Checked by: V1.
- **DATA-4** MUST accept weights per wavelength window, phase and Stokes quantity as settings, with every likelihood. A
  weight w acts as the pixel errors divided by sqrt(w): in the plain chi2 it multiplies the squared
  residuals; in the full likelihood it enters D, the diagonal of pixel errors (DESIGN.md section 5). Weights are written
  in the report header. A weight of 0 removes the pixel. With the full likelihood, whose noise scale is fitted per
  spectrum (LIKE-1), a weight that is the same over a whole spectrum changes -ln L only by a constant: only weights
  that differ inside a spectrum act, and the report says so for a weight that covers whole spectra. Reason: the
  fits this tool was built for weight the polarisation windows (decision #27). (G2) Checked by: V3.
- **DATA-6** If the observed wavelength grid is not regular to 1e-3 of a step, or does not overlap the library, zeemap
  MUST stop with an error that names both ranges. (G5) Checked by: V4.
- **DATA-7** SHOULD offer fits in absolute flux, with the dilution factor (R/d)^2 from radius and distance. Reason for
  SHOULD: the fits compared with zeemap so far are normalised. (G4) Checked by: V2.

### 6.6 Likelihood (LIKE)
- **LIKE-1** The default likelihood ("full") MUST include an unknown noise scale for each spectrum, integrated out in
  closed form. Reason: error files often underestimate the noise, by several times in real data sets. (G2) Checked by: V1, V15.
- **LIKE-2** It MUST include correlated noise between pixels, with a kernel fitted to each object and data set: for
  each spectrum, the sum of two Matern-3/2 kernels (a standard covariance shape), a short one and a long one, plus a
  white part, with the two lengths, the share of the short one and the white part fitted by maximum marginal likelihood at the best end point of a first
  search, refitted once after the band is re-polished, then frozen for the rest of the run (DESIGN.md section 5). The
  fitted values MUST be written to `run.json`, and each parameter that ends at a bound of its range MUST be named per
  spectrum, with its side, in `run.json` and `report.txt`; a configuration MAY instead give the kernel of each spectrum (for
  example from an earlier run), and then nothing is fitted. Reason: one fixed 4 A kernel, made for RE J0317-853, left
  the residuals of a real low-field object correlated, and a fitted kernel changed which geometry was best
  (decision #22). (G2)
  Checked by: V1, V4, V15.
- **LIKE-3** It MUST include a smooth continuum for I: the model I times a Legendre polynomial of degree 3 by default,
  plus a constant where the zero point of I is unknown, integrated out in closed form. The degree, or "off", is a
  setting, and the setting used MUST be stated in the header of every table in the report and in `run.json`. The
  continuum basis MUST be normalised by the model's own norm, so that -ln L does not change when the model I is
  multiplied by a constant (decision #26). Reason: on real data the continuum treatment can move the fitted Teff by
  many kK and change which of two library versions fits better (decision #23). (G2) Checked by: V1, V3, V4, V15.
- **LIKE-4** The long kernel of LIKE-2 is the model-discrepancy term: correlated noise with a long correlation length
  that absorbs smooth misfits the field model cannot produce. Its length and share are fitted per object and data set,
  never per mode (decision #22, which closed D6). (G2) Checked by: V1, V15.
- **LIKE-6** MUST offer a plain chi2 for quick looks and tests: the full likelihood with the noise scale, correlated
  noise and discrepancy switched off, and the continuum coefficients fitted by linear least squares; -ln L = chi2 / 2.
  (G1) Checked by: V1.
- **LIKE-7** If a model cannot be evaluated (for example a field centre outside the star), the likelihood MUST return a
  defined finite worst value, never NaN. (G1) Checked by: V4.
- **LIKE-8** The run MUST count invalid models and write the count to `diagnostics.json`. (G1) Checked by: V4.
- **LIKE-9** Each nuisance term (noise scale, correlated noise, continuum, discrepancy) MUST be switchable off in the
  configuration, so it can be fixed for tests and calibration. (G2) Checked by: V1, V16.
- **LIKE-10** After the kernel is frozen, zeemap MUST check, for each spectrum at the best end point, that the whitened
  residuals are white: their autocorrelation at lags of 1 to 10 pixels stays below 3 / sqrt(n) + 0.05 in absolute
  value (n = the spectrum's number of pixels). The result per spectrum MUST be written to `diagnostics.json` and the
  report. If any spectrum fails, every -ln L difference and every interval of the run MUST carry the flag "noise model
  fails the whiteness check", and no interval may be labelled "calibrated" (UNC-4). Reason: with residuals that are
  not white, the nats are wrong by an unknown factor (decision #22). (G2) Checked by: V1, V4.

### 6.7 Search for modes (SEARCH)
- **SEARCH-1** `zeemap fit` MUST find modes without the user giving starting points. (G1) Checked by: V7, V17.
  How: DESIGN.md section 3. The search uses the full likelihood throughout (decision #15).
- **SEARCH-2** MUST treat symmetric copies of one geometry as one mode (DESIGN.md section 2). (G1) Checked by: V3.
- **SEARCH-3** MUST report for each mode how many starts reached it. (G1) Checked by: V7.
- **SEARCH-4** MUST report the search statistics, per mode and per family (SEARCH-9): number of starts, number of
  distinct modes and families, hits on the best mode and on the best family, and the fraction of the modes in the band
  (SEARCH-7) that were reached by one start only. When that fraction is above
  0.5, the report MUST warn that the chance of a missed better mode cannot be bounded. Reason: when most modes were
  found only once, every new start is likely to find yet another mode, so the starts so far say little about what is
  still unseen. (G1) Checked by: V4, V17.
- **SEARCH-5** With the same seed and configuration, the result MUST be the same whatever the number of worker
  processes. (G5) Checked by: V6.
- **SEARCH-6** At the default budget, the search MUST reach the best mode of the planted test problem in at least 28
  of 30 seeds (VERIFICATION.md V17). (G1) Checked by: V17.
- **SEARCH-7** The mode table MUST contain every mode within a band of the best mode, with a default of 45 nats,
  measured in the nats of the frozen kernel (LIKE-2). The width is provisional: its earlier reason, the tempering of
  UNC-6, was dropped (decisions #10, #24). (G1) Checked by: V7.
- **SEARCH-8** MUST re-polish and re-rank the modes at the error-bar patch count (SURF-3) before reporting, and MUST
  measure the value tolerance (DESIGN.md section 3) at that patch count. Reason: the order of modes changed between
  5000 and 20 000 patches in the research, and the roughness of the fit surface is larger at 5000 (decision #25). (G1)
  Checked by: V3n, V17.
- **SEARCH-9** (a proposal, decision #29: the rule becomes final with D12) MUST group the end points in the band into **families** (decision #29): two end points belong to one
  family when a path between them stays below the family level, f_best + max(value tolerance,
  `search.family.min_level`), where f_best is the best end point's -ln L. A path is the straight segment between the
  two end points (between the closest pair of their symmetric copies), or, when that fails, two straight segments
  through a re-polished midpoint (DESIGN.md section 3b). Paths are evaluated at the error-bar patch count. The
  `modes.json` entry of each mode MUST name its family, and `families.json` MUST list per family its modes, its end
  points and the paths that joined them. The components at the band level (SEARCH-7) MUST also be reported. Reason:
  on a real low-field object the fingerprint cut one connected basin into several modes. (G1, G2) Checked by: V3, V7.

Example (SEARCH-6), Given-When-Then:
*Given* synthetic data from a known geometry whose best mode is reached by 0.5-2 % of random starts, *when* `zeemap
fit` runs at the default budget with seeds 1 to 30, *then* in at least 28 runs the mode table contains the planted best
mode, within the value tolerance of DESIGN.md section 3 of its reference value.

### 6.8 Uncertainty (UNC)
- **UNC-1** When the MCMC runs (`--mcmc`, or an MCMC budget `--mcmc-modes` / `--mcmc-evals`; decision #42), zeemap
  MUST give, for each mode it samples, 68 % and 95 % intervals for every free parameter and every derived quantity
  (FORMATS.md section 7), from the MCMC inside the mode. They MUST be labelled "per mode: conditional on this basin".
  (G2) Checked by: V5, V15, V16.
- **UNC-2** MUST NOT report intervals for quantities whose chains fail the V14 rules; such a quantity is flagged
  "not converged". The rules are applied to each quantity on its own (its R-hat, bulk and tail ESS, and chain length
  and stability against its own autocorrelation time), together with the run's drift and stuck-walker rules; the
  mode's own verdict is reported as well (decision #17). Coordinates a problem declares nuisance (the orientation of a
  single-phase fit, decision #18) are left out of the verdict and get no interval. (G2) Checked by: V14.
- **UNC-3** If the MCMC finds a point that, after polishing, is better than the mode by more than the value tolerance
  of DESIGN.md section 3, zeemap MUST hand that point back to the search and update the mode table, at most 3 times per
  mode. (G1) Checked by: V14.
- **UNC-4** MUST label intervals "uncalibrated" unless the configuration names a passed coverage run (V15) that
  includes injections made with another library version (V15b), made with the same likelihood settings (the frozen
  kernel included), the same library, and data of the same object and data set (the same phases, windows and error
  files), and unless the whiteness check of LIKE-10 passed. A coverage run with same-model injections only (V15a)
  tests the computation; it is named in the report as such and never makes an interval "calibrated". A new star needs
  its own coverage run. The kernel matches when every kernel parameter agrees to 1e-6 relative: in practice the run
  to be calibrated gives the kernel of the coverage run (`objective.kernel_fit = false` with `[[objective.kernel]]`
  copied from it). (G2) Checked by: V4.
- **UNC-5** MUST give, per family and reported quantity (`uncertainty.profile_quantities`; default: inclination,
  |m_d|, the offset along the dipole axis, B10, B50 and B90 at the reference phase, and Teff when free; by default for
  the best family only, `uncertainty.profile_families`; decision #40), a profile: the lowest
  -ln L with the quantity held at each grid value and everything else re-polished. Each profile MUST be walked in both
  directions from at least 2 members of the family (the best end point and the end point with the most distant value
  of the quantity), and the reported profile is the lower envelope of the walks. From it zeemap MUST give the
  intervals where it stays within 0.5 nat and within 2 nats of the family's own best, and the interval where it stays
  below the family level of SEARCH-9 (the one level that defines membership). Reason: on a
  real object, one walk from the best point reported the wrong branch (decision #29). (G2) Checked by:
  V1.
- **UNC-6** (withdrawn by decision #24) MUST NOT report tempered widths. (G2) Checked by: V4.
- **UNC-7** MUST report the jackknife sigma: the spread of each quantity over the leave-one-phase-out refits (SYS-1).
  (G3) Checked by: V18.
- **UNC-8** MUST report, per family and reported quantity, the family envelope: the range of the quantity over the
  family's end points and profile points that lie below the family level (SEARCH-9). A table per quantity MUST show
  next to each other the envelope, the profile intervals of UNC-5, and the per-mode MCMC 68 % interval of the family's
  best mode (UNC-1), with two ratios: R1, the envelope's width over that interval's width, printed and not flagged;
  and R2, the same ratio divided by sqrt(2 h), with h of SEARCH-9, flagged "envelope wider than one Gaussian basin
  gives" above 1.5. For a single Gaussian basin R1 is sqrt(2 h), at least 3.2 at h >= 5 nats, so a flag on it would
  mark every mode; R2 is 1 (decision #30, D13 settled on 2026-10-23). Without the MCMC (UNC-1) the
  mode's 68 % column, R1 and R2 are absent ("-"). Every
  such table states the kernel and the continuum setting in its header (LIKE-2, LIKE-3). Reason: decision #29 asks
  for both views side by side. (G2) Checked by: V4.
  `[NEEDS DECISION D12]`: which of the two views is the unit of uncertainty for published results. Both are built, so no task waits
  for it.

### 6.9 Systematics (SYS)
- **SYS-1** For each mode, zeemap MUST refit under each named swap in the configuration and under the default swaps.
  A swap replaces one input we do not fully trust by an alternative. Default list (decision #21, generated at run time
  while `systematics.defaults = true`): Teff fixed at each selected library slice; V/I scaled by 0.9 and by 1.1
  (`data.vi_scale`); each phase left out in turn; the other library version for each path in
  `systematics.library_alternatives`. Optional, in the configuration: I only; V/I only. (G3)
  Also default (decision #23): the continuum of LIKE-3 at degree 0 (`continuum_deg0`) and off (`continuum_off`: no
  Legendre continuum and no free constant for any I spectrum, whatever `i_offset` says; the report gives the scale
  ratio of each I spectrum next to it, because "off" is meaningful only when the I scale of model and data agree).
  Checked by: V7f, V18.
- **SYS-2** Each swap refit MUST start from at least 4 starting points near the mode, the same for every swap. (G3) Checked by: V18.
- **SYS-3** MUST report each shift separately and MUST NOT add shifts into the error bars. (G3) Checked by: V4.
- **SYS-4** When a swap refit lands in a different mode, zeemap MUST report "mode change" instead of a shift. Whether
  it left the mode is decided on the fingerprint components that the swap does not set: a swap that fixes Teff is
  compared without Teff. (G3) Checked by: V7, V18.
- **SYS-5** MUST flag a quantity "limited by model systematics" when any swap moves it by more than 3 statistical sigma.
  (G3) Checked by: V4.

### 6.10 Report and diagnostics (REP, DIAG)
- **REP-1** The report MUST NOT show a single best fit without the mode table and the family table (one row per
  family). (G1) Checked by: V4.
- **REP-2** The report MUST NOT show mode weights from Bayesian evidence. (G2) Checked by: V4.
- **REP-3** The report MUST flag every interval that reaches within 1 % of a parameter's range limit or of the prior's
  boundary. (G2) Checked by: V4.
- **REP-4** The report MUST flag a quantity "not constrained by the data" when its posterior hardly differs from its
  start distribution: the histogram estimate of KL(posterior || prior), corrected for its plug-in bias, is below
  0.1 nat (DESIGN.md section 4, step 6; decision #19). (G2) Checked by: V4.
- **DIAG-1** MUST compute the floor chi2: the best fit that any non-negative mixture of library spectra can reach at
  each phase (DESIGN.md section 5b2), next to the best mode's plain chi2 for the model I written to `spectra/` (the
  continuum fitted inside the likelihood). It shows how much a better field model could gain (decision #23). (G3) Checked
  by: V2.
- **DIAG-2** MUST write per-pixel residuals and the visible (B, psi) histogram per mode and phase (FORMATS.md
  section 6). (G3) Checked by: V7.
- **DIAG-3** MUST write, for the best mode of each family, a per-window residual table at fixed geometry: for every
  fit window and spectrum, the number of pixels and the whitened chi2 under the run's continuum setting and at degree
  0, for the run's library and for each library in `systematics.library_alternatives`, all at the same parameters
  (no refit). Reason: a comparison of physics versions is made window by window (decision #23). (G3) Checked by: V7.

### 6.11 Configuration and command line (CFG, CLI)
- **CFG-1** One TOML file MUST hold the whole configuration. (G5) Checked by: V1.
- **CFG-2** If the configuration contains an unknown key, zeemap MUST stop with an error naming the key. (G5) Checked
  by: V4.
- **CFG-3** Every key MUST have its default in `src/zeemap/defaults.toml`, with a one-line comment. (G5) Checked by: V1.
- **CLI-2** MUST provide the command `eval`: evaluate one parameter set. (G4) Checked by: V7.
- **CLI-3** MUST provide the command `fit`: search, error bars and systematics. (G1) Checked by: V7.
- **CLI-4** MUST provide the command `inject`: simulate data from a known geometry with the likelihood's noise model.
  (G2) Checked by: V7, V15.

### 6.12 Engineering (ENG)
- **ENG-1** MUST run on Python 3.12 with numpy, scipy, numba and emcee, all versions pinned in a lock file. (G5)
  Checked by: V8.
- **ENG-2** Parallel work MUST use one single-threaded process per core, with numba, BLAS and OpenMP threads set to 1.
  (G1) Checked by: V6.
- **ENG-3** Every random number MUST derive from the run's master seed, which `run.json` records. (G5) Checked by: V6.
- **ENG-4** The code, tests and examples MUST NOT contain absolute paths, and `run.json` MUST NOT either: it records
  each input path relative to `$ZEEMAP_DATA` (written `$ZEEMAP_DATA/...`) or to the configuration's directory. (G5)
  Checked by: V7, V8.
- **ENG-5** The repository MUST NOT contain library files, observations, outputs of real objects, or code copied or
  translated from other fitting programs, Numerical Recipes or MAGWD. (licence) Checked by: V8, and review.
- **ENG-6** Peak memory MUST stay below 1 GB per worker at the default settings. (G1) Checked by: V12.
- **ENG-7** Every output file MUST be written atomically: to a temporary file in the same directory, then renamed, so
  that an interrupted run never leaves a half-written file under a final name. (G5) Checked by: V7.
- **ENG-8** `fit` MUST write each stage's result to `<out>/checkpoint/` as the stage ends (the search also after
  round 0, the kernel stage and each restart round; each mode's MCMC and each systematics refit on its own), and
  `fit --resume` (same directory) or `fit --from <dir>` MUST continue from them, running only the stages whose inputs
  are not checkpointed, with the same outputs as an uninterrupted run. A stage's checkpoint MUST be keyed by what the
  stage reads, so a change of a later stage's settings (the MCMC budget) does not rerun the others. A resume MUST
  refuse, with a one-line message and exit status 2, a run whose library, data, model, parameters, objective or
  search settings, library or data file contents, zeemap version, seed or number of starts differ from the
  checkpointed search's. (G1) Checked by: V7.

## 7. Constraints

- **Clean room.** No code is copied or translated from other fitting programs, Numerical Recipes or MAGWD, and
  contributors do not read their source while writing zeemap code.
- **Configuration first.** The configuration reader is strict (CFG-2) from the first task that reads a configuration.
- **The research prototype is not used.** zeemap is rewritten from this plan alone (decision #14).
- **Real data stay outside the repository.** The library, the observations and every output for a real object stay
  outside git, under `$ZEEMAP_DATA` or a run directory. Tests that need them skip with a message when `$ZEEMAP_DATA`
  is not set. When gate B asks for tier 2, a skipped tier-2 check counts as a failure (GATES.md).

## 8. Open questions

Each is explained in [DECISIONS.md](DECISIONS.md), section "Open questions". Every `[NEEDS DECISION]` marker, in this file,
VERIFICATION.md, FORMATS.md or DESIGN.md, points here.

| marker | question | where it blocks |
|---|---|---|
| D7 | Teff free by default; the Teff grid must extend beyond the expected range | LIB-3 default |
| D8 | user priors for the MCMC | PAR-4 |
| D12 | the unit of uncertainty: fingerprint modes or families (decision #29) | nothing in the code; how published results state their uncertainty |
