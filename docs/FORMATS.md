# File formats

**Status:** describes the current version of zeemap; the history of its decisions is in [DECISIONS.md](DECISIONS.md).
**Who this is for:** anyone preparing a library or observations for zeemap, or reading its outputs.
**What you will know afterwards:** the exact layout of every file zeemap reads, and of its main outputs.

## 1. Library (input, outside the repository)

A library is a directory with one text file `info.dat` and one binary file per temperature.

**`info.dat`** is a list of tokens separated by white space (spaces or new lines), in this order:

| position | token | meaning |
|---|---|---|
| 1 | base directory | ignored by zeemap |
| 2 | library name | text, used in reports |
| 3 | wavelength file | a file name, or `-`; if the file exists, it lists the wavelengths, else they follow from tokens 5-7 |
| 4 | a number (0.3) | unused |
| 5 | NL | number of wavelength points |
| 6, 7 | lambda0, dlambda | first wavelength and step, in A (1 A in all libraries so far) |
| next | NT, then NT values | number of temperatures, then the temperatures in K |
| next | NMU, then NMU values | number of viewing angles mu, then their values, increasing; NMU = 1 with mu = 1 is the usual case; NMU > 1 is a mu-grid library (LIB-8) |
| next | NB, then NB values | number of field strengths, then B in MG, increasing |
| next | NP, then NP values | number of field angles, then psi in degrees, increasing, from 0 to exactly 90 |

**`<Teff>.bin`**, one per temperature (for example `30000.bin`; some older libraries use the index, `0.bin`, `1.bin`, ...;
the reader tries the temperature first). Raw little-endian float32 values (numpy dtype `'<f4'`), no header. The order is: for each B (outer
loop), for each psi (inner loop), NL values of I followed by NL values of V. So the file holds NB x NP x 2 x NL values,
and the reader MUST check the file size against that. **A mu-grid library** (NMU > 1) adds mu as the outermost loop:
for each mu, then each B, then each psi, I then V; NMU x NB x NP x 2 x NL values. (No mu-grid library exists yet; this
order is zeemap's proposal, decision #27; a library writer for zeemap follows it.)

Only psi from 0 to 90 degrees is stored. Angles above 90 degrees follow from I(180 - psi) = I(psi) and
V(180 - psi) = -V(psi). At psi = 90 degrees zeemap ignores the stored V record and uses V = 0.

## 2. Observations (input, outside the repository)

Per object and data set, a directory with, for each phase:

| file | columns (white-space separated) |
|---|---|
| `phNN.dat` | wavelength in A, I (any flux unit), V/I (a fraction, not per cent) |
| `phNN_ierr.dat` | wavelength in A, error of I (same unit as I) |
| `phNN_vierr.dat` | wavelength in A, error of V/I |

All files of a data set share one regular wavelength grid. The configuration states that grid (start, step, number
of points), and the reader MUST check that the files agree with it to 1e-3 of a step. An optional `dataset.json` holds
notes about the data (origin, phases, known problems); zeemap copies it into `run.json`.

## 3. Configuration (input)

One TOML file. `src/zeemap/defaults.toml` holds every key with its default and a one-line comment (task T1.0); it is
the only place where defaults are written. The reader stops on an unknown key (CFG-2). Keys needed from the first
milestone:

| key | type and unit | default | meaning |
|---|---|---|---|
| `library.path` | text | (none) | library directory, relative to `$ZEEMAP_DATA` |
| `library.teffs` | list of K | all | Teff slices to load |
| `library.outside` | `"clamp"` or `"reject"` | `"clamp"` | what a patch with B outside the library gets (LIB-5) |
| `data.path` | text | (none) | data directory, relative to `$ZEEMAP_DATA` |
| `data.grid` | `{start, step, npix}` in A | (none) | the regular data grid |
| `data.windows` | list of `[lo, hi]` or `[lo, hi, weight]` in A | (none) | fit windows |
| `data.exclude` | list of `[lo, hi]` or `[lo, hi, "I"/"VI"]` | `[]` | excluded ranges |
| `data.errors` | `{absolute, floor_I, floor_VI, derive_VI}` | `{true, 1e-6, 1e-5, false}` | error rules (DATA-2) |
| `data.vi_scale` | number > 0 | 1.0 | factor applied to the observed V/I and its error when the data are read (used by the default V/I swaps) |
| `[[data.phase]]` | `phase`, `file`, `ierr`, `vierr`, `use_I`, `use_VI`, `i_offset` | `use_I = use_VI = true`, `i_offset = true` | one block per phase; `i_offset = true` adds a free constant to the I continuum, for data whose zero point of I is not known |
| `[[data.weight]]` | `{stokes, lo, hi, phases, w}` | none | extra weights (DATA-4): the errors of the selected pixels are divided by sqrt(w), in every likelihood; w = 0 removes them |
| `model.field` | `"dqo"`, `"dqo_aligned"`, `"harm4"` | `"dqo"` | field model (DESIGN.md section 2) |
| `model.shift_unit` | `"kms"` or `"A"` | `"kms"` | unit of the `shift` parameter (FWD-13): a velocity, or a constant in A |
| `model.shift_nodes` | integer >= 2 | 5 | a free shift: records at this many shift values over the parameter's [lo, hi], cubic Lagrange weights between them (DESIGN.md section 5b) |
| `model.interp_teff` | `"planck"` or `"log"` | `"planck"` | Teff interpolation between slices (LIB-3, DESIGN.md section 5c) |
| `model.interp_b` | `"linear"` or `"cubic"` (`"pchip"` accepted as an alias of `"cubic"`) | `"linear"` | interpolation in B (FWD-15): `"cubic"` = C1 cubic Hermite in B with three-point slopes, no monotone limiter |
| `model.surface` | `{scheme = "fibonacci", n}` | fibonacci, 5000 | patch grid for the search; error bars use `uncertainty.patches` |
| `model.ld_mode` | 0, 1 or 3 | 3 | limb-darkening law |
| `model.fwhm` | A | (none) | instrumental FWHM |
| `model.phase_offset` | rotations | 0 | added to all phases |
| `objective.kind` | `"full"` or `"chi2"` | `"full"` | likelihood |
| `objective.scale` | true / false | true | unknown noise scale per spectrum |
| `objective.corr` | `{rho, jitter}` in A, or `false` | `{rho = 4.0, jitter = 0.05}` | short correlated noise: the initial kernel before the kernel fit (DESIGN.md section 5) |
| `objective.disc` | `{a_I, rho_I, a_VI, rho_VI}` (rho in A), or `false` | `{0.71, 85.0, 0.42, 205.0}` (research values for RE J0317-853) | the long (discrepancy) term of the initial kernel |
| `objective.kernel_fit` | true / false | true | fit the kernel per spectrum and freeze it (LIKE-2, decision #22); false keeps the initial kernel, or `objective.kernel` |
| `[[objective.kernel]]` | `{spectrum, rho_short, a, rho_long, j}` (spectrum label as in `run.json`, e.g. `"0.250_I"`; lengths in A; j the white part) | none | a kernel given per spectrum (for example copied from an earlier `run.json`); used when `kernel_fit = false`. A label that matches no spectrum of the configuration (the phase as %.3f, then `_I` or `_VI`, for the kinds the phase uses) stops the reader; a systematics swap that removes spectra (the jackknife, `use_VI = false`) drops their blocks |
| `objective.continuum` | `{degree}`, or `false` | `{degree = 3}` | Legendre series P_0 ... P_degree times the model I, normalised by the model's norm (LIKE-3) |
| `[parameters]` | one entry per parameter (below) | | |
| `[search]` | `n_starts` (the key is not `starts`, which names the `[search.starts]` table), `top`, `restarts`, `rounds_max`, `round_gain` (nats), `band` (nats), `lbfgs` (bool: run the L-BFGS-B stage), `lbfgs_maxiter`, `fd_step` (central-difference half-step, in units of each parameter's step), `lm_maxiter`, `lm_stop_nat` (nats: the LM finish stops when the undamped Gauss-Newton step at a fresh Jacobian predicts less; 0 = off), `lm_max_rejections` (the LM finish stops after this many rejected steps in a row; 0 = off), `tol_polishes` (end points per mode re-polished at the error-bar patch count) | 250 (research value; T3.5 sets the default), 10, 12, 3, 5.0, 45.0, true, 150, 1e-4 (was 0.5, decision #25), 100 (provisional, task T7.7; was 20), 1e-3 and 5 (decision #37, provisional; 0 = off), 5 | DESIGN.md section 3 |
| `[search.family]` | `min_level` (nats, > 0), `max_pairs`, `mid_iter`, `interior` (points per straight segment) (integers >= 1) | 5.0, 3000, 50, 15 | families (SEARCH-9, DESIGN.md section 3b) |
| `[search.starts]` | `md` [lo, hi] MG, `bq` max MG, `mo` max MG, `c` radius, `u` [lo, hi], `outside_max`, `rules` (list of extra acceptance rules) | [200, 600], 250, 200, 0.25, [0.03, 1], 0.05, [] | start distribution and prior (DESIGN.md sections 3, 4) |
| `[search.fingerprint]` | `incl` (deg), `angle` ("auto", "incl" or "beta"), `c` (radii), `b50`, `b90`, `b50_rms` (MG), `ref_phase` (index), `relative` (bool), `b50_rel`, `b90_rel`, `b50_rms_rel` (fractions), `extra` (table name -> threshold) | 8, "auto", 0.06, 0.5, 1.0, 1.0, 0, true, 0.04, 0.10, 0.05, {teff = 1000.0} | thresholds that group end points into modes (DESIGN.md section 3, step 3; decision #16). `relative = true`: the B50, B90 and B50-rms thresholds are max(floor, fraction x the larger B50 at the reference phase of the two fingerprints), the floors being `b50`, `b90`, `b50_rms`; `relative = false`: those three are absolute thresholds. `angle = "auto"`: "beta" for one phase, "incl" otherwise. "beta" compares beta, the angle between md and the observer direction at the reference phase (0-180 deg), with the threshold `incl`, and compares the centre in the frame of md and the observer instead of the star's frame. `extra`: non-field parameters compared as \|difference\| / threshold when they are free; a configuration's table replaces the default (`extra = {}` removes Teff) |
| `[search.perturb]` | `b_frac`, `angle` (deg), `c` (radii), `incl` (deg) | 0.05, 8, 0.04, 4 | sizes of restart perturbations |
| `[uncertainty]` | `patches`, `ensembles`, `tau_factor`, `max_evals`, `modes_max` (MCMC for the best N modes only; 0 = all), `calibration` (path of a passed V15 result: its `run.json`, with the configuration and the SHA-256 of every input, and the pass summary of V15a and V15b), `profile_quantities` (`"default"`, or a list of names of section 7, a B quantity with its phase index, `"B50_MG[0]"`; `[]` = no profiles), `profile_steps` (table name -> grid step > 0, the name with or without a phase index; a step in percent is written as a string `"<p>%"`, p a positive decimal number such as `"1%"` or `"2.5%"`, and is p % of max(\|q_best\|, s_q), s_q the quantity's own scale: the largest change of q over one `step` of any free parameter from the family's best node, so that a best value near 0 does not give a vanishing step; DESIGN.md section 4b), `profile_max_steps` (>= 1), `profile_families` (profile the best N families; 0 = all). The reader checks the names (against the number of phases), the steps and the ranges, so a bad setting stops it before the search | 20 000, 4, 50, 600 000 (decision #20; 0 = the same), 0, none, `"default"` = `["incl_deg", "dipole_MG", "c_along_axis", "B10_MG", "B50_MG", "B90_MG", "teff_K"]` (teff_K only when Teff is free; with one phase `beta_deg` instead of `incl_deg`; the B quantities at the reference phase `search.fingerprint.ref_phase` only, decision #40; a quantity that no free parameter moves, e.g. c_along_axis with c fixed, is left out, DESIGN.md section 4b), `{incl_deg = 2.5, beta_deg = 2.5, dipole_MG = "1%", B10_MG = "1%", B50_MG = "1%", B90_MG = "1%", c_along_axis = 0.01, teff_K = 100.0}`, 40, 1 (decision #40) | (`temper_T` was removed with the tempered widths, decision #24: the reader stops on it as an unknown key) |
| `systematics.defaults` | true / false | true | generate the default swaps of SPEC SYS-1 at run time (decisions #21, #23): `teff_<T>` (Teff fixed at each selected library slice T inside the teff range), `vi_scale_0.9` and `vi_scale_1.1` (`data.vi_scale` times 0.9, 1.1), `library_<k>` for each entry of `library_alternatives`, `continuum_deg0` (`objective.continuum.degree = 0`) and `continuum_off` (`objective.continuum = false`: no Legendre continuum and no free constant, whatever `i_offset` says; the scale ratio of each I spectrum is written next to it); a configured swap of the same name replaces a default one |
| `systematics.library_alternatives` | list of library directories (as `library.path`) | [] | other library versions for the default swaps |
| `[[systematics.swap]]` | `{name, set = {key = value, ...}}` | none | one swap per block, run in addition to the defaults |

**Parameters.** Each entry is `name = {value, step, free, lo, hi}`. The names (DESIGN.md section 2): `md`, `mo`, `aq`
(3 numbers each), `bq`, `c` (3 numbers), `u`, `teff`, `shift` (global, FWD-13; a fixed value is built into the forward
model, a free one interpolated between shift nodes; a shift per phase is not built). A vector entry has one `step` and
one length range. Field model "harm4": `md`, `u` and the scalar Gauss coefficients `g20`, `g21`, `h21`, ..., `g44`,
`h44` (MG; a coefficient not given is fixed at 0; a free one starts uniform in [lo, hi]).

Paths: `library.path` and `data.path` are relative to `$ZEEMAP_DATA`; the files of `[[data.phase]]` are relative to
`data.path`; every other relative path is relative to the directory of the configuration file. Absolute paths are allowed in a user's own configuration, never in the
repository.

Sections 4 and 5 were removed on 2026-11-08; sections 6 and 7 keep their numbers.

## 6. Outputs (one directory per run)

Two runs with the same configuration, inputs, seed and `--workers` give the same outputs, apart from wall times.
A `fit` into a directory that holds an earlier fit first removes that fit's files (the names of the table below, and
`mode<k>_*` in `spectra/`, `histograms/` and `samples/`), so no output of the earlier run survives; other files
(`eval.json`, `eval_*` spectra, the user's own) are kept.
Wall times are written only to `run.json`, to the log, and under a top-level `_meta` key of `diagnostics.json`,
`intervals.json` and `systematics.json`. `report.txt` carries none. A comparison that skips `_meta` and `run.json`
therefore finds such a rerun equal at its tolerance.

| file | content and format |
|---|---|
| `eval.json` (from `eval`) | -ln L, `scale_ratio` (median \|model I\| / median \|data I\| per I spectrum, printed by `eval`, with a warning outside [1e-3, 1e3]); the model spectra go to `spectra/` |
| `run.json` | the configuration as used (after defaults), software versions (Python, numpy, numba, llvmlite, zeemap commit), CPU model, master seed, SHA-256 of every input file, start and end time, the initial and the frozen kernel of every spectrum (`kernel`: per spectrum label `rho_short`, `a`, `rho_long`, `j`, `share_short`, the marginal -ln L gained by the fit, and `at_bound`: {parameter: "lower" | "upper"} for each parameter
that ended within 1e-3 of a bound of its range, except an empty range and rho_long when a is at its lower bound),
the continuum setting, the parameters as used (`names`, `free`, `lo`, `hi`, `step`, `value` after the clamps made at load, `length_range`) and `limit_notes` (one sentence per parameter whose value or limits were clamped at load, at present a Teff value or limit up to 0.5 K beyond an edge slice, clamped to that slice, DESIGN.md section 3; [] when none). Input paths are written relative: `$ZEEMAP_DATA/<path>` or relative to the configuration's directory, never absolute (ENG-4) |
| `modes.json` | list of modes; per mode: `id`, canonical parameters, derived quantities (section 7), `neg_log_l` (-ln L at the written canonical parameters), `neg_log_l_mode` (the mode's value: the better of the end point and its equator-mirror image; it ranks the modes), `image` (which of the two), `mirror_asymmetry` (f(mirror) - f(end point)), `delta` (of `neg_log_l_mode`, to the best), `n_starts` (starts that reached it), `patches` used for the final ranking, `family` (the family of its best node, or null when all its nodes lie above the family level), `families_all` (every family its nodes fall into), `band_component` (the id of its component at the band level) |
| `families.json` | the family level and h (nats), the band level; per family: `id`, `modes` (ids), `nodes` (end points: mode id, -ln L, canonical parameters), `n_starts`, `best` (node), `links` (pairs joined, each with `path` = "straight" or "midpoint" and the highest -ln L on it), `capped` (true when `search.family.max_pairs` stopped the grouping); the band components in the same layout; `recomputed_after_handbacks` (mode ids, only when an MCMC hand-back moved a mode by more than the value tolerance and the families were recomputed after the MCMC with its new point as a node, DESIGN.md section 3b) |
| `profiles.json` | top level: `level` (the family level, nats; as in families.json), `h` (nats, DESIGN.md section 3b), `header` (the likelihood line of the tables, `likelihood_header`), `families` keyed by family id (a string); per family: `best_mode` (the id of the mode at the family's best node), `converged` (that mode's MCMC verdict, `converged` of intervals.json; null when intervals.json has no entry for it, e.g. without an MCMC run), `quantities`; per family and quantity: `grid_step`, `grid` and `envelope_profile` (the lower envelope at the grid values where a point met the constraint), `ends` (the closed ends of the range beyond the grid, `lo` and `hi`, each null or `q`, `neg_log_l`, `reason` = "range end" or "constraint not met"), `walks`: each with `start` (`role` = "best", "farthest node" or "walk back"; `node` and `mode`, the family node's ids, null for a walk back; `k`, the grid index it starts at), `points` (per grid index `k`: the grid value `q0`, the value reached `q`, `neg_log_l`, `met` (false: q missed q0 by more than 0.1 dq, so the point is not a value of the profile), `stalled` (true: it missed with no limit in sight, DESIGN.md section 4b, "A missed constraint")), its own `ends`, `nevals` and `retries` (points polished a second time after a miss); the 0.5-nat, 2-nat and family-level intervals (`lo`, `hi`, `open_lo`, `open_hi`), the envelope (UNC-8), `mode_68` (`p16`, `p84`, `median` of the family's best mode, intervals.json; null without one), the ratios `R1` and `R2`, `flags` ("envelope wider than one Gaussian basin gives" (R2 above 1.5); "constraint not met" exactly when `ends` holds an end of that reason; "polish stalled" when a point has `stalled` true; "B quantile is a step function: provisional" on every B10/B50/B90 entry, decision #39), `note` (why `R1` and `R2` are null: "no per-mode interval (no MCMC run, not converged, or a nuisance)"; else null), `minima` (pairs `q`, -ln L), `nevals` (DESIGN.md section 4b) |
| `intervals.json` | per mode and quantity (labelled "per mode: conditional on this basin"): median, 16/84 % and 2.5/97.5 % points, jackknife sigma, R-hat and ESS, `kl_prior` and `kl_prior_corrected` (REP-4), `failed_rules` (the quantity's own failed V14 rules), flags ("not converged", "orientation not determined by one phase", "nuisance, not required", "touches a range limit", "not constrained by the data", "u reached 0"); numbers are null for a quantity that fails V14 on its own or is a declared nuisance (decisions #17, #18). Per mode: `converged` (the mode-level verdict), `convergence` (V14 numbers, `quantity_failed`, `tau_quantity`, `nuisance`, `chart`, `sampling_coordinates`) |
| `systematics.json` | per mode, quantity and swap: shift of the best value, or `"mode change"` when the refit left the mode; per swap its `kind` ("config", "default" or "jackknife") and keys |
| `diagnostics.json` | search statistics (starts, distinct modes and families, hits on the best mode and the best family, fraction reached only once, optimiser scatter, value tolerance, mirror asymmetry: `mirror_asymmetry_max` over the band modes, per mode, `modes_at_mirror_image`, and `mirror_uncertainty_patches` after the re-evaluation on `uncertainty.patches`), the whiteness check per spectrum (`whiteness`: max \|ACF\| at lags 1-10, the threshold, pass), `kernel_at_bound` (per spectrum label whose frozen kernel has a at its upper bound, that kernel's `at_bound`),
floor chi2 per phase (DIAG-1: the floor and the best mode's plain chi2 for the model I of `spectra/`), the per-window residual table (`window_residuals`, DIAG-3: per family's best mode, library, spectrum and window: `family` and `mode` (the ids of families.json; null family without families), pixels, whitened chi2 under the run's continuum setting and at degree 0), per-phase residual summaries, number of invalid models, for mu-grid libraries the area fraction below the smallest mu; every polish's end point (`endpoints`: parameters, -ln L before and after the LM finish, `converged`, `lm_stop` = why the LM finish stopped, "" if it did not run: "gain", "predicted", "no_step" or "held" (a converged stop), "lambda", "rejections", "maxiter" or "invalid" (not converged); `converged` follows `lm_stop` when the LM finish gained more than `search.lm_stop_nat` over L-BFGS-B's end point (with that stop 0: when it accepted at least one step), or with `search.lbfgs = false`, else it is L-BFGS-B's flag, so "rejections" can come with true and "predicted" with false; DESIGN.md section 3, "Polish"); `profiles_error` ("<exception>: <message>") when the profile stage failed, in which case no `profiles.json` is written, every other file is, and `report.txt` says so; `floor_chi2_error` likewise when DIAG-1 failed |
| `spectra/mode<k>_ph<phase>.dat` | wavelength, model I, model V/I, data I, data V/I, residuals in units of the errors |
| `histograms/mode<k>_ph<phase>.dat` | visible area per (B, psi) cell |
| `report.txt` | the same in readable form. Every table header states the likelihood, the kernel ("fitted, frozen" or "given") and the continuum setting. For each family and profiled quantity, two sentences from templates, one per view: "Mode <k>: <q> = <median> +<hi> -<lo> (68 %, per mode, <converged / not converged>)." and "Family <F>: <q> between <lo> and <hi> within <h> nats of the best (profile, <n> walks); minima at <values>, <d> nats apart." |

When the whiteness check (LIKE-10) fails for any spectrum, every file above that holds -ln L differences or intervals
(`modes.json`, `families.json`, `profiles.json`, `systematics.json`, `intervals.json`, whichever the run writes, and
`diagnostics.json`) has two more top-level keys: `noise_model_flag` = "noise model fails the whiteness check" and
`noise_model_failing_spectra` (the labels of the failing spectra, sorted); `intervals.json` then has `calibration` =
"uncalibrated" whatever coverage run is named (UNC-4), and each table of `report.txt` carries the line
"[noise model fails the whiteness check: <spectra>]". A run whose check passes, or that runs no check (kind "chi2"),
writes neither key.

`checkpoint/` (from `fit`, ENG-8) holds one Python pickle per finished stage, read by `fit --resume` and
`fit --from`: `search.pkl` (the search's state after its last finished round, replaced each round, with a hash per
guarded configuration section and its files, and of the code version), `table-<key>.pkl`, `mcmc-<key>.pkl` (one per
mode), `profiles-<key>.pkl` (the walks, without the MCMC intervals) and `sys-<key>.pkl` (one per mode and swap);
`<key>` hashes what the stage reads (cli.py). A resumed run's `run.json` has `resumed`: `from` (the directory read,
relative to the output directory), `search` (true when the search was resumed; then `ms_per_eval_per_worker` and
`n_invalid_main_process` in `diagnostics.json` are null, since they would cover only the resumed part) and
`checkpoints` (the names of the files taken). They are internal to zeemap
(not a stable format; the pickles hold zeemap's own objects), are not removed by a later `fit` into the same
directory, and are read only from runs one made oneself (unpickling runs code).

Outputs of real objects are derived from the user's library and data. They live outside the repository, never in
git.

## 7. Derived quantities

**Areas.** "Visible area" is the sum of patch areas with mu > 0. "Brightness-weighted area" weights each visible patch
with the same weight it has in the disc sum: area x mu x LD(mu). LIB-6, B10/B50/B90 and the fingerprints use the
brightness-weighted area.


Beta (`beta_deg`) is the angle between the dipole axis and the line of sight at the first phase, 0-180 degrees (per
phase: `angle_dipole_observer_deg`). The offset along the dipole axis is `c_along_axis` = c . m_d / |m_d| (signed,
stellar radii); the offset across it, `c_across_axis`, is the length of the rest of c.

Reported for every mode: inclination in degrees (0-90 after the mirror symmetry), |dipole| in MG, centre offset |c| in
stellar radii, beta, the offsets along and across the dipole axis, and per phase B10, B50 and B90 (the field strength below which 10, 50 and 90 % of the
brightness-weighted area lies), plus the fraction of the brightness-weighted area with B outside the library range.

Names in `modes.json`, `profiles.json` and `uncertainty.profile_quantities`: `incl_deg`, `dipole_MG`, `c_abs`,
`beta_deg`, `c_along_axis`, `c_across_axis`, `B10_MG`, `B50_MG`, `B90_MG` (lists over the phases; a profile of a B
quantity is per phase, named `B50_MG[k]` for phase index k), `teff_K`.
