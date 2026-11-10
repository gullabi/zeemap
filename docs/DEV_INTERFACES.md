# Module interfaces (builder's contract)

**Status:** the interfaces between the main modules, written on 2026-10-18 so that the parts could be built in
parallel, and kept as a reference for contributors. The code is the final word: where they differ, the code holds and
this file is corrected. DESIGN.md section 1 says what each part does; this file gives the exact names, shapes and
units. A change to one of these interfaces updates this file in the same commit.

Conventions: numpy float64 everywhere except library storage (float32). Shapes use K = batch size, N = patches,
P = phases, L = library wavelengths, M = data pixels, NB, NP = library B and psi counts, NC = 2 NP - 1 cos-psi nodes.
Angles in the public interfaces are in degrees; internally radians. Field strengths in MG.

## Parameter vector (field + geometry), `field.py`

The search vector `theta` for the `dqo` model, in this order (DESIGN.md section 2):

| index | name | meaning |
|---|---|---|
| 0:3 | `md` | dipole vector, MG (direction = axis, length = polar strength) |
| 3 | `bq` | quadrupole polar strength, MG, signed |
| 4:7 | `aq` | quadrupole axis (direction only; length gauge, kept near 1) |
| 7:10 | `mo` | octupole vector, MG |
| 10:13 | `c` | field centre, stellar radii, star frame |
| 13 | `u` | cos i |

Extra parameters (`teff`, `shift`) are appended after index 13 by `problem.py`, never inside `field.py`.

```python
class FieldModel:                      # registry: field.MODELS[name] -> class; register(name) decorator for toy models
    name: str; names: list[str]        # parameter names, len == dim
    dim: int
    def evaluate(self, points, theta) -> np.ndarray      # points (N,3) star frame -> B (N,3) in MG
    def evaluate_batch(self, points, Theta) -> np.ndarray  # (K,dim) -> (K,N,3)   (default: loop)
    def canonical(self, theta) -> np.ndarray             # DESIGN.md section 2 canonical form
    def valid(self, theta) -> bool                       # |c| < 1 etc.
```
`dqo_aligned`: dim 14 as well (aq and mo direction tied to md); it exposes the same vector but `evaluate` ignores the
free directions. Simpler: `dqo_aligned` has names `md(3), bq, bo, c(3), u` (dim 9); `problem.py` only ever uses
`names`/`dim`. Both are acceptable; the builder of field.py decides and documents.

Helper functions in `field.py`: `axis_from_angles(theta_deg, phi_deg)`, `rotation(theta_deg, phi_deg)` (R_z R_y),
`component_field(points, axis, strength, l)` for l = 1, 2, 3 (and general l up to 5 later), potential `component_potential`.

## Surface, `surface.py`

```python
@dataclass
class Surface:
    scheme: str                 # "fibonacci"
    centres: np.ndarray         # (N,3) unit vectors
    areas: np.ndarray           # (N,)  sum 4 pi
def fibonacci(n) -> Surface
def observer(u, phase) -> np.ndarray      # (3,) n = (sin i cos 2pi phase, -sin i sin 2pi phase, -cos i), u = cos i
def limb_darkening(mu, mode) -> np.ndarray  # mode 0: 1; 1: 0.4+0.6mu; 3: 0.7+0.3mu
```

## Library, `library.py` (reader exists; this adds the slice cache and Teff interpolation)

```python
class Library:
    def __init__(self, libdir, teffs=None)      # teffs: subset to use (K); None = all; loads lazily, once per process
    info: LibraryInfo
    reads: int                                   # number of slice files read (V2f)
    def slice(self, teff) -> np.ndarray          # (NB,NP,2,L) float32, cached
    def at_teff(self, teff) -> np.ndarray         # (NB,NP,2,L) float64, linear in ln Teff; error if outside
    def check_teff(self, teff) / check_wavelengths(lo, hi, margin)  -> raise LibraryError naming the covered range (LIB-7)
    b_min, b_max: float
```

## Data, `data.py`, shared types in `types.py`

```python
@dataclass
class Grid: start: float; step: float; npix: int; wavelengths (property)
@dataclass
class Spectrum:        # one phase and one Stokes quantity, switched on
    phase_index: int; phase: float; kind: str   # "I" | "VI"
    pixels: np.ndarray   # indices into the data grid that are fitted (in windows, not excluded), increasing
    data: np.ndarray; sigma: np.ndarray; weight: np.ndarray   # same length as pixels; weight 1 unless [[data.weight]]
    window: np.ndarray   # window index per pixel
    i_offset: bool       # I only: free constant in the continuum
@dataclass
class DataSet:
    grid: Grid; phases: np.ndarray (P,); spectra: list[Spectrum]
    I: np.ndarray (P,M); VI: np.ndarray (P,M); sig_I; sig_VI        # full arrays on the grid (NaN where missing)
def load_data(cfg) -> DataSet     # cfg: the dict from config.load_config; paths already resolved
```

## Forward model, `forward.py`

```python
class Forward:
    def __init__(self, library: Library, surface: Surface, field: FieldModel, dataset: DataSet, cfg)
    # precomputes once: library records convolved (FWHM) and resampled to the data grid, for each Teff slice in use:
    # rec[(ib, ic)] -> (2, M); cos-psi nodes NC = 2 NP - 1 from +1 to -1 with V mirrored (V = 0 at psi = 90)
    def spectra(self, Theta, teff=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]
        # Theta (K, dim) -> I (K, P, M), VI (K, P, M), invalid (K,) bool. Batched: node weights W (K*P, NB*NC) then W @ records.
    def node_weights(self, theta) -> np.ndarray   # (P, NB, NC) brightness-weighted area per node (for histograms, LIB-6, V2b)
    def outside_fraction(self, theta) -> np.ndarray  # (P,) brightness-weighted area fraction with B outside the library
    def b_quantiles(self, theta, q=(0.1,0.5,0.9)) -> np.ndarray  # (P, 3) B10, B50, B90
```
Options read from cfg: `model.interp_b` ("linear" | "pchip"), `library.outside` ("clamp" | "reject"), `model.ld_mode`,
`model.fwhm`, `model.phase_offset`. The plain-numpy reference for V5a is `forward_reference.py` with the same `spectra` signature and no numba.

## Likelihood, `likelihood.py`

```python
class Objective:
    def __init__(self, dataset: DataSet, cfg)      # objective.kind: "full" | "chi2"; nuisance switches
    def neg_log_l(self, I, VI, invalid) -> np.ndarray   # (K,P,M),(K,P,M),(K,) -> (K,) ; invalid rows get 1e30
    def residuals(self, I, VI) -> list[np.ndarray]      # whitened residuals per spectrum after the best continuum (for LM, 1 row)
    def per_spectrum(self, I, VI) -> dict               # chi2 per spectrum and window (one parameter set)
```

## Problem and search, `problem.py`, `search.py`

```python
class Problem:                              # implements search.CostProblem; module docstring has the details
    def __init__(self, cfg, dataset=None)   # builds library, surface, field, dataset (load_data, or the given one,
                                            # e.g. blank_dataset(cfg) for inject), forward, objective
    names: list[str]; dim: int; free: np.ndarray (bool); lo, hi, step, value: np.ndarray
    field_slice: slice                      # slice(0, field.dim)
    length_range: dict[str, (lo, hi)]       # configured length ranges of the vector entries (md, aq, mo, c)
    iteff: int | None                       # index of teff (after the field numbers), None if not a parameter
    def cost(self, X) -> np.ndarray         # (K,dim) -> (K,) -lnL + prior-cut penalty + aq gauge penalty; 1e30 invalid
    def residuals(self, x) -> np.ndarray    # whitened residuals, all spectra concatenated, noise scales held fixed
    def fingerprint_quantities(self, x)     # {incl (deg, from |u|), c (3,), b50 (P,), b90 (P,)}; optional beta (P,),
                                            # c_frame (P, 3), extra {name: value} (decision #16)
    def canonical(self, x)                  # field.canonical on the field slice
    def outside_fraction(self, x) -> float  # whole-surface area fraction with |B| outside the library (prior cut)
    def accept_start(self, x) -> bool       # starts inside the configured length ranges and u's lo/hi
    def thetas(self, X) / theta(self, x)    # parameter rows -> field vectors
    def spectra(self, X)                    # forward.spectra with teff from the vector
    def model_spectra(self, x) -> (I (P,M), VI (P,M))
    def continuum_models(self, I, VI) -> (P,M)   # model I times the best continuum, as inside the likelihood
    def derived(self, x) -> dict            # FORMATS.md section 7 quantities (+ angle dipole-observer per phase)
    def grouped(self, x) -> dict            # {configuration name: value}, vectors as 3 numbers
    def evaluation(self, x) -> dict         # what eval.json holds (-lnL, penalty, per_spectrum, derived)
    def derived_batch(self, X) -> dict      # for uncertainty.py: incl_deg, dipole_MG, c_abs (K,); B10/B50/B90_MG,
                                            # angle_dipole_observer_deg (K, P); same numbers as `derived`
    aq_penalty_in_cost: bool                # cost holds the ln|aq| gauge penalty (dqo, free aq), so the prior omits it
def blank_dataset(cfg) -> DataSet           # grid, phases and windows of the configuration, I = 1, V/I = 0, errors 1
```
Vector entries of the configuration (md, aq, mo, c) give their 3 components one step and the bounds -hi .. +hi (the
search's safety limits); u keeps its lo/hi; teff its lo/hi, which must lie inside the selected library temperatures.
A field group the configuration omits is fixed at zero (aq: the unit z vector). A Problem pickles as its configuration
(rebuilt once per worker process, or found in the process registry that forked workers inherit).

`search.py` (the number of starts is `search.n_starts`; `[search.starts]` is the start distribution):
`draw_starts(problem, cfg, rng, n)`, `polish(problem, x0, cfg) -> PolishResult`, `fingerprint`, `group_modes`,
`run_search(problem, cfg, seed, n_starts=None, workers=1, log=None) -> SearchResult`.

`inject.py`: `inject(cfg, outdir, seed=1, noise="none"|"white"|"full", sigma_i=0.01, sigma_vi=0.003) -> truth dict`.
`report.py`: `write_eval(outdir, problem, cfg)`, `write_fit(outdir, problem, cfg, result, ..., uncertainty=None,
search_wall=None)` (`uncertainty` = {"intervals": intervals.json dict, "results": {mode id: McmcResult}, "wall": s}
adds intervals.json, samples/mode<k>.npz and the error-bar part of report.txt), `write_spectra`, `write_histograms`
(file layouts in its module docstring and FORMATS.md section 6). `cli.py`: `main(argv)` with the commands `inject`,
`eval`, `fit`; `fit` runs `uncertainty.uncertainty_for_modes` per mode after the search
(only when asked, decision #42: `--mcmc`, or a budget `--mcmc-modes N` / `--mcmc-evals N`; `--no-mcmc` wins; `--mcmc-modes N` or `uncertainty.modes_max`, `--mcmc-evals N` for `uncertainty.max_evals`; mode
k gets SeedSequence([seed, k]); with `--workers > 1` one process per mode; the MCMC Problem uses
`uncertainty.patches` Fibonacci patches when that differs from the search's surface; details in cli.py's docstring).

`systematics.py` (task T5.1; SYS-1 to SYS-5, UNC-7; details in its module docstring):
```python
def apply_swap(cfg, swap) -> dict              # deep copy + swap["set"] (nested tables or dotted keys), checked by
                                               # config.from_dict (ConfigError names the swap); a jackknife swap
                                               # ({"drop_phase": i}) removes [[data.phase]] block i instead
def jackknife_swaps(cfg) -> (list[dict], note) # "drop_phase_<phase>" per phase; ([], note) with one phase
def default_swaps(cfg, problem=None) -> (list[dict], notes)  # SYS-1 defaults (decision #21): teff_<T> per library
                                               # slice, vi_scale_0.9 / 1.1, library_<k> per library_alternatives
def map_point(src, x, dst, fix=True)           # parameter vector of one Problem -> another's (by name);
                                               # fix: dst's fixed values set
def quantities(problem, X) -> {name: (K,)}     # free parameters (canonical) + derived_batch, intervals.json's keys
def mode_starts(problem, x_mode, cfg, seed, k) # the mode + 3 perturbations from SeedSequence([seed, 2, k])
def refit(problem_swapped, table_problem, x_mode, starts, cfg_swapped, rng, ...) -> SwapResult
                                               # SYS-2: 4 starts, polish each, keep the best
def limited_by_systematics(shifts, iv_mode) -> (quantities, note)   # SYS-5, 3 statistical sigma
def run_systematics(problem, cfg, modes, uncertainty, seed, workers=1, log=None, swaps=None) -> SystematicsResult
                                               # uncertainty = fit's {"intervals", "results": {mode id: McmcResult}}
                                               # or None; (mode, swap) pairs in search.make_pool when workers > 1
def attach_jackknife(intervals, sysres_dict) -> list[int]   # only the jackknife_sigma fields of intervals.json
SystematicsResult.to_dict()                    # systematics.json: swaps, rules, modes[{mode, left_mode_rule, d99,
                                               # left_mode_limit, values, swaps{name: {keys, neg_log_l, start_neg_log_l,
                                               # n_starts, x, fingerprint_distance, mode_change, error}},
                                               # shifts{quantity: {swap: shift | "mode change" | "failed"}},
                                               # mode_changes, limited_by_model_systematics, sys5_note,
                                               # jackknife_sigma, jackknife_n, jackknife_note, notes}], notes
```
"Mode change" (DESIGN.md section 3): fingerprint distance of the refit (evaluated on the table's Problem) to the mode
above the 99th percentile of the same distance over the mode's MCMC samples; without samples, above 1 (search
thresholds), and the output says which rule was used. `cli fit --systematics/--no-systematics` (default on),
`--sys-modes N` (default all); `report.write_fit(..., systematics=dict)` writes systematics.json and the report lines.

## Configuration, `config.py`

`load_config(path) -> dict`: nested dict with every key of `defaults.toml` present (defaults filled in), unknown keys
rejected with `ConfigError` naming the key path, `library.path` and `data.path` resolved against `$ZEEMAP_DATA`, other
relative paths against the configuration file's directory, `[[data.phase]]` files against `data.path`.
`defaults()` returns the defaults; `dump(cfg) -> str` (TOML) for `run.json` and V1k.
