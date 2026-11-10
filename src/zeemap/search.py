"""The search for modes (DESIGN.md section 3; SPEC.md SEARCH-1 to SEARCH-4, SEARCH-7, PAR-2, PAR-4,
REP-3, ENG-3).

The search talks to the fitting problem only through the protocol `CostProblem` below, so that it can be tested on
toy problems and does not depend on the forward model. `problem.py` implements the protocol for the real fit.

Steps (DESIGN.md section 3, "Algorithm"; step 6, re-polishing at the error-bar patch count, is SEARCH-8 and lives
outside this module):

1. `draw_starts`: the start distribution of `cfg["search"]["starts"]`, one numpy Generator per start
   (`start_streams`: `SeedSequence(seed).spawn(n)`, ENG-3). A start that the problem rejects (cost >= 1e30), that
   breaks the prior cut (`problem.outside_fraction(x)`, if the problem has it, above `starts.outside_max`) or that
   fails `problem.accept_start(x)` (if the problem has it; this is where `starts.rules` are applied) is redrawn from
   its own stream, at most 100 tries.
2. `polish`: L-BFGS-B in coordinates scaled by `step`, gradient by central differences with half-step
   `fd_step * step`, all 2 nfree + 1 points in one batched `cost` call, one-sided at a polish limit or next to an
   invalid model (`fd_rows`, `fd_derivative`, `cost_gradient`: the one stencil of the polish gradient, the LM Jacobian
   and the family midpoint); then a Levenberg-Marquardt finish of at most `lm_maxiter` iterations on
   `problem.residuals`, a step accepted only if the full cost goes down, with the stops of `LM_STOPS`
   (`lm_stop_nat`, `lm_max_rejections`).
3. `fingerprint`, `same_mode`, `group_modes`: greedy grouping from the best end point down.
4. Restart rounds: each of the `top` best modes is perturbed `restarts` times and polished again; rounds repeat
   until a round improves the best mode by less than `round_gain` nats, at most `rounds_max` rounds.
5. The mode table keeps every mode within `band` nats of the best (SEARCH-7). Statistics (SEARCH-4): number of
   starts, number of distinct modes, hits on the best mode, the fraction of the band modes reached by at most one
   start, and the warning when that fraction is above 0.5. Optimiser scatter and the value tolerance
   max(2, min(2 x scatter, 10)) nats (DESIGN.md section 3, "Value tolerance").

Fingerprint thresholds (DESIGN.md section 3, step 3; decision #16 of DECISIONS.md). Components: the angle
(threshold `incl`, 8 degrees), the field centre (`c`, 0.06 stellar radii, the Euclidean length of the difference of the
two centre vectors), B50 and B90 at the reference phase (index `ref_phase`, 0) and the rms of B50 over the phases, and
the non-field free parameters named in `extra`. Defaults (`FINGERPRINT_DEFAULTS`, the same values in defaults.toml):
- `relative = true`: the B50, B90 and B50-rms thresholds are max(floor, fraction x B), B = the larger |B50| at the
  reference phase of the two fingerprints compared, fractions `b50_rel`, `b90_rel`, `b50_rms_rel` (0.04, 0.10, 0.05),
  floors `b50`, `b90`, `b50_rms` (0.5, 1.0, 1.0 MG). The research's absolute 5 / 25 / 10 MG equal the fractions at
  B50 = 125 / 250 / 200 MG; the small floors bind only below 12.5 / 10 / 20 MG. `relative = false` gives the absolute
  rule with the floors as the thresholds.
- `extra = {teff = 1000.0}`: non-field free parameters enter the fingerprint. The problem supplies
  `fingerprint_quantities(x)["extra"] = {name: value}` for every non-field free parameter (optional; absent = {});
  `Fingerprint.extra` keeps them all (and `to_dict` writes them when there are any), and the distance adds
  |difference| / threshold for each name in the table that both fingerprints have; a name missing from either (a fixed
  Teff, say) is ignored, so the Teff entry acts only when Teff is free. A configuration's `extra` table replaces the
  default table (`extra = {}` removes Teff).
- `angle = "auto"`: "beta" when both fingerprints have one phase and carry beta, "incl" otherwise. "beta" compares
  beta, the angle between the dipole axis and the observer direction at the reference phase (0-180 degrees,
  `fingerprint_quantities(x)["beta"]`, (P,)), with the threshold `incl`, and compares the centre in the frame of the
  dipole axis and the observer (`fingerprint_quantities(x)["c_frame"]`, (P, 3): the component along md, the component
  along the part of the observer direction perpendicular to md, and the absolute value of the third component) instead
  of the centre vector in the star's frame. One phase determines beta and this frame, but not the inclination or the
  azimuth of md about the line of sight, and the centre rotates with md; so with "incl" (or with the star-frame centre)
  one geometry splits into many modes, one per inclination the threshold separates.
  A fingerprint without `c_frame` falls back to the star-frame centre.

Mirror images (`settle_mirror`). The canonical form (u >= 0) is reached by the equator mirror, which is a symmetry of
the problem but not of the Fibonacci patch grid: cost(mirror(x)) - cost(x) is quadrature noise, so the two images of
one geometry differ slightly in -ln L. After the search every mode's end point and its mirror image are both evaluated
(one batched cost call; the mirror image of a canonical end point from `problem.mirror`, when the problem has it); the
better of the two becomes the mode's point and value (`Mode.x`, `Mode.f`, `Mode.image`), and the cost at the canonical
point, the vector the outputs write, is kept as `Mode.f_canonical`. The statistics get `mirror_asymmetry_max` (max
|f(mirror) - f(x)| over the band modes).

Hits. `Mode.hits` counts the end points of the initial starts (round 0) in the mode: SEARCH-3 asks how many starts
reached a mode, and restarts of a mode land in it by construction. `Mode.members` lists every end point (starts and
restarts). The "reached once" fraction of SEARCH-4 counts band modes with hits <= 1 (a mode found only by a restart
has 0 hits and counts as reached once).

Evaluations. `nevals` counts the rows of every `cost` call plus every `residuals` call (one forward-model evaluation
each), including the cost calls made to check starts.

Range limits (DESIGN.md section 3, "Range limits in the vector space"). Free parameters inside `problem.field_slice`
get symmetric safety limits from the problem's `hi`: each component of md, mo, aq and c and the scalars bq, bo within
+-|hi| of that component (for vectors the problem's `hi` is the configured upper limit of the length), and u within
[-1, 1]. Every other free parameter is bounded by [lo, hi]. `touched_limit`: a free parameter within 1 % of the width
of its limit interval from a limit (REP-3).

A problem without field vectors (`field_slice = None`, e.g. a toy problem): nothing here then assumes the 14-vector
layout: starts are uniform in [lo, hi] and restart perturbations are Gaussian with sigma = step for every free parameter.

Parallelism. Every polish is done by the picklable top-level function `polish_start`; `run_search(workers > 1)` maps
it over a `concurrent.futures.ProcessPoolExecutor` (`make_pool`) in order, so the result does not depend on the number
of workers (the start points and their streams are made in the main process). Each worker sets OpenMP, BLAS and
numba to one thread when it starts (`_single_thread_worker`, ENG-2); `thread_state` reports them. Check V6.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import Bounds, minimize

WORST = 1e30  # cost of an invalid model (LIKE-7)
INVALID_RESIDUAL = 1e6  # residual value of an invalid model (keeps Levenberg-Marquardt finite; Problem fills such rows)
MAX_TRIES = 100  # redraws per start
FINGERPRINT_DEFAULTS = {"incl": 8.0, "angle": "auto", "c": 0.06, "b50": 0.5, "b90": 1.0, "b50_rms": 1.0, "ref_phase": 0,
                        "relative": True, "b50_rel": 0.04, "b90_rel": 0.10, "b50_rms_rel": 0.05,
                        "extra": {"teff": 1000.0}}  # decision #16
ANGLE_CHOICES = ("auto", "incl", "beta")
SCATTER_DEFAULT = 5.0  # nats, when no mode has 3 end points (DESIGN.md section 3)

VECTOR_GROUPS = ("md", "aq", "mo", "c")
SCALAR_GROUPS = ("bq", "bo", "u")


class SearchError(RuntimeError):
    """The search cannot proceed (e.g. no valid start in 100 tries)."""


@runtime_checkable
class CostProblem(Protocol):
    """What the search needs from a problem (implemented by problem.Problem and by the toy problems of the tests).

    Optional members, used when present: `outside_fraction(x) -> float or (P,)` (prior cut of the starts and of the
    end points), `accept_start(x) -> bool` (extra start rules, `search.starts.rules`), `rng_draw_start(rng) -> x`
    (replaces the generic start distribution; the full vector, fixed parameters included).
    """

    dim: int
    names: list[str]
    free: np.ndarray  # (dim,) bool
    lo: np.ndarray  # (dim,)
    hi: np.ndarray
    step: np.ndarray
    value: np.ndarray
    field_slice: slice | None  # index range of the field numbers inside x, or None

    def cost(self, X: np.ndarray) -> np.ndarray: ...  # (K, dim) -> (K,), -ln L with penalties, 1e30 if invalid

    def residuals(self, x: np.ndarray) -> np.ndarray: ...  # (dim,) -> whitened residuals; may raise NotImplementedError

    def fingerprint_quantities(self, x: np.ndarray) -> dict: ...  # {incl (deg, 0-90), c (3,), b50 (P,), b90 (P,)}
    # optional keys "extra": {name: value} of the non-field free parameters; "beta" (P,) degrees; "c_frame" (P, 3)

    def canonical(self, x: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers


def _search_cfg(cfg: dict) -> dict:
    return cfg["search"]


def _field_range(problem) -> range:
    fs = getattr(problem, "field_slice", None)
    if fs is None:
        return range(0)
    if isinstance(fs, slice):
        return range(*fs.indices(problem.dim))
    if isinstance(fs, range):
        return fs
    a, b = fs
    return range(int(a), int(b))


def _group_of(name: str) -> str | None:
    for g in VECTOR_GROUPS:
        if name.startswith(g + "_"):
            return g
    return name if name in SCALAR_GROUPS else None


def field_groups(problem) -> dict[str, list[int]]:
    """Indices of the field numbers by group (md, aq, mo, c: 3 each; bq, bo, u: 1), from `problem.names` inside
    `field_slice`. Names inside the slice that belong to no group are returned under "other"."""
    groups: dict[str, list[int]] = {}
    for i in _field_range(problem):
        g = _group_of(problem.names[i]) or "other"
        groups.setdefault(g, []).append(i)
    return groups


def start_streams(seed: int, n: int) -> list[np.random.Generator]:
    """One independent Generator per start: SeedSequence(seed).spawn(n) (ENG-3)."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def restart_streams(seed: int, round_no: int, n: int) -> list[np.random.Generator]:
    """One Generator per restart of round `round_no` (>= 1), from the entropy [seed, 1, round_no]: independent of the
    start streams and of the other rounds."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence([int(seed), 1, int(round_no)]).spawn(n)]


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        v = rng.standard_normal(3)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            return v / n


def _as_streams(rng, n: int) -> list[np.random.Generator]:
    if isinstance(rng, (int, np.integer)):
        return start_streams(int(rng), n)
    if isinstance(rng, np.random.Generator):
        return [rng] * n
    streams = list(rng)
    if len(streams) != n:
        raise ValueError(f"need {n} random streams, got {len(streams)}")
    return streams


def _with_fixed(problem, x: np.ndarray) -> np.ndarray:
    free = np.asarray(problem.free, dtype=bool)
    out = np.array(x, dtype=float)
    out[~free] = np.asarray(problem.value, dtype=float)[~free]
    return out


def polish_bounds(problem) -> tuple[np.ndarray, np.ndarray]:
    """Safety limits of polish (DESIGN.md section 3, "Range limits"), as full-length arrays (lo, hi)."""
    lo = np.array(problem.lo, dtype=float)
    hi = np.array(problem.hi, dtype=float)
    blo, bhi = lo.copy(), hi.copy()
    for i in _field_range(problem):
        g = _group_of(problem.names[i])
        if g == "u":
            blo[i], bhi[i] = -1.0, 1.0
        elif g is not None:
            a = abs(hi[i])
            blo[i], bhi[i] = -a, a
    return blo, bhi


def _touched(x: np.ndarray, idx: np.ndarray, blo: np.ndarray, bhi: np.ndarray, step: np.ndarray) -> bool:
    for j in idx:
        lo, hi = blo[j], bhi[j]
        if np.isfinite(lo) and np.isfinite(hi):
            tol = 0.01 * (hi - lo)
        else:
            fin = lo if np.isfinite(lo) else hi
            tol = 0.01 * max(abs(fin), abs(step[j])) if np.isfinite(fin) else 0.0
        if (np.isfinite(lo) and x[j] - lo <= tol) or (np.isfinite(hi) and hi - x[j] <= tol):
            return True
    return False


def _outside_ok(problem, x: np.ndarray, outside_max: float) -> bool:
    fn = getattr(problem, "outside_fraction", None)
    if fn is None:
        return True
    return float(np.max(np.atleast_1d(fn(x)))) <= outside_max


def _start_ok(problem, x: np.ndarray, outside_max: float) -> tuple[bool, int]:
    """(acceptable, evaluations used)."""
    f = float(np.asarray(problem.cost(x[None, :]), dtype=float)[0])
    if not (f < WORST):
        return False, 1
    if not _outside_ok(problem, x, outside_max):
        return False, 1
    acc = getattr(problem, "accept_start", None)
    if acc is not None and not acc(x):
        return False, 1
    return True, 1


# ---------------------------------------------------------------------------------------------------------------------
# 1. Starts


def start_distribution(problem, cfg: dict) -> dict:
    """The start distribution actually used, for run.json (PAR-4)."""
    names = list(problem.names)
    free = np.asarray(problem.free, dtype=bool)
    fixed = {names[i]: float(problem.value[i]) for i in range(problem.dim) if not free[i]}
    if getattr(problem, "rng_draw_start", None) is not None:
        return {"kind": "problem", "description": str(getattr(problem, "start_description", "problem.rng_draw_start")),
                "fixed": fixed}
    fr = set(_field_range(problem))
    uniform = {names[i]: [float(problem.lo[i]), float(problem.hi[i])] for i in range(problem.dim)
               if free[i] and (i not in fr or _group_of(names[i]) is None)}
    out: dict[str, Any] = {"kind": "field" if fr else "uniform", "uniform": uniform, "fixed": fixed}
    if fr:
        st = _search_cfg(cfg)["starts"]
        out["field"] = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in st.items()}
        out["field"]["groups"] = sorted(g for g in field_groups(problem) if g != "other")
    return out


def _draw_one(problem, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """One draw of the generic start distribution (no acceptance test)."""
    draw = getattr(problem, "rng_draw_start", None)
    if draw is not None:
        return _with_fixed(problem, np.asarray(draw(rng), dtype=float))
    x = np.array(problem.value, dtype=float)
    lo, hi = np.asarray(problem.lo, dtype=float), np.asarray(problem.hi, dtype=float)
    groups = field_groups(problem)
    in_field = set()
    if groups:
        st = _search_cfg(cfg)["starts"]
        for g in ("md", "bq", "aq", "mo", "bo", "c", "u"):  # fixed order: the draws of a stream are reproducible
            idx = groups.get(g)
            if not idx:
                continue
            in_field.update(idx)
            if g == "md":
                x[idx] = rng.uniform(float(st["md"][0]), float(st["md"][1])) * _unit_vector(rng)
            elif g == "bq":
                x[idx] = rng.uniform(-float(st["bq"]), float(st["bq"]))
            elif g == "aq":
                x[idx] = _unit_vector(rng)
            elif g == "mo":
                x[idx] = rng.uniform(0.0, float(st["mo"])) * _unit_vector(rng)
            elif g == "bo":  # dqo_aligned: signed octupole strength along the dipole axis
                x[idx] = rng.uniform(-float(st["mo"]), float(st["mo"]))
            elif g == "c":
                x[idx] = float(st["c"]) * rng.uniform() ** (1.0 / 3.0) * _unit_vector(rng)
            elif g == "u":
                x[idx] = rng.uniform(float(st["u"][0]), float(st["u"][1]))
    for i in range(problem.dim):
        if problem.free[i] and i not in in_field:
            if not (np.isfinite(lo[i]) and np.isfinite(hi[i])):
                raise SearchError(f"parameter {problem.names[i]!r} is free but has no finite [lo, hi] to draw from")
            x[i] = rng.uniform(lo[i], hi[i])
    return _with_fixed(problem, x)


def _draw_starts(problem, cfg: dict, streams: list[np.random.Generator]) -> tuple[np.ndarray, int, int]:
    """(starts (n, dim), evaluations, redraws)."""
    outside_max = float(_search_cfg(cfg)["starts"].get("outside_max", math.inf))
    X = np.empty((len(streams), problem.dim))
    nevals = redraws = 0
    for k, rng in enumerate(streams):
        for tries in range(MAX_TRIES):
            x = _draw_one(problem, cfg, rng)
            ok, ne = _start_ok(problem, x, outside_max)
            nevals += ne
            if ok:
                X[k] = x
                redraws += tries
                break
        else:
            raise SearchError(f"start {k}: no acceptable start in {MAX_TRIES} draws (rejected by the problem or the "
                              f"prior cut); check [search.starts]")
    return X, nevals, redraws


def draw_starts(problem, cfg: dict, rng, n: int) -> np.ndarray:
    """n starts (n, dim) from the start distribution (DESIGN.md section 3).

    rng: an int seed (then `start_streams(seed, n)`), a sequence of n Generators (one per start; what run_search
    passes), or one Generator shared by all starts.
    """
    X, _, _ = _draw_starts(problem, cfg, _as_streams(rng, n))
    return X


def perturb(problem, x: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """One restart point near x (DESIGN.md section 3, step 4; `cfg["search"]["perturb"]`).

    Field numbers: field strengths (|md|, |mo|, bq, bo) times (1 + b_frac N), the directions of md, mo and aq turned by
    a Gaussian angle of sigma `angle` degrees (an isotropic tangent kick), c plus a Gaussian of sigma `c` per
    component, the inclination arccos(u) plus a Gaussian of sigma `incl` degrees. Every other free parameter (all of a
    toy problem's, when field_slice is None) plus a Gaussian of sigma = step. Fixed parameters stay.
    """
    pc = _search_cfg(cfg)["perturb"]
    out = np.array(x, dtype=float)
    groups = field_groups(problem)
    in_field: set[int] = set()
    bfrac, ang = float(pc["b_frac"]), math.radians(float(pc["angle"]))
    for g in ("md", "bq", "aq", "mo", "bo", "c", "u"):
        idx = groups.get(g)
        if not idx:
            continue
        in_field.update(idx)
        v = out[idx]
        if g in ("md", "mo", "aq"):
            s = float(np.linalg.norm(v))
            if s == 0.0:
                d, s = _unit_vector(rng), (s if g != "aq" else 1.0)
            else:
                d = v / s
            kick = rng.standard_normal(3)
            kick -= (kick @ d) * d
            d = d + ang * kick
            d /= np.linalg.norm(d)
            if g != "aq":
                s *= 1.0 + bfrac * rng.standard_normal()
            out[idx] = s * d
        elif g in ("bq", "bo"):
            out[idx] = v * (1.0 + bfrac * rng.standard_normal())
        elif g == "c":
            out[idx] = v + float(pc["c"]) * rng.standard_normal(3)
        elif g == "u":
            th = math.acos(min(1.0, max(-1.0, float(v[0]))))
            out[idx] = math.cos(th + math.radians(float(pc["incl"])) * rng.standard_normal())
    step = np.asarray(problem.step, dtype=float)
    for i in range(problem.dim):
        if problem.free[i] and i not in in_field:
            out[i] += step[i] * rng.standard_normal()
    blo, bhi = polish_bounds(problem)
    return _with_fixed(problem, np.clip(out, blo, bhi))


# ---------------------------------------------------------------------------------------------------------------------
# 2. Polish


def fd_rows(x: np.ndarray, idx: np.ndarray, s: np.ndarray, h: float, blo: np.ndarray | None = None,
            bhi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 2 n + 1 rows of a central difference around x over the free coordinates idx (n of them): row 0 is x, row
    1 + k is x with coordinate idx[k] raised by h s[k], row 1 + n + k the same lowered. A raised (lowered) point beyond
    the upper (lower) polish limit is not taken: its row repeats x, and its flag in (inp, inm) is False. Returns
    (X, inp, inm). The one finite-difference stencil of the search: the L-BFGS-B gradient (`cost_gradient`), the LM
    Jacobian (`_jacobian_batched`) and the family midpoint (`families.Linker.midpoint`) all use it with
    `fd_derivative`."""
    x = np.asarray(x, dtype=float)
    n = idx.size
    k = np.arange(n)
    xp, xm = x[idx] + h * s, x[idx] - h * s
    inp = np.ones(n, dtype=bool) if bhi is None else xp <= bhi[idx]
    inm = np.ones(n, dtype=bool) if blo is None else xm >= blo[idx]
    X = np.repeat(x[None, :], 2 * n + 1, axis=0)
    X[1 + k[inp], idx[inp]] = xp[inp]
    X[1 + n + k[inm], idx[inm]] = xm[inm]
    return X, inp, inm


def fd_derivative(v0: np.ndarray, vp: np.ndarray, vm: np.ndarray, okp: np.ndarray, okm: np.ndarray,
                  h: float) -> np.ndarray:
    """Derivatives (n, ...) in the scaled coordinates from the values at the rows of `fd_rows` (v0 at x, vp and vm
    (n, ...) at the raised and lowered points; okp, okm: (n,) the point was taken and its model is valid): the central
    difference where both sides are usable, the one-sided difference from the usable side where only one is (at a
    polish limit, or next to an invalid model), 0 where neither is."""
    v0 = np.asarray(v0, dtype=float)
    D = np.zeros((okp.size, *v0.shape))
    both = okp & okm
    D[both] = (vp[both] - vm[both]) / (2.0 * h)
    onlyp = okp & ~okm
    D[onlyp] = (vp[onlyp] - v0) / h
    onlym = okm & ~okp
    D[onlym] = (v0 - vm[onlym]) / h
    return D


def cost_gradient(cost: Callable[[np.ndarray], np.ndarray], x: np.ndarray, idx: np.ndarray, s: np.ndarray, h: float,
                  blo: np.ndarray | None = None, bhi: np.ndarray | None = None) -> tuple[float, np.ndarray, int]:
    """-ln L at x and its gradient in the scaled coordinates (x[idx] / s) from one batched `cost` call on the rows of
    `fd_rows` (a point whose cost is 1e30, an invalid model, is not usable). Returns (f, g, rows evaluated); g = 0
    when x itself is invalid."""
    X, inp, inm = fd_rows(x, idx, s, h, blo, bhi)
    c = np.asarray(cost(X), dtype=float)
    n = idx.size
    f0 = float(c[0])
    if not (f0 < WORST):
        return f0, np.zeros(n), X.shape[0]
    fp, fm = c[1:n + 1], c[n + 1:]
    return f0, fd_derivative(f0, fp, fm, inp & (fp < WORST), inm & (fm < WORST), h), X.shape[0]


@dataclass
class PolishResult:
    x: np.ndarray
    f: float
    nevals: int
    converged: bool
    touched_limit: bool
    f_lbfgs: float = math.nan  # -ln L after L-BFGS-B, before the Levenberg-Marquardt finish
    lm_iterations: int = 0
    lm_used: bool = False
    lm_stop: str = ""  # why the LM finish stopped (LM_STOPS); "" when it did not run
    lm_nevals: int = 0  # evaluations of the LM finish (included in nevals)


def polish(problem, x0: np.ndarray, cfg: dict, trace: list | None = None) -> PolishResult:
    """L-BFGS-B then the Levenberg-Marquardt finish (DESIGN.md section 3, "Polish"). `trace`, if given, receives
    (LM iteration, -ln L) after every LM iteration, however it ends (tools/measure_lm_shortfall.py).
    `converged`: the stage that made the end point stopped on a convergence rule: the LM finish, judged by `LM_STOPS`,
    when its total gain over L-BFGS-B's end point exceeds `search.lm_stop_nat` (with that stop off: when it accepted a
    step) or when L-BFGS-B did not run; else the end point is in substance L-BFGS-B's, and so is the verdict
    (L-BFGS-B's success flag, whatever `lm_stop` says)."""
    sc = _search_cfg(cfg)
    free = np.asarray(problem.free, dtype=bool)
    idx = np.flatnonzero(free)
    n = idx.size
    step = np.asarray(problem.step, dtype=float)
    s = step[idx]
    h = float(sc["fd_step"])
    blo, bhi = polish_bounds(problem)
    base = _with_fixed(problem, np.clip(np.asarray(x0, dtype=float), blo, bhi))
    counter = {"n": 0}
    if trace is not None:
        counter["trace"] = trace

    def full(z: np.ndarray) -> np.ndarray:
        x = base.copy()
        x[idx] = z * s
        return x

    if n == 0:
        f = float(problem.cost(base[None, :])[0])
        return PolishResult(base, f, 1, True, False, f)

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        f0, g, rows = cost_gradient(problem.cost, full(z), idx, s, h, blo, bhi)
        counter["n"] += rows
        if not (f0 < WORST):
            return WORST, np.zeros(n)
        return f0, g

    if sc.get("lbfgs", True):
        res = minimize(fun, base[idx] / s, jac=True, method="L-BFGS-B",
                       bounds=Bounds(blo[idx] / s, bhi[idx] / s),
                       options={"maxiter": int(sc["lbfgs_maxiter"])})
        x = full(np.asarray(res.x, dtype=float))
        f = float(res.fun)
        converged = bool(res.success)
    else:  # search.lbfgs = false: the Levenberg-Marquardt finish alone (decision #25)
        x = base.copy()
        f = float(problem.cost(x[None, :])[0])
        counter["n"] += 1
        converged = False
    f_lbfgs = f
    lm_used, lm_it, lm_stop, lm_nevals = False, 0, "", 0
    lm_max = int(sc.get("lm_maxiter", 0))
    if lm_max > 0 and f < WORST:
        try:
            r = np.asarray(problem.residuals(x), dtype=float)
            counter["n"] += 1
            lm_used = True
        except NotImplementedError:
            r = None
        if r is not None:
            n_before = counter["n"]
            x, f, lm_it, lm_conv, lm_stop, _ = _lm(problem, x, f, r, idx, s, h, blo, bhi, lm_max, counter,
                                                   float(sc.get("lm_stop_nat", 0.0)),
                                                   int(sc.get("lm_max_rejections", 0)))
            # the stage that made the end point decides: the LM when its total gain over
            # L-BFGS-B's end point exceeds search.lm_stop_nat (with the stop off, 0: when it accepted a step); else
            # the end point is in substance L-BFGS-B's and so is the flag, also after an LM "predicted" stop, since
            # the Gauss-Newton model of the residuals leaves out the continuum log-det, and after tiny accepted steps
            # that end on "rejections", since on real data the "predicted" stop can fail to fire; with
            # search.lbfgs = false the LM's stop is the only verdict
            if f_lbfgs - f > float(sc.get("lm_stop_nat", 0.0)) or not sc.get("lbfgs", True):
                converged = lm_conv
            lm_nevals = counter["n"] - n_before + 1  # + the residuals call above
    return PolishResult(x, f, counter["n"], converged, _touched(x, idx, blo, bhi, step), f_lbfgs, lm_it, lm_used,
                        lm_stop, lm_nevals)


def _jacobian_batched(problem, x, idx, s, h, counter, blo=None, bhi=None):
    """Central-difference Jacobian of the whitened residuals in the scaled coordinates, half-step h parameter steps,
    in one batched forward call of 2 n + 1 rows, with the noise scales held at x, taken from row 0 of that batch (x
    itself) (decision #25; DESIGN.md section 3, "Polish"). A coordinate whose +h or -h point lies outside the polish
    limits (blo, bhi) or gives an invalid model takes the one-sided difference from the other side, as the gradient of
    `polish` does; that
    point is not evaluated (its row repeats x). A coordinate with neither side valid gets a zero column. Returns
    (r at x, J) or None when x is invalid."""
    n = idx.size
    X, inp, inm = fd_rows(x, idx, s, h, blo, bhi)
    Rb = problem.residuals_batch(X, None)  # scales None: those of row 0, x itself
    counter["n"] += X.shape[0]
    valid = ~np.all(Rb == INVALID_RESIDUAL, axis=1)
    if not valid[0]:
        return None
    r, P, M = Rb[0], Rb[1:n + 1], Rb[n + 1:]
    J = fd_derivative(r, P, M, inp & valid[1:n + 1], inm & valid[n + 1:], h).T
    return r, J


LM_STOPS = {  # PolishResult.lm_stop -> whether it counts as converged; `polish` takes this verdict
    # only when the LM gained more than search.lm_stop_nat over L-BFGS-B's end point
    "gain": True,  # an accepted step gained <= 1e-12 |f|
    "predicted": True,  # the undamped Gauss-Newton step predicts a gain below search.lm_stop_nat
    "no_step": True,  # the step, clipped to the limits, does not move x
    "held": True,  # every coordinate is held on a limit
    "lambda": False,  # lam passed 1e12: every step was rejected (the Gauss-Newton model points uphill)
    "rejections": False,  # search.lm_max_rejections consecutive steps were rejected
    "maxiter": False,  # search.lm_maxiter iterations
    "invalid": False,  # the Jacobian could not be evaluated at x
}


def _lm(problem, x, f, r, idx, s, h, blo, bhi, maxiter, counter, stop_nat=0.0, max_rej=0):
    """Levenberg-Marquardt in the scaled coordinates z = x / step, Marquardt damping lam * diag(J^T J); a step is kept
    only if the full cost goes down; a coordinate on a polish limit whose descent direction points outward is held
    for that step. Jacobian: central differences in one batched call when the problem has `residuals_batch` (decision
    #25; one-sided at the limits), else forward differences (nfree residual calls; backward at the upper limit).

    Stops (`LM_STOPS`; returns (x, f, iterations, converged, stop, accepted steps)): an accepted step gains <= 1e-12
    |f|; with `stop_nat` > 0 (`search.lm_stop_nat`) also a Jacobian (fresh after an accepted step) whose undamped
    Gauss-Newton step predicts a gain below `stop_nat` nats (0.5 g^T A^+ g, the model's whole remaining gain; an
    accepted step that gains less is no stop: after rejections it is damped and short); with
    `max_rej` > 0 (`search.lm_max_rejections`) `max_rej` rejected steps in a row; lam above 1e12; `maxiter`. Only the
    first group counts as converged: lam running away means the Gauss-Newton model found no descent, not that none is
    left (the residuals leave out the 0.5 ln det continuum term, so J^T r can disagree with the true gradient)."""
    lam = 1e-3
    J = None
    stop = "maxiter"
    it = 0
    rejected = 0
    accepted = 0
    batched = hasattr(problem, "residuals_batch") and hasattr(problem, "residual_scales")
    while it < maxiter:
        it += 1
        try:  # the trace gets (it, f) for every iteration, however it ends
            if J is None and batched:
                got = _jacobian_batched(problem, x, idx, s, h, counter, blo, bhi)
                if got is None:
                    stop = "invalid"
                    break
                r, J = got
            if J is None:
                J = np.empty((r.size, idx.size))
                for k, j in enumerate(idx):
                    xk = x.copy()
                    sg = -1.0 if x[j] + h * s[k] > bhi[j] else 1.0  # backward difference at the upper limit
                    xk[j] += sg * h * s[k]
                    J[:, k] = sg * (np.asarray(problem.residuals(xk), dtype=float) - r) / h
                    counter["n"] += 1
            A = J.T @ J
            g = J.T @ r
            d = np.diag(A).copy()
            d = np.where(d > 0, d, 1e-12 * max(float(d.max()), 1.0))
            # a coordinate on a polish limit whose descent direction -g points out of the range is held there (the
            # clipped Gauss-Newton step would otherwise keep pushing it into the limit and stall the others)
            held = ((x[idx] <= blo[idx]) & (g > 0)) | ((x[idx] >= bhi[idx]) & (g < 0))
            if held.all():
                stop = "held"  # a corner of the range where every coordinate is held: no descent step left
                break
            fr = ~held
            if stop_nat > 0.0 and rejected == 0:
                Af, gf = A[np.ix_(fr, fr)], g[fr]
                pred = 0.5 * float(gf @ np.linalg.lstsq(Af, gf, rcond=None)[0])
                if pred < stop_nat:
                    stop = "predicted"
                    break
            dz = np.zeros(idx.size)
            try:
                dz[fr] = -np.linalg.solve((A + lam * np.diag(d))[np.ix_(fr, fr)], g[fr])
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xt = x.copy()
            xt[idx] = np.clip(x[idx] + dz * s, blo[idx], bhi[idx])
            if np.array_equal(xt, x):
                stop = "no_step"
                break
            ft = float(problem.cost(xt[None, :])[0])
            counter["n"] += 1
            if ft < f:
                gain = f - ft
                x, f = xt, ft
                rejected = 0
                accepted += 1
                if not batched:
                    r = np.asarray(problem.residuals(x), dtype=float)
                    counter["n"] += 1
                J = None
                lam = max(lam / 10.0, 1e-12)
                # no stop on a gain below stop_nat: after rejections the step is damped and can gain little while the
                # undamped step still predicts much more; the "predicted" test on the next Jacobian decides
                if gain <= 1e-12 * max(1.0, abs(f)):
                    stop = "gain"
                    break
            else:
                lam *= 10.0
                rejected += 1
                if lam > 1e12:
                    stop = "lambda"  # no descent step left in the Gauss-Newton model
                    break
                if 0 < max_rej <= rejected:
                    stop = "rejections"
                    break
        finally:
            if "trace" in counter:
                counter["trace"].append((it, f))
    return x, f, it, LM_STOPS[stop], stop, accepted


def polish_start(args: tuple) -> PolishResult:
    """Picklable unit of work for a process pool: args = (problem, x0, cfg)."""
    problem, x0, cfg = args
    return polish(problem, x0, cfg)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Fingerprints


@dataclass
class Fingerprint:
    incl: float  # degrees, folded to 0-90
    c: np.ndarray  # (3,)
    b50: np.ndarray  # (P,)
    b90: np.ndarray  # (P,)
    extra: dict = field(default_factory=dict)  # non-field free parameters, {name: value}
    beta: np.ndarray | None = None  # (P,) degrees 0-180, angle between md and the observer (problem-supplied, optional)
    c_frame: np.ndarray | None = None  # (P, 3) centre in the frame of md and the observer (problem-supplied, optional)

    def to_dict(self) -> dict:
        out = {"incl": float(self.incl), "c": [float(v) for v in self.c], "b50": [float(v) for v in self.b50],
               "b90": [float(v) for v in self.b90]}
        if self.beta is not None:
            out["beta"] = [float(v) for v in self.beta]
        if self.c_frame is not None:
            out["c_frame"] = [[float(v) for v in row] for row in self.c_frame]
        if self.extra:
            out["extra"] = {str(k): float(v) for k, v in self.extra.items()}
        return out


def fingerprint_thresholds(cfg: dict | None) -> dict:
    """FINGERPRINT_DEFAULTS overridden by the optional cfg["search"]["fingerprint"] table."""
    out = dict(FINGERPRINT_DEFAULTS)
    if cfg is not None:
        out.update((cfg.get("search", {}) or {}).get("fingerprint", {}) or {})
    out["extra"] = dict(out.get("extra") or {})
    if out.get("angle", "auto") not in ANGLE_CHOICES:
        raise ValueError(f"[search.fingerprint] angle must be one of {ANGLE_CHOICES}, not {out['angle']!r}")
    return out


def _fold(incl: float) -> float:
    a = float(incl) % 180.0
    return 180.0 - a if a > 90.0 else a


def fingerprint(problem, x: np.ndarray) -> Fingerprint:
    """Fingerprint of x after mapping it to the canonical form (DESIGN.md section 3, step 3)."""
    q = problem.fingerprint_quantities(problem.canonical(np.asarray(x, dtype=float)))
    return Fingerprint(_fold(q["incl"]), np.asarray(q["c"], dtype=float).reshape(3),
                       np.atleast_1d(np.asarray(q["b50"], dtype=float)),
                       np.atleast_1d(np.asarray(q["b90"], dtype=float)),
                       {str(k): float(v) for k, v in (q.get("extra") or {}).items()},
                       None if q.get("beta") is None else np.atleast_1d(np.asarray(q["beta"], dtype=float)),
                       None if q.get("c_frame") is None else np.asarray(q["c_frame"], dtype=float).reshape(-1, 3))


def angle_rule(f1: Fingerprint, f2: Fingerprint, angle: str) -> str:
    """The angle component actually compared: "incl" or "beta" ("auto": "beta" when both fingerprints have one phase
    and carry beta)."""
    if angle == "auto":
        one = len(f1.b50) == 1 and len(f2.b50) == 1
        return "beta" if one and f1.beta is not None and f2.beta is not None else "incl"
    return angle


def fingerprint_distance(f1: Fingerprint, f2: Fingerprint, thresholds: dict | None = None) -> float:
    """max over the components of |difference| / threshold (DESIGN.md section 3, step 3; used by UNC-3, SYS-4)."""
    t = dict(FINGERPRINT_DEFAULTS)
    t.update(thresholds or {})
    ref = int(t["ref_phase"])
    db50 = f1.b50 - f2.b50
    t50, t90, trms = t["b50"], t["b90"], t["b50_rms"]
    if t.get("relative", True):  # fractions of the larger B50 at the reference phase, absolute floors (decision #16)
        bref = max(abs(float(f1.b50[ref])), abs(float(f2.b50[ref])))
        t50 = max(t50, t["b50_rel"] * bref)
        t90 = max(t90, t["b90_rel"] * bref)
        trms = max(trms, t["b50_rms_rel"] * bref)
    angle = angle_rule(f1, f2, t.get("angle", "auto"))
    dc = f1.c - f2.c
    if angle == "beta":  # the dipole-observer angle at the reference phase (threshold incl), centre in that frame
        if f1.beta is None or f2.beta is None:
            raise ValueError('[search.fingerprint] angle = "beta" needs a problem that supplies "beta" in '
                             "fingerprint_quantities")
        dang = abs(float(f1.beta[ref]) - float(f2.beta[ref]))
        if f1.c_frame is not None and f2.c_frame is not None:
            dc = f1.c_frame[ref] - f2.c_frame[ref]
    elif angle == "incl":
        dang = abs(f1.incl - f2.incl)
        # polarity (DESIGN.md section 3, step 3): the folded inclination is the same for a
        # geometry and its field-reversed copy (V of opposite sign); beta at the reference phase tells them apart
        if f1.beta is not None and f2.beta is not None:
            dang = max(dang, abs(float(f1.beta[ref]) - float(f2.beta[ref])))
    else:
        raise ValueError(f"[search.fingerprint] angle must be one of {ANGLE_CHOICES}, not {t['angle']!r}")
    d = max(dang / t["incl"],
            float(np.linalg.norm(dc)) / t["c"],
            abs(float(db50[ref])) / t50,
            abs(float(f1.b90[ref] - f2.b90[ref])) / t90,
            float(np.sqrt(np.mean(db50 * db50))) / trms)
    for name, thr in (t.get("extra") or {}).items():  # non-field free parameters named in the table (decision #16)
        if name in f1.extra and name in f2.extra:
            d = max(d, abs(float(f1.extra[name]) - float(f2.extra[name])) / float(thr))
    return d


def same_mode(f1: Fingerprint, f2: Fingerprint, thresholds: dict | None = None) -> bool:
    """True when every component agrees within its threshold."""
    return fingerprint_distance(f1, f2, thresholds) <= 1.0


# ---------------------------------------------------------------------------------------------------------------------
# 4. End points and modes


@dataclass
class EndPoint:
    x: np.ndarray
    f: float
    start: int  # index of the start (round 0) or of the restart within its round
    round: int  # 0 = initial starts, 1.. = restart rounds
    nevals: int
    converged: bool
    touched_limit: bool
    valid: bool  # f < 1e30 and the prior cut holds; invalid end points are dropped from the mode table
    x0: np.ndarray | None = None
    parent: int | None = None  # restart rounds: rank of the perturbed mode
    fingerprint: Fingerprint | None = None
    mode: int | None = None  # index into SearchResult.all_modes
    f_lbfgs: float = math.nan
    lm_stop: str = ""  # PolishResult.lm_stop

    def to_dict(self) -> dict:
        return {"x": [float(v) for v in self.x], "f": float(self.f), "f_lbfgs": float(self.f_lbfgs),
                "start": self.start, "round": self.round, "nevals": self.nevals, "converged": self.converged,
                "lm_stop": self.lm_stop,
                "touched_limit": self.touched_limit, "valid": self.valid, "parent": self.parent, "mode": self.mode,
                "x0": None if self.x0 is None else [float(v) for v in self.x0],
                "fingerprint": None if self.fingerprint is None else self.fingerprint.to_dict()}


@dataclass
class Mode:
    x: np.ndarray  # best end point
    f: float
    fingerprint: Fingerprint
    hits: int  # end points of initial starts (round 0) in this mode
    members: list[int] = field(default_factory=list)  # indices of all its end points (starts and restarts)
    touched_limit: bool = False  # the best end point touches a range limit (REP-3)
    x_canonical: np.ndarray | None = None
    f_canonical: float | None = None  # cost at x_canonical (settle_mirror); None: not evaluated
    image: str = "end point"  # which image is x: "end point" or "mirror image" (of the end point; settle_mirror)
    mirror_asymmetry: float | None = None  # f(mirror image) - f(end point) (settle_mirror)

    def to_dict(self) -> dict:
        return {"x": [float(v) for v in self.x],
                "x_canonical": None if self.x_canonical is None else [float(v) for v in self.x_canonical],
                "f": float(self.f), "f_canonical": None if self.f_canonical is None else float(self.f_canonical),
                "image": self.image,
                "mirror_asymmetry": None if self.mirror_asymmetry is None else float(self.mirror_asymmetry),
                "fingerprint": self.fingerprint.to_dict(), "hits": self.hits,
                "members": list(self.members), "touched_limit": self.touched_limit}

    @property
    def f_written(self) -> float:
        """The cost at the vector the outputs write (x_canonical; x when the canonical point was not evaluated)."""
        return float(self.f if self.f_canonical is None else self.f_canonical)


def settle_mirror(problem, modes: list[Mode], reevaluate: bool = False) -> dict:
    """Evaluate each mode's point and its equator-mirror image, keep the better as the mode's point (see the module
    docstring, "Mirror images"), set x_canonical and f_canonical (the cost at x_canonical) and return
    {"mirror_asymmetry": [f(mirror) - f(x) per mode, None where no mirror image exists or one image is invalid],
    "n_mirror_better": count, "n_rows": rows evaluated}.
    One batched cost call: the mirror images, plus the points themselves when ``reevaluate`` (Mode.f then becomes the
    cost of ``problem`` at x, e.g. on another patch count). The canonical form itself is unchanged."""
    X, others = [], []
    for m in modes:
        x = np.asarray(m.x, dtype=float)
        xc = np.asarray(problem.canonical(x), dtype=float)
        if not np.array_equal(xc, x):
            other = xc
        else:
            mir = getattr(problem, "mirror", None)
            other = None if mir is None else mir(x)
            if other is not None and np.array_equal(np.asarray(other, dtype=float), x):
                other = None
        others.append(None if other is None else np.asarray(other, dtype=float))
        m.x_canonical = xc
    rows = ([np.asarray(m.x, dtype=float) for m in modes] if reevaluate else []) + [o for o in others if o is not None]
    f = np.asarray(problem.cost(np.array(rows)), dtype=float) if rows else np.zeros(0)
    if reevaluate:
        for m, v in zip(modes, f[: len(modes)]):
            m.f = float(v)
        f = f[len(modes):]
    asym, nbetter, j = [], 0, 0
    for m, o in zip(modes, others):
        x = np.asarray(m.x, dtype=float)
        if o is None:
            asym.append(None)
            m.mirror_asymmetry = None
            m.f_canonical = float(m.f)
            continue
        fo = float(f[j])
        j += 1
        a = fo - float(m.f) if (fo < WORST and m.f < WORST) else None  # None: one image invalid
        m.mirror_asymmetry = a
        asym.append(a)
        o_is_canonical = np.array_equal(o, m.x_canonical)
        m.f_canonical = fo if o_is_canonical else float(m.f)
        if fo < m.f:
            m.x, m.f = o, fo
            m.image = "end point" if m.image == "mirror image" else "mirror image"
            nbetter += 1
    return {"mirror_asymmetry": asym, "n_mirror_better": nbetter, "n_rows": len(rows)}


def mirror_statistics(modes: list[Mode], band: list[Mode]) -> dict:
    """The statistics entries of `settle_mirror` (diagnostics.json): max |f(mirror) - f(x)| over the band modes (a
    measure of the patch grid's quadrature noise), per mode in rank order, and the modes whose point is the mirror
    image of the end point."""
    vals = [abs(m.mirror_asymmetry) for m in band if m.mirror_asymmetry is not None]
    return {"mirror_asymmetry_max": max(vals) if vals else None,
            "mirror_asymmetry": [None if m.mirror_asymmetry is None else float(m.mirror_asymmetry) for m in modes],
            "modes_at_mirror_image": [k + 1 for k, m in enumerate(modes) if m.image == "mirror image"]}


def group_modes(endpoints: list[EndPoint], thresholds: dict | None = None) -> list[Mode]:
    """Greedy grouping from the best end point down: each end point joins the best existing mode whose fingerprint
    (that of its best end point) it matches, else it opens a new mode. Invalid end points or ones without a
    fingerprint are skipped. `members` are indices into `endpoints`; ties in f are broken by index (stable)."""
    order = sorted((i for i, e in enumerate(endpoints) if e.valid and e.fingerprint is not None),
                   key=lambda i: (endpoints[i].f, i))
    modes: list[Mode] = []
    for i in order:
        e = endpoints[i]
        for m in modes:
            if same_mode(m.fingerprint, e.fingerprint, thresholds):
                m.members.append(i)
                m.hits += int(e.round == 0)
                break
        else:
            modes.append(Mode(np.array(e.x), float(e.f), e.fingerprint, int(e.round == 0), [i], bool(e.touched_limit)))
    return modes


# ---------------------------------------------------------------------------------------------------------------------
# 5. The search


@dataclass
class SearchResult:
    modes: list[Mode]  # the mode table: every mode within `band` nats of the best, best first (SEARCH-7)
    all_modes: list[Mode]  # every distinct mode, best first
    endpoints: list[EndPoint]
    stats: dict
    start_distribution: dict
    nevals: int
    seed: int
    names: list[str]

    def to_dict(self) -> dict:
        """JSON-ready dict (run.json, diagnostics.json)."""
        return {"seed": self.seed, "names": list(self.names), "nevals": self.nevals, "stats": dict(self.stats),
                "start_distribution": self.start_distribution, "modes": [m.to_dict() for m in self.modes],
                "all_modes": [m.to_dict() for m in self.all_modes], "endpoints": [e.to_dict() for e in self.endpoints]}


def optimiser_scatter(modes: list[Mode], endpoints: list[EndPoint]) -> float:
    """Pooled standard deviation of -ln L over the end points of each mode with at least 3 end points
    (DESIGN.md section 3, "Value tolerance"); SCATTER_DEFAULT if no mode has 3."""
    ss, dof = 0.0, 0
    for m in modes:
        if len(m.members) >= 3:
            fs = np.array([endpoints[i].f for i in m.members])
            ss += float(np.sum((fs - fs.mean()) ** 2))
            dof += fs.size - 1
    return math.sqrt(ss / dof) if dof > 0 else SCATTER_DEFAULT


def value_tolerance(scatter: float) -> float:
    return max(2.0, min(2.0 * scatter, 10.0))


def search_statistics(all_modes: list[Mode], band_modes: list[Mode], n_starts: int) -> dict:
    """SEARCH-4: starts, distinct modes, hits on the best mode, fraction of band modes reached by at most one start,
    and the warning flag (fraction > 0.5)."""
    once = sum(1 for m in band_modes if m.hits <= 1)
    frac = once / len(band_modes) if band_modes else 0.0
    return {"n_starts": int(n_starts), "n_modes": len(all_modes), "n_band_modes": len(band_modes),
            "hits_best": int(all_modes[0].hits) if all_modes else 0, "n_band_once": once, "frac_band_once": frac,
            "warning": bool(frac > 0.5)}


WARNING_TEXT = ("most modes in the band were reached by one start only: the chance that a better mode was missed "
                "cannot be bounded (SEARCH-4)")


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_BLAS_SET = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads", "openblas_set_num_threads64_",
             "openblas_set_num_threads", "MKL_Set_Num_Threads")
_BLAS_GET = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads", "openblas_get_num_threads64_",
             "openblas_get_num_threads", "MKL_Get_Max_Threads")


def _blas_libs() -> list:
    """The BLAS libraries loaded in this process (OpenBLAS of numpy and of scipy, MKL), as ctypes handles; Linux only
    (read from /proc/self/maps), else an empty list."""
    import ctypes

    try:
        with open("/proc/self/maps") as fh:
            paths = sorted({ln.split()[-1] for ln in fh if "openblas" in ln.lower() or "libmkl_rt" in ln.lower()})
    except OSError:
        return []
    libs = []
    for p in paths:
        try:
            libs.append((p, ctypes.CDLL(p)))
        except OSError:
            pass
    return libs


def _single_thread_worker() -> None:
    """Pool initializer (ENG-2): one thread per worker process for OpenMP, BLAS and numba. The variables
    only reach libraries loaded later; BLAS libraries already loaded (fork inherits the parent's) are set through
    their own calls, numba through `set_num_threads` (NUMBA_NUM_THREADS cannot change once numba's threads exist)."""
    import os

    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        import numba

        numba.set_num_threads(1)
    except Exception:  # noqa: BLE001  numba absent or its threading layer unavailable
        pass
    for _p, lib in _blas_libs():
        for s in _BLAS_SET:
            if hasattr(lib, s):
                getattr(lib, s)(1)
                break


def thread_state() -> dict:
    """The thread settings of this process (V6b): the variables of THREAD_VARS and NUMBA_NUM_THREADS, numba's
    `get_num_threads()` and each loaded BLAS library's thread count (file name -> count)."""
    import os

    st: dict = {"pid": os.getpid(), "env": {v: os.environ.get(v) for v in THREAD_VARS + ("NUMBA_NUM_THREADS",)}}
    try:
        import numba

        st["numba"] = int(numba.get_num_threads())
    except Exception:  # noqa: BLE001
        st["numba"] = None
    st["blas"] = {}
    for p, lib in _blas_libs():
        for s in _BLAS_GET:
            if hasattr(lib, s):
                st["blas"][p.rsplit("/", 1)[-1]] = int(getattr(lib, s)())
                break
    return st


@contextmanager
def single_threaded():
    """Run the enclosed code with BLAS and numba at one thread, as in a pool worker, and restore the counts after.
    The serial search (workers = 1) polishes inside it: OpenBLAS sums in a different order with more threads, and the
    polished end points then differ from those of a pool in the last bits, which flat directions amplify (V6a)."""
    libs = []
    for _p, lib in _blas_libs():
        get = next((s for s in _BLAS_GET if hasattr(lib, s)), None)
        put = next((s for s in _BLAS_SET if hasattr(lib, s)), None)
        if get and put:
            libs.append((getattr(lib, put), int(getattr(lib, get)())))
    try:
        import numba

        n_numba = int(numba.get_num_threads())
    except Exception:  # noqa: BLE001
        numba, n_numba = None, None
    try:
        for put, _n in libs:
            put(1)
        if numba is not None:
            numba.set_num_threads(1)
        yield
    finally:
        for put, n in libs:
            put(n)
        if numba is not None:
            numba.set_num_threads(n_numba)


def make_pool(workers: int):
    """The process pool of the search: `workers` processes, each single-threaded (`_single_thread_worker`).
    With the default start method on Linux (fork) the workers inherit the parent's registry of Problems
    (problem.py), so a task ships only the problem's key and configuration."""
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=workers, initializer=_single_thread_worker)


def _polish_many(problem, X0: list[np.ndarray], cfg: dict, workers: int) -> list[PolishResult]:
    tasks = [(problem, x0, cfg) for x0 in X0]
    if workers <= 1 or len(tasks) <= 1:
        with single_threaded():  # the same thread counts as in a worker: BLAS results depend on them (SEARCH-5)
            return [polish_start(t) for t in tasks]
    with make_pool(min(workers, len(tasks))) as pool:
        return list(pool.map(polish_start, tasks))


def _endpoint(problem, pr: PolishResult, start: int, rnd: int, x0, parent, outside_max: float) -> EndPoint:
    valid = bool(pr.f < WORST) and _outside_ok(problem, pr.x, outside_max)
    e = EndPoint(pr.x, pr.f, start, rnd, pr.nevals, pr.converged, pr.touched_limit, valid, np.array(x0), parent,
                 f_lbfgs=pr.f_lbfgs, lm_stop=pr.lm_stop)
    if valid:
        e.fingerprint = fingerprint(problem, pr.x)
    return e


def run_search(problem, cfg: dict, seed: int, n_starts: int | None = None, workers: int = 1,
               log: Callable[[str], None] | None = None, make_problem: Callable | None = None,
               save: Callable[[dict], None] | None = None, state: dict | None = None) -> SearchResult:
    """Steps 1-5 and 7 of DESIGN.md section 3 (see the module docstring). With `make_problem` (configuration -> problem)
    and `objective.kernel_fit` true with the full likelihood, step 2b runs after round 0 (noise.kernel_stage): the
    kernel is fitted and frozen, and every later step uses the problem it returns; that problem and its configuration
    are `result.problem` and `result.cfg` (otherwise the ones passed in).
    Checkpoints: `save(state)` is called after round 0, after step 2b and after each restart round with
    the search's state so far (`step`, the end points, counters, the frozen kernel); `run_search(..., state=state)`
    continues from it with the same result as an uninterrupted run (the streams depend only on the seed and the
    round). The state holds no problem: the frozen kernel is re-applied to `cfg` (noise.with_kernels)."""
    sc = _search_cfg(cfg)
    say = log or (lambda _msg: None)
    n = int(sc["n_starts"] if n_starts is None else n_starts)
    outside_max = float(sc["starts"].get("outside_max", math.inf)) if "starts" in sc else math.inf
    thr = fingerprint_thresholds(cfg)

    def checkpoint(step: str) -> None:
        if save is not None:
            save({"step": step, "endpoints": endpoints, "nevals": nevals, "redraws": redraws, "gains": gains,
                  "rounds": rounds, "kernel_info": kernel_info})

    gains, rounds, kernel_info = [], 0, None
    if state is None:
        X0, nevals, redraws = _draw_starts(problem, cfg, start_streams(seed, n))
        say(f"search: {n} starts drawn ({redraws} redraws)")
        results = _polish_many(problem, list(X0), cfg, workers)
        endpoints = [_endpoint(problem, pr, k, 0, X0[k], None, outside_max) for k, pr in enumerate(results)]
        nevals += sum(pr.nevals for pr in results)
        checkpoint("round 0")
    else:
        endpoints, nevals, redraws = state["endpoints"], state["nevals"], state["redraws"]
        gains, rounds, kernel_info = list(state["gains"]), state["rounds"], state["kernel_info"]
        if kernel_info is not None:  # step 2b ran: continue on the problem with the frozen kernel
            from .noise import with_kernels

            cfg = with_kernels(cfg, kernel_info["frozen"])
            problem = make_problem(cfg)
            problem.objective.kernel_state = kernel_info
        say(f"search: resumed after {state['step']} ({len(endpoints)} end points)")
    modes = group_modes(endpoints, thr)
    if not modes:
        raise SearchError("no valid end point: every polished start was invalid or broke the prior cut")
    best = modes[0].f
    if state is None:
        say(f"search: round 0: {len(modes)} modes, best {best:.3f}")
    obj = cfg.get("objective") or {}  # with_kernels switches kernel_fit off, so a resumed step 2b does not run again
    if make_problem is not None and obj.get("kernel_fit", True) and getattr(problem.objective, "kind", "") == "full":
        from .noise import kernel_stage  # step 2b (LIKE-2, decision #22)

        problem, cfg, endpoints, kernel_info = kernel_stage(
            problem, endpoints, cfg=cfg, make_problem=make_problem,
            polish_many=lambda p, X: _polish_many(p, list(X), cfg, workers),
            make_endpoint=lambda p, pr, e: _endpoint(p, pr, e.start, e.round, e.x0, e.parent, outside_max),
            band=float(sc["band"]), say=say)
        modes = group_modes(endpoints, thr)
        if not modes:
            raise SearchError("no valid end point under the fitted kernel")
        best = modes[0].f
        say(f"search: round 0 under the frozen kernel: {len(modes)} modes, best {best:.3f}")
        checkpoint("the kernel stage")

    top, nre = int(sc["top"]), int(sc["restarts"])
    stopped = bool(gains) and gains[-1] < float(sc["round_gain"])  # a resumed search that had stopped
    for rnd in range(rounds + 1, 0 if stopped else int(sc["rounds_max"]) + 1):
        parents = modes[:top]
        if not parents or nre <= 0:
            break
        streams = restart_streams(seed, rnd, len(parents) * nre)
        starts, parent_of = [], []
        for pi, m in enumerate(parents):
            for j in range(nre):
                rng = streams[pi * nre + j]
                for _ in range(MAX_TRIES):
                    x0 = perturb(problem, m.x, cfg, rng)
                    ok, ne = _start_ok(problem, x0, outside_max)
                    nevals += ne
                    if ok:
                        starts.append(x0)
                        parent_of.append(pi)
                        break
        results = _polish_many(problem, starts, cfg, workers)
        nevals += sum(pr.nevals for pr in results)
        endpoints += [_endpoint(problem, pr, k, rnd, starts[k], parent_of[k], outside_max)
                      for k, pr in enumerate(results)]
        modes = group_modes(endpoints, thr)
        rounds = rnd
        gain = best - modes[0].f
        gains.append(gain)
        best = modes[0].f
        say(f"search: round {rnd}: {len(modes)} modes, best {best:.3f}, gain {gain:.3f}")
        checkpoint(f"round {rnd}")
        if gain < float(sc["round_gain"]):
            break

    settled = settle_mirror(problem, modes)  # value = the better image; f_canonical = the cost at the written vector
    nevals += settled["n_rows"]
    modes.sort(key=lambda m: (m.f, m.members[0]))
    best = modes[0].f
    for k, m in enumerate(modes):
        for i in m.members:
            endpoints[i].mode = k
    band = [m for m in modes if m.f <= best + float(sc["band"])]
    scatter = optimiser_scatter(modes, endpoints)
    stats = search_statistics(modes, band, n)
    stats.update({"rounds": rounds, "round_gains": gains, "redraws": redraws, "optimiser_scatter": scatter,
                  "value_tolerance": value_tolerance(scatter), "nevals": nevals, "band": float(sc["band"]),
                  "fingerprint_thresholds": thr, "n_endpoints": len(endpoints),
                  "n_invalid_endpoints": sum(1 for e in endpoints if not e.valid)})
    stats.update(mirror_statistics(modes, band))
    if stats["warning"]:
        stats["warning_text"] = WARNING_TEXT
        say("search: WARNING: " + WARNING_TEXT)
    if kernel_info is not None:
        stats["kernel"] = kernel_info
    res = SearchResult(band, modes, endpoints, stats, start_distribution(problem, cfg), nevals, int(seed),
                       list(problem.names))
    res.problem, res.cfg = problem, cfg
    return res
