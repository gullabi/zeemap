"""Profiles and envelopes per family, and the side-by-side table (SPEC UNC-5, UNC-8; DESIGN.md section 4b; decisions
#29, #30).

A profile of a quantity q is the lowest -ln L with q held at each grid value q_best + k dq and every free parameter
re-polished. A coordinate of the search is fixed directly (Teff; the inclination through u = cos i, with the sign of the
walk's start); |m_d| and the B quantiles are held exactly by scaling every field strength by q0 / q(x) (`_Scaled`,
`strength_indices`; decision #39), where B is linear in the strengths; any other q is held by the penalty
kappa ((q(x) - q0) / dq)^2 (kappa = 1000 nats), added to the cost and, as one more residual, to the Levenberg-Marquardt
finish. A walk starts at a family member and steps outwards in both
directions (each point polished from the previous one) until the profile exceeds the family's best by max(2 h, 10 nats),
or `uncertainty.profile_max_steps` steps, or the range of q ends: a coordinate fixed directly is polished once more at
the end of its range (0 or 90 degrees, the Teff limits) and the walk stops there; a point held by the penalty whose
polish misses the constraint (|q - q0| > 0.1 dq) is polished once more, with the LM stops off, from a start moved
towards q0; if that misses too and q cannot move towards q0 (`_towards`: no free parameter moves it that way, or all
that do sit at their polish limits, e.g. |m_d| < 0, a polish limit), the walk ends at the q it reached. Either is a
closed end of the range. A miss without such a limit is a stalled polish (the LM stopped short of a reachable q0): the
point is kept as not met, nothing is closed, and the walk goes on from its last point that met the constraint. Each
quantity is walked from the family's best node and from the node with the most distant q (from the far ends of the first
walk, back, when the family has one node). The reported profile is the lower envelope of the walks over the points that
met the constraint; the closed ends bound the intervals (a closed end is where the model cannot take q further, so the
interval ends there and not at the last grid value).

Parallelism (DESIGN.md section 4b): each direction of each walk is one task (`walk`, a pure function of its arguments),
mapped in a fixed order over the search's pool (`search.make_pool`) or, with one worker, run here under
`search.single_threaded`; the result does not depend on the number of workers (V6d).
"""

from __future__ import annotations

import math
import re

import numpy as np

from . import search as S

KAPPA = 1000.0  # nats per (grid step)^2 of the penalty
CONSTRAINT_TOL = 0.1  # |q - q0| > 0.1 dq: "constraint not met"
DEFAULT_STEPS = {"incl_deg": 2.5, "beta_deg": 2.5, "dipole_MG": "1%", "B10_MG": "1%", "B50_MG": "1%", "B90_MG": "1%",
                 "c_along_axis": 0.01, "teff_K": 100.0}
NOT_MET = "constraint not met"
STALLED = "polish stalled"
RANGE_END = "range end"
OPEN = "open (the walk ended inside the interval)"
# decision #39: a B quantile is a step function of the parameters at a finite patch count (forward.b_quantiles),
# so its held polishes can stall above the constrained minimum; its profiles are provisional until the quantile is
# made continuous in the parameters
QUANTILE_STEP = "B quantile is a step function: provisional"
MAX_STALLS = 2  # stalled points in a row after which a walk direction ends, open
NO_LM_STOPS = {"lm_stop_nat": 0.0, "lm_max_rejections": 0}  # the retry after a miss (decision #37's stops off)
# held by scaling the field strengths (decision #39): the quantities proportional to them, and the
# strength parameters of the field models whose B is linear in them (field.py: "dqo" md, bq, mo; "dqo_aligned" md, bq,
# bo; aq is a direction only, c and u are geometry)
SCALED = ("dipole_MG", "B10_MG", "B50_MG", "B90_MG")
STRENGTHS = ("md_x", "md_y", "md_z", "bq", "mo_x", "mo_y", "mo_z", "bo")


def _base(name: str) -> tuple[str, int | None]:
    m = re.fullmatch(r"(\w+)\[(\d+)\]", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def quantity_values(problem, X, name: str, d: dict | None = None) -> np.ndarray:
    """q for the rows of X (FORMATS.md section 7 names; "B50_MG[k]" for phase k); `d`: problem.derived_batch(X), if
    already computed."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    base, k = _base(name)
    if base == "teff_K":
        return X[:, problem.names.index("teff")]
    names = list(problem.names)
    if all(n in names for n in ("md_x", "md_y", "md_z")):
        md = X[:, [names.index(n) for n in ("md_x", "md_y", "md_z")]]
        nmd = np.linalg.norm(md, axis=1)
        if base == "dipole_MG":  # cheap forms of the quantities that need no forward model (derived_batch)
            return nmd
        if base == "incl_deg" and "u" in names:
            return np.degrees(np.arccos(np.minimum(1.0, np.abs(X[:, names.index("u")]))))
        if base in ("c_along_axis", "c_abs") and all(n in names for n in ("c_x", "c_y", "c_z")):
            c = X[:, [names.index(n) for n in ("c_x", "c_y", "c_z")]]
            if base == "c_abs":
                return np.linalg.norm(c, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(nmd > 0, np.einsum("ki,ki->k", c, md) / nmd, np.nan)
    d = problem.derived_batch(X) if d is None else d
    v = np.asarray(d[base], dtype=float)
    return v[:, k] if v.ndim == 2 else v


def default_quantities(problem, ref: int = 0) -> list[str]:
    """The default list of DESIGN.md section 4b, before the quantities that no free parameter moves are left out
    (`profile_names`). The B quantiles are those of phase `ref` only (`search.fingerprint.ref_phase`; decision #40)."""
    P = len(problem.dataset.phases)
    out = ["incl_deg" if P > 1 else "beta_deg", "dipole_MG", "c_along_axis"]
    out += [f"{b}[{ref}]" for b in ("B10_MG", "B50_MG", "B90_MG")]
    if getattr(problem, "iteff", None) is not None:
        out.append("teff_K")
    return out


def moved_by_free(problem, x, names: list[str]) -> list[bool]:
    """Whether each quantity changes when a free parameter moves: q at x against q with each free parameter in turn
    moved by +-its step (within the polish limits). False for a quantity that depends on no free parameter, e.g.
    c_along_axis with c fixed at 0; it cannot be profiled (every walk point would miss the
    constraint)."""
    x = np.asarray(x, dtype=float)
    blo, bhi = S.polish_bounds(problem)
    step = np.asarray(problem.step, dtype=float)
    rows = [x]
    for i in np.flatnonzero(np.asarray(problem.free, dtype=bool)):
        for sgn in (1.0, -1.0):
            r = x.copy()
            r[i] = min(max(x[i] + sgn * step[i], blo[i]), bhi[i])
            rows.append(r)
    X = np.array(rows)
    d = problem.derived_batch(X) if any(_base(n)[0] != "teff_K" for n in names) else None  # once for all
    out = []
    for name in names:
        v = quantity_values(problem, X, name, d)
        v0 = v[0]
        if np.isnan(v0):
            out.append(bool(np.any(~np.isnan(v[1:]))))
        else:
            out.append(bool(np.any(~(np.abs(v[1:] - v0) <= 1e-12 * max(1.0, abs(v0))))))
    return out


def profile_names(problem, cfg, x_best) -> list[str]:
    """The quantities to profile for a family whose best node is x_best: `uncertainty.profile_quantities` as listed, or
    for "default" the default list without the quantities that no free parameter moves (`moved_by_free`)."""
    pq = (cfg.get("uncertainty") or {}).get("profile_quantities", "default")
    if isinstance(pq, list):
        return list(pq)
    names = [n for n in default_quantities(problem, int(S.fingerprint_thresholds(cfg)["ref_phase"]))
             if not (_base(n)[0] == "teff_K" and getattr(problem, "iteff", None) is None)]
    return [n for n, ok in zip(names, moved_by_free(problem, x_best, names)) if ok]


def own_scale(problem, x, name: str) -> float:
    """s_q, the quantity's own scale: the largest |q(x + step_i e_i) - q(x)| over the free parameters i (one `step` of
    the configuration each), at x (the family's best node). It floors a percentage grid step (`grid_step`)."""
    x = np.asarray(x, dtype=float)
    idx = np.flatnonzero(np.asarray(problem.free, dtype=bool))
    X = np.repeat(x[None, :], idx.size + 1, axis=0)
    X[1 + np.arange(idx.size), idx] += np.asarray(problem.step, dtype=float)[idx]
    q = quantity_values(problem, X, name)
    d = np.abs(q[1:] - q[0])
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


def grid_step(name: str, q_best: float, steps: dict, scale=0.0) -> float:
    """dq of quantity `name`: its entry of `steps` (by name, then by the name without a phase index), else
    DEFAULT_STEPS. A step "<p>%" is p % of max(|q_best|, s_q), s_q = `scale` (a number, or a function called only for a
    percentage step; `own_scale`), so that a quantity whose best value is near 0 does not get a vanishing step (a plain
    percentage of a best value near 0 gave 1e-9); when both are 0 (no free parameter moves q) it is p / 100 in the
    quantity's own unit."""
    from .config import percent

    base, _ = _base(name)
    s = steps.get(name, steps.get(base, DEFAULT_STEPS.get(base, 0.01)))
    if isinstance(s, str):
        p = percent(s)
        if p is None:
            raise ValueError(f'profile step of {name}: {s!r} is not "<p>%" with p a positive number')
        ref = max(abs(q_best), float(scale() if callable(scale) else scale))
        return p / 100.0 * (ref if ref > 0 else 1.0)
    return float(s)


class _Held:
    """A view of `problem` with q held near q0 by the penalty (cost and one extra LM residual)."""

    def __init__(self, problem, name, q0, dq):
        self._p, self.name, self.q0, self.dq = problem, name, float(q0), float(dq)

    def __getattr__(self, n):
        return getattr(self._p, n)

    def _pen(self, X):
        q = quantity_values(self._p, X, self.name)
        return (q - self.q0) / self.dq

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = np.asarray(self._p.cost(X), dtype=float)
        ok = f < S.WORST
        out = f.copy()
        if np.any(ok):
            z = self._pen(X[ok])
            out[ok] = f[ok] + KAPPA * z * z
        return out

    def residuals_batch(self, X, scales):
        R = self._p.residuals_batch(X, scales)
        z = self._pen(X)
        return np.concatenate([R, (math.sqrt(2.0 * KAPPA) * z)[:, None]], axis=1)

    def residuals(self, x):
        r = self._p.residuals(x)
        return np.concatenate([r, [math.sqrt(2.0 * KAPPA) * float(self._pen(np.asarray(x)[None])[0])]])


def strength_indices(problem) -> np.ndarray | None:
    """The free field-strength parameters (`STRENGTHS`), when scaling them all by one factor s > 0 scales every
    quantity of `SCALED` by s exactly; else None (the penalty holds the quantity). B of "dqo" and "dqo_aligned" is
    linear in the strengths, the patch weights (area, mu, limb darkening) and the
    visibility depend only on the geometry (u, the phases), and scaling every |B| by s > 0 keeps their order, so the
    weighted quantile of `forward.b_quantiles` and |m_d| scale by s. That needs every strength parameter free or fixed
    at 0 (a fixed non-zero strength would not scale) and at least one free; other field models (plugins) and problems
    without a field are not assumed linear."""
    fld = getattr(problem, "field", None)
    if getattr(fld, "name", None) not in ("dqo", "dqo_aligned"):
        return None
    free = np.asarray(problem.free, dtype=bool)
    value = np.asarray(problem.value, dtype=float)
    idx = [i for i, n in enumerate(problem.names) if n in STRENGTHS]
    if not any(free[i] for i in idx) or any(not free[i] and value[i] != 0.0 for i in idx):
        return None
    return np.array([i for i in idx if free[i]], dtype=int)


class _Scaled:
    """A view of `problem` with a quantity of `SCALED` held at q0 by scaling the free field strengths `idx` by one
    factor (decision #39): a row X of the view stands for `actual(X)`, X with its strengths times
    q0 / q(X), so q = q0 exactly (`strength_indices`). The strength largest in |x| at the start (the pivot) is fixed,
    which removes the scale the factor makes redundant; the other parameters are polished. A row whose scaled strengths
    leave their polish limits is invalid (the polish keeps out of it; `_held_problem` closes the end there)."""

    def __init__(self, problem, name, q0, idx, x0):
        self._p, self.name, self.q0, self.idx = problem, name, float(q0), idx
        pivot = int(idx[np.argmax(np.abs(np.asarray(x0, dtype=float)[idx]))])
        self.free = np.asarray(problem.free, dtype=bool).copy()
        self.free[pivot] = False
        self.value = np.array(x0, dtype=float)
        self._blo, self._bhi = S.polish_bounds(problem)

    def __getattr__(self, n):
        return getattr(self._p, n)

    def _map(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        q = quantity_values(self._p, X, self.name)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = self.q0 / q
        Y = X.copy()
        ok = np.isfinite(f) & (f > 0)
        Y[np.ix_(ok, self.idx)] *= f[ok, None]
        tol = 1e-12 * np.maximum(1.0, np.abs(Y[:, self.idx]))
        ok &= np.all((Y[:, self.idx] >= self._blo[self.idx] - tol) & (Y[:, self.idx] <= self._bhi[self.idx] + tol),
                     axis=1)
        return Y, ok

    def actual(self, x) -> np.ndarray:
        """The problem's parameter vector a row of the view stands for (q = q0)."""
        return self._map(x)[0][0]

    def cost(self, X):
        Y, ok = self._map(X)
        out = np.full(Y.shape[0], S.WORST)
        if np.any(ok):
            out[ok] = np.asarray(self._p.cost(Y[ok]), dtype=float)
        return out

    def residual_scales(self, x):
        Y, ok = self._map(x)
        return self._p.residual_scales(Y[0]) if ok[0] else None

    def residuals_batch(self, X, scales):
        Y, ok = self._map(X)
        R = np.asarray(self._p.residuals_batch(np.where(ok[:, None], Y, Y[ok][0] if ok.any() else Y), scales))
        R[~ok] = S.INVALID_RESIDUAL
        return R

    def residuals(self, x):
        Y, ok = self._map(x)
        r = np.asarray(self._p.residuals(Y[0]), dtype=float)
        return r if ok[0] else np.full_like(r, S.INVALID_RESIDUAL)


def _scale_range(problem, idx, x) -> tuple[float, float]:
    """The factors s > 0 for which s x keeps every strength of `idx` within its polish limits."""
    blo, bhi = S.polish_bounds(problem)
    lo, hi = 0.0, math.inf
    for i in idx:
        v = float(x[i])
        if v > 0:
            lo, hi = max(lo, blo[i] / v), min(hi, bhi[i] / v)
        elif v < 0:
            lo, hi = max(lo, bhi[i] / v), min(hi, blo[i] / v)
    return lo, hi


def _polish(view, x0, cfg):
    """search.polish on a view; for a view held by scaling, the end point is mapped back to the problem's vector."""
    pr = S.polish(view, x0, cfg)
    if hasattr(view, "actual"):
        pr.x = view.actual(pr.x)
    return pr


def _held_problem(problem, name, q0, dq, x_start):
    """(problem view, x0) for one grid value: a direct coordinate is fixed; a quantity proportional to the field
    strengths (`SCALED`, `strength_indices`) is held by scaling them (`_Scaled`; x0 is x_start scaled to q0, or, when
    that would take a strength beyond its polish limits, scaled to the limit, with `view.limit` = the q there); else
    the penalty view."""
    from .uncertainty import _Fixed

    base, _ = _base(name)
    x0 = np.array(x_start, dtype=float)
    idx = strength_indices(problem) if base in SCALED else None
    if idx is not None:
        qs = float(quantity_values(problem, x0, name)[0])
        if np.isfinite(qs) and qs > 0:
            lo, hi = _scale_range(problem, idx, x0)
            f = min(max(q0 / qs, lo), hi)
            x0[idx] *= f
            view = _Scaled(problem, name, f * qs, idx, x0)
            view.limit = None if f == q0 / qs else f * qs
            return view, x0
    if base == "teff_K":
        i = problem.names.index("teff")
        x0[i] = q0
        return _Fixed(problem, i, q0, x0), x0
    if base == "incl_deg" and "u" in problem.names and problem.free[problem.names.index("u")]:
        i = problem.names.index("u")
        sign = 1.0 if x0[i] >= 0 else -1.0
        x0[i] = sign * math.cos(math.radians(q0))
        return _Fixed(problem, i, x0[i], x0), x0
    return _Held(problem, name, q0, dq), x0


def _direct_range(problem, name) -> tuple[float, float] | None:
    """The range of a quantity fixed directly (`_held_problem`): the inclination 0-90 degrees, Teff its polish limits
    (the library); None for a quantity held by the penalty."""
    base, _ = _base(name)
    if base == "teff_K":
        i = problem.names.index("teff")
        blo, bhi = S.polish_bounds(problem)
        return float(blo[i]), float(bhi[i])
    if base == "incl_deg" and "u" in problem.names and problem.free[problem.names.index("u")]:
        return 0.0, 90.0
    return None


def _towards(problem, name, x, q0: float, dq: float, iters: int = 8) -> tuple[np.ndarray, bool]:
    """(a start with q nearer q0, whether q cannot reach q0 from x) for the retry after a missed constraint, from
    evaluations of q only (no likelihood evaluations). Each free parameter in turn moves by +-its `step`
    (within the polish limits); those that move q towards q0 give the direction (each towards its better side, weighted
    by its gain per step), along which a secant search, at most `iters` evaluations of q, looks for q = q0 within
    0.01 dq. q cannot reach q0 (a limit, the second value True) when no free parameter moves q towards q0 by more than
    1e-12 max(1, |q|), or when the search reaches the polish limits of every such parameter still more than 0.1 dq
    short of q0. The start is the point of the search nearest q0 (x itself if none is nearer)."""
    x = np.asarray(x, dtype=float)
    blo, bhi = S.polish_bounds(problem)
    step = np.asarray(problem.step, dtype=float)
    idx = np.flatnonzero(np.asarray(problem.free, dtype=bool))
    rows = [x]
    for i in idx:
        for sgn in (1.0, -1.0):
            r = x.copy()
            r[i] = min(max(x[i] + sgn * step[i], blo[i]), bhi[i])
            rows.append(r)
    X = np.array(rows)
    q = quantity_values(problem, X, name)
    qx = float(q[0])
    if not np.isfinite(qx):
        return x, False
    toward = 1.0 if q0 > qx else -1.0
    n = idx.size
    gain = np.where(np.isfinite(q[1:]), toward * (q[1:] - qx), -np.inf).reshape(n, 2)  # progress of each move
    side = np.argmax(gain, axis=1) if n else np.zeros(0, dtype=int)
    g = gain[np.arange(n), side]
    moved = (X[1:, idx] - x[idx]).reshape(n, 2, n)[np.arange(n), side, np.arange(n)] / step[idx]  # in steps
    use = g > 1e-12 * max(1.0, abs(qx))
    if not use.any():
        return x, True
    rate = np.where(use, g / np.where(use, np.abs(moved), 1.0), 0.0)  # gain per step moved
    v = np.where(use, np.sign(moved) * rate, 0.0)  # direction in steps per unit t
    with np.errstate(divide="ignore"):
        room = np.where(v > 0, (bhi[idx] - x[idx]) / (v * step[idx]),
                        np.where(v < 0, (blo[idx] - x[idx]) / (v * step[idx]), 0.0))
    t_sat = float(room[use].max())  # beyond it no parameter that moves q towards q0 can move further

    def at(t):
        y = x.copy()
        y[idx] = np.clip(x[idx] + t * v * step[idx], blo[idx], bhi[idx])
        return y

    rem0 = abs(q0 - qx)
    ta, ra, tb, rb = 0.0, rem0, None, None  # last t short of q0, first t past it; remaining distance toward q0
    best_t, best_r = 0.0, rem0
    t = min(rem0 / float(np.sum(rate * rate)), t_sat)  # the linear estimate
    limit = False
    for _ in range(iters):
        qy = float(quantity_values(problem, at(t), name)[0])
        if not np.isfinite(qy):
            t = 0.5 * (ta + t)
            continue
        r = toward * (q0 - qy)
        if abs(r) < best_r:
            best_t, best_r = t, abs(r)
        if abs(r) <= 0.01 * dq:
            break
        if r > 0:
            if t >= t_sat:
                limit = r > CONSTRAINT_TOL * dq  # every parameter that moves q towards q0 is at its polish limit
                break
            if tb is None and r >= ra:  # no progress along the line: q turns back here
                break
            t_new = (t + r * (t - ta) / (ra - r)) if tb is None else (t + r * (tb - t) / (r - rb))
            ta, ra = t, r
            t = min(t_new, t_sat) if tb is None else t_new
        else:
            tb, rb = t, r
            t = ta + ra * (tb - ta) / (ra - rb)
    return at(best_t), limit


def walk(problem, cfg, name, x_start, q_grid0, dq, stop, max_steps, direction, k0, first=0, back=False) -> dict:
    """One direction of a walk from x_start: grid indices k0 + direction * step for step = first .. max_steps, each
    polished from the previous point; stops after the first point above `stop`, at an invalid point, or at a closed end
    of the range of q: for a quantity fixed directly, a grid value beyond its range is replaced by the range end itself,
    polished there, and the walk stops. For a quantity held by the penalty, a point whose
    polish misses the constraint (|q - q0| > 0.1 dq) goes to `_towards`, once: where it finds that q cannot reach q0
    (a limit) the point is kept with met = False and the walk stops at the q it reached, a closed end; else the point
    is polished once more with both LM stops off (`NO_LM_STOPS`), from the start `_towards` moved towards q0 (never at
    a known limit, where a retry cannot help). If that misses too, the
    point is kept with met = False and stalled = True, nothing is closed, and the walk goes on from its last point
    that met the constraint, up to `MAX_STALLS` stalled points in a row (then it ends, open). A walk back (`back`, from a
    far end of a first walk, which lies above `stop`) stops above `stop` only once it has been at or below it. Returns
    {"points": {grid index: (-ln L without penalty, x, q, constraint met, stalled)}, "end": None or {"q", "neg_log_l",
    "reason"} (the closed end it stopped at), "nevals", "retries" (points polished a second time)}. A pure function of
    its arguments: the unit of work of the pool (`profiles_for_families`)."""
    out = {}
    nevals = retries = stalls = 0
    end = None
    seen = not back
    lim = _direct_range(problem, name)
    cfg_retry = None
    x = np.array(x_start, dtype=float)
    for step in range(first, max_steps + 1):
        k = k0 + direction * step
        q0 = q_grid0 + k * dq
        at_end = lim is not None and not lim[0] <= q0 <= lim[1]
        if at_end:
            q0 = lim[0] if q0 < lim[0] else lim[1]
            prev = out.get(k - direction)
            if prev is not None and prev[2] == q0:  # the last grid value was the range end itself
                end = {"q": q0, "neg_log_l": prev[0], "reason": RANGE_END}
                break
        view, x0 = _held_problem(problem, name, q0, dq, x)
        blocked = getattr(view, "limit", None) is not None  # a strength reaches its polish limit before q0
        if blocked and abs(view.limit - quantity_values(problem, x, name)[0]) <= 1e-9 * max(1.0, abs(q0)):
            prev = out.get(k - direction)  # the last point already sits at the limit
            end = {"q": float(view.limit), "neg_log_l": prev[0] if prev else float(problem.cost(x[None])[0]),
                   "reason": NOT_MET}
            break
        pr = _polish(view, x0, cfg)
        nevals += pr.nevals
        f = float(problem.cost(np.asarray(pr.x)[None])[0])
        q = float(quantity_values(problem, pr.x, name)[0])
        met = abs(q - q0) <= CONSTRAINT_TOL * dq
        if f >= S.WORST:
            break
        if at_end:
            end = {"q": q, "neg_log_l": f, "reason": RANGE_END}
            break
        if blocked:  # held by scaling: q cannot pass the strength's polish limit, a closed end polished there
            end = {"q": q, "neg_log_l": f, "reason": NOT_MET}
            break
        limit = False
        if not met:  # a limit, or a stalled polish: one search towards q0 tells which and gives the retry's start
            xs, limit = _towards(problem, name, pr.x, q0, dq)
        if not met and not limit:  # stalled: once more with the LM stops off (no retry at a limit)
            retries += 1
            if cfg_retry is None:
                cfg_retry = dict(cfg, search={**cfg["search"], **NO_LM_STOPS})
            pr2 = _polish(view, xs, cfg_retry)
            nevals += pr2.nevals
            f2 = float(problem.cost(np.asarray(pr2.x)[None])[0])
            if f2 < S.WORST:
                pr, f = pr2, f2
                q = float(quantity_values(problem, pr.x, name)[0])
                met = abs(q - q0) <= CONSTRAINT_TOL * dq
        if not met and limit:  # q cannot reach q0: a closed end
            out[k] = (f, np.asarray(pr.x, float), q, False, False)
            end = {"q": q, "neg_log_l": f, "reason": NOT_MET}
            break
        out[k] = (f, np.asarray(pr.x, float), q, met, not met)
        if not met:  # stalled: not a value of the profile, nothing closed; on from the last point that met
            stalls += 1
            if stalls >= MAX_STALLS:
                break
            continue
        stalls = 0
        x = np.asarray(pr.x, float)
        seen = seen or f <= stop
        if f > stop and seen:
            break
    return {"points": out, "end": end, "nevals": nevals, "retries": retries}


def _walk_task(args: tuple) -> dict:
    """Picklable unit of work for the search's pool: args = the arguments of `walk`."""
    return walk(*args)


def _joined(ks, a: int, b: int) -> bool:
    """Whether entries a and b of the envelope are neighbouring grid values (no gap between the walks there)."""
    return ks is None or abs(int(ks[b]) - int(ks[a])) == 1


def _interval(grid, env, thr, ks=None, ends=(None, None)):
    """[lo, hi] where the envelope stays at or below thr around its minimum (linear interpolation at the crossings),
    and whether each side is open: the walked grid ended below thr, or reached a gap between the walks (grid indices
    `ks` not consecutive), across which nothing is interpolated. `ends` = (low, high) closed ends of the range beyond
    the grid's ends, each None or (q, -ln L) (`_closed_ends`): a side that reaches one is closed, at the end's q when
    its -ln L is at or below thr, else interpolated between the last grid value and the end."""
    i = int(np.argmin(env))
    if env[i] > thr:
        return None, None, False, False
    n = len(env)
    lo_i = i
    while lo_i > 0 and _joined(ks, lo_i - 1, lo_i) and env[lo_i - 1] <= thr:
        lo_i -= 1
    hi_i = i
    while hi_i < n - 1 and _joined(ks, hi_i, hi_i + 1) and env[hi_i + 1] <= thr:
        hi_i += 1

    def cross(a, b):
        fa, fb = env[a], env[b]
        if fb == fa:
            return grid[a]
        return grid[a] + (thr - fa) / (fb - fa) * (grid[b] - grid[a])

    def to_end(a, end):
        qe, fe = end
        if fe <= thr:
            return qe
        return grid[a] + (thr - env[a]) / (fe - env[a]) * (qe - grid[a])

    open_lo = lo_i == 0 or not _joined(ks, lo_i - 1, lo_i)
    open_hi = hi_i == n - 1 or not _joined(ks, hi_i, hi_i + 1)
    if lo_i == 0 and ends[0] is not None:
        lo, open_lo = to_end(0, ends[0]), False
    else:
        lo = grid[lo_i] if open_lo else cross(lo_i, lo_i - 1)
    if hi_i == n - 1 and ends[1] is not None:
        hi, open_hi = to_end(n - 1, ends[1]), False
    else:
        hi = grid[hi_i] if open_hi else cross(hi_i, hi_i + 1)
    return float(lo), float(hi), open_lo, open_hi


def _lower_envelope(walks: list[dict]) -> tuple[list[int], np.ndarray]:
    """(grid indices, envelope) over the grid indices where a walk's point met the constraint (|q - q0| <= 0.1 dq): at
    each the lowest such point. A point that missed it is not a value of the profile at its grid value (the model
    cannot take q0 there, or its polish stalled) and is left out; it ended its walk at a closed end, or was a stalled
    polish that left a gap (`walk`, `_closed_ends`)."""
    ks = sorted({k for w in walks for k, p in w["points"].items() if p[3]})
    env = [min(w["points"][k][0] for w in walks if k in w["points"] and w["points"][k][3]) for k in ks]
    return ks, np.array(env, dtype=float)


def _closed_ends(walks: list[dict], grid: np.ndarray, dq: float) -> dict:
    """{"lo", "hi"}: the closed ends of the range beyond the envelope's grid, each None or {"q", "neg_log_l", "reason"}:
    of the ends the walks stopped at (`walk`: a range end, or a point where q cannot reach q0; a stalled polish is never
    an end) those that lie beyond the envelope's last grid value on that side (within half a grid step), the farthest
    out, its q not inside the grid. An end inside the grid (a walk stuck where another one went on) closes nothing."""
    if grid.size == 0:
        return {"lo": None, "hi": None}
    lo = [w["ends"]["lo"] for w in walks if w["ends"].get("lo") and w["ends"]["lo"]["q"] <= grid[0] + 0.5 * dq]
    hi = [w["ends"]["hi"] for w in walks if w["ends"].get("hi") and w["ends"]["hi"]["q"] >= grid[-1] - 0.5 * dq]
    a = min(lo, key=lambda e: (e["q"], e["neg_log_l"])) if lo else None
    b = max(hi, key=lambda e: (e["q"], -e["neg_log_l"])) if hi else None
    return {"lo": dict(a, q=min(float(a["q"]), float(grid[0]))) if a else None,
            "hi": dict(b, q=max(float(b["q"]), float(grid[-1]))) if b else None}


def ratios(envelope: dict, mode_68: dict | None, h: float) -> tuple[float | None, float | None, list[str]]:
    """UNC-8: R1 = envelope width / the per-mode 68 % width (printed, not flagged), R2 = R1 / sqrt(2 h) (flag above
    1.5); decision #30: R1 is at least sqrt(2 h) for one Gaussian basin, so a flag on it would always be on. None
    without a per-mode interval or an envelope."""
    if not mode_68 or envelope.get("lo") is None or mode_68.get("p16") is None or mode_68.get("p84") is None:
        return None, None, []
    w = float(mode_68["p84"]) - float(mode_68["p16"])
    if w <= 0:
        return None, None, []
    r1 = (float(envelope["hi"]) - float(envelope["lo"])) / w
    r2 = r1 / math.sqrt(2.0 * h)
    flags = []
    if r2 > 1.5:
        flags.append("envelope wider than one Gaussian basin gives")
    return r1, r2, flags


def local_minima(grid, env, depth=0.5, ks=None, closed=(False, False)):
    """Local minima of the envelope as (q, -ln L): a point, or the middle of a run of equal values (a plateau), from
    which the envelope rises by at least `depth` nats on each side before it falls below the point's value. A side that
    reaches an open grid end (the walk ended there) or a gap between the walks (`ks`, as in `_interval`) first does not
    count, so the last point of a walk cut short while still falling is no minimum; `closed` = (low end, high end) marks
    ends of the parameter range (the inclination at 0 or 90 degrees), which count."""
    n = len(env)

    def rises(t, step, end_closed, v):
        while True:
            nxt = t + step
            if nxt < 0 or nxt >= n:
                return end_closed
            if not _joined(ks, t, nxt) or env[nxt] < v:
                return False
            if env[nxt] >= v + depth:
                return True
            t = nxt

    out = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and _joined(ks, j, j + 1) and env[j + 1] == env[i]:
            j += 1
        v = env[i]
        if rises(i, -1, closed[0], v) and rises(j, 1, closed[1], v):
            out.append((float(grid[(i + j) // 2]), float(v)))
        i = j + 1
    return out


_SIDE = {1: "hi", -1: "lo"}  # walk direction -> side of the range (grid steps are positive)


def _run_walks(tasks: list[tuple], pool) -> list[dict]:
    if pool is None:  # the same thread counts as in a worker: BLAS results depend on them (search._polish_many)
        with S.single_threaded():
            return [_walk_task(t) for t in tasks]
    return list(pool.map(_walk_task, tasks))


def profiles_for_families(problem, cfg, fams: list[dict], level: float, h: float, quantities=None,
                          mode_intervals: list | None = None, workers: int = 1) -> list[dict]:
    """UNC-5 and UNC-8 for each family of `fams` (entries of families.json with node "x"); `mode_intervals[i]`: the
    intervals.json quantities of family i's best mode, or None. Returns one {quantity: entry} per family (FORMATS.md
    section 6, profiles.json): `profile_entries` of `profile_walks`."""
    return profile_entries(profile_walks(problem, cfg, fams, h, quantities, workers), len(fams), level, h,
                           mode_intervals)


def profile_entries(plans: list[dict], n_fams: int, level: float, h: float, mode_intervals: list | None = None
                    ) -> list[dict]:
    """The profiles.json entries of the walks `plans` (`profile_walks`), one {quantity: entry} per family. The MCMC
    enters only here (`mode_intervals`), not the walks, so checkpointed walks serve a later MCMC."""
    mode_intervals = mode_intervals if mode_intervals is not None else [None] * n_fams
    out = [{} for _ in range(n_fams)]
    for pl in plans:
        out[pl["fam"]][pl["name"]] = _entry(pl, level, h, mode_intervals[pl["fam"]])
    return out


def profile_walks(problem, cfg, fams: list[dict], h: float, quantities=None, workers: int = 1) -> list[dict]:
    """The walks of every family of `fams` and quantity: one plan per (family, quantity) with its walks.

    Every direction of every walk (family, quantity, start node, direction) is an independent task; with workers > 1
    the tasks go to the search's pool (`search.make_pool`, single-threaded workers), else they run in this process under
    `search.single_threaded`. The walks back from the far ends of a one-node family's first walk need that walk, so
    they are a second batch on the same pool. Tasks are mapped in a fixed order and each is a pure function of its
    arguments, so the result does not depend on the number of workers (V6d)."""
    uc = cfg.get("uncertainty") or {}
    steps = dict(DEFAULT_STEPS)
    steps.update(uc.get("profile_steps") or {})
    max_steps = int(uc.get("profile_max_steps", 40))
    plans, tasks, owner = [], [], []  # owner[t] = (plan, walk, direction) of task t
    for fi, fam in enumerate(fams):
        nodes = [np.asarray(n["x"], dtype=float) for n in fam["nodes"]]
        fnodes = np.array([float(n["neg_log_l"]) for n in fam["nodes"]])
        best = int(np.argmin(fnodes))
        f_best = float(fnodes[best])
        names = list(quantities) if quantities is not None else profile_names(problem, cfg, nodes[best])
        for name in names:
            if _base(name)[0] == "teff_K" and getattr(problem, "iteff", None) is None:
                continue
            qn = quantity_values(problem, np.array(nodes), name)
            q_best = float(qn[best])
            dq = grid_step(name, q_best, steps, lambda: own_scale(problem, nodes[best], name))
            far = int(np.argmax(np.abs(qn - q_best)))
            pl = {"fam": fi, "name": name, "qn": qn, "fnodes": fnodes, "f_best": f_best, "q_best": q_best, "dq": dq,
                  "stop": f_best + max(2.0 * h, 10.0), "walks": [], "back": far == best}
            plans.append(pl)
            for start in ([best] if far == best else [best, far]):
                q_s = float(quantity_values(problem, nodes[start], name)[0])
                k0 = int(round((q_s - q_best) / dq))
                nd = fam["nodes"][start]
                pl["walks"].append({"points": {}, "ends": {"lo": None, "hi": None}, "nevals": 0, "retries": 0,
                                    "start": {"role": "best" if start == best else "farthest node",
                                              "node": nd.get("node", start), "mode": nd.get("mode"), "k": k0}})
                for direction, first in ((1, 0), (-1, 1)):  # the start's own grid value once
                    tasks.append((problem, cfg, name, nodes[start], q_best, dq, pl["stop"], max_steps, direction, k0,
                                  first))
                    owner.append((len(plans) - 1, len(pl["walks"]) - 1, direction))
    w = max(1, int(workers))
    pool = S.make_pool(min(w, len(tasks))) if w > 1 and len(tasks) > 1 else None
    try:
        for t, r in zip(owner, _run_walks(tasks, pool)):
            wk = plans[t[0]]["walks"][t[1]]
            wk["points"].update(r["points"])  # the two directions have disjoint grid indices
            wk["nevals"] += r["nevals"]
            wk["retries"] += r["retries"]
            wk["ends"][_SIDE[t[2]]] = r["end"]
        tasks, owner = [], []  # one node: walk back from each far end of the first walk (points that met)
        for pi, pl in enumerate(plans):
            pts = {k: p for k, p in pl["walks"][0]["points"].items() if p[3]}
            if not (pl["back"] and pts):
                continue
            for kk, direction in ((max(pts), -1), (min(pts), 1)):
                x = pts[kk][1]
                k0 = int(round((float(quantity_values(problem, x, pl["name"])[0]) - pl["q_best"]) / pl["dq"]))
                tasks.append((problem, cfg, pl["name"], x, pl["q_best"], pl["dq"], pl["stop"], max_steps, direction,
                              k0, 0, True))
                owner.append((pi, direction, k0))
        for (pi, direction, k0), r in zip(owner, _run_walks(tasks, pool) if tasks else []):
            ends = {"lo": None, "hi": None, _SIDE[direction]: r["end"]}
            plans[pi]["walks"].append({"points": r["points"], "ends": ends, "nevals": r["nevals"],
                                       "retries": r["retries"],
                                       "start": {"role": "walk back", "node": None, "mode": None, "k": k0}})
    finally:
        if pool is not None:
            pool.shutdown()
    return plans


def provisional_flags(name: str) -> list[str]:
    """The flags a quantity carries whatever its walks did: `QUANTILE_STEP` for the B quantiles (decision #39)."""
    return [QUANTILE_STEP] if name.split("[")[0] in ("B10_MG", "B50_MG", "B90_MG") else []


def _entry(pl: dict, level: float, h: float, mode_intervals: dict | None) -> dict:
    """The profiles.json entry of one family and quantity from its walks."""
    name, walks, f_best, q_best, dq = pl["name"], pl["walks"], pl["f_best"], pl["q_best"], pl["dq"]
    ks, env = _lower_envelope(walks)
    grid = np.array([q_best + k * dq for k in ks])
    ends = _closed_ends(walks, grid, dq)
    end_qf = tuple(None if ends[s] is None else (ends[s]["q"], ends[s]["neg_log_l"]) for s in ("lo", "hi"))
    fref = min(f_best, float(env.min())) if env.size else f_best
    ints = {}
    for key, thr in (("delta_0.5", fref + 0.5), ("delta_2", fref + 2.0), ("family_level", level)):
        lo, hi, olo, ohi = _interval(grid, env, thr, ks, end_qf) if env.size else (None, None, False, False)
        ints[key] = {"lo": lo, "hi": hi, "open_lo": olo, "open_hi": ohi}
    # UNC-8 envelope: nodes, grid values that met the constraint, and the closed ends at the q they reached
    below = ([float(q) for q, f in zip(pl["qn"], pl["fnodes"]) if f <= level] +
             [float(g) for g, f in zip(grid, env) if f <= level] +
             [e["q"] for e in ends.values() if e is not None and e["neg_log_l"] <= level])
    envelope = {"lo": min(below), "hi": max(below)} if below else {"lo": None, "hi": None}
    q = (mode_intervals or {}).get(name)
    mode_iv = ({"p16": q["p16"], "p84": q["p84"], "median": q.get("median")}
               if q and q.get("p16") is not None and q.get("p84") is not None else None)
    r1, r2, flags = ratios(envelope, mode_iv, h)
    note = None if mode_iv is not None else "no per-mode interval (no MCMC run, not converged, or a nuisance)"
    closed = (end_qf[0] is not None, end_qf[1] is not None)
    # "constraint not met" = a closed end of that kind in `ends`; "polish stalled" = a point left out without one
    not_met = any(e is not None and e["reason"] == NOT_MET for e in ends.values())
    stalled = any(len(p) > 4 and p[4] for w in walks for p in w["points"].values())
    return {"grid_step": dq, "grid": grid.tolist(), "envelope_profile": env.tolist(), "ends": ends,
            "walks": [_walk_json(w, q_best, dq) for w in walks],
            "intervals": ints, "envelope": envelope, "mode_68": mode_iv, "R1": r1, "R2": r2,
            "flags": flags + ([NOT_MET] if not_met else []) + ([STALLED] if stalled else []) + provisional_flags(name),
            "note": note,
            "minima": local_minima(grid, env, ks=ks, closed=closed), "nevals": sum(w["nevals"] for w in walks)}


def _walk_json(w: dict, q_best: float, dq: float) -> dict:
    """A walk as profiles.json writes it (FORMATS.md section 6): its start, per point (by grid index) the grid value
    q0, the q reached, -ln L, whether the constraint was met and whether the polish stalled there, the
    closed ends it stopped at, its evaluations and the number of points polished a second time."""
    pts = [{"k": int(k), "q0": q_best + k * dq, "q": float(p[2]), "neg_log_l": float(p[0]), "met": bool(p[3]),
            "stalled": bool(len(p) > 4 and p[4])} for k, p in sorted(w["points"].items())]
    return {"start": w["start"], "points": pts, "ends": w["ends"], "nevals": w["nevals"],
            "retries": int(w.get("retries", 0))}


def family_profiles(problem, cfg, fam: dict, level: float, h: float, quantities=None, mode_intervals=None,
                    workers: int = 1) -> dict:
    """UNC-5 and UNC-8 for one family (an entry of families.json with node "x"). `mode_intervals`: the intervals.json
    quantities of the family's best mode, or None. Returns {quantity: entry} (FORMATS.md section 6, profiles.json)."""
    return profiles_for_families(problem, cfg, [fam], level, h, quantities, [mode_intervals], workers)[0]


def sentences(name: str, fam_id: int, mode_id: int, entry: dict, h: float, converged: bool | None) -> tuple[str, str]:
    """The two sentences of the report, one per view (FORMATS.md section 6, report.txt)."""
    m = entry.get("mode_68")
    if m and m.get("median") is not None:
        conv = "converged" if converged else "not converged"
        s1 = (f"Mode {mode_id}: {name} = {m['median']:.4g} +{m['p84'] - m['median']:.3g} "
              f"-{m['median'] - m['p16']:.3g} (68 %, per mode, {conv}).")
    else:
        s1 = f"Mode {mode_id}: {name}: no per-mode interval."
    iv = entry["intervals"]["family_level"]
    mins = [q for q, f in entry["minima"]]
    fm = [f for q, f in entry["minima"]]
    d = (max(fm) - min(fm)) if len(fm) > 1 else 0.0
    if iv["lo"] is None:
        s2 = f"Family {fam_id}: {name}: no profile point below the family level."
    else:
        s2 = (f"Family {fam_id}: {name} between {iv['lo']:.4g} and {iv['hi']:.4g} within {h:.3g} nats of the best "
              f"(profile, {len(entry['walks'])} walks); minima at {', '.join(f'{q:.4g}' for q in mins) or '-'}"
              f"{f', {d:.2g} nats apart' if len(mins) > 1 else ''}.")
    return s1, s2
