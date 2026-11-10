"""Error bars inside a mode: prior, MCMC, run-time convergence rules, hand-back, intervals, profiles (DESIGN.md
section 4; SPEC.md UNC-1, UNC-2, UNC-3, UNC-5, UNC-6, PAR-4, REP-3, REP-4; VERIFICATION.md V13, V14,
V1l; FORMATS.md section 3 `[uncertainty]` and section 6 `intervals.json`).

Like `search.py`, this module sees the fitting problem only through the duck-typed `search.CostProblem` protocol
(cost, residuals, canonical, names, free, lo, hi, step, value, field_slice; optional outside_fraction, accept_start,
fingerprint_quantities) plus two optional members of its own:
- `derived(x) -> {name: float or (P,) array}` (FORMATS.md section 7; array entries are reported as name[0], name[1],
  ...); `derived_batch(X) -> {name: (K,) or (K, P)}` is used instead when present;
- `canonical_batch(X) -> (K, dim)` is used instead of a loop over `canonical` when present;
- `aq_penalty_in_cost = True` when `cost` already contains the gauge penalty (ln|aq|)^2 / (2 x 0.1^2), so that
  `log_prior` does not count it twice (default False: the prior carries it).

Steps:
0. `log_prior` / `draw_prior`: the start distribution of `[search.starts]` (DESIGN.md section 3) as a density in the
   sampling coordinates (DESIGN.md section 4, step 0), see `log_prior`.
1. `gauss_newton_cov`: (J^T J)^-1 from forward differences of `residuals` (step 1e-3 x step), ridge if singular;
   fallback: central-difference Hessian of `cost` when `residuals` raises NotImplementedError.
2. `run_mcmc`: `uncertainty.ensembles` ensembles of walkers over the sampling coordinates of `make_chart` (the
   observer frame of the first phase when the field is in vectors with md free, `_ObserverChart`; else the free
   parameters themselves), each with its own seed (`SeedSequence(seed).spawn`), nwalkers = max(2 d + 2, 16) (even; d
   the number of sampling coordinates), walkers started at mode + 3 x L z (L the Cholesky factor of the Gauss-Newton
   covariance pulled back to the sampling coordinates, z standard normal; invalid points redrawn; each start width
   capped at 0.25 x the width of the polish limits or of the chart's coordinate range: directions the data do not
   constrain), moves `ReflectDEMove` (0.9; DE proposals reflected at the hard bounds of the prior) and `SnookerMove`
   (0.1; a corrected snooker move, see its docstring), log-probability -cost + log_prior + log Jacobian of the chart.
   The ensembles are stepped like emcee's EnsembleSampler (`_Ensemble`, same random-stream use, bit-identical chains
   for a row-independent cost) but in lock-step: every sub-step's proposals of ALL ensembles go to one batched `cost`
   call (`_run_lockstep`). Chunks of `CHUNK` steps; after each chunk the V14 rules (`convergence.check_v14`; the
   autocorrelation time over the sampling coordinates and the free parameters) are applied; the run stops when all
   pass or when the next chunk would exceed `uncertainty.max_evals` (one last partial chunk; default 6e5, decision
   #20; 0 = `MAX_EVALS_DEFAULT`, the same number, noted in `McmcResult.notes`). `McmcResult.samples` holds the free
   parameters. Declared nuisance (`nuisance_names`, decision #18; one phase): the orientation coordinates are sampled
   but left out of the V14 verdict.
3. V14f (`_handback_candidates`): the best 3 distinct samples by -ln L are polished with `search.polish`; those that
   end better than the mode by more than the value tolerance are returned in `McmcResult.better_points`.
   `uncertainty_for_modes` hands the best of them back (re-polish, update the mode, re-run the MCMC), at most
   `MAX_HANDBACKS` = 3 times per mode, then re-sorts the mode table.
4. Intervals (`McmcResult.intervals`): canonical form per sample, median, 16/84 and 2.5/97.5 % points, half-widths
   (labelled PER_MODE_LABEL, UNC-1; the tempered half-widths were dropped, decision #24), flags: "not converged" (per quantity, with its own failed V14
   rules; no numbers are reported for it, UNC-2, decision #17), "orientation not determined by one phase" and
   "nuisance, not required" (declared nuisance, no numbers, decision #18), "touches a range limit" (REP-3), "not
   constrained by the data" (REP-4, KL - its plug-in bias (filled bins - 1) / (2 n_eff) < 0.1 nat, see `_kl_hist`;
   decision #19), "u reached 0".
5. `profile`: -ln L profile of one parameter, the others re-optimised with `search.polish` (UNC-5, V1l).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import emcee
import numpy as np

from . import atomic
from . import convergence as conv
from . import search as S

WORST = S.WORST
MAX_EVALS_DEFAULT = 600_000  # when uncertainty.max_evals = 0: the default of defaults.toml (decision #20)
CHUNK = 200  # steps per ensemble between two V14 checks
MAX_HANDBACKS = 3  # per mode (UNC-3)
VALUE_TOL_DEFAULT = 2.0  # nats, the floor of the value tolerance (DESIGN.md section 3)
AQ_WIDTH = 0.1  # width of the Gaussian on ln|aq| (DESIGN.md section 4, step 0)
START_SPREAD = 3.0  # walkers start within START_SPREAD x the Gauss-Newton widths (V14a)
KL_MIN = 0.1  # nats (DESIGN.md section 4, step 6)
KL_BINS = 20
N_PRIOR_DRAWS = 10_000
LIMIT_FRAC = 0.01  # REP-3
MAX_TRIES = 100
COST_CHUNK = 8  # rows per cost call of the sampler: fixed, so the chains do not depend on the pool
DE_WEIGHT = 0.9  # weight of the (reflecting) DE move; the snooker move takes the rest (measured on the toy star)

NOT_CONVERGED = "not converged"
NOT_CONSTRAINED = "not constrained by the data"
TOUCHES_LIMIT = "touches a range limit"
U_ZERO = "u reached 0"
PER_MODE_LABEL = "per mode: conditional on this basin"  # UNC-1 (decisions #29, #24)
ORIENTATION = "orientation not determined by one phase"
NUISANCE = "nuisance, not required"


# ---------------------------------------------------------------------------------------------------------------------
# 0. Prior


@dataclass
class _PriorSpec:
    groups: dict[str, list[int]]  # field groups whose components are all free
    uniform: list[int]  # free parameters with a uniform prior on [lo, hi]
    lo: np.ndarray
    hi: np.ndarray
    st: dict  # [search.starts]
    outside_max: float
    aq_in_cost: bool


def _prior_spec(problem, cfg: dict) -> _PriorSpec:
    free = np.asarray(problem.free, dtype=bool)
    groups = {}
    in_group: set[int] = set()
    for g, idx in S.field_groups(problem).items():
        if g == "other" or not all(free[i] for i in idx):
            continue  # partly fixed vectors: their free components are treated as uniform in [lo, hi]
        groups[g] = list(idx)
        in_group.update(idx)
    uniform = [i for i in range(problem.dim) if free[i] and i not in in_group]
    st = dict((cfg.get("search", {}) or {}).get("starts", {}) or {})
    return _PriorSpec(groups, uniform, np.asarray(problem.lo, dtype=float), np.asarray(problem.hi, dtype=float), st,
                      float(st.get("outside_max", math.inf)), bool(getattr(problem, "aq_penalty_in_cost", False)))


def _norm(X: np.ndarray, idx: list[int]) -> np.ndarray:
    return np.sqrt(np.sum(X[:, idx] ** 2, axis=1))


def _log_density(spec: _PriorSpec, X: np.ndarray) -> np.ndarray:
    """The density of the start distribution (without the prior cut), up to a constant, for full vectors X (K, dim)."""
    lp = np.zeros(X.shape[0])
    st = spec.st
    with np.errstate(divide="ignore", invalid="ignore"):
        for g, idx in spec.groups.items():
            if g == "md":
                n = _norm(X, idx)
                lo, hi = float(st["md"][0]), float(st["md"][1])
                lp += np.where((n >= lo) & (n <= hi) & (n > 0), -2.0 * np.log(n), -np.inf)
            elif g == "mo":
                n = _norm(X, idx)
                lp += np.where((n > 0) & (n <= float(st["mo"])), -2.0 * np.log(n), -np.inf)
            elif g == "aq":
                n = _norm(X, idx)
                ln = np.log(n)
                # ln|aq| ~ N(0, 0.1^2), direction uniform: density in R^3 = N(ln n) / (4 pi n^3)
                term = -3.0 * ln if spec.aq_in_cost else -0.5 * (ln / AQ_WIDTH) ** 2 - 3.0 * ln
                lp += np.where(n > 0, term, -np.inf)
            elif g == "bq":
                lp += np.where(np.abs(X[:, idx[0]]) <= float(st["bq"]), 0.0, -np.inf)
            elif g == "bo":
                lp += np.where(np.abs(X[:, idx[0]]) <= float(st["mo"]), 0.0, -np.inf)
            elif g == "c":
                lp += np.where(_norm(X, idx) <= float(st["c"]), 0.0, -np.inf)
            elif g == "u":
                a = np.abs(X[:, idx[0]])
                lp += np.where((a >= float(st["u"][0])) & (a <= float(st["u"][1])), 0.0, -np.inf)
        for i in spec.uniform:
            lo, hi = spec.lo[i], spec.hi[i]
            lp += np.where((X[:, i] >= lo) & (X[:, i] <= hi), 0.0, -np.inf)
    return lp


def _cut(problem, spec: _PriorSpec, X: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """The prior cut (outside fraction) and problem.accept_start, applied to the rows with a finite density."""
    of = getattr(problem, "outside_fraction", None)
    acc = getattr(problem, "accept_start", None)
    if of is None and acc is None:
        return lp
    lp = lp.copy()
    for k in np.flatnonzero(np.isfinite(lp)):
        if of is not None and float(np.max(np.atleast_1d(of(X[k])))) > spec.outside_max:
            lp[k] = -np.inf
        elif acc is not None and not acc(X[k]):
            lp[k] = -np.inf
    return lp


def log_prior(problem, cfg: dict, x: np.ndarray) -> float | np.ndarray:
    """Log density of the prior (DESIGN.md section 4, step 0) at full parameter vectors, up to a constant.

    x: (dim,) -> float, or (K, dim) -> (K,). Free parameters only (fixed ones contribute nothing). With a field slice
    (names md_*, bq, aq_*, mo_*, bo, c_*, u), from `cfg["search"]["starts"]`:
    md: -2 ln|md| on the shell [md_lo, md_hi] (uniform strength x uniform direction); bq: uniform in [-bq, bq];
    aq: ln|aq| ~ N(0, 0.1^2) with a uniform direction, i.e. -(ln|aq|)^2 / (2 x 0.01) - 3 ln|aq| (the second term is
    the volume element, so that `draw_prior` and this density agree; without it ln|aq| would be centred at 0.03);
    mo: -2 ln|mo| in the ball of radius mo; bo (dqo_aligned): uniform in [-mo, mo]; c: uniform in the ball of radius
    c; u: |u| uniform in [u_lo, u_hi] (the start range and its mirror). Any other free parameter: uniform in
    [lo, hi]. -inf outside, where the outside fraction exceeds `outside_max`, or where `problem.accept_start` fails.
    """
    X = np.asarray(x, dtype=float)
    one = X.ndim == 1
    X = np.atleast_2d(X)
    spec = _prior_spec(problem, cfg)
    lp = _cut(problem, spec, X, _log_density(spec, X))
    return float(lp[0]) if one else lp


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal((n, 3))
    nv = np.linalg.norm(v, axis=1)
    bad = nv < 1e-12
    while bad.any():
        v[bad] = rng.standard_normal((int(bad.sum()), 3))
        nv = np.linalg.norm(v, axis=1)
        bad = nv < 1e-12
    return v / nv[:, None]


def _draw_raw(problem, spec: _PriorSpec, rng: np.random.Generator, n: int) -> np.ndarray:
    X = np.tile(np.asarray(problem.value, dtype=float), (n, 1))
    st = spec.st
    for g in ("md", "bq", "aq", "mo", "bo", "c", "u"):  # fixed order: reproducible streams
        idx = spec.groups.get(g)
        if not idx:
            continue
        if g == "md":
            X[:, idx] = rng.uniform(float(st["md"][0]), float(st["md"][1]), n)[:, None] * _unit(rng, n)
        elif g == "bq":
            X[:, idx[0]] = rng.uniform(-float(st["bq"]), float(st["bq"]), n)
        elif g == "aq":
            X[:, idx] = np.exp(AQ_WIDTH * rng.standard_normal(n))[:, None] * _unit(rng, n)
        elif g == "mo":
            X[:, idx] = rng.uniform(0.0, float(st["mo"]), n)[:, None] * _unit(rng, n)
        elif g == "bo":
            X[:, idx[0]] = rng.uniform(-float(st["mo"]), float(st["mo"]), n)
        elif g == "c":
            X[:, idx] = (float(st["c"]) * rng.uniform(0.0, 1.0, n) ** (1.0 / 3.0))[:, None] * _unit(rng, n)
        elif g == "u":
            X[:, idx[0]] = rng.uniform(float(st["u"][0]), float(st["u"][1]), n) * rng.choice([-1.0, 1.0], n)
    for i in spec.uniform:
        if not (np.isfinite(spec.lo[i]) and np.isfinite(spec.hi[i])):
            raise ValueError(f"parameter {problem.names[i]!r} is free but has no finite [lo, hi] to draw from")
        X[:, i] = rng.uniform(spec.lo[i], spec.hi[i], n)
    return X


def draw_prior(problem, cfg: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """n direct draws (n, dim) from the prior of `log_prior` (V13, the KL flag of REP-4). Draws that fail the prior
    cut or `accept_start` are redrawn (at most 100 rounds)."""
    spec = _prior_spec(problem, cfg)
    out = _draw_raw(problem, spec, rng, n)
    ok = np.isfinite(_cut(problem, spec, out, np.zeros(n)))
    for _ in range(MAX_TRIES):
        if ok.all():
            return out
        bad = np.flatnonzero(~ok)
        new = _draw_raw(problem, spec, rng, bad.size)
        out[bad] = new
        ok[bad] = np.isfinite(_cut(problem, spec, new, np.zeros(bad.size)))
    raise RuntimeError(f"draw_prior: {int((~ok).sum())} draws still fail the prior cut after {MAX_TRIES} rounds")


# ---------------------------------------------------------------------------------------------------------------------
# 1. Gauss-Newton covariance


def _free_idx(problem) -> np.ndarray:
    return np.flatnonzero(np.asarray(problem.free, dtype=bool))


def _pd_inverse(A: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric matrix that is made positive definite: eigenvalues below 1e-10 x the largest are raised
    to that value (a ridge on the singular directions)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    top = max(float(np.max(np.abs(w))), 1e-300)
    w = np.maximum(w, 1e-10 * top)
    return (V / w) @ V.T


def gauss_newton_cov(problem, x: np.ndarray, cfg: dict | None = None, rel_step: float = 1e-3) -> np.ndarray:
    """(nfree, nfree) covariance at x: (J^T J)^-1 with J the forward-difference Jacobian of `problem.residuals`
    (step rel_step x step per free parameter). Fallback when `residuals` raises NotImplementedError: the inverse of
    the central-difference Hessian of `cost` (step 1e-2 x step, one batched call). Singular or indefinite matrices get
    a ridge (`_pd_inverse`)."""
    x = np.asarray(x, dtype=float)
    idx = _free_idx(problem)
    h = rel_step * np.asarray(problem.step, dtype=float)[idx]
    try:
        r0 = np.asarray(problem.residuals(x), dtype=float)
        J = np.empty((r0.size, idx.size))
        for k, j in enumerate(idx):
            xk = x.copy()
            xk[j] += h[k]
            J[:, k] = (np.asarray(problem.residuals(xk), dtype=float) - r0) / h[k]
        return _pd_inverse(J.T @ J)
    except NotImplementedError:
        pass
    n = idx.size
    hh = 1e-2 * np.asarray(problem.step, dtype=float)[idx]
    rows = [x.copy()]
    pairs = []
    for a in range(n):
        for b in range(a, n):
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                xx = x.copy()
                xx[idx[a]] += sa * hh[a]
                xx[idx[b]] += sb * hh[b]
                rows.append(xx)
            pairs.append((a, b))
    f = np.asarray(problem.cost(np.array(rows)), dtype=float)
    H = np.empty((n, n))
    for k, (a, b) in enumerate(pairs):
        fpp, fpm, fmp, fmm = f[1 + 4 * k:5 + 4 * k]
        H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4.0 * hh[a] * hh[b])
    return _pd_inverse(H)


# ---------------------------------------------------------------------------------------------------------------------
# 2. MCMC


class SnookerMove(emcee.moves.RedBlueMove):
    """The snooker update of Ter Braak & Vrugt 2008 (Statistics and Computing 18, 435, section 3.2), as a red-blue
    move: for walker x, draw z, z1, z2 from the three other quarters of the ensemble; with e = (x - z) / |x - z|,
    propose x* = x + gamma (e . (z1 - z2)) e and accept with min(1, p(x*) / p(x) x (|x* - z| / |x - z|)^(d - 1)).

    Written here because emcee 3.1.6's DESnookerMove differs from the paper (it scales the step by |x - z|^(1/2) and
    uses the exponent (d - 1) / 2): on a 3-d Gaussian with 16 walkers and 2e4 steps it returned variances 15 % too
    small, this move returns them within 1 %."""

    def __init__(self, gamma: float = 1.7, **kwargs):
        kwargs["nsplits"] = 4
        super().__init__(**kwargs)
        self.gamma = float(gamma)

    def get_proposal(self, s, c, random):
        ns, nd = s.shape
        q = np.empty_like(s)
        lf = np.empty(ns)
        for i in range(ns):
            z, z1, z2 = (c[k][random.randint(len(c[k]))] for k in random.permutation(3))
            d = s[i] - z
            n = float(np.linalg.norm(d))
            if n == 0.0:
                q[i], lf[i] = s[i], 0.0
                continue
            e = d / n
            q[i] = s[i] + self.gamma * float(e @ (z1 - z2)) * e
            lf[i] = (nd - 1.0) * (math.log(max(float(np.linalg.norm(q[i] - z)), 1e-300)) - math.log(n))
        return q, lf


def _unc_cfg(cfg: dict) -> dict:
    return dict(cfg.get("uncertainty", {}) or {})


def _max_evals(cfg: dict) -> tuple[int, str | None]:
    m = int(_unc_cfg(cfg).get("max_evals", 0) or 0)
    if m <= 0:
        return MAX_EVALS_DEFAULT, (f"uncertainty.max_evals = 0: the default {MAX_EVALS_DEFAULT} cost evaluations per "
                                   "mode run")
    return m, None


def _fold(Q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Reflect the columns of Q (K, d) into [lo, hi] where both ends are finite (repeated reflection, i.e. folding with
    period 2 (hi - lo)); the other columns are returned unchanged."""
    Q = np.array(Q, dtype=float)
    w = hi - lo
    m = np.isfinite(w) & (w > 0)
    if m.any():
        t = np.mod(Q[:, m] - lo[m], 2.0 * w[m])
        Q[:, m] = lo[m] + np.where(t > w[m], 2.0 * w[m] - t, t)
    return Q


class ReflectDEMove(emcee.moves.DEMove):
    """emcee's DEMove (Ter Braak 2006; Nelson et al. 2014) with its proposals reflected into a box [lo, hi].

    Why this is still a valid Metropolis move: a DE proposal is y = x + d, d = gamma (c_a - c_b) drawn from the
    complementary walkers, independent of x and symmetric (emcee draws both orders of each pair). Folding a translation
    kernel with a symmetric step at the faces of a box keeps it symmetric (q(x -> y) sums s(z - x) over the mirror
    images z of y; the mirror images of x seen from y give the same terms), so the acceptance ratio is p(y) / p(x)
    as before. The box must CONTAIN the support of the target (a face inside the support would make the region beyond
    it unreachable): `run_mcmc` uses the hard bounds of the prior (`_prior_box`). Without the reflection a coordinate
    whose posterior is flat over its whole prior range loses 23 % of the DE proposals at its ends (u when one phase
    leaves the inclination undetermined, measured on the toy star)."""

    def __init__(self, lo, hi, **kwargs):
        super().__init__(**kwargs)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def get_proposal(self, s, c, random):
        q, f = super().get_proposal(s, c, random)
        return _fold(q, self.lo, self.hi), f


def _prior_box(problem, spec: _PriorSpec) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (lo, hi) of a box that contains the support of `log_prior` (inf where there is no hard bound): the
    uniform parameters' [lo, hi]; md, mo, c components within +- the shell or ball radius; bq, bo, u within +- their
    prior end; aq unbounded (Gaussian in ln|aq|)."""
    lo = np.full(problem.dim, -np.inf)
    hi = np.full(problem.dim, np.inf)
    st = spec.st
    rad = {"md": lambda: float(st["md"][1]), "mo": lambda: float(st["mo"]), "c": lambda: float(st["c"]),
           "bq": lambda: float(st["bq"]), "bo": lambda: float(st["mo"]), "u": lambda: float(st["u"][1])}
    for g, idx in spec.groups.items():
        if g in rad:
            r = rad[g]()
            lo[idx], hi[idx] = -r, r
    for i in spec.uniform:
        lo[i], hi[i] = spec.lo[i], spec.hi[i]
    return lo, hi


class _Chart:
    """The sampling coordinates z of the MCMC and the map to the problem's free parameters y. This one is the identity
    (z = y); `_ObserverChart` is the alternative. Interface: `names`, `dim` (of z), `lo`/`hi` (reflection box of z, a
    box that contains the support), `cap_width` (widths that cap the start spread), `to_free(Z) -> (Y, log Jacobian,
    valid)`, `from_free(Y, rng) -> Z`, `jacobian(z) -> dY/dz`."""

    kind = "identity"

    def __init__(self, problem, spec: _PriorSpec):
        self.idx = _free_idx(problem)
        self.free_names = [problem.names[i] for i in self.idx]
        self.names = list(self.free_names)
        self.nfree = self.idx.size
        self.dim = self.nfree
        blo, bhi = _prior_box(problem, spec)
        self.lo, self.hi = blo[self.idx], bhi[self.idx]
        plo, phi = S.polish_bounds(problem)
        self.cap_width = (phi - plo)[self.idx]

    def to_free(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return Z.copy(), np.zeros(Z.shape[0]), np.ones(Z.shape[0], dtype=bool)

    def from_free(self, Y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.atleast_2d(np.asarray(Y, dtype=float)).copy()

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """dY/dz (nfree, dim) by central differences (one-sided at the edge of the chart)."""
        z = np.asarray(z, dtype=float)
        y0 = self.to_free(z[None])[0][0]
        J = np.empty((self.nfree, self.dim))
        for j in range(self.dim):
            h = 1e-6 * max(1.0, abs(z[j]))
            Zp = np.array([z, z])
            Zp[0, j] += h
            Zp[1, j] -= h
            Y, _, ok = self.to_free(Zp)
            if ok[0] and ok[1]:
                J[:, j] = (Y[0] - Y[1]) / (2.0 * h)
            elif ok[0]:
                J[:, j] = (Y[0] - y0) / h
            else:
                J[:, j] = (y0 - Y[1]) / h
        return J


class _ObserverChart(_Chart):
    """Sampling coordinates in the observer frame of the first phase, for a field in vectors whose dipole md is free.

    Why: one phase fixes |md| and the angle beta between md and the line of sight n, while the
    rotation of md about n and the inclination (with md co-rotating) leave the spectra unchanged. In Cartesian md and u
    the posterior is therefore a thin curved shell (|md| = 12 +- 0.07 MG on the toy star) around a ring: a DE step,
    being a straight chord, leaves the shell (median +218 nats on the toy star), and the acceptance fell to 0.02-0.04
    whatever the number of walkers. In these coordinates the same posterior is a product of Gaussians and flat
    directions.

    Frame: e3 = n(u) = sqrt(1 - u^2) a + u b, with a = n(0) and b = n(1) read from `problem.forward.observers` (checked
    to follow this form at construction); e2 = a x b / |a x b| (perpendicular to n for every u); e1 = e2 x e3. A
    change of u is then a rotation about e2, under which the frame co-rotates. Coordinates: md -> (md_r = |md|,
    md_cosb = md . e3 / |md|, md_p, md_q) with the azimuth atan2(md . e2, md . e1) = atan2(md_q, md_p), i.e. the
    azimuth carried by an auxiliary 2-d vector (md_p, md_q) with a standard normal factor (its length is not a
    parameter of the model; it keeps the circle of azimuths free of a periodic boundary); every other free vector group
    (mo, c, aq) -> its components along (e1, e2, e3); scalars unchanged.

    Density: the map (md_r, md_cosb, azimuth) -> md has Jacobian md_r^2, the rotation of the other vectors 1, and
    (md_p, md_q) -> (length, azimuth) has Jacobian = length, which the standard normal factor integrates out:
    p(z) = p(y(z)) x md_r^2 x exp(-(md_p^2 + md_q^2) / 2) (up to a constant), so the marginal of y is the target."""

    kind = "observer"

    def __init__(self, problem, spec: _PriorSpec, a: np.ndarray, b: np.ndarray, iu_full: int):
        super().__init__(problem, spec)
        pos = {int(i): k for k, i in enumerate(self.idx)}
        self.md = [pos[i] for i in spec.groups["md"]]
        self.iu = pos.get(int(iu_full))
        self.u_fixed = float(np.asarray(problem.value, dtype=float)[iu_full])
        self.vecs = [[pos[i] for i in idx] for g, idx in spec.groups.items() if g != "md" and len(idx) == 3]
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        e2 = np.cross(self.a, self.b)
        self.e2 = e2 / np.linalg.norm(e2)
        self.dim = self.nfree + 1
        names = list(self.free_names)
        for k, s in zip(self.md, ("md_r", "md_cosb", "md_p")):
            names[k] = s
        for v in self.vecs:
            g = S._group_of(self.free_names[v[0]])
            for k, s in zip(v, ("e1", "e2", "e3")):
                names[k] = f"{g}_{s}"
        self.names = names + ["md_q"]
        lo = np.append(self.lo, -np.inf)
        hi = np.append(self.hi, np.inf)
        st = spec.st
        lo[self.md], hi[self.md] = [float(st["md"][0]), -1.0, -np.inf], [float(st["md"][1]), 1.0, np.inf]
        self.lo, self.hi = lo, hi
        cw = np.append(self.cap_width, 4.0)
        cw[self.md] = [float(st["md"][1]) - float(st["md"][0]), 2.0, 4.0]
        self.cap_width = cw

    def _u(self, Z: np.ndarray) -> np.ndarray:
        return Z[:, self.iu] if self.iu is not None else np.full(Z.shape[0], self.u_fixed)

    def _frame(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uc = np.clip(u, -1.0, 1.0)
        e3 = np.sqrt(1.0 - uc * uc)[:, None] * self.a + uc[:, None] * self.b
        return np.cross(self.e2, e3), e3

    def to_free(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        Y = Z[:, :self.nfree].copy()
        u = self._u(Z)
        e1, e3 = self._frame(u)
        r, cb, p = (Z[:, k] for k in self.md)
        q = Z[:, -1]
        rho = np.hypot(p, q)
        ok = (np.abs(u) <= 1.0) & (r > 0.0) & (np.abs(cb) <= 1.0) & (rho > 0.0)
        cbc = np.clip(cb, -1.0, 1.0)
        sb = np.sqrt(1.0 - cbc * cbc)
        rs = np.where(rho > 0.0, rho, 1.0)
        m1, m2, m3 = r * sb * p / rs, r * sb * q / rs, r * cbc
        Y[:, self.md] = m1[:, None] * e1 + m2[:, None] * self.e2 + m3[:, None] * e3
        for v in self.vecs:
            w = Z[:, v]
            Y[:, v] = w[:, 0:1] * e1 + w[:, 1:2] * self.e2 + w[:, 2:3] * e3
        with np.errstate(divide="ignore", invalid="ignore"):
            lj = np.where(ok, 2.0 * np.log(np.where(r > 0, r, 1.0)) - 0.5 * (p * p + q * q), -np.inf)
        return Y, lj, ok

    def from_free(self, Y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Y -> Z; the length of (md_p, md_q) is 1 without rng, else drawn from its Rayleigh distribution."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        K = Y.shape[0]
        Z = np.zeros((K, self.dim))
        Z[:, :self.nfree] = Y
        e1, e3 = self._frame(self._u(Y))
        md = Y[:, self.md]
        r = np.linalg.norm(md, axis=1)
        m1, m2, m3 = np.sum(md * e1, axis=1), md @ self.e2, np.sum(md * e3, axis=1)
        cb = np.where(r > 0, m3 / np.where(r > 0, r, 1.0), 1.0)
        al = np.arctan2(m2, m1)
        rho = np.ones(K) if rng is None else np.sqrt(-2.0 * np.log1p(-rng.random(K)))
        Z[:, self.md] = np.column_stack([r, np.clip(cb, -1.0, 1.0), rho * np.cos(al)])
        Z[:, -1] = rho * np.sin(al)
        for v in self.vecs:
            w = Y[:, v]
            Z[:, v] = np.column_stack([np.sum(w * e1, axis=1), w @ self.e2, np.sum(w * e3, axis=1)])
        return Z


def make_chart(problem, cfg: dict, kind: str = "auto") -> _Chart:
    """The sampling coordinates of `run_mcmc`: "identity", "observer" (`_ObserverChart`; ValueError if the problem
    does not allow it) or "auto" (observer when the problem has a field in vectors with md and u (free or fixed) and
    `problem.forward.observers` / `problem.thetas` whose first-phase observer direction has the form of
    `_ObserverChart`; identity otherwise)."""
    spec = _prior_spec(problem, cfg)
    if kind == "identity":
        return _Chart(problem, spec)
    why = None
    groups = S.field_groups(problem)
    fw = getattr(problem, "forward", None)
    obs = getattr(fw, "observers", None)
    th = getattr(problem, "thetas", None)
    if "md" not in spec.groups:
        why = "md is not a free vector"
    elif "u" not in groups:
        why = "no u in the field"
    elif obs is None or th is None:
        why = "the problem has no forward.observers / thetas"
    else:
        iu = int(groups["u"][0])
        base = np.asarray(problem.value, dtype=float)

        def n_of(u):
            X = np.tile(base, (len(u), 1))
            X[:, iu] = u
            return np.asarray(obs(np.asarray(th(X), dtype=float))[0], dtype=float)[:, 0, :]

        try:
            a, b = n_of([0.0, 1.0])
            ut = np.array([-0.9, -0.3, 0.4, 0.8])
            good = (abs(np.linalg.norm(a) - 1.0) < 1e-12 and abs(np.linalg.norm(b) - 1.0) < 1e-12
                    and abs(float(a @ b)) < 1e-12
                    and np.allclose(n_of(ut), np.sqrt(1.0 - ut * ut)[:, None] * a + ut[:, None] * b, atol=1e-12))
        except Exception as exc:  # noqa: BLE001 - any failure means: no observer chart
            good, why = False, f"observers failed: {exc}"
        if good:
            return _ObserverChart(problem, spec, a, b, iu)
        why = why or "the observer direction does not have the form sqrt(1 - u^2) a + u b"
    if kind == "observer":
        raise ValueError(f"make_chart: no observer chart: {why}")
    return _Chart(problem, spec)


def nuisance_names(problem, ch: _Chart, names: list[str]) -> tuple[list[str], list[str]]:
    """Declared nuisance (decision #18): (orientation quantities, nuisance sampling coordinates).

    The orientation quantities are the free parameters named in `problem.nuisance_coordinates` (optional; Problem
    declares the field-vector components and u for one phase), plus `incl_deg` when u is among them. They are reported
    without an interval, flagged ORIENTATION and NUISANCE. The nuisance coordinates are the chart's coordinates that
    carry them: in the observer chart the azimuth of md about the line of sight (md_p, md_q), u, and the e1, e2
    components of the other vectors (they rotate with that azimuth; e3, along the line of sight, is determined); in the
    identity chart the orientation parameters themselves. Both are left out of the V14 verdict
    (`convergence.check_v14(nuisance=...)`)."""
    declared = set(getattr(problem, "nuisance_coordinates", None) or ())
    orient = [n for n in names if n in declared]
    if not orient:
        return [], []
    if "u" in orient:
        orient.append("incl_deg")
    if ch.kind != "observer":
        return orient, [n for n in ch.names if n in declared]
    coords = []
    if any(S._group_of(n) == "md" for n in orient):
        coords += ["md_p", "md_q"]
    if "u" in orient:
        coords.append("u")
    for n in ch.names:
        g, _, tail = n.rpartition("_")
        if tail in ("e1", "e2") and any(S._group_of(o) == g for o in orient):
            coords.append(n)
    return orient, coords


def _cost_task(args) -> np.ndarray:
    """Picklable unit of a pooled cost call: args = (problem, rows)."""
    problem, X = args
    return np.asarray(problem.cost(X), dtype=float)


class _LogProb:
    """Vectorised log-probability over the sampling coordinates of a chart (default: the identity, i.e. the free
    parameters): -cost(X) (x 0 when data_off) + log_prior + the chart's log Jacobian; returns (K, 2) rows
    [log p, -ln L] (-ln L is kept per sample). Counts the rows passed to `cost`. Each cost call is split into chunks of
    COST_CHUNK rows, mapped over `pool` when given, else evaluated in turn: the same bits either way."""

    def __init__(self, problem, cfg: dict, data_off: bool = False, chart: _Chart | None = None, pool=None):
        self.problem = problem
        self.spec = _prior_spec(problem, cfg)
        self.idx = _free_idx(problem)
        self.base = np.asarray(problem.value, dtype=float).copy()
        self.data_off = data_off
        self.chart = chart if chart is not None else _Chart(problem, self.spec)
        self.nevals = 0
        self.ncalls = 0
        self.pool = pool

    def _cost(self, X: np.ndarray) -> np.ndarray:
        chunks = [X[i:i + COST_CHUNK] for i in range(0, X.shape[0], COST_CHUNK)]
        if self.pool is not None and len(chunks) > 1:
            return np.concatenate(list(self.pool.map(_cost_task, [(self.problem, c) for c in chunks])))
        return np.concatenate([_cost_task((self.problem, c)) for c in chunks])

    def full(self, Y: np.ndarray) -> np.ndarray:
        """Free parameters (K, nfree) -> full parameter vectors (K, dim)."""
        X = np.tile(self.base, (Y.shape[0], 1))
        X[:, self.idx] = Y
        return X

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        Y, lj, okc = self.chart.to_free(Z)
        X = self.full(Y)
        lp = np.full(Z.shape[0], -np.inf)
        if okc.any():
            lp[okc] = _cut(self.problem, self.spec, X[okc], _log_density(self.spec, X[okc]))
        nll = np.full(Z.shape[0], np.inf)
        ok = np.flatnonzero(np.isfinite(lp))
        if ok.size:
            c = self._cost(X[ok])
            self.nevals += ok.size
            self.ncalls += 1
            c = np.where(c < WORST, c, np.inf)
            nll[ok] = c
        like = np.where(np.isfinite(nll), 0.0 if self.data_off else -nll, -np.inf)
        good = np.isfinite(lp) & np.isfinite(like)
        val = np.where(good, like + np.where(good, lp + lj, 0.0), -np.inf)
        return np.column_stack([val, nll])


def _nwalkers(nd: int) -> int:
    """Walkers per ensemble: max(2 d + 2, 16), even."""
    n = max(2 * nd + 2, 16)
    return n + n % 2


def pool_width(problem, cfg: dict) -> int:
    """The most pool workers one mode's sampler keeps busy: the chunks of COST_CHUNK rows in its largest cost call,
    ensembles x walkers / 2 rows (a differential-evolution half-step of every ensemble)."""
    E = int(_unc_cfg(cfg).get("ensembles", 4))
    return -(-E * _nwalkers(make_chart(problem, cfg).dim) // 2 // COST_CHUNK)


def _initial_walkers(lpf: _LogProb, z0: np.ndarray, cov: np.ndarray, nwalkers: int, rng: np.random.Generator,
                     widths_cap: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """mode + spread x L z in the sampling coordinates (cov: their Gauss-Newton covariance, each width capped at
    widths_cap), invalid rows redrawn; the spread starts at START_SPREAD and is halved after every 20 rounds that leave
    invalid rows. Returns (positions (nwalkers, dim), spread used, the log-probability rows of the positions)."""
    sig = np.sqrt(np.diag(cov))
    scale = np.minimum(1.0, widths_cap / np.where(sig > 0, sig, 1.0))
    C = cov * np.outer(scale, scale)
    L = np.linalg.cholesky(C + 1e-14 * np.max(np.diag(C)) * np.eye(C.shape[0]))
    spread = START_SPREAD
    P = z0 + spread * rng.standard_normal((nwalkers, z0.size)) @ L.T
    vals = lpf(P)
    ok = np.isfinite(vals[:, 0])
    for t in range(1, 10 * MAX_TRIES):
        if ok.all():
            return P, spread, vals
        if t % 20 == 0:
            spread *= 0.5
        bad = np.flatnonzero(~ok)
        P[bad] = z0 + spread * rng.standard_normal((bad.size, z0.size)) @ L.T
        vals[bad] = lpf(P[bad])
        ok[bad] = np.isfinite(vals[bad, 0])
    raise RuntimeError("run_mcmc: could not place the walkers at valid points near the mode")


class _Ensemble:
    """One ensemble of walkers stepped like emcee 3.1's EnsembleSampler with red-blue moves (same use of its
    RandomState: the move choice, the split shuffle, the proposal, one uniform per walker for the acceptance), but with
    the log-probability supplied from outside: `step()` is a generator that yields each sub-ensemble's proposals and
    receives their (log p, -ln L). This lets `_run_lockstep` pool the proposals of all ensembles into one cost call per
    sub-step; every ensemble keeps its own walkers and random stream, so the chains are those of independent emcee
    samplers (identical draws when the cost of a row does not depend on the other rows of its batch)."""

    def __init__(self, coords: np.ndarray, vals: np.ndarray, random: np.random.RandomState, moves: list):
        self.coords = np.array(coords, dtype=float)
        self.log_prob = np.array(vals[:, 0], dtype=float)
        self.nll = np.array(vals[:, 1], dtype=float)
        self.random = random
        self.moves = [m for m, _ in moves]
        w = np.array([float(x) for _, x in moves])
        self.weights = w / w.sum()
        self.chain: list[np.ndarray] = []
        self.blobs: list[np.ndarray] = []
        self.naccepted = np.zeros(self.coords.shape[0])

    @property
    def acceptance(self) -> float:
        return float(np.mean(self.naccepted / max(1, len(self.chain))))

    def step(self):
        move = self.moves[self.random.choice(len(self.moves), p=self.weights)]
        nw = self.coords.shape[0]
        move.setup(self.coords)
        accepted = np.zeros(nw, dtype=bool)
        all_inds = np.arange(nw)
        inds = all_inds % move.nsplits
        if move.randomize_split:
            self.random.shuffle(inds)
        for split in range(move.nsplits):
            s1 = inds == split
            sets = [self.coords[inds == j] for j in range(move.nsplits)]
            q, factors = move.get_proposal(sets[split], sets[:split] + sets[split + 1:], self.random)
            new_lp, new_nll = yield q
            for j, f, nlp in zip(all_inds[s1], factors, new_lp):
                if f + nlp - self.log_prob[j] > np.log(self.random.rand()):
                    accepted[j] = True
            m1 = s1 & accepted
            m2 = accepted[s1]
            self.coords[m1] = q[m2]
            self.log_prob[m1] = new_lp[m2]
            self.nll[m1] = new_nll[m2]
        self.naccepted += accepted
        self.chain.append(self.coords.copy())
        self.blobs.append(self.nll.copy())


def _run_lockstep(ens: list[_Ensemble], lpf: _LogProb, nsteps: int) -> None:
    """Advance every ensemble by nsteps steps; at each sub-step the pending proposals of all ensembles go to ONE call
    of lpf. An ensemble whose step ends (its move has fewer sub-steps than another's) starts its next step at once, so
    its proposals join the next call instead of leaving calls of a few rows (decision #25; without it a third of the
    calls had 1-10 rows, 37-43 ms for one row against about 10 ms per extra row). Each ensemble still makes its
    own steps in order with its own random stream, so the chains are unchanged for a row-independent cost (V5c)."""
    left = [nsteps] * len(ens)
    gens: dict = {}
    pending: dict = {}
    for k, e in enumerate(ens):
        if left[k] > 0:
            gens[k] = e.step()
            pending[k] = next(gens[k])
            left[k] -= 1
    while pending:
        keys = list(pending)
        out = lpf(np.concatenate([pending[k] for k in keys]))
        nxt = {}
        a = 0
        for k in keys:
            n = pending[k].shape[0]
            try:
                nxt[k] = gens[k].send((out[a:a + n, 0], out[a:a + n, 1]))
            except StopIteration:
                if left[k] > 0:
                    gens[k] = ens[k].step()
                    nxt[k] = next(gens[k])
                    left[k] -= 1
            a += n
        pending = nxt


def default_moves(ch: _Chart) -> list:
    """[(ReflectDEMove in the chart's box, DE_WEIGHT), (SnookerMove, 1 - DE_WEIGHT)]."""
    return [(ReflectDEMove(ch.lo, ch.hi), DE_WEIGHT), (SnookerMove(), 1.0 - DE_WEIGHT)]


@dataclass
class McmcResult:
    names: list[str]  # names of the free parameters (sampling coordinates)
    mode_x: np.ndarray
    mode_f: float
    samples: np.ndarray  # (ensembles, steps, walkers, nfree), the whole chain (burn-in included)
    nll: np.ndarray  # (ensembles, steps, walkers): -ln L of each sample
    report: conv.V14Report
    converged: bool
    nevals: int
    max_evals: int
    nwalkers: int
    seeds: list[int]
    start_spread: float
    cov_gn: np.ndarray
    acceptance: list[float]
    better_points: list[S.PolishResult] = field(default_factory=list)
    polished: list[S.PolishResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    intervals: dict = field(default_factory=dict)  # quantity -> entry (see `compute_intervals`)
    flags: list[str] = field(default_factory=list)  # mode-level flags
    data_off: bool = False
    chart: str = "identity"  # sampling coordinates (make_chart)
    coords: list[str] = field(default_factory=list)  # their names

    @property
    def burn(self) -> int:
        return self.report.burn

    def kept(self) -> np.ndarray:
        """Kept samples, flattened over ensembles, steps and walkers: (n, nfree)."""
        return self.samples[:, self.burn:].reshape(-1, self.samples.shape[-1])

    def to_dict(self) -> dict:
        """intervals.json entry of this mode (FORMATS.md section 6)."""
        return {"neg_log_l": float(self.mode_f), "x": [float(v) for v in self.mode_x], "converged": self.converged,
                "failed_rules": list(self.report.failed), "flags": list(self.flags),
                "convergence": {**self.report.to_dict(), "nevals": int(self.nevals), "max_evals": int(self.max_evals),
                                "nwalkers": int(self.nwalkers), "ensembles": int(self.samples.shape[0]),
                                "steps": int(self.samples.shape[1]), "seeds": [int(s) for s in self.seeds],
                                "start_spread_gn_widths": float(self.start_spread), "chart": self.chart,
                                "sampling_coordinates": list(self.coords)},
                "better_points": [{"x": [float(v) for v in b.x], "neg_log_l": float(b.f)} for b in self.better_points],
                "notes": list(self.notes), "label": PER_MODE_LABEL, "quantities": self.intervals}

    def save_npz(self, path) -> None:
        path = str(path) if str(path).endswith(".npz") else f"{path}.npz"  # np.savez_compressed's own rule
        atomic.savez_compressed(path, samples=self.samples, nll=self.nll, names=np.array(self.names),
                            mode_x=self.mode_x, mode_f=self.mode_f, burn=self.burn, cov_gn=self.cov_gn)


def _canonical_batch(problem, X: np.ndarray) -> np.ndarray:
    fn = getattr(problem, "canonical_batch", None)
    if fn is not None:
        return np.asarray(fn(X), dtype=float)
    return np.array([np.asarray(problem.canonical(x), dtype=float) for x in X])


def _derived_batch(problem, X: np.ndarray) -> dict[str, np.ndarray]:
    """{name: (K,)} for every derived quantity (array entries expanded to name[i]); {} without `problem.derived`."""
    fb = getattr(problem, "derived_batch", None)
    if fb is not None:
        raw = {k: np.asarray(v, dtype=float) for k, v in fb(X).items()}
    elif getattr(problem, "derived", None) is not None:
        rows = [problem.derived(x) for x in X]
        raw = {k: np.array([np.asarray(r[k], dtype=float) for r in rows]) for k in rows[0]} if rows else {}
    else:
        return {}
    out = {}
    for k, v in raw.items():
        if v.ndim == 1:
            out[k] = v
        else:
            for j in range(v.shape[1]):
                out[f"{k}[{j}]"] = v[:, j]
    return out


def _quantities(problem, lpf: _LogProb, chain: np.ndarray, names: list[str], with_derived: bool,
                thin: int = 1) -> dict[str, np.ndarray]:
    """Reported quantities on the chain (E, steps, walkers, nfree), thinned in steps: the free parameters in canonical
    form, plus the derived quantities when asked. Each value: (E, steps', walkers)."""
    ch = chain[:, ::thin]
    E, T, W, nf = ch.shape
    X = lpf.full(ch.reshape(-1, nf))
    Xc = _canonical_batch(problem, X)
    out = {n: Xc[:, j].reshape(E, T, W) for n, j in zip(names, lpf.idx)}
    if with_derived:
        for k, v in _derived_batch(problem, X).items():
            out[k] = v.reshape(E, T, W)
    return out


def run_mcmc(problem, mode, cfg: dict, seed: int, log: Callable[[str], None] | None = None, *,
             value_tol: float = VALUE_TOL_DEFAULT, data_off: bool = False, handback: bool = True,
             chunk: int = CHUNK, n_prior: int = N_PRIOR_DRAWS, chart: str = "auto",
             moves: list | None = None, pool=None) -> McmcResult:
    """MCMC inside one mode (DESIGN.md section 4, steps 1-6; see the module docstring).

    mode: a `search.Mode` (or anything with `.x` (dim,) and `.f`). value_tol: the value tolerance of DESIGN.md
    section 3 (V14f). data_off: multiply -ln L by 0 (V13). handback: polish the best 3 samples (V14f).
    chart: the sampling coordinates (`make_chart`: "auto", "observer" or "identity"). moves: [(emcee move, weight)]
    in the sampling coordinates instead of the default (diagnostics; `default_moves`). pool: a `search.make_pool` pool
    over which the cost calls and the hand-back polishes are spread; None: in this process.
    """
    say = log or (lambda _m: None)
    uc = _unc_cfg(cfg)
    E = int(uc.get("ensembles", 4))
    tau_factor = float(uc.get("tau_factor", 50.0))
    max_evals, note = _max_evals(cfg)
    notes = [note] if note else []
    if note:
        say("mcmc: " + note)
    x_mode = np.asarray(mode.x, dtype=float)
    idx = _free_idx(problem)
    names = [problem.names[i] for i in idx]
    ch = make_chart(problem, cfg, chart)
    nd = ch.dim
    nwalkers = _nwalkers(nd)
    lpf = _LogProb(problem, cfg, data_off, ch, pool)
    if ch.kind == "observer":
        notes.append("sampled in the observer frame of the first phase: " + ", ".join(ch.names)
                     + " (|md|, cos of the dipole-observer angle, the dipole azimuth about the line of sight as a "
                     "2-d vector with a standard normal length factor; other vectors rotated with u)")
    orient, ncoords = nuisance_names(problem, ch, names)
    nuisance = set(orient) | set(ncoords)
    if orient:
        notes.append("declared nuisance (one phase does not determine the orientation): " + ", ".join(ncoords + orient)
                     + " are sampled but not required to converge; no interval is reported for " + ", ".join(orient))

    cov = gauss_newton_cov(problem, x_mode, cfg)
    z0 = ch.from_free(x_mode[idx][None])[0]
    if ch.kind == "identity":
        cov_z = cov
    else:  # pull the Gauss-Newton precision back to the sampling coordinates
        B = ch.jacobian(z0)
        cov_z = _pd_inverse(B.T @ _pd_inverse(cov) @ B)
    cap = np.where(np.isfinite(ch.cap_width), 0.25 * ch.cap_width, np.inf)

    ss = np.random.SeedSequence(int(seed)).spawn(E)
    seeds = [int(s.generate_state(1)[0]) for s in ss]
    moves = moves if moves is not None else default_moves(ch)
    ens, spreads = [], []
    for e in range(E):
        rng = np.random.default_rng(ss[e])
        P, spread, vals = _initial_walkers(lpf, z0, cov_z, nwalkers, rng, cap)
        spreads.append(spread)
        ens.append(_Ensemble(P, vals, np.random.RandomState(seeds[e] % (2**32)), moves))
    if min(spreads) < START_SPREAD:
        notes.append(f"walkers started within {min(spreads):g} (not {START_SPREAD:g}) Gauss-Newton widths: invalid "
                     "points near the mode")

    def chains() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cz = np.stack([np.array(e.chain) for e in ens])  # (E, steps, W, nd)
        cy = ch.to_free(cz.reshape(-1, nd))[0].reshape(cz.shape[:3] + (len(idx),))
        nl = np.stack([np.array(e.blobs) for e in ens])
        return cz, cy, nl

    def for_tau(cz, cy):  # autocorrelation on the sampling coordinates and (other charts) the free parameters too
        return (cz, list(ch.names)) if ch.kind == "identity" else \
            (np.concatenate([cz, cy], axis=-1), list(ch.names) + names)

    report = None
    steps = 0
    while True:
        left = (max_evals - lpf.nevals) // (nwalkers * E)
        if left <= 0:
            break
        n = int(min(chunk, left))
        _run_lockstep(ens, lpf, n)
        steps += n
        cz, chain, nll = chains()
        acc = [e.acceptance for e in ens]
        q = _quantities(problem, lpf, chain, names, with_derived=False)
        ct, nt = for_tau(cz, chain)
        report = conv.check_v14(ct, nll, tau_factor=tau_factor, quantities=q, acceptance=acc, names=nt,
                                nuisance=nuisance)
        say(f"mcmc: {steps} steps, {lpf.nevals} evaluations, tau {report.tau:.1f}, acceptance "
            f"{min(acc):.2f}-{max(acc):.2f}, failed {report.failed}")
        if report.passed or n < chunk:  # n < chunk: the budget allows no further full chunk
            break

    if steps == 0:
        raise RuntimeError(f"run_mcmc: max_evals = {max_evals} allows no step of {E} x {nwalkers} walkers")
    cz, chain, nll = chains()
    acc = [e.acceptance for e in ens]

    # final check including derived quantities (thinned by tau / 4 steps to bound their cost)
    has_derived = getattr(problem, "derived", None) is not None or getattr(problem, "derived_batch", None) is not None
    if has_derived:
        b = report.burn
        thin = max(1, int(report.tau / 4)) if math.isfinite(report.tau) else 1
        q = _quantities(problem, lpf, chain[:, b:], names, with_derived=True, thin=thin)
        ct, nt = for_tau(cz[:, b::thin], chain[:, b::thin])
        rep2 = conv.check_v14(ct, nll[:, b::thin], tau_factor=tau_factor, quantities=q, acceptance=acc, names=nt,
                              burn=0, nuisance=nuisance)
        report.rhat.update(rep2.rhat)
        report.ess_bulk.update(rep2.ess_bulk)
        report.ess_tail.update(rep2.ess_tail)
        for rule in ("V14c_rhat", "V14d_ess_bulk", "V14d_ess_tail"):
            bad = [k for k in rep2.failed_quantities.get(rule, []) if k not in names]
            if bad:
                report.failed_quantities.setdefault(rule, []).extend(bad)
                if rule not in report.failed:
                    report.failed.append(rule)
        # per-quantity verdict of the derived quantities (decision #17): their own tau on the thinned chain (both
        # the kept length and tau in thinned steps), their R-hat and ESS, and the run's -ln L rules
        run_rules = [r for r in ("V14e_drift", "V14e_stuck") if r in report.failed]
        for k, rules in rep2.quantity_failed.items():
            if k not in names:
                report.quantity_failed[k] = [r for r in rules if r not in ("V14e_drift", "V14e_stuck")] + run_rules
                report.tau_quantity[k] = rep2.tau_quantity[k] * thin
        report.passed = not report.failed
    if report.passed is False and lpf.nevals + nwalkers * E > max_evals:
        say(f"mcmc: NOT CONVERGED after {lpf.nevals} evaluations: {report.failed}")
    for w in report.warnings:
        say("mcmc: WARNING: " + w)

    res = McmcResult(names, x_mode, float(mode.f), chain, nll, report, bool(report.passed), lpf.nevals, max_evals,
                     nwalkers, seeds, min(spreads), cov, acc, notes=notes, data_off=data_off, chart=ch.kind,
                     coords=list(ch.names))
    if handback:
        res.polished, res.better_points = _handback_candidates(problem, res, cfg, value_tol, pool)
        res.nevals += sum(p.nevals for p in res.polished)
    res.intervals, res.flags = compute_intervals(problem, cfg, res, lpf, seed=seed, n_prior=n_prior)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# 3. Hand-back (V14f, UNC-3)


def _outside_ok(problem, x: np.ndarray, cfg: dict) -> bool:
    om = float(cfg.get("search", {}).get("starts", {}).get("outside_max", math.inf))
    fn = getattr(problem, "outside_fraction", None)
    return fn is None or float(np.max(np.atleast_1d(fn(x)))) <= om


def _handback_candidates(problem, res: McmcResult, cfg: dict, value_tol: float, pool=None):
    """Polish the best 3 distinct samples by -ln L (all steps, all ensembles); return (polished, better), better =
    those ending below mode_f - value_tol and inside the prior cut, best first."""
    flat = res.samples.reshape(-1, res.samples.shape[-1])
    f = res.nll.reshape(-1)
    order = np.argsort(f, kind="stable")
    picked: list[np.ndarray] = []
    for k in order:
        if not np.isfinite(f[k]):
            break
        if not any(np.array_equal(flat[k], p) for p in picked):
            picked.append(flat[k])
        if len(picked) == 3:
            break
    idx = _free_idx(problem)
    base = np.asarray(problem.value, dtype=float)
    starts = []
    for z in picked:
        x = base.copy()
        x[idx] = z
        starts.append(x)
    tasks = [(problem, x, cfg) for x in starts]
    polished = list(pool.map(S.polish_start, tasks)) if pool is not None else [S.polish_start(t) for t in tasks]
    better = sorted((p for p in polished if p.f < res.mode_f - value_tol and _outside_ok(problem, p.x, cfg)),
                    key=lambda p: p.f)
    return polished, better


@dataclass
class UncertaintyResult:
    modes: list  # the mode table after hand-backs, best first
    results: list[McmcResult]  # final MCMC result per mode, same order as `modes`
    handbacks: list[list[dict]]  # per mode: the hand-backs made
    nevals: int
    calibration: str  # "uncalibrated" or the configured calibration (UNC-4)

    def to_dict(self) -> dict:
        """intervals.json (FORMATS.md section 6)."""
        out = []
        for k, (m, r, hb) in enumerate(zip(self.modes, self.results, self.handbacks)):
            d = r.to_dict()
            d.update({"mode": k, "handbacks": hb})
            out.append(d)
        return {"calibration": self.calibration, "label": PER_MODE_LABEL, "nevals": int(self.nevals),
                "modes": out}


def uncertainty_for_modes(problem, modes: list, cfg: dict, seed: int, *, value_tol: float = VALUE_TOL_DEFAULT,
                          log: Callable[[str], None] | None = None, max_handbacks: int = MAX_HANDBACKS,
                          **kw: Any) -> UncertaintyResult:
    """run_mcmc for every mode; a better polished point (V14f) is polished again with `search.polish`, replaces the
    mode's point and value (fingerprint and canonical form updated when the problem has them) and the MCMC is re-run
    from it, at most `max_handbacks` times per mode (UNC-3). The table is then re-sorted by -ln L.

    Seeds: mode k, attempt a uses SeedSequence([seed, k, a]). Each hand-back records the old and new -ln L and, when
    the problem has fingerprints, the fingerprint distance between the old and the new point; whether the new point
    lies in another mode (DESIGN.md section 3, step 3: against the 99th percentile over the mode's samples) is left to
    the search loop that consumes this table.
    """
    say = log or (lambda _m: None)
    results, hbs = [], []
    nevals = 0
    for k, mode in enumerate(modes):
        hb: list[dict] = []
        attempt = 0
        while True:
            s = int(np.random.SeedSequence([int(seed), k, attempt]).generate_state(1)[0])
            r = run_mcmc(problem, mode, cfg, s, log, value_tol=value_tol, **kw)
            nevals += r.nevals
            if not r.better_points or len(hb) >= max_handbacks:
                if r.better_points:
                    r.flags.append(f"better point found after {max_handbacks} hand-backs (UNC-3 limit)")
                break
            cand = r.better_points[0]
            again = S.polish(problem, cand.x, cfg)
            nevals += again.nevals
            new = again if again.f <= cand.f else cand
            entry = {"attempt": attempt, "old_neg_log_l": float(mode.f), "new_neg_log_l": float(new.f),
                     "x": [float(v) for v in new.x]}
            if getattr(problem, "fingerprint_quantities", None) is not None and getattr(mode, "fingerprint", None):
                fp = S.fingerprint(problem, new.x)
                entry["fingerprint_distance"] = S.fingerprint_distance(mode.fingerprint, fp,
                                                                       S.fingerprint_thresholds(cfg))
                mode.fingerprint = fp
            say(f"mcmc: mode {k}: hand-back {len(hb) + 1}: -ln L {mode.f:.3f} -> {new.f:.3f}")
            mode.x = np.array(new.x)
            mode.f = float(new.f)
            if hasattr(mode, "touched_limit"):
                mode.touched_limit = bool(new.touched_limit)
            if hasattr(mode, "x_canonical"):
                mode.x_canonical = np.asarray(problem.canonical(mode.x), dtype=float)
            hb.append(entry)
            attempt += 1
        results.append(r)
        hbs.append(hb)
    order = sorted(range(len(modes)), key=lambda i: (modes[i].f, i))
    cal = str(_unc_cfg(cfg).get("calibration", "") or "")
    return UncertaintyResult([modes[i] for i in order], [results[i] for i in order], [hbs[i] for i in order], nevals,
                             "uncalibrated" if not cal else f"calibration run named: {cal} (not verified here)")


# ---------------------------------------------------------------------------------------------------------------------
# 4. Intervals and flags


def _kl_hist(post: np.ndarray, prior: np.ndarray, lo: float, hi: float, bins: int = KL_BINS,
             with_bins: bool = False):
    """KL(posterior || prior) from histograms with `bins` bins over [lo, hi]; samples outside are put in the end bins;
    empty prior bins get half a count. with_bins: return (KL, number of non-empty posterior bins).

    The plug-in estimate is biased upwards by about (filled bins - 1) / (2 n) nats for n independent posterior draws
    (Miller 1955; the Miller-Madow correction of the plug-in entropy; the 10 000 prior draws add (bins - 1) / 20 000).
    Measured on the toy star: draws from the prior itself gave a mean KL of 0.081 nat at n = 124 (predicted
    0.077) and 0.025 at n = 400 (0.024); at n = 124 a quarter of them exceeded 0.1 nat. `compute_intervals` therefore
    flags REP-4 on KL - (filled - 1) / (2 n_eff)."""
    if not (hi > lo):
        return (math.inf, 0) if with_bins else math.inf
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(np.clip(post, lo, hi), edges)[0].astype(float)
    q = np.histogram(np.clip(prior, lo, hi), edges)[0].astype(float)
    q = np.where(q > 0, q, 0.5)
    p /= p.sum()
    q /= q.sum()
    m = p > 0
    kl = float(np.sum(p[m] * np.log(p[m] / q[m])))
    return (kl, int(m.sum())) if with_bins else kl


def _limit_ranges(problem, cfg: dict, names: list[str], idx: np.ndarray) -> dict[str, list[tuple[str, float, float]]]:
    """Per reported parameter, the ranges whose ends count as limits (REP-3): the polish safety limits, and for
    scalars of the field the prior range (bq, bo, u in canonical form)."""
    blo, bhi = S.polish_bounds(problem)
    spec = _prior_spec(problem, cfg)
    st = spec.st
    out = {}
    for n, i in zip(names, idx):
        r = [("range", float(blo[i]), float(bhi[i]))]
        g = S._group_of(n) if i in S._field_range(problem) else None
        if g == "u" and "u" in spec.groups:
            r.append(("prior", float(st["u"][0]), float(st["u"][1])))
        elif g == "bq" and "bq" in spec.groups:
            r.append(("prior", -float(st["bq"]), float(st["bq"])))
        elif g == "bo" and "bo" in spec.groups:
            r.append(("prior", -float(st["mo"]), float(st["mo"])))
        out[n] = r
    return out


def _touches(v_lo: float, v_hi: float, ranges) -> bool:
    for _, lo, hi in ranges:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        tol = LIMIT_FRAC * (hi - lo)
        if v_lo - lo <= tol or hi - v_hi <= tol:
            return True
    return False


def _norm_limits(spec: _PriorSpec, blo: np.ndarray, bhi: np.ndarray) -> dict[str, tuple[float, float]]:
    """Length limits of the vector groups (prior shell / ball), for REP-3 on vector lengths."""
    st = spec.st
    out = {}
    if "md" in spec.groups:
        out["md"] = (float(st["md"][0]), float(st["md"][1]))
    if "mo" in spec.groups:
        out["mo"] = (0.0, float(st["mo"]))
    if "c" in spec.groups:
        out["c"] = (0.0, float(st["c"]))
    return out


def compute_intervals(problem, cfg: dict, res: McmcResult, lpf: _LogProb | None = None, *, seed: int = 0,
                      n_prior: int = N_PRIOR_DRAWS) -> tuple[dict, list[str]]:
    """Per quantity (free parameters in canonical form, then derived quantities): median, 16/84 and 2.5/97.5 % points,
    half-widths, R-hat, ESS, KL to the prior, flags.

    UNC-2 per quantity (decision #17): when the mode passes every V14 rule, every quantity gets its numbers, as
    before. When it does not, a quantity still gets its numbers when it passes the V14 rules on its own
    (`report.quantity_failed[name]` empty: R-hat, bulk and tail ESS, chain length and tau stability on its own
    autocorrelation time, and the run's -ln L rules); otherwise they are withheld (None), the quantity is flagged
    "not converged" and `failed_rules` lists its own failed rules (a quantity without a verdict of its own takes the
    mode's failed rules). The per-quantity rules are never applied to a converged mode: a single quantity's tau
    estimate is noisier than the mode's longest tau, so its 5 % stability rule can fail by chance. Declared nuisance (decision #18, `nuisance_names`): the orientation quantities get no
    numbers and the flags ORIENTATION and NUISANCE (their R-hat and ESS are still given). The mode-level flag "not
    converged" is set as before when the mode-level verdict fails."""
    lpf = lpf or _LogProb(problem, cfg)
    names = res.names
    rep = res.report
    b = rep.burn
    mode_flags: list[str] = []
    kept = res.samples[:, b:]
    if kept.shape[1] == 0:
        kept = res.samples[:, -1:]
    # post-processing cost (decision #25): the samples are thinned to a quarter of the shortest integrated
    # autocorrelation time before the canonical form and the derived quantities (the B quantiles over all patches);
    # draws closer than that carry almost no information of their own
    taus = [float(t) for t in (rep.tau_quantity or {}).values() if np.isfinite(t) and t > 0]
    tau_min = min(taus) if taus else (float(rep.tau) if np.isfinite(rep.tau) else 1.0)
    thin = max(1, int(tau_min // 4))
    kept = kept[:, ::thin]
    E, Tn, W, nf = kept.shape
    X = lpf.full(kept.reshape(-1, nf))
    Xc = _canonical_batch(problem, X)
    q = {n: Xc[:, j] for n, j in zip(names, lpf.idx)}
    q.update(_derived_batch(problem, X))

    # u reached 0: raw u changes sign, or canonical u within 1 % of the lower end of the prior's |u| range
    spec = _prior_spec(problem, cfg)
    u_flag = False
    if "u" in spec.groups:
        iu = spec.groups["u"][0]
        u_raw = X[:, iu]
        lo_u, hi_u = float(spec.st["u"][0]), float(spec.st["u"][1])
        u_flag = bool((u_raw.min() < 0 < u_raw.max()) or np.min(np.abs(u_raw)) <= lo_u + LIMIT_FRAC * (hi_u - lo_u))
        if u_flag:
            mode_flags.append(U_ZERO)

    # prior draws for the KL flag (free parameters only)
    prior_c = None
    try:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
        Pd = draw_prior(problem, cfg, rng, int(n_prior))
        Pc = _canonical_batch(problem, Pd)
        prior_c = {n: Pc[:, j] for n, j in zip(names, lpf.idx)}
    except (ValueError, RuntimeError) as exc:  # no finite range to draw from
        res.notes.append(f"KL flag skipped: {exc}")

    limits = _limit_ranges(problem, cfg, names, lpf.idx)
    blo, bhi = S.polish_bounds(problem)
    nlim = _norm_limits(spec, blo, bhi)
    group_touch = {}
    for g, (lo, hi) in nlim.items():
        n = np.sqrt(np.sum(X[:, spec.groups[g]] ** 2, axis=1))
        a, z = np.quantile(n, [0.025, 0.975])
        tol = LIMIT_FRAC * (hi - lo)
        group_touch[g] = bool((g == "md" and a - lo <= tol) or hi - z <= tol)

    if not res.converged:
        mode_flags.insert(0, NOT_CONVERGED)
    orient = set(nuisance_names(problem, lpf.chart, names)[0])
    out = {}
    for name, v in q.items():
        flags = []
        entry: dict[str, Any] = {"rhat": rep.rhat.get(name), "ess_bulk": rep.ess_bulk.get(name),
                                 "ess_tail": rep.ess_tail.get(name), "jackknife_sigma": None}
        if res.converged:  # the mode passes every rule: every quantity is reported, as before
            own = []
        elif name in rep.quantity_failed:
            own = list(rep.quantity_failed[name])
        else:
            own = [r for r in rep.failed if r not in rep.failed_quantities or name in rep.failed_quantities[r]]
        ok = not own and name not in orient
        if name in orient:
            flags += [ORIENTATION, NUISANCE]
            entry["failed_rules"] = own
        elif own:
            flags.append(NOT_CONVERGED)
            entry["failed_rules"] = own
        p = np.quantile(v, [0.5, 0.16, 0.84, 0.025, 0.975])
        h68, h95 = 0.5 * (p[2] - p[1]), 0.5 * (p[4] - p[3])
        if ok:
            entry.update({"median": float(p[0]), "p16": float(p[1]), "p84": float(p[2]), "p2_5": float(p[3]),
                          "p97_5": float(p[4]), "half68": float(h68), "half95": float(h95)})
        else:
            for k in ("median", "p16", "p84", "p2_5", "p97_5", "half68", "half95"):
                entry[k] = None
        if name in limits:
            g = S._group_of(name)
            if _touches(float(p[3]), float(p[4]), limits[name]) or group_touch.get(g, False):
                flags.append(TOUCHES_LIMIT)
            if g == "u" and u_flag:
                flags.append(U_ZERO)
        kl = klc = None
        if prior_c is not None and name in prior_c:
            pr = prior_c[name]
            ne = rep.ess_bulk.get(name)
            n_eff = int(max(2, min(v.size, ne if ne is not None and math.isfinite(ne) else v.size)))
            thinned = v[np.linspace(0, v.size - 1, n_eff).astype(int)]
            kl, filled = _kl_hist(thinned, pr, float(pr.min()), float(pr.max()), with_bins=True)
            klc = kl - (filled - 1) / (2.0 * n_eff)  # Miller-Madow bias of the plug-in estimate (see _kl_hist)
            if klc < KL_MIN:
                flags.append(NOT_CONSTRAINED)
        entry["kl_prior"] = kl
        entry["kl_prior_corrected"] = klc
        entry["flags"] = flags
        out[name] = entry
    return out, mode_flags


# ---------------------------------------------------------------------------------------------------------------------
# 5. Profiles (UNC-5, V1l)


class _Fixed:
    """A view of a problem with one more parameter fixed (free False, value = the grid value)."""

    def __init__(self, problem, index: int, value: float, x: np.ndarray):
        self._p = problem
        self.free = np.asarray(problem.free, dtype=bool).copy()
        self.free[index] = False
        self.value = np.array(x, dtype=float)
        self.value[index] = value

    def __getattr__(self, name):
        return getattr(self._p, name)


def profile(problem, x_best: np.ndarray, index: int, grid, cfg: dict) -> dict:
    """-ln L profile of parameter `index`: at each grid value the other free parameters are re-optimised with
    `search.polish`, starting from the optimum of the nearest grid value already done (the walk goes outwards from the
    grid value closest to x_best). Returns {"grid", "neg_log_l", "x", "nevals"}."""
    grid = np.asarray(grid, dtype=float)
    x_best = np.asarray(x_best, dtype=float)
    f = np.full(grid.size, np.nan)
    xs = np.zeros((grid.size, x_best.size))
    c = int(np.argmin(np.abs(grid - x_best[index])))
    nevals = 0
    for order in (range(c, grid.size), range(c - 1, -1, -1)):
        x0 = x_best.copy()
        for k in order:
            x0[index] = grid[k]
            pr = S.polish(_Fixed(problem, index, grid[k], x0), x0, cfg)
            nevals += pr.nevals
            f[k], xs[k] = pr.f, pr.x
            x0 = pr.x.copy()
    return {"grid": grid, "neg_log_l": f, "x": xs, "nevals": nevals}
