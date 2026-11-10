"""Families: end points joined by paths along which -ln L stays below a level (SPEC SEARCH-9; DESIGN.md section 3b;
decision #29).

- Nodes: end points re-polished at the error-bar patch count (`search.repolish_band`), in canonical form.
- Level: f_best + h, h = max(value tolerance, `search.family.min_level`); one level for all families.
- Link test between two nodes at or below the level: the straight segment in the scaled search coordinates (x / step
  over the free parameters) at `interior` points, between the closest pair of their symmetric copies, with a_q
  rescaled to unit length at every point (its length is a gauge); if it fails, a re-polished midpoint (L-BFGS-B in
  the hyperplane through the segment's midpoint perpendicular to the segment, at most `mid_iter` iterations) and the
  two half segments at 7 interior points each.
- Components by union-find; pairs in order of increasing scaled distance, already joined pairs skipped, at most
  `max_pairs` tested. The same at the band level gives the band components.
- With workers > 1 the next W pairs not yet joined are tested at once in the search's pool and their results applied
  in that order, skipping a pair joined meanwhile: the same families, links and `pairs_tested` as the
  serial loop; only `nevals` can be larger.
- A mode belongs to the family of its best node and lists every family its nodes fall into.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, minimize

from . import search as S
from .config import defaults

WORST = S.WORST
DEFAULTS = {"min_level": 5.0, "max_pairs": 3000, "mid_iter": 50, "interior": 15}
HALF_INTERIOR = 7


@dataclass
class Node:
    mode: int  # 0-based index into the mode table
    x: np.ndarray  # canonical parameters
    f: float


@dataclass
class Grouping:
    level: float  # absolute -ln L level
    h: float  # level - f_best
    component: list[int]  # per node: component id (-1 for a node above the level)
    links: list[dict] = field(default_factory=list)
    pairs_tested: int = 0
    capped: bool = False


def family_settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(((cfg.get("search") or {}).get("family")) or {})
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _aq_index(problem):
    names = list(problem.names)
    if all(n in names for n in ("aq_x", "aq_y", "aq_z")):
        return np.array([names.index(n) for n in ("aq_x", "aq_y", "aq_z")])
    return None


def _gauge(problem, X):
    """a_q rescaled to unit length at every row (its length is a gauge: DESIGN.md section 2)."""
    ia = _aq_index(problem)
    if ia is None or not np.all(np.asarray(problem.free)[ia]):
        return X
    X = np.array(X, dtype=float)
    n = np.linalg.norm(X[:, ia], axis=1)
    ok = n > 0
    X[np.ix_(ok, ia)] = X[np.ix_(ok, ia)] / n[ok, None]
    return X


def symmetric_copies(problem, x) -> list[np.ndarray]:
    """x and the copies that give identical spectra (the equator mirror, the reversed quadrupole axis)."""
    out = [np.asarray(x, dtype=float)]
    m = problem.mirror(x) if hasattr(problem, "mirror") else None
    if m is not None:
        out.append(np.asarray(m, dtype=float))
    ia = _aq_index(problem)
    if ia is not None:
        for y in list(out):
            z = y.copy()
            z[ia] = -z[ia]
            out.append(z)
    return out


class Linker:
    """The link test of DESIGN.md section 3b on one problem (the error-bar patch count). `fd_step`: the half-step of
    the midpoint's finite differences in parameter steps (`search.fd_step`; None = its default). The midpoint stays
    within the polish limits (`search.polish_bounds`) when the problem has `lo` and `hi`."""

    def __init__(self, problem, settings: dict, fd_step: float | None = None):
        self.p = problem
        self.st = settings
        self.free = np.flatnonzero(np.asarray(problem.free, dtype=bool))
        self.step = np.asarray(problem.step, dtype=float)
        self.h = float(defaults()["search"]["fd_step"] if fd_step is None else fd_step)
        self.bounds = S.polish_bounds(problem) if hasattr(problem, "lo") and hasattr(problem, "hi") else None
        self.nevals = 0

    def scaled(self, x):
        return np.asarray(x, dtype=float)[self.free] / self.step[self.free]

    def closest_copy(self, xi, xj):
        zi = self.scaled(xi)
        best, bd = None, math.inf
        for c in symmetric_copies(self.p, xj):
            d = float(np.linalg.norm(self.scaled(c) - zi))
            if d < bd:
                best, bd = c, d
        return best, bd

    def _cost(self, X):
        X = _gauge(self.p, np.atleast_2d(X))
        self.nevals += X.shape[0]
        return np.asarray(self.p.cost(X), dtype=float)

    def segment_max(self, xa, xb, k):
        t = np.arange(1, k + 1) / (k + 1)
        X = xa[None, :] + t[:, None] * (xb - xa)[None, :]
        return float(np.max(self._cost(X)))

    def midpoint(self, xa, xb):
        """Polish from the midpoint within the hyperplane perpendicular to the segment (scaled coordinates), inside
        the polish limits. The gradient is the polish's (`search.cost_gradient`: central differences at `self.h`,
        one-sided at a limit or next to an invalid model), projected onto the hyperplane."""
        m0 = 0.5 * (xa + xb)
        d = self.scaled(xb) - self.scaled(xa)
        nd = float(np.linalg.norm(d))
        if nd == 0:
            return m0, float(self._cost(m0)[0])
        u = d / nd
        n = self.free.size
        s = self.step[self.free]
        blo, bhi = self.bounds if self.bounds is not None else (None, None)

        def full(z):
            zz = z - u * float(u @ z)  # projected onto the hyperplane
            x = m0.copy()
            x[self.free] = m0[self.free] + zz * s
            return x if blo is None else np.clip(x, blo, bhi)  # m0 lies inside: the clip acts only at a limit

        def fun(z):
            f0, g, _ = S.cost_gradient(self._cost, full(z), self.free, s, self.h, blo, bhi)
            if not (f0 < WORST):
                return WORST, np.zeros(n)
            return f0, g - u * float(u @ g)

        zb = None if blo is None else Bounds((blo[self.free] - m0[self.free]) / s, (bhi[self.free] - m0[self.free]) / s)
        r = minimize(fun, np.zeros(n), jac=True, method="L-BFGS-B", bounds=zb,
                     options={"maxiter": int(self.st["mid_iter"])})
        xm = full(r.x)
        return xm, float(self._cost(xm)[0])

    def linked(self, xi, xj, level) -> tuple[bool, dict]:
        xj2, dist = self.closest_copy(xi, xj)
        top = self.segment_max(xi, xj2, int(self.st["interior"]))
        if top <= level:
            return True, {"path": "straight", "max_neg_log_l": top, "distance": dist}
        xm, fm = self.midpoint(xi, xj2)
        if fm > level:
            return False, {"path": "midpoint", "max_neg_log_l": max(top, fm), "distance": dist}
        top2 = max(fm, self.segment_max(xi, xm, HALF_INTERIOR), self.segment_max(xm, xj2, HALF_INTERIOR))
        return top2 <= level, {"path": "midpoint", "max_neg_log_l": top2, "distance": dist}


def _link_task(args: tuple) -> tuple:
    """Picklable unit of work for the search's pool: args = (linker, xi, xj, level) -> (linked, info, evaluations)."""
    lk, xi, xj, level = args
    n0 = lk.nevals
    ok, info = lk.linked(xi, xj, level)
    return ok, info, lk.nevals - n0


def group(linker: Linker, nodes: list[Node], level: float, max_pairs: int, pool=None, workers: int = 1) -> Grouping:
    f_best = min(n.f for n in nodes)
    below = [i for i, n in enumerate(nodes) if n.f <= level]
    uf = _UF(len(nodes))
    pairs = []
    for a in range(len(below)):
        for b in range(a + 1, len(below)):
            i, j = below[a], below[b]
            pairs.append((linker.closest_copy(nodes[i].x, nodes[j].x)[1], i, j))
    pairs.sort()
    g = Grouping(level=level, h=level - f_best, component=[-1] * len(nodes))
    w = max(1, int(workers)) if pool is not None else 1
    k = 0
    while k < len(pairs):
        n = min(w, max_pairs - g.pairs_tested)  # the next n pairs not joined now; w = 1: the serial loop
        batch = []
        while k < len(pairs) and len(batch) < max(n, 1):
            if uf.find(pairs[k][1]) != uf.find(pairs[k][2]):
                batch.append(pairs[k])
            k += 1
        if n <= 0:
            g.capped = bool(batch)  # a pair left to test beyond the cap
            break
        tasks = [(linker, nodes[i].x, nodes[j].x, level) for _, i, j in batch]
        if pool is None:
            res = [(*linker.linked(*t[1:]), 0) for t in tasks]  # counted in linker.nevals already
        else:
            res = list(pool.map(_link_task, tasks))
            linker.nevals += sum(r[2] for r in res)
        for (_, i, j), (ok, info, _) in zip(batch, res):  # applied in the serial order
            if uf.find(i) == uf.find(j):
                continue  # joined by an earlier pair of the batch: the serial loop skips it
            g.pairs_tested += 1
            if ok:
                uf.union(i, j)
                g.links.append({"nodes": [i, j], **info})
    roots = {}
    for i in below:
        r = uf.find(i)
        roots.setdefault(r, len(roots))
    # component ids ordered by their best node
    order = sorted(roots, key=lambda r: min(nodes[i].f for i in below if uf.find(i) == r))
    rid = {r: k for k, r in enumerate(order)}
    for i in below:
        g.component[i] = rid[uf.find(i)]
    return g


def find_families(problem, nodes: list[Node], value_tol: float, band: float, cfg: dict, workers: int = 1) -> dict:
    """Families at the family level and components at the band level (SEARCH-9). Returns the families.json content
    plus per-mode assignments: {"level", "h", "band_level", "families": [...], "band_components": [...],
    "mode_family": {mode: id or None}, "mode_families_all": {mode: [ids]}, "mode_band_component": {...}, "nevals"}.
    With workers > 1 the link tests run in the search's pool (single-threaded workers), else in this process under
    `search.single_threaded`, so the result does not depend on the number of workers."""
    st = family_settings(cfg)
    lk = Linker(problem, st, (cfg.get("search") or {}).get("fd_step"))
    f_best = min(n.f for n in nodes)
    h = max(float(value_tol), float(st["min_level"]))
    w = max(1, int(workers))
    with (S.make_pool(w) if w > 1 else nullcontext()) as pool, \
            (S.single_threaded() if pool is None else nullcontext()):
        fam = group(lk, nodes, f_best + h, int(st["max_pairs"]), pool, w)
        bnd = group(lk, nodes, f_best + float(band), int(st["max_pairs"]), pool, w)

    def describe(g: Grouping):
        out = []
        for cid in sorted({c for c in g.component if c >= 0}):
            members = [i for i, c in enumerate(g.component) if c == cid]
            best = min(members, key=lambda i: nodes[i].f)
            out.append({"id": cid + 1, "modes": sorted({nodes[i].mode + 1 for i in members}),
                        "nodes": [{"node": i, "mode": nodes[i].mode + 1, "neg_log_l": nodes[i].f,
                                   "x": [float(v) for v in nodes[i].x]} for i in members],
                        "best": best, "neg_log_l": nodes[best].f,
                        "links": [ln for ln in g.links if g.component[ln["nodes"][0]] == cid]})
        return out

    modes = sorted({n.mode for n in nodes})
    mode_family, mode_all, mode_band = {}, {}, {}
    for m in modes:
        idx = [i for i, n in enumerate(nodes) if n.mode == m]
        best = min(idx, key=lambda i: nodes[i].f)
        mode_family[m + 1] = None if fam.component[best] < 0 else fam.component[best] + 1
        mode_all[m + 1] = sorted({fam.component[i] + 1 for i in idx if fam.component[i] >= 0})
        mode_band[m + 1] = None if bnd.component[best] < 0 else bnd.component[best] + 1
    return {"level": fam.level, "h": fam.h, "band_level": bnd.level, "families": describe(fam),
            "band_components": describe(bnd), "capped": fam.capped, "band_capped": bnd.capped,
            "pairs_tested": fam.pairs_tested + bnd.pairs_tested, "mode_family": mode_family,
            "mode_families_all": mode_all, "mode_band_component": mode_band, "nevals": lk.nevals}
