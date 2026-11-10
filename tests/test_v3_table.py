"""V3n: the table stage (SPEC SEARCH-8, SEARCH-9; DESIGN.md section 3, steps 6 and 6b; table.py),
on a toy -ln L surface whose wells change depth between the search's patch count and the
error-bar patch count, so the answer is known.

The toy: four wells of width 1 in two coordinates, far apart, so near well k the cost is d_k + |x - m_k|^2 / 2 to
within ~1e-20. A configuration with `search.lbfgs = false` and `search.lm_maxiter = 0` makes the re-polish return its
start unchanged, so each end point's -ln L at the error-bar patch count is d_k + e with e chosen per end point.

Search (1000 patches): A (depth -5), D (2), B (3), C (4) in that order. Error bars (500 patches): B 0, A 2, C 15, and
D's region is invalid. Re-polished values: A 8, 10, 12, 14 (4 end points); B 0, 2, 4 (3); C 15, 30 (2). Expected:
the order flips to B, A; D is dropped as invalid, C as outside the 10-nat band; the pooled scatter uses A and B only
(sqrt(28 / 5) = 2.366 nats; with C it would be sqrt(140.5 / 6) = 4.84), the value tolerance is twice that.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from zeemap import search as S
from zeemap.config import defaults
from zeemap.table import refresh_families, table_stage

pytestmark = pytest.mark.tier1

CENTRES = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (0.0, 10.0), "D": (60.0, 0.0)}
DEPTHS = {1000: {"A": -5.0, "B": 3.0, "C": 4.0, "D": 2.0}, 500: {"A": 2.0, "B": 0.0, "C": 15.0, "D": 2.0}}
OFFSETS = {"A": [6.0, 8.0, 10.0, 12.0], "D": [0.0], "B": [0.0, 2.0, 4.0], "C": [0.0, 15.0]}  # e per end point
DIRECTION = {"A": (0.0, -1.0), "B": (1.0, 0.0), "C": (0.0, 1.0), "D": (0.0, 1.0)}  # away from the other wells


class Wells:
    names = ["p", "q"]
    dim = 2
    free = np.array([True, True])
    step = np.array([1.0, 1.0])
    lo = np.array([-100.0, -100.0])
    hi = np.array([100.0, 100.0])
    value = np.zeros(2)
    field_slice = None

    def __init__(self, n):
        self.surface = SimpleNamespace(n=n)
        self.keys = list(CENTRES)
        self.m = np.array([CENTRES[k] for k in self.keys])
        self.d = np.array([DEPTHS[n][k] for k in self.keys])

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        q = 0.5 * np.sum((X[:, None, :] - self.m[None]) ** 2, axis=2)
        f = -logsumexp(-self.d[None] - q, axis=1)
        if self.surface.n == 500:  # D's region is invalid on the error-bar patches
            f = np.where(X[:, 0] > 50.0, S.WORST, f)
        return f

    def canonical(self, x):
        return np.asarray(x, dtype=float)


def make_problem(cfg):
    return Wells(int(cfg["model"]["surface"]["n"]))


def point(k, e):
    return np.array(CENTRES[k]) + math.sqrt(2.0 * e) * np.array(DIRECTION[k])


@pytest.fixture()
def setup():
    cfg = defaults()
    cfg["model"]["surface"] = {"scheme": "fibonacci", "n": 1000}
    cfg["uncertainty"]["patches"] = 500
    cfg["search"]["band"] = 10.0
    cfg["search"]["lbfgs"] = False
    cfg["search"]["lm_maxiter"] = 0
    cfg["search"]["family"]["min_level"] = 2.0  # the family level is the value tolerance (4.73 nats here)
    p = Wells(1000)
    endpoints, modes = [], []
    for k in ("A", "D", "B", "C"):  # the search's order
        mem = []
        for e in OFFSETS[k]:
            x = point(k, e)
            endpoints.append(S.EndPoint(x, float(p.cost(x[None])[0]), len(endpoints), 0, 1, True, False, True))
            mem.append(len(endpoints) - 1)
        best = min(mem, key=lambda i: endpoints[i].f)
        modes.append(S.Mode(endpoints[best].x, endpoints[best].f, None, len(mem), mem))
        modes[-1].name = k
    res = S.SearchResult(modes, list(modes), endpoints, {}, {}, 0, 1, list(p.names))
    return p, cfg, res


def test_v3n_table_reranks_at_the_error_bar_patch_count(setup):
    p, cfg, res = setup
    assert [m.name for m in res.modes] == ["A", "D", "B", "C"]
    ts = table_stage(p, cfg, res, workers=1, make_problem=make_problem, say=lambda _m: None)
    st = ts["stats"]
    assert st["table_repolished"] is True and st["table_patches"] == 500
    # new order: B before A; D (invalid on 500 patches) and C (15 nats above the best, band 10) are dropped
    assert [m.name for m in ts["modes"]] == ["B", "A"]
    assert ts["modes"][0].f == pytest.approx(0.0, abs=1e-9)
    assert ts["modes"][1].f == pytest.approx(8.0, abs=1e-9)
    np.testing.assert_allclose(ts["modes"][1].x, point("A", 6.0), atol=1e-12)
    # pooled scatter over the modes with >= 3 re-polished points (A, B), before the band cut (C has only 2)
    scatter = math.sqrt(28.0 / 5.0)
    assert st["optimiser_scatter_table"] == pytest.approx(scatter, rel=1e-9)
    assert st["value_tolerance_table"] == pytest.approx(S.value_tolerance(scatter), rel=1e-9)
    assert st["value_tolerance_table"] == pytest.approx(2.0 * scatter, rel=1e-9)
    # the search's modes are unchanged (the table works on copies)
    assert [m.name for m in res.modes] == ["A", "D", "B", "C"] and res.modes[0].f == pytest.approx(1.0, abs=1e-9)
    # families: the level is f_best + 4.73; only B's nodes lie below it, so mode 1 (B) is family 1 and A has none.
    # Each mode carries its re-polished nodes, aligned after the drops and the re-ranking.
    fams = ts["families"]
    assert fams["h"] == pytest.approx(2.0 * scatter, rel=1e-9)
    assert [fm["modes"] for fm in fams["families"]] == [[1]]
    assert len(fams["families"][0]["nodes"]) == 3
    assert ts["modes"][0].family == 1 and ts["modes"][1].family is None
    assert [round(f, 9) for _, f in ts["modes"][0].nodes] == [0.0, 2.0, 4.0]
    assert [round(f, 9) for _, f in ts["modes"][1].nodes] == [8.0, 10.0, 12.0, 14.0]


def test_v3n_families_recomputed_after_a_hand_back(setup):
    p, cfg, res = setup
    ts = table_stage(p, cfg, res, workers=1, make_problem=make_problem, say=lambda _m: None)
    up, modes = ts["problem"], ts["modes"]
    tol = ts["stats"]["value_tolerance_table"]
    band = float(cfg["search"]["band"])
    # no hand-back: nothing to recompute
    assert refresh_families(up, cfg, modes, tol, band, say=lambda _m: None) is None
    # a hand-back (uncertainty_for_modes) moves A from 8 to its well's minimum, 2: more than the value tolerance
    moved = copy.copy(modes[1])
    moved.x = np.array(CENTRES["A"])
    moved.f = float(up.cost(moved.x[None])[0])
    assert moved.f < modes[1].f - tol
    modes = [modes[0], moved]
    got = refresh_families(up, cfg, modes, tol, band, say=lambda _m: None)
    assert got is not None
    fams, st = got
    assert fams["recomputed_after_handbacks"] == [2]
    assert [fm["modes"] for fm in fams["families"]] == [[1], [2]]
    assert modes[1].family == 2 and st["n_families"] == 2
    # the profiles start from a family's best node: for family 2 that is the hand-back's point
    fam2 = fams["families"][1]
    best = next(n for n in fam2["nodes"] if n["node"] == fam2["best"])
    np.testing.assert_allclose(best["x"], CENTRES["A"], atol=1e-12)
    assert best["neg_log_l"] == pytest.approx(2.0, abs=1e-9)
