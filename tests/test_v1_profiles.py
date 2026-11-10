"""V1l (profiles: the walks, the lower envelope, intervals and minima, on toy problems with a known profile), V4i (the
UNC-8 ratios and flags) and V6d (the same profiles with 1 and 2 workers); profiles.py. The end-to-end
profile runs are in test_v7_cli.py (V7j) and test_v6_determinism.py (V6d)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from zeemap import profiles as PR
from zeemap.config import defaults

pytestmark = pytest.mark.tier1


def test_v1l_intervals_of_a_known_profile():
    """A parabola with curvature 1/sigma^2 (sigma = 2): the Delta < 0.5 interval is +-sigma, Delta < 2 is +-2 sigma, to
    the grid's linear-interpolation error, and a walk that ends inside an interval marks that side open."""
    grid = np.arange(-10.0, 10.0 + 1e-9, 0.25)
    env = 100.0 + 0.5 * (grid / 2.0) ** 2
    lo, hi, olo, ohi = PR._interval(grid, env, 100.5)
    assert lo == pytest.approx(-2.0, abs=0.25) and hi == pytest.approx(2.0, abs=0.25) and not (olo or ohi)
    lo, hi, _, _ = PR._interval(grid, env, 102.0)
    assert lo == pytest.approx(-4.0, abs=0.25) and hi == pytest.approx(4.0, abs=0.25)
    lo, hi, olo, ohi = PR._interval(grid[:30], env[:30], 140.0)
    assert olo and ohi  # the walked grid ended inside the interval on both sides


class Quad:
    """-ln L = (x - m)^T A (x - m) / 2 over (teff, p, y), with residuals L^T (x - m) (A = L L^T) and the derived
    quantity q = c . x. The profile of a coordinate t is (t - m_t)^2 / (2 (A^-1)_tt); that of q is
    (q - c . m)^2 / (2 V), V = c^T A^-1 c. Held by the penalty kappa ((q - q0) / dq)^2, the polished point has
    q - q* = (q0 - q*) / (1 + e), e = dq^2 / (2 kappa V), so the reported (unpenalised) value is P(q0) / (1 + e)^2."""

    names = ["teff", "p", "y"]
    dim = 3
    field_slice = None
    iteff = 0

    def __init__(self):
        self.m = np.array([30.0, 1.0, -2.0])
        self.L = np.array([[1.0 / 3.0, 0.0, 0.0], [0.2, 1.5, 0.0], [-0.1, 0.6, 0.8]])
        self.A = self.L @ self.L.T
        self.c = np.array([0.0, 1.0, 0.5])
        self.free = np.ones(3, dtype=bool)
        self.lo, self.hi = np.array([0.0, -50.0, -50.0]), np.array([100.0, 50.0, 50.0])
        self.step, self.value = np.ones(3), np.zeros(3)

    def cost(self, X):
        R = self.residuals_batch(np.atleast_2d(X), 1.0)
        return 0.5 * np.sum(R * R, axis=1)

    def residual_scales(self, x):
        return 1.0

    def residuals_batch(self, X, scales):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self.m) @ self.L

    def residuals(self, x):
        return self.residuals_batch(x, 1.0)[0]

    def derived_batch(self, X):
        return {"q": np.atleast_2d(X) @ self.c}


class Branches:
    """Two branches of -ln L over (teff, y): b1 = ((t - 25) / 3)^2 / 2 at y = -5 and b2 = 0.4 + ((t - 44) / 3)^2 / 2 at
    y = +5, each with width 0.5 in y; -ln L is the lower of the two. With t held, a polish started on one branch stays
    on it (the other lies 200 nats higher at its y), so a walk follows its own branch past the crossing at t = 34.69
    (hysteresis); the true profile is min(b1, b2)."""

    names = ["teff", "y"]
    dim = 2
    field_slice = None
    iteff = 0

    def __init__(self):
        self.free = np.ones(2, dtype=bool)
        self.lo, self.hi = np.array([0.0, -10.0]), np.array([100.0, 10.0])
        self.step, self.value = np.ones(2), np.zeros(2)

    @staticmethod
    def branch(t, b):
        return 0.5 * ((t - 25.0) / 3.0) ** 2 if b == 0 else 0.4 + 0.5 * ((t - 44.0) / 3.0) ** 2

    def _res(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        r1 = np.stack([(X[:, 0] - 25.0) / 3.0, (X[:, 1] + 5.0) / 0.5, np.zeros(len(X))], axis=1)
        r2 = np.stack([(X[:, 0] - 44.0) / 3.0, (X[:, 1] - 5.0) / 0.5, np.full(len(X), math.sqrt(0.8))], axis=1)
        f1, f2 = 0.5 * np.sum(r1 * r1, axis=1), 0.5 * np.sum(r2 * r2, axis=1)
        return np.where((f1 <= f2)[:, None], r1, r2)

    def cost(self, X):
        R = self._res(X)
        return 0.5 * np.sum(R * R, axis=1)

    def residual_scales(self, x):
        return 1.0

    def residuals_batch(self, X, scales):
        return self._res(X)

    def residuals(self, x):
        return self._res(x)[0]

    def derived_batch(self, X):
        return {}


class Bounded:
    """A bounded toy: q = p >= 0 (a polish limit, held by the penalty), best at p = 0.3;
    -ln L = ((p - 0.3) / 3)^2 / 2 + ((y - p / 2) / 0.5)^2 / 2, so the profile of q is ((q - 0.3) / 3)^2 / 2 for q >= 0
    and the model cannot reach q < 0."""

    names = ["p", "y"]
    dim = 2
    field_slice = None
    iteff = None

    def __init__(self):
        self.free = np.ones(2, dtype=bool)
        self.lo, self.hi = np.array([0.0, -50.0]), np.array([50.0, 50.0])
        self.step, self.value = np.ones(2), np.zeros(2)

    def residuals_batch(self, X, scales):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.stack([(X[:, 0] - 0.3) / 3.0, (X[:, 1] - 0.5 * X[:, 0]) / 0.5], axis=1)

    def cost(self, X):
        R = self.residuals_batch(X, 1.0)
        return 0.5 * np.sum(R * R, axis=1)

    def residual_scales(self, x):
        return 1.0

    def residuals(self, x):
        return self.residuals_batch(x, 1.0)[0]

    def derived_batch(self, X):
        return {"q": np.atleast_2d(np.asarray(X, dtype=float))[:, 0]}


class Vec:
    """A toy in the vector space with two phases: (md, c, u) with -ln L a quadratic around md = (0, 0, 300),
    c = (0.02, 0, 0.01), u = cos(3 deg); c fixed at 0 when `c_free` is False. The derived quantities of FORMATS.md
    section 7 that the default profile list uses (B quantiles: simple functions of |md| and u)."""

    names = ["md_x", "md_y", "md_z", "c_x", "c_y", "c_z", "u"]
    dim = 7
    field_slice = None
    iteff = None

    class _Data:
        phases = [0.0, 0.5]

    def __init__(self, c_free=True):
        self.dataset = Vec._Data()
        self.free = np.array([True, True, True, c_free, c_free, c_free, True])
        self.lo = np.array([-1000.0, -1000.0, -1000.0, -0.3, -0.3, -0.3, -1.0])
        self.hi = -self.lo
        self.step = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01, 0.05])
        self.value = np.zeros(7)
        self.m = np.array([0.0, 0.0, 300.0, 0.02, 0.0, 0.01, math.cos(math.radians(3.0))])
        self.s = np.array([20.0, 20.0, 10.0, 0.01, 0.01, 0.01, 0.05])

    def best(self):
        x = self.m.copy()
        if not self.free[3]:
            x[3:6] = 0.0
        return x

    def residuals_batch(self, X, scales):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self.m) / self.s

    def cost(self, X):
        R = self.residuals_batch(X, 1.0)
        return 0.5 * np.sum(R * R, axis=1)

    def residual_scales(self, x):
        return 1.0

    def residuals(self, x):
        return self.residuals_batch(x, 1.0)[0]

    def derived_batch(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        nmd = np.linalg.norm(X[:, :3], axis=1)
        u = X[:, 6]
        out = {"beta_deg": np.degrees(np.arccos(np.clip(X[:, 2] / nmd, -1.0, 1.0)))}
        for name, a in (("B10_MG", 0.6), ("B50_MG", 0.8), ("B90_MG", 0.95)):
            out[name] = np.stack([nmd * (a + 0.05 * p * u) for p in range(2)], axis=1)
        return out


def _cfg(steps, max_steps=40):
    cfg = defaults()
    cfg["uncertainty"]["profile_steps"] = steps
    cfg["uncertainty"]["profile_max_steps"] = max_steps
    return cfg


def _branch_family(toy):
    return {"nodes": [{"x": [25.0, -5.0], "neg_log_l": float(toy.cost([25.0, -5.0])[0])},
                      {"x": [44.0, 5.0], "neg_log_l": float(toy.cost([44.0, 5.0])[0])}]}


def test_v1l_known_profile_through_the_walks():
    """A quadratic toy (one family member, so the second walks start from the far ends of the first): Teff, fixed
    directly, gives (t - m_t)^2 / (2 (A^-1)_tt) at every grid value; q, held by the penalty, gives the closed form
    P(q0) / (1 + e)^2 (class Quad); both to 1e-6, every point meets the constraint, the walks reach the stop level
    f_best + max(2 h, 10) on both sides, the walks back cross the whole profile, and the intervals equal their closed
    forms to one grid step."""
    toy = Quad()
    h = 5.0
    Ainv = np.linalg.inv(toy.A)
    V, qs = float(toy.c @ Ainv @ toy.c), float(toy.c @ toy.m)
    dq = 0.25 * math.sqrt(V)
    fam = {"nodes": [{"x": toy.m.tolist(), "neg_log_l": 0.0}]}
    # both LM stops off: the closed forms hold to 1e-6; with decision #37's defaults each point lies within
    # search.lm_stop_nat above them (checked at the end)
    cfg = _cfg({"teff_K": 1.0, "q": dq})
    cfg["search"].update(lm_stop_nat=0.0, lm_max_rejections=0)
    out = PR.family_profiles(toy, cfg, fam, level=h, h=h, quantities=["teff_K", "q"])
    e = out["teff_K"]
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    np.testing.assert_allclose(env, (g - toy.m[0]) ** 2 / (2.0 * Ainv[0, 0]), rtol=0, atol=1e-6)
    assert len(e["walks"]) == 3 and all(p["met"] for w in e["walks"] for p in w["points"]) and e["flags"] == []
    assert e["ends"] == {"lo": None, "hi": None}
    for w in e["walks"][1:]:  # each walk back starts above the stop level and crosses the whole profile
        assert len(w["points"]) == len(g)
    assert env[0] > 10.0 and env[-1] > 10.0 and env[1] <= 10.0 and env[-2] <= 10.0  # stop level reached both sides
    s = math.sqrt(Ainv[0, 0])
    for key, d in (("delta_0.5", 0.5), ("delta_2", 2.0), ("family_level", h)):
        iv = e["intervals"][key]
        assert iv["lo"] == pytest.approx(toy.m[0] - s * math.sqrt(2 * d), abs=1.0)
        assert iv["hi"] == pytest.approx(toy.m[0] + s * math.sqrt(2 * d), abs=1.0)
        assert not (iv["open_lo"] or iv["open_hi"])
    assert e["minima"] == [(30.0, pytest.approx(0.0, abs=1e-9))]
    e = out["q"]
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    eps = dq * dq / (2.0 * PR.KAPPA * V)
    np.testing.assert_allclose(env, 0.5 * (g - qs) ** 2 / V / (1.0 + eps) ** 2, rtol=0, atol=1e-6)
    assert all(p["met"] for w in e["walks"] for p in w["points"]) and env[0] > 10.0 and env[-1] > 10.0
    iv = e["intervals"]["delta_2"]
    assert iv["lo"] == pytest.approx(qs - 2.0 * math.sqrt(V), abs=dq)
    assert iv["hi"] == pytest.approx(qs + 2.0 * math.sqrt(V), abs=dq)
    cfg = _cfg({"teff_K": 1.0, "q": dq})
    dflt = PR.family_profiles(toy, cfg, fam, level=h, h=h, quantities=["teff_K", "q"])
    for k in ("teff_K", "q"):
        assert dflt[k]["grid"] == out[k]["grid"]
        d = np.array(dflt[k]["envelope_profile"]) - np.array(out[k]["envelope_profile"])
        assert np.all(d >= -1e-6) and np.all(d <= cfg["search"]["lm_stop_nat"])


def test_v1l_lower_envelope_removes_hysteresis():
    """Two branches (class Branches), walks from the two family members (t = 25 on branch 1, t = 44 on branch 2): each
    walk follows its own branch past the crossing (hysteresis, more than 4 nats above the true profile at the far
    grid values); the lower envelope equals min(b1, b2) at every grid value (1e-6); its intervals equal their closed
    forms to one grid step (the family-level one spans both branches); the UNC-8 envelope equals the range of the
    toy's points below the level; both minima are found."""
    toy = Branches()
    h = level = 6.0
    out = PR.family_profiles(toy, _cfg({"teff_K": 2.5}), _branch_family(toy), level=level, h=h,
                             quantities=["teff_K"])
    e = out["teff_K"]
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    true = np.minimum(Branches.branch(g, 0), Branches.branch(g, 1))
    np.testing.assert_allclose(env, true, rtol=0, atol=1e-6)
    w1 = {p["q0"]: p["neg_log_l"] for p in e["walks"][0]["points"]}
    w2 = {p["q0"]: p["neg_log_l"] for p in e["walks"][1]["points"]}
    assert w1[37.5] == pytest.approx(Branches.branch(37.5, 0), abs=1e-6)  # walk 1 stays on branch 1 past the crossing
    assert w1[37.5] - Branches.branch(37.5, 1) > 4.0
    assert w2[32.5] == pytest.approx(Branches.branch(32.5, 1), abs=1e-6)  # walk 2 stays on branch 2
    assert w2[32.5] - Branches.branch(32.5, 0) > 4.0
    assert all(p["met"] for w in e["walks"] for p in w["points"]) and e["flags"] == []
    iv = e["intervals"]
    for key, lo_c, hi_c in (("delta_0.5", 22.0, 28.0), ("delta_2", 19.0, 31.0)):  # b1 <= 0.5, 2 around t = 25
        assert iv[key]["lo"] == pytest.approx(lo_c, abs=2.5) and iv[key]["hi"] == pytest.approx(hi_c, abs=2.5)
    lo_c, hi_c = 25.0 - 3.0 * math.sqrt(2 * level), 44.0 + 3.0 * math.sqrt(2 * (level - 0.4))
    fl = iv["family_level"]
    assert fl["lo"] == pytest.approx(lo_c, abs=2.5) and fl["hi"] == pytest.approx(hi_c, abs=2.5)
    assert not (fl["open_lo"] or fl["open_hi"])
    below = [25.0, 44.0] + [q for q, f in zip(g, true) if f <= level]  # the toy's points below the level
    assert e["envelope"] == {"lo": min(below), "hi": max(below)} == {"lo": 15.0, "hi": 52.5}
    assert [q for q, _ in e["minima"]] == [25.0, 45.0]


def test_v1l_gap_between_walks_splits_the_interval():
    """Walks cut short (profile_max_steps = 2) cover 20-30 and 40-50 with a gap between: the interval around the best
    ends at the gap, open, instead of being interpolated across it."""
    toy = Branches()
    out = PR.family_profiles(toy, _cfg({"teff_K": 2.5}, max_steps=2), _branch_family(toy), level=6.0, h=6.0,
                             quantities=["teff_K"])
    e = out["teff_K"]
    assert e["grid"] == [20.0, 22.5, 25.0, 27.5, 30.0, 40.0, 42.5, 45.0, 47.5, 50.0]
    fl = e["intervals"]["family_level"]
    assert (fl["lo"], fl["hi"], fl["open_lo"], fl["open_hi"]) == (20.0, 30.0, True, True)
    d = e["intervals"]["delta_0.5"]  # inside one piece: closed, interpolated
    assert d["lo"] == pytest.approx(22.0, abs=0.5) and d["hi"] == pytest.approx(28.0, abs=0.5)
    assert not (d["open_lo"] or d["open_hi"])
    assert [q for q, _ in e["minima"]] == [25.0, 45.0]


def test_v1l_interval_does_not_cross_a_gap():
    grid = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    env = np.array([3.0, 0.0, 0.5, 0.2, 3.0])
    ks = [0, 1, 2, 5, 6]
    lo, hi, olo, ohi = PR._interval(grid, env, 1.0, ks)
    assert lo == pytest.approx(1.0 - 1.0 / 3.0) and hi == 2.0 and not olo and ohi


def test_v1l_envelope_holds_only_points_that_met_the_constraint():
    """At a grid value the envelope takes the lowest point that met |q - q0| <= 0.1 dq; a grid value where no walk's
    point met it is left out (not the lowest not-met point with a flag, which is not a value of the profile), so the
    envelope has a gap there. A closed end beyond the grid bounds the interval at its q (below the threshold) or at
    the crossing towards it (above), and is not open."""
    x = np.zeros(2)
    walks = [{"points": {0: (1.0, x, 0.0, False), 1: (3.0, x, 0.0, False)}},
             {"points": {0: (2.0, x, 0.0, True), 1: (2.5, x, 0.0, False), 2: (4.0, x, 0.0, True)}}]
    ks, env = PR._lower_envelope(walks)
    assert ks == [0, 2] and env.tolist() == [2.0, 4.0]
    grid, env = np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.0, 0.6])
    assert PR._interval(grid, env, 1.0) == (1.0, 3.0, True, True)
    assert PR._interval(grid, env, 1.0, ends=((0.6, 0.4), None)) == (0.6, 3.0, False, True)
    lo, hi, olo, ohi = PR._interval(grid, env, 1.0, ends=((0.6, 1.8), (3.5, 1.6)))  # crossings towards the ends
    assert lo == pytest.approx(1.0 - 0.4 * 0.5) and hi == pytest.approx(3.0 + 0.5 * 0.4) and not (olo or ohi)
    w = [{"points": {}, "ends": {"lo": {"q": 0.6, "neg_log_l": 0.4, "reason": PR.NOT_MET}, "hi": None}},
         {"points": {}, "ends": {"lo": {"q": 1.5, "neg_log_l": 0.1, "reason": PR.NOT_MET},  # inside: closes nothing
                                 "hi": {"q": 3.0, "neg_log_l": 0.6, "reason": PR.RANGE_END}}}]
    ends = PR._closed_ends(w, grid, 1.0)
    assert ends["lo"]["q"] == 0.6 and ends["hi"] == {"q": 3.0, "neg_log_l": 0.6, "reason": PR.RANGE_END}


def test_v1l_local_minima_skip_open_ends_plateaus_and_shallow_dips():
    grid = np.arange(8.0)
    assert PR.local_minima(grid, np.array([5.0, 3.0, 1.0, 2.0, 4.0, 3.0, 2.0, 1.5])) == [(2.0, 1.0)]  # falling end
    assert PR.local_minima(grid, np.array([5.0, 3.0, 1.0, 1.0, 1.0, 3.0, 5.0, 6.0])) == [(3.0, 1.0)]  # one plateau
    assert PR.local_minima(grid, np.array([5.0, 1.0, 1.3, 1.2, 4.0, 0.0, 3.0, 5.0])) == [(1.0, 1.0), (5.0, 0.0)]
    assert PR.local_minima(grid, np.array([5.0, 1.0, 1.3, 1.2, 4.0, 0.0, 3.0, 5.0]), depth=0.05) == [
        (1.0, 1.0), (3.0, 1.2), (5.0, 0.0)]
    env = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 5.0])
    ks = [0, 1, 2, 3, 10, 11, 12, 13]  # a gap between index 3 and 4: index 3 ends a piece (open), not a minimum
    assert PR.local_minima(grid, env, ks=ks) == [(1.0, 1.0), (5.0, 0.0)]
    assert PR.local_minima(grid, np.array([0.5, 1.0, 3.0]), closed=(True, False)) == [(0.0, 0.5)]  # a range end
    assert PR.local_minima(grid, np.array([0.5, 1.0, 3.0])) == []


def test_v6d_profiles_same_for_1_and_2_workers():
    """The walks of every family and quantity run as independent tasks in the search's pool (DESIGN.md section 4b):
    1 and 2 workers give the same profiles.json content (two families, two quantities, one family with one node)."""
    toy = Branches()
    fams = [_branch_family(toy), {"nodes": [{"x": [44.0, 5.0], "neg_log_l": float(toy.cost([44.0, 5.0])[0])}]}]
    cfg = _cfg({"teff_K": 2.5}, max_steps=6)
    runs = [PR.profiles_for_families(toy, cfg, fams, level=6.0, h=6.0, quantities=["teff_K"], workers=w)
            for w in (1, 2)]
    a, b = (json.dumps(r, sort_keys=True) for r in runs)
    assert a == b
    assert len(runs[0]) == 2 and len(runs[0][1]["teff_K"]["walks"]) == 3
    one = PR.family_profiles(toy, cfg, fams[0], level=6.0, h=6.0, quantities=["teff_K"])
    assert json.dumps(one, sort_keys=True) == json.dumps(runs[0][0], sort_keys=True)


def test_v4i_ratios_and_flags():
    h = 6.0
    mode = {"p16": 10.0, "p84": 12.0}  # width 2
    r1, r2, flags = PR.ratios({"lo": 8.4, "hi": 11.6}, mode, h)  # envelope width 3.2: R1 = 1.6
    assert r1 == pytest.approx(1.6) and r2 == pytest.approx(1.6 / math.sqrt(12.0))
    assert flags == []  # R1 is printed, never flagged (decision #30)
    r1, _, flags = PR.ratios({"lo": 9.0, "hi": 11.8}, mode, h)  # R1 = 1.4
    assert r1 == pytest.approx(1.4) and flags == []
    w = 2.0 * 1.6 * math.sqrt(12.0)  # R2 = 1.6
    _, r2, flags = PR.ratios({"lo": 0.0, "hi": w}, mode, h)
    assert r2 == pytest.approx(1.6) and flags == ["envelope wider than one Gaussian basin gives"]
    w = 2.0 * 1.4 * math.sqrt(12.0)  # R2 = 1.4 (R1 = 4.8): no flag
    _, r2, flags = PR.ratios({"lo": 0.0, "hi": w}, mode, h)
    assert r2 == pytest.approx(1.4) and flags == []
    # a Gaussian basin: the envelope below h spans 2 sqrt(2 h) sigma, the 68 % interval 2 sigma: R2 = 1
    s = 0.7
    _, r2, _ = PR.ratios({"lo": -math.sqrt(2 * h) * s, "hi": math.sqrt(2 * h) * s}, {"p16": -s, "p84": s}, h)
    assert r2 == pytest.approx(1.0)
    assert PR.ratios({"lo": 0.0, "hi": 1.0}, None, h) == (None, None, [])


def _points(walk):
    """{grid index: point} of a profiles.json walk (FORMATS.md section 6)."""
    return {p["k"]: p for p in walk["points"]}


def test_v1l_walk_ends_closed_where_the_constraint_cannot_be_met():
    """Regression: q = p >= 0 held by the penalty, best at 0.3, grid step 0.25 (class Bounded). The walk down
    stops at the first point that misses the constraint (q0 = -0.2, the polish stays at q = 0): that point is written
    with met = False and ends the direction, and the lower side is closed at the q it reached (0), not open and not
    stretched over values the model cannot take (before the fix: 39 of 81 points not met, delta_0.5 = [-9.7, 3.3]).
    No not-met point enters the envelope, the intervals or the minima; the flag "constraint not met" is set."""
    toy = Bounded()
    h = 5.0
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}, max_steps=60), fam, level=h, h=h, quantities=["q"])["q"]
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    assert env[-1] > 10.0  # the walks up reach the stop level
    assert g.min() == pytest.approx(0.05) and np.all(g >= 0.0)  # no grid value the model cannot reach
    eps = 0.25 ** 2 / (2.0 * PR.KAPPA * 9.0)  # the penalty's pull (class Quad), V = 9
    np.testing.assert_allclose(env, 0.5 * ((g - 0.3) / 3.0) ** 2 / (1.0 + eps) ** 2, rtol=0, atol=1e-6)
    for key, hi in (("delta_0.5", 0.3 + 3.0), ("family_level", 0.3 + 3.0 * math.sqrt(2 * h))):
        iv = e["intervals"][key]
        assert iv["lo"] == pytest.approx(0.0, abs=1e-6) and not iv["open_lo"]  # closed at the limit
        assert iv["hi"] == pytest.approx(hi, abs=0.25) and not iv["open_hi"]
    assert e["envelope"]["lo"] == pytest.approx(0.0, abs=1e-6)
    assert e["ends"]["lo"]["reason"] == PR.NOT_MET and e["ends"]["lo"]["q"] == pytest.approx(0.0, abs=1e-6)
    assert e["ends"]["hi"] is None
    assert PR.NOT_MET in e["flags"]
    assert [q for q, _ in e["minima"]] == [pytest.approx(0.3)]
    for w in e["walks"]:  # a not-met point only as the last point of a walk down; every other point met
        pts = sorted(w["points"], key=lambda p: p["k"])
        bad = [p for p in pts if not p["met"]]
        assert len(bad) <= 1 and all(p["q0"] >= 0.0 for p in pts if p["met"])
        if bad:
            assert bad[0] is pts[0] and bad[0]["q0"] == pytest.approx(-0.2)
            assert bad[0]["q"] == pytest.approx(0.0, abs=1e-6)
    assert sum(1 for w in e["walks"] for p in w["points"] if not p["met"]) == 2  # the first walk and one walk back


def test_v1l_inclination_interval_closed_at_0_degrees():
    """Regression: the inclination (fixed directly through u) of a toy whose best lies at 3 degrees and whose
    profile stays low down to 0: the walk down polishes at 0 degrees itself (the end of the range) and stops; the
    intervals reaching it are closed at 0 (before the fix: open, printed '<'), and the end counts for the minima."""
    toy = Vec()
    h = 5.0
    fam = {"nodes": [{"x": toy.best().tolist(), "neg_log_l": float(toy.cost(toy.best())[0])}]}
    e = PR.family_profiles(toy, _cfg({"incl_deg": 2.5}), fam, level=h, h=h, quantities=["incl_deg"])["incl_deg"]
    g = np.array(e["grid"])
    assert g.min() == pytest.approx(0.5) and np.all(g >= 0.0)
    for key in ("delta_0.5", "delta_2", "family_level"):
        iv = e["intervals"][key]
        assert iv["lo"] == 0.0 and not iv["open_lo"] and not iv["open_hi"], (key, iv)
    assert e["ends"]["lo"]["reason"] == PR.RANGE_END and e["ends"]["lo"]["q"] == 0.0
    assert PR.NOT_MET not in e["flags"]
    assert [q for q, _ in e["minima"]] == [pytest.approx(3.0)]
    from zeemap.report import side_by_side_text

    text = "\n".join(side_by_side_text({1: {"best_mode": 1, "converged": None, "quantities": {"incl_deg": e}}},
                                       {"h": h}, None))
    row = next(r for r in text.splitlines() if "incl_deg" in r and ".." in r)
    assert "<0" not in row and " 0.." in row


def test_v1l_default_list_leaves_out_quantities_no_free_parameter_moves():
    """Regression: with c fixed at 0, c_along_axis = c . m_d / |m_d| depends on no free parameter, so the default
    list leaves it out (before: profiled, 1 of 13 points met); with c free it stays in. `uncertainty.profile_quantities
    = "default"` (the shipped default) gives that list (before: the string "default" was walked letter by letter,
    KeyError 'd'). The B quantiles are those of the reference phase only (`search.fingerprint.ref_phase`, decision
    #40; before: every phase, 3 + 3 P quantities)."""
    for c_free in (True, False):
        toy = Vec(c_free=c_free)
        x = toy.best()
        names = PR.profile_names(toy, defaults(), x)
        want = ["incl_deg", "dipole_MG"] + (["c_along_axis"] if c_free else []) + [
            f"{b}[0]" for b in ("B10_MG", "B50_MG", "B90_MG")]
        assert names == want, (c_free, names)
    cfg1 = defaults()
    cfg1["search"]["fingerprint"]["ref_phase"] = 1
    assert PR.profile_names(Vec(c_free=False), cfg1, x)[2:] == [f"{b}[1]" for b in ("B10_MG", "B50_MG", "B90_MG")]
    toy = Vec(c_free=False)
    fam = {"nodes": [{"x": toy.best().tolist(), "neg_log_l": float(toy.cost(toy.best())[0])}]}
    out = PR.family_profiles(toy, _cfg({}, max_steps=1), fam, level=5.0, h=5.0)
    assert list(out) == want  # the loop's last `want`: c fixed


def test_v1l_profiles_json_walks_hold_start_node_q_and_met():
    """FORMATS.md section 6, profiles.json: each walk names its start (role, family node and mode
    id, grid index) and lists per point the grid index, the grid value q0, the q reached, -ln L and the flag met; the
    entry has the closed ends of the range ("ends")."""
    toy = Branches()
    fam = _branch_family(toy)
    fam["nodes"][0].update(node=7, mode=3)
    fam["nodes"][1].update(node=8, mode=5)
    e = PR.family_profiles(toy, _cfg({"teff_K": 2.5}, max_steps=3), fam, level=6.0, h=6.0,
                           quantities=["teff_K"])["teff_K"]
    s0, s1 = e["walks"][0]["start"], e["walks"][1]["start"]
    assert s0 == {"role": "best", "node": 7, "mode": 3, "k": 0}
    assert s1 == {"role": "farthest node", "node": 8, "mode": 5, "k": round((44.0 - 25.0) / 2.5)}
    for w in e["walks"]:
        for p in w["points"]:
            assert set(p) == {"k", "q0", "q", "neg_log_l", "met", "stalled"} and not p["stalled"]
            assert p["q0"] == pytest.approx(25.0 + 2.5 * p["k"]) and p["q"] == pytest.approx(p["q0"]) and p["met"]
    json.dumps(e)  # plain JSON types
    assert set(e["ends"]) == {"lo", "hi"}
    one = PR.family_profiles(toy, _cfg({"teff_K": 2.5}, max_steps=3), {"nodes": fam["nodes"][:1]}, level=6.0, h=6.0,
                             quantities=["teff_K"])["teff_K"]
    assert [w["start"]["role"] for w in one["walks"]] == ["best", "walk back", "walk back"]
    assert one["walks"][1]["start"]["node"] is None and one["walks"][1]["start"]["k"] == 3


def test_v4i_side_by_side_prints_constraint_not_met():
    """The side-by-side table of report.txt shows the flag "constraint not met" in the row of a
    quantity whose walk stopped where the constraint could not be met."""
    from zeemap.report import side_by_side_text

    toy = Bounded()
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}), fam, level=5.0, h=5.0, quantities=["q"])["q"]
    lines = side_by_side_text({1: {"best_mode": 1, "converged": None, "quantities": {"q": e}}}, {"h": 5.0}, None)
    row = next(r for r in lines if r.split()[:2] == ["1", "q"])
    assert PR.NOT_MET in row and " 0..3." in row


def test_v1l_percent_step_floor_at_the_quantitys_own_scale():
    """A step "<p>%" is p % of max(|q_best|, s_q), s_q the largest change of q over one `step` of any
    free parameter from the best node (the quantity's own scale). The toy Quad has q_best = c . m = 0, where the plain
    percentage gave the step 1e-9; here s_q = max_i |c_i| step_i = 1, so "1%" gives 0.01. Away from 0 nothing changes;
    an absolute step is never floored; a malformed percentage raises (the configuration reader refuses it at load)."""
    toy = Quad()
    assert float(toy.c @ toy.m) == 0.0
    assert PR.own_scale(toy, toy.m, "q") == pytest.approx(1.0, rel=1e-12)
    toy.free[1] = False  # only the parameters that move: y moves q by 0.5 per step
    assert PR.own_scale(toy, toy.m, "q") == pytest.approx(0.5, rel=1e-12)
    toy.free[1] = True
    out = PR.family_profiles(toy, _cfg({"q": "1%"}, max_steps=1), {"nodes": [{"x": toy.m.tolist(), "neg_log_l": 0.0}]},
                             level=5.0, h=5.0, quantities=["q"])
    assert out["q"]["grid_step"] == pytest.approx(0.01, rel=1e-12)
    assert PR.grid_step("dipole_MG", 300.0, {"dipole_MG": "1%"}, 20.0) == pytest.approx(3.0)  # |q_best| > s_q
    assert PR.grid_step("dipole_MG", 1e-12, {"dipole_MG": "1%"}, 20.0) == pytest.approx(0.2)
    assert PR.grid_step("dipole_MG", 0.0, {"dipole_MG": "2.5%"}, lambda: 4.0) == pytest.approx(0.1)
    assert PR.grid_step("incl_deg", 0.0, {"incl_deg": 2.5}, 99.0) == 2.5
    for bad in ("two%", "-5%", "0%"):
        with pytest.raises(ValueError, match="dipole_MG"):
            PR.grid_step("dipole_MG", 300.0, {"dipole_MG": bad}, 20.0)


class Rough(Bounded):
    """The toy of class Bounded without the limit (p free in [-50, 50]) and with a rough quantity
    q = p + a s(p), s a square wave of period 3e-4 (a = 0.005): q jumps by 2 a every 1.5e-4, as a B quantile jumps when
    two neighbouring patches swap places, so the central differences of the penalty over the FD step (1e-4 steps) are
    wrong and a polish can stop short of q0 although every q0 is reachable. The profile of q is
    ((q - 0.3) / 3)^2 / 2 to within |q - 0.3| a / 9 (0.006 nats at the stop level)."""

    a, period = 0.005, 3e-4

    def __init__(self):
        super().__init__()
        self.lo = np.array([-50.0, -50.0])

    def derived_batch(self, X):
        p = np.atleast_2d(np.asarray(X, dtype=float))[:, 0]
        return {"q": p + self.a * np.sign(np.sin(2.0 * math.pi * p / self.period))}


def test_v1l_reachable_values_missed_by_a_first_polish_are_reached_by_the_retry():
    """On the rough toy (class Rough) the first polish misses the constraint at most grid values,
    although every q0 is reachable. Each miss is polished once more (LM stops off, from a start moved towards q0) and
    reaches it: no closed end, no flag, the walks reach the stop level on both sides, the envelope is the toy's profile
    to 0.03 nats (the constraint tolerance 0.1 dq at the far grid values) and the family-level interval its closed form
    to one grid step. Before the fix both sides closed within two grid steps of the best ("constraint not met" at
    q = 0.23 and 0.87; family-level interval 0.23..0.87 instead of -9.2..9.8; 2050 evaluations, now about 7e4)."""
    toy = Rough()
    h = 5.0
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}, max_steps=60), fam, level=h, h=h, quantities=["q"])["q"]
    assert e["ends"] == {"lo": None, "hi": None} and e["flags"] == []
    assert sum(w["retries"] for w in e["walks"]) > 10  # the first polish missed often
    assert all(p["met"] for w in e["walks"] for p in w["points"])
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    assert env[0] > 10.0 and env[-1] > 10.0
    np.testing.assert_allclose(env, 0.5 * ((g - 0.3) / 3.0) ** 2, rtol=0, atol=0.03)
    iv = e["intervals"]["family_level"]
    assert iv["lo"] == pytest.approx(0.3 - 3.0 * math.sqrt(2 * h), abs=0.25)
    assert iv["hi"] == pytest.approx(0.3 + 3.0 * math.sqrt(2 * h), abs=0.25)
    assert not (iv["open_lo"] or iv["open_hi"])


def _stalling_polish(monkeypatch, q0s, only_with_stops=True):
    """Make `search.polish` stall at the grid values q0s of a quantity held by the penalty: it returns the point the
    walk's first polish there started from (the previous point, as a stalled polish does), either while
    one of the LM stops of decision #37 is on (`only_with_stops`) or always. Returns the calls there as (q0, stops on).
    The walks run in this process (one worker), one after the other."""
    from zeemap import search as S

    real = S.polish
    prev = {}
    calls = []

    def fake(view, x0, cfg, trace=None):
        q0 = getattr(view, "q0", None)
        on = cfg["search"]["lm_stop_nat"] > 0 or cfg["search"]["lm_max_rejections"] > 0
        if q0 is not None and any(abs(q0 - t) < 1e-9 for t in q0s):
            calls.append((q0, on))
            if on:  # the walk's first polish there (the defaults): remember where it started
                prev["x"] = np.array(x0, dtype=float)
            if on or not only_with_stops:
                f = float(np.asarray(view.cost(prev["x"][None]))[0])
                return S.PolishResult(prev["x"].copy(), f, 1, False, False, f)
        return real(view, x0, cfg, trace)

    monkeypatch.setattr(S, "polish", fake)
    return calls


def test_v1l_stalled_polish_is_retried_with_the_lm_stops_off(monkeypatch):
    """Under decision #37's LM stops a polish on a smooth problem can stop short of a reachable q0 and
    return the previous point. On the quadratic toy (q held by the penalty, one family member) every polish at the grid
    value k = -2 stalls while an LM stop is on: the walk re-polishes that point once with both stops off, which reaches
    it, and walks on; no side is closed, no flag is set, the profile equals its closed form (to search.lm_stop_nat above
    it, 1e-6 at the retried point) down to the stop level on both sides. Before the fix the walk ended at k = -2 as a
    closed end "constraint not met" at q(k = -1), and the interval was closed there."""
    toy = Quad()
    h = 5.0
    Ainv = np.linalg.inv(toy.A)
    V, qs = float(toy.c @ Ainv @ toy.c), float(toy.c @ toy.m)
    dq = 0.25 * math.sqrt(V)
    calls = _stalling_polish(monkeypatch, [qs - 2.0 * dq])
    fam = {"nodes": [{"x": toy.m.tolist(), "neg_log_l": 0.0}]}
    cfg = _cfg({"q": dq})
    e = PR.family_profiles(toy, cfg, fam, level=h, h=h, quantities=["q"])["q"]
    assert e["ends"] == {"lo": None, "hi": None} and e["flags"] == []
    assert any(on for _, on in calls) and any(not on for _, on in calls)  # stalled with the stops on, retried without
    g, env = np.array(e["grid"]), np.array(e["envelope_profile"])
    eps = dq * dq / (2.0 * PR.KAPPA * V)
    d = env - 0.5 * (g - qs) ** 2 / V / (1.0 + eps) ** 2
    assert np.all(d >= -1e-6) and np.all(d <= cfg["search"]["lm_stop_nat"])
    k2 = int(np.argmin(np.abs(g - (qs - 2.0 * dq))))
    assert g[k2] == pytest.approx(qs - 2.0 * dq) and abs(d[k2]) <= 1e-6  # the retried point, polished with stops off
    assert all(p["met"] and not p["stalled"] for w in e["walks"] for p in w["points"])
    assert env[0] > 10.0 and env[-1] > 10.0  # the walks reach the stop level on both sides
    assert sum(w["retries"] for w in e["walks"]) == 3  # the first walk and both walks back retried k = -2 once
    iv = e["intervals"]["family_level"]
    assert iv["lo"] == pytest.approx(qs - math.sqrt(2.0 * h * V), abs=dq) and not (iv["open_lo"] or iv["open_hi"])


def test_v1l_polish_that_stalls_again_keeps_the_side_open_and_the_walk_goes_on(monkeypatch):
    """When the retry stalls too and nothing shows a limit (q is linear in two free parameters, no
    polish limit near), the point is written with met = false and stalled = true, the flag "polish stalled" is set and
    "constraint not met" is not, no end is closed, and the walk goes on from the last point that met the constraint: the
    grid continues past k = -2 down to the stop level (in the first walk and in the walk back, which therefore does not
    end at the same stall). The envelope has a gap at k = -2, so the intervals that reach it are open there ('<' in the
    side-by-side table), not closed at a value the model can pass."""
    from zeemap.report import side_by_side_text

    toy = Quad()
    h = 5.0
    Ainv = np.linalg.inv(toy.A)
    V, qs = float(toy.c @ Ainv @ toy.c), float(toy.c @ toy.m)
    dq = 0.25 * math.sqrt(V)
    _stalling_polish(monkeypatch, [qs - 2.0 * dq], only_with_stops=False)
    fam = {"nodes": [{"x": toy.m.tolist(), "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": dq}), fam, level=h, h=h, quantities=["q"])["q"]
    ks = [round((q - qs) / dq) for q in e["grid"]]
    assert e["ends"] == {"lo": None, "hi": None} and -3 in ks
    assert PR.STALLED in e["flags"] and PR.NOT_MET not in e["flags"]
    assert -2 not in ks and -1 in ks and -3 in ks and e["envelope_profile"][0] > 10.0
    for w in e["walks"]:
        pts = _points(w)
        if -2 in pts:
            assert not pts[-2]["met"] and pts[-2]["stalled"] and w["ends"] == {"lo": None, "hi": None}
            assert min(pts) < -2  # it went on past the stall
        assert all(p["met"] for k, p in pts.items() if k != -2)
    assert [w["start"]["role"] for w in e["walks"]] == ["best", "walk back", "walk back"]
    assert all(-2 in _points(w) for w in e["walks"])
    for key in ("delta_0.5", "delta_2", "family_level"):  # all reach k = -2 (0.125 nats): open at the gap
        iv = e["intervals"][key]
        assert iv["open_lo"] and iv["lo"] == pytest.approx(qs - dq) and not iv["open_hi"], (key, iv)
    lines = side_by_side_text({1: {"best_mode": 1, "converged": None, "quantities": {"q": e}}}, {"h": h}, None)
    row = next(r for r in lines if r.split()[:2] == ["1", "q"])
    assert PR.STALLED in row and PR.NOT_MET not in row and "<" in row
    assert "polish stalled" in lines[0]  # the legend explains the flag


def test_v1l_closed_end_needs_a_limit_and_the_flag_follows_the_ends():
    """On the toy q = p >= 0 (class Bounded) the walk down closes at the polish limit (p sits on
    its lower limit and y does not move q, so no free parameter moves q towards q0 < 0), with no stalled point and
    without a retry, which cannot pass a limit. The flag "constraint not met" is set exactly
    when `ends` holds such an end: on every profile of this module's toys the two agree."""
    toy = Bounded()
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}), fam, level=5.0, h=5.0, quantities=["q"])["q"]
    assert e["ends"]["lo"]["reason"] == PR.NOT_MET and e["ends"]["lo"]["q"] == pytest.approx(0.0, abs=1e-6)
    assert e["flags"] == [PR.NOT_MET]
    assert not any(p["stalled"] for w in e["walks"] for p in w["points"])
    bad = [p for w in e["walks"] for p in w["points"] if not p["met"]]  # each polished once, then closed
    assert len(bad) >= 1 and sum(w["retries"] for w in e["walks"]) == 0
    assert all(p["q0"] == pytest.approx(-0.2) and p["q"] == pytest.approx(0.0, abs=1e-6) for p in bad)
    others = [PR.family_profiles(Branches(), _cfg({"teff_K": 2.5}), _branch_family(Branches()), level=6.0, h=6.0,
                                 quantities=["teff_K"])["teff_K"], e]
    for x in others:
        closed = any(v is not None and v["reason"] == PR.NOT_MET for v in x["ends"].values())
        assert (PR.NOT_MET in x["flags"]) == closed


def test_v1l_b_quantile_profiles_are_flagged_provisional():
    """Decision #39: the B quantiles (a step function of the parameters at a finite patch count) carry the flag
    "B quantile is a step function: provisional"; |m_d|, the inclination, Teff and c do not."""
    for name in ("B10_MG[0]", "B50_MG[1]", "B90_MG[3]"):
        assert PR.provisional_flags(name) == [PR.QUANTILE_STEP]
    for name in ("dipole_MG", "incl_deg", "beta_deg", "teff_K", "c_along_axis"):
        assert PR.provisional_flags(name) == []


def test_v1l_polish_short_of_a_bound_closes_by_the_second_limit_test(monkeypatch):
    """On the toy q = p >= 0 (class Bounded) a polish at q0 <= -0.2 that stops just short of the
    bound, at p = 0.01 (planted: every polish there returns it), leaves p free to move q towards q0, so the first limit
    test of `_towards` (no free parameter moves q) does not fire; the second one does (the search towards q0 reaches the
    polish limits of every parameter that moves q, here p = 0, still more than 0.1 dq short). The walk down closes at
    q = 0.01 with "constraint not met", no point stalled. Without the second test (mutant `limit = False`) the side was
    open, marked "polish stalled"."""
    from zeemap import search as S

    real = S.polish
    planted = []

    def fake(view, x0, cfg, trace=None):
        q0 = getattr(view, "q0", None)
        if q0 is not None and q0 <= -0.2 + 1e-9:
            planted.append(q0)
            x = np.array([0.01, 0.005])
            f = float(np.asarray(view.cost(x[None]))[0])
            return S.PolishResult(x, f, 1, True, False, f)
        return real(view, x0, cfg, trace)

    monkeypatch.setattr(S, "polish", fake)
    toy = Bounded()
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}), fam, level=5.0, h=5.0, quantities=["q"])["q"]
    assert planted and all(q0 == pytest.approx(-0.2) for q0 in planted)
    assert e["ends"]["lo"]["reason"] == PR.NOT_MET and e["ends"]["lo"]["q"] == pytest.approx(0.01)
    assert e["flags"] == [PR.NOT_MET]
    assert not any(p["stalled"] for w in e["walks"] for p in w["points"])
    iv = e["intervals"]["family_level"]
    assert iv["lo"] == pytest.approx(0.01) and not iv["open_lo"]


def test_v1l_direction_ends_after_max_stalls_stalled_points(monkeypatch):
    """`MAX_STALLS` = 2 stalled points in a row end a walk direction, open. On the quadratic toy
    every polish at k = -2 and k = -3 stalls (also the retry): each walk down writes both as stalled and ends there, so
    no walk reaches k = -4 and the lower side of every interval that reaches k = -1 is open there. With 3 the walks
    went on to k = -4 (which a polish reaches) and below."""
    toy = Quad()
    h = 5.0
    V, qs = float(toy.c @ np.linalg.inv(toy.A) @ toy.c), float(toy.c @ toy.m)
    dq = 0.25 * math.sqrt(V)
    _stalling_polish(monkeypatch, [qs - 2.0 * dq, qs - 3.0 * dq], only_with_stops=False)
    fam = {"nodes": [{"x": toy.m.tolist(), "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": dq}), fam, level=h, h=h, quantities=["q"])["q"]
    assert PR.MAX_STALLS == 2
    assert e["ends"] == {"lo": None, "hi": None} and PR.STALLED in e["flags"] and PR.NOT_MET not in e["flags"]
    down = [w for w in e["walks"] if -2 in _points(w)]
    assert down and min(k for w in e["walks"] for k in _points(w)) == -3
    for w in down:
        pts = _points(w)
        assert min(pts) == -3 and all(pts[k]["stalled"] and not pts[k]["met"] for k in (-2, -3))
    assert min(round((q - qs) / dq) for q in e["grid"]) == -1
    for key in ("delta_0.5", "delta_2", "family_level"):
        iv = e["intervals"][key]
        assert iv["open_lo"] and iv["lo"] == pytest.approx(qs - dq), (key, iv)


def test_v1l_no_retry_where_the_first_search_finds_a_limit():
    """Regression: after a missed constraint `walk` once called `_towards` twice, before the retry (for its start) and
    after it (for the limit), and ran the retry even when the first call had already found that q cannot reach q0.
    The first call's verdict is kept: on the toy q = p >= 0 (class Bounded, the walk down misses q0 = -0.2 with p on
    its limit) the end closes at once, with no retry (it was one per walk that reached q0 = -0.2), and the
    evaluations of the profile fall."""
    toy = Bounded()
    fam = {"nodes": [{"x": [0.3, 0.15], "neg_log_l": 0.0}]}
    e = PR.family_profiles(toy, _cfg({"q": 0.25}), fam, level=5.0, h=5.0, quantities=["q"])["q"]
    assert e["ends"]["lo"]["reason"] == PR.NOT_MET and e["ends"]["lo"]["q"] == pytest.approx(0.0, abs=1e-6)
    assert sum(w["retries"] for w in e["walks"]) == 0
    assert e["nevals"] < 10503
