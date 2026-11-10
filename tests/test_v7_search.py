"""V7c (search part), V4g and search determinism on toy problems (DESIGN.md section 3).

The toy CostProblem: cost(x) = -logsumexp_k(-d_k - |x - m_k|^2 / (2 s_k^2)), wells at known centres m_k with known
depths d_k. The wells are far apart compared with their widths, so near m_k the cost is d_k + |x - m_k|^2 / (2 s_k^2)
to within ~1e-14: the minima are at m_k with values d_k, and the residuals of the dominant well, (x - m_k) / s_k, give
cost = d_k + |r|^2 / 2 there, so the Levenberg-Marquardt finish runs. The fingerprint is a linear map of x.
(V4g's tests live here because they need the same toy.)
"""

import copy
import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from zeemap import search as S
from zeemap.config import defaults

pytestmark = pytest.mark.tier1


class Wells:
    def __init__(self, centres, depths, widths, names=None, lo=-10.0, hi=10.0):
        self.m = np.asarray(centres, dtype=float)
        self.d = np.asarray(depths, dtype=float)
        self.s = np.asarray(widths, dtype=float)
        self.dim = self.m.shape[1]
        self.names = list(names) if names else [f"p{i}" for i in range(self.dim)]
        self.free = np.ones(self.dim, dtype=bool)
        self.lo = np.full(self.dim, lo)
        self.hi = np.full(self.dim, hi)
        self.step = np.ones(self.dim)
        self.value = np.zeros(self.dim)
        self.field_slice = None
        self.ncost = 0  # rows of cost calls
        self.nres = 0  # residual calls

    def _terms(self, X):
        q = 0.5 * np.sum((X[:, None, :] - self.m[None]) ** 2, axis=2) / self.s[None] ** 2
        return -self.d[None] - q

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.ncost += X.shape[0]
        return -logsumexp(self._terms(X), axis=1)

    def residuals(self, x):
        self.nres += 1
        k = int(np.argmax(self._terms(np.asarray(x, dtype=float)[None])[0]))
        return (x - self.m[k]) / self.s[k]

    def canonical(self, x):
        return np.array(x, dtype=float)

    def fingerprint_quantities(self, x):
        return {"incl": 40.0 + 3.0 * x[0], "c": 0.02 * np.asarray(x[1:4]),
                "b50": np.array([100.0 + 2.0 * x[0] + x[3], 150.0 + x[1]]),
                "b90": np.array([300.0 + 5.0 * x[2], 320.0])}


THREE = dict(centres=[[-4, 3, 2, -1], [5, -4, -3, 2], [-3, -7, 7, 6]], depths=[0.0, 5.0, 60.0], widths=[1.0, 1.0, 3.0])


def three_wells(**kw):
    return Wells(**THREE, **kw)


@pytest.fixture(scope="module")
def cfg():
    return defaults()


@pytest.fixture(scope="module")
def three_result(cfg):
    p = three_wells()
    return p, S.run_search(p, cfg, seed=1, n_starts=40)


# ---------------------------------------------------------------------------------------------------------------------
# V7c


def test_v7c_every_start_recorded_and_hits_listed(three_result):
    p, r = three_result
    round0 = [e for e in r.endpoints if e.round == 0]
    assert [e.start for e in round0] == list(range(40))
    assert all(e.x0 is not None and e.fingerprint is not None for e in round0)
    assert sum(m.hits for m in r.all_modes) == 40
    for k, m in enumerate(r.all_modes):
        assert m.hits == sum(1 for i in m.members if r.endpoints[i].round == 0)
        assert all(r.endpoints[i].mode == k for i in m.members)
    assert r.stats["n_starts"] == 40 and r.stats["n_modes"] == len(r.all_modes) == 3
    assert r.stats["hits_best"] == r.all_modes[0].hits > 0


def test_v7c_band_holds_exactly_the_two_shallow_wells(three_result):
    p, r = three_result
    assert len(r.all_modes) == 3  # the 60-nat well is found ...
    assert r.all_modes[2].f == pytest.approx(60.0, abs=1e-6)
    assert len(r.modes) == 2  # ... but lies outside the 45-nat band
    for m, centre, depth in zip(r.modes, p.m[:2], p.d[:2]):
        assert np.max(np.abs(m.x - centre)) < 1e-3
        assert m.f == pytest.approx(depth, abs=1e-6)
    assert np.max(np.abs(r.modes[0].x - p.m[0])) < 1e-6  # best mode: L-BFGS-B + LM on a smooth well
    assert not any(m.touched_limit for m in r.modes)


def test_v7c_start_distribution_and_json(three_result):
    p, r = three_result
    sd = r.start_distribution
    assert sd["kind"] == "uniform" and sd["uniform"] == {n: [-10.0, 10.0] for n in p.names}
    d = json.loads(json.dumps(r.to_dict()))
    assert d["start_distribution"] == sd
    assert len(d["endpoints"]) == len(r.endpoints) and len(d["modes"]) == 2
    assert d["stats"]["value_tolerance"] == 2.0  # scatter ~ 0 on exact wells -> the 2-nat floor
    assert 0.0 <= r.stats["optimiser_scatter"] < 1e-6


def test_v7c_restart_rounds(three_result, cfg):
    p, r = three_result
    later = [e for e in r.endpoints if e.round >= 1]
    n_parents = min(cfg["search"]["top"], 3)
    assert len(later) == r.stats["rounds"] * n_parents * cfg["search"]["restarts"]
    assert r.stats["rounds"] == 1  # the first round gains nothing on exact wells, so the search stops
    assert all(e.parent is not None for e in later)


def test_v7c_evaluations_are_counted(cfg):
    p = three_wells()
    r = S.run_search(p, cfg, seed=3, n_starts=10)
    assert r.nevals == p.ncost + p.nres == r.stats["nevals"]
    assert sum(e.nevals for e in r.endpoints) <= r.nevals


class Bowl:
    """cost = |A (x - x*)|^2 / 2 with residuals A (x - x*); Gauss-Newton sigma_i = sqrt(((A^T A)^-1)_ii)."""

    def __init__(self, residuals=True):
        rng = np.random.default_rng(7)
        self.dim = 5
        self.A = np.diag([3.0, 1.0, 0.5, 2.0, 0.2]) + 0.3 * rng.standard_normal((5, 5))
        self.xs = np.array([0.3, -1.2, 2.0, 0.7, -0.4])
        self.names = [f"q{i}" for i in range(5)]
        self.free = np.ones(5, dtype=bool)
        self.lo, self.hi = np.full(5, -10.0), np.full(5, 10.0)
        self.step = np.array([0.1, 0.3, 0.5, 0.2, 1.0])
        self.value = np.zeros(5)
        self.field_slice = None
        self.with_res = residuals
        self.sigma = np.sqrt(np.diag(np.linalg.inv(self.A.T @ self.A)))

    def cost(self, X):
        R = (np.atleast_2d(X) - self.xs) @ self.A.T
        return 0.5 * np.sum(R * R, axis=1)

    def residuals(self, x):
        if not self.with_res:
            raise NotImplementedError
        return self.A @ (np.asarray(x) - self.xs)

    def canonical(self, x):
        return np.array(x, dtype=float)

    def fingerprint_quantities(self, x):
        return {"incl": 45.0, "c": np.zeros(3), "b50": np.zeros(1), "b90": np.zeros(1)}


def test_v7c_polish_reaches_bowl_minimum_within_gauss_newton_sigma(cfg):
    p = Bowl()
    x0 = p.xs + 3.0 * p.sigma * np.array([1, -1, 1, 1, -1])
    pr = S.polish(p, x0, cfg)
    assert pr.lm_used and pr.converged and not pr.touched_limit
    assert np.all(np.abs(pr.x - p.xs) < 0.1 * p.sigma)
    assert pr.f <= pr.f_lbfgs


def test_v7c_lm_finish_does_the_work_when_lbfgs_is_cut_short(cfg):
    c = copy.deepcopy(cfg)
    c["search"]["lbfgs_maxiter"] = 1
    p = Bowl()
    x0 = p.xs + 3.0 * p.sigma
    pr = S.polish(p, x0, c)
    assert pr.lm_used and pr.lm_iterations >= 1 and pr.f < pr.f_lbfgs
    assert np.all(np.abs(pr.x - p.xs) < 0.1 * p.sigma)


def test_v7c_polish_without_residuals_skips_lm(cfg):
    p = Bowl(residuals=False)
    pr = S.polish(p, p.xs + 2.0 * p.sigma, cfg)
    assert not pr.lm_used and pr.lm_iterations == 0 and pr.f == pr.f_lbfgs
    assert np.all(np.abs(pr.x - p.xs) < 0.1 * p.sigma)


def test_v7c_polish_counts_batched_gradient_rows(cfg):
    class Counting(Bowl):
        calls = []

        def cost(self, X):
            self.calls.append(np.atleast_2d(X).shape[0])
            return super().cost(X)

    p = Counting(residuals=False)
    pr = S.polish(p, p.xs + p.sigma, cfg)
    assert set(p.calls) == {2 * 5 + 1}  # every L-BFGS-B evaluation is one batch: the point and the +- points
    assert pr.nevals == sum(p.calls)


def test_v7c_polish_touched_limit(cfg):
    p = Bowl(residuals=False)
    p.hi = np.full(5, 1.0)  # x*_2 = 2 lies beyond the limit: the end point sits on it
    pr = S.polish(p, np.zeros(5), cfg)
    assert pr.touched_limit and pr.x[2] == pytest.approx(1.0)


class Misled:
    """Residuals x - 1 whose Gauss-Newton model points uphill. -ln L = |x - 1|^2 / 2 + 3 sum(x) (the
    linear term stands in for the 0.5 ln det continuum term, which the LM residuals leave out), so from x = 0 every LM
    step towards 1 raises -ln L and is rejected; the true minimum is at x = -2."""

    def __init__(self, n=2):
        self.dim = n
        self.names = [f"q{i}" for i in range(n)]
        self.free = np.ones(n, dtype=bool)
        self.lo, self.hi = np.full(n, -10.0), np.full(n, 10.0)
        self.step = np.ones(n)
        self.value = np.zeros(n)
        self.field_slice = None

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return 0.5 * np.sum((X - 1.0) ** 2, axis=1) + 3.0 * np.sum(X, axis=1)

    def residuals(self, x):
        return np.asarray(x, dtype=float) - 1.0

    def canonical(self, x):
        return np.array(x, dtype=float)

    def fingerprint_quantities(self, x):
        return {"incl": 45.0, "c": np.zeros(3), "b50": np.zeros(1), "b90": np.zeros(1)}


class Banana(Misled):
    """Rosenbrock residuals (10 (x1 - x0^2), 1 - x0): the LM finish creeps along the valley with shrinking gains."""

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return 0.5 * ((10.0 * (X[:, 1] - X[:, 0] ** 2)) ** 2 + (1.0 - X[:, 0]) ** 2)

    def residuals(self, x):
        return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def test_v7c_lm_lambda_runaway_is_not_converged(cfg):
    """Before the fix the exit after lam passed 1e12 set `converged`, here 16 rejected steps from a point 2 units from
    the minimum, with the LM alone (search.lbfgs = false)."""
    c = copy.deepcopy(cfg)
    c["search"].update(lbfgs=False, lm_max_rejections=0)  # the rejection cap (decision #37) would end it first
    pr = S.polish(Misled(), np.zeros(2), c)
    assert pr.lm_used and pr.lm_stop == "lambda" and not pr.converged
    assert pr.lm_iterations == 16 and pr.f == 1.0  # no step was taken: f(0) = 2 x |0 - 1|^2 / 2


def test_v7c_lm_stop_keys_defaults_and_their_effect(cfg):
    """`search.lm_stop_nat` and `search.lm_max_rejections` (defaults.toml) default to 1e-3 nat and 5 (decision #37);
    set to 0 (off) the LM finish runs as with the keys absent, i.e. as before them. A rejection cap of 5 ends the
    runaway after 5 rejected steps (not converged); an absolute stop of 1e-3 nat ends the Rosenbrock creep early
    (converged), less than that 1e-3 nat above the end point without it."""
    assert cfg["search"]["lm_stop_nat"] == 1e-3 and cfg["search"]["lm_max_rejections"] == 5
    c = copy.deepcopy(cfg)
    c["search"].update(lbfgs=False, lm_stop_nat=0.0, lm_max_rejections=0)
    keys_absent = copy.deepcopy(c)
    del keys_absent["search"]["lm_stop_nat"], keys_absent["search"]["lm_max_rejections"]
    x0 = np.array([-1.2, 1.0])
    base = S.polish(Banana(), x0, c)
    same = S.polish(Banana(), x0, keys_absent)
    assert (base.f, base.nevals, base.lm_iterations, base.lm_stop) == (same.f, same.nevals, same.lm_iterations,
                                                                      same.lm_stop)
    np.testing.assert_array_equal(base.x, same.x)
    assert base.converged and base.f < 1e-10

    capped = copy.deepcopy(c)
    capped["search"]["lm_max_rejections"] = 5
    pr = S.polish(Misled(), np.zeros(2), capped)
    run = S.polish(Misled(), np.zeros(2), c)
    assert pr.lm_stop == "rejections" and not pr.converged and pr.lm_iterations == 5
    assert pr.nevals < run.nevals and pr.f == run.f

    stop = copy.deepcopy(c)
    stop["search"]["lm_stop_nat"] = 1e-3
    pr = S.polish(Banana(), x0, stop)
    assert pr.converged and pr.lm_stop in ("gain", "predicted")
    assert pr.lm_iterations < base.lm_iterations and pr.f - base.f < stop["search"]["lm_stop_nat"]


class Blind(Misled):
    """Residuals (x0 - 1) that do not see x1; -ln L = (x0 - 1)^2 / 2 + x1^4. At x0 = 1 the Gauss-Newton model predicts
    no gain at all, while -ln L still falls along x1 (the reverse case of Misled)."""

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return 0.5 * (X[:, 0] - 1.0) ** 2 + X[:, 1] ** 4

    def residuals(self, x):
        return np.array([x[0] - 1.0])


def test_v7c_converged_is_the_lm_verdict_only_after_an_accepted_step(cfg):
    """Regression: `converged` once took the LM's verdict whenever the LM ran an iteration, also when it accepted no
    step, so the end point was L-BFGS-B's but the flag the LM's. (1) L-BFGS-B reaches the minimum of Misled (x = -2,
    converged); the LM then rejects 5 steps (the Gauss-Newton model points uphill): `lm_stop` "rejections", the end
    point L-BFGS-B's, and `converged` L-BFGS-B's True (it was False). (2) The reverse case: L-BFGS-B stops at its
    iteration limit (not converged) at x0 = 1, where the LM's first Jacobian predicts no gain: `lm_stop` "predicted",
    no step, and `converged` stays L-BFGS-B's False (it was True), since the Gauss-Newton model of the residuals
    is not a convergence test for -ln L by itself (it leaves out the continuum log-det). With `search.lm_stop_nat` > 0
    "an accepted step" reads "a total LM gain above `search.lm_stop_nat`" (next test); (3) with that stop off (0) the
    rule is "an accepted step"."""
    pr = S.polish(Misled(), np.zeros(2), cfg)
    assert pr.lm_used and pr.lm_stop == "rejections" and pr.lm_iterations == 5
    assert pr.f == pr.f_lbfgs and pr.f == pytest.approx(-3.0, abs=1e-6)  # 2 x 3^2 / 2 + 3 x (-4) at x = -2
    assert pr.converged

    c = copy.deepcopy(cfg)
    c["search"]["lbfgs_maxiter"] = 1
    pr = S.polish(Blind(), np.array([1.0, 2.0]), c)
    assert pr.lm_used and pr.lm_stop == "predicted" and pr.lm_iterations == 1 and pr.f == pr.f_lbfgs
    assert pr.f > 1e-3  # x1 is not at its minimum: L-BFGS-B ran out of iterations
    assert not pr.converged

    # (3) with the LM stops off (search.lm_stop_nat = 0) the accepted-step rule holds unchanged, an LM that
    # accepted no step leaves L-BFGS-B's flag and one that accepted a step decides
    off = copy.deepcopy(cfg)
    off["search"]["lm_stop_nat"] = 0.0
    pr = S.polish(Misled(), np.zeros(2), off)
    assert pr.lm_stop == "rejections" and pr.f == pr.f_lbfgs and pr.converged
    pr = S.polish(Dented(), np.array([-3.0]), off)
    assert pr.lm_stop == "rejections" and 0.0 < pr.f_lbfgs - pr.f and not pr.converged


class Dented(Misled):
    """Misled in one coordinate with a narrow dent at x = -0.5, where the LM's step from the minimum
    x = -2 lands at lam = 1 (the third rejection raised lam from 1e-3 to 1): -ln L = |x - 1|^2 / 2 + 3 x - (1.125 +
    `gain`) exp(-(x + 0.5)^2 / (2 0.01^2)). L-BFGS-B, which never comes near the dent from x = -3, converges at -2
    (-ln L = -1.5); the LM accepts the step into the dent, gaining `gain`, and then rejects 5 steps (the Gauss-Newton
    model of x - 1 still predicts 1.125 nat): `lm_stop` "rejections" after an accepted step."""

    def __init__(self, gain=5e-4):
        super().__init__(1)
        self.depth = 1.125 + gain

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return super().cost(X) - self.depth * np.exp(-0.5 * ((X[:, 0] + 0.5) / 0.01) ** 2)


def test_v7c_converged_after_an_lm_gain_below_lm_stop_nat_is_lbfgs_verdict(cfg):
    """Regression: an LM that accepted any step once decided `converged`, and on the full objective
    the Gauss-Newton prediction never falls below `search.lm_stop_nat` (decision #37), so the LM ended on
    "rejections" and the flag read False for exactly the polishes the LM improved (synth_single, 20 starts: 8 of 20).
    Now the LM's stop decides only when its total gain over L-BFGS-B's end point exceeds `search.lm_stop_nat` (or
    L-BFGS-B did not run). (1) L-BFGS-B converged, the LM accepts a step of 5e-4 nat, then hits the rejection cap:
    converged True (it was False). (2) The same with a gain of 1e-2 nat > lm_stop_nat: the LM decides, False."""
    pr = S.polish(Dented(), np.array([-3.0]), cfg)
    assert pr.lm_used and pr.lm_stop == "rejections" and pr.lm_iterations == 9  # 3 rejections, 1 accept, 5 rejections
    assert pr.f_lbfgs == pytest.approx(-1.5, abs=1e-9) and pr.f_lbfgs - pr.f == pytest.approx(5e-4, rel=1e-3)
    assert pr.converged

    pr = S.polish(Dented(gain=1e-2), np.array([-3.0]), cfg)
    assert pr.lm_stop == "rejections" and pr.f_lbfgs - pr.f == pytest.approx(1e-2, rel=1e-3)
    assert not pr.converged

    # L-BFGS-B off: the LM's stop is the only verdict, whatever the gain
    c = copy.deepcopy(cfg)
    c["search"]["lbfgs"] = False
    pr = S.polish(Dented(), np.array([-2.0]), c)
    assert pr.lm_stop == "rejections" and pr.f < pr.f_lbfgs and not pr.converged


def test_v7c_lm_small_gain_after_damping_is_not_a_stop(cfg):
    """Regression: an accepted step that gained less than `search.lm_stop_nat` ended the LM ("gain", converged)
    also when rejected steps had raised the damping lam, so that the damped step was short while the undamped
    Gauss-Newton step still predicted far more: Rosenbrock, LM alone, from (-0.0803, -0.0712) it stopped at -ln L =
    0.148, all of which the undamped step still predicted (7e-22 with the stops off). The small-gain stop is gone; a
    fresh Jacobian's prediction decides. Over 200 seeded starts every converged LM end point lies within
    `search.lm_stop_nat` of the end point with the stops off (before the fix 3 of them did not: 0.25, 0.025, 0.021)."""
    c = copy.deepcopy(cfg)
    c["search"]["lbfgs"] = False
    off = copy.deepcopy(c)
    off["search"].update(lm_stop_nat=0.0, lm_max_rejections=0)
    stop_nat = c["search"]["lm_stop_nat"]
    pr = S.polish(Banana(), np.array([-0.0803, -0.0712]), c)
    ref = S.polish(Banana(), np.array([-0.0803, -0.0712]), off)
    assert ref.f < 1e-15
    assert pr.converged and pr.f - ref.f < stop_nat, (pr.f, pr.lm_stop)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x0 = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 3.0)])
        pr = S.polish(Banana(), x0, c)
        if pr.converged:
            assert pr.f - S.polish(Banana(), x0, off).f < stop_nat, (x0, pr.f, pr.lm_stop)


def test_v7c_lm_trace_has_every_iteration(cfg, monkeypatch):
    """An LM iteration that ends in LinAlgError (or in any stop) writes its trace entry, so that
    tools/measure_lm_shortfall.py reads f at every iteration number."""
    c = copy.deepcopy(cfg)
    c["search"]["lbfgs"] = False
    solve = np.linalg.solve
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 2:
            raise np.linalg.LinAlgError("planted")
        return solve(a, b)

    monkeypatch.setattr(S.np.linalg, "solve", flaky)
    trace: list = []
    pr = S.polish(Banana(), np.array([-1.2, 1.0]), c, trace=trace)
    assert [t[0] for t in trace] == list(range(1, pr.lm_iterations + 1))
    assert trace[1][1] == trace[0][1]  # the planted LinAlgError: no step, f unchanged
    assert trace[-1][1] == pr.f


def test_v7c_invalid_starts_are_redrawn(cfg):
    class HalfInvalid(Wells):
        def cost(self, X):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            out = super().cost(X)
            return np.where(X[:, 0] > 0.0, S.WORST, out)

    p = HalfInvalid(**THREE)
    X = S.draw_starts(p, cfg, S.start_streams(5, 40), 40)
    assert X.shape == (40, 4) and np.all(X[:, 0] <= 0.0)
    assert np.all((X >= -10) & (X <= 10))
    assert p.ncost > 40  # some draws were rejected and redrawn

    class AllInvalid(Wells):
        def cost(self, X):
            return np.full(np.atleast_2d(X).shape[0], S.WORST)

    with pytest.raises(S.SearchError, match="100 draws"):
        S.draw_starts(AllInvalid(**THREE), cfg, 0, 2)


def test_v7c_prior_cut_redraws_starts(cfg):
    class Cut(Wells):
        def outside_fraction(self, x):
            return np.array([0.0, 0.5 if x[1] > 0 else 0.0])  # per phase; the largest counts

    X = S.draw_starts(Cut(**THREE), cfg, 9, 30)
    assert np.all(X[:, 1] <= 0.0)


def test_v7c_field_start_distribution(cfg):
    """Starts of a 14-number dqo layout plus teff, and the polish safety limits (DESIGN.md section 3)."""
    from zeemap.field import DQO

    class FieldToy:
        names = DQO.names + ["teff"]
        dim = 15
        free = np.ones(15, dtype=bool)
        lo = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, -1, 15000.0])
        hi = np.array([1200] * 4 + [2.0] * 3 + [1200] * 3 + [0.4] * 3 + [1.0, 25000.0])
        step = np.ones(15)
        value = np.zeros(15)
        field_slice = slice(0, 14)

        def cost(self, X):
            return np.zeros(np.atleast_2d(X).shape[0])

    p = FieldToy()
    p.free = p.free.copy()
    p.free[5] = False  # aq_y fixed
    p.value = p.value.copy()
    p.value[5] = 0.25
    X = S.draw_starts(p, cfg, 11, 400)
    st = cfg["search"]["starts"]
    md, aq, mo, c = (np.linalg.norm(X[:, a:a + 3], axis=1) for a in (0, 4, 7, 10))
    assert np.all((md >= st["md"][0]) & (md <= st["md"][1]))
    assert np.all(np.abs(X[:, 3]) <= st["bq"]) and np.any(X[:, 3] < 0)
    assert np.all(X[:, 5] == 0.25)
    assert np.all(mo <= st["mo"]) and np.all(c <= st["c"])
    assert np.all((X[:, 13] >= st["u"][0]) & (X[:, 13] <= st["u"][1]))
    assert np.all((X[:, 14] >= 15000) & (X[:, 14] <= 25000))
    # uniform direction: mean of the md directions near 0; c uniform in the ball: P(|c| < R/2) = 1/8
    assert np.all(np.abs(np.mean(X[:, 0:3] / md[:, None], axis=0)) < 0.15)
    assert abs(np.mean(c < st["c"] / 2) - 0.125) < 0.06
    blo, bhi = S.polish_bounds(p)
    assert np.array_equal(blo[:13], -p.hi[:13]) and np.array_equal(bhi[:13], p.hi[:13])
    assert (blo[13], bhi[13]) == (-1.0, 1.0) and (blo[14], bhi[14]) == (15000.0, 25000.0)
    sd = S.start_distribution(p, cfg)
    assert sd["kind"] == "field" and sd["uniform"] == {"teff": [15000.0, 25000.0]} and sd["fixed"] == {"aq_y": 0.25}
    assert sd["field"]["md"] == [200.0, 600.0]
    # perturbations keep fixed parameters and stay inside the safety limits
    x = X[0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = S.perturb(p, x, cfg, rng)
        assert y[5] == 0.25 and np.all((y >= blo) & (y <= bhi))
        assert abs(np.linalg.norm(y[0:3]) / np.linalg.norm(x[0:3]) - 1) < 0.5


def test_v7c_same_seed_same_result_bitwise(cfg):
    a = S.run_search(three_wells(), cfg, seed=2, n_starts=12)
    b = S.run_search(three_wells(), cfg, seed=2, n_starts=12)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())
    for ea, eb in zip(a.endpoints, b.endpoints, strict=True):
        assert np.array_equal(ea.x, eb.x) and ea.f == eb.f
    c = S.run_search(three_wells(), cfg, seed=4, n_starts=12)
    assert not np.array_equal(a.endpoints[0].x0, c.endpoints[0].x0)


# ---------------------------------------------------------------------------------------------------------------------
# V4g: the SEARCH-4 warning


class ManyWells(Wells):
    """12 wells within 10 nats; start k is drawn next to well k, so every start finds its own well."""

    start_description = "one start next to each well, in turn"

    def __init__(self):
        rng = np.random.default_rng(12)
        centres = []
        while len(centres) < 12:  # well-separated centres in [-8, 8]^4
            c = rng.uniform(-8, 8, 4)
            if all(np.linalg.norm(c - o) > 6 for o in centres):
                centres.append(c)
        super().__init__(centres, np.linspace(0.0, 9.0, 12), np.full(12, 0.8))
        self._next = 0

    def rng_draw_start(self, rng):
        k = self._next % 12
        self._next += 1
        return self.m[k] + 0.3 * rng.standard_normal(self.dim)

    def fingerprint_quantities(self, x):
        return {"incl": 40.0 + 3.0 * x[0], "c": 0.05 * np.asarray(x[1:4]), "b50": np.array([100.0 + 3 * x[3]]),
                "b90": np.array([300.0])}


def test_v4g_warning_when_most_band_modes_were_reached_once(cfg):
    c = copy.deepcopy(cfg)
    c["search"]["restarts"] = 2
    r = S.run_search(ManyWells(), c, seed=1, n_starts=12)
    assert r.stats["n_band_modes"] == 12 and all(m.hits == 1 for m in r.modes)
    assert r.stats["frac_band_once"] == 1.0 and r.stats["warning"] is True
    assert "cannot be bounded" in r.stats["warning_text"]
    assert r.start_distribution["kind"] == "problem"


def test_v4g_no_warning_with_three_wells_and_40_starts(three_result):
    _, r = three_result
    assert r.stats["frac_band_once"] <= 0.5 and r.stats["warning"] is False
    assert "warning_text" not in r.stats


def test_v4g_statistics_boundary():
    def mode(h):
        fp = S.Fingerprint(0.0, np.zeros(3), np.zeros(1), np.zeros(1))
        return S.Mode(np.zeros(1), 0.0, fp, h)

    half = [mode(1), mode(3)]
    assert S.search_statistics(half, half, 4)["warning"] is False  # exactly 0.5 does not warn
    more = [mode(1), mode(0), mode(5)]
    st = S.search_statistics(more, more, 6)
    assert st["warning"] is True and st["n_band_once"] == 2 and math.isclose(st["frac_band_once"], 2 / 3)


# ---------------------------------------------------------------------------------------------------------------------
# V7c: fingerprint extensions (defaults of decision #16)


class TeffWells(Wells):
    """Two wells with one geometry and Teff 1500 K apart (x[4] in units of 250 K)."""

    def __init__(self):
        super().__init__([[0, 0, 0, 0, -3.0], [0, 0, 0, 0, 3.0]], [0.0, 1.0], [1.0, 1.0],
                         names=["p0", "p1", "p2", "p3", "teff"], lo=-8.0, hi=8.0)

    def fingerprint_quantities(self, x):
        q = super().fingerprint_quantities(x)
        q["extra"] = {"teff": 13000.0 + 250.0 * float(x[4])}
        return q


def test_v7c_extra_teff_separates_modes_in_the_search(cfg):
    p = TeffWells()
    c0 = copy.deepcopy(cfg)
    c0["search"]["fingerprint"]["extra"] = {}
    base = S.run_search(p, c0, seed=3, n_starts=20)
    split = S.run_search(TeffWells(), cfg, seed=3, n_starts=20)
    # extra = {}: Teff ignored, both wells fall into one mode; with the default teff = 1000 K (decision #16) they are
    # two
    assert len(base.all_modes) == 1
    assert len(split.all_modes) == 2
    assert min(m.hits for m in split.all_modes) >= 2  # both wells reached by starts, so the default merged them
    assert sum(m.hits for m in split.all_modes) == base.all_modes[0].hits == 20
    teffs = sorted(m.fingerprint.extra["teff"] for m in split.all_modes)
    assert teffs == pytest.approx([12250.0, 13750.0], abs=10.0)
    assert split.all_modes[0].fingerprint.to_dict()["extra"]["teff"] == pytest.approx(12250.0, abs=10.0)  # best


def test_v7c_fingerprint_defaults_of_decision_16(cfg):
    fp = cfg["search"]["fingerprint"]
    assert fp["relative"] is True and fp["extra"] == {"teff": 1000.0} and fp["angle"] == "auto"
    assert (fp["b50"], fp["b90"], fp["b50_rms"]) == (0.5, 1.0, 1.0)
    assert {k: fp[k] for k in S.FINGERPRINT_DEFAULTS} == S.FINGERPRINT_DEFAULTS
