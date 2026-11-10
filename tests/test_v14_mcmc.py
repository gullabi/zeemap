"""V14a-f: the MCMC inside a mode and its run-time convergence rules, on toy problems with known answers.

Toys:
- `Gauss`: cost = 0.5 (x - m)^T C^-1 (x - m) with whitened residuals L^T (x - m) (C^-1 = L L^T), field_slice None, a
  wide uniform prior: the posterior is N(m, C).
- `Wells` (V14f): cost = -logsumexp_k(-d_k - |x - m_k|^2 / (2 s_k^2)); a shallow well (the mode handed to the MCMC)
  next to a deeper one within reach of walkers started at 3 Gauss-Newton widths.
- `FieldToy` (prior checks): the 14 "dqo" names inside field_slice, cost 0.
Synthetic chains test the rules of `convergence.py` directly (AR(1) with known tau, iid draws, a trend, a stuck walker).
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from zeemap import convergence as C
from zeemap import search as S
from zeemap import uncertainty as U
from zeemap.config import defaults

pytestmark = pytest.mark.tier1

COV = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.5]])
MEAN = np.array([1.0, -2.0, 3.0])


class Gauss:
    def __init__(self, m=MEAN, cov=COV, residuals=True):
        self.m = np.asarray(m, dtype=float)
        self.C = np.asarray(cov, dtype=float)
        self.Ci = np.linalg.inv(self.C)
        self.Lt = np.linalg.cholesky(self.Ci).T
        self.dim = self.m.size
        self.names = [f"p{i}" for i in range(self.dim)]
        self.free = np.ones(self.dim, dtype=bool)
        self.lo = np.full(self.dim, -50.0)
        self.hi = np.full(self.dim, 50.0)
        self.step = np.sqrt(np.diag(self.C))
        self.value = np.zeros(self.dim)
        self.field_slice = None
        self._res = residuals

    def cost(self, X):
        D = np.atleast_2d(np.asarray(X, dtype=float)) - self.m
        return 0.5 * np.einsum("ki,ij,kj->k", D, self.Ci, D)

    def residuals(self, x):
        if not self._res:
            raise NotImplementedError
        return self.Lt @ (np.asarray(x, dtype=float) - self.m)

    def canonical(self, x):
        return np.array(x, dtype=float)


class Wells:
    def __init__(self, centres, depths, widths):
        self.mk = np.asarray(centres, dtype=float)
        self.d = np.asarray(depths, dtype=float)
        self.s = np.asarray(widths, dtype=float)
        self.dim = self.mk.shape[1]
        self.names = [f"p{i}" for i in range(self.dim)]
        self.free = np.ones(self.dim, dtype=bool)
        self.lo = np.full(self.dim, -10.0)
        self.hi = np.full(self.dim, 10.0)
        self.step = np.ones(self.dim)
        self.value = np.zeros(self.dim)
        self.field_slice = None

    def _terms(self, X):
        q = 0.5 * np.sum((X[:, None, :] - self.mk[None]) ** 2, axis=2) / self.s[None] ** 2
        return -self.d[None] - q

    def cost(self, X):
        return -logsumexp(self._terms(np.atleast_2d(np.asarray(X, dtype=float))), axis=1)

    def residuals(self, x):
        k = int(np.argmax(self._terms(np.asarray(x, dtype=float)[None])[0]))
        return (x - self.mk[k]) / self.s[k]

    def canonical(self, x):
        return np.array(x, dtype=float)

    def fingerprint_quantities(self, x):
        return {"incl": 40.0 + 3.0 * x[0], "c": np.array([0.02 * x[1], 0.0, 0.0]), "b50": np.array([100.0 + x[0]]),
                "b90": np.array([300.0 + x[1]])}


NAMES14 = ["md_x", "md_y", "md_z", "bq", "aq_x", "aq_y", "aq_z", "mo_x", "mo_y", "mo_z", "c_x", "c_y", "c_z", "u"]


class FieldToy:
    def __init__(self, outside=None):
        self.dim = 14
        self.names = list(NAMES14)
        self.free = np.ones(14, dtype=bool)
        self.lo = np.array([-1200.0] * 3 + [-1200.0] + [-1.0] * 3 + [-1200.0] * 3 + [-0.4] * 3 + [-1.0])
        self.hi = -self.lo
        self.step = np.ones(14)
        self.value = np.zeros(14)
        self.value[4:7] = [0.0, 0.0, 1.0]
        self.field_slice = slice(0, 14)
        if outside is not None:
            self.outside_fraction = outside

    def cost(self, X):
        return np.zeros(np.atleast_2d(X).shape[0])

    def residuals(self, x):
        raise NotImplementedError

    def canonical(self, x):
        return np.array(x, dtype=float)


def _cfg(max_evals=0):
    cfg = defaults()
    cfg["uncertainty"]["max_evals"] = int(max_evals)
    return cfg


class GaussDerived(Gauss):
    def derived(self, x):
        return {"s": float(x[0] + x[1]), "v": np.array([x[0], x[2]])}


class HalfFlat(Gauss):
    """p0: a unit Gaussian at 0 cut by its lower limit lo = 0 (touches a range limit); p1: not in the cost at all
    (not constrained by the data)."""

    def __init__(self):
        super().__init__(m=[0.0, 0.0], cov=np.eye(2))
        self.lo = np.array([0.0, -1.0])
        self.hi = np.array([50.0, 1.0])
        self.step = np.array([1.0, 0.5])

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return 0.5 * X[:, 0] ** 2

    def residuals(self, x):
        return np.array([x[0], 1e-3 * x[1]])


@pytest.fixture(scope="module")
def gauss_run():
    p = GaussDerived()
    r = U.run_mcmc(p, S.Mode(p.m.copy(), 0.0, None, 1), _cfg(), seed=1)
    return p, r


# ---------------------------------------------------------------------------------------------------------------------
# V14a: ensembles, seeds, start spread, burn-in; the prior; the Gaussian toy end to end


def test_v14a_gaussian_toy_converges_mean_cov_and_coverage(gauss_run):
    p, r = gauss_run
    assert r.converged, r.report.failed
    assert r.notes and "the default 600000" in r.notes[0]
    assert r.nevals <= r.max_evals == U.MAX_EVALS_DEFAULT
    k = r.kept()
    n_eff = min(r.report.ess_bulk.values())
    sig = np.sqrt(np.diag(COV))
    # mean within 3 standard errors
    assert np.all(np.abs(k.mean(axis=0) - MEAN) < 3.0 * sig / math.sqrt(n_eff))
    # covariance within 3 standard errors: var(S_ij) = (C_ii C_jj + C_ij^2) / n for a Gaussian
    se = np.sqrt((np.outer(np.diag(COV), np.diag(COV)) + COV**2) / n_eff)
    assert np.all(np.abs(np.cov(k.T) - COV) < 3.0 * se)
    # 16/84 % points at m -+ sigma within 3 standard errors of a quantile (1.515 sigma / sqrt(n) at p = 0.16)
    for i, n in enumerate(p.names):
        q = r.intervals[n]
        assert q["p16"] == pytest.approx(MEAN[i] - sig[i], abs=3 * 1.515 * sig[i] / math.sqrt(n_eff))
        assert q["p84"] == pytest.approx(MEAN[i] + sig[i], abs=3 * 1.515 * sig[i] / math.sqrt(n_eff))
        assert "tempered_half68" not in q  # dropped, decision #24
        assert q["flags"] == []
    # coverage: the posterior of this toy is translation invariant (N(m', C) for data giving centre m'), so the
    # interval for a realisation m' = m + delta is the run's interval shifted by delta; 68 % of realisations cover m
    rng = np.random.default_rng(5)
    delta = rng.multivariate_normal(np.zeros(3), COV, 4000)
    for i, n in enumerate(p.names):
        q = r.intervals[n]
        cover = np.mean((q["p16"] + delta[:, i] <= MEAN[i]) & (MEAN[i] <= q["p84"] + delta[:, i]))
        assert cover == pytest.approx(0.68, abs=0.035)
        q95 = np.mean((q["p2_5"] + delta[:, i] <= MEAN[i]) & (MEAN[i] <= q["p97_5"] + delta[:, i]))
        assert q95 == pytest.approx(0.95, abs=0.02)


def test_v14a_ensembles_seeds_spread_and_burn_in(gauss_run):
    p, r = gauss_run
    E, steps, W, nf = r.samples.shape
    assert E == 4 and W == 16 and nf == 3
    assert len(set(r.seeds)) == 4
    assert r.start_spread == 3.0
    # the first step is still spread over about 3 Gauss-Newton widths
    first = r.samples[:, 0].reshape(-1, nf)
    assert np.all(first.std(axis=0) / np.sqrt(np.diag(COV)) > 1.8)
    assert r.burn >= 5 * r.report.tau * 0.5 and r.burn > 0
    assert r.report.kept == steps - r.burn
    # the run is reproducible from its seed and different for another seed
    r2 = U.run_mcmc(p, S.Mode(p.m.copy(), 0.0, None, 1), _cfg(4000), seed=1, handback=False)
    r3 = U.run_mcmc(p, S.Mode(p.m.copy(), 0.0, None, 1), _cfg(4000), seed=1, handback=False)
    r4 = U.run_mcmc(p, S.Mode(p.m.copy(), 0.0, None, 1), _cfg(4000), seed=2, handback=False)
    assert np.array_equal(r2.samples, r3.samples)
    assert not np.array_equal(r2.samples, r4.samples)


def test_v14a_gauss_newton_covariance():
    p = Gauss()
    assert np.allclose(U.gauss_newton_cov(p, MEAN + 0.3), COV, rtol=1e-8, atol=1e-10)
    q = Gauss(residuals=False)  # fallback: Hessian of the cost
    assert np.allclose(U.gauss_newton_cov(q, MEAN), COV, rtol=1e-6, atol=1e-8)


def test_v14a_derived_quantities(gauss_run):
    p, r = gauss_run
    for name in ("s", "v[0]", "v[1]"):
        q = r.intervals[name]
        assert q["rhat"] < 1.01 and q["ess_bulk"] >= 400 and q["ess_tail"] >= 400
        assert q["kl_prior"] is None  # no prior range for derived quantities
    sd = math.sqrt(COV[0, 0] + COV[1, 1] + 2 * COV[0, 1])
    n_eff = r.report.ess_bulk["s"]
    assert r.intervals["s"]["median"] == pytest.approx(MEAN[0] + MEAN[1], abs=3 * 1.26 * sd / math.sqrt(n_eff))
    assert r.intervals["v[1]"]["median"] == pytest.approx(r.intervals["p2"]["median"], abs=0.05)


def test_v14a_flags_range_limit_and_not_constrained():
    p = HalfFlat()
    r = U.run_mcmc(p, S.Mode(np.array([0.5, 0.0]), 0.125, None, 1), _cfg(20000), seed=4, handback=False)
    f0, f1 = r.intervals["p0"]["flags"], r.intervals["p1"]["flags"]
    assert U.TOUCHES_LIMIT in f0 and U.NOT_CONSTRAINED not in f0
    assert U.NOT_CONSTRAINED in f1
    assert r.intervals["p0"]["kl_prior"] > 1.0 and r.intervals["p1"]["kl_prior"] < 0.1


def test_v14a_intervals_json_shape(gauss_run):
    p, r = gauss_run
    d = r.to_dict()
    for key in ("neg_log_l", "converged", "failed_rules", "flags", "convergence", "quantities", "label"):
        assert key in d
    for key in ("tau", "burn", "kept", "rhat", "ess_bulk", "ess_tail", "drift", "stuck", "acceptance", "nevals"):
        assert key in d["convergence"]
    q = d["quantities"]["p0"]
    for key in ("median", "p16", "p84", "p2_5", "p97_5", "half68", "half95",
                "rhat", "ess_bulk", "ess_tail", "kl_prior", "jackknife_sigma", "flags"):
        assert key in q
    import json

    json.dumps(d)


def test_v14a_prior_draws_match_density():
    """Tier-1 stand-in for V13: draw_prior samples the density of log_prior."""
    p = FieldToy()
    cfg = _cfg()
    st = cfg["search"]["starts"]
    X = U.draw_prior(p, cfg, np.random.default_rng(3), 20000)
    lp = U.log_prior(p, cfg, X)
    assert np.all(np.isfinite(lp))

    def flat(v, lo, hi):
        h = np.histogram(v, 20, (lo, hi))[0]
        return stats.chisquare(h).pvalue

    md = np.linalg.norm(X[:, 0:3], axis=1)
    assert flat(md, *st["md"]) > 0.01
    assert flat(np.linalg.norm(X[:, 7:10], axis=1), 0.0, st["mo"]) > 0.01
    assert flat((np.linalg.norm(X[:, 10:13], axis=1) / st["c"]) ** 3, 0.0, 1.0) > 0.01
    assert flat(np.abs(X[:, 13]), *st["u"]) > 0.01
    assert 0.45 < np.mean(X[:, 13] > 0) < 0.55  # the mirror image is in the prior
    assert flat(X[:, 3], -st["bq"], st["bq"]) > 0.01
    assert flat(X[:, 2] / md, -1.0, 1.0) > 0.01  # uniform direction
    assert stats.kstest(np.log(np.linalg.norm(X[:, 4:7], axis=1)), stats.norm(0.0, 0.1).cdf).pvalue > 0.01
    # the density is -2 ln|md| on the shell
    x = X[0].copy()
    y = x.copy()
    y[0:3] *= 1.1 if md[0] * 1.1 <= st["md"][1] else 0.95
    ratio = np.linalg.norm(y[0:3]) / md[0]
    assert U.log_prior(p, cfg, y) - U.log_prior(p, cfg, x) == pytest.approx(-2.0 * math.log(ratio), abs=1e-12)
    # outside points
    for k, v in ((0, 700.0), (13, 0.01), (13, -1.01), (3, 300.0), (10, 0.3)):
        z = x.copy()
        if k == 0:
            z[0:3] = [v, 0.0, 0.0]
        else:
            z[k] = v
        assert U.log_prior(p, cfg, z) == -np.inf
    # the prior cut: an outside fraction above outside_max sets the density to 0, and draws respect it
    q = FieldToy(outside=lambda x: 0.5 if x[2] > 0 else 0.0)
    Xq = U.draw_prior(q, cfg, np.random.default_rng(4), 500)
    assert np.all(Xq[:, 2] <= 0)
    z = x.copy()
    z[2] = abs(z[2]) + 1.0
    if st["md"][0] <= np.linalg.norm(z[0:3]) <= st["md"][1]:
        assert U.log_prior(q, cfg, z) == -np.inf


# ---------------------------------------------------------------------------------------------------------------------
# V14b: chain length and autocorrelation time


def _ar1(phi, steps, walkers, rng, ndim=1):
    x = np.zeros((steps, walkers, ndim))
    e = rng.standard_normal((steps, walkers, ndim)) * math.sqrt(1 - phi**2)
    x[0] = rng.standard_normal((walkers, ndim))
    for t in range(1, steps):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.95])
def test_v14b_ar1_known_tau(phi):
    rng = np.random.default_rng(int(phi * 100))
    x = _ar1(phi, 20000, 16, rng)
    tau = C.integrated_time(x)[0]
    assert tau == pytest.approx((1 + phi) / (1 - phi), rel=0.2)


def test_v14b_short_chain_flagged_not_converged():
    p = Gauss()
    r = U.run_mcmc(p, S.Mode(p.m.copy(), 0.0, None, 1), _cfg(4000), seed=3)
    assert not r.converged
    assert "V14b_length" in r.report.failed
    assert r.flags[0] == U.NOT_CONVERGED
    for q in r.intervals.values():
        assert U.NOT_CONVERGED in q["flags"]
        assert q["median"] is None and q["p16"] is None and q["half68"] is None  # UNC-2
        assert "V14b_length" in q["failed_rules"]
    assert r.to_dict()["failed_rules"] == r.report.failed


def test_v14b_tau_change_rule():
    rng = np.random.default_rng(8)
    good = _ar1(0.5, 3000, 16, rng, 2)[None].repeat(1, axis=0)
    nll = 0.5 * np.sum(good**2, axis=-1)
    rep = C.check_v14(good, nll, tau_factor=50)
    assert "V14b_tau_stable" not in rep.failed and "V14b_length" not in rep.failed
    # the last quarter becomes strongly correlated: tau changes
    bad = good.copy()
    slow = _ar1(0.97, 750, 16, rng, 2)
    bad[0, -750:] = slow
    rep2 = C.check_v14(bad, 0.5 * np.sum(bad**2, axis=-1), tau_factor=50, burn=0)
    assert "V14b_tau_stable" in rep2.failed


# ---------------------------------------------------------------------------------------------------------------------
# V14c: R-hat


def test_v14c_rhat_identical_and_shifted():
    rng = np.random.default_rng(11)
    same = rng.standard_normal((4, 4000))
    assert C.rhat(same) < 1.01
    shifted = rng.standard_normal((4, 4000)) + np.array([0.0, 0.0, 1.0, -1.0])[:, None]
    assert C.rhat(shifted) > 1.1
    wider = rng.standard_normal((4, 4000)) * np.array([1.0, 1.0, 1.0, 3.0])[:, None]  # caught by the folded R-hat
    assert C.rhat(wider) > 1.01
    # a run whose ensembles sit at different places fails V14c
    chains = rng.standard_normal((4, 2000, 16, 1)) + np.array([0.0, 0.0, 0.0, 1.0])[:, None, None, None]
    rep = C.check_v14(chains, 0.5 * chains[..., 0] ** 2, tau_factor=50)
    assert "V14c_rhat" in rep.failed
    assert rep.failed_quantities["V14c_rhat"] == ["p0"]


def test_v14b_too_short_chain_reports_every_quantity():
    """V14b: a chain with fewer than 8 kept steps fails every rule and still lists every quantity in both per-quantity
    tables (their own failed rules, and tau = inf). Regression: run_mcmc's thinned second check read tau_quantity for
    every key of quantity_failed and stopped with a KeyError on such a chain (found by V6c)."""
    rng = np.random.default_rng(5)
    chains = rng.standard_normal((2, 6, 8, 2))
    q = {"a": chains[..., 0] + chains[..., 1], "b": chains[..., 0] * 2.0}
    rep = C.check_v14(chains, 0.5 * np.sum(chains**2, axis=-1), tau_factor=50, quantities=q, burn=0)
    assert rep.passed is False
    assert set(rep.quantity_failed) == set(rep.tau_quantity) == {"a", "b"}
    assert all(math.isinf(v) for v in rep.tau_quantity.values())


def test_v14c_gaussian_run_rhat(gauss_run):
    _, r = gauss_run
    assert all(v < 1.01 for v in r.report.rhat.values())


# ---------------------------------------------------------------------------------------------------------------------
# V14d: ESS


def test_v14d_ess_of_iid_draws_is_n():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((8, 2500))
    n = x.size
    assert C.ess(x) == pytest.approx(n, rel=0.1)
    assert C.ess_bulk(x) == pytest.approx(n, rel=0.1)
    assert C.ess_tail(x) == pytest.approx(n, rel=0.2)


def test_v14d_ess_of_ar1_is_n_over_tau():
    rng = np.random.default_rng(13)
    phi = 0.8
    x = _ar1(phi, 10000, 16, rng)[:, :, 0].T  # (walkers, steps)
    tau = (1 + phi) / (1 - phi)
    assert C.ess_bulk(x) == pytest.approx(x.size / tau, rel=0.2)


def test_v14d_low_ess_fails():
    rng = np.random.default_rng(14)
    chains = np.stack([_ar1(0.95, 200, 4, rng) for _ in range(4)])  # n / tau = 3200 / 39 ~ 80
    rep = C.check_v14(chains, 0.5 * chains[..., 0] ** 2, tau_factor=1, burn=0)
    assert "V14d_ess_bulk" in rep.failed and "V14d_ess_tail" in rep.failed


# ---------------------------------------------------------------------------------------------------------------------
# V14e: drift, stuck walkers, acceptance


def _iid_run(rng, e=4, steps=2000, walkers=16, ndim=2):
    ch = rng.standard_normal((e, steps, walkers, ndim))
    return ch, 0.5 * np.sum(ch**2, axis=-1)


def test_v14e_drifting_chain_flagged():
    rng = np.random.default_rng(21)
    ch, nll = _iid_run(rng)
    rep = C.check_v14(ch, nll, tau_factor=50)
    assert rep.passed, rep.failed
    trend = nll + 4.0 * np.linspace(0.0, 1.0, nll.shape[1])[None, :, None]
    rep2 = C.check_v14(ch, trend, tau_factor=50)
    assert not rep2.passed and "V14e_drift" in rep2.failed
    assert max(rep2.drift) > 1.0


def test_v14e_stuck_walker_flagged():
    rng = np.random.default_rng(22)
    ch, nll = _iid_run(rng)
    nll[2, :, 5] += 10.0
    rep = C.check_v14(ch, nll, tau_factor=50)
    assert "V14e_stuck" in rep.failed
    assert rep.stuck[2] == [5] and all(not s for k, s in enumerate(rep.stuck) if k != 2)
    idx, thr = C.stuck_walkers(nll[2])
    assert idx == [5] and thr == pytest.approx(3.0)


def test_v14e_acceptance_warning():
    rng = np.random.default_rng(23)
    ch, nll = _iid_run(rng)
    rep = C.check_v14(ch, nll, tau_factor=50, acceptance=[0.3, 0.1, 0.45, 0.6])
    assert rep.passed
    assert len(rep.warnings) == 2 and "ensemble 1" in rep.warnings[0] and "ensemble 3" in rep.warnings[1]


# ---------------------------------------------------------------------------------------------------------------------
# V14f: hand-back of a planted better point


def test_v14f_planted_better_point_handed_back_once():
    # mode A (handed to the MCMC) is a shallow well at the origin, cost 0, width 1; the planted better point is a
    # narrow well (cost -20, width 0.3) 3 widths of A away, whose basin (where it dominates the sum) reaches to about
    # 1.5 from the origin, so walkers started at 3 Gauss-Newton widths find it. The table also holds that deep well as
    # mode B (already found by the search): its own MCMC finds nothing better.
    p = Wells([[0.0, 0.0], [3.0, 0.0]], [0.0, -20.0], [1.0, 0.3])
    p.step = np.full(2, 0.3)  # polish's central differences (half-step 0.5 x step) must resolve the narrow well
    cfg = _cfg(30000)
    modes = []
    for x in ([3.0, 0.0], [0.0, 0.0]):
        x = np.array(x)
        modes.append(S.Mode(x, float(p.cost(x[None])[0]), S.fingerprint(p, x), 1))
    mode_a = modes[1]
    assert modes[0].f == pytest.approx(-20.0, abs=1e-6) and mode_a.f == pytest.approx(0.0, abs=1e-6)
    res = U.uncertainty_for_modes(p, modes, cfg, seed=7, value_tol=2.0)
    k_a = next(k for k, m in enumerate(res.modes) if m is mode_a)
    k_b = 1 - k_a
    assert len(res.handbacks[k_a]) == 1  # handed back exactly once
    assert res.handbacks[k_b] == []
    hb = res.handbacks[k_a][0]
    assert hb["old_neg_log_l"] == pytest.approx(0.0, abs=1e-6)
    assert hb["new_neg_log_l"] == pytest.approx(-20.0, abs=1e-6)
    assert hb["fingerprint_distance"] > 1.0
    # the mode table is updated: mode A now holds the better point, its fingerprint and value
    assert mode_a.f == pytest.approx(-20.0, abs=1e-6)
    assert np.allclose(mode_a.x, [3.0, 0.0], atol=1e-4)
    assert S.fingerprint_distance(mode_a.fingerprint, S.fingerprint(p, np.array([3.0, 0.0]))) < 1e-3
    assert res.results[k_a].better_points == []  # the re-run from the new point finds nothing better
    assert [m.f for m in res.modes] == sorted(m.f for m in res.modes)
    d = res.to_dict()
    assert d["calibration"] == "uncalibrated" and len(d["modes"]) == 2
    assert len(d["modes"][k_a]["handbacks"]) == 1


def test_v14f_no_handback_within_tolerance(gauss_run):
    _, r = gauss_run
    assert len(r.polished) == 3
    assert r.better_points == []


# ---------------------------------------------------------------------------------------------------------------------
# Sampler details: lock-step ensembles, reflecting DE move, observer-frame chart, budget end, KL bias


COS40 = math.cos(math.radians(40.0))


class _Observers:
    phases = np.array([0.0])

    def observers(self, Th):
        u = Th[:, 3]
        bad = ~(np.abs(u) <= 1.0)
        uc = np.clip(u, -1.0, 1.0)
        n = np.stack([np.sqrt(1.0 - uc * uc), np.zeros_like(uc), -uc], axis=-1)
        return n[:, None, :], bad


class StarToy:
    """The geometry of a one-phase dipole fit (the toy star): -ln L depends only on |md| (12 +-
    0.07 MG) and on the cosine of the angle between md and the observer n(u) = (sqrt(1 - u^2), 0, -u) (cos 40 deg +-
    0.03), plus a unit Gaussian in t. The rotation of md about n and the inclination (md co-rotating) are free, so in
    md and u the posterior is a thin curved shell around a ring."""

    def __init__(self):
        self.names = ["md_x", "md_y", "md_z", "u", "t"]
        self.dim = 5
        self.free = np.ones(5, dtype=bool)
        self.lo = np.array([-40.0, -40.0, -40.0, -1.0, -5.0])
        self.hi = -self.lo
        self.step = np.array([1.0, 1.0, 1.0, 0.1, 1.0])
        n = np.array([0.8, 0.0, -0.6])  # the mode: u = 0.6, md 40 deg from n(0.6) in the x-z plane
        e1 = np.cross([0.0, 1.0, 0.0], n)
        self.value = np.zeros(5)
        self.value[0:3] = 12.0 * (COS40 * n + math.sin(math.radians(40.0)) * e1)
        self.value[3] = 0.6
        self.field_slice = slice(0, 4)
        self.forward = _Observers()

    def thetas(self, X):
        return np.atleast_2d(np.asarray(X, dtype=float))[:, 0:4]

    def _parts(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = self.forward.observers(X)[0][:, 0, :]
        r = np.linalg.norm(X[:, 0:3], axis=1)
        cb = np.einsum("ki,ki->k", X[:, 0:3], n) / r
        return np.column_stack([(r - 12.0) / 0.07, (cb - COS40) / 0.03, X[:, 4]])

    def cost(self, X):
        return 0.5 * np.sum(self._parts(X) ** 2, axis=1)

    def residuals(self, x):
        return self._parts(x)[0]

    def canonical(self, x):
        return np.array(x, dtype=float)


def _star_cfg(max_evals):
    cfg = _cfg(max_evals)
    cfg["search"]["starts"]["md"] = [2.0, 40.0]
    return cfg


def test_v14a_lockstep_ensembles_match_emcee_bit_for_bit():
    """The lock-step stepping (one cost call per sub-step for all ensembles) draws exactly what independent emcee
    samplers draw from the same seeds when the cost of a row does not depend on its batch."""
    import emcee

    p = Gauss()
    lpf = U._LogProb(p, _cfg())
    rng = np.random.default_rng(0)
    P = [p.m + rng.standard_normal((16, 3)) for _ in range(3)]
    moves = [(emcee.moves.DEMove(), 0.8), (U.SnookerMove(), 0.2)]
    ens = [U._Ensemble(P[e], lpf(P[e]), np.random.RandomState(100 + e), moves) for e in range(3)]
    calls = lpf.ncalls
    U._run_lockstep(ens, lpf, 150)
    assert lpf.ncalls - calls <= 4 * 150  # one call per sub-step for the 3 ensembles together
    for e in range(3):
        s = emcee.EnsembleSampler(16, 3, lpf, moves=moves, vectorize=True)
        s.run_mcmc(emcee.State(P[e].copy(), random_state=np.random.RandomState(100 + e).get_state()), 150,
                   progress=False)
        assert np.array_equal(s.get_chain(), np.array(ens[e].chain))
        assert np.array_equal(np.asarray(s.get_blobs(), dtype=float).reshape(150, 16), np.array(ens[e].blobs))
        assert ens[e].acceptance == pytest.approx(float(np.mean(s.acceptance_fraction)), abs=1e-15)


def test_v14a_reflecting_de_move_samples_bounded_target():
    """ReflectDEMove keeps detailed balance: a uniform on [0, 1] and a half-normal on [0, 10] (its mode at the face)
    are recovered, and no proposal leaves the box."""
    import emcee

    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 10.0])

    def lp(Z):
        Z = np.atleast_2d(Z)
        inside = np.all((Z >= lo) & (Z <= hi), axis=1)
        return np.where(inside, -0.5 * Z[:, 1] ** 2, -np.inf)

    s = emcee.EnsembleSampler(16, 2, lp, moves=[(U.ReflectDEMove(lo, hi), 1.0)], vectorize=True)
    rng = np.random.default_rng(3)
    P = np.column_stack([rng.uniform(0, 1, 16), np.abs(rng.standard_normal(16))])
    s.run_mcmc(emcee.State(P, random_state=np.random.RandomState(3).get_state()), 6000, progress=False)
    ch = s.get_chain()[500:]
    tau = float(np.max(C.integrated_time(ch)))
    x = ch[:: max(1, int(3 * tau))].reshape(-1, 2)
    assert np.all(np.isfinite(s.get_log_prob()))  # every proposal landed inside
    assert stats.kstest(x[:, 0], stats.uniform(0, 1).cdf).pvalue > 0.01
    assert stats.kstest(x[:, 1], stats.halfnorm.cdf).pvalue > 0.01
    assert 0.2 < float(np.mean(s.acceptance_fraction)) < 0.9


def test_v14a_observer_chart_round_trip():
    p = StarToy()
    ch = U.make_chart(p, _star_cfg(0))
    assert ch.kind == "observer" and ch.names == ["md_r", "md_cosb", "md_p", "u", "t", "md_q"]
    assert U.make_chart(p, _star_cfg(0), "identity").kind == "identity"
    with pytest.raises(ValueError):
        U.make_chart(Gauss(), _cfg(), "observer")
    rng = np.random.default_rng(1)
    Y = np.column_stack([rng.uniform(-20, 20, (50, 3)), rng.uniform(-1, 1, 50), rng.standard_normal(50)])
    Z = ch.from_free(Y, rng)
    Y2, lj, ok = ch.to_free(Z)
    assert ok.all() and np.allclose(Y2, Y, rtol=0, atol=1e-12)
    assert np.allclose(Z[:, 0], np.linalg.norm(Y[:, 0:3], axis=1))
    n = p.forward.observers(Y)[0][:, 0, :]
    assert np.allclose(Z[:, 1], np.einsum("ki,ki->k", Y[:, 0:3], n) / Z[:, 0])
    # the likelihood does not depend on the azimuth (md_p, md_q) nor on u: the posterior is flat along both
    z = ch.from_free(p.value[None])[0]
    Zs = np.tile(z, (9, 1))
    Zs[:, 3] = np.linspace(-0.9, 0.9, 9)
    Zs[:, 2], Zs[:, 5] = np.cos(np.arange(9.0)), np.sin(np.arange(9.0))
    X = ch.to_free(Zs)[0]
    assert np.allclose(p.cost(X), p.cost(p.value[None])[0], atol=1e-9)


def test_v14a_observer_chart_samples_the_prior_with_the_data_off():
    """V13 stand-in for the chart: with -ln L x 0 the samples follow log_prior (|md| uniform on its shell, md direction
    uniform, |u| uniform on [0.03, 1] and its mirror), so the Jacobian md_r^2 and the auxiliary factor are right.

    Pass rule as V13's (VERIFICATION.md): p > 0.01 / (number of KS tests), here 3, on draws thinned to 2 integrated
    autocorrelation times. With p > 0.01 each on draws thinned to one tau, the test failed on
    a BLAS build whose different rounding gave another realisation of the same chain (p = 0.005, OPENBLAS_CORETYPE =
    Haswell): draws one tau apart are still correlated, which makes KS p-values too small, and three tests at 0.01 each
    fail a correct sampler about 3 % of the time."""
    p = StarToy()
    cfg = _star_cfg(60000)
    r = U.run_mcmc(p, S.Mode(p.value.copy(), 0.0, None, 1), cfg, seed=5, data_off=True, handback=False)
    assert r.chart == "observer"
    k = r.samples[:, r.samples.shape[1] // 4:]
    k = k[:, :: max(1, int(2 * r.report.tau))].reshape(-1, 5)
    md = np.linalg.norm(k[:, 0:3], axis=1)
    st = cfg["search"]["starts"]
    p_min = 0.01 / 3
    assert stats.kstest(md, stats.uniform(st["md"][0], st["md"][1] - st["md"][0]).cdf).pvalue > p_min
    assert stats.kstest(k[:, 2] / md, stats.uniform(-1, 2).cdf).pvalue > p_min
    assert stats.kstest(np.abs(k[:, 3]), stats.uniform(st["u"][0], st["u"][1] - st["u"][0]).cdf).pvalue > p_min
    assert 0.4 < np.mean(k[:, 3] > 0) < 0.6


def test_v14a_star_toy_acceptance_and_convergence():
    """The diagnosis on a toy with the toy star's geometry: in md and u the acceptance falls far
    below 0.2 (DE chords leave the curved shell); in the observer-frame chart it lies in 0.2-0.5, the run converges,
    and |md|, the dipole-observer angle and t are recovered while u is flagged "not constrained by the data"."""
    p = StarToy()
    mode = S.Mode(p.value.copy(), 0.0, None, 1)
    bad = U.run_mcmc(p, mode, _star_cfg(12000), seed=2, chart="identity", handback=False)
    assert max(bad.acceptance) < 0.1
    r = U.run_mcmc(p, mode, _star_cfg(200000), seed=2, handback=False)
    assert r.chart == "observer"
    assert all(0.2 <= a <= 0.5 for a in r.acceptance), r.acceptance
    assert r.converged, r.report.failed
    k = r.kept()
    md = np.linalg.norm(k[:, 0:3], axis=1)
    n = p.forward.observers(k)[0][:, 0, :]
    cb = np.einsum("ki,ki->k", k[:, 0:3], n) / md
    assert np.median(md) == pytest.approx(12.0, abs=0.01) and np.std(md) == pytest.approx(0.07, rel=0.1)
    assert np.median(cb) == pytest.approx(COS40, abs=0.005) and np.std(cb) == pytest.approx(0.03, rel=0.1)
    assert np.std(k[:, 4]) == pytest.approx(1.0, rel=0.1)
    assert U.NOT_CONSTRAINED in r.intervals["u"]["flags"] and U.NOT_CONSTRAINED not in r.intervals["t"]["flags"]


def test_v14a_budget_ends_with_one_partial_chunk():
    """Rows rejected by the prior are not evaluated, so the budget is not used up by whole steps; the run must still
    end after one partial chunk instead of a series of 1-step chunks each followed by a V14 check."""
    p = StarToy()
    lines: list[str] = []
    r = U.run_mcmc(p, S.Mode(p.value.copy(), 0.0, None, 1), _star_cfg(5000), seed=3, chart="identity",
                   handback=False, log=lines.append)
    checks = [m for m in lines if " steps, " in m]
    assert len(checks) == 1, checks
    assert r.nevals <= 5000


def test_v14a_kl_bias_correction():
    """REP-4: the plug-in KL of n draws from the prior itself is about (bins - 1) / (2 n) nats (0.063 at n = 150, 20
    bins), so the raw estimate would call the prior "constrained" in a fair share of short chains; the corrected value
    used by the flag is near 0."""
    rng = np.random.default_rng(31)
    prior = rng.uniform(0.0, 1.0, 10000)
    raw, cor = [], []
    for _ in range(200):
        kl, filled = U._kl_hist(rng.uniform(0.0, 1.0, 150), prior, 0.0, 1.0, with_bins=True)
        raw.append(kl)
        cor.append(kl - (filled - 1) / 300.0)
    assert np.mean(raw) == pytest.approx(19 / 300.0, rel=0.25)
    assert abs(np.mean(cor)) < 0.01
    assert np.mean(np.array(cor) >= U.KL_MIN) == 0.0
    # a constrained posterior keeps a corrected KL far above the threshold
    kl, filled = U._kl_hist(rng.normal(0.5, 0.05, 150), prior, 0.0, 1.0, with_bins=True)
    assert kl - (filled - 1) / 300.0 > 1.0


# ---------------------------------------------------------------------------------------------------------------------
# UNC-2 per quantity (decision #17) and declared nuisance (decision #18)


class NuisanceGauss(Gauss):
    """Gauss with p1 declared nuisance (as Problem declares the orientation of a single-phase fit)."""

    nuisance_coordinates = ("p1",)


def _split_run(seed=11, E=4, T=3000, W=16):
    """Synthetic chains on a 2-d toy: p0 independent N(0, 1) draws (converged); p1 a flat direction that each ensemble
    has explored only near its own value (e, 2e, ... : R-hat far above 1.01, unconverged)."""
    rng = np.random.default_rng(seed)
    ch = np.empty((E, T, W, 2))
    ch[..., 0] = rng.standard_normal((E, T, W))
    ch[..., 1] = 2.0 * np.arange(E)[:, None, None] + 0.1 * rng.standard_normal((E, T, W))
    nll = 0.5 * ch[..., 0] ** 2
    return ch, nll


def _result(problem, ch, nll, report):
    return U.McmcResult(list(problem.names), np.zeros(2), 0.0, ch, nll, report, bool(report.passed), 0, 0,
                        ch.shape[2], [0] * ch.shape[0], 3.0, np.eye(2), [0.3] * ch.shape[0])


def test_v14c_per_quantity_flat_unconverged_direction_gets_no_interval():
    """VERIFICATION.md V14 (decision #17): one direction flat and unconverged, the other converged: the run fails V14
    (mode-level verdict unchanged), the converged quantity gets its interval, the unconverged one is flagged "not
    converged" with its own failed rule and no numbers."""
    ch, nll = _split_run()
    rep = C.check_v14(ch, nll, tau_factor=50.0, names=["p0", "p1"])
    assert not rep.passed and "V14c_rhat" in rep.failed and rep.failed_quantities["V14c_rhat"] == ["p1"]
    assert rep.quantity_failed["p0"] == [] and "V14c_rhat" in rep.quantity_failed["p1"]
    p = Gauss(m=[0.0, 0.0], cov=np.eye(2))
    res = _result(p, ch, nll, rep)
    iv, flags = U.compute_intervals(p, _cfg(), res, n_prior=2000)
    assert flags[0] == U.NOT_CONVERGED  # the mode-level flag stays
    q0, q1 = iv["p0"], iv["p1"]
    assert q0["median"] == pytest.approx(0.0, abs=0.05) and q0["p84"] - q0["p16"] == pytest.approx(2.0, abs=0.05)
    assert U.NOT_CONVERGED not in q0["flags"] and "failed_rules" not in q0
    assert q1["median"] is None and q1["p16"] is None and U.NOT_CONVERGED in q1["flags"]
    assert q1["failed_rules"] == rep.quantity_failed["p1"] and q1["rhat"] > 1.5
    # a run-level -ln L rule (drift) fails every quantity: no interval at all
    nll_drift = nll + np.linspace(0.0, 5.0, ch.shape[1])[None, :, None]
    rep_d = C.check_v14(ch, nll_drift, tau_factor=50.0, names=["p0", "p1"])
    assert "V14e_drift" in rep_d.failed and "V14e_drift" in rep_d.quantity_failed["p0"]
    iv_d, _ = U.compute_intervals(p, _cfg(), _result(p, ch, nll_drift, rep_d), n_prior=2000)
    assert iv_d["p0"]["median"] is None and iv_d["p1"]["median"] is None


def test_v14c_declared_nuisance_is_left_out_of_the_verdict():
    """Decision #18: the same run with p1 declared nuisance passes the mode-level verdict; p1 keeps its R-hat and ESS,
    gets no interval and the flags "orientation not determined by one phase" and "nuisance, not required"."""
    ch, nll = _split_run()
    rep = C.check_v14(ch, nll, tau_factor=50.0, names=["p0", "p1"], nuisance={"p1"})
    assert rep.passed, rep.failed
    assert rep.nuisance == ["p1"] and "V14c_rhat" in rep.quantity_failed["p1"]
    assert rep.to_dict()["nuisance"] == ["p1"]
    p = NuisanceGauss(m=[0.0, 0.0], cov=np.eye(2))
    assert U.nuisance_names(p, U.make_chart(p, _cfg()), list(p.names)) == (["p1"], ["p1"])
    iv, flags = U.compute_intervals(p, _cfg(), _result(p, ch, nll, rep), n_prior=2000)
    assert U.NOT_CONVERGED not in flags
    assert iv["p0"]["median"] == pytest.approx(0.0, abs=0.05)
    assert iv["p1"]["median"] is None and iv["p1"]["rhat"] > 1.5
    assert U.ORIENTATION in iv["p1"]["flags"] and U.NUISANCE in iv["p1"]["flags"]
    assert U.NOT_CONVERGED not in iv["p1"]["flags"]


class _Rows(Gauss):
    """Gauss recording the rows of every cost call."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def cost(self, X):
        self.rows.append(np.atleast_2d(X).shape[0])
        return super().cost(X)


def test_v14a_pool_spreads_cost_calls_same_chains():
    """Every cost call of the sampler is split into chunks of at most COST_CHUNK rows, evaluated in a pool
    when one is given (one mode then uses every worker), else one after another; the chains and the hand-back
    polishes are the same bits either way."""
    p = _Rows()
    mode = S.Mode(p.m.copy(), 0.0, None, 1)
    r1 = U.run_mcmc(p, mode, _cfg(3000), seed=5)
    assert max(p.rows) == U.COST_CHUNK  # a sub-step proposes 4 ensembles x 16 / 2 = 32 rows: split
    with S.make_pool(2) as pool:
        r2 = U.run_mcmc(p, mode, _cfg(3000), seed=5, pool=pool)
    assert np.array_equal(r1.samples, r2.samples) and np.array_equal(r1.nll, r2.nll)
    assert [q.f for q in r1.polished] == [q.f for q in r2.polished]


def test_v14a_mcmc_layout_choice():
    """One mode keeps at most pool_width workers busy (4 ensembles x 16 walkers / 2 = 32 rows =
    4 chunks on the toy); one process per mode once n >= min(workers, width), else the modes in turn over the pool."""
    from zeemap.cli import _per_mode

    assert U.pool_width(_Rows(), _cfg(3000)) == 4
    assert not _per_mode(1, 12, 4)  # one mode: in turn, its calls over 4 workers
    assert not _per_mode(3, 12, 4)  # 3 processes would idle the 4th worker one mode could use
    assert _per_mode(4, 12, 4) and _per_mode(12, 12, 8) and _per_mode(2, 2, 8)
