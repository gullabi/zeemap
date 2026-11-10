"""V5b: the fast likelihood against the plain reference (likelihood_reference.py), on random small problems, both
kinds and every switch, 1e-10 relative."""

from __future__ import annotations

import numpy as np
import pytest

from zeemap import likelihood_reference as ref
from zeemap.likelihood import Objective
from zeemap.types import Grid
from test_v1_likelihood import make_dataset, perturbed_models

pytestmark = pytest.mark.tier1


def random_cfg(rng):
    kind = "chi2" if rng.random() < 0.3 else "full"
    obj = {"kind": kind, "scale": bool(rng.random() < 0.7)}
    obj["corr"] = False if rng.random() < 0.25 else {"rho": float(rng.uniform(2, 8)), "jitter": float(rng.uniform(0.03, 0.3))}
    obj["disc"] = False if rng.random() < 0.3 else {"a_I": float(rng.uniform(0.1, 1)), "rho_I": float(rng.uniform(20, 120)),
                                                     "a_VI": float(rng.uniform(0.1, 1)), "rho_VI": float(rng.uniform(20, 250))}
    obj["continuum"] = False if rng.random() < 0.2 else {"degree": int(rng.integers(0, 4))}
    return {"objective": obj}


@pytest.mark.parametrize("seed", range(24))
def test_v5b_fast_equals_reference(seed):
    rng = np.random.default_rng(500 + seed)
    cfg = random_cfg(rng)
    weights = cfg["objective"]["kind"] == "chi2" and rng.random() < 0.5
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0, 0.4), i_offset=bool(rng.random() < 0.5),
                                       n_fit=int(rng.integers(12, 30)), weights=weights)
    for sp in ds.spectra:  # a per-spectrum offset choice
        if sp.kind == "I":
            sp.i_offset = bool(rng.random() < 0.5)
    K = 4
    I, VI = perturbed_models(rng, I_true, VI_true, K, size=0.1)
    obj = Objective(ds, cfg)
    fast = obj.neg_log_l(I, VI, np.zeros(K, bool))
    for k in range(K):
        want, const = ref.neg_log_l(ds, cfg, I[k], VI[k])
        assert fast[k] == pytest.approx(want, rel=1e-10), cfg
        assert obj.constant == pytest.approx(const, rel=1e-10, abs=1e-12)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v5b_fast_equals_reference_realistic_size():
    """One I and one V/I spectrum of 300 pixels with the default settings."""
    rng = np.random.default_rng(77)
    grid = Grid(start=3800.0, step=2.5, npix=320)
    from zeemap.types import DataSet, Spectrum
    lam = grid.wavelengths
    pix = np.arange(10, 310)
    I_true = (1 - 0.3 * np.exp(-0.5 * ((lam - 4200) / 15) ** 2))[None]
    VI_true = (0.05 * np.sin(lam / 40.0))[None]
    sI = 0.01 * np.ones(pix.size)
    sV = 0.004 * np.ones(pix.size)
    yI = I_true[0, pix] * (1 + 0.02 * (lam[pix] - 4200) / 300) + 3 * sI * rng.standard_normal(pix.size)
    yV = VI_true[0, pix] + 3 * sV * rng.standard_normal(pix.size)
    spectra = [Spectrum(0, 0.0, "I", pix, yI, sI, np.ones(pix.size), np.zeros(pix.size, int), True),
               Spectrum(0, 0.0, "VI", pix, yV, sV, np.ones(pix.size), np.zeros(pix.size, int), True)]
    ds = DataSet(grid, np.array([0.0]), spectra, np.full((1, 320), np.nan), np.full((1, 320), np.nan),
                 np.full((1, 320), np.nan), np.full((1, 320), np.nan))
    I = np.stack([I_true, I_true * 1.01])
    VI = np.stack([VI_true, VI_true * 0.9])
    for cfg in ({}, {"objective": {"kind": "chi2"}}):
        obj = Objective(ds, cfg)
        fast = obj.neg_log_l(I, VI)
        for k in range(2):
            want, const = ref.neg_log_l(ds, cfg, I[k], VI[k])
            assert fast[k] == pytest.approx(want, rel=1e-10)
            assert obj.constant == pytest.approx(const, rel=1e-10, abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# V5b, the pre-whitened-record path (likelihood.py bind_forward / neg_log_l_nodes; forward.py node_weights_batch;
# decision #15): the same -ln L as the exact path (forward.spectra, then neg_log_l), 1e-10.


@pytest.fixture(scope="module")
def fast_env(synth_lib, tmp_path_factory):
    """The small synthetic problem of test_v7_cli.py, with two Teff slices and Teff free (rows interpolate), a
    narrower window (301 pixels per spectrum, for speed); one Problem per kind, shared by the tests below."""
    import os

    from zeemap import cli
    from zeemap.config import load_config
    from zeemap.problem import Problem
    from test_v7_cli import config_text

    root = tmp_path_factory.mktemp("zeemap_data_v5b_fast")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)

    def text(data, kind):
        t = config_text(data, kind).replace("teffs = [30000.0]", "teffs = [20000.0, 30000.0]")
        t = t.replace("windows = [[4300.0, 6700.0]]", "windows = [[4500.0, 5100.0]]")
        return t.replace("[search]\n", 'teff = { value = 25000.0, step = 1000.0, lo = 20000.0, hi = 30000.0, '
                                       'free = true }\n\n[search]\n')

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "truth.toml").write_text(text("inj", "chi2"))
        assert cli.main(["inject", str(root / "truth.toml"), "-o", str(root / "inj"), "--noise", "white",
                         "--seed", "3"]) == 0
        cfgs = {}
        for kind in ("full", "chi2"):
            p = root / f"{kind}.toml"
            p.write_text(text("inj", kind))
            cfgs[kind] = load_config(p)
        problems = {kind: Problem(cfg) for kind, cfg in cfgs.items()}
        yield root, cfgs, problems


def _fast_batch(pr, rng, K=7):
    X = np.repeat(pr.value[None, :], K, axis=0)
    X[:, pr.free] += rng.normal(0.0, 0.5, (K, int(pr.free.sum()))) * pr.step[pr.free]
    X[:, pr.iteff] = rng.uniform(20500.0, 29500.0, K)
    X[1, pr.iteff] = 30000.0  # exactly on a slice
    X[2, pr.iteff] = 35000.0  # outside the selection: invalid
    X[3, pr.names.index("u")] = 1.5  # |cos i| > 1: invalid
    return X


def _force(monkeypatch, mode):
    """mode "pre" / "raw": every pre-whitened spectrum takes that branch of neg_log_l_nodes; "auto": the estimate."""
    if mode != "auto":
        monkeypatch.setattr(Objective, "_prefer_pre", staticmethod(lambda *a: mode == "pre"))


@pytest.mark.parametrize("mode", ["auto", "pre", "raw"])
@pytest.mark.parametrize("kind", ["full", "chi2"])
def test_v5b_fast_path_equals_exact_path(fast_env, kind, mode, monkeypatch):
    from zeemap.likelihood import WORST

    _force(monkeypatch, mode)
    pr = fast_env[2][kind]
    assert pr.objective.bind_forward(pr.forward) == "on", pr.objective.fast_path
    rng = np.random.default_rng(31)
    X = _fast_batch(pr, rng)
    I, VI, bad = pr.spectra(X)
    exact = pr.objective.neg_log_l(I, VI, bad)
    Wn, sw, bad2, _, _ = pr.forward.node_weights_batch(pr.thetas(X), teff=pr._teff(X))
    np.testing.assert_array_equal(bad, bad2)
    used = np.flatnonzero(np.any(Wn != 0.0, axis=(0, 1)))
    assert 0 < used[0] and used[-1] < Wn.shape[2] - 1  # the node range is restricted (exercises the views)
    assert len(sw) == 2 and any(np.any((c > 0) & (c < 1)) for _, c in sw)  # rows interpolate between slices
    fast = pr.objective.neg_log_l_nodes(Wn, sw, bad2)
    assert np.all(exact[[2, 3]] == WORST) and np.all(fast[[2, 3]] == WORST)
    ok = exact < WORST
    assert ok.sum() == len(X) - 2
    np.testing.assert_allclose(fast[ok], exact[ok], rtol=1e-10, atol=0)
    # differences between rows (what a search sees) agree as well
    np.testing.assert_allclose(fast[ok] - fast[ok][0], exact[ok] - exact[ok][0], rtol=0,
                               atol=1e-10 * np.abs(exact[ok]).max())
    # Problem.cost takes the fast path; the old path (fast_path off) gives the same cost, single rows and the batch
    n0 = pr.objective.n_evals
    cost_fast = pr.cost(X)
    single = np.array([pr.cost(x[None, :])[0] for x in X])
    assert pr.objective.n_evals == n0 + 2 * len(X)
    pr.objective.fast_path = "off: test"  # bind_forward returns the cached state for this forward model
    try:
        cost_old = pr.cost(X)
    finally:
        pr.objective.fast_path = "on"
    np.testing.assert_array_equal(cost_fast >= WORST, cost_old >= WORST)
    np.testing.assert_allclose(cost_fast[ok], cost_old[ok], rtol=1e-10, atol=0)
    np.testing.assert_allclose(single[ok], cost_old[ok], rtol=1e-10, atol=0)


@pytest.mark.parametrize("seed", range(8))
def test_v5b_fast_path_every_switch(fast_env, seed, monkeypatch):
    """Random objective settings (corr, disc, scale, continuum degree or none, i_offset per spectrum) on the same
    forward model: neg_log_l_nodes equals neg_log_l(forward.spectra(...)) to 1e-10 relative."""
    pr = fast_env[2]["full"]
    rng = np.random.default_rng(900 + seed)
    saved = [sp.i_offset for sp in pr.dataset.spectra]
    try:
        for sp in pr.dataset.spectra:
            if sp.kind == "I":
                sp.i_offset = bool(rng.random() < 0.5)
        obj = Objective(pr.dataset, random_cfg(rng))
    finally:
        for sp, v in zip(pr.dataset.spectra, saved):
            sp.i_offset = v
    assert obj.bind_forward(pr.forward) == "on"
    assert obj.bind_forward(pr.forward) == "on"  # idempotent
    assert obj.fast_path_bytes(pr.forward) == (sum(a.nbytes for d in obj._fast["pre"] for a in d.values())
                                               + sum(L.nbytes for L, _ in obj._fast["chol"].values()))
    X = _fast_batch(pr, rng, K=5)
    Th, teff = pr.thetas(X), pr._teff(X)
    I, VI, bad = pr.forward.spectra(Th, teff=teff)
    exact = obj.neg_log_l(I, VI, bad)
    nodes = pr.forward.node_weights_batch(Th, teff=teff)[:3]
    ok = exact < 1e30
    for mode in ("pre", "raw"):
        _force(monkeypatch, mode)
        fast = obj.neg_log_l_nodes(*nodes)
        np.testing.assert_array_equal(exact == 1e30, fast == 1e30)
        np.testing.assert_allclose(fast[ok], exact[ok], rtol=1e-10, atol=0)
    assert obj.n_evals == 3 * len(X) and obj.n_invalid == 3 * int((~ok).sum())


def test_v5b_fast_node_weights_batch_reproduces_spectra(fast_env):
    """spectra() = node_weights_batch + one product per Teff slice (the refactor changes no bits)."""
    pr = fast_env[2]["chi2"]
    fw = pr.forward
    X = _fast_batch(pr, np.random.default_rng(5), K=4)
    Th, teff = pr.thetas(X), pr._teff(X)
    I, VI, bad = fw.spectra(Th, teff=teff)
    Wn, sw, bad2, frac, _ = fw.node_weights_batch(Th, teff=teff)
    np.testing.assert_array_equal(bad, bad2)
    K, P = Wn.shape[:2]
    S = np.zeros((K * P, 2 * fw.M))
    per_pixel = any(np.ndim(c) == 2 for _, c in sw)  # "planck" Teff weights (decision #26)
    C = np.zeros((len(fw.records), K, fw.M)) if per_pixel else np.zeros((len(fw.records), K))
    for t, c in sw:
        C[t] = c if np.ndim(c) == C.ndim - 1 else np.asarray(c)[:, None]
    S = fw._disc_sum(Wn.reshape(K * P, -1), C, K, P)
    I2, VI2 = fw._finish(S, bad2)
    np.testing.assert_array_equal(np.isnan(I), np.isnan(I2))
    np.testing.assert_array_equal(I[~np.isnan(I)], I2[~np.isnan(I2)])
    np.testing.assert_array_equal(VI[~np.isnan(VI)], VI2[~np.isnan(VI2)])
    for k in (0, 4 % K):
        if not bad[k]:
            np.testing.assert_array_equal(frac[k], fw.outside_fraction(Th[k]))


def test_v5b_fast_path_declines_with_reason(fast_env, monkeypatch):
    import zeemap.likelihood as lk

    pr = fast_env[2]["full"]
    obj = Objective(pr.dataset, {"objective": {"kind": "full"}})
    assert obj.fast_path.startswith("off: bind_forward not called")
    with pytest.raises(RuntimeError):
        obj.neg_log_l_nodes(np.zeros((1, 2, 1)), [], None)
    full_bytes, chol_bytes = obj.fast_path_bytes(pr.forward), obj.fast_path_bytes(pr.forward, pre=False)
    assert 0 < chol_bytes < full_bytes
    # over budget for the pre-whitened matrices only: on without them, same values as the exact path
    monkeypatch.setattr(lk, "FAST_PATH_MAX_BYTES", full_bytes - 1)
    assert obj.bind_forward(pr.forward) == "on"
    assert obj.fast_path_detail.startswith("on, without pre-whitened matrices") and not obj._fast["where"]
    X = _fast_batch(pr, np.random.default_rng(8), K=5)
    Th, teff = pr.thetas(X), pr._teff(X)
    exact = obj.neg_log_l(*pr.forward.spectra(Th, teff=teff))
    fast = obj.neg_log_l_nodes(*pr.forward.node_weights_batch(Th, teff=teff)[:3])
    np.testing.assert_allclose(fast, exact, rtol=1e-10, atol=0)
    # even the Cholesky factors over budget: off, with the reason
    obj2 = Objective(pr.dataset, {"objective": {"kind": "full"}})
    monkeypatch.setattr(lk, "FAST_PATH_MAX_BYTES", chol_bytes - 1)
    assert obj2.bind_forward(pr.forward).startswith("off: the Cholesky factors alone would take")
    assert obj2.fast_path_detail == obj2.fast_path
    with pytest.raises(RuntimeError):
        obj2.neg_log_l_nodes(np.zeros((1, 2, 1)), [], None)


# ---------------------------------------------------------------------------------------------------------------------
# V5a, the whole-surface outside fraction of the prior cut comes from the forward model's node stage (forward.py
# `_node_stage`; problem.py `cost`): no second field evaluation, and `cost` is unchanged.


def _old_smooth_outside(pr, Th):
    """The prior-cut fraction as problem.py computed it before (a second field evaluation at the patch centres)."""
    from scipy.special import expit

    with np.errstate(all="ignore"):
        B = pr.field.evaluate_batch(pr.surface.centres, Th)
    b = np.sqrt(np.einsum("kni,kni->kn", B, B))
    s = 2.0 / 4.0
    f = expit((b - pr.library.b_max) / s) + expit((pr.library.b_min - b) / s)
    f = np.where(np.isfinite(b), f, 1.0)
    return f @ (pr.surface.areas / pr.surface.areas.sum())


def _old_cost(pr, X):
    from zeemap.likelihood import WORST
    from zeemap.problem import PRIOR_CUT_WEIGHT

    I, VI, bad = pr.spectra(X)
    f = pr.objective.neg_log_l(I, VI, bad)
    ok = f < WORST
    Th = pr.thetas(X[ok])
    f[ok] = f[ok] + PRIOR_CUT_WEIGHT * np.maximum(0.0, _old_smooth_outside(pr, Th) - pr.outside_max)
    f[~np.isfinite(f)] = WORST
    return np.minimum(f, WORST)


@pytest.mark.parametrize("kind", ["full", "chi2"])
def test_v5a_cost_with_surface_fraction_from_node_stage_unchanged(fast_env, kind):
    """cost() with the outside fraction from the node stage equals the former cost (separate field evaluation) on 50
    draws of the start distribution, a third of them pushed above the library's B range, to 1e-12 relative, on the
    pre-whitened path and on the exact path."""
    from zeemap import search
    from zeemap.likelihood import WORST

    cfg, pr = fast_env[1][kind], fast_env[2][kind]
    rng = np.random.default_rng(2026)
    X = np.array([search._draw_one(pr, cfg, rng) for _ in range(50)])
    X[::3, 0:3] *= 3.0  # strong fields: part of the surface above b_max, the prior cut is active
    X[1, pr.names.index("u")] = 1.5  # an invalid row
    want = _old_cost(pr, X)
    ok = want < WORST
    assert ok.sum() >= 40 and not ok[1]
    pen = want[ok] - pr.objective.neg_log_l(*pr.spectra(X[ok]))
    assert np.sum(pen > 0) >= 5  # the prior cut is exercised
    assert pr.objective.bind_forward(pr.forward) == "on"
    got = pr.cost(X)
    np.testing.assert_array_equal(got >= WORST, ~ok)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-12, atol=0)
    pr.objective.fast_path = "off: test"  # the exact path (spectra_with_outside)
    try:
        got_exact = pr.cost(X)
    finally:
        pr.objective.fast_path = "on"
    np.testing.assert_array_equal(got_exact >= WORST, ~ok)
    np.testing.assert_allclose(got_exact[ok], want[ok], rtol=1e-12, atol=0)


def test_v5a_surface_fraction_delegates_to_forward(fast_env):
    """node_weights_batch, spectra_with_outside, Problem.smooth_outside and forward.surface_outside give the same
    smoothed fraction; Problem.outside_fraction (hard edges) equals the former formula; spectra() is unchanged."""
    from zeemap import search

    cfg, pr = fast_env[1]["full"], fast_env[2]["full"]
    fw = pr.forward
    rng = np.random.default_rng(77)
    X = np.array([search._draw_one(pr, cfg, rng) for _ in range(12)])
    X[::2, 0:3] *= 3.0
    Th, teff = pr.thetas(X), pr._teff(X)
    fs = fw.node_weights_batch(Th, teff=teff)[4]
    I, VI, bad, fs2 = fw.spectra_with_outside(Th, teff=teff)
    I0, VI0, bad0 = fw.spectra(Th, teff=teff)
    np.testing.assert_array_equal(fs, fs2)
    np.testing.assert_array_equal(I, I0)
    np.testing.assert_array_equal(VI, VI0)
    np.testing.assert_array_equal(bad, bad0)
    np.testing.assert_allclose(pr.smooth_outside(Th), fs, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(fs, _old_smooth_outside(pr, Th), rtol=1e-14, atol=1e-16)
    assert np.any(fs > 0.05) and np.any(fs < 0.05)
    areas = pr.surface.areas / pr.surface.areas.sum()
    for k in range(len(X)):
        with np.errstate(all="ignore"):
            B = pr.field.evaluate(pr.surface.centres, Th[k])
        b = np.sqrt(np.einsum("ij,ij->i", B, B))
        out = ~np.isfinite(b) | (b < pr.library.b_min) | (b > pr.library.b_max)
        assert pr.outside_fraction(X[k]) == pytest.approx(float(out @ areas), rel=1e-14, abs=1e-16)


@pytest.mark.parametrize("kind", ["full", "chi2"])
def test_v3h_scale_fast_path_records_times_1e15(fast_env, kind, monkeypatch):
    """The pre-whitened-record path with the library records x 1e15 (I and V, so V/I is unchanged): both branches
    ("pre" and "raw") equal the exact path at that scale (1e-10 relative), and -ln L does not move (decision #26; without it, it would move by (deg + 1) ln(1e15) per I
    spectrum (kind full; 0 for chi2) from the unscaled records, the same for every row (1e-6 nats)."""
    import copy

    pr = fast_env[2][kind]
    fw = copy.copy(pr.forward)
    fw.records = pr.forward.records * 1e15
    fw._flat = [fw.records[i].reshape(fw.NB * fw.NC, 2 * fw.M) for i in range(fw.records.shape[0])]
    fw.factors = [None if f is None else (f[0], f[1] * 1e15) for f in pr.forward.factors]  # factored records
    ob0 = Objective(pr.dataset, pr.cfg)
    ob1 = Objective(pr.dataset, pr.cfg)
    assert ob0.bind_forward(pr.forward) == "on" and ob1.bind_forward(fw) == "on"
    X = _fast_batch(pr, np.random.default_rng(32))
    Th, teff = pr.thetas(X), pr._teff(X)
    I, VI, bad = fw.spectra(Th, teff=teff)
    exact = ob1.neg_log_l(I, VI, bad)
    ok = exact < 1e30
    n_i = sum(1 for sp in pr.dataset.spectra if sp.kind == "I")
    shift = 0.0  # the normalised continuum basis (decision #26): no shift; before, (deg + 1) ln(1e15) per I spectrum
    del n_i
    for mode in ("pre", "raw"):
        _force(monkeypatch, mode)
        f0 = ob0.neg_log_l_nodes(*pr.forward.node_weights_batch(Th, teff=teff)[:3])
        f1 = ob1.neg_log_l_nodes(*fw.node_weights_batch(Th, teff=teff)[:3])
        np.testing.assert_array_equal(f1 >= 1e30, ~ok)
        np.testing.assert_allclose(f1[ok], exact[ok], rtol=1e-10, atol=0)
        np.testing.assert_allclose(f1[ok] - f0[ok], shift, rtol=0, atol=1e-6)
