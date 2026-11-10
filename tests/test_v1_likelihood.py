"""V1: known answers, likelihood items: V1g (closed-form integrals against brute-force integration), V1h
(plain chi2 known answers), V1m (noise model: kernel formula and whitening). The helpers here build small synthetic
DataSets directly from arrays (no data.py) and are reused by the V3, V4 and V5 likelihood tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import eval_legendre, logsumexp

from zeemap.likelihood import Objective
from zeemap.types import DataSet, Grid, Spectrum

pytestmark = pytest.mark.tier1

GRID = Grid(start=4000.0, step=2.5, npix=80)


def line_model(lam, depth=0.3, centre=4100.0, width=12.0, slope=0.0):
    """A smooth model I: a continuum with one Gaussian line."""
    return (1.0 + slope * (lam - lam.mean()) / 100.0) * (1.0 - depth * np.exp(-0.5 * ((lam - centre) / width) ** 2))


def make_dataset(rng, phases=(0.0, 0.25), kinds=("I", "VI"), grid=GRID, i_offset=True, n_fit=None, noise=3.0,
                 weights=False):
    """A DataSet with one spectrum per (phase, kind), fitted pixels in two windows, data = a model with a continuum
    tilt plus noise `noise` times the quoted errors. Returns (dataset, truth I (P, M), truth VI (P, M))."""
    lam = grid.wavelengths
    P, M = len(phases), grid.npix
    I_true = np.empty((P, M))
    VI_true = np.empty((P, M))
    for p in range(P):
        I_true[p] = line_model(lam, depth=0.3 + 0.1 * p, centre=4080.0 + 15 * p)
        VI_true[p] = 0.05 * np.sin((lam - 4000.0) / 30.0 + p)
    I_obs = np.full((P, M), np.nan)
    VI_obs = np.full((P, M), np.nan)
    sI = np.full((P, M), np.nan)
    sVI = np.full((P, M), np.nan)
    spectra = []
    for p in range(P):
        for kind in kinds:
            if n_fit is None:
                pix = np.r_[np.arange(3, 30), np.arange(38, 75)]
                win = np.r_[np.zeros(27, int), np.ones(37, int)]
            else:
                pix = np.arange(n_fit) + 5
                win = np.zeros(n_fit, int)
            n = pix.size
            if kind == "I":
                sig = 0.01 * (1.0 + 0.3 * rng.random(n))
                tilt = 1.0 + 0.03 * (lam[pix] - lam[pix].mean()) / 100.0
                y = I_true[p, pix] * tilt + 0.02 + noise * sig * rng.standard_normal(n)
                I_obs[p, pix], sI[p, pix] = y, sig
            else:
                sig = 0.004 * (1.0 + 0.3 * rng.random(n))
                y = VI_true[p, pix] + noise * sig * rng.standard_normal(n)
                VI_obs[p, pix], sVI[p, pix] = y, sig
            w = (0.5 + rng.random(n)) if weights else np.ones(n)
            spectra.append(Spectrum(phase_index=p, phase=float(phases[p]), kind=kind, pixels=pix, data=y, sigma=sig,
                                    weight=w, window=win, i_offset=i_offset))
    ds = DataSet(grid=grid, phases=np.asarray(phases, float), spectra=spectra, I=I_obs, VI=VI_obs, sig_I=sI, sig_VI=sVI)
    return ds, I_true, VI_true


def perturbed_models(rng, I_true, VI_true, K, size=0.03):
    """A batch (K, P, M) of models near the truth (different line depths and V/I amplitudes)."""
    lam = GRID.wavelengths
    I = np.empty((K,) + I_true.shape)
    VI = np.empty((K,) + VI_true.shape)
    for k in range(K):
        I[k] = I_true * (1.0 - size * rng.standard_normal() * np.exp(-0.5 * ((lam - 4090.0) / 20.0) ** 2))
        VI[k] = VI_true * (1.0 + size * rng.standard_normal()) + 0.002 * rng.standard_normal()
    return I, VI


def cfg_of(**objective):
    return {"objective": objective}


# ---------------------------------------------------------------- V1g


def brute_neg_log_l(spec, lam, model, n_lns=561, n_u=97, u_max=12.0):
    """-ln of the marginal likelihood of one I spectrum (K = identity, Legendre degree 1, no offset), integrated
    numerically over ln sigma (Jeffreys, unit density) and the two continuum coefficients (flat, unit density). The
    integrand is the raw Gaussian likelihood; the grids are centred on the least-squares point and scaled with the
    coefficients' standard errors times sigma (an affine change of variables, Jacobian included), spanning +-12 standard
    errors, i.e. a flat prior range wide enough to hold all of the mass."""
    y, sig = spec.data, spec.sigma
    n = y.size
    x = 2.0 * (lam - lam[0]) / (lam[-1] - lam[0]) - 1.0
    A = np.column_stack([model * eval_legendre(0, x), model * eval_legendre(1, x)]) / sig[:, None]
    r = (y - model) / sig
    beta0, *_ = np.linalg.lstsq(A, r, rcond=None)
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    chi2min = float(np.sum((r - A @ beta0) ** 2))
    lns_hat = 0.5 * math.log(chi2min / (n - 2))
    lns = lns_hat + np.linspace(-3.0, 4.0, n_lns)
    h_lns = lns[1] - lns[0]
    u = np.linspace(-u_max, u_max, n_u)
    h_u = u[1] - u[0]
    U0, U1 = np.meshgrid(u, u, indexing="ij")
    U = np.column_stack([U0.ravel(), U1.ravel()])  # (n_u^2, 2)
    out = np.empty(n_lns)
    for i, ls in enumerate(lns):
        s = math.exp(ls)
        beta = beta0[None, :] + s * se[None, :] * U
        e = r[None, :] - beta @ A.T
        logf = -0.5 * n * math.log(2 * math.pi * s * s) - np.sum(np.log(sig)) - np.sum(e * e, axis=1) / (2 * s * s)
        logf += math.log(s * se[0] * s * se[1])  # Jacobian of beta = beta0 + s * se * u
        out[i] = logsumexp(logf) + 2 * math.log(h_u)
    # the flat prior is on the coefficients of the normalised basis A / nu, nu = |W m| / |W 1| (decision #26):
    # integral over those = nu^k x the integral over beta, k = 2 here
    nu = float(np.linalg.norm(model / sig) / np.linalg.norm(1.0 / sig))
    return -(logsumexp(out) + math.log(h_lns)) - 2.0 * math.log(nu)


def test_v1g_closed_form_equals_brute_force_integration():
    rng = np.random.default_rng(11)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0,), kinds=("I",), i_offset=False, n_fit=12)
    cfg = cfg_of(kind="full", scale=True, corr=False, disc=False, continuum={"degree": 1})
    obj = Objective(ds, cfg)
    assert obj.n_params == [2]
    lam = GRID.wavelengths
    I = np.stack([I_true, I_true * (1.0 + 0.05 * np.exp(-0.5 * ((lam - 4030.0) / 6.0) ** 2))])
    VI = np.stack([VI_true, VI_true])
    nll = obj.neg_log_l(I, VI, np.zeros(2, bool))
    sp = ds.spectra[0]
    brute = np.array([brute_neg_log_l(sp, lam[sp.pixels], I[k, 0, sp.pixels]) for k in range(2)])
    d_closed, d_brute = nll[1] - nll[0], brute[1] - brute[0]
    assert abs(d_closed) > 0.1  # a real difference between the two parameter sets
    assert abs(d_closed - d_brute) <= 1e-6 * abs(d_brute)
    # with the exposed constant, the absolute value is the full marginal likelihood
    np.testing.assert_allclose(nll + obj.constant, brute, rtol=1e-6)


def test_v1g_closed_form_noise_scale_off_equals_brute_force():
    """Scale off: sigma fixed at 1 (the quoted errors); only the continuum is integrated (2-D grid)."""
    rng = np.random.default_rng(12)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0,), kinds=("I",), i_offset=False, n_fit=12, noise=1.0)
    obj = Objective(ds, cfg_of(scale=False, corr=False, disc=False, continuum={"degree": 1}))
    sp = ds.spectra[0]
    lam = GRID.wavelengths[sp.pixels]
    x = 2.0 * (lam - lam[0]) / (lam[-1] - lam[0]) - 1.0
    vals = []
    models = [I_true[0, sp.pixels], I_true[0, sp.pixels] * (1.0 + 0.04 * x**2)]
    for m in models:
        A = np.column_stack([m, m * x]) / sp.sigma[:, None]
        r = (sp.data - m) / sp.sigma
        b0, *_ = np.linalg.lstsq(A, r, rcond=None)
        se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
        u = np.linspace(-12, 12, 121)
        U0, U1 = np.meshgrid(u, u, indexing="ij")
        beta = b0 + se * np.column_stack([U0.ravel(), U1.ravel()])
        e = r[None] - beta @ A.T
        logf = -0.5 * sp.data.size * math.log(2 * math.pi) - np.sum(np.log(sp.sigma)) - 0.5 * np.sum(e * e, axis=1)
        nu = float(np.linalg.norm(m / sp.sigma) / np.linalg.norm(1.0 / sp.sigma))  # normalised basis, decision #26
        vals.append(-(logsumexp(logf) + math.log(se[0] * se[1]) + 2 * math.log(u[1] - u[0])) - 2.0 * math.log(nu))
    I = np.stack([I_true, I_true])
    I[1, 0, sp.pixels] = models[1]
    nll = obj.neg_log_l(I, np.stack([VI_true, VI_true]), None)
    np.testing.assert_allclose(nll + obj.constant, vals, rtol=1e-9)
    assert abs((nll[1] - nll[0]) - (vals[1] - vals[0])) <= 1e-6 * abs(vals[1] - vals[0])


# ---------------------------------------------------------------- V1h


def test_v1h_chi2_of_model_against_itself_is_zero():
    rng = np.random.default_rng(1)
    ds, I_true, VI_true = make_dataset(rng)
    obj = Objective(ds, cfg_of(kind="chi2"))
    I = np.zeros((1,) + I_true.shape)
    VI = np.zeros((1,) + VI_true.shape)
    for sp in ds.spectra:
        (I if sp.kind == "I" else VI)[0, sp.phase_index, sp.pixels] = sp.data
    assert obj.neg_log_l(I, VI, np.zeros(1, bool))[0] == 0.0
    assert all(v["chi2"] == 0.0 for v in obj.per_spectrum(I[0], VI[0]).values())


def test_v1h_chi2_of_known_residual():
    rng = np.random.default_rng(2)
    ds, I_true, VI_true = make_dataset(rng)
    obj = Objective(ds, cfg_of(kind="chi2", continuum=False))
    I = np.zeros((1,) + I_true.shape)
    VI = np.zeros((1,) + VI_true.shape)
    want_total = 0.0
    want = {}
    for sp in ds.spectra:
        r = 0.01 * rng.standard_normal(sp.pixels.size)
        (I if sp.kind == "I" else VI)[0, sp.phase_index, sp.pixels] = sp.data - r
        c = float(np.sum((r / sp.sigma) ** 2))
        want[sp.label] = (c, {w: float(np.sum((r[sp.window == w] / sp.sigma[sp.window == w]) ** 2))
                              for w in np.unique(sp.window)})
        want_total += c
    got = obj.neg_log_l(I, VI, None)[0]
    assert got == pytest.approx(0.5 * want_total, rel=1e-12)
    per = obj.per_spectrum(I[0], VI[0])
    for label, (c, wins) in want.items():
        assert per[label]["chi2"] == pytest.approx(c, rel=1e-12)
        for w, cw in wins.items():
            assert per[label]["windows"][int(w)]["chi2"] == pytest.approx(cw, rel=1e-12)
    res = obj.residuals(I[0], VI[0])
    assert sum(float(v @ v) for v in res) == pytest.approx(want_total, rel=1e-12)


def test_v1h_residuals_and_per_spectrum_with_continuum():
    """chi2 kind with the default continuum: residuals square-sum to 2 (-ln L); per_spectrum (unweighted) agrees;
    full kind with the scale on: each residual vector square-sums to n_s - k_s."""
    rng = np.random.default_rng(3)
    ds, I_true, VI_true = make_dataset(rng)
    I, VI = perturbed_models(rng, I_true, VI_true, 1)
    obj = Objective(ds, cfg_of(kind="chi2"))
    nll = obj.neg_log_l(I, VI)[0]
    res = obj.residuals(I[0], VI[0])
    assert sum(float(v @ v) for v in res) == pytest.approx(2 * nll, rel=1e-12)
    per = obj.per_spectrum(I[0], VI[0])
    assert sum(v["chi2"] for v in per.values()) == pytest.approx(2 * nll, rel=1e-12)
    full = Objective(ds, cfg_of())
    for v, n, k in zip(full.residuals(I[0], VI[0]), [s.pixels.size for s in ds.spectra], full.n_params):
        assert float(v @ v) == pytest.approx(n - k, rel=1e-12)


# ---------------------------------------------------------------- V1m


def matern(d, rho):
    return (1 + math.sqrt(3) * d / rho) * math.exp(-math.sqrt(3) * d / rho)


def test_v1m_kernel_entries_equal_formula():
    rng = np.random.default_rng(4)
    ds, *_ = make_dataset(rng, phases=(0.0,), n_fit=20)
    cfg = cfg_of(corr={"rho": 4.0, "jitter": 0.05}, disc={"a_I": 0.71, "rho_I": 85.0, "a_VI": 0.42, "rho_VI": 205.0})
    obj = Objective(ds, cfg)
    for idx, sp in enumerate(ds.spectra):
        lam = GRID.wavelengths[sp.pixels]
        a, rl = (0.71, 85.0) if sp.kind == "I" else (0.42, 205.0)
        K = obj.kernel(idx)
        for i in range(lam.size):
            for j in range(lam.size):
                d = abs(lam[i] - lam[j])
                want = matern(d, 4.0) + a * a * matern(d, rl) + (0.05**2 if i == j else 0.0)
                assert K[i, j] == pytest.approx(want, rel=1e-14, abs=1e-300)
    # switches: corr off -> identity (plus the discrepancy term); both off -> identity
    K = Objective(ds, cfg_of(corr=False, disc=False)).kernel(0)
    np.testing.assert_array_equal(K, np.eye(20))
    K = Objective(ds, cfg_of(corr=False)).kernel(0)
    lam = GRID.wavelengths[ds.spectra[0].pixels]
    assert K[0, 1] == pytest.approx(0.71**2 * matern(lam[1] - lam[0], 85.0), rel=1e-14)
    assert K[0, 0] == pytest.approx(1 + 0.71**2, rel=1e-14)


def test_v1m_whitened_noise_has_identity_covariance():
    rng = np.random.default_rng(5)
    ds, *_ = make_dataset(rng, phases=(0.0,), kinds=("I",), n_fit=20)
    obj = Objective(ds, cfg_of())
    sp = ds.spectra[0]
    lam = GRID.wavelengths[sp.pixels]
    K = np.array([[matern(abs(a - b), 4.0) + 0.71**2 * matern(abs(a - b), 85.0) + (0.05**2 if a == b else 0.0)
                   for b in lam] for a in lam])
    C = np.diag(sp.sigma) @ K @ np.diag(sp.sigma)
    N = 10_000
    draws = np.random.default_rng(20261018).multivariate_normal(np.zeros(20), C, size=N, method="eigh")  # (N, 20)
    z = obj.whiten(0, draws.T)  # (20, N)
    S = z @ z.T / N
    se = np.where(np.eye(20, dtype=bool), math.sqrt(2.0 / N), math.sqrt(1.0 / N))
    assert np.all(np.abs(S - np.eye(20)) < 5 * se), np.max(np.abs(S - np.eye(20)) / se)


# ---------------------------------------------------------------- V1h: the model's scale (a model in physical flux units)


def _one_i_spectrum(n=400, i_offset=False, with_vi=False):
    """The reproduction of the round-off found on the first real-data fit: one n-pixel I spectrum at ~9 (sigma/I 0.4 %),
    a Gaussian bump on a flat continuum; optionally a V/I spectrum on the same pixels."""
    rng = np.random.default_rng(0)
    grid = Grid(start=4000.0, step=2.5, npix=n)
    lam = grid.wavelengths
    shape = 1 + 0.3 * np.exp(-0.5 * ((lam - 4400) / 30) ** 2)
    data = 9.0 * shape * (1 + 0.004 * rng.standard_normal(n))
    sig = 0.004 * 9.0 * np.ones(n)
    pix = np.arange(n)
    spectra = [Spectrum(phase_index=0, phase=0.0, kind="I", pixels=pix, data=data, sigma=sig, weight=np.ones(n),
                        window=np.zeros(n, int), i_offset=i_offset)]
    if with_vi:
        spectra.append(Spectrum(phase_index=0, phase=0.0, kind="VI", pixels=pix, data=1e-3 * rng.standard_normal(n),
                                sigma=np.full(n, 1e-3), weight=np.ones(n), window=np.zeros(n, int), i_offset=False))
    ds = DataSet(grid=grid, phases=np.array([0.0]), spectra=spectra, I=data[None], VI=np.zeros((1, n)), sig_I=sig[None],
                 sig_VI=np.full((1, n), 1e-3))
    return ds, lam, shape, data, sig


@pytest.mark.parametrize("i_offset", [False, True])
def test_v1h_scale_chi2_equals_lstsq_and_does_not_depend_on_model_scale(i_offset):
    ds, lam, shape, data, sig = _one_i_spectrum(i_offset=i_offset)
    ob = Objective(ds, cfg_of(kind="chi2"))
    x = 2.0 * (lam - lam[0]) / (lam[-1] - lam[0]) - 1.0
    cols = [shape * eval_legendre(j, x) for j in range(4)] + ([np.ones_like(lam)] if i_offset else [])
    A = np.column_stack(cols) / sig[:, None]
    b = data / sig
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    want = float(np.sum((b - A @ beta) ** 2))
    VI = np.zeros((1, 1, lam.size))
    chi2 = [2.0 * ob.terms((scale * shape)[None, None, :], VI)[0, 0] for scale in (1.0, 1e15, 1e-12)]
    assert chi2[0] == pytest.approx(want, rel=1e-10)
    assert chi2[1] == pytest.approx(chi2[0], rel=1e-8)  # 4.46e6 instead of 412.87 before the fix
    assert chi2[2] == pytest.approx(chi2[0], rel=1e-8)
    # per_spectrum (plain chi2) and residuals take the same path
    for scale in (1.0, 1e15):
        I = (scale * shape)[None, :]
        assert ob.per_spectrum(I, VI[0])[ob.labels[0]]["chi2"] == pytest.approx(want, rel=1e-8)
        r = ob.residuals(I, VI[0])[0]
        assert float(r @ r) == pytest.approx(want, rel=1e-8)


def test_v1h_scale_check_scale_reports_the_ratio():
    ds, lam, shape, data, sig = _one_i_spectrum(with_vi=True)
    ob = Objective(ds, cfg_of())
    ratio = ob.check_scale((1e15 * shape)[None, :])
    assert list(ratio) == [ob.labels[0]]  # I spectra only
    assert ratio[ob.labels[0]] == pytest.approx(1e15 * np.median(shape) / np.median(data), rel=1e-12)
