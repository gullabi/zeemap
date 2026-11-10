"""V2a, V2b, V2d: known answers of the whole forward model on the synthetic library.

V2a: the same B and psi on every patch, divided by the sum of the weights, equals that library record after
convolution and resampling, computed here independently (1e-5, the float32 row).
V2b: a centred dipole seen pole-on: node weights equal an independent plain-numpy sum over the same patches (1e-10);
B10, B50, B90 of the brightness-weighted |B| distribution match the closed form (quadrature tolerance 1e-3).
V2d: B outside the library: "clamp" gives the edge record, "reject" marks the model invalid; the outside fraction of a
pole-on centred dipole whose equatorial field is below b_min matches the closed form (quadrature tolerance).

Closed form for V2b and V2d. Pole-on (u = 1) the observer is n = (0, 0, -1); the dipole md = (0, 0, Bp) is centred.
On the visible hemisphere put mu = n . r = -z. The dipole field at the unit sphere, B = (Bp / 2) (3 z r - e_z), has
|B| = (Bp / 2) sqrt(1 + 3 z^2) = (Bp / 2) sqrt(1 + 3 mu^2), increasing with mu. On the sphere dA = dmu dphi, so the
brightness-weighted distribution of mu has density proportional to mu LD(mu) on [0, 1]: its cumulative distribution is
F(mu) = mu^2 for LD = 1, and F(mu) = (0.35 mu^2 + 0.1 mu^3) / 0.45 for LD = 0.7 + 0.3 mu. A quantile q of |B| is
(Bp / 2) sqrt(1 + 3 mu_q^2) with F(mu_q) = q; the fraction below b_min is F(mu*) with (Bp / 2) sqrt(1 + 3 mu*^2) = b_min.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

import synth
from zeemap import field
from zeemap.forward import Forward
from zeemap.library import Library
from zeemap.surface import fibonacci
from zeemap.types import DataSet, Grid, Spectrum

pytestmark = pytest.mark.tier1


@field.register("toy_const")
class ToyConst(field.FieldModel):
    """The same field vector (bx, by, bz) on every patch; u = cos i."""

    names = ["bx", "by", "bz", "u"]
    dim = 4

    def evaluate(self, points, theta):
        t = self._theta(theta)
        return np.tile(t[:3], (np.asarray(points).reshape(-1, 3).shape[0], 1))

    def valid(self, theta):
        return bool(np.all(np.isfinite(theta)))


def dataset(phases=(0.0,), start=4100.0, step=2.5, npix=1100):
    grid = Grid(start, step, npix)
    P, M = len(phases), npix
    sp = [Spectrum(0, float(phases[0]), "VI", np.arange(M), np.zeros(M), np.ones(M), np.ones(M), np.zeros(M, int))]
    nan = np.full((P, M), np.nan)
    return DataSet(grid, np.asarray(phases, float), sp, nan, nan.copy(), nan.copy(), nan.copy())


def cfg(outside="clamp", **model):
    m = {"interp_b": "linear", "ld_mode": 3, "fwhm": 7.0, "phase_offset": 0.0}
    m.update(model)
    return {"model": m, "library": {"outside": outside}}


def pole_on(b, cospsi):
    return np.array([b * math.sqrt(max(0.0, 1.0 - cospsi * cospsi)), 0.0, -b * cospsi, 1.0])


def dipole(bp, u=1.0):
    th = np.zeros(14)
    th[2] = bp
    th[4:7] = [0.0, 0.0, 1.0]
    th[13] = u
    return th


@pytest.fixture(scope="module")
def lib(synth_lib):
    return Library(synth_lib, teffs=[30000.0])


def independent_record(rec, fwhm, grid):
    """Gaussian (4 sigma, sum 1) by np.convolve, then the bin average via 0.25 A sub-cells (pixel centres on whole or
    half A, 2.5 A pixels: exactly 10 sub-cells per pixel)."""
    s = fwhm / (2 * math.sqrt(2 * math.log(2)))
    h = int(math.floor(4 * s))
    lags = np.arange(-h, h + 1)
    k = np.exp(-(lags**2) / (2 * s * s))
    conv = np.convolve(rec, k / k.sum(), mode="same")
    fine = np.repeat(conv, 4)  # sub-cell q covers 3999.5 + 0.25 q .. + 0.25
    lam = grid.wavelengths
    first = np.rint((lam - grid.step / 2 - 3999.5) / 0.25).astype(int)
    return np.array([fine[a : a + 10].mean() for a in first])


@pytest.mark.parametrize("ld_mode", [0, 3])
@pytest.mark.parametrize("ib,ipsi,mirrored", [(12, 4, False), (12, 4, True), (0, 0, False), (30, 7, True), (5, 8, False)])
def test_v2a_uniform_star_equals_the_convolved_record(lib, ib, ipsi, mirrored, ld_mode):
    ds = dataset(phases=(0.0, 0.3, 0.7))
    fw = Forward(lib, fibonacci(1000), field.get_model("toy_const"), ds, cfg(ld_mode=ld_mode))
    b = lib.info.bs[ib]
    c = synth.cos_psi(lib.info.psis[ipsi]) * (-1 if mirrored else 1)
    theta = pole_on(b, c)
    I, VI, bad = fw.spectra(theta)
    wsum = fw.node_weights(theta).sum(axis=(1, 2))  # (P,)
    assert not bad[0]
    raw = lib.slice(30000.0)[ib, ipsi].astype(float)
    want_I = independent_record(raw[0], 7.0, ds.grid)
    want_V = independent_record(raw[1], 7.0, ds.grid) * (-1 if mirrored else 1)
    V = VI[0] * I[0]
    for p in range(3):
        np.testing.assert_allclose(I[0, p] / wsum[p], want_I, rtol=0, atol=1e-5 * np.abs(want_I).max())
        np.testing.assert_allclose(V[p] / wsum[p], want_V, rtol=0, atol=1e-5 * np.abs(want_I).max())
        np.testing.assert_allclose(VI[0, p], want_V / want_I, rtol=0, atol=1e-5)


def test_v2a_weight_sum_is_the_limb_darkened_disc(lib):
    """Sum of the weights = integral of mu LD(mu) dA over the visible hemisphere = 2 pi (0.7 / 2 + 0.3 / 3) for law 3."""
    fw = Forward(lib, fibonacci(20000), field.get_model("toy_const"), dataset(), cfg(ld_mode=3))
    w = fw.node_weights(pole_on(400.0, 0.3)).sum()
    assert w == pytest.approx(2 * math.pi * (0.7 / 2 + 0.3 / 3), rel=1e-4)


def hat_weights(bmag, cospsi, w, bnodes, cnodes):
    """Independent bilinear node weights: hat functions in B and cos psi via np.interp of unit vectors."""
    NB, NC = len(bnodes), len(cnodes)
    cinc = cnodes[::-1]
    hb = np.stack([np.interp(bmag, bnodes, np.eye(NB)[i]) for i in range(NB)])  # (NB, N)
    hc = np.stack([np.interp(cospsi, cinc, np.eye(NC)[NC - 1 - j]) for j in range(NC)])  # (NC, N)
    return np.einsum("in,jn,n->ij", hb, hc, w)


@pytest.mark.parametrize("ld_mode", [0, 3])
def test_v2b_pole_on_dipole_node_weights_against_a_plain_sum(lib, ld_mode):
    surf = fibonacci(3000)
    fw = Forward(lib, surf, field.get_model("dqo"), dataset(phases=(0.0, 0.4)), cfg(ld_mode=ld_mode))
    bp = 400.0
    W = fw.node_weights(dipole(bp))
    # independent: closed-form dipole field at the centres, pole-on observer
    r = surf.centres
    B = 0.5 * bp * (3 * r[:, 2:3] * r - np.array([0.0, 0.0, 1.0]))
    n = np.array([0.0, 0.0, -1.0])
    mu = r @ n
    vis = mu > 0
    ld = {0: np.ones_like(mu), 3: 0.7 + 0.3 * mu}[ld_mode]
    w = surf.areas * mu * ld
    bmag = np.linalg.norm(B, axis=1)
    want = hat_weights(bmag[vis], (B @ n)[vis] / bmag[vis], w[vis], lib.info.bs, lib.cos_psi_nodes())
    for p in range(2):
        np.testing.assert_allclose(W[p], want, rtol=0, atol=1e-10 * want.max())


def _cdf(mu, ld_mode):
    return mu**2 if ld_mode == 0 else (0.35 * mu**2 + 0.1 * mu**3) / 0.45


@pytest.mark.parametrize("ld_mode", [0, 3])
def test_v2b_b_quantiles_match_the_closed_form(lib, ld_mode):
    fw = Forward(lib, fibonacci(20000), field.get_model("dqo"), dataset(), cfg(ld_mode=ld_mode))
    bp = 400.0
    got = fw.b_quantiles(dipole(bp), q=(0.1, 0.5, 0.9))[0]
    want = []
    for q in (0.1, 0.5, 0.9):
        mq = brentq(lambda m: _cdf(m, ld_mode) - q, 0.0, 1.0, xtol=1e-14)
        want.append(0.5 * bp * math.sqrt(1 + 3 * mq * mq))
    if ld_mode == 0:
        assert want[1] == pytest.approx(200.0 * math.sqrt(2.5), rel=1e-12)
    np.testing.assert_allclose(got, want, rtol=1e-3)


def test_v2d_clamp_gives_the_edge_record_and_reject_marks_invalid(lib):
    surf = fibonacci(500)
    toy = field.get_model("toy_const")
    ds = dataset()
    fc = Forward(lib, surf, toy, ds, cfg())
    fr = Forward(lib, surf, toy, ds, cfg("reject"))
    c = synth.cos_psi(22.5)
    for b_out, b_edge in [(900.0, lib.b_max), (30.0, lib.b_min)]:
        I, VI, bad = fc.spectra(np.stack([pole_on(b_out, c), pole_on(b_edge, c)]))
        assert not bad.any()
        np.testing.assert_allclose(I[0], I[1], rtol=1e-12)
        np.testing.assert_allclose(VI[0], VI[1], rtol=1e-12, atol=1e-15)
        assert fc.outside_fraction(pole_on(b_out, c)) == pytest.approx([1.0], abs=1e-15)
        I, VI, bad = fr.spectra(np.stack([pole_on(b_out, c), pole_on(b_edge, c)]))
        assert bad.tolist() == [True, False]
        assert np.isnan(I[0]).all() and np.isfinite(I[1]).all()


@pytest.mark.parametrize("ld_mode", [0, 3])
@pytest.mark.parametrize("bp", [80.0, 90.0])
def test_v2d_outside_fraction_of_a_pole_on_dipole(lib, ld_mode, bp):
    fw = Forward(lib, fibonacci(20000), field.get_model("dqo"), dataset(phases=(0.0, 0.5)), cfg(ld_mode=ld_mode))
    mstar = math.sqrt(((2 * lib.b_min / bp) ** 2 - 1) / 3)
    want = _cdf(mstar, ld_mode)
    np.testing.assert_allclose(fw.outside_fraction(dipole(bp)), [want, want], atol=1e-3)
    fr = Forward(lib, fibonacci(2000), field.get_model("dqo"), dataset(), cfg("reject", ld_mode=ld_mode))
    assert fr.spectra(dipole(bp))[2].tolist() == [True]
    assert fr.spectra(dipole(2 * lib.b_min + 1))[2].tolist() == [False]
