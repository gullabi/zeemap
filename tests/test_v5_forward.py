"""V5a: the forward model (numba scatter, one matrix product per batch) against the plain-numpy reference
(forward_reference.py), single and batched calls, random parameters, both outside rules, Teff per row; 1e-10 relative
(VERIFICATION.md V5)."""

import numpy as np
import pytest

from zeemap import field
from zeemap.forward import Forward
from zeemap.forward_reference import ForwardReference
from zeemap.library import Library
from zeemap.surface import fibonacci
from zeemap.types import DataSet, Grid, Spectrum

pytestmark = pytest.mark.tier1

PHASES = (0.0, 0.3, 0.65)


def dataset(phases=PHASES, start=4100.0, step=2.5, npix=1100):
    grid = Grid(start, step, npix)
    P, M = len(phases), npix
    sp = [Spectrum(0, float(phases[0]), "VI", np.arange(M), np.zeros(M), np.ones(M), np.ones(M), np.zeros(M, int))]
    nan = np.full((P, M), np.nan)
    return DataSet(grid, np.asarray(phases, float), sp, nan, nan.copy(), nan.copy(), nan.copy())


def cfg(outside="clamp", **model):
    m = {"interp_b": "linear", "ld_mode": 3, "fwhm": 7.0, "phase_offset": 0.1}
    m.update(model)
    return {"model": m, "library": {"outside": outside}}


def random_thetas(rng, K):
    Th = np.zeros((K, 14))
    d = rng.normal(size=(K, 3))
    Th[:, 0:3] = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(100, 700, (K, 1))
    Th[:, 3] = rng.uniform(-150, 150, K)
    Th[:, 4:7] = rng.normal(size=(K, 3))
    Th[:, 7:10] = rng.normal(0, 40, (K, 3))
    Th[:, 10:13] = rng.uniform(-0.2, 0.2, (K, 3))
    Th[:, 13] = rng.uniform(-1, 1, K)
    return Th


def assert_same(a, b, rtol=1e-10):
    for x, y in zip(a[:2], b[:2]):
        np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
        ok = ~np.isnan(x)
        scale = np.abs(y[ok]).max() if ok.any() else 1.0
        np.testing.assert_allclose(x[ok], y[ok], rtol=0, atol=rtol * scale)
    np.testing.assert_array_equal(a[2], b[2])


@pytest.fixture(scope="module")
def lib1(synth_lib):
    return Library(synth_lib, teffs=[30000.0])


@pytest.fixture(scope="module")
def lib2(synth_lib):
    return Library(synth_lib, teffs=[20000.0, 30000.0, 40000.0])


@pytest.mark.parametrize("outside", ["clamp", "reject"])
def test_v5a_forward_against_reference(lib1, outside):
    surf = fibonacci(1500)
    c = cfg(outside)
    fast = Forward(lib1, surf, field.get_model("dqo"), dataset(), c)
    ref = ForwardReference(lib1, surf, field.get_model("dqo"), dataset(), c)
    rng = np.random.default_rng(11)
    Th = random_thetas(rng, 6)
    Th[5, 0:3] *= 0.2  # weak field: patches below b_min (clamped or rejected)
    batch = fast.spectra(Th)
    assert_same(batch, ref.spectra(Th))
    if outside == "reject":
        assert batch[2][5]
    for k in (0, 5):  # single calls equal the batched rows
        assert_same(fast.spectra(Th[k]), tuple(x[k : k + 1] for x in batch))
        np.testing.assert_allclose(fast.node_weights(Th[k]), ref.node_weights(Th[k]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(fast.outside_fraction(Th[k]), ref.outside_fraction(Th[k]), rtol=0, atol=1e-12)


def test_v5a_teff_per_row(lib2):
    surf = fibonacci(320)
    c = cfg()
    fast = Forward(lib2, surf, field.get_model("dqo"), dataset(), c)
    ref = ForwardReference(lib2, surf, field.get_model("dqo"), dataset(), c)
    Th = random_thetas(np.random.default_rng(5), 5)
    teff = np.array([20000.0, 24500.0, 30000.0, 39000.0, 45000.0])  # the last is outside the selection: invalid
    got = fast.spectra(Th, teff=teff)
    assert_same(got, ref.spectra(Th, teff=teff))
    assert got[2].tolist() == [False, False, False, False, True]
    with pytest.raises(ValueError, match="Teff slices"):
        fast.spectra(Th)


@pytest.mark.parametrize("interp_b", ["cubic", "pchip"])
@pytest.mark.parametrize("outside", ["clamp", "reject"])
def test_v5a_cubic_b_interpolation_against_reference(lib1, lib2, interp_b, outside):
    """The cubic Hermite node weights (numba, slope matrix) against the written-out reference (Lagrange slopes)."""
    surf = fibonacci(1500)
    c = cfg(outside, interp_b=interp_b)
    fast = Forward(lib1, surf, field.get_model("dqo"), dataset(), c)
    ref = ForwardReference(lib1, surf, field.get_model("dqo"), dataset(), c)
    rng = np.random.default_rng(13)
    Th = random_thetas(rng, 6)
    Th[5, 0:3] *= 0.2  # weak field: patches below b_min, including the first B intervals (one-sided end slopes)
    batch = fast.spectra(Th)
    assert_same(batch, ref.spectra(Th))
    for k in (0, 5):
        assert_same(fast.spectra(Th[k]), tuple(x[k : k + 1] for x in batch))
        np.testing.assert_allclose(fast.node_weights(Th[k]), ref.node_weights(Th[k]), rtol=0, atol=1e-12)
    lin = Forward(lib1, surf, field.get_model("dqo"), dataset(), cfg(outside))
    assert np.nanmax(np.abs(lin.spectra(Th)[0] - batch[0])) > 1e-4  # the option changes the model (I)
    if outside == "clamp":
        fast2 = Forward(lib2, surf, field.get_model("dqo"), dataset(), c)
        ref2 = ForwardReference(lib2, surf, field.get_model("dqo"), dataset(), c)
        teff = np.array([20000.0, 24500.0, 30000.0, 39000.0, 33000.0, 21000.0])
        assert_same(fast2.spectra(Th, teff=teff), ref2.spectra(Th, teff=teff))


def test_v5a_node_stage_surface_outside_fraction(lib1):
    """The smoothed whole-surface outside fraction of the node stage (all patches, visible or not; problem.py's prior
    cut) equals forward.surface_outside and the formula written out."""
    from scipy.special import expit

    from zeemap.forward import SMOOTH_WIDTH_MG

    surf = fibonacci(1500)
    fw = Forward(lib1, surf, field.get_model("dqo"), dataset(), cfg())
    Th = random_thetas(np.random.default_rng(5), 6)
    Th[5, 0:3] *= 0.2
    Th[4, 10:13] = [0.9, 0.5, 0.0]  # |c| > 1: invalid row, the fraction is still defined
    fs = fw.node_weights_batch(Th)[4]
    np.testing.assert_array_equal(fs, fw.surface_outside(Th))
    with np.errstate(all="ignore"):
        B = field.get_model("dqo").evaluate_batch(surf.centres, Th)
    b = np.sqrt(np.einsum("kni,kni->kn", B, B))
    s = SMOOTH_WIDTH_MG / 4.0
    f = np.where(np.isfinite(b), expit((b - lib1.b_max) / s) + expit((lib1.b_min - b) / s), 1.0)
    want = f @ (surf.areas / surf.areas.sum())
    np.testing.assert_allclose(fs, want, rtol=1e-14, atol=1e-16)
    assert 0.0 <= fs.min() and fs.max() <= 1.0 and fs[5] > 0.0
    hard = fw.surface_outside(Th, smooth=False)
    out = ~np.isfinite(b) | (b < lib1.b_min) | (b > lib1.b_max)
    np.testing.assert_allclose(hard, out @ (surf.areas / surf.areas.sum()), rtol=1e-14, atol=1e-16)


def test_v5_records_factorised_to_float32_rounding(lib2):
    """Each slice's records are stored as A @ B (A orthonormal, nodes x r) when that holds every Stokes part
    to COMPRESS_TOL relative (below the float32 rounding of the stored library) and saves work. A matrix with decaying
    singular values and no exact rank, its V part independent of its I part and 1e-3 of it, with NaN pixels: each part
    within the tolerance (one global norm instead of the per-part one leaves V about 800 times above it), NaN kept; a full-rank one is left alone (None); the forward model on the synthetic library uses the
    factors and its disc sum equals W @ records, the records being A @ B."""
    from zeemap.forward import COMPRESS_TOL, factor_records

    rng = np.random.default_rng(3)
    n, M = 300, 200
    sv = 10.0 ** -np.arange(0, 12, 0.5)
    R = np.stack([(rng.normal(size=(n, sv.size)) * sv) @ rng.normal(size=(sv.size, M)) for _ in range(2)], axis=1)
    R[:, 1] *= 1e-3
    R[:, :, :7] = np.nan
    A, B = factor_records(R)
    assert 5 < A.shape[1] < 2 * sv.size  # I and V independent: rank up to 2 x 24
    got = (A @ B).reshape(n, 2, M)
    assert np.all(np.isnan(got[:, :, :7]))
    for k in (0, 1):
        assert np.linalg.norm(got[:, k, 7:] - R[:, k, 7:]) <= COMPRESS_TOL * np.linalg.norm(R[:, k, 7:])
    assert factor_records(rng.normal(size=(n, 2, M))) is None

    fw = Forward(lib2, fibonacci(500), field.get_model("dqo"), dataset(), cfg())
    assert all(f is not None and f[0].shape[1] < fw.records.shape[1] // 4 for f in fw.factors)
    W = rng.random((6, fw.records.shape[1]))
    C = np.ones((3, 3))
    S = fw._disc_sum(W, C, 3, 2).reshape(6, -1)
    want = sum(W @ f for f in fw._flat)
    ok = np.isfinite(want)
    np.testing.assert_allclose(S[ok], want[ok], rtol=1e-12, atol=1e-12 * np.abs(want[ok]).max())
