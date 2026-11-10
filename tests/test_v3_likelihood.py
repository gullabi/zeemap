"""V3: symmetries and consistency, likelihood items: V3e (phase order), V3g (plain chi2 scalings and
weights), V3h (full likelihood under scaled errors), V3i (switching a spectrum off)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zeemap.likelihood import Objective
from test_v1_likelihood import cfg_of, make_dataset, perturbed_models

pytestmark = pytest.mark.tier1


def scaled(ds, factor=None, weight=None):
    """A copy of ds with every error multiplied by `factor`, or every weight set from `weight(spectrum)`."""
    spectra = []
    for sp in ds.spectra:
        kw = {}
        if factor is not None:
            kw["sigma"] = sp.sigma * factor
        if weight is not None:
            kw["weight"] = weight(sp)
        spectra.append(dataclasses.replace(sp, **kw))
    return dataclasses.replace(ds, spectra=spectra)


@pytest.mark.parametrize("cfg", [cfg_of(), cfg_of(kind="chi2"), cfg_of(scale=False, continuum={"degree": 2})])
def test_v3e_phase_order_does_not_matter(cfg):
    rng = np.random.default_rng(30)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0, 0.3, 0.7))
    I, VI = perturbed_models(rng, I_true, VI_true, 5)
    perm = np.array([2, 0, 1])  # new phase q holds old phase perm[q]
    inv = np.argsort(perm)
    spectra = [dataclasses.replace(sp, phase_index=int(inv[sp.phase_index])) for sp in ds.spectra]
    spectra = [spectra[i] for i in rng.permutation(len(spectra))]
    ds2 = dataclasses.replace(ds, phases=ds.phases[perm], spectra=spectra, I=ds.I[perm], VI=ds.VI[perm],
                              sig_I=ds.sig_I[perm], sig_VI=ds.sig_VI[perm])
    a = Objective(ds, cfg)
    b = Objective(ds2, cfg)
    np.testing.assert_allclose(b.neg_log_l(I[:, perm], VI[:, perm]), a.neg_log_l(I, VI), rtol=1e-12)
    assert b.constant == pytest.approx(a.constant, rel=1e-12)


def test_v3g_chi2_errors_times_k_divide_chi2_by_k_squared():
    rng = np.random.default_rng(31)
    ds, I_true, VI_true = make_dataset(rng)
    I, VI = perturbed_models(rng, I_true, VI_true, 4)
    k = 2.7
    a = Objective(ds, cfg_of(kind="chi2")).neg_log_l(I, VI)
    b = Objective(scaled(ds, factor=k), cfg_of(kind="chi2")).neg_log_l(I, VI)
    np.testing.assert_allclose(b, a / k**2, rtol=1e-12)


@pytest.mark.parametrize("kind", ["chi2", "full"])
def test_v3g_weight_equals_errors_divided_by_sqrt_weight(kind):
    """DATA-4 (decision #27): with every likelihood a weight w acts as the errors divided by sqrt(w); the plain-numpy
    reference agrees for kind full."""
    from zeemap import likelihood_reference as ref

    rng = np.random.default_rng(32)
    ds, I_true, VI_true = make_dataset(rng, weights=True)
    I, VI = perturbed_models(rng, I_true, VI_true, 4)
    a = Objective(ds, cfg_of(kind=kind)).neg_log_l(I, VI)
    spectra = [dataclasses.replace(sp, sigma=sp.sigma / np.sqrt(sp.weight), weight=np.ones_like(sp.weight))
               for sp in ds.spectra]
    ds_b = dataclasses.replace(ds, spectra=spectra)
    b = Objective(ds_b, cfg_of(kind=kind)).neg_log_l(I, VI)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    if kind == "full":
        for k in range(2):
            want, _ = ref.neg_log_l(ds, cfg_of(kind=kind), I[k], VI[k])
            assert a[k] == pytest.approx(want, rel=1e-10)


def test_v3g_huge_error_equals_pixel_left_out():
    rng = np.random.default_rng(33)
    ds, I_true, VI_true = make_dataset(rng)
    I, VI = perturbed_models(rng, I_true, VI_true, 4)
    for j in (0, 1):  # an I and a V/I spectrum
        cut = 17
        big = [dataclasses.replace(sp, sigma=sp.sigma.copy()) for sp in ds.spectra]
        big[j].sigma[cut] *= 1e6
        keep = np.arange(ds.spectra[j].pixels.size) != cut
        sp = ds.spectra[j]
        out = list(ds.spectra)
        out[j] = dataclasses.replace(sp, pixels=sp.pixels[keep], data=sp.data[keep], sigma=sp.sigma[keep],
                                     weight=sp.weight[keep], window=sp.window[keep])
        a = Objective(dataclasses.replace(ds, spectra=big), cfg_of(kind="chi2")).neg_log_l(I, VI)
        b = Objective(dataclasses.replace(ds, spectra=out), cfg_of(kind="chi2")).neg_log_l(I, VI)
        np.testing.assert_allclose(a, b, rtol=1e-9)


@pytest.mark.parametrize("cfg", [cfg_of(), cfg_of(corr=False, disc=False), cfg_of(continuum=False)])
def test_v3h_full_errors_times_k_shift_by_a_constant(cfg):
    rng = np.random.default_rng(34)
    ds, I_true, VI_true = make_dataset(rng)
    I, VI = perturbed_models(rng, I_true, VI_true, 5)
    k = 3.3
    oa = Objective(ds, cfg)
    ob = Objective(scaled(ds, factor=k), cfg)
    a, b = oa.neg_log_l(I, VI), ob.neg_log_l(I, VI)
    shift = b - a
    np.testing.assert_allclose(shift - shift[0], 0.0, atol=1e-9 * np.max(np.abs(a)))
    # the shift is exactly what the exposed constant takes up: -ln L + constant is unchanged
    np.testing.assert_allclose(b + ob.constant, a + oa.constant, rtol=1e-9)


@pytest.mark.parametrize("cfg", [cfg_of(), cfg_of(kind="chi2"), cfg_of(scale=False)])
def test_v3i_switching_a_spectrum_off_removes_its_term(cfg):
    rng = np.random.default_rng(35)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0, 0.5))
    I, VI = perturbed_models(rng, I_true, VI_true, 4)
    obj = Objective(ds, cfg)
    full = obj.neg_log_l(I, VI)
    T = obj.terms(I, VI)
    np.testing.assert_allclose(T.sum(axis=1), full, rtol=1e-12)
    for j in range(len(ds.spectra)):
        rest = dataclasses.replace(ds, spectra=[sp for i, sp in enumerate(ds.spectra) if i != j])
        got = Objective(rest, cfg).neg_log_l(I, VI)
        np.testing.assert_allclose(got, full - T[:, j], rtol=1e-12, atol=1e-12 * np.max(np.abs(full)))


@pytest.mark.parametrize("i_offset", [False, True])
@pytest.mark.parametrize("cfg", [cfg_of(), cfg_of(scale=False), cfg_of(corr=False, disc=False), cfg_of(kind="chi2")])
def test_v3h_scale_model_times_1e15_shifts_by_a_constant(cfg, i_offset):
    """Multiplying the model I by 1e15 (V/I unchanged: it is scale-free) leaves -ln L unchanged, to 1e-6 nats: the
    continuum basis is normalised by the model's size (decision #26; without that it would move by (deg + 1) ln(1e15)
    per I spectrum, the ln det(A^T A) term); the likelihood reference agrees at both scales."""
    from zeemap import likelihood_reference as ref

    rng = np.random.default_rng(71)
    ds, I_true, VI_true = make_dataset(rng, i_offset=i_offset)
    I, VI = perturbed_models(rng, I_true, VI_true, 10, size=0.1)
    ob = Objective(ds, cfg)
    a = ob.neg_log_l(I, VI)
    b = ob.neg_log_l(1e15 * I, VI)
    n_i = sum(1 for sp in ds.spectra if sp.kind == "I")
    deg = ob.settings["continuum"]["degree"]
    del n_i, deg
    np.testing.assert_allclose(b - a, 0.0, rtol=0, atol=1e-6)
    for k in range(3):
        for scale, val in ((1.0, a[k]), (1e15, b[k])):
            want, _ = ref.neg_log_l(ds, cfg, scale * I[k], VI[k])
            assert val == pytest.approx(want, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("deg", [0, 1, 3])
@pytest.mark.parametrize("i_offset", [False, True])
@pytest.mark.parametrize("c", [0.01, 7.0, 100.0])
def test_v3l_full_likelihood_does_not_change_with_the_model_scale(deg, i_offset, c):
    """LIKE-3 (decision #26): with the normalised continuum basis, multiplying the model I by c leaves the full -ln L
    unchanged (1e-10 relative), with and without the free constant, at degrees 0, 1 and 3; the reference agrees."""
    from zeemap import likelihood_reference as ref

    rng = np.random.default_rng(81)
    ds, I_true, VI_true = make_dataset(rng, i_offset=i_offset)
    I, VI = perturbed_models(rng, I_true, VI_true, 3, size=0.1)
    cfg = cfg_of(continuum={"degree": deg})
    ob = Objective(ds, cfg)
    a, b = ob.neg_log_l(I, VI), ob.neg_log_l(c * I, VI)
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-9)
    want, _ = ref.neg_log_l(ds, cfg, c * I[0], VI[0])
    assert b[0] == pytest.approx(want, rel=1e-10, abs=1e-9)
