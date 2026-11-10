"""V1o (kernel fit) and V1p (whiteness check): noise.py (SPEC LIKE-2, LIKE-4, LIKE-10; decision #22)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zeemap import noise
from zeemap.likelihood import Objective
from test_v1_likelihood import GRID, cfg_of, make_dataset, perturbed_models

pytestmark = pytest.mark.tier1


def _given(sp, th):
    return {"spectrum": sp.label, "rho_short": th[0], "a": th[1], "rho_long": th[2], "j": th[3]}


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("i_offset", [False, True])
def test_v1o_marginal_equals_objective_and_reference(scale, i_offset):
    """The kernel-fit objective is the exact marginal likelihood with ln det K: equal to Objective.neg_log_l + constant
    for the same kernel given per spectrum, and to the plain-numpy reference (term + constant), to 1e-8."""
    from zeemap import likelihood_reference as ref

    rng = np.random.default_rng(91)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0,), i_offset=i_offset, n_fit=30)
    I, VI = perturbed_models(rng, I_true, VI_true, 1, size=0.05)
    th = (3.0, 1.7, 25.0, 0.4)
    lam_all = GRID.wavelengths
    cfg = cfg_of(scale=scale, continuum={"degree": 2})
    cfg["objective"]["kernel"] = [_given(sp, th) for sp in ds.spectra]
    ob = Objective(ds, cfg)
    want = ob.neg_log_l(I, VI)[0] + ob.constant
    got = 0.0
    for sp in ds.spectra:
        model = (I[0] if sp.kind == "I" else VI[0])[sp.phase_index, sp.pixels]
        deg = 2 if sp.kind == "I" else -1
        got += noise.marginal_neg_log_l(lam_all[sp.pixels], sp.data, sp.sigma, model, sp.kind, deg,
                                        sp.i_offset and sp.kind == "I", th, scale)
    assert got == pytest.approx(want, rel=1e-8, abs=1e-8)
    # the reference with the same kernel for every spectrum (its corr/disc tables)
    rcfg = cfg_of(scale=scale, continuum={"degree": 2}, corr={"rho": th[0], "jitter": th[3]},
                  disc={"a_I": th[1], "rho_I": th[2], "a_VI": th[1], "rho_VI": th[2]})
    term, const = ref.neg_log_l(ds, rcfg, I[0], VI[0])
    assert got == pytest.approx(term + const, rel=1e-8, abs=1e-8)


def _draw(rng, lam, th, n_draws):
    K = noise.kernel_matrix(lam, *th)
    L = np.linalg.cholesky(K)
    return [L @ rng.standard_normal(lam.size) for _ in range(n_draws)]


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v1o_fit_recovers_a_known_kernel():
    """8 noise draws (300 pixels of 2.5 A) from two Materns, 2.5 A and 40 A, a = 2: the fit never ends below the value
    at the truth (it maximises), and the medians of rho_long, rho_short and a lie within a factor 1.5 of the truth."""
    rng = np.random.default_rng(92)
    lam = 4000.0 + 2.5 * np.arange(300)
    truth = (2.5, 2.0, 40.0, 0.05)
    sig = np.ones(lam.size)
    model = np.zeros(lam.size)
    fits = []
    for y in _draw(rng, lam, truth, 8):
        f_true = noise.marginal_neg_log_l(lam, y, sig, model, "VI", -1, False, truth)
        r = noise.fit_spectrum_kernel(lam, y, sig, model, "VI", -1, False, 2.5, 20.0, (4.0, 0.42, 205.0, 0.05))
        assert r["neg_log_l"] <= f_true + 1e-6
        fits.append(r)
    for key, t in (("rho_long", 40.0), ("rho_short", 2.5), ("a", 2.0)):
        med = float(np.median([r[key] for r in fits]))
        assert t / 1.5 <= med <= t * 1.5, (key, med)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v1o_white_noise_fitted_kernel_whitens_it():
    """10 white Gaussian series of 400 pixels: after the kernel fit (whose white part can describe white noise),
    the whitened residuals pass LIKE-10."""
    rng = np.random.default_rng(93)
    lam = 4000.0 + 2.5 * np.arange(400)
    sig = np.ones(lam.size)
    model = np.zeros(lam.size)
    pix = np.arange(lam.size)
    for _ in range(10):
        y = rng.standard_normal(lam.size)
        r = noise.fit_spectrum_kernel(lam, y, sig, model, "VI", -1, False, 2.5, 20.0, (4.0, 0.42, 205.0, 0.05))
        K = noise.kernel_matrix(lam, r["rho_short"], r["a"], r["rho_long"], r["j"])
        e = np.linalg.solve(np.linalg.cholesky(K), y)
        assert noise.acf_check(e, pix)["pass"], r


def test_v1o_bounds_follow_the_fwhm_and_the_window():
    lam = 4000.0 + 2.5 * np.r_[np.arange(100), np.arange(150, 400)]  # runs of 247.5 A and 622.5 A
    b = noise.kernel_bounds(lam, 2.5, 20.0)
    assert b[0] == (1.25, 20.0) and b[2] == (20.0, 0.25 * 622.5) and b[3] == (noise.J_MIN, noise.J_MAX)
    b = noise.kernel_bounds(lam[:20], 2.5, 20.0)  # a quarter of 47.5 A is below the FWHM: rho_long = FWHM
    assert b[2] == (20.0, 20.0)


def test_v1o_parameters_at_a_bound_are_named():
    """White noise ends in the white corner (rho_short and a lower, j upper; rho_long, unidentified there,
    not named) and is not reported; a smooth misfit of 5 sigma ends at a = A_MAX with rho_long and j free and is
    reported; a parameter held by an empty range (rho_long on 40 pixels, FWHM 13) is not named."""
    rng = np.random.default_rng(0)
    lam = 4000.0 + np.arange(400)
    sig = np.full(lam.size, 0.01)
    model = np.ones(lam.size)
    white = model + sig * rng.standard_normal(lam.size)
    smooth = white + 0.05 * np.sin(2 * np.pi * (lam - 4000.0) / 150.0)
    fits = {}
    for name, y in (("white", white), ("smooth", smooth)):
        r = noise.fit_spectrum_kernel(lam, y, sig, model, "VI", -1, False, 1.0, 3.0, (2.0, 0.1, 30.0, 0.3))
        fits[name] = {"spectrum": name, **r}
    assert fits["white"]["at_bound"] == {"rho_short": "lower", "a": "lower", "j": "upper"}
    assert fits["smooth"]["at_bound"] == {"rho_short": "lower", "a": "upper"}
    assert noise.capped_kernels(list(fits.values())) == {"smooth": {"rho_short": "lower", "a": "upper"}}
    r = noise.fit_spectrum_kernel(lam[:40], smooth[:40], sig[:40], model[:40], "VI", -1, False, 1.0, 13.0,
                                  (2.0, 0.1, 30.0, 0.3))
    assert r["a"] > 0.01 and r["rho_long"] == 13.0 and "rho_long" not in r["at_bound"]


@pytest.mark.parametrize("i_offset", [False, True])
def test_v1o_plain_chi2_uses_the_likelihoods_continuum(i_offset):
    """per_spectrum (DIAG-1's chi2 of the mode, eval.json) is the plain chi2 of the model I with the continuum
    fitted under the spectrum's kernel, the one spectra/ writes, not of a separate least-squares continuum (under a
    fitted kernel the two can differ by a factor of 100)."""
    rng = np.random.default_rng(94)
    ds, I_true, VI_true = make_dataset(rng, phases=(0.0,), i_offset=i_offset)
    I, VI = perturbed_models(rng, I_true, VI_true, 1, size=0.05)
    ob = Objective(ds, cfg_of(kernel=[_given(sp, (2.5, 10.0, 40.0, 0.05)) for sp in ds.spectra]))
    lam = GRID.wavelengths
    for j, sp in enumerate(ds.spectra):
        m = (I if sp.kind == "I" else VI)[0, 0, sp.pixels]
        y, sig = sp.data, sp.sigma
        A = m[:, None]
        if sp.kind == "I":
            x = 2.0 * (lam[sp.pixels] - lam[sp.pixels][0]) / (lam[sp.pixels][-1] - lam[sp.pixels][0]) - 1.0
            A = np.column_stack([m * np.polynomial.legendre.legval(x, np.eye(4)[d]) for d in range(4)]
                                + ([np.ones(m.size)] if i_offset else []))
            W = np.linalg.inv(np.linalg.cholesky(ob.kernel(j))) / sig[None, :]
            coef, *_ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
            fitted = A @ coef
        else:
            fitted = m
        want = float(np.sum(((y - fitted) / sig) ** 2))
        assert ob.per_spectrum(I[0], VI[0])[sp.label]["chi2"] == pytest.approx(want, rel=1e-8)


def test_v1p_whiteness_rule():
    """200 white series of 900 pixels pass (a correct check fails one of them with probability about 0.014 over
    seeds); 20 AR(1) series with coefficient 0.3 fail; the threshold depends on n; pairs across a gap are not counted."""
    rng = np.random.default_rng(94)
    pix = np.arange(900)
    assert all(noise.acf_check(rng.standard_normal(900), pix)["pass"] for _ in range(200))
    for _ in range(20):
        e = np.empty(900)
        e[0] = rng.standard_normal()
        for i in range(1, 900):
            e[i] = 0.3 * e[i - 1] + rng.standard_normal()
        r = noise.acf_check(e, pix)
        assert not r["pass"] and r["lag"] == 1
    assert noise.acf_check(rng.standard_normal(400), np.arange(400))["threshold"] == pytest.approx(0.2)
    assert noise.acf_check(rng.standard_normal(900), pix)["threshold"] == pytest.approx(0.15)
    # a series correlated only across a gap: the pairs that span the gap are not counted
    e = rng.standard_normal(200)
    e[100:] = e[:100]  # pixel 100 + i equals pixel i; with a gap of 50 grid steps, no lag <= 10 joins them
    gap = np.r_[np.arange(100), np.arange(150, 250)]
    assert noise.acf_check(e, gap)["max_abs_acf"] < 0.3
    # isolated pixels whose array neighbours are equal (each value twice in a row) have a lag-1
    # autocorrelation of 0.5 over the array, but no two of them are fewer than 20 grid steps apart, so no pair is
    # counted and the check passes with 0; without the contiguity mask it fails
    e = np.repeat(rng.standard_normal(150), 2)
    r = noise.acf_check(e, 20 * np.arange(300))
    assert r["max_abs_acf"] == 0.0 and r["pass"]
