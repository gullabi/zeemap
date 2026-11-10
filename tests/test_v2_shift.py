"""V2g: the wavelength shift (FWD-13; decision #27; DESIGN.md section 5b), on the synthetic library.

Data are made by the model with a fixed shift (23 km/s, between the shift nodes, or 0.7 A); fitting only the shift
(free, its records at 5 nodes over [lo, hi], cubic Lagrange weights) returns it within 0.5 km/s or 0.01 A on
noise-free data; the data arrays the likelihood receives are the input arrays, unchanged; the model with the shift
fixed at the truth reproduces the data (cost below 1e-8).

Because the data and the fit share the model, those checks cannot see the sign or the unit of the shift (flipping
both signs in `Forward.shifted_wavelengths` passes them). The convention is anchored
outside the model: the synthetic line k = 2 lies at 5500 A for every B (tests/synth.py), so in `Forward.spectra` with
a fixed shift v its centroid must sit at 5500 (1 + v / c) for a velocity and at 5500 + v for a constant in A; a
positive v moves the line to longer wavelengths (a redshift; DESIGN.md section 5b).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zeemap import cli
from zeemap import search as S
from zeemap.config import load_config
from zeemap.data import load_data
from zeemap.forward import C_KMS
from zeemap.problem import Problem, blank_dataset

from test_v7_cli import MD_TRUE, U_TRUE, config_text

pytestmark = pytest.mark.tier1


def _with_shift(txt, value, free, lo, hi, step, unit):
    txt = txt.replace("[search]\n", f"shift = {{ value = {value}, step = {step}, lo = {lo}, hi = {hi}, free = {free} }}"
                      "\n\n[search]\n")
    return txt.replace("[model]\n", f'[model]\nshift_unit = "{unit}"\n')


@pytest.mark.parametrize("unit,truth,lo,hi,step,tol", [("kms", 23.0, -60.0, 60.0, 5.0, 0.5),
                                                       ("A", 0.7, -1.5, 1.5, 0.1, 0.01)])
def test_v2g_shift_recovered_and_data_unchanged(synth_lib, tmp_path, unit, truth, lo, hi, step, tol):
    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    base = config_text("inj", kind="chi2", md=MD_TRUE, u=U_TRUE)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "truth.toml").write_text(_with_shift(base, truth, "false", lo, hi, step, unit))
        assert cli.main(["inject", str(root / "truth.toml"), "-o", str(root / "inj"), "--noise", "none"]) == 0
        fit_txt = _with_shift(base, 0.0, "true", lo, hi, step, unit)
        fit_txt = fit_txt.replace("md = { value", "md = { value").replace("free = true }\nbq", "free = false }\nbq")
        fit_txt = fit_txt.replace("u = { value = 0.5, step = 0.1, lo = -1.0, hi = 1.0, free = true }",
                                  "u = { value = 0.5, step = 0.1, lo = -1.0, hi = 1.0, free = false }")
        (root / "fit.toml").write_text(fit_txt)
        cfg = load_config(root / "fit.toml")
        p = Problem(cfg)
        raw = load_data(cfg)
    assert list(np.flatnonzero(p.free)) == [p.names.index("shift")]  # only the shift is fitted
    np.testing.assert_array_equal(p.dataset.I, raw.I)  # FWD-13: the data are not interpolated
    np.testing.assert_array_equal(p.dataset.VI, raw.VI)
    r = S.polish(p, np.array(p.value, float), cfg)
    got = float(r.x[p.names.index("shift")])
    assert abs(got - truth) <= tol, (got, truth)
    # a fixed shift equal to the truth reproduces the data exactly (no interpolation in v); the free problem at the
    # truth is not exact (the Lagrange weights between the shift nodes leave about 0.04 at 23 km/s, 0.005 at 0.7 A)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        pt = Problem(load_config(root / "truth.toml"))
    assert pt.forward.shift_nodes is None and pt.forward.shift_fixed == truth
    assert float(pt.cost(np.array(pt.value, float)[None])[0]) < 1e-8


# The line k = 2 of the synthetic library is at 5500 A for every B; on the data grid (2 A pixels, 5500 A on a pixel)
# its centroid over [5470, 5530] A is measured to about 4e-4 A, so 0.01 A (0.5 % of a pixel) is a sub-pixel tolerance
# far below the 11 A between the two signs at these shifts.
LINE_A, HALF_WINDOW_A, CENTROID_TOL_A = 5500.0, 30.0, 0.01


def line_centroid(lam, I):
    """Centroid of the depression below the continuum of I over LINE_A +- HALF_WINDOW_A; the continuum is the mean of
    the three end pixels at each side of the window (no other line comes within 30 A of the window, tests/synth.py)."""
    sel = np.abs(lam - LINE_A) <= HALF_WINDOW_A
    x, y = lam[sel], I[sel]
    d = 0.5 * (y[:3].mean() + y[-3:].mean()) - y
    return float(np.sum(d * x) / np.sum(d))


@pytest.mark.parametrize("unit,v,want", [("kms", 300.0, LINE_A * (1.0 + 300.0 / C_KMS)),
                                         ("kms", -300.0, LINE_A * (1.0 - 300.0 / C_KMS)),
                                         ("A", 5.0, LINE_A + 5.0), ("A", -5.0, LINE_A - 5.0),
                                         ("kms", 0.0, LINE_A)])
def test_v2g_shift_moves_the_line_to_the_stated_wavelength(synth_lib, tmp_path, unit, v, want):
    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    lim = 1000.0 if unit == "kms" else 10.0
    txt = _with_shift(config_text("inj", kind="chi2", md=MD_TRUE, u=U_TRUE), v, "false", -lim, lim, 1.0, unit)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "t.toml").write_text(txt)
        cfg = load_config(root / "t.toml")
        p = Problem(cfg, blank_dataset(cfg))
    assert p.forward.shift_fixed == v
    I, _, bad = p.forward.spectra(p.thetas(np.array(p.value, float)[None]))
    assert not bad.any()
    g = p.dataset.grid
    lam = g.start + g.step * np.arange(g.npix)
    for ph in range(I.shape[1]):
        got = line_centroid(lam, I[0, ph])
        assert abs(got - want) <= CENTROID_TOL_A, (unit, v, ph, got, want)


# ------------------------------------------------------------------------------------------------------------- V2i


@pytest.mark.parametrize("truth,teffs,teff", [(0.0, [30000.0], None),
                                              (30.0, [20000.0, 30000.0, 40000.0], 30000.0),
                                              (-30.0, [20000.0, 30000.0, 40000.0], 35000.0)])
def test_v2i_floor_chi2_with_a_free_shift(synth_lib, tmp_path, truth, teffs, teff):
    """DIAG-1 with a free shift: the records are stored per (Teff slice, shift node), index t * n_shift + j, and the
    floor takes the bracketing slices' records at the mode's shift, combined over the shift nodes with the forward
    model's Lagrange weights. Noise-free data made with a fixed shift at a node (0, 30 or -30 km/s; nodes at -60, -30,
    0, 30, 60): floor below 1e-8 x the chi2 of the zero model, as with the shift fixed; one Teff slice, or three with
    the truth on a slice or between two (the synthetic slices differ by a constant factor, so a Teff between them is
    still a non-negative mixture under the "log" Teff weights; the "planck" weights vary with wavelength). Before the fix the floor took record set t (Teff t // 5, shift node t % 5) and gave
    about 400 per phase in the first case. At a wrong shift (-60 km/s) the floor is far above 0."""
    from zeemap.diagnostics import floor_chi2

    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    base = config_text("inj", kind="chi2", md=MD_TRUE, u=U_TRUE).replace("n = 1000", "n = 300")
    base = base.replace("teffs = [30000.0]", f"teffs = {teffs!r}").replace("[model]\n", '[model]\ninterp_teff = "log"\n')
    if teff is not None:
        base = base.replace("[search]\n", f"teff = {{ value = {teff}, step = 1000.0, lo = 20000.0, hi = 40000.0, "
                                          "free = true }\n\n[search]\n")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "truth.toml").write_text(_with_shift(base, truth, "false", -60.0, 60.0, 5.0, "kms"))
        assert cli.main(["inject", str(root / "truth.toml"), "-o", str(root / "inj"), "--noise", "none"]) == 0
        (root / "fit.toml").write_text(_with_shift(base, truth, "true", -60.0, 60.0, 5.0, "kms"))
        p = Problem(load_config(root / "fit.toml"))
    assert p.forward.n_shift == 5 and p.forward.records.shape[0] == 5 * len(teffs)
    x = np.array(p.value, float)
    assert float(p.cost(x[None])[0]) < 1e-8  # the model at the truth reproduces the data
    zero = sum(float(np.sum((np.asarray(s.data) / np.asarray(s.sigma)) ** 2)) for s in p.dataset.spectra)
    fc = floor_chi2(p, x)
    assert len(fc) == 2
    want = [teff or 30000.0] if teff in (None, 30000.0) else [30000.0, 40000.0]
    for v in fc.values():
        assert v["chi2_floor"] < 1e-8 * zero, v
        assert v["slices"] == want
    x[p.names.index("shift")] = -60.0
    for v in floor_chi2(p, x).values():
        assert v["chi2_floor"] > 1e-6 * zero, v
