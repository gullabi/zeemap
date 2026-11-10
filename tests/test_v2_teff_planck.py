"""V2c, the "planck" Teff weights (LIB-3, decision #26; DESIGN.md section 5c).

A synthetic library whose spectra are a fixed shape times the Planck function (tests/synth.py, variant "planck") is
reproduced between its slices by the continuum-matched weights: exactly (up to the float32 records) on a data grid
equal to the library grid with a negligible convolution, and at FWHM 7 A to 1e-5 in I and 1e-4 of max |V| in V (the
weight varies across the kernel, and V is steep in the lines); "log" weights are off by more.
"""

from __future__ import annotations

import numpy as np
import pytest

import synth
from zeemap import field
from zeemap.forward import Forward
from zeemap.library import Library
from zeemap.surface import fibonacci
from zeemap.types import DataSet, Grid, Spectrum

from test_v2_forward_model import cfg, pole_on  # noqa: F401  (registers the field model "toy_const")

pytestmark = pytest.mark.tier1

TEFFS_BETWEEN = [22000.0, 25000.0, 28000.0, 33000.0, 45000.0]


@pytest.fixture(scope="module")
def planck_libs(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_planck")
    synth.write_library(d, "planck")
    e = tmp_path_factory.mktemp("synth_planck_exact")
    synth.write_library(e, "planck", teffs=TEFFS_BETWEEN)  # the exact slices at the in-between temperatures
    return d, e


def _dataset(step):
    grid = Grid(4100.0, step, int(round(2800.0 / step)) + 1)
    M = grid.npix
    sp = [Spectrum(0, 0.0, "VI", np.arange(M), np.zeros(M), np.ones(M), np.ones(M), np.zeros(M, int))]
    nan = np.full((1, M), np.nan)
    return DataSet(grid, np.array([0.0]), sp, nan, nan.copy(), nan.copy(), nan.copy())


def _spectra(libdir, teffs, teff, fwhm, step, mode):
    lib = Library(libdir, teffs=teffs)
    fw = Forward(lib, fibonacci(500), field.get_model("toy_const"), _dataset(step),
                 cfg(ld_mode=0, fwhm=fwhm, interp_teff=mode))
    I, VI, bad = fw.spectra(pole_on(300.0, 0.6), teff=None if teffs is not None and len(teffs) == 1 else [teff])
    assert not bad[0]
    return I[0, 0], I[0, 0] * VI[0, 0]


@pytest.mark.parametrize("fwhm,step,tol,tol_v", [(0.05, 1.0, 1e-6, 1e-6), (7.0, 2.5, 1e-5, 1e-4)])
def test_v2c_planck_weights_reproduce_a_planck_library(planck_libs, fwhm, step, tol, tol_v):
    grid_lib, exact_lib = planck_libs
    worst_log = 0.0
    for t in TEFFS_BETWEEN:
        I_e, V_e = _spectra(exact_lib, [t], t, fwhm, step, "planck")
        I_p, V_p = _spectra(grid_lib, None, t, fwhm, step, "planck")
        np.testing.assert_allclose(I_p, I_e, rtol=tol, atol=0)
        np.testing.assert_allclose(V_p, V_e, rtol=0, atol=tol_v * np.max(np.abs(V_e)))
        I_l, _ = _spectra(grid_lib, None, t, fwhm, step, "log")
        worst_log = max(worst_log, float(np.max(np.abs(I_l / I_e - 1.0))))
    assert worst_log > 1e-3  # the "log" rule is not exact for such a library


def test_v2c_planck_weights_at_a_slice_return_the_slice(synth_lib):
    for t in (20000.0, 30000.0, 50000.0):
        I_p, V_p = _spectra(synth_lib, None, t, 3.0, 2.5, "planck")
        I_l, V_l = _spectra(synth_lib, None, t, 3.0, 2.5, "log")
        np.testing.assert_array_equal(I_p, I_l)
        np.testing.assert_array_equal(V_p, V_l)
