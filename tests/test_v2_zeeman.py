"""Checks of tools/zeeman_library.py, the synthetic Zeeman toy library and observation writer (run with V2e)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import zeeman_library as zl  # noqa: E402

from zeemap.library import read_info, read_slice  # noqa: E402

pytestmark = pytest.mark.tier1

HALPHA = 6562.8


@pytest.fixture(scope="module")
def zlib(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_zeeman")
    zl.main([str(d), "--fast"])
    return d


def _min_near(lam, spec, centre, half=40.0):
    m = np.abs(lam - centre) < half
    return lam[m][np.argmin(spec[m])]


def test_v2e_zeeman_library_reads_back(zlib):
    inf = read_info(zlib)
    assert inf.nl == 3601 and inf.wavelengths[0] == 3600.0 and inf.wavelengths[-1] == 7200.0
    assert inf.teffs.tolist() == zl.TEFFS_FAST.tolist()
    assert inf.bs.tolist() == list(np.arange(1.0, 40.0, 5.0))
    assert inf.psis.tolist() == list(np.arange(0.0, 90.1, 11.25)) and inf.mus.tolist() == [1.0]
    assert (zlib / "README.txt").read_text().startswith("SYNTHETIC TOY LIBRARY")
    for t in inf.teffs:
        sl = read_slice(zlib, inf, t)
        assert np.all(np.isfinite(sl)) and np.all(sl[:, :, 0] > 0)
        assert np.all(np.abs(sl[:, :, 1]) <= sl[:, :, 0])
    # a stored record equals the formula (float32)
    i_ref, v_ref = zl.stokes(inf.wavelengths, 14000.0, inf.bs[2], zl.cos_psi(inf.psis[3]))
    sl = read_slice(zlib, inf, 14000.0)
    np.testing.assert_allclose(sl[2, 3, 0], i_ref, rtol=1e-6)
    np.testing.assert_allclose(sl[2, 3, 1], v_ref, rtol=1e-6, atol=1e-7)


def test_v2e_zeeman_halpha_sigma_shift_at_10mg():
    assert zl.linear_shift(HALPHA, 10.0) == pytest.approx(201.0, abs=2.0)
    lam = zl.WAVELENGTHS
    i0, _ = zl.stokes(lam, 12000.0, 10.0, zl.cos_psi(0.0))
    q = zl.quadratic_shift(HALPHA, 3, 10.0)
    blue = _min_near(lam, i0, HALPHA + q - 201.0)
    red = _min_near(lam, i0, HALPHA + q + 201.0)
    assert red - blue == pytest.approx(2 * 201.0, abs=4.0)
    # pattern centre is blueshifted by the quadratic term (negative, order 10 A at 10 MG)
    assert -30.0 < q < -5.0
    assert (red + blue) / 2 == pytest.approx(HALPHA + q, abs=2.0)


def test_v2e_zeeman_v_zero_at_psi_90_and_mirror_not_stored(zlib):
    inf = read_info(zlib)
    assert inf.psis[-1] == 90.0  # psi > 90 is not stored; V(180 - psi) = -V(psi) comes from the lookup mirror
    for t in inf.teffs:
        sl = read_slice(zlib, inf, t)
        assert np.all(sl[:, -1, 1] == 0.0)
        assert np.any(sl[:, 0, 1] != 0.0)
    lam = zl.WAVELENGTHS
    c = zl.cos_psi(33.75)
    i_a, v_a = zl.stokes(lam, 12000.0, 10.0, c)
    i_b, v_b = zl.stokes(lam, 12000.0, 10.0, -c)
    np.testing.assert_array_equal(i_a, i_b)
    np.testing.assert_allclose(v_a, -v_b)


def test_v2e_zeeman_no_pi_component_at_psi_0():
    lam = zl.WAVELENGTHS
    centre = HALPHA + zl.quadratic_shift(HALPHA, 3, 10.0)
    k = np.argmin(np.abs(lam - centre))
    cont = zl.continuum(lam, 12000.0)
    i0, _ = zl.stokes(lam, 12000.0, 10.0, zl.cos_psi(0.0))
    i90, _ = zl.stokes(lam, 12000.0, 10.0, zl.cos_psi(90.0))
    assert 1 - i0[k] / cont[k] < 0.02  # only far sigma wings
    assert 1 - i90[k] / cont[k] > 0.2  # pi at full strength (d/2)
    a = zl.single_line_absorption(lam, "H-alpha", 12000.0, 10.0, 0.0)
    assert a[k] < 0.02
    # weights add to one when the components overlap: an unsplit line has depth d at every psi
    for psi in (0.0, 45.0, 90.0):
        a = zl.single_line_absorption(np.array([HALPHA]), "H-alpha", 12000.0, 0.0, psi)
        assert a[0] == pytest.approx(zl.depth(0.60, 12000.0))


def test_v2e_zeeman_hbeta_equivalent_width():
    lam = np.arange(3000.0, 9000.0, 0.25)
    ew = np.trapezoid(zl.single_line_absorption(lam, "H-beta", 12000.0, 1.0, 0.0), lam)
    assert 10.0 < ew < 60.0


def test_v2e_zeeman_observation_files(tmp_path):
    truth = zl.make_observation(tmp_path, npoints=200, seed=3)
    start, step, n = zl.OBS_GRID
    expected = start + step * np.arange(n)
    assert n == 1501 and expected[-1] == 6800.0
    for name in ("ph00.dat", "ph00_ierr.dat", "ph00_vierr.dat"):
        a = np.loadtxt(tmp_path / name)
        assert a.shape[0] == n
        np.testing.assert_allclose(a[:, 0], expected, rtol=0, atol=1e-3 * step)
    obs = np.loadtxt(tmp_path / "ph00.dat")
    assert np.all(np.isfinite(obs)) and np.all(obs[:, 1] > 0)
    ds = json.loads((tmp_path / "dataset.json").read_text())
    tr = json.loads((tmp_path / "truth.json").read_text())
    assert tr == ds["truth"] == json.loads(json.dumps(truth))
    assert tr["teff_K"] == 13000.0 and tr["bp_MG"] == 12.0 and tr["beta_deg"] == 40.0 and tr["noise"]["seed"] == 3
