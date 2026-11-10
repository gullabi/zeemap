"""V2k: mu-grid libraries (LIB-8; DESIGN.md section 5d), on synthetic libraries with 3 mu values."""

from __future__ import annotations

import numpy as np
import pytest

import synth
from zeemap import field
from zeemap.forward import Forward
from zeemap.library import Library, LibraryError, read_info, read_slice
from zeemap.surface import fibonacci

from test_v2_forward_model import cfg, dataset

pytestmark = pytest.mark.tier1

MUS = [0.001, 0.5, 1.0]


@pytest.fixture(scope="module")
def mu_libs(tmp_path_factory):
    flat = tmp_path_factory.mktemp("mu_flat")
    synth.write_library(flat, "a", mus=MUS)  # no mu dependence
    lin = tmp_path_factory.mktemp("mu_lin")
    synth.write_library(lin, "a", mus=MUS, mu_factor=lambda m: 0.4 + 0.6 * m)  # limb law 1, linear in mu
    return flat, lin


def test_v2k_reader_and_size_check(mu_libs, tmp_path):
    flat, _ = mu_libs
    inf = read_info(flat)
    assert inf.mus.tolist() == MUS and inf.slice_shape() == (3, 31, 9, 2, 3001)
    s = read_slice(flat, inf, 30000.0)
    assert s.shape == (3, 31, 9, 2, 3001)
    np.testing.assert_array_equal(s[0], s[2])
    bad = tmp_path / "bad"
    bad.mkdir()
    synth.write_library(bad, "a", mus=MUS)
    with open(bad / "30000.bin", "ab") as fh:
        fh.write(b"\0\0\0\0")
    with pytest.raises(LibraryError, match="NMU x NB x NP x 2 x NL = 3 x 31 x 9 x 2 x 3001"):
        read_slice(bad, read_info(bad), 30000.0)


def _spectra(libdir, ld_mode, theta):
    lib = Library(libdir, teffs=[30000.0])
    fw = Forward(lib, fibonacci(3000), field.get_model("dqo"), dataset(phases=(0.0, 0.4)), cfg(ld_mode=ld_mode))
    I, VI, bad = fw.spectra(theta)
    assert not bad[0]
    return I[0], VI[0], fw


def test_v2k_mu_grid_reproduces_the_limb_law(mu_libs, synth_lib):
    """A mu-grid library without mu dependence equals the mu = 1 library with no limb darkening; one whose records
    scale as 0.4 + 0.6 mu (linear in mu, so the interpolation in mu is exact) equals the mu = 1 library with
    limb law 1 (patches below mu = 0.001 aside: 1e-6)."""
    flat, lin = mu_libs
    th = np.array([[300.0, 50.0, 200.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]])
    I0, V0, _ = _spectra(synth_lib, 0, th)
    I1, V1, fw = _spectra(flat, 3, th)  # the configured limb law is ignored for a mu-grid library
    ok = np.isfinite(I0)
    np.testing.assert_allclose(I1[ok], I0[ok], rtol=1e-10)
    np.testing.assert_allclose(V1[ok], V0[ok], rtol=1e-10, atol=1e-15)
    IL, VL, _ = _spectra(synth_lib, 1, th)
    I2, V2, fw2 = _spectra(lin, 3, th)
    np.testing.assert_allclose(I2[ok], IL[ok], rtol=1e-6)
    np.testing.assert_allclose(V2[ok], VL[ok], rtol=1e-6, atol=1e-12)
    below = fw2.mu_below_fraction(th[0])
    assert below.shape == (2,) and np.all(below >= 0) and np.all(below < 1e-5)
    assert fw2.node_weights(th[0]).shape == (2, 31, 17)  # the histograms sum over mu


def test_v2k_mu_grid_needs_the_linear_lookup(mu_libs):
    flat, _ = mu_libs
    with pytest.raises(ValueError, match="mu-grid library"):
        Forward(Library(flat, teffs=[30000.0]), fibonacci(200), field.get_model("dqo"), dataset(),
                cfg(interp_b="cubic"))
