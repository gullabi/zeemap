"""V4a: the library stops a fit before it starts, with a message naming the covered range (LIB-7).
Also the psi mirror of FORMATS.md section 1 (`Library.mirror_slice`)."""

import numpy as np
import pytest

from zeemap.library import Library, LibraryError

pytestmark = pytest.mark.tier1


def test_v4a_check_teff_outside_range(synth_lib):
    lib = Library(synth_lib)
    for teff in (19000.0, 50001.0):
        with pytest.raises(LibraryError) as e:
            lib.check_teff(teff)
        assert "20000" in str(e.value) and "50000" in str(e.value)
    for teff in (20000.0, 33333.0, 50000.0):
        lib.check_teff(teff)


def test_v4a_at_teff_outside_selected_range(synth_lib):
    lib = Library(synth_lib, teffs=[30000.0, 40000.0])
    for teff in (25000.0, 45000.0):
        with pytest.raises(LibraryError) as e:
            lib.at_teff(teff)
        assert "30000" in str(e.value) and "40000" in str(e.value)
    assert lib.reads == 0  # stopped before reading anything


def test_v4a_check_wavelengths(synth_lib):
    lib = Library(synth_lib)  # 4000 to 7000 A
    margin = 4 * 2.0
    lib.check_wavelengths(4008.0, 6992.0, margin)  # exactly at the margin: allowed
    for lo, hi in ((4005.0, 6000.0), (4500.0, 6995.0), (3900.0, 5000.0), (5000.0, 7100.0)):
        with pytest.raises(LibraryError) as e:
            lib.check_wavelengths(lo, hi, margin)
        assert "4000" in str(e.value) and "7000" in str(e.value)


def test_v4a_cos_psi_nodes(synth_lib):
    lib = Library(synth_lib)
    c = lib.cos_psi_nodes()
    npsi = len(lib.info.psis)
    assert len(c) == 2 * npsi - 1
    assert c[0] == 1.0 and c[npsi - 1] == 0.0 and c[-1] == -1.0
    np.testing.assert_array_equal(c[npsi:], -c[npsi - 2 :: -1])
    assert np.all(np.diff(c) < 0)


def test_v4a_mirror_slice_symmetry(synth_lib):
    lib = Library(synth_lib)
    s = lib.slice(30000.0).copy()
    s[:, -1, 1] = 0.123  # a stored V at psi = 90 must be ignored
    m = lib.mirror_slice(s)
    npsi = len(lib.info.psis)
    nc = 2 * npsi - 1
    assert m.shape == (s.shape[0], nc, 2, s.shape[3])
    assert np.all(m[:, npsi - 1, 1] == 0.0)
    np.testing.assert_array_equal(m[:, :npsi, 0], s[:, :, 0])
    np.testing.assert_array_equal(m[:, : npsi - 1, 1], s[:, : npsi - 1, 1])
    for k in range(nc):
        np.testing.assert_array_equal(m[:, k, 0], m[:, nc - 1 - k, 0])  # I even
        np.testing.assert_array_equal(m[:, k, 1], -m[:, nc - 1 - k, 1])  # V odd
    with pytest.raises(LibraryError, match="shape"):
        lib.mirror_slice(s[:, :-1])
