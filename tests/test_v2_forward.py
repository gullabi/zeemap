"""V2: known answers of the whole forward model (synthetic library). Here: V2e, the library format."""

import numpy as np
import pytest

import synth
from zeemap.library import LibraryError, read_info, read_slice, write_info, write_slice

pytestmark = pytest.mark.tier1


def test_v2e_library_written_and_read_back_unchanged(synth_lib):
    info = read_info(synth_lib)
    ref = synth.info("a")
    assert info.name == ref.name
    np.testing.assert_array_equal(info.wavelengths, ref.wavelengths)
    np.testing.assert_array_equal(info.teffs, ref.teffs)
    np.testing.assert_array_equal(info.bs, ref.bs)
    np.testing.assert_array_equal(info.psis, ref.psis)
    assert info.mus.tolist() == [1.0]
    assert info.slice_shape() == (31, 9, 2, 3001)
    for teff in info.teffs:
        got = read_slice(synth_lib, info, teff)
        want = synth.make_slice(teff, "a", info)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, want)  # float32 written, float32 read: byte for byte here


def test_v2e_synthetic_formulas_by_hand(synth_lib):
    """Line 2 does not move with B; V vanishes at psi = 90; V scales with cos psi; I scales with Teff."""
    info = read_info(synth_lib)
    s30 = read_slice(synth_lib, info, 30000.0)
    s60 = read_slice(synth_lib, info, 50000.0)
    lam = info.wavelengths
    i5500 = np.flatnonzero(lam == 5500.0)[0]
    # line 2 centre at 5500 A for every B: depth 0.3 there (other lines are >= 500 A away)
    assert np.allclose(s30[:, 0, 0, i5500], 1.0 - 0.3, atol=1e-6)
    assert np.all(s30[:, -1, 1] == 0.0)  # psi = 90: V exactly 0 (synth.cos_psi)
    np.testing.assert_allclose(s30[:, 4, 1], s30[:, 0, 1] * np.float32(np.cos(np.radians(45.0))), rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(s60[:, :, 0], s30[:, :, 0] * np.float32(50000.0 / 30000.0), rtol=2e-6)
    # V = 0.01 dI/dlambda: check against a central difference of the stored I at one node
    b = 12
    di = (s30[b, 0, 0, 2:] - s30[b, 0, 0, :-2]) / 2.0
    np.testing.assert_allclose(s30[b, 0, 1, 1:-1], 0.01 * di, atol=3e-5)  # finite-difference error ~ 1e-4 of 0.3/4^2


def test_v2e_wrong_size_refused(tmp_path, synth_lib):
    info = read_info(synth_lib)
    write_info(tmp_path, info)
    data = synth.make_slice(20000.0, "a", info)
    p = write_slice(tmp_path, info, 20000.0, data)
    with open(p, "ab") as f:
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(LibraryError, match="bytes"):
        read_slice(tmp_path, info, 20000.0)
    p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises(LibraryError, match="bytes"):
        read_slice(tmp_path, info, 20000.0)


def test_v2e_more_than_one_mu_refused(tmp_path, synth_lib):
    text = (synth_lib / "info.dat").read_text().replace("\n1 1\n", "\n2 1 0.5\n")
    assert "2 1 0.5" in text
    (tmp_path / "info.dat").write_text(text)
    with pytest.raises(LibraryError, match="mu"):
        read_info(tmp_path)


def test_v2e_index_named_slice_files_are_found(tmp_path, synth_lib):
    """Older libraries name the slices 0.bin, 1.bin, ... in Teff order."""
    info = read_info(synth_lib)
    write_info(tmp_path, info)
    data = synth.make_slice(40000.0, "a", info)
    np.ascontiguousarray(data, dtype="<f4").tofile(tmp_path / "2.bin")
    np.testing.assert_array_equal(read_slice(tmp_path, info, 40000.0), data)
    with pytest.raises(LibraryError, match="no slice file"):
        read_slice(tmp_path, info, 20000.0)
