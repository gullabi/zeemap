"""V2c (Teff interpolation) and V2f (each slice read once) of the library (LIB-2, LIB-3)."""

import numpy as np
import pytest

from zeemap.library import Library, LibraryError

pytestmark = pytest.mark.tier1


def test_v2c_at_library_teff_returns_slice_exactly(synth_lib):
    lib = Library(synth_lib)
    for teff in lib.teffs:
        got = lib.at_teff(teff)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, lib.slice(teff).astype(np.float64))


def test_v2c_halfway_in_log_teff_gives_mean(synth_lib):
    lib = Library(synth_lib)
    mean = 0.5 * (lib.slice(30000.0).astype(np.float64) + lib.slice(40000.0).astype(np.float64))
    got = lib.at_teff(np.sqrt(30000.0 * 40000.0))
    np.testing.assert_allclose(got, mean, rtol=0, atol=1e-12)
    # the arithmetic midpoint is not halfway in log Teff
    assert not np.allclose(lib.at_teff(35000.0), mean, rtol=0, atol=1e-6)


@pytest.mark.parametrize("teff", [20000.0, 23456.7, 30000.0, 34641.0, 47000.0, 50000.0])
def test_v2c_interp_weights_consistent_with_at_teff(synth_lib, teff):
    lib = Library(synth_lib)
    (i0, w0), (i1, w1) = lib.interp_weights(teff)
    assert w0 + w1 == pytest.approx(1.0, abs=1e-15)
    assert 0.0 <= w0 <= 1.0 and 0.0 <= w1 <= 1.0
    assert lib.teffs[i0] - 0.5 <= teff <= lib.teffs[i1] + 0.5
    comb = w0 * lib.slice(lib.teffs[i0]).astype(np.float64) + w1 * lib.slice(lib.teffs[i1]).astype(np.float64)
    np.testing.assert_allclose(lib.at_teff(teff), comb, rtol=0, atol=1e-12)
    if lib.teffs[i0] != lib.teffs[i1]:
        t0, t1 = lib.teffs[i0], lib.teffs[i1]
        assert w1 == pytest.approx(np.log(teff / t0) / np.log(t1 / t0), rel=1e-12)


def test_v2c_subset_interpolates_across_gap(synth_lib):
    """With 20000 and 40000 selected, 30000 is interpolated from those two, not taken from the file."""
    lib = Library(synth_lib, teffs=[20000.0, 40000.0])
    (i0, w0), (i1, w1) = lib.interp_weights(30000.0)
    assert (lib.teffs[i0], lib.teffs[i1]) == (20000.0, 40000.0)
    assert w1 == pytest.approx(np.log(1.5) / np.log(2.0))
    with pytest.raises(LibraryError, match="not a selected"):
        lib.slice(30000.0)


def test_v2f_each_slice_read_once(synth_lib):
    lib = Library(synth_lib, teffs=[30000.0, 40000.0])
    assert lib.reads == 0  # lazy
    for _ in range(5):
        lib.slice(30000.0)
        lib.at_teff(35000.0)
        lib.at_teff(36000.0)
        lib.slice(40000.0)
        lib.interp_weights(31000.0)
    assert lib.reads == 2
    assert lib.slice(30000.0) is lib.slice(30000.0)  # the cached array itself


def test_v2f_reads_count_distinct_slices_touched(synth_lib):
    lib = Library(synth_lib)
    for _ in range(3):
        lib.at_teff(30000.0)  # exact: only one slice
    assert lib.reads == 1
    for _ in range(3):
        lib.at_teff(25000.0)  # 20000 and 30000
    assert lib.reads == 2
    lib.slice(50000.0)
    lib.slice(50000.0)
    assert lib.reads == 3


def test_v2f_cached_slice_read_only(synth_lib):
    s = Library(synth_lib).slice(20000.0)
    with pytest.raises(ValueError):
        s[0, 0, 0, 0] = 1.0


def test_v2f_selection_tolerance_and_non_library_teff(synth_lib):
    lib = Library(synth_lib, teffs=[30000.4, 40000.0])
    assert lib.teffs.tolist() == [30000.0, 40000.0]
    with pytest.raises(LibraryError) as e:
        Library(synth_lib, teffs=[30000.0, 35000.0])
    msg = str(e.value)
    assert "35000" in msg and "20000" in msg and "50000" in msg
    with pytest.raises(LibraryError):
        Library(synth_lib, teffs=[30000.6])
