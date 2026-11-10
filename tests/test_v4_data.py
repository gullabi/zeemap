"""V4b: the data reader stops with a message naming both ranges (DATA-6)."""

import numpy as np
import pytest

from zeemap.data import DataError, load_data

pytestmark = pytest.mark.tier1

START, STEP, NPIX = 5000.0, 2.5, 20


def write(d, lam, n=None):
    n = len(lam) if n is None else n
    lam = np.asarray(lam)[:n]
    np.savetxt(d / "ph01.dat", np.column_stack([lam, np.ones(n), np.zeros(n)]))
    np.savetxt(d / "ph01_ierr.dat", np.column_stack([lam, 0.01 * np.ones(n)]))
    np.savetxt(d / "ph01_vierr.dat", np.column_stack([lam, 0.001 * np.ones(n)]))


def cfg(d):
    return {
        "data": {
            "path": str(d),
            "grid": {"start": START, "step": STEP, "npix": NPIX},
            "windows": [[START, START + STEP * (NPIX - 1)]],
            "phase": [{"phase": 0.0, "file": "ph01.dat", "ierr": "ph01_ierr.dat", "vierr": "ph01_vierr.dat"}],
        },
    }


GRID_LAM = START + STEP * np.arange(NPIX)


def test_v4b_offset_within_tolerance_passes(tmp_path):
    """The tolerance is 1e-3 of a step (DATA-6, FORMATS.md section 2): 5e-4 and 9e-4 of a step pass."""
    for frac in (5e-4, -9e-4):
        write(tmp_path, GRID_LAM + frac * STEP)
        ds = load_data(cfg(tmp_path))
        assert ds.grid.npix == NPIX


def test_v4b_offset_of_one_hundredth_step_fails(tmp_path):
    """0.01 of a step is ten times the tolerance of DATA-6, so it stops (an error, not a pass)."""
    write(tmp_path, GRID_LAM + 0.01 * STEP)
    with pytest.raises(DataError) as e:
        load_data(cfg(tmp_path))
    assert "5000.0250-5047.5250" in str(e.value) and "5000.0000-5047.5000" in str(e.value)


def test_v4b_half_step_offset_fails_naming_both_ranges(tmp_path):
    write(tmp_path, GRID_LAM + 0.5 * STEP)
    with pytest.raises(DataError) as e:
        load_data(cfg(tmp_path))
    msg = str(e.value)
    assert "5001.2500-5048.7500" in msg  # the file's range
    assert "5000.0000-5047.5000" in msg  # the grid's range


def test_v4b_irregular_grid_fails(tmp_path):
    lam = GRID_LAM.copy()
    lam[7] += 0.3 * STEP
    write(tmp_path, lam)
    with pytest.raises(DataError, match="5000.0000-5047.5000"):
        load_data(cfg(tmp_path))


def test_v4b_wrong_row_count_fails(tmp_path):
    write(tmp_path, GRID_LAM, n=NPIX - 1)
    with pytest.raises(DataError) as e:
        load_data(cfg(tmp_path))
    msg = str(e.value)
    assert "19 rows" in msg and "20 pixels" in msg
    assert "5000.0000-5045.0000" in msg and "5000.0000-5047.5000" in msg


def test_v4b_error_file_checked_too(tmp_path):
    write(tmp_path, GRID_LAM)
    np.savetxt(tmp_path / "ph01_ierr.dat", np.column_stack([GRID_LAM + 0.5 * STEP, 0.01 * np.ones(NPIX)]))
    with pytest.raises(DataError, match="ph01_ierr.dat"):
        load_data(cfg(tmp_path))


def test_v4b_file_not_overlapping_the_grid_fails(tmp_path):
    write(tmp_path, 7000.0 + STEP * np.arange(NPIX))
    with pytest.raises(DataError) as e:
        load_data(cfg(tmp_path))
    msg = str(e.value)
    assert "do not overlap" in msg
    assert "7000.0000-7047.5000" in msg and "5000.0000-5047.5000" in msg
