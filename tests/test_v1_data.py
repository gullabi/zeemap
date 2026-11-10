"""V1i, V1j: errors, fit windows and excluded ranges of the data reader (DATA-2, DATA-3, DATA-4)."""

import json

import numpy as np
import pytest

from zeemap.data import load_data
from zeemap.types import Grid

pytestmark = pytest.mark.tier1

GRID = {"start": 5000.0, "step": 2.5, "npix": 10}
LAM = 5000.0 + 2.5 * np.arange(10)
I_OBS = np.array([1.0, 2.0, 0.5, 4.0, 1.0, 2.0, 1.0, 1.0, 5.0, 2.0])
VI_OBS = np.array([0.0, 0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2, 0.01, -0.5])
IERR = np.array([0.1, 0.2, 1e-8, 0.4, 0.05, 0.02, 0.1, 1e-9, 0.5, 0.2])
VIERR = np.array([0.01, 2e-6, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])


def write_phase(d, name, I=I_OBS, VI=VI_OBS, ierr=IERR, vierr=VIERR, lam=LAM):
    np.savetxt(d / f"{name}.dat", np.column_stack([lam, I, VI]), header="lambda I V/I")
    np.savetxt(d / f"{name}_ierr.dat", np.column_stack([lam, ierr]))
    if vierr is not None:
        np.savetxt(d / f"{name}_vierr.dat", np.column_stack([lam, vierr]))


def make_cfg(d, errors=None, windows=None, exclude=None, phases=None, weight=None):
    if phases is None:
        phases = [{"phase": 0.25, "file": "ph01.dat", "ierr": "ph01_ierr.dat", "vierr": "ph01_vierr.dat"}]
    data = {"path": str(d), "grid": dict(GRID), "windows": windows or [[5000.0, 5022.5]], "phase": phases}
    if errors is not None:
        data["errors"] = errors
    if exclude is not None:
        data["exclude"] = exclude
    if weight is not None:
        data["weight"] = weight
    return {"data": data}


def spec(ds, kind, phase_index=0):
    (s,) = [s for s in ds.spectra if s.kind == kind and s.phase_index == phase_index]
    return s


# ---------------------------------------------------------------- V1i: errors


def test_v1i_absolute_errors_with_floors(tmp_path):
    write_phase(tmp_path, "ph01")
    ds = load_data(make_cfg(tmp_path, errors={"absolute": True, "floor_I": 1e-6, "floor_VI": 1e-5, "derive_VI": False}))
    want_I = np.array([0.1, 0.2, 1e-6, 0.4, 0.05, 0.02, 0.1, 1e-6, 0.5, 0.2])
    want_VI = np.array([0.01, 1e-5, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    np.testing.assert_allclose(ds.sig_I[0], want_I, rtol=1e-12)
    np.testing.assert_allclose(ds.sig_VI[0], want_VI, rtol=1e-12)
    np.testing.assert_allclose(spec(ds, "I").sigma, want_I, rtol=1e-12)
    np.testing.assert_allclose(ds.I[0], I_OBS)
    np.testing.assert_allclose(ds.VI[0], VI_OBS)


def test_v1i_relative_errors_with_floor(tmp_path):
    write_phase(tmp_path, "ph01")
    ds = load_data(make_cfg(tmp_path, errors={"absolute": False, "floor_I": 1e-3, "floor_VI": 1e-5}))
    # sigma = err * I: [0.1, 0.4, 5e-9, 1.6, 0.05, 0.04, 0.1, 1e-9, 2.5, 0.4], then the floor 1e-3
    want = np.array([0.1, 0.4, 1e-3, 1.6, 0.05, 0.04, 0.1, 1e-3, 2.5, 0.4])
    np.testing.assert_allclose(ds.sig_I[0], want, rtol=1e-12)


def test_v1i_derived_vi_error_formula(tmp_path):
    write_phase(tmp_path, "ph01")
    ds = load_data(make_cfg(tmp_path, errors={"absolute": True, "floor_I": 1e-6, "floor_VI": 0.02, "derive_VI": True}))
    sI = np.array([0.1, 0.2, 1e-6, 0.4, 0.05, 0.02, 0.1, 1e-6, 0.5, 0.2])  # after the I floor
    rel = sI / I_OBS
    want = np.maximum(rel * np.sqrt(1.0 + VI_OBS**2), 0.02)
    # by hand for two pixels: pixel 1: 0.1 * sqrt(1.01); pixel 2: floor (2e-6 * sqrt(1.04) < 0.02)
    assert want[1] == pytest.approx(0.1 * np.sqrt(1.01), rel=1e-12)
    assert want[2] == 0.02
    np.testing.assert_allclose(ds.sig_VI[0], want, rtol=1e-12)


def test_v1i_missing_vi_error_file_derives(tmp_path):
    write_phase(tmp_path, "ph01", vierr=None)
    phases = [{"phase": 0.25, "file": "ph01.dat", "ierr": "ph01_ierr.dat", "vierr": None}]
    ds = load_data(make_cfg(tmp_path, phases=phases))
    sI = np.maximum(IERR, 1e-6)
    np.testing.assert_allclose(ds.sig_VI[0], np.maximum(sI / I_OBS * np.sqrt(1 + VI_OBS**2), 1e-5), rtol=1e-12)


def test_v1i_dataset_json_copied_to_notes(tmp_path):
    write_phase(tmp_path, "ph01")
    (tmp_path / "dataset.json").write_text(json.dumps({"origin": "synthetic", "phases": 1}))
    ds = load_data(make_cfg(tmp_path))
    assert ds.notes == {"origin": "synthetic", "phases": 1}


# ---------------------------------------------------------------- V1j: windows and excluded ranges


def test_v1j_window_edge_rule(tmp_path):
    """Window 5001-5016: first pixel at or above 5001 is 5002.5 (index 1), last at or below 5016 is 5015 (index 6)."""
    write_phase(tmp_path, "ph01")
    ds = load_data(make_cfg(tmp_path, windows=[[5001.0, 5016.0]]))
    assert spec(ds, "I").pixels.tolist() == [1, 2, 3, 4, 5, 6]
    assert spec(ds, "VI").pixels.tolist() == [1, 2, 3, 4, 5, 6]
    # exact pixel edges are inclusive
    ds = load_data(make_cfg(tmp_path, windows=[[5002.5, 5015.0]]))
    assert spec(ds, "I").pixels.tolist() == [1, 2, 3, 4, 5, 6]


def test_v1j_two_windows_with_weights_and_index(tmp_path):
    write_phase(tmp_path, "ph01")
    ds = load_data(make_cfg(tmp_path, windows=[[5000.0, 5005.0, 2.0], [5014.0, 5021.0]]))
    s = spec(ds, "I")
    assert s.pixels.tolist() == [0, 1, 2, 6, 7, 8]
    assert s.window.tolist() == [0, 0, 0, 1, 1, 1]
    assert s.weight.tolist() == [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    np.testing.assert_array_equal(s.data, I_OBS[[0, 1, 2, 6, 7, 8]])


def test_v1j_exclusions_both_and_vi_only(tmp_path):
    write_phase(tmp_path, "ph01")
    ds = load_data(
        make_cfg(
            tmp_path,
            windows=[[5000.0, 5022.5]],
            exclude=[[5005.0, 5007.5], [5016.0, 5020.0, "VI"], [5021.0, 5030.0, "I"]],
        )
    )
    assert spec(ds, "I").pixels.tolist() == [0, 1, 4, 5, 6, 7, 8]  # 2, 3 and 9 excluded
    assert spec(ds, "VI").pixels.tolist() == [0, 1, 4, 5, 6, 9]  # 2, 3 and 7, 8 excluded
    # full arrays keep every pixel
    assert np.isfinite(ds.I[0]).all() and np.isfinite(ds.VI[0]).all()


def test_v1j_extra_weights_multiply(tmp_path):
    write_phase(tmp_path, "ph01")
    write_phase(tmp_path, "ph02")
    phases = [
        {"phase": 0.1, "file": "ph01.dat", "ierr": "ph01_ierr.dat", "vierr": "ph01_vierr.dat"},
        {"phase": 0.6, "file": "ph02.dat", "ierr": "ph02_ierr.dat", "vierr": "ph02_vierr.dat", "i_offset": False},
    ]
    weight = [
        {"stokes": "VI", "lo": 5005.0, "hi": 5010.0, "phases": [0.6], "w": 3.0},
        {"stokes": "VI", "lo": 5010.0, "hi": 5012.5, "phases": "all", "w": 0.5},
    ]
    ds = load_data(make_cfg(tmp_path, windows=[[5000.0, 5012.5, 2.0]], phases=phases, weight=weight))
    assert spec(ds, "I", 0).weight.tolist() == [2.0] * 6
    assert spec(ds, "I", 1).weight.tolist() == [2.0] * 6
    assert spec(ds, "VI", 0).weight.tolist() == [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    assert spec(ds, "VI", 1).weight.tolist() == [2.0, 2.0, 6.0, 6.0, 3.0, 1.0]
    assert spec(ds, "I", 0).i_offset is True and spec(ds, "I", 1).i_offset is False
    assert spec(ds, "VI", 0).i_offset is False


def test_v1j_per_phase_switches(tmp_path):
    write_phase(tmp_path, "ph01")
    write_phase(tmp_path, "ph02")
    phases = [
        {"phase": 0.7, "file": "ph01.dat", "ierr": "ph01_ierr.dat", "vierr": "ph01_vierr.dat", "use_VI": False},
        {"phase": 0.2, "file": "ph02.dat", "ierr": "ph02_ierr.dat", "vierr": "ph02_vierr.dat", "use_I": False},
    ]
    ds = load_data(make_cfg(tmp_path, phases=phases))
    assert ds.phases.tolist() == [0.7, 0.2]  # as given, not reordered
    assert [(s.phase_index, s.kind) for s in ds.spectra] == [(0, "I"), (1, "VI")]
    assert np.isnan(ds.VI[0]).all() and np.isnan(ds.sig_VI[0]).all()
    assert np.isnan(ds.I[1]).all() and np.isnan(ds.sig_I[1]).all()
    assert ds.grid == Grid(5000.0, 2.5, 10)
