"""Observation reader, errors, windows, excluded ranges and weights.

Requirements DATA-1 to DATA-4 and DATA-6 (docs/SPEC.md section 6.5); file layout docs/FORMATS.md sections 2 and 3.
Interfaces: docs/DEV_INTERFACES.md, section "Data".

`load_data(cfg)` takes the nested configuration dict with paths already resolved (config.load_config); missing
optional keys take the defaults of FORMATS.md section 3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from zeemap.types import DataSet, Grid, Spectrum

GRID_TOL = 1e-3  # wavelength agreement with the configured grid, in steps (DATA-6)
EDGE_EPS = 1e-9  # pixel edge rules ("at or above", "at or below"), in steps, against rounding only

ERROR_DEFAULTS = {"absolute": True, "floor_I": 1e-6, "floor_VI": 1e-5, "derive_VI": False}


class DataError(ValueError):
    """Wrong or inconsistent observation input; the message names the file or key and the ranges involved."""


# --------------------------------------------------------------------------------------------------------------------
# pixel rules


def first_at_or_above(grid: Grid, lam: float) -> int:
    """Index of the first grid pixel with wavelength >= lam (may be npix if none)."""
    k = math.ceil((lam - grid.start) / grid.step - EDGE_EPS)
    return max(k, 0)


def last_at_or_below(grid: Grid, lam: float) -> int:
    """Index of the last grid pixel with wavelength <= lam (may be -1 if none)."""
    k = math.floor((lam - grid.start) / grid.step + EDGE_EPS)
    return min(k, grid.npix - 1)


def range_mask(grid: Grid, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of the pixels with lo <= wavelength <= hi (inclusive both ends)."""
    m = np.zeros(grid.npix, dtype=bool)
    a, b = first_at_or_above(grid, lo), last_at_or_below(grid, hi)
    if a <= b:
        m[a : b + 1] = True
    return m


# --------------------------------------------------------------------------------------------------------------------
# files


def _fmt_range(lo: float, hi: float) -> str:
    return f"{lo:.4f}-{hi:.4f} A"


def read_columns(path: Path, ncol: int, grid: Grid) -> np.ndarray:
    """Read a white-space separated file with `ncol` columns (`#` comments), check its wavelengths against the grid.

    Returns the array (npix, ncol). Raises DataError naming the file's and the grid's ranges (DATA-6).
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"data file {path} does not exist")
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2, dtype=float)
    except ValueError as exc:
        raise DataError(f"data file {path}: cannot read numbers ({exc})") from exc
    if arr.size == 0:
        raise DataError(f"data file {path} has no rows; the grid {_fmt_range(grid.start, grid.end)} has {grid.npix}")
    if arr.shape[1] < ncol:
        raise DataError(f"data file {path}: {arr.shape[1]} columns, expected {ncol}")
    arr = arr[:, :ncol]
    lam = arr[:, 0]
    frange = _fmt_range(float(lam.min()), float(lam.max()))
    grange = _fmt_range(grid.start, grid.end)
    if lam.max() < grid.start - GRID_TOL * grid.step or lam.min() > grid.end + GRID_TOL * grid.step:
        raise DataError(
            f"data file {path}: its wavelengths {frange} do not overlap the configured grid {grange} "
            f"(step {grid.step} A, {grid.npix} pixels)"
        )
    if len(lam) != grid.npix:
        raise DataError(
            f"data file {path}: {len(lam)} rows covering {frange}, but the configured grid {grange} "
            f"(step {grid.step} A) has {grid.npix} pixels"
        )
    dev = np.abs(lam - grid.wavelengths) / grid.step
    if np.max(dev) > GRID_TOL:
        k = int(np.argmax(dev))
        raise DataError(
            f"data file {path}: its wavelengths {frange} do not agree with the configured grid {grange} "
            f"(step {grid.step} A) to {GRID_TOL} of a step; largest deviation {dev[k]:.4g} steps at row {k + 1}"
        )
    return arr


def _resolve(base: Path, name) -> Path:
    p = Path(name)
    return p if p.is_absolute() else base / p


# --------------------------------------------------------------------------------------------------------------------
# errors (DATA-2)


def errors_I(I: np.ndarray, err: np.ndarray, absolute: bool, floor: float) -> np.ndarray:
    """sigma_I = err (absolute) or err * |I| (relative), then max(sigma, floor)."""
    sig = np.asarray(err, float) if absolute else np.asarray(err, float) * np.abs(I)
    return np.maximum(sig, floor)


def derived_errors_VI(I: np.ndarray, VI: np.ndarray, sig_I: np.ndarray, floor: float) -> np.ndarray:
    """sigma_VI = (sigma_I / |I|) * sqrt(1 + (V/I)^2), sigma_I after its floor; then max(sigma_VI, floor)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = sig_I / np.abs(I)
    return np.maximum(rel * np.sqrt(1.0 + VI**2), floor)


# --------------------------------------------------------------------------------------------------------------------
# selection (DATA-3, DATA-4)


def _windows(dcfg: dict) -> list[tuple[float, float, float]]:
    raw = dcfg.get("windows")
    if not raw:
        raise DataError("data.windows: no fit window configured (list of [lo, hi] or [lo, hi, weight] in A)")
    out = []
    for i, w in enumerate(raw):
        if len(w) not in (2, 3):
            raise DataError(f"data.windows[{i}]: expected [lo, hi] or [lo, hi, weight], got {w!r}")
        lo, hi = float(w[0]), float(w[1])
        if hi < lo:
            raise DataError(f"data.windows[{i}]: lo {lo} above hi {hi}")
        out.append((lo, hi, float(w[2]) if len(w) == 3 else 1.0))
    return out


def window_index(grid: Grid, windows) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: window index (-1 outside every window) and window weight. A pixel in two windows goes to the first."""
    idx = np.full(grid.npix, -1, dtype=int)
    wt = np.ones(grid.npix)
    for k, (lo, hi, w) in enumerate(windows):
        a, b = first_at_or_above(grid, lo), last_at_or_below(grid, hi)
        if a > b:
            continue
        sel = np.zeros(grid.npix, dtype=bool)
        sel[a : b + 1] = True
        sel &= idx < 0
        idx[sel] = k
        wt[sel] = w
    return idx, wt


def exclusion_masks(grid: Grid, exclude) -> dict[str, np.ndarray]:
    """Boolean masks of excluded pixels for "I" and "VI" ([lo, hi] inclusive; third entry limits to one quantity)."""
    out = {"I": np.zeros(grid.npix, dtype=bool), "VI": np.zeros(grid.npix, dtype=bool)}
    for i, e in enumerate(exclude or []):
        if len(e) not in (2, 3):
            raise DataError(f"data.exclude[{i}]: expected [lo, hi] or [lo, hi, \"I\"|\"VI\"], got {e!r}")
        kinds = ("I", "VI")
        if len(e) == 3:
            if e[2] not in kinds:
                raise DataError(f"data.exclude[{i}]: third entry must be \"I\" or \"VI\", got {e[2]!r}")
            kinds = (e[2],)
        m = range_mask(grid, float(e[0]), float(e[1]))
        for k in kinds:
            out[k] |= m
    return out


def _phase_matches(sel, phase: float, index: int) -> bool:
    if sel is None or sel == "all":
        return True
    return any(abs(float(p) - phase) <= 1e-9 for p in sel)


def extra_weights(grid: Grid, entries, kind: str, phase: float, index: int) -> np.ndarray:
    """Product of the [[data.weight]] factors that apply to this Stokes quantity and phase (default 1)."""
    w = np.ones(grid.npix)
    for i, e in enumerate(entries or []):
        stokes = e.get("stokes")
        if stokes not in ("I", "VI"):
            raise DataError(f"data.weight[{i}].stokes must be \"I\" or \"VI\", got {stokes!r}")
        if stokes != kind or not _phase_matches(e.get("phases", "all"), phase, index):
            continue
        w[range_mask(grid, float(e["lo"]), float(e["hi"]))] *= float(e["w"])
    return w


# --------------------------------------------------------------------------------------------------------------------
# the reader


def load_data(cfg: dict) -> DataSet:
    """Read the observations of cfg["data"] into a DataSet (DATA-1 to DATA-4, DATA-6)."""
    dcfg = cfg.get("data") or {}
    if "path" not in dcfg:
        raise DataError("data.path: not configured")
    base = Path(dcfg["path"])
    g = dcfg.get("grid")
    if not g or any(k not in g for k in ("start", "step", "npix")):
        raise DataError("data.grid: needs start, step and npix")
    grid = Grid(float(g["start"]), float(g["step"]), int(g["npix"]))
    if grid.step <= 0 or grid.npix < 1:
        raise DataError(f"data.grid: step must be > 0 and npix >= 1, got step {grid.step}, npix {grid.npix}")
    errs = {**ERROR_DEFAULTS, **(dcfg.get("errors") or {})}
    vi_scale = float(dcfg.get("vi_scale", 1.0))
    windows = _windows(dcfg)
    excl = exclusion_masks(grid, dcfg.get("exclude", []))
    phase_cfg = dcfg.get("phase") or []
    if not phase_cfg:
        raise DataError("data.phase: no phase configured")

    win_idx, win_wt = window_index(grid, windows)
    in_window = win_idx >= 0
    P, M = len(phase_cfg), grid.npix
    I = np.full((P, M), np.nan)
    VI = np.full((P, M), np.nan)
    sI = np.full((P, M), np.nan)
    sVI = np.full((P, M), np.nan)
    phases = np.empty(P)
    spectra: list[Spectrum] = []

    for p, pc in enumerate(phase_cfg):
        for key in ("phase", "file"):
            if key not in pc:
                raise DataError(f"data.phase[{p}].{key}: missing")
        phases[p] = float(pc["phase"])
        use_I = bool(pc.get("use_I", True))
        use_VI = bool(pc.get("use_VI", True))
        i_off = bool(pc.get("i_offset", True))
        obs = read_columns(_resolve(base, pc["file"]), 3, grid)
        if pc.get("ierr") is None:
            raise DataError(f"data.phase[{p}].ierr: missing (the error file of I)")
        ierr = read_columns(_resolve(base, pc["ierr"]), 2, grid)[:, 1]
        Ip, VIp = obs[:, 1], obs[:, 2]
        sIp = errors_I(Ip, ierr, bool(errs["absolute"]), float(errs["floor_I"]))
        if bool(errs["derive_VI"]) or not pc.get("vierr"):
            sVIp = derived_errors_VI(Ip, VIp, sIp, float(errs["floor_VI"]))
        else:  # V/I errors are read as given (V/I is already a fraction), then floored
            vierr = read_columns(_resolve(base, pc["vierr"]), 2, grid)[:, 1]
            sVIp = np.maximum(vierr, float(errs["floor_VI"]))
        if vi_scale != 1.0:  # data.vi_scale: a calibration factor on the measured V/I and its error (decision #21)
            VIp = VIp * vi_scale
            sVIp = sVIp * abs(vi_scale)
        for kind, on, d, s, full_d, full_s in (
            ("I", use_I, Ip, sIp, I, sI),
            ("VI", use_VI, VIp, sVIp, VI, sVI),
        ):
            if not on:
                continue
            full_d[p], full_s[p] = d, s
            pix = np.flatnonzero(in_window & ~excl[kind])
            w = (win_wt * extra_weights(grid, dcfg.get("weight"), kind, phases[p], p))[pix]
            keep = w > 0.0  # DATA-4: a weight of 0 removes the pixel (it then counts in no n_s)
            pix, w = pix[keep], w[keep]
            spectra.append(
                Spectrum(
                    phase_index=p,
                    phase=float(phases[p]),
                    kind=kind,
                    pixels=pix,
                    data=d[pix].copy(),
                    sigma=s[pix].copy(),
                    weight=w,
                    window=win_idx[pix].copy(),
                    i_offset=i_off if kind == "I" else False,
                )
            )

    notes = {}
    jf = base / "dataset.json"
    if jf.is_file():
        try:
            notes = json.loads(jf.read_text())
        except json.JSONDecodeError as exc:
            raise DataError(f"{jf}: not valid JSON ({exc})") from exc
    return DataSet(grid=grid, phases=phases, spectra=spectra, I=I, VI=VI, sig_I=sI, sig_VI=sVI, notes=notes)
