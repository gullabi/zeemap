"""Run diagnostics for the paper's comparisons (SPEC DIAG-1, DIAG-3; DESIGN.md section 5b2; decision #23).

- `floor_chi2`: per phase, the smallest plain chi2 that a non-negative mixture of the library records reaches, with I
  and V stacked (V = (V/I)_obs x I_obs, because V/I of a mixture is not linear), solved by scipy's NNLS on the
  error-scaled system; next to it the plain chi2 of the given mode at the same pixels (Objective.per_spectrum: the
  model I with the continuum fitted inside the likelihood, as written to spectra/; the floor has none, so the two are
  not ordered). The columns are the records of
  the Teff slices that bracket the mode's Teff (all selected slices would make the system too wide for NNLS on real
  libraries; DESIGN.md section 5b2), at the mode's wavelength shift (a free shift: the record sets of the shift nodes
  combined with the forward model's Lagrange weights, FWD-13).
- `window_residuals`: per library version (the run's and each `systematics.library_alternatives`), spectrum and fit
  window, at fixed parameters (no refit): the pixel count and the whitened chi2 under the run's continuum setting and
  at degree 0. `fit` evaluates it at the best mode of each family (`family_best_points`) and tags each row with the
  family and mode ids.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import nnls

from .library import SLICE_SNAP_REL, on_slice
from .likelihood import Objective


def _bracketing_slices(problem, x) -> list[int]:
    teffs = np.asarray(problem.library.teffs, dtype=float)
    if len(teffs) == 1 or getattr(problem, "iteff", None) is None:
        if len(teffs) == 1:
            return [0]
        t = float(problem.value[problem.names.index("teff")]) if "teff" in problem.names else float(teffs[0])
    else:
        t = float(np.asarray(x, dtype=float)[problem.iteff])
    # the slices whose interpolation weights are nonzero (Library.interp_weights: one slice at a selected Teff, to
    # rounding, or beyond an edge slice; else the two around t)
    tol = SLICE_SNAP_REL * abs(t)
    if t <= teffs[0] + tol:
        return [0]
    if t >= teffs[-1] - tol:
        return [len(teffs) - 1]
    i1 = int(np.searchsorted(teffs, t))
    if on_slice(t, teffs[i1 - 1]):
        return [i1 - 1]
    if on_slice(t, teffs[i1]):
        return [i1]
    return [i1 - 1, i1]


def _records_at_shift(problem, x, t: int) -> np.ndarray:
    """The records (nodes, 2, M) of Teff slice t at the mode's shift. A fixed shift is built into the records (one set
    per slice); a free one has a set per (slice, shift node), index t * n_shift + j (forward.py), which the forward
    model combines with the cubic Lagrange weights of the row's shift (`Forward._coeffs`): the same combination here."""
    fw = problem.forward
    if fw.shift_nodes is None:
        return fw.records[t]
    v = float(np.asarray(x, dtype=float)[problem.ishift])
    w = fw.shift_weights(v)[0]
    return sum(w[j] * fw.records[t * fw.n_shift + j] for j in np.flatnonzero(w))


def floor_chi2(problem, x) -> dict:
    """DIAG-1 per phase index: {"phase", "n_I", "n_V", "chi2_floor", "chi2_floor_I", "chi2_floor_V", "chi2_mode",
    "slices"} (plain chi2: the errors and weights of LIKE-6; no kernel, no continuum, no noise scale)."""
    fw = problem.forward
    ds = problem.dataset
    slices = _bracketing_slices(problem, x)
    recs = {t: _records_at_shift(problem, x, t) for t in slices}
    per_spec = problem.objective.per_spectrum(*problem.model_spectra(np.asarray(x, dtype=float)))
    out = {}
    for p in range(len(ds.phases)):
        rows_y, rows_A, kinds, nI, nV = [], [], [], 0, 0
        for sp in ds.spectra:
            if sp.phase_index != p:
                continue
            pix = np.asarray(sp.pixels)
            sig = np.asarray(sp.sigma, float) / np.sqrt(np.asarray(sp.weight, float))
            cols = np.concatenate([recs[t][:, 0 if sp.kind == "I" else 1, :][:, pix] for t in slices], axis=0).T
            if sp.kind == "I":
                y = np.asarray(sp.data, float)
                nI = pix.size
            else:
                Iobs = ds.I[p, pix]
                if not np.all(np.isfinite(Iobs)):
                    continue  # V rows need the observed I at the same pixels
                y = np.asarray(sp.data, float) * Iobs
                sig = sig * np.abs(Iobs)
                nV = pix.size
            ok = np.all(np.isfinite(cols), axis=1) & np.isfinite(y) & (sig > 0)
            rows_y.append((y / sig)[ok])
            rows_A.append((cols / sig[:, None])[ok])
            kinds.append(np.full(int(ok.sum()), sp.kind == "I"))
        if not rows_y:
            continue
        b = np.concatenate(rows_y)
        A = np.concatenate(rows_A, axis=0)
        c, rnorm = nnls(A, b, maxiter=50 * A.shape[1])
        r = A @ c - b
        isI = np.concatenate(kinds)
        cI, cV = float(np.sum(r[isI] ** 2)), float(np.sum(r[~isI] ** 2))
        mode = sum(v["chi2"] for lab, v in per_spec.items() if lab.startswith(f"{ds.phases[p]:.3f}_"))
        out[int(p)] = {"phase": float(ds.phases[p]), "n_I": int(nI), "n_V": int(nV), "chi2_floor": float(rnorm ** 2),
                       "chi2_floor_I": cI, "chi2_floor_V": cV, "chi2_mode": float(mode),
                       "slices": [float(problem.library.teffs[t]) for t in slices]}
    return out


def _window_chi2(objective, I, VI) -> dict:
    res = objective.residuals(I, VI, scales=np.ones(len(objective.labels)))
    out = {}
    for sp, r in zip(objective.dataset.spectra, res):
        win = np.asarray(sp.window)
        out[sp.label] = {int(w): {"n": int(np.sum(win == w)), "chi2": float(np.sum(r[win == w] ** 2))}
                         for w in np.unique(win)}
    return out


def family_best_points(families: dict | None, modes) -> list[tuple]:
    """DIAG-3's parameter sets: (family id, mode id, x) for the best mode of each family, at the family's best node (the
    canonical end point of families.json, where its profiles start); without families, (None, 1, best mode's x)."""
    if families and families.get("families"):
        out = []
        for fam in families["families"]:
            node = next(n for n in fam["nodes"] if n["node"] == fam["best"])
            out.append((int(fam["id"]), int(node["mode"]), np.asarray(node["x"], dtype=float)))
        return out
    return [(None, 1, np.asarray(modes[0].x, dtype=float))] if modes else []


def window_residuals(problems: dict, x) -> list[dict]:
    """DIAG-3: rows {library, spectrum, window, n, chi2, chi2_deg0} for each problem (library name -> problem, all with
    the same data set and parameter layout) at the parameters x."""
    rows = []
    for name, pr in problems.items():
        I, VI = pr.model_spectra(np.asarray(x, dtype=float))
        base = _window_chi2(pr.objective, I, VI)
        cfg0 = copy.deepcopy(pr.cfg)
        cont = cfg0["objective"].get("continuum")
        deg0 = None
        if cont is not False:
            cfg0["objective"]["continuum"] = {"degree": 0}
            deg0 = _window_chi2(Objective(pr.dataset, cfg0), I, VI)
        for lab, wins in base.items():
            for w, v in wins.items():
                rows.append({"library": name, "spectrum": lab, "window": w, "n": v["n"], "chi2": v["chi2"],
                             "chi2_deg0": None if deg0 is None else deg0[lab][w]["chi2"]})
    return rows
