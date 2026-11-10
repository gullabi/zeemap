"""`zeemap inject`: simulated observations from a known geometry (SPEC.md CLI-4; VERIFICATION.md V7b).

The configuration's parameter `value`s are evaluated with the Problem's forward model on the configured data grid,
phases and windows (no observation files are read: `problem.blank_dataset`). Written to the output directory, in the
FORMATS.md section 2 layout and under the file names of the configuration's `[[data.phase]]` blocks (their base names),
so that the same configuration with `data.path` pointing at the directory fits the injected data:

- `<file>`: wavelength, I, V/I (all grid pixels; NaN where the library does not cover the pixel);
- `<ierr>`: wavelength, error of I: sigma_I = `sigma_i` x the median model I over the fitted pixels of that phase,
  constant over the grid (written relative to I when `data.errors.absolute = false`);
- `<vierr>` (when configured): wavelength, sigma_VI = `sigma_vi` (absolute), constant;
- `dataset.json` (origin and truth) and `truth.json` (parameters, derived quantities, noise settings).

Noise: "none" (V7b), "white" (independent Gaussian per pixel with the sigmas above), "full" (on the fitted pixels of
each spectrum a draw from the objective's covariance sigma^2 D K_s D, K_s = objective.kernel(s): the short correlated
term, the discrepancy term and the jitter of the full likelihood with noise scale 1; white noise on the other
pixels; for kind "chi2" K_s is the identity, so "full" equals "white"). The random stream is
numpy.random.default_rng(seed), drawn in a fixed order (phase by phase: I, then V/I).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import atomic
from .likelihood import Objective
from .problem import Problem, blank_dataset

NOISE_KINDS = ("none", "white", "full")


def inject(cfg: dict, outdir, seed: int = 1, noise: str = "none", sigma_i: float = 0.01,
           sigma_vi: float = 0.003) -> dict:
    """Write the simulated data set to `outdir`; returns the truth dict (also in truth.json)."""
    if noise not in NOISE_KINDS:
        raise ValueError(f"noise must be one of {', '.join(NOISE_KINDS)}, got {noise!r}")
    if not (sigma_i > 0 and sigma_vi > 0):
        raise ValueError("--sigma-i and --sigma-vi must be > 0")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ds = blank_dataset(cfg)
    problem = Problem(cfg, dataset=ds)
    x = problem.value.copy()
    I, VI, bad = problem.spectra(x[None, :])
    if bad[0]:
        raise ValueError("the configured parameter values give an invalid model (check |c| < 1, u in [-1, 1], Teff)")
    I, VI = I[0], VI[0]
    lam = ds.grid.wavelengths
    P = len(ds.phases)
    sig_I = np.empty((P, lam.size))
    sig_VI = np.full((P, lam.size), float(sigma_vi))
    for p in range(P):
        pix = np.concatenate([s.pixels for s in ds.spectra if s.phase_index == p] or [np.arange(lam.size)])
        ref = float(np.nanmedian(I[p, pix]))
        sig_I[p] = sigma_i * ref
    # noisy data
    rng = np.random.default_rng(seed)
    dI, dVI = I.copy(), VI.copy()
    kernels = {}
    if noise == "full":
        for s in ds.spectra:  # the objective on the blank data set only supplies the kernels (sigma enters below)
            s.sigma = (sig_I if s.kind == "I" else sig_VI)[s.phase_index, s.pixels].copy()
        obj = Objective(ds, cfg)
        for j, s in enumerate(ds.spectra):
            kernels[(s.phase_index, s.kind)] = (s.pixels, np.linalg.cholesky(obj.kernel(j)))
    if noise != "none":
        for p in range(P):
            for kind, d, sg in (("I", dI, sig_I), ("VI", dVI, sig_VI)):
                e = rng.standard_normal(lam.size)
                if (p, kind) in kernels:
                    pix, L = kernels[(p, kind)]
                    e[pix] = L @ e[pix]
                d[p] += sg[p] * e
    absolute = bool(cfg["data"]["errors"]["absolute"])
    files = []
    fmt = "%.10g"
    for p, pc in enumerate(cfg["data"]["phase"]):
        f = out / os.path.basename(pc["file"])
        atomic.savetxt(f, np.column_stack([lam, dI[p], dVI[p]]), fmt=fmt)
        fi = out / os.path.basename(pc["ierr"])
        ierr = sig_I[p] if absolute else sig_I[p] / np.abs(dI[p])
        atomic.savetxt(fi, np.column_stack([lam, ierr]), fmt=fmt)
        files += [f.name, fi.name]
        if pc.get("vierr"):
            fv = out / os.path.basename(pc["vierr"])
            atomic.savetxt(fv, np.column_stack([lam, sig_VI[p]]), fmt=fmt)
            files.append(fv.name)
    truth = {
        "parameters": problem.grouped(x),
        "vector": {n: float(v) for n, v in zip(problem.names, x)},
        "derived": problem.derived(x),
        "noise": {"kind": noise, "seed": int(seed), "sigma_i_relative_to_median_I": float(sigma_i),
                  "sigma_I": [float(sig_I[p, 0]) for p in range(P)], "sigma_VI": float(sigma_vi)},
        "model": {"field": problem.field.name, "surface": cfg["model"]["surface"], "fwhm": cfg["model"]["fwhm"],
                  "ld_mode": cfg["model"]["ld_mode"],
                  "library": os.path.basename(cfg["library"]["path"].rstrip("/")),
                  "objective_kind": cfg["objective"]["kind"]},
        "phases": [float(v) for v in ds.phases],
        "files": files,
    }
    dataset = {"origin": "SYNTHETIC: zeemap inject (simulated with zeemap's own forward model)",
               "phases": truth["phases"], "truth": truth}
    atomic.write_text(out / "dataset.json", json.dumps(dataset, indent=2) + "\n")
    atomic.write_text(out / "truth.json", json.dumps(truth, indent=2) + "\n")
    return truth
