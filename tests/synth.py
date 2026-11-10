"""Synthetic library generator (VERIFICATION.md section 4): a small library in the library format, fixed by formulas.

- Teff: 20 000, 30 000, 40 000, 50 000 K; files 20000.bin ... 50000.bin
- B: 31 values, 50 to 800 MG in steps of 25 MG
- psi: 9 values, 0 to 90 degrees in steps of 11.25 degrees
- wavelength: 4000 to 7000 A in steps of 1 A, both ends included (3001 points); wavelength file `-`
- I(lambda) = (Teff / 30 000 K) x [1 - sum_{k=0..4} 0.3 exp(-(lambda - lambda_k)^2 / (2 x 4^2))],
  lambda_k = 4500 + 500 k + SHIFT (k - 2) B, SHIFT = 0.02 A/MG (variant a) or 0.022 (variant b)
- V(lambda) = 0.01 x cos psi x dI/dlambda x (1 A), derivative analytic

Variant "planck" (check V2c): the 30 000 K spectra of variant a times P(Teff, lambda) / P(30 000 K, lambda).
Usage: python tests/synth.py <outdir> [--variant a|b|planck]
Nothing here comes from MAGWD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zeemap.library import LibraryInfo, write_info, write_slice  # noqa: E402

TEFFS = np.array([20000.0, 30000.0, 40000.0, 50000.0])
BS = np.arange(50.0, 800.0 + 1e-9, 25.0)  # 31 values
PSIS = np.arange(0.0, 90.0 + 1e-9, 11.25)  # 9 values
WAVELENGTHS = np.arange(4000.0, 7000.0 + 1e-9, 1.0)  # 3001 values
SIGMA = 4.0
DEPTH = 0.3
SHIFT = {"a": 0.02, "b": 0.022, "planck": 0.02}
PLANCK_X_AK = 1.438776877e8  # h c / k in A K


def planck_factor(lam: np.ndarray, teff: float) -> np.ndarray:
    """P(Teff, lambda) / P(30 000 K, lambda), P the Planck function B_lambda (variant "planck", check V2c)."""
    x = PLANCK_X_AK / np.asarray(lam, dtype=float)
    return np.expm1(x / 30000.0) / np.expm1(x / float(teff))


def info(variant: str = "a") -> LibraryInfo:
    return LibraryInfo(
        name=f"synth_{variant}",
        wavelengths=WAVELENGTHS,
        teffs=TEFFS,
        mus=np.array([1.0]),
        bs=BS,
        psis=PSIS,
        wavelength_file="-",
        base_dir=".",
        unused=0.3,
    )


def line_centres(b: float, variant: str = "a") -> np.ndarray:
    k = np.arange(5)
    return 4500.0 + 500.0 * k + SHIFT[variant] * (k - 2) * b


def intensity(lam: np.ndarray, teff: float, b: float, variant: str = "a") -> np.ndarray:
    """I(lambda) for one (Teff, B); independent of psi."""
    centres = line_centres(b, variant)
    prof = np.exp(-((lam[:, None] - centres[None, :]) ** 2) / (2 * SIGMA**2))
    return (teff / 30000.0) * (1.0 - DEPTH * prof.sum(axis=1))


def d_intensity(lam: np.ndarray, teff: float, b: float, variant: str = "a") -> np.ndarray:
    """dI/dlambda, analytic."""
    centres = line_centres(b, variant)
    d = lam[:, None] - centres[None, :]
    prof = np.exp(-(d**2) / (2 * SIGMA**2))
    return (teff / 30000.0) * (DEPTH * (d / SIGMA**2) * prof).sum(axis=1)


def cos_psi(psi_deg: float) -> float:
    """cos psi, exactly 0 at 90 degrees (np.cos gives 6e-17 there); the library's V is then exactly 0 at psi = 90."""
    return 0.0 if psi_deg == 90.0 else float(np.cos(np.radians(psi_deg)))


def stokes_v(lam: np.ndarray, teff: float, b: float, psi_deg: float, variant: str = "a") -> np.ndarray:
    return 0.01 * cos_psi(psi_deg) * d_intensity(lam, teff, b, variant) * 1.0


def make_slice(teff: float, variant: str = "a", inf: LibraryInfo | None = None) -> np.ndarray:
    inf = inf or info(variant)
    out = np.empty(inf.slice_shape(), dtype=np.float32)
    for ib, b in enumerate(inf.bs):
        if variant == "planck":  # the 30 000 K spectra of variant a times the Planck ratio: exact for "planck" weights
            f = planck_factor(inf.wavelengths, teff)
            i_spec = intensity(inf.wavelengths, 30000.0, b, "a") * f
            di = d_intensity(inf.wavelengths, 30000.0, b, "a") * f
        else:
            i_spec = intensity(inf.wavelengths, teff, b, variant)
            di = d_intensity(inf.wavelengths, teff, b, variant)
        for ip, psi in enumerate(inf.psis):
            out[ib, ip, 0] = i_spec
            out[ib, ip, 1] = 0.01 * cos_psi(psi) * di
    return out


def write_library(outdir: str | Path, variant: str = "a", teffs=None, mus=None, mu_factor=None) -> LibraryInfo:
    """`mus` (increasing in (0, 1]) makes a mu-grid library (LIB-8, check V2k) whose records are those of the variant
    times `mu_factor(mu)` (default 1)."""
    import dataclasses

    inf = info(variant)
    if teffs is not None:  # other temperatures (tests: an exact slice between the standard ones)
        inf = dataclasses.replace(inf, teffs=np.asarray(teffs, dtype=float))
    if mus is not None:
        mus = np.asarray(mus, dtype=float)
        inf = dataclasses.replace(inf, mus=mus)
        write_info(outdir, inf)
        base = dataclasses.replace(inf, mus=np.array([1.0]))
        fac = (lambda m: 1.0) if mu_factor is None else mu_factor
        for teff in inf.teffs:
            one = make_slice(teff, variant, base)
            write_slice(outdir, inf, teff, np.stack([one * np.float32(fac(m)) for m in mus]))
        return inf
    write_info(outdir, inf)
    for teff in inf.teffs:
        write_slice(outdir, inf, teff, make_slice(teff, variant, inf))
    return inf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("outdir")
    p.add_argument("--variant", choices=sorted(SHIFT), default="a")
    a = p.parse_args(argv)
    inf = write_library(a.outdir, a.variant)
    print(f"wrote synthetic library {inf.name} to {a.outdir}: {len(inf.teffs)} slices of {inf.slice_nbytes()} bytes")


if __name__ == "__main__":
    main()
