"""Synthetic Zeeman library and test observation for a magnetic DA white dwarf (DAH) at a few to a few tens of MG.

THIS IS A TOY WITH RECOGNISABLE PHYSICS, NOT A MODEL ATMOSPHERE. There is no radiative transfer, no atmosphere
structure, no Stark theory and no magnetic line theory beyond the linear Zeeman triplet plus a quadratic blueshift.
It exists so that end-to-end fits (the quick start, the tests) have a library and an observation that look like a hydrogen-atmosphere
white dwarf with split Balmer lines in Stokes I and V. Everything here is written from textbook formulas; nothing
comes from any other code.

Model (per (Teff, B, psi); log g fixed at 8, implicit):
- Wavelengths 3600-7200 A, 1 A steps (3601 points). Lines H-alpha ... H-epsilon (lambda0 in LINES, upper level n,
  lower level 2).
- Continuum C(lambda) = B_lambda(Teff) / B_lambda(Teff, 5500 A) (Planck, normalised to 1 at 5500 A). The Balmer jump
  (3646 A) lies outside the grid and is ignored.
- Line profile P(x): a Voigt profile normalised to peak 1 (scipy.special.voigt_profile), Gaussian sigma = thermal
  Doppler width lambda0 sqrt(k Teff / m_H) / c + 0.5 A, Lorentz HWHM gamma_n = GAMMA0 (n^2 - 4)^GAMMA_P, a stand-in for
  Stark broadening at log g = 8 (FWHM 15 A for H-alpha, 30 A for H-beta, broader for the higher lines; independent
  of Teff).
- Central depth d_n(Teff) = DMAX_n exp(-(ln(Teff / 12000 K) / 0.35)^2): peaks at the Balmer maximum near 12000 K.
- Linear Zeeman triplet (normal Zeeman effect, valid only while the shift is small against the level spacing,
  roughly B < ~20 MG for the lower Balmer lines, less for the higher ones): pi unshifted, sigma components at
  +-dl_Z = +-4.67e-13 lambda0^2 B[G] A (lambda0 in A). For H-alpha at 10 MG this is 201 A.
- Quadratic Zeeman shift (blueshift of the whole pattern; an order-of-magnitude estimate): diamagnetic term H_2 = e^2 B^2 (x^2 + y^2) /
  (8 m_e c^2) (Gaussian units), with <x^2 + y^2> ~ n^4 a0^2 (the n^4 scaling for m = 0; the true expectation value
  depends on l and m, so this is an order-of-magnitude estimate). Both levels shift up; the transition energy
  grows by dE = K_DIA B^2 (n_u^4 - n_l^4), K_DIA = e^2 a0^2 / (8 m_e c^2) = 9.86e-31 erg G^-2, and
  dl_Q = -lambda0^2 dE / (h c). For H-alpha (n 3 -> 2) this is -13.9 A at 10 MG (-0.14 A at 1 MG); with the upper
  level alone (n_u^4) it would be -17.3 A, which matches the usual textbook form -4.97e-23 lambda0^2 n^4 B[G]^2 A.
  Magnetic intensification and the field dependence of the pi/sigma strengths are ignored.
- Angular weights (Unno-type, weak field), psi = angle between field and line of sight: for each line the absorption
  a = d [ sin^2(psi)/2 P_pi + (1 + cos^2 psi)/4 (P_sigma+ + P_sigma-) ]; when the components overlap (B -> 0) the
  weights add to sin^2/2 + 2 (1 + cos^2)/4 = 1, i.e. depth d, as for an unsplit line.
  I = C prod_lines (1 - a_line);
  V = sum_lines C prod_{other lines} (1 - a_k) d cos(psi) / 2 (P_sigma- - P_sigma+).
- Sign convention: sigma+ is the red component (lambda0 + dl_Z), sigma- the blue one. For a field pointing towards
  the observer (cos psi > 0) V is positive in the blue sigma component and negative in the red one; the red sigma
  component is then taken as left circularly polarised. Only psi 0..90 is stored; V(180 - psi) = -V(psi) follows
  from the library mirror rule (FORMATS.md section 1). At psi = 90 V is exactly 0 (cos 90 is taken as exactly 0).

Library grid (default): Teff 10000, 12000, 14000, 17000, 20000 K; B 1..40 MG step 1; psi 0..90 step 11.25; mu = 1.
`--fast`: Teff 10000, 14000, 20000; B 1..36 step 5.

The observation writer (`inject`) integrates over the visible disc of a centred dipole directly with the formulas
above (no library lookup, independent of the zeemap forward model), convolves with a Gaussian of FWHM 3 A, rebins
to 2 A pixels from 3800 to 6800 A, adds Gaussian noise and writes FORMATS.md section 2 files.

Usage:
  python tools/zeeman_library.py <outdir> [--teffs T ...] [--bmax 40] [--bstep 1] [--fast]
  python tools/zeeman_library.py inject <outdir> [--teff 13000] [--bp 12] [--beta 40] [--snr 40]
                                        [--sigma-vi 0.003] [--seed 1] [--npoints 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import voigt_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zeemap.library import LibraryInfo, write_info, write_slice  # noqa: E402

# --- grids ---------------------------------------------------------------------------------------------------------
WAVELENGTHS = np.arange(3600.0, 7200.0 + 1e-9, 1.0)  # 3601 points
TEFFS = np.array([10000.0, 12000.0, 14000.0, 17000.0, 20000.0])
TEFFS_FAST = np.array([10000.0, 14000.0, 20000.0])
PSIS = np.arange(0.0, 90.0 + 1e-9, 11.25)  # 9 values

# --- physics constants (CGS) ---------------------------------------------------------------------------------------
E_CHARGE = 4.80320e-10  # esu
A0 = 5.29177e-9  # cm, Bohr radius
M_E = 9.10938e-28  # g
M_H = 1.67353e-24  # g, hydrogen atom
C_CGS = 2.99792458e10  # cm/s
K_B = 1.380649e-16  # erg/K
HC_ERG_A = 6.62607e-27 * 2.99792458e18  # h c in erg A
HC_OVER_K = 1.438777e8  # A K (second radiation constant)
K_DIA = E_CHARGE**2 * A0**2 / (8.0 * M_E * C_CGS**2)  # erg G^-2, diamagnetic coefficient
ZEEMAN_LIN = 4.67e-13  # A^-1 G^-1: dl = ZEEMAN_LIN lambda0^2 B, normal Zeeman effect (e / (4 pi m_e c^2))

# --- lines: name, lambda0 (A, air), upper level n, maximum depth ---------------------------------------------------
LINES = (
    ("H-alpha", 6562.8, 3, 0.60),
    ("H-beta", 4861.3, 4, 0.55),
    ("H-gamma", 4340.5, 5, 0.50),
    ("H-delta", 4101.7, 6, 0.45),
    ("H-epsilon", 3970.1, 7, 0.40),
)
N_LOWER = 2
# Lorentz HWHM gamma_n = GAMMA0 (n^2 - 4)^GAMMA_P: 7.5 A (H-alpha, FWHM 15) and 15 A (H-beta, FWHM 30).
GAMMA_P = np.log(15.0 / 7.5) / np.log(12.0 / 5.0)
GAMMA0 = 7.5 / 5.0**GAMMA_P
T_BALMER = 12000.0
DEPTH_WIDTH = 0.35


def cos_psi(psi_deg: float) -> float:
    """cos psi, exactly 0 at 90 degrees, so that the stored V is exactly 0 there."""
    return 0.0 if float(psi_deg) == 90.0 else float(np.cos(np.radians(psi_deg)))


def continuum(lam: np.ndarray, teff: float) -> np.ndarray:
    """Planck B_lambda(Teff) normalised to 1 at 5500 A."""

    def planck(x):
        return x**-5.0 / np.expm1(HC_OVER_K / (x * teff))

    return planck(np.asarray(lam, float)) / planck(5500.0)


def gamma_lorentz(n: int) -> float:
    return float(GAMMA0 * (n * n - 4) ** GAMMA_P)


def sigma_gauss(lambda0: float, teff: float) -> float:
    return float(lambda0 * np.sqrt(K_B * teff / M_H) / C_CGS + 0.5)


def depth(dmax: float, teff: float) -> float:
    return float(dmax * np.exp(-((np.log(teff / T_BALMER) / DEPTH_WIDTH) ** 2)))


def linear_shift(lambda0: float, b_mg: float) -> float:
    """Sigma-component shift in A (positive), linear Zeeman effect."""
    return ZEEMAN_LIN * lambda0**2 * b_mg * 1e6


def quadratic_shift(lambda0: float, n_upper: int, b_mg: float) -> float:
    """Quadratic Zeeman shift of the whole pattern in A (negative: to the blue; an estimate), see module docstring."""
    de = K_DIA * (b_mg * 1e6) ** 2 * (n_upper**4 - N_LOWER**4)
    return -(lambda0**2) * de / HC_ERG_A


def _profile(x: np.ndarray, sig: float, gam: float) -> np.ndarray:
    return voigt_profile(x, sig, gam) / voigt_profile(0.0, sig, gam)


def line_components(lam: np.ndarray, line, teff: float, b_mg: float):
    """(d, P_pi, P_sigma+ (red), P_sigma- (blue)) of one line at field b_mg (MG), broadcast over lam."""
    _, l0, n, dmax = line
    sig, gam = sigma_gauss(l0, teff), gamma_lorentz(n)
    lc = l0 + quadratic_shift(l0, n, b_mg)
    dz = linear_shift(l0, b_mg)
    x = lam - lc
    return depth(dmax, teff), _profile(x, sig, gam), _profile(x - dz, sig, gam), _profile(x + dz, sig, gam)


def stokes(lam: np.ndarray, teff: float, b_mg, cpsi) -> tuple[np.ndarray, np.ndarray]:
    """Stokes I and V. `b_mg` and `cpsi` (cos psi, signed) may be scalars or arrays of shape (npoints, 1) to evaluate
    many surface points at once; lam then broadcasts along the last axis."""
    lam = np.asarray(lam, float)
    b = np.asarray(b_mg, float)
    c = np.asarray(cpsi, float)
    w_pi = (1.0 - c * c) / 2.0
    w_sig = (1.0 + c * c) / 4.0
    cont = continuum(lam, teff)
    trans, vparts = [], []
    for line in LINES:
        _, l0, n, dmax = line
        sig, gam = sigma_gauss(l0, teff), gamma_lorentz(n)
        d = depth(dmax, teff)
        lc = l0 + (-(l0**2) * K_DIA * (b * 1e6) ** 2 * (n**4 - N_LOWER**4) / HC_ERG_A)
        dz = ZEEMAN_LIN * l0**2 * b * 1e6
        x = lam - lc
        p_pi = _profile(x, sig, gam)
        p_red = _profile(x - dz, sig, gam)
        p_blue = _profile(x + dz, sig, gam)
        trans.append(1.0 - d * (w_pi * p_pi + w_sig * (p_red + p_blue)))
        vparts.append(d * c / 2.0 * (p_blue - p_red))
    trans = np.array(np.broadcast_arrays(*trans))
    total = np.prod(trans, axis=0)
    i_spec = cont * total
    v_spec = np.zeros_like(i_spec)
    for k, vp in enumerate(vparts):
        others = np.prod(np.delete(trans, k, axis=0), axis=0)
        v_spec = v_spec + cont * others * vp
    return i_spec, v_spec


def single_line_absorption(lam: np.ndarray, line_name: str, teff: float, b_mg: float, psi_deg: float) -> np.ndarray:
    """1 - I/C for one line alone (for equivalent widths)."""
    line = next(ln for ln in LINES if ln[0] == line_name)
    d, p_pi, p_red, p_blue = line_components(np.asarray(lam, float), line, teff, b_mg)
    c = cos_psi(psi_deg)
    return d * ((1 - c * c) / 2 * p_pi + (1 + c * c) / 4 * (p_red + p_blue))


# --- library -------------------------------------------------------------------------------------------------------
def library_info(teffs=TEFFS, bmax: float = 40.0, bstep: float = 1.0, name: str = "synth_zeeman") -> LibraryInfo:
    bs = np.arange(1.0, bmax + 1e-9, bstep)
    return LibraryInfo(
        name=name,
        wavelengths=WAVELENGTHS,
        teffs=np.array(sorted(float(t) for t in teffs)),
        mus=np.array([1.0]),
        bs=bs,
        psis=PSIS,
        wavelength_file="-",
        base_dir=".",
        unused=0.3,
    )


def make_slice(teff: float, inf: LibraryInfo) -> np.ndarray:
    out = np.empty(inf.slice_shape(), dtype=np.float32)
    for ib, b in enumerate(inf.bs):
        for ip, psi in enumerate(inf.psis):
            i_spec, v_spec = stokes(inf.wavelengths, teff, b, cos_psi(psi))
            out[ib, ip, 0] = i_spec
            out[ib, ip, 1] = v_spec
    return out


README_TEXT = """\
SYNTHETIC TOY LIBRARY - not a model atmosphere.
Written by tools/zeeman_library.py (zeemap): a hydrogen (DA) white dwarf toy with Balmer lines H-alpha..H-epsilon as
peak-normalised Voigt profiles on a Planck continuum, split by the linear Zeeman effect (pi + sigma triplet) plus a
quadratic blueshift estimate, with Unno-type weak-field angular weights for Stokes I and V. No radiative transfer.
Valid in spirit only for B below ~20 MG. For tests and the quick start only; do not use for science.
Grid: {nt} Teff {teffs} K; {nb} B {b0:g}..{b1:g} MG; psi 0..90 deg ({npsi} values); mu = 1; {nl} wavelengths
{l0:g}..{l1:g} A step 1 A.
"""


def write_library(outdir, teffs=TEFFS, bmax: float = 40.0, bstep: float = 1.0) -> LibraryInfo:
    outdir = Path(outdir)
    inf = library_info(teffs, bmax, bstep, name=outdir.name or "synth_zeeman")
    write_info(outdir, inf)
    for teff in inf.teffs:
        write_slice(outdir, inf, teff, make_slice(teff, inf))
    (outdir / "README.txt").write_text(
        README_TEXT.format(
            nt=len(inf.teffs),
            teffs=", ".join(f"{t:g}" for t in inf.teffs),
            nb=len(inf.bs),
            b0=inf.bs[0],
            b1=inf.bs[-1],
            npsi=len(inf.psis),
            nl=inf.nl,
            l0=inf.wavelengths[0],
            l1=inf.wavelengths[-1],
        )
    )
    return inf


# --- observation ---------------------------------------------------------------------------------------------------
OBS_GRID = (3800.0, 2.0, 1501)  # start, step, number of pixels (to 6800 A)


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """n points of equal area on the visible hemisphere (z = mu > 0, observer along +z); shape (n, 3)."""
    i = np.arange(n)
    mu = 1.0 - (i + 0.5) / n
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=1)


def dipole_field(points: np.ndarray, bp_mg: float, beta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """|B| (MG) and cos psi (field vs line of sight +z) of a centred dipole with polar field bp_mg, axis in the x-z
    plane at beta_deg to the line of sight: B = Bp/2 (3 (m.r) r - m)."""
    beta = np.radians(beta_deg)
    m = np.array([np.sin(beta), 0.0, np.cos(beta)])
    mr = points @ m
    bvec = bp_mg / 2.0 * (3.0 * mr[:, None] * points - m[None, :])
    bmag = np.linalg.norm(bvec, axis=1)
    return bmag, bvec[:, 2] / bmag


def disc_spectrum(teff: float, bp_mg: float, beta_deg: float, npoints: int = 2000, chunk: int = 100):
    """Disc-integrated I and V on WAVELENGTHS: weights area x mu x (0.7 + 0.3 mu), equal areas."""
    pts = fibonacci_hemisphere(npoints)
    bmag, cpsi = dipole_field(pts, bp_mg, beta_deg)
    mu = pts[:, 2]
    w = mu * (0.7 + 0.3 * mu)
    i_sum = np.zeros_like(WAVELENGTHS)
    v_sum = np.zeros_like(WAVELENGTHS)
    for s in range(0, npoints, chunk):
        sl = slice(s, s + chunk)
        i_loc, v_loc = stokes(WAVELENGTHS, teff, bmag[sl, None], cpsi[sl, None])
        i_sum += w[sl] @ i_loc
        v_sum += w[sl] @ v_loc
    return i_sum / w.sum(), v_sum / w.sum()


def _smooth_rebin(spec: np.ndarray, fwhm: float = 3.0) -> np.ndarray:
    """Gaussian convolution (FWHM in A, 1 A input grid), then 2 A pixels (trapezoid over each pixel)."""
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = int(np.ceil(5 * sig))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.pad(spec, half, mode="edge")
    conv = np.convolve(pad, k, mode="valid")
    start, step, n = OBS_GRID
    centres = start + step * np.arange(n)
    idx = np.rint(centres - WAVELENGTHS[0]).astype(int)
    return 0.25 * conv[idx - 1] + 0.5 * conv[idx] + 0.25 * conv[idx + 1]


def make_observation(
    outdir,
    teff: float = 13000.0,
    bp_mg: float = 12.0,
    beta_deg: float = 40.0,
    snr: float = 40.0,
    sigma_vi: float = 0.003,
    seed: int = 1,
    npoints: int = 2000,
    phase: str = "ph00",
) -> dict:
    """Write one noisy observed spectrum (FORMATS.md section 2) of a centred-dipole toy DAH; returns the truth."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    i_hr, v_hr = disc_spectrum(teff, bp_mg, beta_deg, npoints)
    i_obs = _smooth_rebin(i_hr)
    v_obs = _smooth_rebin(v_hr)
    start, step, n = OBS_GRID
    lam = start + step * np.arange(n)
    vi = v_obs / i_obs
    sigma_i = float(i_obs[np.argmin(np.abs(lam - 5500.0))] / snr)
    rng = np.random.default_rng(seed)
    i_noisy = i_obs + rng.normal(0.0, sigma_i, n)
    vi_noisy = vi + rng.normal(0.0, sigma_vi, n)
    fmt = "%.7g"
    np.savetxt(outdir / f"{phase}.dat", np.column_stack([lam, i_noisy, vi_noisy]), fmt=fmt)
    np.savetxt(outdir / f"{phase}_ierr.dat", np.column_stack([lam, np.full(n, sigma_i)]), fmt=fmt)
    np.savetxt(outdir / f"{phase}_vierr.dat", np.column_stack([lam, np.full(n, sigma_vi)]), fmt=fmt)
    truth = {
        "teff_K": teff,
        "bp_MG": bp_mg,
        "beta_deg": beta_deg,
        "geometry": "centred dipole, axis at beta to the line of sight",
        "noise": {"snr_I_at_5500A": snr, "sigma_I": sigma_i, "sigma_VI": sigma_vi, "seed": seed},
        "disc_points": npoints,
        "limb_darkening": "0.7 + 0.3 mu",
        "resolution_fwhm_A": 3.0,
        "grid": {"start_A": start, "step_A": step, "n": n},
    }
    dataset = {
        "origin": "SYNTHETIC: tools/zeeman_library.py inject (toy DAH, not a real star, not a model atmosphere)",
        "phases": [phase],
        "truth": truth,
        "notes": "Linear Zeeman triplet + quadratic blueshift estimate on Voigt Balmer lines; see the tool docstring.",
    }
    (outdir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (outdir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


# --- command line --------------------------------------------------------------------------------------------------
def _main_inject(argv) -> None:
    p = argparse.ArgumentParser(prog="zeeman_library.py inject", description="write a synthetic DAH observation")
    p.add_argument("outdir")
    p.add_argument("--teff", type=float, default=13000.0)
    p.add_argument("--bp", type=float, default=12.0, help="polar field, MG")
    p.add_argument("--beta", type=float, default=40.0, help="dipole inclination to the line of sight, degrees")
    p.add_argument("--snr", type=float, default=40.0)
    p.add_argument("--sigma-vi", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--npoints", type=int, default=2000)
    a = p.parse_args(argv)
    make_observation(a.outdir, a.teff, a.bp, a.beta, a.snr, a.sigma_vi, a.seed, a.npoints)
    size = sum(f.stat().st_size for f in Path(a.outdir).iterdir() if f.is_file())
    print(f"wrote synthetic observation (Teff {a.teff:g} K, Bp {a.bp:g} MG, beta {a.beta:g} deg) to {a.outdir}: "
          f"{size} bytes")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "inject":
        _main_inject(argv[1:])
        return
    p = argparse.ArgumentParser(description="write the synthetic Zeeman toy library (see the module docstring)")
    p.add_argument("outdir", nargs="?", default="synth_zeeman")
    p.add_argument("--teffs", type=float, nargs="+", default=None)
    p.add_argument("--bmax", type=float, default=40.0)
    p.add_argument("--bstep", type=float, default=1.0)
    p.add_argument("--fast", action="store_true", help="reduced grid for tests: 3 Teff, B 1..36 step 5")
    a = p.parse_args(argv)
    if a.fast:
        teffs, bmax, bstep = (a.teffs or TEFFS_FAST), 40.0, 5.0
    else:
        teffs, bmax, bstep = (a.teffs or TEFFS), a.bmax, a.bstep
    inf = write_library(a.outdir, teffs, bmax, bstep)
    total = len(inf.teffs) * inf.slice_nbytes()
    print(f"wrote synthetic Zeeman library {inf.name} to {a.outdir}: {len(inf.teffs)} slices of "
          f"{inf.slice_nbytes()} bytes ({total / 1e6:.1f} MB; NB {len(inf.bs)}, NP {len(inf.psis)}, NL {inf.nl})")


if __name__ == "__main__":
    main()
