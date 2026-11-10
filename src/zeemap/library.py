"""Reader and writer of the library format (FORMATS.md section 1; SPEC.md LIB-1).

A library is a directory with `info.dat` (white-space separated tokens) and one raw little-endian float32 file per
temperature, `<Teff>.bin` (older libraries: `<index>.bin`), holding for each B (outer loop) and each psi (inner loop)
NL values of I followed by NL values of V: NB x NP x 2 x NL values in all. The reader checks that size.

Only psi from 0 to 90 degrees is stored. The reader functions return the stored data unchanged; `Library.mirror_slice`
applies the mirror rule (I(180 - psi) = I(psi), V(180 - psi) = -V(psi)) and the V = 0 rule at psi = 90 degrees.

`Library` (LIB-2, LIB-3, LIB-7) adds the slice cache, the interpolation in Teff and the range checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE = np.dtype("<f4")


class LibraryError(ValueError):
    """A library that cannot be used; the message names what is wrong."""


@dataclass(frozen=True)
class LibraryInfo:
    """The contents of `info.dat`."""

    name: str
    wavelengths: np.ndarray  # A, NL values
    teffs: np.ndarray  # K, increasing
    mus: np.ndarray  # viewing angles: one value (1), or several increasing in (0, 1] (a mu-grid library, LIB-8)
    bs: np.ndarray  # MG, increasing
    psis: np.ndarray  # degrees, increasing, 0 to 90
    wavelength_file: str = "-"
    base_dir: str = "."
    unused: float = 0.3

    @property
    def nl(self) -> int:
        return len(self.wavelengths)

    def slice_shape(self) -> tuple[int, ...]:
        """(NB, NP, 2, NL), or (NMU, NB, NP, 2, NL) for a mu-grid library: the array one `.bin` file holds."""
        if len(self.mus) > 1:
            return (len(self.mus), len(self.bs), len(self.psis), 2, self.nl)
        return (len(self.bs), len(self.psis), 2, self.nl)

    def slice_nbytes(self) -> int:
        return int(np.prod(self.slice_shape())) * DTYPE.itemsize


def _tokens(text: str) -> list[str]:
    return text.split()


def read_info(libdir: str | Path) -> LibraryInfo:
    """Read `info.dat`. Wavelengths come from the wavelength file if it exists, else from NL, lambda0, dlambda."""
    libdir = Path(libdir)
    path = libdir / "info.dat"
    if not path.is_file():
        raise LibraryError(f"library {libdir}: no info.dat")
    tok = _tokens(path.read_text())
    try:
        pos = 0

        def take(n: int) -> list[str]:
            nonlocal pos
            out = tok[pos : pos + n]
            if len(out) < n:
                raise LibraryError(f"library {libdir}: info.dat ends early (token {pos + 1})")
            pos += n
            return out

        base_dir, name, wavelength_file, unused = take(4)
        nl = int(take(1)[0])
        lambda0, dlambda = (float(x) for x in take(2))
        nt = int(take(1)[0])
        teffs = np.array([float(x) for x in take(nt)])
        nmu = int(take(1)[0])
        mus = np.array([float(x) for x in take(nmu)])
        nb = int(take(1)[0])
        bs = np.array([float(x) for x in take(nb)])
        npsi = int(take(1)[0])
        psis = np.array([float(x) for x in take(npsi)])
    except ValueError as e:  # int()/float() failures
        raise LibraryError(f"library {libdir}: cannot parse info.dat ({e})") from e

    wfile = libdir / wavelength_file
    if wavelength_file != "-" and wfile.is_file():
        wavelengths = np.loadtxt(wfile, ndmin=1, dtype=float)
        if len(wavelengths) != nl:
            raise LibraryError(
                f"library {libdir}: wavelength file {wavelength_file} has {len(wavelengths)} values, info.dat says NL = {nl}"
            )
    else:
        wavelengths = lambda0 + dlambda * np.arange(nl)

    info = LibraryInfo(
        name=name,
        wavelengths=wavelengths,
        teffs=teffs,
        mus=mus,
        bs=bs,
        psis=psis,
        wavelength_file=wavelength_file,
        base_dir=base_dir,
        unused=float(unused),
    )
    check_info(info, libdir)
    return info


def check_info(info: LibraryInfo, libdir: str | Path = "") -> None:
    """Refuse libraries zeemap does not support (FORMATS.md section 1)."""
    where = f"library {libdir}: " if libdir else ""
    if len(info.mus) == 0 or np.any(np.diff(info.mus) <= 0) or info.mus[0] <= 0.0 or info.mus[-1] > 1.0:
        raise LibraryError(f"{where}mu values must increase within (0, 1], got {info.mus.tolist()}")
    if len(info.mus) == 1 and info.mus[0] != 1.0:
        raise LibraryError(f"{where}a library with one mu value must have mu = 1, got {info.mus.tolist()}")
    if np.any(np.diff(info.bs) <= 0):
        raise LibraryError(f"{where}B values are not increasing")
    if np.any(np.diff(info.psis) <= 0) or info.psis[0] != 0.0 or info.psis[-1] != 90.0:
        raise LibraryError(f"{where}psi values must increase from 0 to exactly 90 degrees, got {info.psis.tolist()}")
    if len(info.teffs) == 0 or np.any(np.diff(info.teffs) <= 0):
        raise LibraryError(f"{where}Teff values must be increasing and not empty, got {info.teffs.tolist()}")


def slice_path(libdir: str | Path, info: LibraryInfo, teff: float) -> Path:
    """`<Teff>.bin` if it exists, else `<index>.bin` (older libraries)."""
    libdir = Path(libdir)
    idx = _teff_index(info, teff)
    for cand in (f"{int(round(teff))}.bin", f"{idx}.bin"):
        if (libdir / cand).is_file():
            return libdir / cand
    raise LibraryError(f"library {libdir}: no slice file for Teff {teff:g} ({int(round(teff))}.bin or {idx}.bin)")


def _teff_index(info: LibraryInfo, teff: float) -> int:
    hits = np.flatnonzero(np.isclose(info.teffs, teff, rtol=0, atol=0.5))
    if len(hits) != 1:
        raise LibraryError(f"Teff {teff:g} is not a library temperature; the library has {info.teffs.tolist()}")
    return int(hits[0])


def read_slice(libdir: str | Path, info: LibraryInfo, teff: float) -> np.ndarray:
    """One Teff slice as an array (NB, NP, 2, NL) of float32 (index 2: 0 = I, 1 = V). Checks the file size."""
    path = slice_path(libdir, info, teff)
    expected = info.slice_nbytes()
    actual = path.stat().st_size
    if actual != expected:
        mu = f"NMU x " if len(info.mus) > 1 else ""
        mun = f"{len(info.mus)} x " if len(info.mus) > 1 else ""
        raise LibraryError(
            f"{path}: {actual} bytes, but info.dat implies {mu}NB x NP x 2 x NL = "
            f"{mun}{len(info.bs)} x {len(info.psis)} x 2 x {info.nl} float32 = {expected} bytes"
        )
    data = np.fromfile(path, dtype=DTYPE)
    return data.reshape(info.slice_shape())


def write_info(libdir: str | Path, info: LibraryInfo) -> Path:
    """Write `info.dat` in the token order of FORMATS.md section 1."""
    libdir = Path(libdir)
    libdir.mkdir(parents=True, exist_ok=True)
    w = info.wavelengths
    dl = float(w[1] - w[0]) if len(w) > 1 else 1.0
    lines = [
        f"{info.base_dir} {info.name} {info.wavelength_file} {info.unused:g}",
        f"{info.nl} {w[0]:.6f} {dl:.6f}",
        f"{len(info.teffs)} " + " ".join(f"{t:g}" for t in info.teffs),
        f"{len(info.mus)} " + " ".join(f"{m:g}" for m in info.mus),
        f"{len(info.bs)} " + " ".join(f"{b:g}" for b in info.bs),
        f"{len(info.psis)} " + " ".join(f"{p:g}" for p in info.psis),
    ]
    path = libdir / "info.dat"
    path.write_text("\n".join(lines) + "\n")
    if info.wavelength_file != "-":
        np.savetxt(libdir / info.wavelength_file, w, fmt="%.6f")
    return path


def write_slice(libdir: str | Path, info: LibraryInfo, teff: float, data: np.ndarray) -> Path:
    """Write one slice (NB, NP, 2, NL) as `<Teff>.bin`, raw little-endian float32."""
    libdir = Path(libdir)
    data = np.asarray(data)
    if data.shape != info.slice_shape():
        raise LibraryError(f"slice for Teff {teff:g} has shape {data.shape}, expected {info.slice_shape()}")
    _teff_index(info, teff)
    path = libdir / f"{int(round(teff))}.bin"
    np.ascontiguousarray(data, dtype=DTYPE).tofile(path)
    return path


# --------------------------------------------------------------------------------------------------------------------
# Library: slice cache, Teff interpolation, range checks

# A Teff within this fraction of a selected Teff is that slice (interp_weights): a rounding tolerance only, so that the
# weights stay continuous on the scale of a finite-difference step (a window of 0.5 K made them constant there).
SLICE_SNAP_REL = 1e-9


def on_slice(teff: float, slice_teff: float) -> bool:
    """True when `teff` reads as the slice `slice_teff` (the snap rule of `Library.interp_weights`: within
    `SLICE_SNAP_REL` x Teff). The one rule for every "is this Teff a slice" test."""
    return bool(abs(float(slice_teff) - float(teff)) <= SLICE_SNAP_REL * abs(float(teff)))


class Library:
    """A library opened for use: `info.dat` read at once, Teff slices read lazily and cached (LIB-2).

    `teffs` selects the library temperatures to use (K, each within 0.5 K of a library Teff); None = all of them.
    Each selected slice is read from disk at most once over the life of the object; `reads` counts the files read (V2f).
    """

    def __init__(self, libdir: str | Path, teffs=None):
        self.libdir = Path(libdir)
        self.info: LibraryInfo = read_info(self.libdir)
        lib_t = self.info.teffs
        if teffs is None:
            idx = list(range(len(lib_t)))
        else:
            sel = np.atleast_1d(np.asarray(teffs, dtype=float))
            if sel.size == 0:
                raise LibraryError(f"library {self.libdir}: empty Teff selection; the library has {lib_t.tolist()}")
            idx = []
            for t in sel:
                hits = np.flatnonzero(np.abs(lib_t - t) <= 0.5)
                if len(hits) != 1:
                    raise LibraryError(
                        f"library {self.libdir}: Teff {t:g} K is not a library temperature; "
                        f"the library has {lib_t.tolist()}"
                    )
                idx.append(int(hits[0]))
            idx = sorted(set(idx))
        self._lib_index = np.array(idx, dtype=int)  # selected -> library index
        self.teffs = lib_t[self._lib_index].copy()  # selected temperatures, increasing
        self._cache: dict[int, np.ndarray] = {}
        self.reads = 0

    # ---- properties
    @property
    def b_min(self) -> float:
        return float(self.info.bs[0])

    @property
    def b_max(self) -> float:
        return float(self.info.bs[-1])

    @property
    def wavelength_min(self) -> float:
        return float(self.info.wavelengths[0])

    @property
    def wavelength_max(self) -> float:
        return float(self.info.wavelengths[-1])

    @property
    def teff_min(self) -> float:
        """Lowest selected Teff (K)."""
        return float(self.teffs[0])

    @property
    def teff_max(self) -> float:
        """Highest selected Teff (K)."""
        return float(self.teffs[-1])

    # ---- slices
    def _selected_index(self, teff: float) -> int:
        hits = np.flatnonzero(np.abs(self.teffs - teff) <= 0.5)
        if len(hits) != 1:
            raise LibraryError(
                f"Teff {teff:g} K is not a selected library temperature; selected: {self.teffs.tolist()}"
            )
        return int(hits[0])

    def _slice_by_index(self, i: int) -> np.ndarray:
        """Slice of the i-th selected Teff, read on first use and cached (LIB-2)."""
        arr = self._cache.get(i)
        if arr is None:
            arr = read_slice(self.libdir, self.info, float(self.teffs[i]))
            arr.flags.writeable = False  # shared by every caller
            self._cache[i] = arr
            self.reads += 1
        return arr

    def slice(self, teff: float) -> np.ndarray:
        """The stored slice (NB, NP, 2, NL) float32 of a selected library Teff (read-only, cached)."""
        return self._slice_by_index(self._selected_index(teff))

    # ---- Teff interpolation
    def check_teff(self, teff: float) -> None:
        """LIB-7: raise LibraryError naming the covered range if `teff` lies outside the selected temperatures."""
        lo, hi = self.teff_min, self.teff_max
        if not np.isfinite(teff) or teff < lo - 0.5 or teff > hi + 0.5:
            raise LibraryError(
                f"Teff {teff:g} K is outside the library range {lo:g} to {hi:g} K "
                f"(selected temperatures {self.teffs.tolist()})"
            )

    def interp_weights(self, teff: float) -> tuple[tuple[int, float], tuple[int, float]]:
        """Interpolation weights in Teff: ((i0, w0), (i1, w1)), with i0, i1 indices into `self.teffs`.

        at_teff(teff) == w0 * slice(teffs[i0]) + w1 * slice(teffs[i1]) (in float64). The forward model can
        apply the same weights to records it precomputed per slice (for example convolved spectra), since the
        interpolation is linear in the slice data. The weights are linear in ln Teff (LIB-3). At a selected Teff (within `SLICE_SNAP_REL` x Teff, i.e. rounding):
        ((i, 1.0), (i, 0.0)), the stored slice exactly. Up to 0.5 K beyond the first or last selected Teff (the
        tolerance of `check_teff`): that edge slice. Further outside: LibraryError naming the range (LIB-7).

        The weights are continuous in Teff: the snap onto a slice is a rounding tolerance, not a 0.5 K window. A 0.5 K
        window made the weights constant there, so a finite difference of `search.fd_step` (a Teff half-step of 0.05 K
        at a step of 500 K) saw no change within about 0.4 K of every slice.
        """
        self.check_teff(teff)
        t = self.teffs
        near = [i for i, ti in enumerate(t) if on_slice(teff, ti)]
        if len(near):
            i = int(near[0])
            return (i, 1.0), (i, 0.0)
        if teff < t[0] or teff > t[-1]:  # within check_teff's 0.5 K beyond an edge slice
            i = 0 if teff < t[0] else len(t) - 1
            return (i, 1.0), (i, 0.0)
        i1 = int(np.searchsorted(t, teff))  # t[i1-1] < teff < t[i1]
        i0 = i1 - 1
        x, x0, x1 = np.log(teff), np.log(t[i0]), np.log(t[i1])
        w1 = float((x - x0) / (x1 - x0))
        return (i0, 1.0 - w1), (i1, w1)

    def at_teff(self, teff: float) -> np.ndarray:
        """The library interpolated to `teff`: (NB, NP, 2, NL) float64 (see `interp_weights`)."""
        (i0, w0), (i1, w1) = self.interp_weights(teff)
        a = self._slice_by_index(i0).astype(np.float64)
        if w1 == 0.0:
            return a
        return w0 * a + w1 * self._slice_by_index(i1).astype(np.float64)

    # ---- wavelengths
    def check_wavelengths(self, lo: float, hi: float, margin_A: float = 0.0) -> None:
        """LIB-7: the fit window [lo, hi] (A) must lie at least `margin_A` (4 kernel sigma) inside the library range."""
        wmin, wmax = self.wavelength_min, self.wavelength_max
        if lo > hi:
            raise LibraryError(f"wavelength window {lo:g} to {hi:g} A is empty (lower end above upper end)")
        if lo - margin_A < wmin or hi + margin_A > wmax:
            raise LibraryError(
                f"wavelength window {lo:g} to {hi:g} A with a margin of {margin_A:g} A (4 kernel sigma) needs "
                f"{lo - margin_A:g} to {hi + margin_A:g} A; the library covers {wmin:g} to {wmax:g} A"
            )

    # ---- psi mirror (FORMATS.md section 1)
    def cos_psi_nodes(self) -> np.ndarray:
        """The NC = 2 NP - 1 values of cos psi from +1 (psi = 0) down to -1 (psi = 180).

        psi = 0 and 90 give exactly 1 and 0; the mirrored angles 180 - psi give exactly the negatives.
        """
        psis = self.info.psis
        c = np.cos(np.radians(psis))
        c[psis == 0.0] = 1.0
        c[psis == 90.0] = 0.0
        return np.concatenate([c, -c[-2::-1]])

    def mirror_slice(self, slice_: np.ndarray) -> np.ndarray:
        """(NB, NP, 2, L) -> (NB, NC, 2, L) on the nodes of `cos_psi_nodes`: I(180 - psi) = I(psi),
        V(180 - psi) = -V(psi), and V = 0 exactly at psi = 90 (the stored V record there is ignored).
        """
        s = np.asarray(slice_)
        npsi = len(self.info.psis)
        if s.ndim != 4 or s.shape[1] != npsi or s.shape[2] != 2:
            raise LibraryError(f"mirror_slice: expected shape (NB, {npsi}, 2, L), got {s.shape}")
        out = np.empty((s.shape[0], 2 * npsi - 1, 2, s.shape[3]), dtype=s.dtype)
        out[:, :npsi] = s
        out[:, npsi - 1, 1] = 0.0
        out[:, npsi:, 0] = s[:, npsi - 2 :: -1, 0]
        out[:, npsi:, 1] = -s[:, npsi - 2 :: -1, 1]
        return out
