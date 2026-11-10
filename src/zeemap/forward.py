"""Forward model: parameters -> model I and V/I on the data grid, for a batch.

Requirements FWD-1 to FWD-6, FWD-10, FWD-11, LIB-5, LIB-6 (docs/SPEC.md); design docs/DESIGN.md sections 5b and 7.
Interface: docs/DEV_INTERFACES.md, "Forward model".

Once, in `Forward.__init__`, for every selected Teff slice: the stored records (NB, NP, 2, L) are convolved with the
instrumental Gaussian and resampled to the data grid by one matrix T (M, L) = resampling @ convolution, then mirrored
to the NC = 2 NP - 1 cos-psi nodes (library.mirror_slice). The result, `records[t]` of shape (NB * NC, 2, M), is kept
in **float64** (C-contiguous, so `records[t].reshape(NB * NC, 2 * M)` is a view for the matrix product).

- Convolution (DESIGN.md section 5b): sigma = FWHM / 2.3548, cut at 4 sigma, normalised to a sum of 1, the ends
  padded with zeros. Lag l: out[j] = sum_l k(l) in[j - l].
- Resampling: each library value stands for a cell between the midpoints to its neighbours; each data pixel is the
  overlap-weighted mean over [lambda - d/2, lambda + d/2] (exact bin average). Pixels whose bin is not fully covered
  by library cells are flagged in `data_pixels_in_library` and the model there is NaN.
- LIB-7: the fitted pixels of the data set must lie at least 4 kernel sigma inside the library range, else
  LibraryError (a ValueError) naming the ranges.

Records as two factors: each slice's (NB * NC, 2 M) records matrix is factored once, R = A B with A
orthonormal (nodes x r), when r components hold every Stokes part to `COMPRESS_TOL` relative (Frobenius, over the
finite pixels) and the two thin products cost at most half the full one (`factor_records`; truncated SVD of the
matrix with each part scaled to unit norm, so a V part much smaller than I is held to the same relative accuracy).
The records are then REPLACED by A B, so every path (the disc sum, likelihood.py's pre-whitened records, the
reference model of V5a, diagnostics) uses the same model; it differs from the stored library by less than the
float32 rounding of the stored values (2^-24 / sqrt(3), about 3.4e-8 relative rms). `W @ R` becomes `(W @ A) @ B`.
Factors are cached per process by the records' bytes (the kernel stage and the swaps rebuild the same records).

Per evaluation of a batch Theta (K, dim): the field at the patch centres for all K at once;
observers for every phase (data phases + `model.phase_offset`); the numba scatter (kernels.py) gives node weights
W (K * P, NB * NC) and the disc sum is ONE matrix product W @ R per Teff slice touched; a batch is whatever the caller
passes, never pooled here (the caller owns the parallelism). V/I = V / I where I != 0, else 0 (FWD-5). Teff per row
(optional): the per-slice products are combined with library.interp_weights ("log") or the "planck" weights.
`node_weights_batch` returns the node-weight stage alone (node weights, per-slice Teff coefficients, invalid rows),
for likelihood.py's pre-whitened-record path (`Objective.neg_log_l_nodes`); `spectra` is that stage plus the products.

Whole-surface outside fraction (the prior cut of DESIGN.md section 3, used by problem.py): the node stage also returns,
per row, f = sum over ALL patches (visible or not) of a_i [expit((|B_i| - b_max) / s) + expit((b_min - |B_i|) / s)],
a_i = area_i / total area, s = SMOOTH_WIDTH_MG / 4 = 0.5 MG (each logistic step rises from 12 % to 88 % over
SMOOTH_WIDTH_MG = 2 MG), and 1 for a patch whose |B| is not finite; |B| at the patch centres is the one the scatter
already has, so the field is evaluated once per row. `surface_outside` gives the same number (or the hard-edged
fraction) for callers outside the node stage.

Interpolation in B (`model.interp_b`, FWD-6, FWD-15): "linear" (default, bilinear in B and cos psi) or "cubic"
(cubic Hermite in B on up to 4 B nodes with data-independent three-point slopes, `hermite_slope_matrix`; C1 in B;
cos psi stays linear). "pchip" is accepted as an alias of "cubic": the Fritsch-Carlson monotonicity limiter is NOT
applied (it depends on each record's data and is not a fixed node weight), so the interpolant may overshoot between
nodes.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from scipy.special import expit

from zeemap import kernels
from zeemap.library import Library, LibraryError
from zeemap.surface import Surface

FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548...
CUT_SIGMAS = 4.0
C_KMS = 299792.458  # speed of light, km/s
PLANCK_X_AK = 1.438776877e8  # h c / k in A K (CODATA 2018: second radiation constant c2 = 1.438776877e-2 m K)
LD_COEFFS = {0: (1.0, 0.0), 1: (0.4, 0.6), 3: (0.7, 0.3)}  # LD(mu) = a + b mu (surface.limb_darkening)
OVERLAP_TOL = 1e-9  # of a library cell: overlaps below this count as none (coverage)
SMOOTH_WIDTH_MG = 2.0  # width of the logistic steps of the smoothed whole-surface outside fraction (prior cut)
COMPRESS_TOL = 3e-8  # relative truncation error of the factored records per Stokes part (module docstring)
_FACTORS: dict = {}  # factor_records cache: records bytes -> (A, B) or None; never released (DESIGN.md section 7)


def factor_records(R: np.ndarray):
    """(A (n, r), B (r, 2 M)) with A orthonormal and A @ B = R (n, 2, M) to COMPRESS_TOL relative in each Stokes part
    over the finite pixels (NaN pixel columns stay NaN in B), or None when no r with n r + r c <= n c / 2 (c finite
    columns) reaches that. Truncated SVD (single-threaded BLAS, so parent and pool workers get the same bits) of R
    with each part scaled to unit Frobenius norm: the discarded tail then bounds each part's relative error."""
    n, _, M = R.shape
    X = R.reshape(n, 2 * M)
    key = (X.shape, hashlib.sha1(np.ascontiguousarray(X).tobytes()).digest())
    if key in _FACTORS:
        return _FACTORS[key]
    fin = np.all(np.isfinite(X), axis=0)
    parts = [np.flatnonzero(fin[:M]), M + np.flatnonzero(fin[M:])]
    norms = [float(np.linalg.norm(X[:, j])) for j in parts]
    keep = [(j, nm) for j, nm in zip(parts, norms) if nm > 0.0]
    c = sum(j.size for j, _ in keep)
    out = None
    if keep and n * c > 0:
        from .search import single_threaded  # function level: search imports problem, which imports this module

        with single_threaded():
            U, s, Vt = np.linalg.svd(np.concatenate([X[:, j] / nm for j, nm in keep], axis=1), full_matrices=False)
        tail = np.append(np.sqrt(np.cumsum(s[::-1] ** 2))[::-1], 0.0)  # tail[r]: norm of the components r, r + 1, ...
        r = int(np.argmax(tail <= COMPRESS_TOL))
        if 0 < r and n * r + r * c <= 0.5 * n * c:
            B = np.full((r, 2 * M), np.nan)
            B[:, fin] = 0.0  # finite parts of zero norm
            a = 0
            for j, nm in keep:
                B[:, j] = s[:r, None] * Vt[:r, a:a + j.size] * nm
                a += j.size
            out = (np.ascontiguousarray(U[:, :r]), B)
    _FACTORS[key] = out
    return out


# --------------------------------------------------------------------------------------------------------------------
# Convolution and resampling (DESIGN.md section 5b)


def kernel(fwhm: float, step: float) -> np.ndarray:
    """Kernel values at lags -h .. h (in library steps), length 2 h + 1."""
    if fwhm < 0:
        raise ValueError(f"model.fwhm must be >= 0, got {fwhm}")
    if fwhm == 0:
        return np.array([1.0])
    s = fwhm / FWHM_TO_SIGMA / step
    h = int(math.floor(CUT_SIGMAS * s + 1e-9))
    lags = np.arange(-h, h + 1, dtype=float)
    k = np.exp(-(lags**2) / (2.0 * s * s))
    return k / k.sum()


def cell_edges(wavelengths: np.ndarray) -> np.ndarray:
    """Edges (L + 1,) of the cells the library values stand for: midpoints, half a spacing beyond the ends."""
    w = np.asarray(wavelengths, float)
    if len(w) < 2:
        raise ValueError("the library needs at least two wavelengths")
    mid = 0.5 * (w[1:] + w[:-1])
    return np.concatenate([[w[0] - (mid[0] - w[0])], mid, [w[-1] + (w[-1] - mid[-1])]])


def resampling_matrix(lib_wavelengths, data_wavelengths, d: float) -> tuple[np.ndarray, np.ndarray]:
    """(M, L) overlap-weighted bin-average matrix and the (M,) mask of pixels fully covered by library cells.

    Rows of uncovered pixels are NaN.
    """
    e = cell_edges(lib_wavelengths)
    lam = np.asarray(data_wavelengths, float)
    a, b = lam - 0.5 * d, lam + 0.5 * d
    cell = np.min(np.diff(e))
    tol = OVERLAP_TOL * cell
    inside = (a >= e[0] - tol) & (b <= e[-1] + tol)
    ov = np.clip(np.minimum(b[:, None], e[None, 1:]) - np.maximum(a[:, None], e[None, :-1]), 0.0, None)
    ov[ov <= tol] = 0.0
    tot = ov.sum(axis=1)
    R = np.full_like(ov, np.nan)
    ok = inside & (tot > 0)
    R[ok] = ov[ok] / tot[ok, None]
    return R, ok


def convolution_resampling_matrix(lib_wavelengths, grid, fwhm):
    """T (M, L) with model_on_grid = T @ library_record, and the coverage mask (M,)."""
    w = np.asarray(lib_wavelengths, float)
    L = len(w)
    step = (w[-1] - w[0]) / (L - 1)
    if not np.allclose(np.diff(w), step, rtol=1e-6, atol=0.0):
        raise ValueError("the convolution needs a library on a regular wavelength grid")
    Rs, ok = resampling_matrix(w, grid.wavelengths, grid.step)
    k = kernel(fwhm, step)
    h = (len(k) - 1) // 2
    # T[m, i] = sum_c Rs[m, c] Cv[c, i], Cv[c, i] = k(c - i) for 0 <= c, i < L (zero padding): T[:, i] += k(l) Rs[:, i + l]
    Rz = np.where(ok[:, None], Rs, 0.0)
    T = np.zeros_like(Rz)
    for j, kl in enumerate(k):
        l = j - h
        if l >= 0:
            T[:, : L - l] += kl * Rz[:, l:]
        else:
            T[:, -l:] += kl * Rz[:, : L + l]
    T[~ok] = np.nan
    return T, ok


# --------------------------------------------------------------------------------------------------------------------
# Interpolation in B with a continuous first derivative (FWD-15)

INTERP_B = {"linear": "linear", "cubic": "cubic", "pchip": "cubic"}


def hermite_slope_matrix(bnodes) -> np.ndarray:
    """(NB, NB) matrix S with the node slopes m = S @ f of the cubic Hermite interpolant in B ("cubic", FWD-15).

    Data-independent three-point slopes (the derivative of the parabola through three neighbouring nodes):
      interior: m_i = (h_i-1 d_i + h_i d_i-1) / (h_i-1 + h_i),   d_i = (f_i+1 - f_i) / h_i, h_i = x_i+1 - x_i;
      ends (one-sided, through the three end nodes): m_0 = ((2 h_0 + h_1) d_0 - h_0 d_1) / (h_0 + h_1), mirrored at
      the last node; with two nodes only, m = d_0 (the interpolant is then linear).
    These are the starting slopes of Fritsch & Carlson (1980, SIAM J. Numer. Anal. 17, 238) [verify] WITHOUT their
    monotonicity limiter: the limiter depends on the data of each record and cannot be written as fixed node weights,
    which the scatter (one GEMM per Teff slice) needs. The interpolant is therefore C1 and exact for data quadratic in
    B, but not monotone: between nodes it may overshoot a monotone run of data. "pchip" is accepted as an alias.
    """
    x = np.asarray(bnodes, dtype=np.float64)
    n = len(x)
    S = np.zeros((n, n))
    h = np.diff(x)
    if n == 2:
        S[:, 0], S[:, 1] = -1.0 / h[0], 1.0 / h[0]
        return S

    def dd(i):  # coefficients of d_i on the nodes i, i + 1
        return {i: -1.0 / h[i], i + 1: 1.0 / h[i]}

    def put(row, a, i, b, j):  # S[row] = a d_i + b d_j
        for jj, v in dd(i).items():
            S[row, jj] += a * v
        for jj, v in dd(j).items():
            S[row, jj] += b * v

    for i in range(1, n - 1):
        H = h[i - 1] + h[i]
        put(i, h[i - 1] / H, i, h[i] / H, i - 1)
    put(0, (2 * h[0] + h[1]) / (h[0] + h[1]), 0, -h[0] / (h[0] + h[1]), 1)
    put(n - 1, (2 * h[-1] + h[-2]) / (h[-1] + h[-2]), n - 2, -h[-1] / (h[-1] + h[-2]), n - 3)
    return S


# --------------------------------------------------------------------------------------------------------------------


def _get(d, key, default):
    return d.get(key, default) if isinstance(d, dict) else default


class Forward:
    """The forward model (DEV_INTERFACES.md). See the module docstring for what is precomputed and how."""

    def __init__(self, library: Library, surface: Surface, field, dataset, cfg: dict):
        mcfg = cfg.get("model", {}) or {}
        lcfg = cfg.get("library", {}) or {}
        self.library, self.surface, self.field, self.dataset = library, surface, field, dataset
        self.interp_b = _get(mcfg, "interp_b", "linear")
        self.ld_mode = int(_get(mcfg, "ld_mode", 3))
        self.fwhm = float(mcfg["fwhm"])
        self.phase_offset = float(_get(mcfg, "phase_offset", 0.0))
        self.outside = _get(lcfg, "outside", "clamp")
        self.teff_mode = _get(mcfg, "interp_teff", "planck")
        if self.teff_mode not in ("log", "planck"):
            raise ValueError(f"model.interp_teff {self.teff_mode!r}: use 'planck' or 'log'")
        if self.interp_b not in INTERP_B:
            raise ValueError(f"model.interp_b {self.interp_b!r}: use 'linear' or 'cubic' ('pchip' = 'cubic')")
        self.interp_b = INTERP_B[self.interp_b]  # "pchip" -> "cubic" (no monotone limiter; see hermite_slope_matrix)
        if self.outside not in ("clamp", "reject"):
            raise ValueError(f"library.outside {self.outside!r}: use 'clamp' or 'reject'")
        if self.ld_mode not in LD_COEFFS:
            raise ValueError(f"model.ld_mode {self.ld_mode!r}: use 0, 1 or 3")
        if "u" not in field.names:
            raise ValueError(f"field model {field.name!r} has no parameter 'u' (cos i)")
        self.iu = field.names.index("u")
        self.ld_a, self.ld_b = LD_COEFFS[self.ld_mode]

        # nodes
        self.bnodes = np.ascontiguousarray(library.info.bs, dtype=np.float64)
        self.cnodes = np.ascontiguousarray(library.cos_psi_nodes(), dtype=np.float64)
        if len(self.bnodes) < 2:
            raise ValueError("the library needs at least two B values")
        self.NB, self.NC = len(self.bnodes), len(self.cnodes)
        self.mus = np.ascontiguousarray(library.info.mus, dtype=np.float64)
        self.NMU = len(self.mus) if len(self.mus) > 1 else 1
        if self.NMU > 1 and self.interp_b != "linear":
            raise ValueError("a mu-grid library (LIB-8) needs model.interp_b = 'linear'")
        if self.NMU > 1:  # the library holds the angular dependence: no limb-darkening law (brightness weight area x mu)
            self.ld_a, self.ld_b = 1.0, 0.0
        self.bslopes = np.ascontiguousarray(hermite_slope_matrix(self.bnodes)) if self.interp_b == "cubic" else None

        # LIB-7: fitted pixels at least 4 kernel sigma inside the library range
        grid = dataset.grid
        lam = grid.wavelengths
        self.sigma_A = self.fwhm / FWHM_TO_SIGMA
        pix = [s.pixels for s in dataset.spectra if len(s.pixels)]
        if pix:
            allpix = np.concatenate(pix)
            library.check_wavelengths(float(lam[allpix.min()]), float(lam[allpix.max()]), CUT_SIGMAS * self.sigma_A)

        # wavelength shift (FWD-13, decision #27; DESIGN.md section 5b): applied to the library wavelengths before the
        # convolution and resampling matrix is built; a free shift gets records at `model.shift_nodes` shift values
        # over its range, interpolated per row with cubic Lagrange weights. The data are never interpolated.
        pcfg = (cfg.get("parameters") or {}).get("shift")
        self.shift_unit = _get(mcfg, "shift_unit", "kms")
        if self.shift_unit not in ("kms", "A"):
            raise ValueError(f"model.shift_unit {self.shift_unit!r}: use 'kms' or 'A'")
        if pcfg is not None and pcfg.get("free"):
            nv = int(_get(mcfg, "shift_nodes", 5))
            if nv < 2:
                raise ValueError("model.shift_nodes must be at least 2")
            self.shift_nodes = np.linspace(float(pcfg["lo"]), float(pcfg["hi"]), nv)
        else:
            self.shift_nodes = None
        shifts = [float(pcfg["value"]) if pcfg is not None else 0.0] if self.shift_nodes is None else list(
            self.shift_nodes)
        self.shift_fixed = None if self.shift_nodes is not None else shifts[0]
        # records: convolved, resampled, mirrored, per selected Teff (and per shift node: index t * n_shift + j)
        self.M = grid.npix
        self.grid_wavelengths = np.asarray(lam, dtype=np.float64)  # data grid (A), for the "planck" Teff weights
        self.n_teff = len(library.teffs)
        self.n_shift = len(shifts)
        recs = []
        oks = []
        mats = []
        for v in shifts:
            lam_lib = self.shifted_wavelengths(library.info.wavelengths, v)
            T, ok = convolution_resampling_matrix(lam_lib, grid, self.fwhm)
            oks.append(ok)
            mats.append(np.ascontiguousarray(T[ok].T))  # (L, M_ok)
        ok = np.logical_and.reduce(oks)
        self.data_pixels_in_library = ok
        for t in library.teffs:
            s = np.asarray(library.slice(float(t)), dtype=np.float64)
            if s.ndim == 5:  # a mu-grid library (LIB-8): mu is the outer index of the record
                s = s.reshape(-1, *s.shape[2:])
            nb, npsi, two, L = s.shape
            for okv, Tok in zip(oks, mats):
                r = np.full((nb, npsi, 2, self.M), np.nan)
                r[..., okv] = (s.reshape(-1, L) @ Tok).reshape(nb, npsi, 2, -1)
                r[..., ~ok] = np.nan
                recs.append(np.ascontiguousarray(library.mirror_slice(r).reshape(nb * self.NC, 2, self.M)))
        self.records = np.stack(recs)  # (n_teff * n_shift, [NMU *] NB * NC, 2, M) float64
        self.factors = [factor_records(r) for r in self.records]  # (A, B) or None per record set
        for i, f in enumerate(self.factors):
            if f is not None:
                self.records[i] = (f[0] @ f[1]).reshape(self.records.shape[1:])
        self._flat = [self.records[i].reshape(self.records.shape[1], 2 * self.M) for i in range(len(recs))]

        # geometry
        self.phases = np.asarray(dataset.phases, dtype=float) + self.phase_offset
        self.centres = np.ascontiguousarray(surface.centres, dtype=np.float64)
        self.areas = np.ascontiguousarray(surface.areas, dtype=np.float64)
        self.area_fractions = self.areas / self.areas.sum()  # whole-surface weights of the outside fraction

    # ---------------------------------------------------------------------------------------------------- pieces

    def _theta(self, Theta) -> np.ndarray:
        Theta = np.atleast_2d(np.asarray(Theta, dtype=float))
        if Theta.shape[1] != self.field.dim:
            raise ValueError(f"Theta has {Theta.shape[1]} columns; field model {self.field.name!r} takes {self.field.dim}")
        return Theta

    def observers(self, Theta) -> tuple[np.ndarray, np.ndarray]:
        """(K, P, 3) observer directions for each row (u = cos i) and phase; (K,) bool: u outside [-1, 1]."""
        u = Theta[:, self.iu]
        bad = ~(np.abs(u) <= 1.0)
        uc = np.clip(np.nan_to_num(u), -1.0, 1.0)
        si = np.sqrt(np.maximum(0.0, 1.0 - uc * uc))
        a = 2.0 * np.pi * self.phases
        obs = np.empty((Theta.shape[0], len(a), 3))
        obs[:, :, 0] = si[:, None] * np.cos(a)[None, :]
        obs[:, :, 1] = -si[:, None] * np.sin(a)[None, :]
        obs[:, :, 2] = -uc[:, None]
        return obs, bad

    def field_values(self, Theta) -> np.ndarray:
        """(K, N, 3): field at the patch centres."""
        with np.errstate(all="ignore"):
            return np.ascontiguousarray(self.field.evaluate_batch(self.centres, Theta))

    def _scatter(self, Theta):
        """Node weights W (K * P, NB * NC), outside weight and total visible weight (K, P), bad (K,), and the field
        |B| at the patch centres (K, N) (for the whole-surface outside fraction)."""
        K, P = Theta.shape[0], len(self.phases)
        obs, bad = self.observers(Theta)
        bad = bad | np.array([not self.field.valid(th) for th in Theta], dtype=bool)
        Bc = self.field_values(Theta)
        W = np.zeros((K * P, self.NMU * self.NB * self.NC))
        wout = np.zeros((K, P))
        wtot = np.zeros((K, P))
        reject = self.outside == "reject"
        if self.NMU > 1:  # a mu-grid library (LIB-8)
            self._wmu = np.zeros((K, P))
            kernels.scatter_interp_mu(Bc, self.centres, self.areas, obs, self.bnodes, self.cnodes, self.mus, reject, W,
                                      wout, wtot, self._wmu, bad)
        elif self.interp_b == "cubic":
            kernels.scatter_interp_cubic(Bc, self.centres, self.areas, obs, self.ld_a, self.ld_b, self.bnodes,
                                         self.bslopes, self.cnodes, reject, W, wout, wtot, bad)
        else:
            kernels.scatter_interp(Bc, self.centres, self.areas, obs, self.ld_a, self.ld_b, self.bnodes, self.cnodes,
                                   reject, W, wout, wtot, bad)
        return W, wout, wtot, bad, _bmag(Bc)

    def _surface_fraction(self, bmag, smooth: bool = True) -> np.ndarray:
        """(K,) whole-surface area fraction with centre |B| (K, N) outside [b_min, b_max] of the library: logistic
        steps of scale SMOOTH_WIDTH_MG / 4 (smooth) or hard edges; a non-finite |B| counts as outside."""
        lo, hi = self.library.b_min, self.library.b_max
        if smooth:
            s = SMOOTH_WIDTH_MG / 4.0
            f = expit((bmag - hi) / s) + expit((lo - bmag) / s)
            f = np.where(np.isfinite(bmag), f, 1.0)
        else:
            f = ~np.isfinite(bmag) | (bmag < lo) | (bmag > hi)
        return f @ self.area_fractions

    def surface_outside(self, Theta, smooth: bool = True) -> np.ndarray:
        """(K,) whole-surface outside fraction for rows Theta (K, field.dim), smoothed as in the node stage (the
        prior-cut penalty) or hard-edged (smooth=False; the prior cut of the starts). One field evaluation at the
        patch centres per row."""
        Theta = self._theta(Theta)
        with np.errstate(all="ignore"):
            B = self.field.evaluate_batch(self.centres, Theta)
        return self._surface_fraction(_bmag(B), smooth)

    def shifted_wavelengths(self, lam: np.ndarray, v: float) -> np.ndarray:
        """The library wavelengths shifted by v: a velocity in km/s multiplies them by (1 + v / c), a constant in A adds."""
        lam = np.asarray(lam, dtype=np.float64)
        if v == 0.0:
            return lam
        return lam * (1.0 + v / C_KMS) if self.shift_unit == "kms" else lam + v

    def shift_weights(self, shift) -> np.ndarray:
        """(K, n_shift) cubic Lagrange weights of the shift nodes per row (exact at a node; the 4 nearest nodes, or
        linear with 2-3 nodes); rows outside the node range are clamped to the nearest end."""
        v = np.atleast_1d(np.asarray(shift, dtype=float))
        nodes = self.shift_nodes
        n = nodes.size
        out = np.zeros((v.size, n))
        for k, x in enumerate(v):
            x = min(max(float(x), nodes[0]), nodes[-1])
            i = int(np.clip(np.searchsorted(nodes, x) - 1, 0, n - 2))
            if n < 4:
                t = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
                out[k, i], out[k, i + 1] = 1.0 - t, t
                continue
            a = int(np.clip(i - 1, 0, n - 4))
            idx = range(a, a + 4)
            for j in idx:
                w = 1.0
                for m in idx:
                    if m != j:
                        w *= (x - nodes[m]) / (nodes[j] - nodes[m])
                out[k, j] = w
        return out

    def planck_weight(self, teff: float, t0: float, t1: float) -> np.ndarray:
        """(M,) continuum-matched weight of slice t1 at Teff `teff` between slices t0 < t1 on the data grid (DESIGN.md
        section 5c): w = (P(T) - P(T0)) / (P(T1) - P(T0)) with P the Planck function B_lambda; the factor 2 h c^2 /
        lambda^5 cancels, so P is replaced by 1 / (exp(x / T) - 1), x = h c / (lambda k) (expm1 for accuracy)."""
        x = PLANCK_X_AK / self.grid_wavelengths  # K
        p = [1.0 / np.expm1(x / float(t)) for t in (teff, t0, t1)]
        return (p[0] - p[1]) / (p[2] - p[1])

    def _coeffs(self, K, teff, shift):
        """Coefficients of the record sets (n_teff * n_shift, K[, M]): the Teff coefficients times the shift weights."""
        C, bad = self._teff_coeffs(K, teff)
        if self.shift_nodes is None:
            return C, bad
        Ws = self.shift_weights(np.zeros(K) if shift is None else np.broadcast_to(np.asarray(shift, float).reshape(-1),
                                                                                  (K,)))
        nt, nv = C.shape[0], Ws.shape[1]
        out = np.zeros((nt * nv, *C.shape[1:]))
        for t in range(nt):
            for j in range(nv):
                w = Ws[:, j] if C.ndim == 2 else Ws[:, j, None]
                out[t * nv + j] = C[t] * w
        return out, bad

    def _teff_coeffs(self, K, teff):
        """(n_teff, K) coefficients of the slices per row, and (K,) bool rows whose Teff is outside the selection. With
        `teff_mode` "planck" and a row between two slices the coefficients are per data pixel: (n_teff, K, M)."""
        nt = len(self.library.teffs)
        C = np.zeros((nt, K))
        bad = np.zeros(K, dtype=bool)
        if teff is None:
            if nt != 1:
                raise ValueError(
                    f"{nt} Teff slices are selected ({self.library.teffs.tolist()}); pass teff per row or select one"
                )
            C[0] = 1.0
            return C, bad
        teff = np.broadcast_to(np.asarray(teff, dtype=float).reshape(-1), (K,))
        planck = self.teff_mode == "planck"
        Cp = None
        for k, t in enumerate(teff):
            try:
                (i0, w0), (i1, w1) = self.library.interp_weights(float(t))
            except LibraryError:
                bad[k] = True
                continue
            if planck and w1 != 0.0:
                if Cp is None:
                    Cp = np.repeat(C[:, :, None], self.M, axis=2)
                w = self.planck_weight(float(t), float(self.library.teffs[i0]), float(self.library.teffs[i1]))
                Cp[i0, k] += 1.0 - w
                Cp[i1, k] += w
                continue
            C[i0, k] += w0
            C[i1, k] += w1
            if Cp is not None:
                Cp[i0, k] += w0
                Cp[i1, k] += w1
        return (C if Cp is None else Cp), bad

    def _product(self, W, s):
        """W @ records of set s (rows, 2 M), as (W @ A) @ B when the set is factored."""
        f = self.factors[s]
        return W @ self._flat[s] if f is None else (W @ f[0]) @ f[1]

    def _disc_sum(self, W, C, K, P):
        """sum over slices of C[s, k] * (W @ R_s): one GEMM per slice touched -> (K, P, 2, M). C (n_teff, K), or per
        data pixel (n_teff, K, M) ("planck" Teff weights between slices), applied to I and V alike."""
        if C.ndim == 3:
            S = np.zeros((K * P, 2, self.M))
            for s in range(C.shape[0]):
                c = np.repeat(C[s], P, axis=0)  # (K * P, M)
                rows = np.flatnonzero(np.any(c != 0.0, axis=1))
                if rows.size:
                    S[rows] += c[rows, None, :] * self._product(W[rows], s).reshape(rows.size, 2, self.M)
            return S.reshape(K, P, 2, self.M)
        S = np.zeros((K * P, 2 * self.M))
        for s in range(C.shape[0]):
            c = np.repeat(C[s], P)  # per row of W
            if not np.any(c):
                continue
            if np.all(c == 1.0):
                S += self._product(W, s)
            else:
                rows = np.flatnonzero(c)
                S[rows] += c[rows, None] * self._product(W[rows], s)
        return S.reshape(K, P, 2, self.M)

    def _finish(self, S, bad):
        I = S[:, :, 0, :]
        V = S[:, :, 1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            VI = np.where(I != 0.0, V / np.where(I != 0.0, I, 1.0), 0.0)
        VI[np.isnan(I)] = np.nan
        I = I.copy()
        I[bad] = np.nan
        VI[bad] = np.nan
        return I, VI

    # ---------------------------------------------------------------------------------------------------- public

    def _node_stage(self, Theta, teff, shift):
        """The node-weight stage shared by `spectra` and `node_weights_batch`: Theta (K, field.dim), W (K * P, NB * NC),
        C (n_teff, K), outside and total visible weight (K, P), invalid (K,), smoothed whole-surface outside fraction
        (K,) (module docstring; computed for every row, invalid ones included)."""
        Theta = self._theta(Theta)
        if shift is not None and self.shift_nodes is None and np.any(np.asarray(shift) != (self.shift_fixed or 0.0)):
            raise ValueError("a shift per row needs parameters.shift free (the fixed shift is built in)")
        C, tbad = self._coeffs(Theta.shape[0], teff, shift)
        out = self._scatter(Theta)
        W, wout, wtot, bad = out[:4]
        # the |B| the scatter already has; a subclass whose _scatter returns only four items (forward_reference.py)
        # gets the fraction from a separate field evaluation
        fsurf = self._surface_fraction(out[4]) if len(out) > 4 else self.surface_outside(Theta)
        return Theta, W, C, wout, wtot, bad | tbad, fsurf

    def spectra(self, Theta, teff=None, shift=None):
        """Theta (K, dim) -> I (K, P, M), VI (K, P, M), invalid (K,) bool. Invalid rows are NaN."""
        return self.spectra_with_outside(Theta, teff, shift)[:3]

    def spectra_with_outside(self, Theta, teff=None, shift=None):
        """`spectra` plus the smoothed whole-surface outside fraction (K,) of the same field evaluation."""
        Theta, W, C, _, _, bad, fsurf = self._node_stage(Theta, teff, shift)
        K, P = Theta.shape[0], len(self.phases)
        return (*self._finish(self._disc_sum(W, C, K, P), bad), bad, fsurf)

    def node_weights_batch(self, Theta, teff=None, shift=None):
        """The node-weight stage of `spectra` alone, for likelihoods that take node weights (likelihood.py
        `Objective.neg_log_l_nodes`, the pre-whitened-record path; decision #15).

        Returns Wnodes (K, P, NB * NC) (row k, phase p: the brightness-weighted area per library node, the matrix that
        `spectra` multiplies with `records[t]`), slice_weights = [(t, (K,) coefficients)] for the Teff slices with a
        non-zero coefficient in some row (the per-row interpolation weights; `spectra` = sum over them of
        coefficient x Wnodes @ records[t], then V/I), invalid (K,) bool, the visible outside fraction (K, P) as in
        `outside_fraction`, and the smoothed whole-surface outside fraction (K,) (module docstring; the prior cut).
        """
        Theta, W, C, wout, wtot, bad, fsurf = self._node_stage(Theta, teff, shift)
        K, P = Theta.shape[0], len(self.phases)
        weights = [(t, C[t]) for t in range(C.shape[0]) if np.any(C[t])]  # C[t]: (K,), or (K, M) per pixel
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(wtot > 0, wout / wtot, 0.0)
        return W.reshape(K, P, W.shape[1]), weights, bad, frac, fsurf

    def node_weights(self, theta) -> np.ndarray:
        """(P, NB, NC) brightness-weighted area per library node for one parameter set."""
        W = self._scatter(self._theta(theta)[:1])[0]
        return W.reshape(len(self.phases), self.NMU, self.NB, self.NC).sum(axis=1)  # summed over mu (LIB-8)

    def mu_below_fraction(self, theta) -> np.ndarray:
        """(P,) brightness-weighted visible area fraction below the smallest library mu (mu-grid libraries, LIB-8)."""
        if self.NMU <= 1:
            return np.zeros(len(self.phases))
        _, _, wtot = self._scatter(self._theta(theta)[:1])[:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(wtot[0] > 0, self._wmu[0] / wtot[0], 0.0)

    def outside_fraction(self, theta) -> np.ndarray:
        """(P,) brightness-weighted visible area fraction whose centre |B| lies outside the library (LIB-6)."""
        _, wout, wtot = self._scatter(self._theta(theta)[:1])[:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(wtot[0] > 0, wout[0] / wtot[0], 0.0)

    def b_quantiles(self, theta, q=(0.1, 0.5, 0.9)) -> np.ndarray:
        """(P, len(q)) quantiles of the brightness-weighted |B| distribution over the visible patch centres.

        Weighted quantile: |B| sorted, each patch at the midpoint of its share of the cumulative weight, linear in between.
        """
        Theta = self._theta(theta)[:1]
        obs, _ = self.observers(Theta)
        with np.errstate(all="ignore"):
            B = self.field.evaluate_batch(self.centres, Theta)[0]
        bmag = np.sqrt(np.einsum("ij,ij->i", B, B))
        mu = obs[0] @ self.centres.T  # (P, N)
        out = np.empty((len(self.phases), len(q)))
        for p in range(len(self.phases)):
            vis = mu[p] > 0.0
            w = self.areas[vis] * mu[p, vis] * (self.ld_a + self.ld_b * mu[p, vis])
            b = bmag[vis]
            o = np.argsort(b, kind="stable")
            b, w = b[o], w[o]
            cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
            out[p] = np.interp(np.asarray(q, float), cdf, b)
        return out


def _bmag(B) -> np.ndarray:
    """|B| (K, N) of field vectors (K, N, 3) (a C-contiguous copy first, so strided views sum in the same order)."""
    B = np.ascontiguousarray(B)
    return np.sqrt(np.einsum("kni,kni->kn", B, B))
