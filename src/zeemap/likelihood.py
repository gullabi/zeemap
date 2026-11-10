"""The likelihood: plain chi2 and the full likelihood (DESIGN.md section 5; SPEC LIKE-1 to LIKE-4, LIKE-6 to LIKE-9).

Configuration: the dict ``cfg["objective"]`` (FORMATS.md section 3, keys ``objective.*``); missing keys take
the defaults of ``DEFAULTS`` below.

Notation for one spectrum s (one phase and one Stokes quantity; n_s fitted pixels, data y, errors sigma, model m):

- kernel K_s = M(rho_short) + a^2 M(rho_long) + j^2 I, with the Matern-3/2 kernel
  M_ij = (1 + sqrt(3) d_ij / rho) exp(-sqrt(3) d_ij / rho), d_ij = |lambda_i - lambda_j| in A; a, rho_long from
  ``disc`` (per Stokes kind), rho_short and j from ``corr``. ``corr = false`` replaces M(rho_short) + j^2 I by the
  identity; ``disc = false`` drops the a^2 M(rho_long) term. Covariance C_s = sigma_s^2 D K_s D, D = diag(sigma).
- whitening W_s = L_s^-1 D^-1 with K_s = L_s L_s^T (so W_s^T W_s = (D K_s D)^-1).
- continuum (I spectra only): the model I times a Legendre series P_0 .. P_degree on the fitted pixels mapped linearly
  onto [-1, 1], plus a free constant when the spectrum's ``i_offset`` is set. Whitened design matrix
  A_s = [W_s (m * P_0), ..., W_s (m * P_degree), W_s 1]; k_s = its number of columns; k_s = 0 for V/I.
  ``continuum = false`` switches the whole continuum off, the constant included (k_s = 0 for I too): LIKE-3 defines the
  constant as part of the continuum, and LIKE-9 wants the continuum fixable as a whole.
- r_s = W_s (y - m); chi2_s = min over beta |r_s - A_s beta|^2 (the whitened residual after the best continuum). For I
  with a continuum (P_0 m is a column of A_s) this equals min over beta |W_s (y - c m) - A_s beta|^2 for any c, and
  is computed with c a power of two that brings the model to the data's size: chi2_s does not depend on the scale of
  the model; see `_fit_whitened`.
- normalised basis (decision #26, DESIGN.md section 5): the Legendre columns of A_s are divided by
  nu_s = |W_s m| / |W_s 1|, the size of the model measured with the whitening (it scales with the model and does not
  change when all errors are scaled), so that ln det(A_s^T A_s), and with it -ln L, does not change when the model I
  is multiplied by a constant (before, it moved by 2 (deg+1) ln(scale): the flat prior on the coefficients favoured
  faint models). The constant column (``i_offset``) is not divided.

-ln L per mode (``neg_log_l``; the parameter-independent ``constant`` is NOT included, it is exposed as an attribute):

- kind "full", scale on:  sum_s 0.5 [(n_s - k_s) ln chi2_s + ln det(A_s^T A_s)]
  (Jeffreys prior on sigma_s and flat priors on the continuum coefficients, integrated out in closed form).
- kind "full", scale off: sum_s 0.5 [chi2_s + ln det(A_s^T A_s)]  (sigma_s = 1, continuum integrated out).
- kind "chi2" (LIKE-6):   sum_s 0.5 chi2_s with W_s = diag(sqrt(weight) / sigma); scale, corr and disc are forced off
  and the continuum coefficients are fitted by linear least squares, with NO ln det term. This differs from "full" with
  every switch off, which keeps 0.5 ln det(A_s^T A_s) (the volume factor of the integrated continuum).

``constant`` = sum_s [0.5 ln det K_s + sum_i ln sigma_i] plus, with scale on,
sum_s [(n_s - k_s)/2 ln pi - ln Gamma((n_s - k_s)/2) + ln 2], and with scale off sum_s (n_s - k_s)/2 ln(2 pi);
``neg_log_l + constant`` is then the exact -ln of the marginal likelihood for unit-density (improper) priors
d ln sigma and d beta (V1g checks this against numerical integration). For kind "chi2", ``constant`` = 0.

Invalid models (the forward model's flag, a non-finite model value on a fitted pixel, or a non-finite result such as
a singular continuum design) get exactly ``WORST`` = 1e30 (LIKE-7), and ``n_invalid`` counts them (LIKE-8).

Weights (DATA-4, decision #27) act in every kind as the errors divided by sqrt(weight): W_s = diag(sqrt(weight) /
sigma) for "chi2", D = diag(sigma / sqrt(weight)) in the covariance of "full". Pixels of weight 0 are removed by the
data reader; a weight of 0 that reaches "full" is refused.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import cho_factor, solve_triangular
from scipy.special import gammaln

from .types import DataSet

WORST = 1e30

# The pre-whitened-record path (``bind_forward`` / ``neg_log_l_nodes``; decision #15): its
# precomputed matrices are used only below this size per process (ENG-6 allows 1 GB per worker; the matrices are
# built once per process and shared with forked workers). They are kept in float64 (agreement with the exact path
# ~1e-15 relative): float32 storage was measured on the toy star to move -ln L (kind full) by up to
# 1.8e-4 absolute, 1.3e-8 relative, noise of the size of a finite-difference step's change in -ln L.
FAST_PATH_MAX_BYTES = 1.5e9
FAST_PATH_DTYPE = np.float64
SCALE_RATIO_RANGE = (1e-3, 1e3)  # check_scale: median |model I| / median |data I| outside this is reported
FAST_GEMM_ROWS = 16  # rows at which a matrix product on one core turns from memory- to compute-bound (_prefer_pre)

DEFAULTS = {
    "kind": "full",
    "scale": True,
    "corr": {"rho": 4.0, "jitter": 0.05},
    "disc": {"a_I": 0.71, "rho_I": 85.0, "a_VI": 0.42, "rho_VI": 205.0},
    "continuum": {"degree": 3},
    "kernel_fit": True,  # used by the search (noise.py, DESIGN.md section 3 step 2b), not here
    "kernel": [],  # [{spectrum, rho_short, a, rho_long, j}]: a given kernel per spectrum label (LIKE-2)
}


def objective_settings(cfg) -> dict:
    """The objective settings with defaults filled in. ``cfg`` is the whole configuration (a dict with an "objective"
    table), or None for all defaults. ``true`` for corr, disc or continuum means the default table."""
    obj = dict((cfg or {}).get("objective", {}) or {})
    unknown = set(obj) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown key(s) objective.{', objective.'.join(sorted(unknown))}")
    out = copy.deepcopy(DEFAULTS)
    for key, val in obj.items():
        if key == "kernel":
            out[key] = [dict(k) for k in (val or [])]
            for k in out[key]:
                bad = set(k) - {"spectrum", "rho_short", "a", "rho_long", "j"}
                if bad or "spectrum" not in k:
                    raise ValueError(f"objective.kernel entry {k!r}: needs spectrum, rho_short, a, rho_long, j")
            continue
        if key in ("corr", "disc", "continuum"):
            if val is False or val is None:
                out[key] = False
            elif val is True:
                pass
            else:
                sub = dict(val)
                bad = set(sub) - set(DEFAULTS[key])
                if bad:
                    raise ValueError(f"unknown key(s) in objective.{key}: {sorted(bad)}")
                out[key].update(sub)
        else:
            out[key] = val
    if out["kind"] not in ("full", "chi2"):
        raise ValueError(f"objective.kind must be 'full' or 'chi2', not {out['kind']!r}")
    if out["continuum"] is not False:
        deg = out["continuum"]["degree"]
        if int(deg) != deg or deg < 0:
            raise ValueError(f"objective.continuum.degree must be an integer >= 0, not {deg!r}")
        out["continuum"]["degree"] = int(deg)
    return out


def matern32(lam: np.ndarray, rho: float) -> np.ndarray:
    """Matern-3/2 kernel matrix M_ij = (1 + sqrt(3) d / rho) exp(-sqrt(3) d / rho), d = |lam_i - lam_j|."""
    d = np.abs(lam[:, None] - lam[None, :])
    x = math.sqrt(3.0) * d / float(rho)
    return (1.0 + x) * np.exp(-x)


def legendre_x(lam: np.ndarray) -> np.ndarray:
    """The fitted pixels' wavelengths mapped linearly onto [-1, 1] (first pixel -> -1, last -> +1)."""
    if lam.size < 2 or lam[-1] == lam[0]:
        return np.zeros_like(lam)
    return 2.0 * (lam - lam[0]) / (lam[-1] - lam[0]) - 1.0


class _Spec:
    """Precomputed quantities of one spectrum (internal)."""

    __slots__ = ("spec", "n", "k", "K", "W", "wdiag", "y_w", "one_w", "basis", "deg", "offset", "logdet_K", "w1norm", "sig")


class Objective:
    """-ln L of a batch of model spectra against a DataSet (DEV_INTERFACES.md, "Likelihood").

    Attributes: ``kind``, ``settings`` (the filled-in objective table), ``constant`` (parameter independent, see the
    module docstring), ``n_invalid`` (invalid model rows seen by ``neg_log_l``), ``n_evals`` (rows evaluated),
    ``labels`` (spectrum labels, in the order of the columns of ``terms``).
    """

    def __init__(self, dataset: DataSet, cfg=None):
        st = objective_settings(cfg)
        self.settings = st
        self.kind = st["kind"]
        self.dataset = dataset
        self.n_invalid = 0
        self.n_evals = 0
        chi2 = self.kind == "chi2"
        self.scale = bool(st["scale"]) and not chi2
        corr = False if chi2 else st["corr"]
        disc = False if chi2 else st["disc"]
        cont = st["continuum"]
        given = {str(k["spectrum"]): k for k in st.get("kernel") or []}  # [[objective.kernel]] (LIKE-2)
        unmatched = sorted(set(given) - {sp.label for sp in dataset.spectra})
        if unmatched:  # silently ignored, a mistyped label would leave its spectrum with the initial kernel
            from .config import ConfigError

            raise ConfigError(f"objective.kernel: spectrum {', '.join(repr(u) for u in unmatched)} matches no spectrum "
                              f"of the data set (spectra: {', '.join(sp.label for sp in dataset.spectra)})")
        lam_all = dataset.grid.wavelengths
        self._specs: list[_Spec] = []
        self.labels = []
        const = 0.0
        for sp in dataset.spectra:
            if sp.kind not in ("I", "VI"):
                raise ValueError(f"spectrum {sp.label}: kind must be 'I' or 'VI', not {sp.kind!r}")
            pix = np.asarray(sp.pixels)
            y = np.asarray(sp.data, dtype=np.float64)
            sig = np.asarray(sp.sigma, dtype=np.float64)
            w = np.asarray(sp.weight, dtype=np.float64)
            n = pix.size
            if not (y.shape == sig.shape == w.shape == (n,)):
                raise ValueError(f"spectrum {sp.label}: pixels, data, sigma and weight differ in length")
            if not (np.all(np.isfinite(y)) and np.all(np.isfinite(sig)) and np.all(sig > 0)):
                raise ValueError(f"spectrum {sp.label}: data and errors must be finite and errors > 0")
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError(f"spectrum {sp.label}: weights must be finite and >= 0")
            if not chi2 and np.any(w <= 0.0):
                raise ValueError(f"spectrum {sp.label}: a weight of 0 reached the full likelihood (the data reader "
                                 "removes such pixels)")
            if not chi2:  # DATA-4: the weight scales the error, D = diag(sigma / sqrt(w))
                sig = sig / np.sqrt(w)
            lam = lam_all[pix]
            s = _Spec()
            s.spec, s.n = sp, n
            s.sig = sig  # the errors as used in D (divided by sqrt(weight) for kind full)
            s.deg = cont["degree"] if (cont is not False and sp.kind == "I") else -1
            s.offset = bool(sp.i_offset) and cont is not False and sp.kind == "I"
            s.k = (s.deg + 1) + int(s.offset)
            s.basis = legendre.legvander(legendre_x(lam), s.deg).T if s.deg >= 0 else None  # (deg+1, n)
            corr_s, disc_s = corr, disc
            if not chi2 and sp.label in given:  # a given (or frozen fitted) kernel for this spectrum
                g = given[sp.label]
                corr_s = {"rho": float(g["rho_short"]), "jitter": float(g["j"])}
                disc_s = {"a_I": float(g["a"]), "rho_I": float(g["rho_long"]), "a_VI": float(g["a"]),
                          "rho_VI": float(g["rho_long"])}
            if corr_s is False and disc_s is False:
                s.K = None  # identity
                s.wdiag = (np.sqrt(w) if chi2 else 1.0) / sig
                s.W = None
                s.logdet_K = 0.0
            else:
                K = np.eye(n) if corr_s is False else matern32(lam, corr_s["rho"]) + corr_s["jitter"] ** 2 * np.eye(n)
                if disc_s is not False:
                    a, rho_l = (disc_s["a_I"], disc_s["rho_I"]) if sp.kind == "I" else (disc_s["a_VI"], disc_s["rho_VI"])
                    K = K + a**2 * matern32(lam, rho_l)
                L, _ = cho_factor(K, lower=True)
                L = np.tril(L)
                s.K = K
                s.logdet_K = 2.0 * float(np.sum(np.log(np.diag(L))))
                # W = L^-1 D^-1 as an explicit matrix, so that whitening a batch is one GEMM per spectrum. The forward
                # model may later supply pre-whitened library records, which makes this step a pure GEMM on the
                # library side (decision #15).
                s.W = solve_triangular(L, np.diag(1.0 / sig), lower=True)
                s.wdiag = None
            s.y_w = self._whiten(s, y)
            w1 = self._whiten(s, np.ones(n))
            s.one_w = w1 if s.offset else None
            s.w1norm = math.sqrt(float(w1 @ w1))  # |W_s 1|, the reference size of the normalised basis (decision #26)
            m = n - s.k
            if m <= 0:
                raise ValueError(f"spectrum {sp.label}: {n} pixels for {s.k} continuum coefficients")
            if not chi2:
                const += 0.5 * s.logdet_K + float(np.sum(np.log(sig)))
                if self.scale:
                    const += 0.5 * m * math.log(math.pi) - gammaln(0.5 * m) + math.log(2.0)
                else:
                    const += 0.5 * m * math.log(2.0 * math.pi)
            self._specs.append(s)
            self.labels.append(sp.label)
        self.constant = float(const)
        self.fast_path = "off: bind_forward not called"
        self.fast_path_detail = self.fast_path
        self._bound = None
        self._fast = None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _whiten(s: _Spec, x: np.ndarray) -> np.ndarray:
        """W_s x for x of shape (n,) or (n, K)."""
        if s.W is None:
            return x * (s.wdiag if x.ndim == 1 else s.wdiag[:, None])
        return s.W @ x

    def whiten(self, index: int, x: np.ndarray) -> np.ndarray:
        """Apply spectrum ``index``'s whitening W_s to x of shape (n_s,) or (n_s, K)."""
        return self._whiten(self._specs[index], np.asarray(x, dtype=np.float64))

    def kernel(self, index: int) -> np.ndarray:
        """The kernel K_s of spectrum ``index`` (identity when corr and disc are off)."""
        s = self._specs[index]
        return np.eye(s.n) if s.K is None else s.K.copy()

    @property
    def n_params(self) -> list[int]:
        """k_s, the number of continuum coefficients of each spectrum."""
        return [s.k for s in self._specs]

    def _rows(self, s: _Spec, I: np.ndarray, VI: np.ndarray) -> np.ndarray:
        src = I if s.spec.kind == "I" else VI
        return src[:, s.spec.phase_index, :][:, s.spec.pixels]  # (K, n)

    def _fit(self, s: _Spec, mrows: np.ndarray):
        """Best continuum for model rows (K, n): whitened residual after it (K, n) and ln det(A^T A) (K,)."""
        K = mrows.shape[0]
        if s.deg >= 0:
            X = (mrows[:, None, :] * s.basis[None, :, :]).reshape(K * (s.deg + 1), s.n)
            WX = self._whiten(s, X.T).reshape(s.n, K, s.deg + 1).transpose(1, 0, 2)  # (K, n, deg+1)
            wm = WX[:, :, 0]  # P_0 = 1: the whitened model itself
        else:
            WX = None
            wm = self._whiten(s, mrows.T).T
        return self._fit_whitened(s, wm, WX)

    def _fit_whitened(self, s: _Spec, wm: np.ndarray, WX):
        """`_fit` from the whitened model wm = W_s m (K, n) and the whitened continuum columns WX = W_s (m P_j)
        (K, n, deg+1), or None without a Legendre continuum."""
        K = wm.shape[0]
        if s.k == 0:  # no continuum (V/I, or the continuum switched off): the residual is y_w - wm itself
            return s.y_w[None, :] - wm, np.zeros(K)
        # k_s > 0 implies a Legendre continuum (the offset alone is never fitted), so the P_0 column of A is wm itself
        # and min_beta |y_w - wm - A beta| = min_gamma |y_w - c wm - A gamma| for any constant c: the model may be
        # subtracted at any scale before span(A) is projected out. Subtracting it at its own scale would leave
        # ~eps |wm| per pixel of round-off, fatal for a model in physical flux units (|wm| ~ 5e16 against
        # |y_w| ~ 500). So each row's model columns are first scaled by a
        # power of two c = 2^-e, e = round(log2(|wm| / |y_w|)), so that they are of the size of the data (exact in
        # floating point: no new rounding, and c = 1, i.e. the plain arithmetic bit for bit, for a model already of
        # the data's size), and r = y_w - c wm is projected. ln det(A^T A) is corrected exactly: the deg+1 model
        # columns scaled by c multiply det(A^T A) by c^(2 (deg+1)); the offset column is not scaled.
        ynorm = math.sqrt(float(s.y_w @ s.y_w))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            wnorm = np.sqrt(np.einsum("kn,kn->k", wm, wm))
            ratio = wnorm / ynorm
            e = np.where(np.isfinite(ratio) & (ratio > 0.0), np.rint(np.log2(ratio)), 0.0)
        e = np.clip(e, -1000.0, 1000.0)
        if np.any(e != 0.0):
            WX = WX * np.ldexp(1.0, -e.astype(np.int64))[:, None, None]
        cols = [WX]
        if s.offset:
            cols.append(np.broadcast_to(s.one_w[None, :, None], (K, s.n, 1)))
        A = np.concatenate(cols, axis=2) if len(cols) > 1 else cols[0]
        Q, R = np.linalg.qr(A)  # batched reduced QR: (K, n, k), (K, k, k)
        r = s.y_w[None, :] - WX[:, :, 0]  # P_0 = 1: the (scaled) whitened model
        res = r - np.einsum("knj,kj->kn", Q, np.einsum("knj,kn->kj", Q, r))
        logdet = 2.0 * np.sum(np.log(np.abs(np.diagonal(R, axis1=1, axis2=2))), axis=1)
        # undo the power-of-two scaling, then normalise the deg+1 model columns by nu = |W m| / |W 1| (decision #26):
        # det(A^T A) of the columns divided by nu is det of the unscaled ones times nu^(-2 (deg+1)); in total
        # logdet(normalised) = logdet(scaled) + 2 (deg+1) (e ln 2 - ln nu)
        with np.errstate(divide="ignore", invalid="ignore"):
            lognu = np.log(wnorm / s.w1norm)
        logdet = logdet + 2.0 * (s.deg + 1) * (math.log(2.0) * e - lognu)
        return res, logdet

    def _term(self, s: _Spec, res: np.ndarray, logdet: np.ndarray) -> np.ndarray:
        chi2 = np.einsum("kn,kn->k", res, res)
        if self.kind == "chi2":
            return 0.5 * chi2
        if self.scale:
            return 0.5 * ((s.n - s.k) * np.log(chi2) + logdet)
        return 0.5 * (chi2 + logdet)

    def _terms_valid(self, I, VI) -> np.ndarray:
        K = I.shape[0]
        out = np.empty((K, len(self._specs)))
        for j, s in enumerate(self._specs):
            res, logdet = self._fit(s, self._rows(s, I, VI))
            out[:, j] = self._term(s, res, logdet)
        return out

    @staticmethod
    def _batch(I, VI):
        I = np.asarray(I, dtype=np.float64)
        VI = np.asarray(VI, dtype=np.float64)
        if I.ndim == 2:
            I, VI = I[None], VI[None]
        return I, VI

    # ---------------------------------------------------------------- public

    def terms(self, I, VI) -> np.ndarray:
        """Per-spectrum terms (K, S) of -ln L (without ``constant``), columns in the order of ``labels``. No validity
        handling: a bad model row gives inf or NaN here (``neg_log_l`` maps those to 1e30)."""
        I, VI = self._batch(I, VI)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self._terms_valid(I, VI)

    def neg_log_l(self, I, VI, invalid=None) -> np.ndarray:
        """-ln L (without ``constant``) for a batch: I, VI (K, P, M), invalid (K,) bool -> (K,). Invalid rows -> 1e30."""
        I, VI = self._batch(I, VI)
        K = I.shape[0]
        bad = np.zeros(K, dtype=bool) if invalid is None else np.asarray(invalid, dtype=bool).reshape(K).copy()
        for s in self._specs:
            bad |= ~np.all(np.isfinite(self._rows(s, I, VI)), axis=1)
        out = np.full(K, WORST)
        ok = np.flatnonzero(~bad)
        if ok.size:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                val = self._terms_valid(I[ok], VI[ok]).sum(axis=1)
            fin = np.isfinite(val)
            out[ok[fin]] = val[fin]
            bad[ok[~fin]] = True
        self.n_invalid += int(bad.sum())
        self.n_evals += K
        return out

    # ---------------------------------------------------------------- the pre-whitened-record path

    def _fast_layout(self):
        """Per phase with a pre-whitened spectrum: [(spectrum index, column offset, number of blocks)] and the column
        count. Pre-whitened are the I spectra with a dense whitening W_s (corr or disc on), in blocks W_s (m P_j),
        j = 0 .. deg (one block W_s m without a Legendre continuum). V/I is not linear in the records, and a diagonal
        W_s (kind "chi2", or corr and disc off) costs nothing to apply, so those spectra take I and V from the forward
        model's own records (a view of the node range, no copy) and are whitened as in the exact path."""
        groups: dict[int, list] = {}
        ncols: dict[int, int] = {}
        for j, s in enumerate(self._specs):
            if s.spec.kind != "I" or s.W is None:
                continue
            p = int(s.spec.phase_index)
            nblk = max(s.deg, 0) + 1
            groups.setdefault(p, []).append((j, ncols.get(p, 0), nblk))
            ncols[p] = ncols.get(p, 0) + nblk * s.n
        return [(p, groups[p], ncols[p]) for p in sorted(groups)]

    @staticmethod
    def _pre_rows(forward, t: int) -> int:
        """Rows of the pre-whitened matrix of record set t: r of its factors, else the node count."""
        f = forward.factors[t]
        return forward.records.shape[1] if f is None else f[0].shape[1]

    def fast_path_bytes(self, forward, pre: bool = True) -> int:
        """Bytes of the matrices `bind_forward` would precompute for this forward model: the pre-whitened records (all
        selected Teff slices; left out with pre=False) and the Cholesky factor of each spectrum with a dense
        whitening."""
        cols = sum(c for _, _, c in self._fast_layout()) if pre else 0
        rows = sum(self._pre_rows(forward, t) for t in range(forward.records.shape[0]))
        chol = sum(s.n * s.n for s in self._specs if s.W is not None)
        return int(rows * cols * np.dtype(FAST_PATH_DTYPE).itemsize + 8 * chol)

    def _fast_decline(self, forward) -> str:
        if forward.dataset is not self.dataset:
            return "the forward model belongs to another data set"
        nbytes = self.fast_path_bytes(forward, pre=False)
        if nbytes > FAST_PATH_MAX_BYTES:
            return (f"the Cholesky factors alone would take {nbytes / 1e6:.0f} MB, above FAST_PATH_MAX_BYTES = "
                    f"{FAST_PATH_MAX_BYTES / 1e6:.0f} MB")
        recs = forward.records
        for s in self._specs:  # a non-finite record would poison the sums over a node range (the exact path flags it)
            if not np.all(np.isfinite(recs[:, :, :, s.spec.pixels])):
                return f"spectrum {s.spec.label}: library records not finite on its fitted pixels"
        return ""

    def bind_forward(self, forward) -> str:
        """Prepare the pre-whitened-record path for this forward model and return ``fast_path`` ("on", or
        "off: <reason>"; ``fast_path_detail`` says more). Idempotent: a second call with the same forward model returns
        at once. Above FAST_PATH_MAX_BYTES the pre-whitened matrices are left out and the path stays on with its other
        two savings (node range, triangular whitening); "off" only when even the Cholesky factors are over budget.

        For every selected Teff slice t and every phase p with a pre-whitened spectrum (``_fast_layout``), one
        matrix (NB * NC, columns) of FAST_PATH_DTYPE, node-major so that a range of nodes is a view: for each such I
        spectrum s the blocks (W_s (R_t[:, I, pix_s]^T * P_j[:, None]))^T, j = 0 .. deg (P_0 = 1: the whitened
        model). Built with BLAS and numba at one thread (search.single_threaded), so the matrices are the same bits in
        the parent, in serial searches and in pool workers (SEARCH-5, V6a)."""
        if getattr(self, "_bound", None) is forward:
            return self.fast_path
        self._bound, self._fast = forward, None
        reason = self._fast_decline(forward)
        if reason:
            self.fast_path = "off: " + reason
            self.fast_path_detail = self.fast_path
            return self.fast_path
        from .search import single_threaded  # function level: search imports problem, which imports this module

        layout = self._fast_layout()
        nbytes = self.fast_path_bytes(forward)
        if nbytes > FAST_PATH_MAX_BYTES:  # over budget: no pre-whitened matrices, the node range and L_s remain
            layout = []
            self.fast_path_detail = (f"on, without pre-whitened matrices: they would take {nbytes / 1e6:.0f} MB, above "
                                     f"FAST_PATH_MAX_BYTES = {FAST_PATH_MAX_BYTES / 1e6:.0f} MB")
        else:
            self.fast_path_detail = f"on, {nbytes / 1e6:.0f} MB precomputed"
        recs = forward.records  # (n_teff, nodes, 2, M) float64
        nodes = recs.shape[1]
        pre = []
        with single_threaded():
            for t in range(recs.shape[0]):
                f = forward.factors[t]  # factored records: the matrices are built from B, (r, 2, M)
                src = recs[t] if f is None else f[1].reshape(f[1].shape[0], 2, -1)
                rows = src.shape[0]
                per_phase = {}
                for p, group, ncol in layout:
                    A = np.empty((rows, ncol), dtype=FAST_PATH_DTYPE)
                    for j, off, nblk in group:
                        s = self._specs[j]
                        n = s.n
                        RI = src[:, 0, :][:, s.spec.pixels]  # (rows, n)
                        X = (RI[None, :, :] * s.basis[:, None, :]).reshape(nblk * rows, n) if s.deg >= 0 else RI
                        WX = self._whiten(s, X.T)  # (n, nblk * rows)
                        A[:, off : off + nblk * n] = WX.reshape(n, nblk, rows).transpose(2, 1, 0).reshape(rows, -1)
                    per_phase[p] = A
                pre.append(per_phase)
        where = {j: (p, off, nblk) for p, group, _ in layout for j, off, nblk in group}
        # the Cholesky factors again (the exact path keeps only W_s): whitening raw rows by a triangular solve reads
        # half the matrix and was measured 2x faster than the product with W_s for 4 to 27 columns (900, 1451 pixels)
        chol = {j: (np.asfortranarray(np.tril(cho_factor(s.K, lower=True)[0])), 1.0 / s.sig)
                for j, s in enumerate(self._specs) if s.W is not None}
        self._fast = {"layout": {p: ncol for p, _, ncol in layout}, "where": where, "pre": pre, "nodes": nodes,
                      "forward": forward, "chol": chol}
        self.fast_path = "on"
        return self.fast_path

    @staticmethod
    def _prefer_pre(s: _Spec, K: int, nodes: int, n_slices: int) -> bool:
        """Whether the pre-whitened product beats raw model + whitening for spectrum s, by a roofline estimate on
        one core: a product of K rows with an (a x b) matrix costs ~ a b max(1, K / FAST_GEMM_ROWS); pre-whitened:
        n_slices (deg+1) n x nodes with K rows; whitening: n x n with (deg+1) K columns."""
        nb = max(s.deg, 0) + 1
        pre = n_slices * nb * s.n * nodes * max(1.0, K / FAST_GEMM_ROWS)
        white = 0.5 * s.n * s.n * max(1.0, nb * K / FAST_GEMM_ROWS)  # triangular solve: half the matrix
        return pre < white

    def _fit_raw(self, j: int, s: _Spec, mrows: np.ndarray):
        """`_fit` with a dense W_s applied as L_s^-1 D^-1 by a triangular solve (same result to rounding)."""
        if s.W is None:
            return self._fit(s, mrows)
        L, isig = self._fast["chol"][j]

        def white(x):  # (n, cols)
            return solve_triangular(L, x * isig[:, None], lower=True, check_finite=False)

        K = mrows.shape[0]
        if s.deg >= 0:
            X = (mrows[:, None, :] * s.basis[None, :, :]).reshape(K * (s.deg + 1), s.n)
            WX = white(X.T).reshape(s.n, K, s.deg + 1).transpose(1, 0, 2)
            return self._fit_whitened(s, WX[:, :, 0], WX)
        return self._fit_whitened(s, white(mrows.T).T, None)

    def neg_log_l_nodes(self, Wnodes, slice_weights, invalid=None) -> np.ndarray:
        """-ln L (without ``constant``) from node weights (``bind_forward`` first): Wnodes (K, P, NB * NC),
        slice_weights [(t, (K,) coefficients)], invalid (K,) bool -> (K,), as ``neg_log_l(*forward.spectra(...))``
        (forward.node_weights_batch gives the arguments). Invalid rows -> 1e30.

        All products are restricted to the contiguous range of nodes with a non-zero weight in some row of the batch
        (the other nodes contribute exact zeros). Raw I and V for all phases: one product per Teff slice touched with
        the forward model's records, as forward.spectra, but over the node range only. Each pre-whitened I spectrum
        (``_fast_layout``) is then taken, when ``_prefer_pre`` says so for this batch, from one product per phase and
        slice with the precomputed matrices (no whitening left), else from the raw model like every other spectrum
        (whitening as in the exact path, a dense W_s applied by a triangular solve with L_s); then the continuum QR."""
        if self.fast_path != "on":
            raise RuntimeError(f"the pre-whitened-record path is not available ({self.fast_path})")
        fast = self._fast
        Wn = np.asarray(Wnodes, dtype=np.float64)
        if Wn.ndim != 3 or Wn.shape[2] != fast["nodes"]:
            raise ValueError(f"Wnodes must be (K, P, {fast['nodes']}), got {Wn.shape}")
        K, P = Wn.shape[:2]
        bad = np.zeros(K, dtype=bool) if invalid is None else np.asarray(invalid, dtype=bool).reshape(K).copy()
        used = np.flatnonzero(np.any(Wn != 0.0, axis=(0, 1)))
        r0, r1 = (int(used[0]), int(used[-1]) + 1) if used.size else (0, 0)
        weights = [(t, c) for t, c in ((t, np.asarray(c, dtype=np.float64)) for t, c in slice_weights) if np.any(c)]
        weights = [(t, c if c.ndim == 2 else c.reshape(K)) for t, c in weights]
        # per-pixel Teff weights ("planck", DESIGN.md section 5c) cannot act after the whitening: the pre-whitened
        # products are then not used, and the raw I and V are combined per pixel before it
        per_pixel = any(c.ndim == 2 for _, c in weights)
        fw = fast["forward"]
        recs = fw.records  # (n_teff, nodes, 2, M), C-contiguous: a node range is a view
        M = recs.shape[3]

        def product(Wr, mats, ncol, rep):  # sum over slices of c_t (per row, repeated rep times) * (Wr @ mats(t))
            O = np.zeros((Wr.shape[0], ncol))
            if r1 <= r0 or ncol == 0:
                return O
            for t, c in weights:
                A = mats(t)
                f = fw.factors[t]
                Wt = Wr if f is None else Wr @ f[0][r0:r1]  # factored records: Wr @ R = (Wr @ A) @ B
                if c.ndim == 2:  # (K, M) per pixel, raw records only: columns are [I (M), V (M)]
                    cc = np.repeat(c, rep, axis=0)
                    rows = np.flatnonzero(np.any(cc != 0.0, axis=1))
                    O[rows] += ((Wt[rows] @ A).reshape(rows.size, 2, -1) * cc[rows, None, :]).reshape(rows.size, -1)
                elif np.all(c == 1.0):
                    O += Wt @ A
                else:
                    cc = np.repeat(c, rep)
                    rows = np.flatnonzero(cc)
                    O[rows] += cc[rows, None] * (Wt[rows] @ A)
            return O

        nodes_eff = max([r1 - r0 if fw.factors[t] is None else fw.factors[t][0].shape[1] for t, _ in weights],
                        default=r1 - r0)  # rows of the products after the node range (r for factored records)
        modes = {j: (not per_pixel) and self._prefer_pre(self._specs[j], K, nodes_eff, len(weights)) for j in fast["where"]}
        terms = np.zeros((K, len(self._specs)))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw = None
            if not all(modes.get(j, False) for j in range(len(self._specs))):
                Wr = np.ascontiguousarray(Wn[:, :, r0:r1].reshape(K * P, r1 - r0))
                raw = product(Wr, lambda t: (recs[t, r0:r1].reshape(r1 - r0, 2 * M) if fw.factors[t] is None
                                             else fw.factors[t][1]), 2 * M, P).reshape(K, P, 2, M)
            pre_out = {}
            for j, s in enumerate(self._specs):
                p, n = int(s.spec.phase_index), s.n
                if modes.get(j, False):
                    if p not in pre_out:
                        Wp = np.ascontiguousarray(Wn[:, p, r0:r1], dtype=FAST_PATH_DTYPE)
                        pre_out[p] = product(Wp, lambda t: (fast["pre"][t][p][r0:r1] if fw.factors[t] is None
                                                            else fast["pre"][t][p]), fast["layout"][p], 1)
                    _, off, nblk = fast["where"][j]
                    blk = pre_out[p][:, off : off + nblk * n].reshape(K, nblk, n)
                    bad |= ~np.all(np.isfinite(blk), axis=(1, 2))
                    ok = np.flatnonzero(~bad)
                    if not ok.size:
                        continue
                    if s.deg >= 0:
                        WX = blk[ok].transpose(0, 2, 1)  # (K, n, deg+1)
                        res, logdet = self._fit_whitened(s, WX[:, :, 0], WX)
                    else:
                        res, logdet = self._fit_whitened(s, blk[ok, 0, :], None)
                else:
                    Ir = raw[:, p, 0, s.spec.pixels]
                    if s.spec.kind == "VI":
                        Vr = raw[:, p, 1, s.spec.pixels]
                        m = np.where(Ir != 0.0, Vr / np.where(Ir != 0.0, Ir, 1.0), 0.0)
                        m[np.isnan(Ir)] = np.nan
                    else:
                        m = Ir
                    bad |= ~np.all(np.isfinite(m), axis=1)
                    ok = np.flatnonzero(~bad)
                    if not ok.size:
                        continue
                    res, logdet = self._fit_raw(j, s, m[ok])
                terms[ok, j] = self._term(s, res, logdet)
        out = np.full(K, WORST)
        ok = np.flatnonzero(~bad)
        if ok.size:
            val = terms[ok].sum(axis=1)
            fin = np.isfinite(val)
            out[ok[fin]] = val[fin]
            bad[ok[~fin]] = True
        self.n_invalid += int(bad.sum())
        self.n_evals += K
        return out

    def noise_scales(self, I, VI) -> np.ndarray:
        """The noise-scale estimates sqrt(chi2_s / (n_s - k_s)) for one parameter set (I, VI of shape (P, M)); ones
        when the scale is off (and always for kind "chi2")."""
        I, VI = self._batch(I, VI)
        out = np.ones(len(self._specs))
        if self.scale:
            for j, s in enumerate(self._specs):
                res, _ = self._fit(s, self._rows(s, I, VI))
                out[j] = math.sqrt(float(res[0] @ res[0]) / (s.n - s.k))
        return out

    def residuals(self, I, VI, scales=None) -> list[np.ndarray]:
        """Whitened residuals per spectrum after the best continuum, for one parameter set (I, VI of shape (P, M)),
        each divided by its noise-scale estimate sqrt(chi2_s / (n_s - k_s)) when the scale is on (kind "full").

        With the scale on, the sum of squares of each vector is n_s - k_s whatever the model, so a least-squares
        solver must hold the scales fixed: pass ``scales`` (from ``noise_scales`` at the current point) to divide by
        those instead. The ln det(A^T A) term of the full likelihood is not represented in the residuals."""
        I, VI = self._batch(I, VI)
        out = []
        for j, s in enumerate(self._specs):
            res, _ = self._fit(s, self._rows(s, I, VI))
            res = res[0]
            if scales is not None:
                res = res / scales[j]
            elif self.scale:
                res = res / math.sqrt(float(res @ res) / (s.n - s.k))
            out.append(res)
        return out

    def per_spectrum(self, I, VI) -> dict:
        """Plain chi2 per spectrum and fit window for one parameter set: label -> {"chi2", "n", "k", "windows":
        {window index -> {"chi2", "n"}}}. The chi2 is unwhitened and unweighted, sum ((y - model') / sigma)^2, where
        model' carries the continuum as fitted inside the likelihood (same whitening, degree and offset): the model I
        that `Problem.continuum_models` writes to spectra/ (a continuum fitted here by plain least squares
        gave a chi2 up to 100x below that of the written model I under a fitted kernel)."""
        I, VI = self._batch(I, VI)
        out = {}
        for s in self._specs:
            sp = s.spec
            res, _ = self._fit(s, self._rows(s, I, VI))  # W (y - model')
            r = res[0] / s.wdiag if s.W is None else solve_triangular(s.W, res[0], lower=True, check_finite=False)
            res2 = (r / np.asarray(sp.sigma, dtype=np.float64)) ** 2
            win = np.asarray(sp.window)
            windows = {int(w): {"chi2": float(res2[win == w].sum()), "n": int(np.sum(win == w))} for w in np.unique(win)}
            out[sp.label] = {"chi2": float(res2.sum()), "n": int(s.n), "k": int(s.k), "windows": windows}
        return out

    def check_scale(self, I) -> dict:
        """Scale sanity check for one model I (P, M): label -> median |model I| / median |data| on the fitted pixels of
        each I spectrum. -ln L does not depend on this ratio except through the ln det(A^T A) term (a constant shift,
        `_fit_whitened`), but a ratio far from 1 (outside [1e-3, 1e3], ``SCALE_RATIO_RANGE``) usually means the library
        and the data are in different units, and an I spectrum without a continuum fit then cannot match at all."""
        I = np.asarray(I, dtype=np.float64)
        out = {}
        for s in self._specs:
            if s.spec.kind != "I":
                continue
            m = I[s.spec.phase_index, s.spec.pixels]
            d = float(np.median(np.abs(np.asarray(s.spec.data, dtype=np.float64))))
            out[s.spec.label] = float(np.median(np.abs(m))) / d if d > 0 else math.inf
        return out
