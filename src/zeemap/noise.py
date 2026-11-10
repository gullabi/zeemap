"""The noise model fitted per object (SPEC LIKE-2, LIKE-4, LIKE-10; DESIGN.md sections 3 step 2b and 5; decision #22).

- `marginal_neg_log_l`: the exact -ln of the marginal likelihood of one spectrum at a fixed model, as a function of its
  kernel K = M(rho_short) + a^2 M(rho_long) + j^2 I (M the Matern-3/2 kernel), with the noise scale and the continuum
  integrated out as in likelihood.Objective (the normalised continuum basis, decision #26), and ln det K included,
  because the kernel varies here. It equals `Objective.neg_log_l + Objective.constant` for that kernel (V1o).
- `fit_spectrum_kernel` / `fit_kernels`: maximise it per spectrum over (rho_short, a, rho_long, j) in their bounds
  (DESIGN.md section 5), by L-BFGS-B in the logarithms, from 3 starts.
- `kernel_stage`: step 2b of the search (fit at the best end point, re-polish the band, refit, freeze), used by
  search.run_search through its `kernel_hook`; the frozen kernel goes into the configuration as `[[objective.kernel]]`
  entries with `objective.kernel_fit = false`, so worker processes, swaps and run.json carry it.
- `whiteness`: the run-time check of LIKE-10.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln

from .likelihood import legendre_x, matern32

J_MIN, J_MAX = 0.05, 10.0  # the white part j (the lower bound keeps K well conditioned)
A_MIN, A_MAX = 1e-3, 10.0
MAX_LAG = 10  # LIKE-10: lags 1-10 pixels
BOUND_REL = 1e-3  # a fitted kernel parameter within this relative distance of a bound is reported at it
FAIL_FLAG = "noise model fails the whiteness check"  # LIKE-10, on every -ln L difference and interval


_DIST: dict = {}  # |lambda_i - lambda_j| per wavelength vector (the kernel fit evaluates one spectrum many times)


def _distances(lam: np.ndarray) -> np.ndarray:
    key = (lam.size, float(lam[0]), float(lam[-1]), hash(lam.tobytes()))
    d = _DIST.get(key)
    if d is None:
        if len(_DIST) > 64:
            _DIST.clear()
        d = _DIST[key] = math.sqrt(3.0) * np.abs(lam[:, None] - lam[None, :])
    return d


def kernel_matrix(lam: np.ndarray, rho_short: float, a: float, rho_long: float, j: float) -> np.ndarray:
    """M(rho_short) + a^2 M(rho_long) + j^2 I (the same numbers as likelihood.matern32, with cached distances)."""
    lam = np.asarray(lam, dtype=np.float64)
    d = _distances(lam)
    xs, xl = d / float(rho_short), d / float(rho_long)
    K = (1.0 + xs) * np.exp(-xs)
    K += (a * a) * ((1.0 + xl) * np.exp(-xl))
    K[np.diag_indices_from(K)] += j * j
    return K


def marginal_neg_log_l(lam, y, sig, model, kind: str, deg: int, offset: bool, theta, scale: bool = True) -> float:
    """-ln of the marginal likelihood of one spectrum (see the module docstring). `theta` = (rho_short, a, rho_long,
    j); `deg` < 0 means no continuum (V/I, or the continuum off); `sig` the errors as used in D (weights applied)."""
    lam, y, sig, model = (np.asarray(v, dtype=np.float64) for v in (lam, y, sig, model))
    n = y.size
    try:
        L, _ = cho_factor(kernel_matrix(lam, *theta), lower=True)
    except np.linalg.LinAlgError:
        return math.inf
    L = np.tril(L)
    logdet_K = 2.0 * float(np.sum(np.log(np.diag(L))))

    def white(x):
        return solve_triangular(L, (x.T / sig).T, lower=True, check_finite=False)

    y_w = white(y)
    if kind == "I" and deg >= 0:
        P = legendre.legvander(legendre_x(lam), deg)  # (n, deg + 1)
        m_w = white(model)
        nu = float(np.linalg.norm(m_w) / np.linalg.norm(white(np.ones(n))))
        cols = [white(model[:, None] * P) / nu]
        if offset:
            cols.append(white(np.ones(n))[:, None])
        A = np.concatenate(cols, axis=1)
        Q, R = np.linalg.qr(A)
        res = y_w - Q @ (Q.T @ y_w)  # the model is in span(A): the residual after the best continuum
        k = A.shape[1]
        logdet_A = 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))
    else:
        res = y_w - white(model)
        k, logdet_A = 0, 0.0
    chi2 = float(res @ res)
    m = n - k
    out = 0.5 * logdet_K + float(np.sum(np.log(sig))) + 0.5 * logdet_A
    if scale:
        return out + 0.5 * m * math.log(chi2) + 0.5 * m * math.log(math.pi) - float(gammaln(0.5 * m)) + math.log(2.0)
    return out + 0.5 * chi2 + 0.5 * m * math.log(2.0 * math.pi)


def kernel_bounds(lam: np.ndarray, step: float, fwhm: float) -> list[tuple[float, float]]:
    """Bounds of (rho_short, a, rho_long, j) for one spectrum (DESIGN.md section 5): rho_short in [0.5 pixel, FWHM],
    rho_long in [FWHM, a quarter of the longest run of contiguous fitted pixels] (= FWHM when that is shorter)."""
    runs, start = [], 0
    pix = np.rint((lam - lam[0]) / step).astype(int)
    for i in range(1, pix.size + 1):
        if i == pix.size or pix[i] != pix[i - 1] + 1:
            runs.append(lam[i - 1] - lam[start])
            start = i
    longest = max(runs) if runs else 0.0
    lo_s, hi_s = 0.5 * step, max(fwhm, 0.5 * step)
    lo_l, hi_l = hi_s, max(hi_s, 0.25 * longest)
    return [(lo_s, hi_s), (A_MIN, A_MAX), (lo_l, hi_l), (J_MIN, J_MAX)]


def _clip(theta, bounds):
    return [min(max(float(v), lo), hi) for v, (lo, hi) in zip(theta, bounds)]


def fit_spectrum_kernel(lam, y, sig, model, kind, deg, offset, step, fwhm, initial, scale: bool = True) -> dict:
    """The kernel of one spectrum that maximises its marginal likelihood (DESIGN.md section 5): {rho_short, a, rho_long,
    j, neg_log_l, neg_log_l_initial, gain, at_bound}; `at_bound` names each parameter that ended at a bound of its
    range, {name: "lower" | "upper"}, except one with an empty range (lo == hi) and rho_long when a is at
    its lower bound (the long kernel then has no share, so its length is not identified)."""
    bounds = kernel_bounds(np.asarray(lam, float), step, fwhm)
    zb = [(math.log(lo), math.log(hi)) for lo, hi in bounds]

    def f(z):
        v = marginal_neg_log_l(lam, y, sig, model, kind, deg, offset, np.exp(z), scale)
        return v if math.isfinite(v) else 1e30

    def geo(b):
        return math.sqrt(b[0] * b[1])

    init = _clip(initial, bounds)
    starts = [init, [geo(bounds[0]), 1.0, geo(bounds[2]), 0.3], [geo(bounds[0]), 3.0, geo(bounds[2]), 0.1]]
    f0 = marginal_neg_log_l(lam, y, sig, model, kind, deg, offset, init, scale)
    best = None
    for s in starts:
        z0 = np.log(_clip(s, bounds))
        r = minimize(f, z0, method="L-BFGS-B", bounds=zb, options={"maxiter": 200})
        if best is None or r.fun < best.fun:
            best = r
    th = np.exp(best.x)
    at_bound = {}
    for name, v, (lo, hi) in zip(("rho_short", "a", "rho_long", "j"), th, bounds):
        if hi <= lo or (name == "rho_long" and th[1] <= A_MIN * (1.0 + BOUND_REL)):
            continue
        if v >= hi * (1.0 - BOUND_REL):
            at_bound[name] = "upper"
        elif v <= lo * (1.0 + BOUND_REL):
            at_bound[name] = "lower"
    return {"rho_short": float(th[0]), "a": float(th[1]), "rho_long": float(th[2]), "j": float(th[3]),
            "share_short": float(1.0 / (1.0 + th[1] ** 2)), "neg_log_l": float(best.fun),
            "neg_log_l_initial": float(f0), "gain": float(f0 - best.fun), "at_bound": at_bound}


def capped_kernels(kernels: list[dict]) -> dict:
    """{spectrum label: at_bound} of the kernels whose long (discrepancy) part ended at its cap a = A_MAX: these
    spectra count little in the fit. White residuals end in the white corner (a and rho_short lower,
    j upper), which is the healthy case and is not named here."""
    return {k["spectrum"]: k["at_bound"] for k in kernels if (k.get("at_bound") or {}).get("a") == "upper"}


def initial_kernel(settings: dict, kind: str) -> tuple[float, float, float, float]:
    """(rho_short, a, rho_long, j) of the initial kernel (objective.corr and objective.disc)."""
    corr, disc = settings.get("corr"), settings.get("disc")
    rs, j = (4.0, 0.05) if not corr else (float(corr["rho"]), float(corr["jitter"]))
    if not disc:
        a, rl = A_MIN, 85.0
    else:
        a, rl = ((disc["a_I"], disc["rho_I"]) if kind == "I" else (disc["a_VI"], disc["rho_VI"]))
    return rs, float(a), float(rl), max(j, J_MIN)


def fit_kernels(problem, x, fwhm: float) -> list[dict]:
    """Fit the kernel of every spectrum of `problem` at the parameters x (one model evaluation). Returns one entry per
    spectrum, in the layout of `[[objective.kernel]]` plus the fit's numbers."""
    ob = problem.objective
    if ob.kind != "full":
        raise ValueError("the kernel fit needs objective.kind = 'full'")
    I, VI = problem.model_spectra(np.asarray(x, dtype=float))
    lam_all = problem.dataset.grid.wavelengths
    step = float(problem.dataset.grid.step)
    cont = ob.settings["continuum"]
    out = []
    for sp in problem.dataset.spectra:
        pix = np.asarray(sp.pixels)
        lam = lam_all[pix]
        sig = np.asarray(sp.sigma, float) / np.sqrt(np.asarray(sp.weight, float))
        model = (I if sp.kind == "I" else VI)[sp.phase_index, pix]
        deg = cont["degree"] if (cont is not False and sp.kind == "I") else -1
        offset = bool(sp.i_offset) and cont is not False and sp.kind == "I"
        r = fit_spectrum_kernel(lam, np.asarray(sp.data, float), sig, model, sp.kind, deg, offset, step, fwhm,
                                initial_kernel(ob.settings, sp.kind), scale=bool(ob.scale))
        out.append({"spectrum": sp.label, **r})
    return out


def with_kernels(cfg: dict, kernels: list[dict]) -> dict:
    """A copy of the loaded configuration with the kernels given per spectrum and the kernel fit switched off."""
    c = copy.deepcopy(cfg)
    c["objective"]["kernel"] = [{k: e[k] for k in ("spectrum", "rho_short", "a", "rho_long", "j")} for e in kernels]
    c["objective"]["kernel_fit"] = False
    return c


def error_bar_cfg(cfg: dict) -> dict:
    """The configuration on the error-bar patch count (`uncertainty.patches`, Fibonacci surfaces only)."""
    c = copy.deepcopy(cfg)
    surf = c["model"]["surface"]
    patches = int((c.get("uncertainty") or {}).get("patches", 0) or 0)
    if surf.get("scheme") == "fibonacci" and patches > 0:
        c["model"]["surface"] = {"scheme": "fibonacci", "n": patches}
    return c


def kernel_stage(problem, endpoints: list, *, cfg: dict, make_problem: Callable, polish_many: Callable,
                 make_endpoint: Callable, band: float, say: Callable[[str], None]):
    """Step 2b of DESIGN.md section 3. Returns (the problem with the frozen kernel, its configuration, the end points
    under it, a summary for run.json). `polish_many(problem, X0) -> [PolishResult]`, `make_endpoint(problem, pr, e)`
    builds the end point of a re-polish of end point e."""
    fwhm = float(cfg["model"]["fwhm"])
    valid = [e for e in endpoints if e.valid]
    best = min(valid, key=lambda e: e.f)
    # (i) fit at the best end point, on the error-bar patch count
    k1 = fit_kernels(make_problem(error_bar_cfg(cfg)), best.x, fwhm)
    cfg1 = with_kernels(cfg, k1)
    p1 = make_problem(cfg1)
    # (ii) re-polish the band (as measured under the initial kernel) with the fitted kernel
    in_band = [i for i, e in enumerate(endpoints) if e.valid and e.f <= best.f + band]
    prs = polish_many(p1, [endpoints[i].x for i in in_band])
    eps = list(endpoints)
    for i, pr in zip(in_band, prs):
        eps[i] = make_endpoint(p1, pr, endpoints[i])
    # (iii) refit at the new best end point and freeze
    best2 = min((e for e in eps if e.valid), key=lambda e: e.f)
    k2 = fit_kernels(make_problem(error_bar_cfg(cfg1)), best2.x, fwhm)
    cfg2 = with_kernels(cfg, k2)
    p2 = make_problem(cfg2)
    # (iv) every end point under the frozen kernel (no polish)
    X = np.array([e.x for e in eps])
    F = p2.cost(X)
    for e, fv in zip(eps, F):
        e.f = float(fv)
        e.valid = bool(e.valid and fv < 1e30)
    gain = sum(k["gain"] for k in k2)
    say(f"search: kernel fitted per spectrum and frozen ({len(k2)} spectra, {len(in_band)} band end points "
        f"re-polished; -ln L of the best end point {gain:.1f} nats better than with the initial kernel)")
    info = {"state": "fitted, frozen", "first_fit": k1, "frozen": k2, "n_repolished": len(in_band)}
    p2.objective.kernel_state = info
    return p2, cfg2, eps, info


def acf_check(e, pix) -> dict:
    """LIKE-10 for one whitened residual vector `e` on the grid pixel indices `pix` (increasing): the largest |ACF| at
    lags 1-10 over pairs of pixels k grid steps apart with every pixel between them fitted, and the verdict."""
    e = np.asarray(e, float)
    pix = np.asarray(pix)
    n = pix.size
    denom = float(e @ e)
    best, lag = 0.0, 0
    for k in range(1, MAX_LAG + 1):
        if n <= k:
            break
        ok = (pix[k:] - pix[:-k]) == k  # contiguous: no excluded pixel between them
        if not np.any(ok):
            continue
        r = float(np.sum(e[k:][ok] * e[:-k][ok])) / denom
        if abs(r) > abs(best):
            best, lag = r, k
    thr = 3.0 / math.sqrt(n) + 0.05
    return {"max_abs_acf": abs(best), "lag": lag, "threshold": thr, "n": int(n), "pass": bool(abs(best) < thr)}


def whiteness(problem, x) -> dict:
    """LIKE-10 at parameters x: per spectrum label {max_abs_acf, lag, threshold, n, pass}; pairs of fitted pixels k grid
    steps apart with every pixel between them fitted (no pair spans a gap). Whitened residuals after the best
    continuum, as the likelihood forms them."""
    I, VI = problem.model_spectra(np.asarray(x, dtype=float))
    res = problem.objective.residuals(I, VI)
    return {sp.label: acf_check(e, sp.pixels) for sp, e in zip(problem.dataset.spectra, res)}
