"""Slow, direct reference of the likelihood for V5b (VERIFICATION.md V5): one parameter set at a time, explicit
covariance matrices, numpy.linalg.slogdet and generalised least squares by the normal equations. Written for clarity,
not speed; it shares no code with likelihood.py except the settings reader and the result convention (see that
module's docstring for the formulas).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import eval_legendre, gammaln

from .likelihood import objective_settings


def _matern32(d, rho):
    return (1.0 + math.sqrt(3.0) * d / rho) * math.exp(-math.sqrt(3.0) * d / rho)


def _kernel(lam, kind, corr, disc):
    n = len(lam)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(lam[i] - lam[j])
            if corr is False:
                v = 1.0 if i == j else 0.0
            else:
                v = _matern32(d, corr["rho"]) + (corr["jitter"] ** 2 if i == j else 0.0)
            if disc is not False:
                a = disc["a_I"] if kind == "I" else disc["a_VI"]
                rho = disc["rho_I"] if kind == "I" else disc["rho_VI"]
                v += a * a * _matern32(d, rho)
            K[i, j] = v
    return K


def _design(spec, lam, model, cont):
    """Unwhitened continuum design matrix (n, k): model * P_j(x), and a column of ones for the offset."""
    cols = []
    if spec.kind == "I" and cont is not False:
        x = 2.0 * (lam - lam.min()) / (lam.max() - lam.min()) - 1.0
        for j in range(cont["degree"] + 1):
            cols.append(model * eval_legendre(j, x))
        if spec.i_offset:
            cols.append(np.ones_like(model))
    return np.column_stack(cols) if cols else np.zeros((len(model), 0))


def spectrum_term(spec, lam, model, settings) -> tuple[float, float]:
    """(term, constant) of one spectrum for one model vector on its fitted pixels."""
    kind = settings["kind"]
    y = np.asarray(spec.data, float)
    sig = np.asarray(spec.sigma, float)
    w = np.asarray(spec.weight, float)
    n = len(y)
    if kind == "chi2":
        C = np.diag(sig**2 / w)
    else:
        sig = sig / np.sqrt(w)  # DATA-4 (decision #27): a weight scales the error
        C = np.diag(sig) @ _kernel(lam, spec.kind, settings["corr"], settings["disc"]) @ np.diag(sig)
    L = np.linalg.cholesky(C)  # C = L L^T; whitened quantities x_w = L^-1 x, so x^T C^-1 x = |x_w|^2
    A = _design(spec, lam, model, settings["continuum"])
    k = A.shape[1]
    if k:
        # The design contains the model itself (P_0 = 1), so the residual after the best continuum is that of y alone,
        # y - A gamma: least squares on the whitened y by numpy's SVD solver, on columns scaled to unit length (scale
        # invariant, robust to a model in physical flux units, 1e15 x the data). ln det(A^T C^-1 A) from the scaled
        # columns plus 2 sum ln(column lengths).
        Aw = np.linalg.solve(L, A)
        yw = np.linalg.solve(L, y)
        d = np.linalg.norm(Aw, axis=0)
        As = Aw / d
        gamma, *_ = np.linalg.lstsq(As, yw, rcond=None)
        ew = yw - As @ gamma
        logdet_G = np.linalg.slogdet(As.T @ As)[1] + 2.0 * float(np.sum(np.log(d)))
        # normalised basis (decision #26): the deg+1 model columns divided by nu = rms of the whitened model, so -ln L
        # does not depend on the model's scale; the offset column is not divided
        nleg = settings["continuum"]["degree"] + 1
        nu = float(np.linalg.norm(np.linalg.solve(L, model))) / float(np.linalg.norm(np.linalg.solve(L, np.ones(n))))
        logdet_G -= 2.0 * nleg * math.log(nu)
    else:
        ew, logdet_G = np.linalg.solve(L, y - model), 0.0
    chi2 = float(ew @ ew)
    if kind == "chi2":
        return 0.5 * chi2, 0.0
    m = n - k
    logdet_C = np.linalg.slogdet(C)[1]
    if settings["scale"]:
        term = 0.5 * (m * math.log(chi2) + logdet_G)
        const = 0.5 * logdet_C + 0.5 * m * math.log(math.pi) - gammaln(0.5 * m) + math.log(2.0)
    else:
        term = 0.5 * (chi2 + logdet_G)
        const = 0.5 * logdet_C + 0.5 * m * math.log(2.0 * math.pi)
    return term, const


def neg_log_l(dataset, cfg, I, VI) -> tuple[float, float]:
    """(-ln L without the constant, the constant) for one parameter set, I and VI of shape (P, M)."""
    st = objective_settings(cfg)
    lam_all = dataset.grid.wavelengths
    total, const = 0.0, 0.0
    for spec in dataset.spectra:
        lam = lam_all[spec.pixels]
        src = I if spec.kind == "I" else VI
        model = np.asarray(src[spec.phase_index, spec.pixels], float)
        t, c = spectrum_term(spec, lam, model, st)
        total += t
        const += c
    return total, const
