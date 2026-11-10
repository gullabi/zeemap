"""Plain-numpy reference of the forward model, written for clarity (check V5a; VERIFICATION.md V5).

`ForwardReference` has the constructor and the `spectra` signature of `forward.Forward` and shares its precomputed
records (convolution and resampling are plain numpy there already and are checked on their own by V1n and V2a).
It replaces the two parts that the fast code does differently:
- the node weights: per parameter set and phase, numpy arrays over the patches instead of the numba scatter (for
  `interp_b = "cubic"` the Hermite weights over all B nodes, with the slopes written out in Lagrange form);
- the disc sum: per parameter set and phase, an explicit weighted sum over the library nodes (einsum) instead of one
  matrix product for the whole batch, and the Teff interpolation applied to the interpolated records.
No numba here.
"""

from __future__ import annotations

import numpy as np

from zeemap.forward import Forward


def _bracket(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Index i (0 .. n - 2) with nodes[i] <= x <= nodes[i + 1] for increasing nodes."""
    return np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)


def bmag_cospsi(B: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|B| and cos psi = B . n / |B| (clipped to [-1, 1], 0 where |B| = 0) for B (..., 3)."""
    b = np.sqrt(np.sum(B * B, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(b > 0, np.sum(B * n, axis=-1) / np.where(b > 0, b, 1.0), 0.0)
    return b, np.clip(c, -1.0, 1.0)


def hermite_weights_reference(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(len(b), NB) cubic Hermite weights in B with three-point slopes (forward.hermite_slope_matrix), written out.

    Slope of node i as weights on the nodes, from the parabola through nodes (a, a + 1, a + 2) differentiated at x_i,
    with a = i - 1 inside, a = 0 at the first node and a = NB - 3 at the last (Lagrange form); two nodes: the secant.
    """
    n = len(x)
    slope = np.zeros((n, n))
    for i in range(n):
        if n == 2:
            slope[i] = [-1.0 / (x[1] - x[0]), 1.0 / (x[1] - x[0])]
            continue
        a = min(max(i - 1, 0), n - 3)
        idx = [a, a + 1, a + 2]
        for j in idx:  # d/dx of the Lagrange basis l_j at x_i
            others = [m for m in idx if m != j]
            denom = np.prod([x[j] - x[m] for m in others])
            slope[i, j] = sum(np.prod([x[i] - x[m] for m in others if m != o]) for o in others) / denom
    i = _bracket(b, x)
    h = x[i + 1] - x[i]
    t = (b - x[i]) / h
    W = h[:, None] * (((t**3 - 2 * t**2 + t))[:, None] * slope[i] + (t**3 - t**2)[:, None] * slope[i + 1])
    rows = np.arange(len(b))
    W[rows, i] += 2 * t**3 - 3 * t**2 + 1
    W[rows, i + 1] += -2 * t**3 + 3 * t**2
    return W


class ForwardReference(Forward):
    """Same results as Forward (to rounding), computed the plain way."""

    def _scatter(self, Theta):
        K, P = Theta.shape[0], len(self.phases)
        NB, NC = self.NB, self.NC
        obs, bad = self.observers(Theta)
        bad = bad | np.array([not self.field.valid(th) for th in Theta], dtype=bool)
        Bc = self.field_values(Theta)  # (K, N, 3)
        bmin, bmax = self.bnodes[0], self.bnodes[-1]
        reject = self.outside == "reject"
        cinc = self.cnodes[::-1]  # increasing cos-psi nodes; index j there is NC - 1 - j in cnodes
        W = np.zeros((K * P, NB * NC))
        wout = np.zeros((K, P))
        wtot = np.zeros((K, P))
        for k in range(K):
            for p in range(P):
                n = obs[k, p]
                mu = self.centres @ n
                vis = mu > 0
                w = self.areas[vis] * mu[vis] * (self.ld_a + self.ld_b * mu[vis])
                b, c = bmag_cospsi(Bc[k, vis], n)  # (Nv,)
                finite = np.isfinite(b)
                if not np.all(finite):
                    bad[k] = True
                b, c, w = b[finite], c[finite], w[finite]
                wtot[k, p] = w.sum()
                out_c = (b < bmin) | (b > bmax)
                wout[k, p] = w[out_c].sum()
                grid = np.zeros((NB, NC))
                keep = ~out_c if reject else np.ones(len(w), dtype=bool)
                if reject and np.any(out_c):
                    bad[k] = True
                bb = np.clip(b[keep], bmin, bmax)
                cc = c[keep]
                ww = w[keep]
                ib = _bracket(bb, self.bnodes)
                t = (bb - self.bnodes[ib]) / (self.bnodes[ib + 1] - self.bnodes[ib])
                jinc = _bracket(cc, cinc)  # cinc[jinc] <= c <= cinc[jinc + 1]
                jc = NC - 2 - jinc  # cnodes[jc] >= c >= cnodes[jc + 1]
                s = (self.cnodes[jc] - cc) / (self.cnodes[jc] - self.cnodes[jc + 1])
                if self.interp_b == "cubic":
                    wb = hermite_weights_reference(bb, self.bnodes)  # (Nv, NB)
                    for col, f in ((jc, 1 - s), (jc + 1, s)):  # grid.T[col] is the column of cos-psi node col
                        np.add.at(grid.T, col, (ww * f)[:, None] * wb)
                    W[k * P + p] = grid.ravel()
                    continue
                np.add.at(grid, (ib, jc), ww * (1 - t) * (1 - s))
                np.add.at(grid, (ib, jc + 1), ww * (1 - t) * s)
                np.add.at(grid, (ib + 1, jc), ww * t * (1 - s))
                np.add.at(grid, (ib + 1, jc + 1), ww * t * s)
                W[k * P + p] = grid.ravel()
        return W, wout, wtot, bad

    def _disc_sum(self, W, C, K, P):
        out = np.zeros((K, P, 2, self.M))
        for k in range(K):
            if C.ndim == 3:  # "planck" Teff weights per data pixel (DESIGN.md section 5c)
                rec = np.einsum("tm,tnsm->nsm", C[:, k], self.records)
            else:
                rec = np.einsum("t,tnsm->nsm", C[:, k], self.records)  # records interpolated in Teff for this row
            for p in range(P):
                out[k, p] = np.einsum("n,nsm->sm", W[k * P + p], rec)
        return out
