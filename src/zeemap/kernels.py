"""numba kernels of the forward model: the scatter of visible patches onto the library nodes.

Lookups: bilinear (`scatter_interp`, FWD-6) and cubic Hermite in B times linear in cos psi (`scatter_interp_cubic`,
FWD-15).

The scatter turns, for every parameter set k and phase p, the visible patches (|B|, cos psi, brightness weight) into
weights on the (NB, NC) library nodes: row k * P + p of W (K * P, NB * NC). It is the hot loop of an evaluation; the
disc sum itself is one matrix product W @ records (forward.py). Plain-numpy twin for V5a: forward_reference.py.

Conventions (DEV_INTERFACES.md):
- visible: mu = n . centre > 0; weight = area * mu * (ld_a + ld_b * mu) (FWD-1, FWD-2);
- cos psi = (B . n) / |B|, clipped to [-1, 1]; 0 when |B| = 0;
- B nodes increase; cos-psi nodes decrease from +1 to -1 (library.cos_psi_nodes);
- outside the library (|B| < b_min or > b_max at the patch centre): its weight is added to `wout` (LIB-6); with
  `reject` the parameter set is marked bad (LIB-5) and the patch is skipped, else B is clamped to the edge node;
- a non-finite field value marks the parameter set bad.
All arithmetic in float64; `fastmath` off; compiled once and cached on disk (`cache=True`).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def bracket_increasing(x, nodes):
    """i in [0, n - 2] with nodes[i] <= x <= nodes[i + 1] (x already inside [nodes[0], nodes[-1]])."""
    lo = 0
    hi = nodes.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nodes[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def bracket_decreasing(x, nodes):
    """j in [0, n - 2] with nodes[j] >= x >= nodes[j + 1] (x already inside [nodes[-1], nodes[0]])."""
    lo = 0
    hi = nodes.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nodes[mid] >= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bmag_cospsi(bx, by, bz, nx, ny, nz):
    b = math.sqrt(bx * bx + by * by + bz * bz)
    if b > 0.0:
        c = (bx * nx + by * ny + bz * nz) / b
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
    else:
        c = 0.0
    return b, c


@njit(cache=True)
def scatter_interp(Bc, centres, areas, obs, ld_a, ld_b, bnodes, cnodes, reject, W, wout, wtot, bad):
    """Bilinear lookup in B and cos psi (FWD-6).

    Bc (K, N, 3) field at the patch centres; centres (N, 3); areas (N,); obs (K, P, 3) observer directions;
    bnodes (NB,) increasing; cnodes (NC,) decreasing; reject: bool.
    Outputs (accumulated, zero them before): W (K * P, NB * NC), wout (K, P), wtot (K, P), bad (K,).
    """
    K, N = Bc.shape[0], Bc.shape[1]
    P = obs.shape[1]
    NC = cnodes.shape[0]
    bmin, bmax = bnodes[0], bnodes[bnodes.shape[0] - 1]
    for k in range(K):
        for p in range(P):
            nx, ny, nz = obs[k, p, 0], obs[k, p, 1], obs[k, p, 2]
            row = k * P + p
            for i in range(N):
                mu = nx * centres[i, 0] + ny * centres[i, 1] + nz * centres[i, 2]
                if not mu > 0.0:
                    continue
                w = areas[i] * mu * (ld_a + ld_b * mu)
                b, c = _bmag_cospsi(Bc[k, i, 0], Bc[k, i, 1], Bc[k, i, 2], nx, ny, nz)
                if not math.isfinite(b):
                    bad[k] = True
                    continue
                wtot[k, p] += w
                if b < bmin or b > bmax:
                    wout[k, p] += w
                    if reject:
                        bad[k] = True
                        continue
                    b = min(max(b, bmin), bmax)
                ib = bracket_increasing(b, bnodes)
                t = (b - bnodes[ib]) / (bnodes[ib + 1] - bnodes[ib])
                jc = bracket_decreasing(c, cnodes)
                s = (cnodes[jc] - c) / (cnodes[jc] - cnodes[jc + 1])
                base = ib * NC + jc
                W[row, base] += w * (1.0 - t) * (1.0 - s)
                W[row, base + 1] += w * (1.0 - t) * s
                W[row, base + NC] += w * t * (1.0 - s)
                W[row, base + NC + 1] += w * t * s


@njit(cache=True)
def scatter_interp_mu(Bc, centres, areas, obs, bnodes, cnodes, mus, reject, W, wout, wtot, wmu, bad):
    """`scatter_interp` for a mu-grid library (LIB-8; DESIGN.md section 5d): trilinear in (mu, B, cos psi), node index
    (imu * NB + ib) * NC + jc, patch weight area x mu (no limb-darkening law: the library holds the angular
    dependence). A patch below the smallest library mu takes that mu; its weight is added to wmu (K, P)."""
    K, N = Bc.shape[0], Bc.shape[1]
    P = obs.shape[1]
    NC = cnodes.shape[0]
    NB = bnodes.shape[0]
    NMU = mus.shape[0]
    bmin, bmax = bnodes[0], bnodes[NB - 1]
    for k in range(K):
        for p in range(P):
            nx, ny, nz = obs[k, p, 0], obs[k, p, 1], obs[k, p, 2]
            row = k * P + p
            for i in range(N):
                mu = nx * centres[i, 0] + ny * centres[i, 1] + nz * centres[i, 2]
                if not mu > 0.0:
                    continue
                w = areas[i] * mu
                b, c = _bmag_cospsi(Bc[k, i, 0], Bc[k, i, 1], Bc[k, i, 2], nx, ny, nz)
                if not math.isfinite(b):
                    bad[k] = True
                    continue
                wtot[k, p] += w
                if b < bmin or b > bmax:
                    wout[k, p] += w
                    if reject:
                        bad[k] = True
                        continue
                    b = min(max(b, bmin), bmax)
                m = mu
                if m < mus[0]:
                    wmu[k, p] += w
                    m = mus[0]
                if m > mus[NMU - 1]:
                    m = mus[NMU - 1]
                if NMU > 1:
                    im = bracket_increasing(m, mus)
                    r = (m - mus[im]) / (mus[im + 1] - mus[im])
                else:
                    im = 0
                    r = 0.0
                ib = bracket_increasing(b, bnodes)
                t = (b - bnodes[ib]) / (bnodes[ib + 1] - bnodes[ib])
                jc = bracket_decreasing(c, cnodes)
                s = (cnodes[jc] - c) / (cnodes[jc] - cnodes[jc + 1])
                for dm in range(2):
                    fm = (1.0 - r) if dm == 0 else r
                    if fm == 0.0:
                        continue
                    base = ((im + dm) * NB + ib) * NC + jc
                    W[row, base] += w * fm * (1.0 - t) * (1.0 - s)
                    W[row, base + 1] += w * fm * (1.0 - t) * s
                    W[row, base + NC] += w * fm * t * (1.0 - s)
                    W[row, base + NC + 1] += w * fm * t * s


@njit(cache=True)
def hermite_b_weights(b, bnodes, slopes, wb):
    """Cubic Hermite weights in B (FWD-15): fills wb[0:4] for the nodes j0 .. j0 + 3 and returns j0.

    b inside [bnodes[0], bnodes[-1]]; slopes (NB, NB): node slope m_i = sum_j slopes[i, j] f_j (forward.py,
    `hermite_slope_matrix`; row i has non-zeros only at i - 1 .. i + 1, or 0 .. 2 and NB - 3 .. NB - 1 at the ends).
    On [x_i, x_i+1], h = x_i+1 - x_i, t = (b - x_i) / h:
      f = h00 f_i + h01 f_i+1 + h (h10 m_i + h11 m_i+1),
      h00 = 2t^3 - 3t^2 + 1, h01 = -2t^3 + 3t^2, h10 = t^3 - 2t^2 + t, h11 = t^3 - t^2.
    The weights of the nodes i - 1 .. i + 2 (clipped to the node range; j0 = max(0, min(i - 1, NB - 4))) sum to 1.
    """
    nb = bnodes.shape[0]
    i = bracket_increasing(b, bnodes)
    h = bnodes[i + 1] - bnodes[i]
    t = (b - bnodes[i]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h01 = -2.0 * t3 + 3.0 * t2
    h10 = (t3 - 2.0 * t2 + t) * h
    h11 = (t3 - t2) * h
    j0 = i - 1
    if j0 > nb - 4:
        j0 = nb - 4
    if j0 < 0:
        j0 = 0
    for q in range(4):
        wb[q] = 0.0
    for q in range(4):
        j = j0 + q
        if j >= nb:
            break
        v = h10 * slopes[i, j] + h11 * slopes[i + 1, j]
        if j == i:
            v += h00
        elif j == i + 1:
            v += h01
        wb[q] = v
    return j0


@njit(cache=True)
def scatter_interp_cubic(Bc, centres, areas, obs, ld_a, ld_b, bnodes, slopes, cnodes, reject, W, wout, wtot, bad):
    """Cubic Hermite in B (up to 4 B nodes, `hermite_b_weights`) times linear in cos psi (2 nodes) (FWD-15).

    Arguments as in `scatter_interp`, plus slopes (NB, NB). Visibility, outside rule and cos psi as there.
    """
    K, N = Bc.shape[0], Bc.shape[1]
    P = obs.shape[1]
    NC = cnodes.shape[0]
    NB = bnodes.shape[0]
    nq = 4 if NB >= 4 else NB
    bmin, bmax = bnodes[0], bnodes[NB - 1]
    wb = np.zeros(4)
    for k in range(K):
        for p in range(P):
            nx, ny, nz = obs[k, p, 0], obs[k, p, 1], obs[k, p, 2]
            row = k * P + p
            for i in range(N):
                mu = nx * centres[i, 0] + ny * centres[i, 1] + nz * centres[i, 2]
                if not mu > 0.0:
                    continue
                w = areas[i] * mu * (ld_a + ld_b * mu)
                b, c = _bmag_cospsi(Bc[k, i, 0], Bc[k, i, 1], Bc[k, i, 2], nx, ny, nz)
                if not math.isfinite(b):
                    bad[k] = True
                    continue
                wtot[k, p] += w
                if b < bmin or b > bmax:
                    wout[k, p] += w
                    if reject:
                        bad[k] = True
                        continue
                    b = min(max(b, bmin), bmax)
                j0 = hermite_b_weights(b, bnodes, slopes, wb)
                jc = bracket_decreasing(c, cnodes)
                s = (cnodes[jc] - c) / (cnodes[jc] - cnodes[jc + 1])
                for q in range(nq):
                    base = (j0 + q) * NC + jc
                    W[row, base] += w * wb[q] * (1.0 - s)
                    W[row, base + 1] += w * wb[q] * s


@njit(cache=True)
def field_multipoles(points, centres, axes, strengths, out):
    """Field of up to three centred multipoles (l = 1, 2, 3) per row, at all points.

    points (N, 3); centres (K, 3) field centre per row; axes (3, K, 3) unit axis of component l = 1, 2, 3 per row;
    strengths (3, K) signed polar strength (0: the component is absent); out (K, N, 3), overwritten. The formula of
    `field.component_field`: B = a / ((l + 1) r^(l+2)) [((l + 1) P_l + mu P_l') r_hat - P_l' n], mu = n . r_hat, with
    P_1 = mu, P_2 = (3 mu^2 - 1) / 2, P_3 = (5 mu^3 - 3 mu) / 2 and their derivatives 1, 3 mu, (15 mu^2 - 3) / 2.
    """
    K, N = centres.shape[0], points.shape[0]
    for k in range(K):
        for i in range(N):
            x = points[i, 0] - centres[k, 0]
            y = points[i, 1] - centres[k, 1]
            z = points[i, 2] - centres[k, 2]
            r = math.sqrt(x * x + y * y + z * z)
            hx, hy, hz = x / r, y / r, z / r
            bx = by = bz = 0.0
            for c in range(3):
                a = strengths[c, k]
                if a == 0.0:
                    continue
                nx, ny, nz = axes[c, k, 0], axes[c, k, 1], axes[c, k, 2]
                mu = nx * hx + ny * hy + nz * hz
                l = c + 1
                if l == 1:
                    p, dp = mu, 1.0
                elif l == 2:
                    p, dp = 0.5 * (3.0 * mu * mu - 1.0), 3.0 * mu
                else:
                    p, dp = 0.5 * (5.0 * mu * mu * mu - 3.0 * mu), 0.5 * (15.0 * mu * mu - 3.0)
                pref = a / ((l + 1) * r ** (l + 2))
                f = (l + 1) * p + mu * dp
                bx += pref * (f * hx - dp * nx)
                by += pref * (f * hy - dp * ny)
                bz += pref * (f * hz - dp * nz)
            out[k, i, 0] = bx
            out[k, i, 1] = by
            out[k, i, 2] = bz
