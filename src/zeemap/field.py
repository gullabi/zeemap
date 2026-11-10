"""Field models "dqo", "dqo_aligned" and "harm4" (SPEC.md FIELD-1 to FIELD-4, PAR-1, PAR-2).

Formulas: DESIGN.md section 2. Interfaces: DEV_INTERFACES.md, "Parameter vector (field + geometry), field.py".

Units: field strengths in MG, lengths in stellar radii, angles in degrees.

Frames: the star's frame has z along the rotation axis. Angles (theta, phi) are polar angles: theta from +z, phi from
+x toward +y. A vector v given in a component's frame is R_z(phi) R_y(theta) v in the star's frame (right-handed
rotation matrices), and that component's axis is (sin theta cos phi, sin theta sin phi, cos theta).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import kernels
from .config import HARM_NAMES

AQ_LENGTH_WIDTH = 0.1  # width of the Gaussian penalty on ln|aq| (DESIGN.md section 4, step 0)

# ---------------------------------------------------------------------------------------------------------------------
# Rotations and axes


def _rz(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation(theta_deg: float, phi_deg: float) -> np.ndarray:
    """R_z(phi) R_y(theta): maps a vector given in the component's frame into the star's frame."""
    return _rz(math.radians(phi_deg)) @ _ry(math.radians(theta_deg))


def axis_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """(sin th cos ph, sin th sin ph, cos th); the third column of `rotation(theta_deg, phi_deg)`."""
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def angles_from_axis(axis) -> tuple[float, float]:
    """Inverse of `axis_from_angles` for any non-zero vector (its length is ignored).

    Returns (theta_deg, phi_deg) with theta in [0, 180] and phi in (-180, 180]. On the z axis phi is 0 by convention
    (the azimuth is undefined there). The zero vector gives (0, 0).
    """
    x, y, z = (float(v) for v in np.asarray(axis, dtype=float))
    rho = math.hypot(x, y)
    theta = math.degrees(math.atan2(rho, z))
    phi = math.degrees(math.atan2(y, x)) if rho > 0.0 else 0.0
    if phi <= -180.0:
        phi += 360.0
    return theta, phi


# ---------------------------------------------------------------------------------------------------------------------
# One centred axisymmetric multipole component of order l


def _legendre(l: int, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P_l(mu) and P_l'(mu): Bonnet's recurrence k P_k = (2k - 1) mu P_{k-1} - (k - 1) P_{k-2}, and the derivative
    recurrence P_k' = P_{k-2}' + (2k - 1) P_{k-1}, which (unlike l (mu P_l - P_{l-1}) / (mu^2 - 1)) holds at |mu| = 1."""
    p_prev, p = np.ones_like(mu), mu.copy()
    d_prev, d = np.zeros_like(mu), np.ones_like(mu)
    if l == 0:
        return p_prev, d_prev
    for k in range(2, l + 1):
        p_next = ((2 * k - 1) * mu * p - (k - 1) * p_prev) / k
        d_next = d_prev + (2 * k - 1) * p
        p_prev, p, d_prev, d = p, p_next, d, d_next
    return p, d


def _check_l(l: int) -> None:
    if not (isinstance(l, (int, np.integer)) and 1 <= l <= 5):
        raise ValueError(f"multipole order l must be an integer in 1..5, got {l!r}")


def _unit(axis) -> np.ndarray:
    n = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if not norm > 0.0:
        raise ValueError("component axis must be a non-zero vector")
    return n / norm


def component_potential(points, axis, a: float, l: int) -> np.ndarray:
    """Magnetic scalar potential of one centred component, with B = -grad Phi.

    Phi(r) = a P_l(cos theta) / ((l + 1) r^(l+1)),   cos theta = (n . r) / r,   n = axis / |axis|.

    Derivation of the normalisation. Write Phi = k a r^-(l+1) P_l(mu). On the axis (r = z > 0, mu = 1, P_l(1) = 1),
    Phi = k a z^-(l+1), so B_z = -dPhi/dz = k a (l + 1) z^-(l+2). Requiring B_z = a at the pole (z = 1) gives
    k = 1 / (l + 1). For l = 1 this is Phi = a z / (2 r^3), whose negative gradient is a (3 z r - r^2 e_z) / (2 r^5),
    the dipole row of DESIGN.md section 2.

    points: (N, 3) relative to the field centre, in any frame in which `axis` is given. Returns (N,).
    """
    _check_l(l)
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    n = _unit(axis)
    r = np.sqrt(np.einsum("ij,ij->i", p, p))
    mu = (p @ n) / r
    pl, _ = _legendre(l, mu)
    return a * pl / ((l + 1) * r ** (l + 1))


def component_field(points, axis, a: float, l: int) -> np.ndarray:
    """B = -grad Phi of `component_potential`, in MG, for l = 1..5.

    With r_hat = r / r, mu = n . r_hat and grad mu = (n - mu r_hat) / r:
      grad (r^-(l+1) P_l(mu)) = r^-(l+2) [ -(l+1) P_l r_hat + P_l' (n - mu r_hat) ],
    so
      B = a / ((l + 1) r^(l+2)) [ ((l + 1) P_l + mu P_l') r_hat - P_l' n ].
    At the pole (r = n, mu = 1, P_l' (1) = l (l+1) / 2) this is a n: B_z = a. The form is frame-free (only n and r
    enter), so no azimuth about the axis is needed. The table formulas of DESIGN.md section 2 (l = 1, 2, 3) follow
    from it; `_table_field_l123` writes them out for the tests.

    points: (N, 3) relative to the field centre. Returns (N, 3). A point at the centre gives inf/nan.
    """
    _check_l(l)
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    n = _unit(axis)
    r = np.sqrt(np.einsum("ij,ij->i", p, p))
    rhat = p / r[:, None]
    mu = rhat @ n
    pl, dpl = _legendre(l, mu)
    pref = a / ((l + 1) * r ** (l + 2))
    return pref[:, None] * (((l + 1) * pl + mu * dpl)[:, None] * rhat - dpl[:, None] * n[None, :])


def _table_field_l123(points_component_frame, a: float, l: int) -> np.ndarray:
    """The table of DESIGN.md section 2, written out literally (component frame, z along the axis). Tests only."""
    p = np.asarray(points_component_frame, dtype=float).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if l == 1:
        f = 3 * a * z / (2 * r**5)
        bz = a * (3 * z**2 - r2) / (2 * r**5)
    elif l == 2:
        f = a * (5 * z**2 - r2) / (2 * r**7)
        bz = a * z * (5 * z**2 - 3 * r2) / (2 * r**7)
    elif l == 3:
        f = 5 * a * z * (7 * z**2 - 3 * r2) / (8 * r**9)
        bz = a * (35 * z**4 - 30 * z**2 * r2 + 3 * r2**2) / (8 * r**9)
    else:
        raise ValueError("the table covers l = 1, 2, 3 only")
    return np.stack([f * x, f * y, bz], axis=1)


def _unit_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(v / |v|, |v|) per row of v (K, 3); a zero row gives the axis z."""
    s = np.sqrt(np.sum(v * v, axis=1))
    n = v / np.where(s > 0.0, s, 1.0)[:, None]
    n[s == 0.0] = (0.0, 0.0, 1.0)
    return n, s


def _multipoles(points, c, axes, strengths) -> np.ndarray:
    """(K, N, 3) field of the three components (axes (3, K, 3), signed strengths (3, K)) by the numba kernel."""
    pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
    out = np.empty((c.shape[0], pts.shape[0], 3))
    kernels.field_multipoles(pts, np.ascontiguousarray(c, dtype=float), np.ascontiguousarray(axes, dtype=float),
                             np.ascontiguousarray(strengths, dtype=float), out)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Models and registry


class FieldModel:
    """Interface of a field model (DEV_INTERFACES.md). Subclasses set name, names and dim."""

    name: str = ""
    names: list[str] = []
    dim: int = 0

    def evaluate(self, points, theta) -> np.ndarray:  # (N,3) star frame -> (N,3) MG
        raise NotImplementedError

    def evaluate_batch(self, points, Theta) -> np.ndarray:
        """(K, dim) -> (K, N, 3), by a loop over `evaluate`."""
        Theta = np.atleast_2d(np.asarray(Theta, dtype=float))
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.empty((Theta.shape[0], pts.shape[0], 3))
        for k, th in enumerate(Theta):
            out[k] = self.evaluate(pts, th)
        return out

    def canonical(self, theta) -> np.ndarray:
        raise NotImplementedError

    def mirror(self, theta) -> np.ndarray:
        """The equator-mirror image (Problem.mirror, search.settle_mirror): u and the parameters named *_z negated. A
        model whose mirror has other signs overrides it ("harm4")."""
        t = np.array(self._theta(theta), dtype=float)
        for i, n in enumerate(self.names):
            if n == "u" or n.endswith("_z"):
                t[i] = -t[i]
        return t

    def valid(self, theta) -> bool:
        raise NotImplementedError

    def _theta(self, theta) -> np.ndarray:
        t = np.asarray(theta, dtype=float).reshape(-1)
        if t.shape[0] != self.dim:
            raise ValueError(f"model {self.name!r} takes {self.dim} parameters, got {t.shape[0]}")
        return t


MODELS: dict[str, type[FieldModel]] = {}


def register(name: str) -> Callable[[type[FieldModel]], type[FieldModel]]:
    """Class decorator: MODELS[name] = cls, cls.name = name (also for toy models in tests)."""

    def deco(cls: type[FieldModel]) -> type[FieldModel]:
        cls.name = name
        MODELS[name] = cls
        return cls

    return deco


def get_model(name: str) -> FieldModel:
    """An instance of the registered model `name`."""
    try:
        return MODELS[name]()
    except KeyError:
        raise KeyError(f"unknown field model {name!r}; known: {', '.join(sorted(MODELS))}") from None


def _mirror(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[1], -v[2]])


@register("dqo")
class DQO(FieldModel):
    """Dipole + quadrupole + octupole, each with its own axis, one common centre (FIELD-1).

    theta (14): md (3, MG), bq (MG, signed), aq (3, direction only), mo (3, MG), c (3, stellar radii, star frame),
    u = cos i. `evaluate` ignores u (it is the observer's, used by the forward model).
    """

    names = ["md_x", "md_y", "md_z", "bq", "aq_x", "aq_y", "aq_z", "mo_x", "mo_y", "mo_z", "c_x", "c_y", "c_z", "u"]
    dim = 14

    @staticmethod
    def split(theta):
        t = np.asarray(theta, dtype=float)
        return t[0:3], t[3], t[4:7], t[7:10], t[10:13], t[13]

    def evaluate(self, points, theta) -> np.ndarray:
        return self.evaluate_batch(points, self._theta(theta)[None, :])[0]

    def evaluate_batch(self, points, Theta) -> np.ndarray:
        """(K, 14) -> (K, N, 3) by one numba loop over rows and points: dipole md, quadrupole bq along aq
        (absent when aq = 0), octupole mo, each as `component_field`."""
        T = np.atleast_2d(np.asarray(Theta, dtype=float))
        if T.shape[1] != self.dim:
            raise ValueError(f"model {self.name!r} takes {self.dim} parameters, got {T.shape[1]}")
        (nd, sd), (nq, sq), (no, so) = _unit_rows(T[:, 0:3]), _unit_rows(T[:, 4:7]), _unit_rows(T[:, 7:10])
        return _multipoles(points, T[:, 10:13], np.stack([nd, nq, no]),
                           np.stack([sd, np.where(sq > 0.0, T[:, 3], 0.0), so]))

    def valid(self, theta) -> bool:
        """Finite, |c| < 1 (the centre inside the star), and a non-zero aq when bq != 0 (its axis must exist)."""
        t = np.asarray(theta, dtype=float).reshape(-1)
        if t.shape[0] != self.dim or not np.all(np.isfinite(t)):
            return False
        md, bq, aq, mo, c, u = self.split(t)
        if not float(np.linalg.norm(c)) < 1.0:
            return False
        return not (bq != 0.0 and float(np.linalg.norm(aq)) == 0.0)

    def canonical(self, theta) -> np.ndarray:
        """DESIGN.md section 2: |aq| = 1, aq into z >= 0 (ties: x > 0, then y > 0), u >= 0.

        u < 0 is removed first by the equator mirror (negate the z components of md, aq, mo and c, and u; bq keeps its
        sign: under the mirror every component, with the same strength, gives B'(M p) = M B(p) with M = diag(1,1,-1),
        and the observer direction goes to M n, so B . n and |B| are unchanged). Then aq is normalised and reversed if
        needed (an even multipole is unchanged by reversing its axis, bq unchanged). A zero aq is left as it is.
        """
        md, bq, aq, mo, c, u = (np.array(v, dtype=float) for v in self.split(self._theta(theta)))
        if u < 0.0:
            md, aq, mo, c, u = _mirror(md), _mirror(aq), _mirror(mo), _mirror(c), -u
        naq = float(np.linalg.norm(aq))
        if naq > 0.0:
            aq = aq / naq
            if aq[2] < 0.0 or (aq[2] == 0.0 and (aq[0] < 0.0 or (aq[0] == 0.0 and aq[1] < 0.0))):
                aq = -aq
        return np.concatenate([md, [float(bq)], aq, mo, c, [float(u)]])


@register("dqo_aligned")
class DQOAligned(FieldModel):
    """Dipole + quadrupole + octupole on one axis, the dipole's (FIELD-2).

    theta (9): md (3, MG), bq (MG, signed), bo (MG, signed, along md's direction), c (3), u. Equal to "dqo" with
    aq = md / |md| and mo = bo md / |md|. If md = 0 the common axis is undefined: `evaluate` uses z as a placeholder
    and `valid` is False when bq or bo is non-zero.
    """

    names = ["md_x", "md_y", "md_z", "bq", "bo", "c_x", "c_y", "c_z", "u"]
    dim = 9

    @staticmethod
    def split(theta):
        t = np.asarray(theta, dtype=float)
        return t[0:3], t[3], t[4], t[5:8], t[8]

    @staticmethod
    def _axis(md: np.ndarray) -> np.ndarray:
        s = float(np.linalg.norm(md))
        return md / s if s > 0.0 else np.array([0.0, 0.0, 1.0])

    def evaluate(self, points, theta) -> np.ndarray:
        return self.evaluate_batch(points, self._theta(theta)[None, :])[0]

    def evaluate_batch(self, points, Theta) -> np.ndarray:
        """(K, 9) -> (K, N, 3) by the numba loop of "dqo"; the common axis is md's (z when md = 0)."""
        T = np.atleast_2d(np.asarray(Theta, dtype=float))
        if T.shape[1] != self.dim:
            raise ValueError(f"model {self.name!r} takes {self.dim} parameters, got {T.shape[1]}")
        n, s = _unit_rows(T[:, 0:3])
        return _multipoles(points, T[:, 5:8], np.stack([n, n, n]), np.stack([s, T[:, 3], T[:, 4]]))

    def valid(self, theta) -> bool:
        """Finite, |c| < 1, and md != 0 whenever bq or bo is non-zero (the common axis must exist; the z axis used
        for md = 0 is a placeholder that breaks the mirror and reversal symmetries)."""
        t = np.asarray(theta, dtype=float).reshape(-1)
        if t.shape[0] != self.dim or not np.all(np.isfinite(t)):
            return False
        if not float(np.linalg.norm(t[5:8])) < 1.0:
            return False
        return not ((t[3] != 0.0 or t[4] != 0.0) and float(np.linalg.norm(t[0:3])) == 0.0)

    def to_dqo(self, theta) -> np.ndarray:
        """The equivalent "dqo" vector (aq = axis, mo = bo * axis)."""
        md, bq, bo, c, u = self.split(self._theta(theta))
        n = self._axis(md)
        return np.concatenate([md, [bq], n, bo * n, c, [u]])

    def canonical(self, theta) -> np.ndarray:
        """u >= 0 via the equator mirror (negate the z components of md and c, and u; bq and bo unchanged)."""
        md, bq, bo, c, u = (np.array(v, dtype=float) for v in self.split(self._theta(theta)))
        if u < 0.0:
            md, c, u = _mirror(md), _mirror(c), -u
        return np.concatenate([md, [float(bq), float(bo)], c, [float(u)]])


# ---------------------------------------------------------------------------------------------------------------------
# Full multipole expansion up to l = 4 (FIELD-4)

#: (l, m, "g" or "h") of the l >= 2 Gauss coefficients of "harm4", in the order of config.HARM_NAMES
HARM_TERMS = [(int(n[1]), int(n[2]), n[0]) for n in HARM_NAMES]


def _harm_basis(points) -> np.ndarray:
    """(24, N, 3): the fields of the 24 "harm4" field numbers at unit value (md_x, md_y, md_z, then HARM_TERMS).

    B = -grad V with V = sum (g cos m phi + h sin m phi) P_l^m(cos theta) / r^(l+1) (R = 1; Langel 1987),
    P_l^m Schmidt semi-normalised without the Condon-Shortley phase. Per term with T = cos m phi or sin m phi:
    B_r = (l + 1) P T / r^(l+2), B_theta = -dP/dtheta T / r^(l+2), B_phi = -(P / sin theta) dT/dphi / r^(l+2).
    P, dP/dtheta and P / sin theta (m >= 1) come from the Schmidt recursion in l and its theta derivative, so no
    division by sin theta occurs; on the z axis phi = 0, the limit along that meridian. The l = 1 field equals a
    centred dipole md = 2 (g11, h11, g10), so md_x = 2 g11, md_y = 2 h11, md_z = 2 g10.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.sqrt(np.einsum("ij,ij->i", p, p))
    rho = np.hypot(p[:, 0], p[:, 1])
    x, s, phi = p[:, 2] / r, rho / r, np.arctan2(p[:, 1], p[:, 0])
    P, D, Q = {(0, 0): np.ones_like(x)}, {(0, 0): np.zeros_like(x)}, {}
    for m in range(0, 5):
        if m > 0:  # P_m^m from P_{m-1}^{m-1}; Q = P / sin theta
            f = 1.0 if m == 1 else math.sqrt((2 * m - 1) / (2 * m))
            pm, dm = P[(m - 1, m - 1)], D[(m - 1, m - 1)]
            P[(m, m)], D[(m, m)], Q[(m, m)] = f * s * pm, f * (x * pm + s * dm), f * pm
        for l in range(m + 1, 5):
            a, b = (2 * l - 1) / math.sqrt(l * l - m * m), math.sqrt(((l - 1) ** 2 - m * m) / (l * l - m * m))
            p1, d1 = P[(l - 1, m)], D[(l - 1, m)]
            p2, d2 = P.get((l - 2, m), 0.0), D.get((l - 2, m), 0.0)
            P[(l, m)], D[(l, m)] = a * x * p1 - b * p2, a * (x * d1 - s * p1) - b * d2
            if m > 0:
                Q[(l, m)] = a * x * Q[(l - 1, m)] - b * Q.get((l - 2, m), 0.0)
    cph, sph = np.cos(phi), np.sin(phi)
    er = np.stack([s * cph, s * sph, x], axis=1)
    et = np.stack([x * cph, x * sph, -s], axis=1)
    ep = np.stack([-sph, cph, np.zeros_like(x)], axis=1)

    def term(l, m, k):
        T, dT = (np.cos(m * phi), -m * np.sin(m * phi)) if k == "g" else (np.sin(m * phi), m * np.cos(m * phi))
        bp = -Q[(l, m)] * dT if m > 0 else np.zeros_like(x)
        b = ((l + 1) * P[(l, m)] * T)[:, None] * er - (D[(l, m)] * T)[:, None] * et + bp[:, None] * ep
        return b / r[:, None] ** (l + 2)

    dip = [0.5 * term(1, 1, "g"), 0.5 * term(1, 1, "h"), 0.5 * term(1, 0, "g")]
    return np.stack(dip + [term(l, m, k) for l, m, k in HARM_TERMS])


#: sign of each "harm4" number under the equator mirror z -> -z: (-1)^(l+m) for g_lm, h_lm (P_l^m(-x) = (-1)^(l+m)
#: P_l^m(x)); md_z and u change sign
HARM_MIRROR = np.array([1.0, 1.0, -1.0] + [(-1.0) ** (l + m) for l, m, _ in HARM_TERMS] + [-1.0])


@register("harm4")
class Harm4(FieldModel):
    """Full multipole expansion up to l = 4, centred (FIELD-4; Euchner et al. 2006, A&A 451, 671, Table 4; Jordan
    2003, astro-ph/0302058).

    theta (25): md (3, MG; the l = 1 part, md = 2 (g11, h11, g10)), the 21 Gauss coefficients HARM_NAMES in MG (star
    frame, z the rotation axis, Schmidt semi-normalised, B = -grad V; `_harm_basis`), u = cos i. The field is linear in
    the 24 field numbers: `evaluate_batch` is one product with the basis of the points, kept for the last points.
    Published sets do not state every convention (normalisation, Condon-Shortley phase, sign, observer side);
    provisionally (DESIGN.md section 2, evidence not a recorded run) a published set enters by negating its odd-m
    coefficients (Condon-Shortley), `rotated` (a tilted frame) and `mirror` with u kept (an observer on the +z side).
    """

    names = ["md_x", "md_y", "md_z", *HARM_NAMES, "u"]
    dim = 25

    def __init__(self):
        self._pts = None
        self._basis = None

    def _basis_for(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._pts is None or self._pts.shape != p.shape or not np.array_equal(self._pts, p):
            self._pts, self._basis = p.copy(), _harm_basis(p).reshape(24, -1)
        return self._basis

    def evaluate(self, points, theta) -> np.ndarray:
        return self.evaluate_batch(points, self._theta(theta)[None, :])[0]

    def evaluate_batch(self, points, Theta) -> np.ndarray:
        T = np.atleast_2d(np.asarray(Theta, dtype=float))
        if T.shape[1] != self.dim:
            raise ValueError(f"model {self.name!r} takes {self.dim} parameters, got {T.shape[1]}")
        basis = self._basis_for(points)
        return (T[:, :24] @ basis).reshape(T.shape[0], -1, 3)

    def valid(self, theta) -> bool:
        t = np.asarray(theta, dtype=float).reshape(-1)
        return t.shape[0] == self.dim and bool(np.all(np.isfinite(t)))

    def mirror(self, theta) -> np.ndarray:
        """The equator-mirror image (B'(M p) = M B(p), M = diag(1, 1, -1), u -> -u): same B . n and |B|."""
        return self._theta(theta) * HARM_MIRROR

    def canonical(self, theta) -> np.ndarray:
        """u >= 0 via the equator mirror."""
        t = self._theta(theta)
        return self.mirror(t) if t[-1] < 0.0 else t.copy()

    def rotated(self, theta, R) -> np.ndarray:
        """The numbers of the rotated field B'(p) = R B(R^T p) (u unchanged): a field given in a frame whose axes are
        the columns of R in the star's frame (e.g. `rotation(theta_deg, phi_deg)`). Exact up to rounding: each l is
        closed under rotations, so a least-squares fit on 200 points of the unit sphere recovers it."""
        t = self._theta(theta)
        R = np.asarray(R, dtype=float)
        pts = np.random.default_rng(0).normal(size=(200, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        A = _harm_basis(pts).reshape(24, -1).T
        b = (R @ self.evaluate(pts @ R, t).T).T.reshape(-1)
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
        return np.concatenate([coef, [t[-1]]])


def aq_length_penalty(theta) -> float:
    """Gauge penalty (ln|aq|)^2 / (2 * 0.1^2) on the length of the quadrupole axis of a "dqo" vector
    (DESIGN.md section 4, step 0). Returns inf for aq = 0."""
    aq = np.asarray(theta, dtype=float).reshape(-1)[4:7]
    n = float(np.linalg.norm(aq))
    if n == 0.0:
        return math.inf
    return math.log(n) ** 2 / (2.0 * AQ_LENGTH_WIDTH**2)
