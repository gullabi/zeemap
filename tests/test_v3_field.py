"""V3a, V3b, V3c, V3f at the field level: symmetries of "dqo" and "dqo_aligned"."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zeemap.field import DQO, DQOAligned, component_field

pytestmark = pytest.mark.tier1

SETTINGS = settings(max_examples=200, derandomize=True, deadline=None, database=None)
M = np.diag([1.0, 1.0, -1.0])


def _fib(n):
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    ph = np.pi * (1 + 5**0.5) * k
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=1)


PTS = _fib(64)  # surface points (stellar radius 1)

strength = st.floats(-800.0, 800.0, allow_nan=False)
unitish = st.floats(-1.0, 1.0, allow_nan=False)
vec = st.tuples(strength, strength, strength)
axis = st.tuples(unitish, unitish, unitish).filter(lambda v: np.linalg.norm(v) > 0.1)
centre = st.tuples(unitish, unitish, unitish).map(lambda v: 0.7 * np.array(v) / max(1.0, np.linalg.norm(v)))


@st.composite
def dqo_theta(draw):
    return np.concatenate([draw(vec), [draw(strength)], draw(axis), draw(vec), draw(centre), [draw(unitish)]])


@st.composite
def aligned_theta(draw):
    # md != 0: with md = 0 the common axis is undefined (DQOAligned.valid is False when bq or bo is non-zero)
    return np.concatenate([draw(vec.filter(lambda v: np.linalg.norm(v) > 1e-3)), [draw(strength), draw(strength)], draw(centre), [draw(unitish)]])


def _close(got, want):
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale)


@SETTINGS
@given(dqo_theta())
def test_v3a_field_equator_mirror_dqo(theta):
    m = DQO()
    mir = theta.copy()
    for s in (slice(0, 3), slice(4, 7), slice(7, 10), slice(10, 13)):
        mir[s] = M @ theta[s]
    mir[13] = -theta[13]
    _close(m.evaluate(PTS @ M, mir), m.evaluate(PTS, theta) @ M)
    # canonical form: u >= 0, unit aq with aq.z >= 0, the same field up to the mirror
    can = m.canonical(theta)
    assert can[13] >= 0.0 and can[6] >= 0.0 and abs(np.linalg.norm(can[4:7]) - 1.0) < 1e-12
    pts, b = (PTS @ M, m.evaluate(PTS, theta) @ M) if theta[13] < 0 else (PTS, m.evaluate(PTS, theta))
    _close(m.evaluate(pts, can), b)
    np.testing.assert_allclose(m.canonical(can), can, rtol=0, atol=1e-15)


@SETTINGS
@given(aligned_theta())
def test_v3a_field_equator_mirror_aligned(theta):
    m = DQOAligned()
    mir = theta.copy()
    mir[0:3] = M @ theta[0:3]
    mir[5:8] = M @ theta[5:8]
    mir[8] = -theta[8]
    _close(m.evaluate(PTS @ M, mir), m.evaluate(PTS, theta) @ M)
    assert m.canonical(theta)[8] >= 0.0 and m.valid(theta)
    assert not m.valid(np.r_[0.0, 0.0, 0.0, theta[3], 1.0, theta[5:]])  # octupole without an axis


def test_v3a_field_mirror_keeps_b_dot_n_and_modulus():
    """Spectra-relevant quantities: |B| and B . n for the mirrored observer (n -> M n) agree."""
    m = DQO()
    theta = np.array([30.0, -40.0, 200.0, -70.0, 0.3, 0.2, -0.9, 10.0, 5.0, -60.0, 0.1, -0.2, 0.3, -0.4])
    can = m.canonical(theta)
    n = np.array([0.5, -0.3, 0.8])
    b, bc = m.evaluate(PTS, theta), m.evaluate(PTS @ M, can)
    _close(np.linalg.norm(bc, axis=1), np.linalg.norm(b, axis=1))
    _close(bc @ (M @ n), b @ n)


@SETTINGS
@given(dqo_theta(), st.floats(0.2, 5.0))
def test_v3b_field_quadrupole_axis_reversed_or_rescaled(theta, scale):
    m = DQO()
    rev = theta.copy()
    rev[4:7] = -scale * theta[4:7]
    _close(m.evaluate(PTS, rev), m.evaluate(PTS, theta))


@SETTINGS
@given(dqo_theta())
def test_v3c_field_reversed_components_dqo(theta):
    m = DQO()
    neg = theta.copy()
    neg[0:4] = -theta[0:4]
    neg[7:10] = -theta[7:10]
    _close(m.evaluate(PTS, neg), -m.evaluate(PTS, theta))


@SETTINGS
@given(aligned_theta())
def test_v3c_field_reversed_components_aligned(theta):
    m = DQOAligned()
    neg = theta.copy()
    neg[0:3] = -theta[0:3]  # reverses the common axis: the octupole along it flips with it, bo keeps its sign
    neg[3] = -theta[3]
    _close(m.evaluate(PTS, neg), -m.evaluate(PTS, theta))


@SETTINGS
@given(aligned_theta())
def test_v3f_field_aligned_equals_dqo_with_equal_axes(theta):
    a, d = DQOAligned(), DQO()
    md = theta[0:3]
    n = md / np.linalg.norm(md)
    full = np.concatenate([md, [theta[3]], n, theta[4] * n, theta[5:8], [theta[8]]])
    _close(a.evaluate(PTS, theta), d.evaluate(PTS, full))
    _close(d.evaluate(PTS, a.to_dqo(theta)), d.evaluate(PTS, full))


@SETTINGS
@given(vec.filter(lambda v: np.linalg.norm(v) > 1e-3), axis, st.floats(0.5, 2.0))
def test_v3f_field_centred_dipole_formula(md, aq, rscale):
    md = np.array(md)
    pts = PTS * rscale
    a = np.linalg.norm(md)
    n = md / a
    r = np.linalg.norm(pts, axis=1)
    rh = pts / r[:, None]
    want = a * (3 * (rh @ n)[:, None] * rh - n[None, :]) / (2 * r[:, None] ** 3)
    theta = np.concatenate([md, [0.0], aq, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3]])
    _close(DQO().evaluate(pts, theta), want)
    _close(DQOAligned().evaluate(pts, np.concatenate([md, [0.0, 0.0], [0.0, 0.0, 0.0], [0.3]])), want)
    _close(component_field(pts, n, a, 1), want)


def test_v3f_field_evaluate_batch_equals_loop():
    m = DQO()
    rng = np.random.default_rng(1)
    T = rng.normal(size=(5, 14)) * 50
    T[:, 10:13] = 0.1
    got = m.evaluate_batch(PTS, T)
    assert got.shape == (5, len(PTS), 3)
    for k in range(5):
        np.testing.assert_array_equal(got[k], m.evaluate(PTS, T[k]))
    assert m.valid(T[0]) and not m.valid(np.r_[T[0][:10], [1.0, 0.0, 0.0], T[0][13]])


def test_v3f_batched_field_equals_component_sum():
    """`evaluate_batch` of "dqo" and "dqo_aligned" is one numba loop (`kernels.field_multipoles`); it equals the sum of
    `component_field` per component (1e-12 relative), with zero dipole, quadrupole or octupole strengths, a negative
    bq, a zero aq with bq = 0, and offsets."""
    rng = np.random.default_rng(4)
    T = rng.normal(size=(6, 14)) * 80
    T[:, 10:13] = rng.uniform(-0.3, 0.3, (6, 3))
    T[1, 7:10] = 0.0  # no octupole
    T[2, 3] = -120.0  # negative bq
    T[3, 3], T[3, 4:7] = 0.0, 0.0  # no quadrupole, zero axis
    T[4, 0:3] = 0.0  # no dipole

    def comp(v, l, r):
        s = float(np.linalg.norm(v))
        return np.zeros_like(r) if s == 0.0 else component_field(r, v / s, s, l)

    got = DQO().evaluate_batch(PTS, T)
    for k, t in enumerate(T):
        r = PTS - t[10:13]
        want = comp(t[0:3], 1, r) + comp(t[7:10], 3, r)
        if t[3] != 0.0:
            want = want + component_field(r, t[4:7] / np.linalg.norm(t[4:7]), t[3], 2)
        np.testing.assert_allclose(got[k], want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
    Ta = np.c_[T[:, 0:5], T[:, 10:14]]
    Ta[4, 3:5] = 0.0  # md = 0: bq and bo must be 0 (valid)
    got = DQOAligned().evaluate_batch(PTS, Ta)
    for k, t in enumerate(Ta):
        r = PTS - t[5:8]
        s = float(np.linalg.norm(t[0:3]))
        n = t[0:3] / s if s > 0 else np.array([0.0, 0.0, 1.0])
        want = comp(t[0:3], 1, r) + component_field(r, n, t[3], 2) + component_field(r, n, t[4], 3)
        np.testing.assert_allclose(got[k], want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
