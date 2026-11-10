"""V1a and V1f at the field level: component formulas, gradients of the potentials, axes and angles."""

from types import SimpleNamespace

import numpy as np
import pytest

from zeemap import field
from zeemap.field import (
    DQO,
    DQOAligned,
    _table_field_l123,
    angles_from_axis,
    axis_from_angles,
    component_field,
    component_potential,
    rotation,
)
from zeemap.problem import Problem

pytestmark = pytest.mark.tier1


def _random_points(rng, n, rmin=0.6, rmax=2.0):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * rng.uniform(rmin, rmax, size=n)[:, None]


def _random_axis(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_v1a_component_matches_design_table(l):
    rng = np.random.default_rng(100 + l)
    for _ in range(20):
        axis = _random_axis(rng)
        a = rng.uniform(-300.0, 300.0)
        pts = _random_points(rng, 50)
        got = component_field(pts, axis, a, l)
        # the table in the component frame: points into it with R^T, the field back with R
        R = rotation(*angles_from_axis(axis))
        want = _table_field_l123(pts @ R, a, l) @ R.T
        scale = np.abs(want).max()
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_v1a_table_along_z_without_rotation(l):
    rng = np.random.default_rng(7 + l)
    pts = _random_points(rng, 200)
    a = 37.5
    np.testing.assert_allclose(component_field(pts, [0, 0, 1], a, l), _table_field_l123(pts, a, l), rtol=0, atol=1e-12 * a * 10)


@pytest.mark.parametrize("l", [1, 2, 3, 4, 5])
def test_v1a_field_is_minus_gradient_of_potential(l):
    rng = np.random.default_rng(200 + l)
    h = 1e-5
    for _ in range(10):
        axis = _random_axis(rng)
        a = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
        pts = _random_points(rng, 40, 0.8, 1.5)
        b = component_field(pts, axis, a, l)
        grad = np.empty_like(pts)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            grad[:, k] = (component_potential(pts + e, axis, a, l) - component_potential(pts - e, axis, a, l)) / (2 * h)
        np.testing.assert_allclose(b, -grad, rtol=0, atol=1e-7)


@pytest.mark.parametrize("l", [1, 2, 3, 4, 5])
def test_v1a_polar_field_equals_strength(l):
    axis = axis_from_angles(40.0, -75.0)
    b = component_field(axis[None, :], axis, 123.0, l)[0]
    np.testing.assert_allclose(b, 123.0 * axis, rtol=0, atol=1e-12 * 123.0)


def test_v1a_spec_example_dipole_pole_and_equator():
    """SPEC.md FIELD-3 example: centred 100 MG dipole along z: |B| = 100 at the pole, 50 at the equator."""
    m = DQO()
    theta = np.zeros(14)
    theta[2] = 100.0
    theta[4:7] = [0, 0, 1]
    theta[13] = 0.5
    b = m.evaluate(np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]), theta)
    np.testing.assert_allclose(np.linalg.norm(b, axis=1), [100.0, 50.0, 50.0], rtol=0, atol=1e-12)


def test_v1a_l_out_of_range_is_an_error():
    with pytest.raises(ValueError):
        component_field(np.ones((1, 3)), [0, 0, 1], 1.0, 6)


def test_v1f_axis_and_angles_inverse():
    rng = np.random.default_rng(5)
    for _ in range(200):
        th, ph = rng.uniform(0.1, 179.9), rng.uniform(-179.9, 180.0)
        t2, p2 = angles_from_axis(axis_from_angles(th, ph) * rng.uniform(0.1, 10.0))
        assert abs(t2 - th) < 1e-12 and abs(p2 - ph) < 1e-12
    assert angles_from_axis([0, 0, 1]) == (0.0, 0.0)
    assert angles_from_axis([-1, 0, 0]) == (90.0, 180.0)
    np.testing.assert_allclose(rotation(33.0, -50.0)[:, 2], axis_from_angles(33.0, -50.0), atol=1e-15)


def test_v1f_axis_worked_example():
    np.testing.assert_allclose(axis_from_angles(30.0, 60.0), [0.25, 0.433013, 0.866025], atol=1e-6)


def test_v1f_registry_and_missing_keys():
    assert isinstance(field.get_model("dqo"), DQO) and isinstance(field.get_model("dqo_aligned"), DQOAligned)
    assert len(DQO().names) == DQO.dim == 14 and len(DQOAligned().names) == DQOAligned.dim == 9
    with pytest.raises(KeyError):
        field.get_model("nope")
    th = np.zeros(14)
    th[4:7] = [0.0, 0.0, np.e]
    assert field.aq_length_penalty(th) == pytest.approx(1.0 / (2 * 0.01))


def _harm_potential(points, l, m, k):
    """V of one harm4 term, written independently with scipy: Schmidt factor times lpmv without its Condon-Shortley
    phase (-1)^m (Langel 1987)."""
    import math

    from scipy.special import lpmv

    p = np.asarray(points, dtype=float)
    r = np.linalg.norm(p, axis=1)
    phi = np.arctan2(p[:, 1], p[:, 0])
    schmidt = math.sqrt((2 - (m == 0)) * math.factorial(l - m) / math.factorial(l + m))
    t = np.cos(m * phi) if k == "g" else np.sin(m * phi)
    return schmidt * (-1) ** m * lpmv(m, l, p[:, 2] / r) * t / r ** (l + 1)


def test_v1a_harm4_is_minus_the_gradient_of_the_schmidt_potential():
    # each of the 24 field numbers at unit value against central differences of its potential (sign, Schmidt
    # normalisation, the theta derivative and B_phi); md = 2 (g11, h11, g10)
    pts = _random_points(np.random.default_rng(40), 60)
    m4 = field.get_model("harm4")
    terms = [(1, 1, "g", 0.5), (1, 1, "h", 0.5), (1, 0, "g", 0.5)] + [(l, m, k, 1.0) for l, m, k in field.HARM_TERMS]
    h = 1e-5
    for j, (l, m, k, scale) in enumerate(terms):
        th = np.zeros(25)
        th[j] = 1.0
        grad = np.stack([(_harm_potential(pts + h * e, l, m, k) - _harm_potential(pts - h * e, l, m, k)) / (2 * h)
                         for e in np.eye(3)], axis=1)
        np.testing.assert_allclose(m4.evaluate(pts, th), -scale * grad, atol=1e-7, err_msg=f"{k}{l}{m}")
    # on the z axis (no azimuth) the field is the limit along phi = 0
    th = np.random.default_rng(41).normal(size=25) * 50.0
    for z in (1.0, -1.0):
        np.testing.assert_allclose(m4.evaluate([[0.0, 0.0, z]], th), m4.evaluate([[1e-9, 0.0, z]], th), atol=1e-6)


def test_v1a_harm4_dipole_part_is_the_centred_dipole():
    pts = _random_points(np.random.default_rng(42), 50)
    md = np.array([30.0, -20.0, 100.0])
    th = np.zeros(25)
    th[:3] = md
    np.testing.assert_allclose(field.get_model("harm4").evaluate(pts, th),
                               component_field(pts, md, float(np.linalg.norm(md)), 1), rtol=1e-12, atol=1e-10)


def test_v1f_harm4_mirror_and_rotation():
    # mirror: B'(M p) = M B(p) (the signs (-1)^(l+m)); rotated: B'(p) = R B(R^T p) (Euchner's tilted-axis tables)
    rng = np.random.default_rng(43)
    pts = _random_points(rng, 50, 1.0, 1.0)
    m4 = field.get_model("harm4")
    th = rng.normal(size=25) * 50.0
    th[-1] = -0.4
    M = np.diag([1.0, 1.0, -1.0])
    np.testing.assert_allclose(m4.evaluate(pts @ M, m4.mirror(th)), m4.evaluate(pts, th) @ M, atol=1e-10)
    assert m4.canonical(th)[-1] == 0.4
    R = rotation(22.0, 191.0)
    np.testing.assert_allclose(m4.evaluate(pts, m4.rotated(th, R)), m4.evaluate(pts @ R, th) @ R.T, atol=1e-8)
    # the Problem uses the model's mirror, and with one phase every coefficient is a declared nuisance (decision #18)
    pr = SimpleNamespace(field=m4, field_slice=slice(0, 25), forward=SimpleNamespace(phases=[0.0]))
    x = np.append(th, 30000.0)
    np.testing.assert_array_equal(Problem.mirror(pr, x), np.append(m4.mirror(th), 30000.0))
    assert set(Problem.nuisance_coordinates.fget(pr)) == set(m4.names)
