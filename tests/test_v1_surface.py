"""V1b, V1c, V1d: known answers of the surfaces, the observer direction and limb darkening.

SPEC.md SURF-1, FWD-2; VERIFICATION.md V1b-V1d.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from zeemap.surface import (
    Surface,
    fibonacci,
    limb_darkening,
    make_surface,
    observer,
    observers,
    visible_weights,
)

pytestmark = pytest.mark.tier1

FOUR_PI = 4.0 * np.pi
LD_CLOSED = {0: np.pi, 1: 0.8 * np.pi, 3: 0.9 * np.pi}  # 2 pi int_0^1 (a + b mu) mu dmu = 2 pi (a/2 + b/3)


def _directions(k=100, seed=20261018):
    v = np.random.default_rng(seed).normal(size=(k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _projected_area_rel_errors(surface, dirs):
    mu = dirs @ surface.centres.T  # (K, N)
    s = np.where(mu > 0.0, mu, 0.0) @ surface.areas
    return (s - np.pi) / np.pi


# ---------------------------------------------------------------- V1b


@pytest.mark.parametrize("n", [1, 7, 5000, 20000])
def test_v1b_fibonacci_areas_sum_to_4pi(n):
    s = fibonacci(n)
    assert s.scheme == "fibonacci" and s.n == n
    assert s.centres.shape == (n, 3) and s.areas.shape == (n,)
    np.testing.assert_allclose(s.areas, FOUR_PI / n, rtol=0, atol=1e-15)
    assert abs(s.areas.sum() - FOUR_PI) <= 1e-12 * FOUR_PI


def test_v1b_centres_are_unit_vectors():
    np.testing.assert_allclose(np.linalg.norm(fibonacci(5000).centres, axis=1), 1.0, atol=1e-14)


def test_v1b_make_surface_reads_the_config():
    a = make_surface({"scheme": "fibonacci", "n": 100})
    assert isinstance(a, Surface) and a.scheme == "fibonacci" and a.n == 100
    with pytest.raises(ValueError, match="hexagonal"):
        make_surface({"scheme": "hexagonal", "n": 10})


# ---------------------------------------------------------------- V1c


# Measured (100 directions, seed 20261018, hard mu > 0 cut): n = 5000: max relative error 5.06e-5,
# rms 1.61e-5; n = 20000: max 8.10e-6, rms 2.61e-6; rms ratio 6.2 (VERIFICATION.md expects about 5.6).
def test_v1c_fibonacci_visible_projected_area_is_pi_at_5000():
    errs = _projected_area_rel_errors(fibonacci(5000), _directions())
    assert np.max(np.abs(errs)) <= 1e-4, f"max relative error {np.max(np.abs(errs)):.3e}"


def test_v1c_fibonacci_projected_area_error_falls_by_2_from_5000_to_20000():
    dirs = _directions()
    rms5 = np.sqrt(np.mean(_projected_area_rel_errors(fibonacci(5000), dirs) ** 2))
    rms20 = np.sqrt(np.mean(_projected_area_rel_errors(fibonacci(20000), dirs) ** 2))
    assert rms5 / rms20 >= 2.0, f"rms 5000: {rms5:.3e}, rms 20000: {rms20:.3e}, ratio {rms5 / rms20:.2f}"


def test_v1c_visible_weights_matches_the_definition():
    s = fibonacci(500)
    n = observer(0.3, 0.17)
    mu, w = visible_weights(s, n, 1)
    np.testing.assert_allclose(mu, s.centres @ n, atol=1e-15)
    want = np.where(mu > 0, s.areas * mu * (0.4 + 0.6 * mu), 0.0)
    np.testing.assert_allclose(w, want, rtol=1e-15, atol=0)
    assert np.all(w[mu <= 0] == 0)


def test_v1c_observer_known_directions():
    np.testing.assert_allclose(observer(1.0, 0.0), [0.0, 0.0, -1.0], atol=1e-15)
    np.testing.assert_allclose(observer(1.0, 0.37), [0.0, 0.0, -1.0], atol=1e-15)
    np.testing.assert_allclose(observer(0.0, 0.25), [0.0, -1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(observer(0.0, 0.0), [1.0, 0.0, 0.0], atol=1e-15)
    u, ph = 0.6, np.array([0.0, 0.1, 0.5, 0.9])
    many = observers(u, ph)
    assert many.shape == (4, 3)
    for k, p in enumerate(ph):
        np.testing.assert_allclose(many[k], observer(u, p), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(many, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(many[:, 2], -u, atol=1e-15)


# ---------------------------------------------------------------- V1d


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_v1d_limb_darkening_disc_integral_closed_form_quad(mode):
    val, _ = quad(lambda m: float(limb_darkening(m, mode)) * m, 0.0, 1.0)
    assert abs(2 * np.pi * val - LD_CLOSED[mode]) <= 1e-12


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_v1d_limb_darkening_disc_integral_fibonacci_20000(mode):
    s = fibonacci(20000)
    for n in _directions(10, seed=7):
        _, w = visible_weights(s, n, mode)
        assert abs(w.sum() - LD_CLOSED[mode]) / LD_CLOSED[mode] <= 1e-3


def test_v1d_limb_darkening_laws_and_bad_mode():
    mu = np.linspace(0, 1, 11)
    np.testing.assert_array_equal(limb_darkening(mu, 0), np.ones_like(mu))
    np.testing.assert_allclose(limb_darkening(mu, 1), 0.4 + 0.6 * mu, rtol=1e-15)
    np.testing.assert_allclose(limb_darkening(mu, 3), 0.7 + 0.3 * mu, rtol=1e-15)
    with pytest.raises(ValueError, match="2"):
        limb_darkening(mu, 2)
