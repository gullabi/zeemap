"""V3k: symmetric copies of a geometry get the same fingerprint and land in one mode (SEARCH-2).

Uses the real "dqo" canonical form with a fake `fingerprint_quantities` that depends on the canonical parameters only
through quantities that are NOT themselves invariant under the symmetries (c_z, md_z, aq . md), so that the test fails
if `fingerprint` does not map to the canonical form first.
"""

import math

import numpy as np
import pytest

from zeemap import search as S
from zeemap.field import DQO

pytestmark = pytest.mark.tier1

MODEL = DQO()


class DQOFake:
    names = DQO.names
    dim = 14
    field_slice = slice(0, 14)

    def canonical(self, x):
        return MODEL.canonical(x)

    def fingerprint_quantities(self, x):
        md, bq, aq, mo, c, u = MODEL.split(x)
        nmd = float(np.linalg.norm(md))
        return {"incl": math.degrees(math.acos(min(1.0, u))), "c": c,
                "b50": np.array([nmd * (1 + 0.1 * c[2]) + 0.05 * bq * float(aq @ md) / nmd,
                                 nmd * (1 - 0.1 * c[2]) + float(np.linalg.norm(mo))]),
                "b90": np.array([nmd + abs(bq) + 20.0 * md[2] / nmd, nmd + 0.5 * abs(bq)])}


M = np.array([1.0, 1.0, -1.0])


def mirror(x):
    y = np.array(x, dtype=float)
    for a in (0, 4, 7, 10):
        y[a:a + 3] *= M
    y[13] = -y[13]
    return y


def reverse_aq(x, scale=1.0):
    y = np.array(x, dtype=float)
    y[4:7] *= -scale
    return y


def random_theta(rng):
    return np.concatenate([rng.normal(0, 300, 3), [rng.uniform(-250, 250)], rng.normal(0, 1, 3),
                           rng.normal(0, 100, 3), rng.uniform(-0.2, 0.2, 3), [rng.uniform(-1, 1)]])


def _copies(x):
    return [x, mirror(x), reverse_aq(x), reverse_aq(x, 2.5), mirror(reverse_aq(x, 0.4))]


def _assert_same(f1, f2):
    assert f1.incl == pytest.approx(f2.incl, abs=1e-10)
    for a, b in ((f1.c, f2.c), (f1.b50, f2.b50), (f1.b90, f2.b90)):
        assert np.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_v3k_symmetric_copies_same_fingerprint():
    p = DQOFake()
    rng = np.random.default_rng(31)
    for _ in range(50):
        x = random_theta(rng)
        fps = [S.fingerprint(p, y) for y in _copies(x)]
        for f in fps[1:]:
            _assert_same(fps[0], f)
            assert S.fingerprint_distance(fps[0], f) < 1e-10
        # not vacuous: without the canonical form the mirror changes the fake quantities
        raw = p.fingerprint_quantities(x), p.fingerprint_quantities(mirror(x))
        assert not np.allclose(raw[0]["c"], raw[1]["c"])


def test_v3k_symmetric_copies_land_in_one_mode():
    p = DQOFake()
    rng = np.random.default_rng(32)
    xa = np.concatenate([[300.0, 50.0, 200.0], [120.0], [0.3, -0.2, 0.9], [40.0, 10.0, -60.0], [0.05, -0.02, 0.1],
                         [0.6]])
    xb = xa.copy()
    xb[13] = 0.2  # another inclination: another mode
    eps = []
    for k, y in enumerate(_copies(xa) + _copies(xb)):
        f = 10.0 + 1e-3 * rng.standard_normal() + (30.0 if k >= 5 else 0.0)
        eps.append(S.EndPoint(y, f, k, 0, 1, True, False, True, fingerprint=S.fingerprint(p, y)))
    modes = S.group_modes(eps)
    assert len(modes) == 2
    assert sorted(modes[0].members) == [0, 1, 2, 3, 4] and modes[0].hits == 5
    assert sorted(modes[1].members) == [5, 6, 7, 8, 9]
    assert np.allclose(MODEL.canonical(modes[0].x), MODEL.canonical(xa), atol=1e-12)


# the absolute rule of DESIGN.md section 3 before decision #16 (relative = false with the research's thresholds)
ABSOLUTE = dict(S.FINGERPRINT_DEFAULTS, relative=False, b50=5.0, b90=25.0, b50_rms=10.0, angle="incl", extra={})


def test_v3k_thresholds_and_distance():
    a = S.Fingerprint(30.0, np.zeros(3), np.array([200.0, 210.0]), np.array([400.0, 380.0]))
    t = ABSOLUTE
    cases = [
        (S.Fingerprint(30.0 + 7.9, np.zeros(3), a.b50, a.b90), True),
        (S.Fingerprint(30.0 + 8.1, np.zeros(3), a.b50, a.b90), False),
        (S.Fingerprint(30.0, np.array([0.059, 0, 0]), a.b50, a.b90), True),
        (S.Fingerprint(30.0, np.array([0.04, 0.04, 0.04]), a.b50, a.b90), False),  # |dc| = 0.069
        (S.Fingerprint(30.0, np.zeros(3), a.b50 + [5.1, 0], a.b90), False),
        (S.Fingerprint(30.0, np.zeros(3), a.b50, a.b90 + [24.0, 100.0]), True),  # B90 only at the reference phase
    ]
    for f, same in cases:
        assert S.same_mode(a, f, t) is same
    # B50 rms over the 2 phases: 14 MG at one phase gives rms 9.9 (same mode), 14.5 MG gives 10.25 (not)
    f = S.Fingerprint(30.0, np.zeros(3), a.b50 + [0.0, 14.0], a.b90)
    assert S.fingerprint_distance(a, f, t) == pytest.approx(14.0 / math.sqrt(2) / 10.0)
    assert S.same_mode(a, f, t) is True
    assert S.same_mode(a, S.Fingerprint(30.0, np.zeros(3), a.b50 + [0.0, 14.5], a.b90), t) is False
    assert S.fingerprint_distance(a, S.Fingerprint(34.0, np.zeros(3), a.b50, a.b90), t) == pytest.approx(0.5)
    # thresholds from cfg["search"]["fingerprint"] override the defaults
    thr = S.fingerprint_thresholds({"search": {"fingerprint": dict(ABSOLUTE, incl=20.0)}})
    assert thr["incl"] == 20.0 and thr["c"] == 0.06
    assert S.same_mode(a, cases[1][0], thr) is True
    # the defaults (decision #16) are relative: 5.1 MG at B50 = 205 MG is within 0.04 x 205.1 = 8.2 MG
    assert S.same_mode(a, cases[4][0]) is True
    assert S.fingerprint_distance(a, cases[4][0]) == pytest.approx(5.1 / (0.04 * 205.1))
    # inclination folded to 0-90
    assert S._fold(170.0) == pytest.approx(10.0) and S._fold(-10.0) == pytest.approx(10.0)


def _fp(b50, b90=None, extra=None):
    b50 = np.atleast_1d(np.asarray(b50, dtype=float))
    return S.Fingerprint(30.0, np.zeros(3), b50, b50 * 2.0 if b90 is None else np.atleast_1d(b90), dict(extra or {}))


def test_v3k_relative_thresholds_both_regimes_and_floor():
    """Decision #16: relative = true (the default) scales the B thresholds with the larger B50; the absolute values
    are floors (0.5, 1.0, 1.0 MG by default)."""
    rel = S.fingerprint_thresholds({})
    assert rel["relative"] is True and rel["b50_rel"] == 0.04 and rel["b90_rel"] == 0.10 and rel["b50_rms_rel"] == 0.05
    assert (rel["b50"], rel["b90"], rel["b50_rms"]) == (0.5, 1.0, 1.0)
    assert S.fingerprint_thresholds(None) == rel
    # low field: 2.5 % apart is one mode, 8 % apart two (threshold 0.04 x 13 = 0.52 MG)
    assert S.same_mode(_fp(12.0, 30.0), _fp(12.3, 30.0), rel) is True
    assert S.same_mode(_fp(12.0, 30.0), _fp(13.0, 30.0), rel) is False
    assert S.fingerprint_distance(_fp(12.0, 30.0), _fp(13.0, 30.0), rel) == pytest.approx(1.0 / 0.52)
    # the floor binds below 12.5 MG: 5 vs 5.4 MG, threshold max(0.5, 0.04 x 5.4 = 0.216) = 0.5 MG
    assert S.fingerprint_distance(_fp(5.0, 30.0), _fp(5.4, 30.0), rel) == pytest.approx(0.4 / 0.5)
    # high field: 300 vs 310 MG, threshold 0.04 x 310 = 12.4 MG, one mode
    assert S.fingerprint_distance(_fp(300.0, 600.0), _fp(310.0, 600.0), rel) == pytest.approx(10.0 / 12.4)
    assert S.same_mode(_fp(300.0, 600.0), _fp(310.0, 600.0), rel) is True
    assert S.same_mode(_fp(300.0, 600.0), _fp(313.0, 600.0), rel) is False  # 13 > 0.04 x 313 = 12.52
    # B90 threshold 0.10 x 310 = 31 MG
    assert S.same_mode(_fp(310.0, 600.0), _fp(310.0, 630.0), rel) is True
    assert S.same_mode(_fp(310.0, 600.0), _fp(310.0, 632.0), rel) is False
    # larger floors (the research's 5 MG) bind at low field: 12 vs 13 MG stays one mode (threshold 5 MG)
    big = S.fingerprint_thresholds({"search": {"fingerprint": {"b50": 5.0, "b90": 25.0, "b50_rms": 10.0}}})
    assert S.fingerprint_distance(_fp(12.0, 30.0), _fp(13.0, 30.0), big) == pytest.approx(1.0 / 5.0)
    # the rms threshold scales too: 2 phases, 0 and 18 MG apart, rms 12.7 > 10 MG but < 0.05 x 310 = 15.5
    a, b = _fp([310.0, 300.0], [600.0, 600.0]), _fp([310.0, 318.0], [600.0, 600.0])
    assert S.same_mode(a, b, ABSOLUTE) is False and S.same_mode(a, b, rel) is True
    # relative = false leaves the absolute rule: 300 vs 310 MG is two modes with the research's 5 MG
    assert S.same_mode(_fp(300.0, 600.0), _fp(310.0, 600.0), ABSOLUTE) is False


def test_v3k_extra_parameters_enter_the_fingerprint():
    """Decision #16: extra = {teff = 1000} (the default) puts Teff into the fingerprint; unnamed extras are ignored."""
    thr = S.fingerprint_thresholds({})
    assert thr["extra"] == {"teff": 1000.0}
    a = _fp(300.0, extra={"teff": 13000.0, "other": 1.0})
    assert S.same_mode(a, _fp(300.0, extra={"teff": 14500.0, "other": 1.0}), thr) is False
    assert S.same_mode(a, _fp(300.0, extra={"teff": 13500.0, "other": 99.0}), thr) is True
    assert S.fingerprint_distance(a, _fp(300.0, extra={"teff": 14500.0}), thr) == pytest.approx(1.5)
    assert S.fingerprint_distance(a, _fp(300.0, extra={"teff": 14500.0})) == pytest.approx(1.5)  # thresholds None
    # a configuration's table replaces the default one: extra = {} ignores Teff; a name missing from one fingerprint
    # (a fixed Teff) is ignored
    none = S.fingerprint_thresholds({"search": {"fingerprint": {"extra": {}}}})
    assert S.same_mode(a, _fp(300.0, extra={"teff": 14500.0}), none) is True
    assert S.same_mode(a, _fp(300.0), thr) is True
    # the defaults table is not shared between calls
    thr["extra"]["teff"] = 1.0
    assert S.fingerprint_thresholds(None)["extra"] == {"teff": 1000.0} == S.FINGERPRINT_DEFAULTS["extra"]
    # fingerprint() takes the optional "extra" of fingerprint_quantities; to_dict writes it only when present
    p = _ToyExtra()
    f = S.fingerprint(p, np.array([13000.0]))
    assert f.extra == {"teff": 13000.0} and f.to_dict()["extra"] == {"teff": 13000.0}
    assert "extra" not in _fp(300.0).to_dict()
    # group_modes with end points 1500 K and 500 K apart in teff
    eps = [S.EndPoint(np.array([t]), fv, k, 0, 1, True, False, True, fingerprint=S.fingerprint(p, np.array([t])))
           for k, (t, fv) in enumerate([(13000.0, 0.0), (14500.0, 1.0), (13500.0, 2.0)])]
    modes = S.group_modes(eps, S.fingerprint_thresholds({}))
    assert [m.members for m in modes] == [[0, 2], [1]]
    assert len(S.group_modes(eps, none)) == 1


def test_v3k_auto_angle_and_centre_frame():
    """Decision #16: angle = "auto" compares beta and the centre in the frame of md and the observer for one phase,
    the inclination and the star-frame centre for more phases."""
    one = dict(b50=np.array([20.0]), b90=np.array([50.0]))
    f1 = S.Fingerprint(10.0, np.array([0.2, 0.0, 0.0]), beta=np.array([117.0]),
                       c_frame=np.array([[-0.22, 0.03, 0.0]]), **one)
    f2 = S.Fingerprint(70.0, np.array([0.0, -0.2, 0.1]), beta=np.array([118.0]),
                       c_frame=np.array([[-0.22, 0.02, 0.01]]), **one)
    assert S.angle_rule(f1, f2, "auto") == "beta"
    assert S.same_mode(f1, f2) is True  # auto: beta 1 deg, centre in the frame 0.014 R
    assert S.same_mode(f1, f2, dict(S.FINGERPRINT_DEFAULTS, angle="incl")) is False
    assert S.fingerprint_distance(f1, f2) == pytest.approx(max(1.0 / 8.0, math.sqrt(2e-4) / 0.06,
                                                               0.0, 0.0, 0.0))
    # without c_frame the star-frame centre is compared (|dc| = 0.3 R)
    g1, g2 = S.Fingerprint(10.0, f1.c, beta=f1.beta, **one), S.Fingerprint(70.0, f2.c, beta=f2.beta, **one)
    assert S.fingerprint_distance(g1, g2) == pytest.approx(math.sqrt(0.04 + 0.04 + 0.01) / 0.06)
    # two phases: auto = incl
    two = dict(b50=np.array([20.0, 21.0]), b90=np.array([50.0, 50.0]))
    h1 = S.Fingerprint(10.0, np.zeros(3), beta=np.array([117.0, 90.0]), **two)
    h2 = S.Fingerprint(70.0, np.zeros(3), beta=np.array([117.0, 90.0]), **two)
    assert S.angle_rule(h1, h2, "auto") == "incl" and S.same_mode(h1, h2) is False
    # no beta: auto = incl; an unknown choice is refused
    assert S.angle_rule(_fp(20.0), _fp(20.0), "auto") == "incl"
    with pytest.raises(ValueError):
        S.fingerprint_thresholds({"search": {"fingerprint": {"angle": "psi"}}})


class _ToyExtra:
    def canonical(self, x):
        return x

    def fingerprint_quantities(self, x):
        return {"incl": 30.0, "c": np.zeros(3), "b50": [300.0], "b90": [600.0], "extra": {"teff": float(x[0])}}
