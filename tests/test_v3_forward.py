"""V3a, V3c, V3d, V3f at the spectrum level: symmetries of the whole forward model.

V3a: the equator mirror (z of md, aq, mo and c negated, u negated) agrees within 1e-3 on the Fibonacci grid (the grid
is not mirror-symmetric, so the difference is quadrature noise).
V3c: reversing the whole field (md, bq, mo negated) flips V and V/I and leaves I (1e-12).
V3d: adding 1 to every phase, or adding a phase offset and subtracting it from the data phases: identical (1e-12).
V3f: "dqo_aligned" equals "dqo" with the axes tied (1e-12). Centred dipole, inclination i and dipole colatitude beta
swapped (same azimuth): the spectra depend only on a . n = sin i sin beta cos(alpha + 2 pi phase) - cos i cos beta,
which the swap keeps. The patch grid is fixed in the star, so in general the swap moves the patches relative to the
field: identical to the quadrature tolerance (1e-3, Fibonacci).
Hypothesis with a fixed seed; fewer cases than the field-level checks (each case costs a whole forward evaluation).
Tolerances are relative to the largest |I| (I, V) or absolute (V/I).
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zeemap import field
from zeemap.forward import Forward
from zeemap.library import Library
from zeemap.surface import Surface, fibonacci
from zeemap.types import DataSet, Grid, Spectrum

pytestmark = pytest.mark.tier1

SETTINGS = settings(max_examples=20, derandomize=True, deadline=None, database=None)
PHASES = (0.0, 0.2, 0.55, 0.8)


def dataset(phases=PHASES, start=4100.0, step=2.5, npix=1100):
    grid = Grid(start, step, npix)
    P, M = len(phases), npix
    sp = [Spectrum(0, float(phases[0]), "VI", np.arange(M), np.zeros(M), np.ones(M), np.ones(M), np.zeros(M, int))]
    nan = np.full((P, M), np.nan)
    return DataSet(grid, np.asarray(phases, float), sp, nan, nan.copy(), nan.copy(), nan.copy())


def cfg(**model):
    m = {"interp_b": "linear", "ld_mode": 3, "fwhm": 7.0, "phase_offset": 0.0}
    m.update(model)
    return {"model": m, "library": {"outside": "clamp"}}


@pytest.fixture(scope="module")
def lib(synth_lib):
    return Library(synth_lib, teffs=[30000.0])


@pytest.fixture(scope="module")
def fws(lib):
    dqo = field.get_model("dqo")
    ds = dataset()
    f = fibonacci(2000)  # and its mirror image in the equator: a mirror-symmetric patch set for the exact checks
    sym = Surface("fibonacci", np.vstack([f.centres, f.centres * [1.0, 1.0, -1.0]]), np.r_[f.areas, f.areas] / 2)
    return {
        ("fib", "interp"): Forward(lib, fibonacci(4000), dqo, ds, cfg()),
        ("sym", "interp"): Forward(lib, sym, dqo, ds, cfg()),
    }


def close(a, b, tol, sign=1.0):
    Ia, VIa, ba = a
    Ib, VIb, bb = b
    assert not ba.any() and not bb.any()
    scale = np.abs(Ia).max()
    np.testing.assert_allclose(Ia, Ib, rtol=0, atol=tol * scale)
    np.testing.assert_allclose(VIa, sign * VIb, rtol=0, atol=tol)
    np.testing.assert_allclose(VIa * Ia, sign * VIb * Ib, rtol=0, atol=tol * scale)


comp = st.floats(-1.0, 1.0)
thetas = st.tuples(
    st.tuples(comp, comp, comp), st.floats(-150.0, 150.0), st.tuples(comp, comp, comp),
    st.tuples(comp, comp, comp), st.tuples(*[st.floats(-0.15, 0.15)] * 3), st.floats(-1.0, 1.0),
    st.floats(150.0, 450.0), st.floats(0.0, 60.0),
)


def make_theta(t):
    md, bq, aq, mo, c, u, bp, bo = t
    md = np.asarray(md) + np.array([0.01, 0.02, 0.03])
    md = bp * md / np.linalg.norm(md)
    aq = np.asarray(aq) + np.array([0.0, 0.0, 1e-2])
    mo = np.asarray(mo) * bo
    return np.concatenate([md, [bq], aq, mo, c, [u]])


def mirror(th):
    m = th.copy()
    for i in (2, 6, 9, 12, 13):
        m[i] = -m[i]
    return m


@SETTINGS
@given(thetas)
def test_v3a_spec_equator_mirror(fws, t):
    th = make_theta(t)
    close(fws[("sym", "interp")].spectra(th), fws[("sym", "interp")].spectra(mirror(th)), 1e-12)


@settings(max_examples=5, derandomize=True, deadline=None, database=None)
@given(thetas)
def test_v3a_spec_equator_mirror_fibonacci_quadrature(fws, t):
    th = make_theta(t)
    fw = fws[("fib", "interp")]
    close(fw.spectra(th), fw.spectra(mirror(th)), 1e-3)


@SETTINGS
@given(thetas)
def test_v3c_spec_reversed_field_flips_v(fws, t):
    th = make_theta(t)
    rev = th.copy()
    rev[0:4] *= -1
    rev[7:10] *= -1
    fw = fws[("fib", "interp")]
    close(fw.spectra(th), fw.spectra(rev), 1e-12, sign=-1.0)


def test_v3d_phase_plus_one_and_phase_offset(lib):
    dqo = field.get_model("dqo")
    surf = fibonacci(500)
    base = Forward(lib, surf, dqo, dataset(), cfg())
    plus1 = Forward(lib, surf, dqo, dataset(tuple(p + 1.0 for p in PHASES)), cfg())
    off = 0.137
    shifted = Forward(lib, surf, dqo, dataset(tuple(p - off for p in PHASES)), cfg(phase_offset=off))
    rng = np.random.default_rng(3)
    for _ in range(5):
        th = make_theta((tuple(rng.uniform(-1, 1, 3)), rng.uniform(-150, 150), tuple(rng.uniform(-1, 1, 3)),
                         tuple(rng.uniform(-1, 1, 3)), tuple(rng.uniform(-0.15, 0.15, 3)), rng.uniform(-1, 1),
                         rng.uniform(150, 450), rng.uniform(0, 60)))
        ref = base.spectra(th)
        close(plus1.spectra(th), ref, 1e-12)
        close(shifted.spectra(th), ref, 1e-12)


@pytest.fixture(scope="module")
def aligned_pairs(lib):
    surf = fibonacci(200)
    return [(Forward(lib, surf, field.get_model("dqo_aligned"), dataset(), cfg()),
             Forward(lib, surf, field.get_model("dqo"), dataset(), cfg()))]


@SETTINGS
@given(st.tuples(comp, comp, comp), st.floats(-150.0, 150.0), st.floats(-60.0, 60.0),
       st.tuples(*[st.floats(-0.15, 0.15)] * 3), st.floats(-1.0, 1.0), st.floats(150.0, 450.0))
def test_v3f_spec_aligned_equals_dqo(aligned_pairs, md, bq, bo, c, u, bp):
    md = np.asarray(md) + np.array([0.01, 0.02, 0.03])  # never zero: the aligned axis must exist
    md = bp * md / np.linalg.norm(md)
    al = field.get_model("dqo_aligned")
    th_al = np.concatenate([md, [bq, bo], c, [u]])
    th_dqo = al.to_dqo(th_al)
    for f_al, f_dqo in aligned_pairs:
        close(f_al.spectra(th_al), f_dqo.spectra(th_dqo), 1e-12)


def centred_dipole(bp, incl_deg, beta_deg, alpha_deg):
    th = np.zeros(14)
    th[0:3] = bp * field.axis_from_angles(beta_deg, alpha_deg)
    th[4:7] = [0.0, 0.0, 1.0]
    th[13] = math.cos(math.radians(incl_deg))
    return th


@pytest.mark.parametrize("incl,alpha", [(60.0, 25.0), (35.0, -70.0), (80.0, 140.0)])
def test_v3f_spec_dipole_inclination_axis_swap_exact_case(fws, incl, alpha):
    """beta = 180 - i: the swap equals the equator mirror; a mirror-symmetric patch set gives identical spectra."""
    beta = 180.0 - incl
    fw = fws[("sym", "interp")]
    close(fw.spectra(centred_dipole(300.0, incl, beta, alpha)), fw.spectra(centred_dipole(300.0, beta, incl, alpha)), 1e-10)


@pytest.mark.parametrize("incl,beta,alpha", [(30.0, 70.0, 0.0), (50.0, 15.0, 120.0), (110.0, 40.0, -45.0)])
def test_v3f_spec_dipole_inclination_axis_swap_quadrature(lib, incl, beta, alpha):
    fw = Forward(lib, fibonacci(20000), field.get_model("dqo"), dataset(), cfg())
    close(fw.spectra(centred_dipole(300.0, incl, beta, alpha)), fw.spectra(centred_dipole(300.0, beta, incl, alpha)),
          1e-3)
