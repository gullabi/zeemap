"""V1e, V1n: known answers of the forward model's small parts.

V1e: the library lookup at a node returns that node; halfway between two nodes (in B, in cos psi) the mean of the two
(node weights, 1e-12). The cubic interpolation in B
(`interp_b = "cubic"`, alias "pchip"): at a node the node, exact for data linear (and quadratic) in B,
weights sum to 1 on at most 4 B nodes, continuous first derivative at the nodes (1e-8).
V1n: resampling a linear spectrum onto pixel centres on whole or half angstroms gives its value at the centres
(1e-12); the convolution of a constant spectrum away from the ends is that constant (1e-12).
"""

import math

import numpy as np
import pytest

from zeemap import field
from zeemap.forward import Forward, convolution_resampling_matrix, kernel, resampling_matrix
from zeemap.library import Library
from zeemap.surface import fibonacci
from zeemap.types import DataSet, Grid, Spectrum

pytestmark = pytest.mark.tier1


@field.register("toy_const")
class ToyConst(field.FieldModel):
    """The same field vector (bx, by, bz) on every patch; u = cos i."""

    names = ["bx", "by", "bz", "u"]
    dim = 4

    def evaluate(self, points, theta):
        t = self._theta(theta)
        return np.tile(t[:3], (np.asarray(points).reshape(-1, 3).shape[0], 1))

    def valid(self, theta):
        return bool(np.all(np.isfinite(theta)))


def dataset(phases=(0.0,), start=4100.0, step=2.5, npix=1100):
    grid = Grid(start, step, npix)
    P, M = len(phases), npix
    sp = [Spectrum(0, float(phases[0]), "VI", np.arange(M), np.zeros(M), np.ones(M), np.ones(M), np.zeros(M, int))]
    nan = np.full((P, M), np.nan)
    return DataSet(grid, np.asarray(phases, float), sp, nan, nan.copy(), nan.copy(), nan.copy())


def cfg(**model):
    m = {"interp_b": "linear", "ld_mode": 3, "fwhm": 7.0, "phase_offset": 0.0}
    m.update(model)
    return {"model": m, "library": {"outside": "clamp"}}


def pole_on(b, cospsi):
    """toy_const vector with |B| = b and B . n / |B| = cospsi for the pole-on observer n = (0, 0, -1) (u = 1)."""
    return np.array([b * math.sqrt(max(0.0, 1.0 - cospsi * cospsi)), 0.0, -b * cospsi, 1.0])


@pytest.fixture(scope="module")
def lib(synth_lib):
    return Library(synth_lib, teffs=[30000.0])


def normalised_weights(fw, theta):
    w = fw.node_weights(theta)[0]
    return w / w.sum()


def test_v1e_lookup_at_a_node_returns_that_node(lib):
    fw = Forward(lib, fibonacci(500), field.get_model("toy_const"), dataset(), cfg())
    for ib, jc in [(0, 0), (12, 4), (30, 8), (7, 12), (20, 16)]:
        w = normalised_weights(fw, pole_on(fw.bnodes[ib], fw.cnodes[jc]))
        want = np.zeros_like(w)
        want[ib, jc] = 1.0
        np.testing.assert_allclose(w, want, rtol=0, atol=1e-12)


def test_v1e_interp_halfway_in_b_and_in_cospsi_is_the_mean(lib):
    fw = Forward(lib, fibonacci(500), field.get_model("toy_const"), dataset(), cfg())
    b, c = fw.bnodes, fw.cnodes
    for ib, jc in [(3, 2), (17, 9), (29, 15)]:
        w = normalised_weights(fw, pole_on(0.5 * (b[ib] + b[ib + 1]), c[jc]))
        want = np.zeros_like(w)
        want[ib, jc] = want[ib + 1, jc] = 0.5
        np.testing.assert_allclose(w, want, rtol=0, atol=1e-12)
        w = normalised_weights(fw, pole_on(b[ib], 0.5 * (c[jc] + c[jc + 1])))
        want = np.zeros_like(w)
        want[ib, jc] = want[ib, jc + 1] = 0.5
        np.testing.assert_allclose(w, want, rtol=0, atol=1e-12)
        # a quarter of the way in both: bilinear weights
        w = normalised_weights(fw, pole_on(0.75 * b[ib] + 0.25 * b[ib + 1], 0.75 * c[jc] + 0.25 * c[jc + 1]))
        want = np.zeros_like(w)
        want[ib, jc], want[ib + 1, jc], want[ib, jc + 1], want[ib + 1, jc + 1] = 0.5625, 0.1875, 0.1875, 0.0625
        np.testing.assert_allclose(w, want, rtol=0, atol=1e-12)


# ---- V1e, cubic interpolation in B (FWD-15): node weights of the cubic Hermite with three-point slopes


@pytest.fixture(scope="module")
def fw_cubic(lib):
    return Forward(lib, fibonacci(200), field.get_model("toy_const"), dataset(), cfg(interp_b="cubic"))


def test_v1e_cubic_pchip_is_an_alias(lib, fw_cubic):
    fw = Forward(lib, fibonacci(200), field.get_model("toy_const"), dataset(), cfg(interp_b="pchip"))
    assert fw.interp_b == "cubic" and fw_cubic.interp_b == "cubic"
    th = pole_on(0.3 * fw.bnodes[5] + 0.7 * fw.bnodes[6], 0.2)
    np.testing.assert_array_equal(fw.node_weights(th), fw_cubic.node_weights(th))
    with pytest.raises(ValueError, match="interp_b"):
        Forward(lib, fibonacci(100), field.get_model("toy_const"), dataset(), cfg(interp_b="spline"))


def test_v1e_cubic_at_a_node_returns_that_node(fw_cubic):
    fw = fw_cubic
    for ib, jc in [(0, 0), (1, 3), (12, 4), (30, 8), (7, 12), (fw.NB - 1, 16), (fw.NB - 2, 2)]:
        w = normalised_weights(fw, pole_on(fw.bnodes[ib], fw.cnodes[jc]))
        want = np.zeros_like(w)
        want[ib, jc] = 1.0
        np.testing.assert_allclose(w, want, rtol=0, atol=1e-12)


def b_weights(fw, b, jc=0):
    """B weights (NB,) at cos psi = node jc (pole-on, toy field): node weights summed over cos psi, normalised.
    jc = 0 (cos psi = 1): B = (0, 0, -b), so |B| = b without rounding (needed for the 1e-6 differences)."""
    w = fw.node_weights(pole_on(b, fw.cnodes[jc]))[0]
    return w.sum(axis=1) / w.sum()


def test_v1e_cubic_weights_sum_to_one_and_use_at_most_four_b_nodes(fw_cubic):
    x = fw_cubic.bnodes
    for b in np.linspace(x[0], x[-1], 157):
        wb = b_weights(fw_cubic, b)
        assert wb.sum() == pytest.approx(1.0, abs=1e-12)
        nz = np.flatnonzero(wb)
        assert len(nz) <= 4 and nz.max() - nz.min() <= 3


def test_v1e_cubic_reproduces_linear_and_quadratic_data_in_b(fw_cubic):
    """Exact for data linear in B (V1e; 1e-12 on the weights), and for quadratics (three-point slopes)."""
    x = fw_cubic.bnodes
    rng = np.random.default_rng(3)
    bs = np.concatenate([rng.uniform(x[0], x[-1], 60), [x[0] + 1e-3, x[-1] - 1e-3, 0.5 * (x[0] + x[1])]])
    for b in bs:
        wb = b_weights(fw_cubic, b)
        assert wb @ x == pytest.approx(b, abs=1e-12 * x[-1])
        assert wb @ (x - b) ** 2 == pytest.approx(0.0, abs=1e-10 * (x[-1] - x[0]) ** 2)


def test_v1e_cubic_has_a_continuous_first_derivative_at_the_nodes(fw_cubic, lib):
    """One-sided derivatives of the node weights at each interior node, second-order differences with
    e = 1e-6: left and right agree to 1e-8 (times the node spacing). The linear interpolation jumps there."""
    x = fw_cubic.bnodes
    lin = Forward(lib, fibonacci(200), field.get_model("toy_const"), dataset(), cfg())
    e = 1e-6

    def d_left_right(fw, b):
        w = {s: b_weights(fw, b + s * e) for s in (-2, -1, 0, 1, 2)}
        right = (-3 * w[0] + 4 * w[1] - w[2]) / (2 * e)
        left = (3 * w[0] - 4 * w[-1] + w[-2]) / (2 * e)
        return left, right

    for ib in [1, 2, 7, 15, len(x) // 2, len(x) - 3, len(x) - 2]:
        h = 0.5 * (x[ib + 1] - x[ib - 1])
        left, right = d_left_right(fw_cubic, x[ib])
        np.testing.assert_allclose(left * h, right * h, rtol=0, atol=1e-8)
        assert np.abs(left).max() > 0.1 / h  # not trivially zero
    left, right = d_left_right(lin, x[7])
    assert np.abs(left - right).max() * (x[8] - x[7]) > 0.5


@pytest.mark.parametrize("start,step", [(4100.0, 2.5), (4100.5, 2.5), (4100.5, 2.0), (4100.0, 1.0), (4100.5, 1.0)])
def test_v1n_resampling_a_linear_spectrum_gives_its_value_at_the_centres(start, step):
    lam = np.arange(4000.0, 7000.0 + 1e-9, 1.0)
    f = 3.0 + 2e-4 * (lam - 5000.0)
    grid = Grid(start, step, 500)
    R, ok = resampling_matrix(lam, grid.wavelengths, step)
    assert ok.all()
    np.testing.assert_allclose(R @ f, 3.0 + 2e-4 * (grid.wavelengths - 5000.0), rtol=1e-12, atol=0)


def test_v1n_resampling_worked_example():
    """2.5 A pixels on 1 A cells: 3502.5 overlaps four cells, 3505.0 three (exact bin averages)."""
    lam = np.arange(3400.0, 3600.0 + 1e-9, 1.0)
    rng = np.random.default_rng(0)
    f = rng.normal(size=lam.size)
    R, _ = resampling_matrix(lam, np.array([3502.5, 3505.0]), 2.5)
    i = lambda x: int(x - 3400.0)  # noqa: E731
    want0 = (0.25 * f[i(3501)] + f[i(3502)] + f[i(3503)] + 0.25 * f[i(3504)]) / 2.5
    want1 = (0.75 * f[i(3504)] + f[i(3505)] + 0.75 * f[i(3506)]) / 2.5
    np.testing.assert_allclose(R @ f, [want0, want1], rtol=1e-13)


def test_v1n_convolution_of_a_constant_is_that_constant():
    lam = np.arange(4000.0, 7000.0 + 1e-9, 1.0)
    grid = Grid(4100.0, 2.5, 1100)
    T, ok = convolution_resampling_matrix(lam, grid, 7.0)
    np.testing.assert_allclose(T @ np.full(lam.size, 2.5), 2.5, rtol=1e-12)
    k = kernel(7.0, 1.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-14)
    assert len(k) == 2 * math.floor(4 * 7.0 / 2.3548200450309493) + 1

