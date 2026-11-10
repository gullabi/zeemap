"""V3m: families (SEARCH-9; DESIGN.md section 3b; families.py), on toy -ln L surfaces with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from zeemap.families import Linker, Node, family_settings, find_families, group

pytestmark = pytest.mark.tier1


class Toy:
    """Two free coordinates; f = two quadratic basins at (-2, 0) and (+2, 0), joined over a saddle of height s above
    the better minimum (the second minimum lies d0 above the first)."""

    names = ["p", "q"]
    free = np.array([True, True])
    step = np.array([1.0, 1.0])

    def __init__(self, s=8.0, d0=0.5, ridge=0.0):
        self.s, self.d0, self.ridge = s, d0, ridge

    def cost(self, X):
        X = np.atleast_2d(X)
        p, q = X[:, 0], X[:, 1]
        a = (p + 2.0) ** 2 + q**2
        b = (p - 2.0) ** 2 + q**2 + self.d0
        # smooth minimum of the two bowls, scaled so the barrier on the line q = 0 has height s at p = 0
        f = np.minimum(a, b) * self.s / 4.0
        if self.ridge:  # a narrow ridge across q = 0 at p = 0 that a path bending to |q| > 1 avoids
            f = f + self.ridge * np.exp(-(p / 0.2) ** 2) * np.exp(-(q / 0.6) ** 2)
        return f

    def mirror(self, x):
        return None

    def canonical(self, x):
        return np.asarray(x, dtype=float)


def _nodes(points, toy):
    return [Node(i, np.array(p, float), float(toy.cost(np.array(p, float))[0])) for i, p in enumerate(points)]


def test_v3m_two_basins_one_family_above_the_saddle_two_below():
    toy = Toy(s=8.0)
    nodes = _nodes([(-2.0, 0.0), (2.0, 0.0)], toy)
    lk = Linker(toy, family_settings({}))
    assert max(g for g in group(lk, nodes, 9.0, 100).component) == 0  # one family: level above the saddle
    g = group(lk, nodes, 7.0, 100)
    assert g.component == [0, 1]  # two families below it


def test_v3m_slope_points_join_their_basin():
    toy = Toy(s=30.0)
    pts = [(-2.0, 0.0), (-1.4, 0.3), (-2.5, -0.4), (2.0, 0.0), (2.3, 0.1), (2.6, 0.2)]
    nodes = _nodes(pts, toy)
    out = find_families(toy, nodes, value_tol=2.0, band=45.0, cfg={"search": {"family": {"min_level": 6.0}}})
    fam_of = out["mode_family"]
    assert fam_of[1] == fam_of[2] == fam_of[3] == 1
    assert fam_of[4] == fam_of[5] == 2
    assert fam_of[6] is None  # 6.75 nats above the best: above the 6-nat family level
    assert len(out["band_components"]) == 1  # 45 nats is above the 30-nat saddle


def test_v3m_midpoint_bends_around_a_ridge():
    toy = Toy(s=4.0, d0=0.0, ridge=50.0)  # straight path crosses a 50-nat ridge; going around costs little
    nodes = _nodes([(-0.6, 0.1), (0.6, 0.1)], toy)
    lk = Linker(toy, family_settings({}))
    ok, info = lk.linked(nodes[0].x, nodes[1].x, level=9.0)  # around the ridge the path tops out near 6.5
    assert ok and info["path"] == "midpoint"


def test_v3m_order_does_not_matter():
    toy = Toy(s=8.0)
    pts = [(-2.0, 0.0), (2.0, 0.0), (-1.7, 0.2), (2.3, -0.1)]
    a = find_families(toy, _nodes(pts, toy), 2.0, 45.0, {"search": {"family": {"min_level": 9.0}}})
    b = find_families(toy, _nodes(pts[::-1], toy), 2.0, 45.0, {"search": {"family": {"min_level": 9.0}}})
    sa = sorted(sorted(tuple(map(tuple, (n["x"] for n in f["nodes"])))) for f in a["families"])
    sb = sorted(sorted(tuple(map(tuple, (n["x"] for n in f["nodes"])))) for f in b["families"])
    assert sa == sb


class ToyMirror(Toy):
    """The same surface, symmetric under q -> -q, with that mirror as a symmetry of the problem."""

    def mirror(self, x):
        y = np.array(x, dtype=float)
        y[1] = -y[1]
        return y


def test_v3m_symmetric_copies_are_one_family():
    toy = ToyMirror(s=3.0, ridge=80.0)  # the direct path between q = +1 and q = -1 at p = 0 crosses the ridge
    a, b = np.array([0.05, 1.5]), np.array([0.05, -1.5])
    nodes = [Node(0, a, float(toy.cost(a)[0])), Node(1, b, float(toy.cost(b)[0]))]
    out = find_families(toy, nodes, 2.0, 45.0, {"search": {"family": {"min_level": 1.0}}})
    assert out["mode_family"][1] == out["mode_family"][2] == 1  # joined through the copy b -> mirror(b) = a


class ToyEdge(Toy):
    """The ridge toy next to a validity edge. Below q = `edge` the model is invalid (cost 1e30), as at
    the edge of the Teff slices or beyond |u| = 1; the two nodes lie on the edge, the straight path between them
    crosses the ridge, and the path bending to q > 1 stays low. With `limits`, the edge is also a polish limit
    (problem.lo / hi) and the region beyond it is valid and falls away steeply, so that only the limits keep the
    midpoint inside."""

    def __init__(self, edge=0.1, limits=False):
        super().__init__(s=4.0, d0=0.0, ridge=50.0)
        self.edge, self.limits = edge, limits
        self.dim, self.field_slice = 2, None
        if limits:
            self.lo, self.hi = np.array([-10.0, edge]), np.array([10.0, 10.0])
            self.value = np.zeros(2)

    def cost(self, X):
        X = np.atleast_2d(X)
        f = super().cost(X)
        below = X[:, 1] < self.edge
        if self.limits:
            return np.where(below, f - 1e3 * (self.edge - X[:, 1]), f)
        return np.where(below, 1e30, f)


@pytest.mark.parametrize("limits", [False, True])
def test_v3m_midpoint_at_a_validity_edge_or_a_limit(limits):
    """The re-polished midpoint uses the one-sided differences of the polish next to an invalid model and stays within
    the polish limits (`search.polish_bounds`), with the half-step `search.fd_step`. Before the fix (plain central
    differences, h = 1e-4 hard-coded, no limits) the midpoint of the pair below did not move off the edge (an invalid
    neighbour gave a gradient of about 5e33) and the pair was not linked; with limits, the midpoint left the range."""
    toy = ToyEdge(limits=limits)
    xa, xb = np.array([-0.6, 0.1]), np.array([0.6, 0.1])
    lk = Linker(toy, family_settings({}))
    assert lk.segment_max(xa, xb, 15) > 9.0  # the straight path crosses the ridge
    xm, fm = lk.midpoint(xa, xb)
    assert xm[1] >= 0.1  # inside the range
    ok, info = lk.linked(xa, xb, level=9.0)  # around the ridge the path tops out near 6.5
    assert ok and info["path"] == "midpoint", info


@pytest.mark.parametrize("max_pairs", [3000, 3])
def test_v3m_pool_gives_the_serial_grouping(max_pairs):
    """The link tests run in batches over the search's pool and are applied in the serial order, so the
    families, band components, links, pairs tested, cap and mode assignments equal the serial ones; only `nevals` may
    differ (pairs tested in a batch and joined meanwhile)."""
    toy = Toy(s=30.0)
    pts = [(-2.0, 0.0), (-1.4, 0.3), (-2.5, -0.4), (-1.8, -0.2), (2.0, 0.0), (2.3, 0.1), (1.7, -0.3), (2.6, 0.2)]
    cfg = {"search": {"family": {"min_level": 6.0, "max_pairs": max_pairs}}}
    a, b = (find_families(toy, _nodes(pts, toy), 2.0, 45.0, cfg, workers=w) for w in (1, 3))
    assert a["capped"] == (max_pairs == 3) and len(a["families"]) == (4 if a["capped"] else 2)
    assert b["nevals"] >= a["nevals"] > 0  # the workers' evaluations are counted
    a.pop("nevals"), b.pop("nevals")
    assert a == b


def test_v3m_cap_reached_exactly_is_not_capped():
    """With `max_pairs` equal to the pairs the uncapped grouping tests, nothing is left to test,
    so the grouping is complete and not capped, serially and in the pool."""
    from zeemap.search import make_pool

    toy = Toy(s=30.0)
    nodes = _nodes([(-2.0, 0.0), (-1.4, 0.3), (-2.5, -0.4), (-1.8, -0.2), (2.0, 0.0), (2.3, 0.1), (1.7, -0.3)], toy)
    lk = Linker(toy, family_settings({}))
    full = group(lk, nodes, 40.0, 3000)  # above the 30-nat saddle: one family, the farthest pairs already joined
    with make_pool(3) as pool:
        for p in (None, pool):
            g = group(lk, nodes, 40.0, full.pairs_tested, p, 3)
            assert not g.capped and g.component == full.component and g.pairs_tested == full.pairs_tested
