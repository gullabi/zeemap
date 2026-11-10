"""V1q: the Jacobian of the Levenberg-Marquardt finish (decision #25; DESIGN.md section 3, "Polish").

The LM finish uses central differences with a half-step of `search.fd_step` (1e-4) parameter steps, computed in one
batched forward call (`search._jacobian_batched`, `Problem.residuals_batch`). Checked here: the batched Jacobian equals
the same differences taken one coordinate at a time through `Problem.residuals` (rtol 1e-9; the absolute floor is
1e-9 of the largest element or 4 d / (2h), d the measured batch-vs-single rounding of the residuals, whichever is
larger), and it agrees with a Richardson
extrapolation from half-steps of 1e-3 and 5e-4 (an O(h^4) reference where the model is smooth) to 1e-3 relative in
the Frobenius norm (the reference's larger steps cross the kinks of the piecewise-linear interpolation in B and
cos psi at some patches: measured 1.7e-4), on the
synthetic library at 3 random geometries with the full likelihood and Teff free with both Teff weight rules.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zeemap import cli
from zeemap import search as S
from zeemap.config import load_config
from zeemap.problem import Problem

from test_v7_cli import config_text

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def problems(synth_lib, tmp_path_factory):
    return _problems(synth_lib, tmp_path_factory.mktemp("jac_data"))


def _problems(libdir, root):
    os.symlink(libdir, root / "synth", target_is_directory=True)
    teff = 'teff = { value = 32000.0, step = 500.0, lo = 20000.0, hi = 40000.0, free = true }\n'
    txt = config_text("inj", kind="full").replace("teffs = [30000.0]", "teffs = [20000.0, 30000.0, 40000.0]")
    txt = txt.replace("[search]\n", teff + "\n[search]\n").replace("windows = [[4300.0, 6700.0]]",
                                                                    "windows = [[4400.0, 5200.0]]")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "t.toml").write_text(txt)
        assert cli.main(["inject", str(root / "t.toml"), "-o", str(root / "inj"), "--noise", "white", "--seed",
                         "3"]) == 0
        for mode in ("planck", "log"):
            cfg = load_config(root / "t.toml")
            cfg["model"]["interp_teff"] = mode
            out[mode] = Problem(cfg)
    return out


def _jac(problem, x, idx, h):
    got = S._jacobian_batched(problem, x, idx, np.asarray(problem.step, float)[idx], h, {"n": 0})
    assert got is not None
    return got[1]


@pytest.mark.parametrize("mode", ["planck", "log"])
def test_v1q_batched_jacobian_equals_differences_and_richardson(problems, mode):
    p = problems[mode]
    idx = np.flatnonzero(p.free)
    s = np.asarray(p.step, float)[idx]
    rng = np.random.default_rng(7)
    for _ in range(3):
        x = np.array(p.value, float)
        x[idx] += 0.5 * s * rng.standard_normal(idx.size)
        h = float(p.cfg["search"]["fd_step"])
        J = _jac(p, x, idx, h)
        scales = p.residual_scales(x)
        # Batch-vs-single rounding d: the same 2n+1 rows as _jacobian_batched, evaluated once as the batch and once one
        # row at a time. Multithreaded BLAS rounds a GEMM differently for different row counts (measured d ~ 1e-13 on
        # the 14-core workstation, 0 with OPENBLAS_NUM_THREADS=1), and the difference quotient amplifies that by
        # 1/(2h). A column takes two rows, so it can differ from the single-row column by at most 2 d / (2h); the floor
        # below is 4 d / (2h) (factor 2 margin). d itself is held to 1e-11 of the largest residual, so a real
        # batch-vs-single disagreement still fails here.
        X = np.repeat(x[None, :], 2 * idx.size + 1, axis=0)
        X[1 + np.arange(idx.size), idx] += h * s
        X[1 + idx.size + np.arange(idx.size), idx] -= h * s
        Rb = p.residuals_batch(X, scales)
        d = max(float(np.max(np.abs(Rb[i] - p.residuals_batch(X[i][None], scales)[0]))) for i in range(X.shape[0]))
        assert d <= 1e-11 * float(np.max(np.abs(Rb)))
        atol = max(1e-9 * np.max(np.abs(J)), 4.0 * d / (2 * h))
        # one coordinate at a time through Problem.residuals (the anchor holds the noise scales)
        r0 = p.residuals(x)
        for k, j in enumerate(idx):
            xp, xm = x.copy(), x.copy()
            xp[j] += h * s[k]
            xm[j] -= h * s[k]
            col = (p.residuals_batch(xp[None], scales)[0] - p.residuals_batch(xm[None], scales)[0]) / (2 * h)
            np.testing.assert_allclose(J[:, k], col, rtol=1e-9, atol=atol)
        assert r0.shape[0] == J.shape[0]
        # Richardson: (4 D(h/2) - D(h)) / 3 from h = 1e-3
        D1, D2 = _jac(p, x, idx, 1e-3), _jac(p, x, idx, 5e-4)
        ref = (4.0 * D2 - D1) / 3.0
        assert np.linalg.norm(J - ref) <= 1e-3 * np.linalg.norm(ref)


class EdgeBowl:
    """Linear residuals A (x - x*) whose unconstrained minimum x* lies beyond the upper limit of coordinate 2 and below
    the lower limit of coordinate 0; outside the limits the model is invalid (residuals_batch fills the row with
    INVALID_RESIDUAL, as Problem does at the edge of the Teff slices or at u = 1)."""

    def __init__(self):
        rng = np.random.default_rng(11)
        self.dim = 4
        self.A = np.diag([2.0, 1.0, 0.7, 1.5]) + 0.2 * rng.standard_normal((4, 4))
        self.xs = np.array([-1.5, 0.4, 2.0, -0.3])
        self.names = [f"q{i}" for i in range(4)]
        self.free = np.ones(4, dtype=bool)
        self.lo = np.array([-1.0, -5.0, -5.0, -5.0])
        self.hi = np.array([5.0, 5.0, 1.0, 5.0])
        self.step = np.array([0.1, 0.2, 0.05, 0.3])
        self.value = np.zeros(4)
        self.field_slice = None

    def _inside(self, X):
        return np.all((X >= self.lo) & (X <= self.hi), axis=1)

    def cost(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        R = (X - self.xs) @ self.A.T
        return np.where(self._inside(X), 0.5 * np.sum(R * R, axis=1), S.WORST)

    def residuals(self, x):
        return self.residuals_batch(np.asarray(x, dtype=float)[None], self.residual_scales(x))[0]

    def residual_scales(self, x):
        return np.ones(1) if self._inside(np.asarray(x, dtype=float)[None])[0] else None

    def residuals_batch(self, X, scales):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        R = (X - self.xs) @ self.A.T
        R[~self._inside(X)] = S.INVALID_RESIDUAL
        return R

    def canonical(self, x):
        return np.array(x, dtype=float)

    def fingerprint_quantities(self, x):
        return {"incl": 45.0, "c": np.zeros(3), "b50": np.zeros(1), "b90": np.zeros(1)}


def test_v1q_jacobian_one_sided_at_the_limits_and_lm_reaches_the_bound_minimum():
    from scipy.optimize import lsq_linear

    from zeemap.config import defaults

    p = EdgeBowl()
    idx = np.arange(4)
    s = p.step
    h = 1e-4
    blo, bhi = S.polish_bounds(p)
    # a point on both limits: coordinate 0 at its lower, coordinate 2 at its upper limit
    x = np.array([-1.0, 0.1, 1.0, 0.2])
    got = S._jacobian_batched(p, x, idx, s, h, {"n": 0}, blo, bhi)
    assert got is not None
    r, J = got
    # the exact Jacobian in the scaled coordinates is A diag(s); one-sided differences reproduce it for linear residuals
    exact = p.A * s[None, :]
    assert np.all(np.abs(J) < 1e3)  # no INVALID_RESIDUAL leaks into a column
    np.testing.assert_allclose(J, exact, rtol=1e-8, atol=1e-10)
    # (1e-9: a one-row and a batched matrix product round differently)
    np.testing.assert_allclose(J[:, 0], (p.residuals(x + h * s * np.eye(4)[0]) - r) / h, rtol=1e-9)
    np.testing.assert_allclose(J[:, 2], (r - p.residuals(x - h * s * np.eye(4)[2])) / h, rtol=1e-9)
    # the Levenberg-Marquardt finish alone, from that corner, reaches the bound-constrained minimum
    truth = lsq_linear(p.A, p.A @ p.xs, bounds=(p.lo, p.hi), tol=1e-14)
    f_true = 0.5 * float(np.sum((p.A @ (truth.x - p.xs)) ** 2))
    # (both LM stops off: this checks the finish itself; decision #37's defaults are checked after it)
    cfg = defaults()
    cfg["search"].update(lbfgs=False, lm_stop_nat=0.0, lm_max_rejections=0)
    pr = S.polish(p, x, cfg)
    assert pr.lm_used and pr.converged
    assert pr.f == pytest.approx(f_true, abs=1e-9)
    np.testing.assert_allclose(pr.x, truth.x, atol=1e-5)
    # with the default stops the finish ends converged within search.lm_stop_nat of that minimum
    cfg = defaults()
    cfg["search"]["lbfgs"] = False
    pr = S.polish(p, x, cfg)
    assert pr.lm_used and pr.converged
    assert -1e-12 <= pr.f - f_true <= cfg["search"]["lm_stop_nat"]


def _polish_gradient(problem, x, monkeypatch):
    """The gradient that `search.polish` hands to L-BFGS-B at x (scaled coordinates): scipy's minimize is replaced by a
    stub that evaluates the polish's own objective once at the start and returns."""
    seen = {}

    class _Res:
        pass

    def stub(fun, z0, jac=True, method=None, bounds=None, options=None):
        seen["f"], seen["g"] = fun(np.asarray(z0, dtype=float))
        r = _Res()
        r.x, r.fun, r.success = np.asarray(z0, dtype=float), seen["f"], True
        return r

    from zeemap.config import defaults

    cfg = defaults()
    cfg["search"]["lm_maxiter"] = 0
    cfg["search"]["fd_step"] = problem.cfg["search"]["fd_step"]
    with monkeypatch.context() as mp:
        mp.setattr(S, "minimize", stub)
        S.polish(problem, x, cfg)
    return seen["g"]


@pytest.fixture(scope="module")
def problems_depth(tmp_path_factory):
    """As `problems`, on a library whose line depth grows with Teff: I(T) = 1 - (T / 30 000 K) (1 - I_a(30 000 K)),
    V(T) = (T / 30 000 K) V_a(30 000 K) (I_a, V_a: the synthetic library, variant a). In variant a Teff only scales the
    spectra, which the continuum takes up, so with "log" weights -ln L does not depend on Teff at all."""
    import synth

    from zeemap.library import write_slice

    d = tmp_path_factory.mktemp("synth_depth")
    inf = synth.write_library(d, "a")
    base = synth.make_slice(30000.0, "a", inf)
    for t in inf.teffs:
        a = base.copy()
        g = np.float32(t / 30000.0)
        a[:, :, 0] = 1.0 - g * (1.0 - base[:, :, 0])
        a[:, :, 1] = g * base[:, :, 1]
        write_slice(d, inf, float(t), a)
    return _problems(d, tmp_path_factory.mktemp("jac_depth_data"))


@pytest.mark.parametrize("mode", ["planck", "log"])
def test_v1q_teff_derivative_not_zero_at_a_slice_or_a_limit(problems_depth, mode, monkeypatch):
    """Regression: `Library.interp_weights` once snapped a Teff within 0.5 K onto a slice, so with `fd_step` 1e-4
    (here a Teff half-step of 0.05 K) both points of the central difference landed on the same slice and the Teff
    column of the LM Jacobian and the Teff component of the L-BFGS-B gradient were exactly 0 within about 0.4 K of
    every slice, the polish limits (the edge slices) included. Both must be nonzero there and of the size measured
    2 K away (inside the limits), where the snap never reached."""
    p = problems_depth[mode]
    idx = np.flatnonzero(p.free)
    s = np.asarray(p.step, float)[idx]
    h = float(p.cfg["search"]["fd_step"])
    blo, bhi = S.polish_bounds(p)
    kt = int(np.flatnonzero(idx == p.iteff)[0])
    assert blo[p.iteff] == 20000.0 and bhi[p.iteff] == 40000.0  # the polish limits are the edge slices
    for t, t_ref in ((30000.0, 30002.0), (30000.3, 30002.0), (20000.0, 20002.0), (20000.4, 20002.0),
                     (40000.0, 39998.0), (39999.7, 39998.0)):
        cols, grads = [], []
        for tt in (t, t_ref):
            x = np.array(p.value, float)
            x[p.iteff] = tt
            got = S._jacobian_batched(p, x, idx, s, h, {"n": 0}, blo, bhi)
            assert got is not None
            cols.append(np.linalg.norm(got[1][:, kt]))
            grads.append(abs(float(_polish_gradient(p, x, monkeypatch)[kt])))
        assert cols[0] > 0.2 * cols[1] > 0.0, (t, cols)
        assert grads[0] > 0.2 * grads[1] > 0.0, (t, grads)


@pytest.mark.parametrize("mode", ["planck", "log"])
def test_v1q_teff_limits_beyond_an_edge_slice_are_clamped(problems_depth, mode, monkeypatch):
    """Regression: `parameters.teff` may reach up to 0.5 K beyond the first or last selected slice (the tolerance
    of `Library.check_teff`), where the model is that edge slice, constant in Teff. Polish limits there let the polish
    sit on a flat piece (Teff column and gradient exactly 0) and gave the edge stencil half the slope. The Teff limits
    are clamped to the slice range at load: with lo = 19 999.6 and hi = 40 000.4 the polish limits are the edge slices,
    and at each limit the Teff column of the LM Jacobian and the L-BFGS-B gradient are of the size measured 2 K
    inside."""
    import copy

    cfg = copy.deepcopy(problems_depth[mode].cfg)
    cfg["parameters"]["teff"].update(lo=19999.6, hi=40000.4)
    p = Problem(cfg)
    idx = np.flatnonzero(p.free)
    s = np.asarray(p.step, float)[idx]
    h = float(p.cfg["search"]["fd_step"])
    blo, bhi = S.polish_bounds(p)
    assert (blo[p.iteff], bhi[p.iteff]) == (20000.0, 40000.0)
    kt = int(np.flatnonzero(idx == p.iteff)[0])
    for t, t_ref in ((blo[p.iteff], 20002.0), (bhi[p.iteff], 39998.0)):
        cols, grads = [], []
        for tt in (t, t_ref):
            x = np.array(p.value, float)
            x[p.iteff] = tt
            got = S._jacobian_batched(p, x, idx, s, h, {"n": 0}, blo, bhi)
            assert got is not None
            cols.append(np.linalg.norm(got[1][:, kt]))
            grads.append(abs(float(_polish_gradient(p, x, monkeypatch)[kt])))
        assert cols[0] > 0.2 * cols[1] > 0.0, (t, cols)
        assert grads[0] > 0.2 * grads[1] > 0.0, (t, grads)


def test_v1q_teff_range_wholly_beyond_an_edge_slice(problems_depth):
    """Regression: the clamp of the test above once clamped lo and hi on their own, so a Teff range lying wholly beyond an edge slice
    (within `check_teff`'s 0.5 K, so accepted) gave lo > hi: free, lo 40 000.2 and hi 40 000.4 (edge slice 40 000)
    became 40 000.2 / 40 000 and the first polish raised "An upper bound is less than the corresponding lower bound";
    fixed at 40 000.3 the problem had lo > hi with the value outside. The value and both limits are now clamped into
    the slice range, so such a range becomes lo = hi = value = the edge slice (the model there was that slice anyway),
    and the problem keeps a note, written to run.json (`limit_notes`). A value alone beyond an edge (19 999.7) is
    clamped too; limits inside the range give no note."""
    import copy

    from zeemap.report import parameter_space

    base = problems_depth["log"].cfg
    assert parameter_space(problems_depth["log"])["limit_notes"] == []
    for kw, edge in ((dict(lo=40000.2, hi=40000.4, value=40000.3, free=True), 40000.0),
                     (dict(lo=40000.3, hi=40000.3, value=40000.3, free=False), 40000.0),
                     (dict(lo=19999.6, hi=19999.8, value=19999.7, free=True), 20000.0)):
        cfg = copy.deepcopy(base)
        cfg["parameters"]["teff"].update(**kw)
        p = Problem(cfg)
        i = p.iteff
        assert p.lo[i] == p.hi[i] == p.value[i] == edge, (kw, p.lo[i], p.hi[i], p.value[i])
        notes = parameter_space(p)["limit_notes"]
        assert len(notes) == 1 and "parameters.teff" in notes[0] and f"{edge:g}" in notes[0], notes
        c = copy.deepcopy(cfg)
        c["search"].update(lbfgs_maxiter=2, lm_maxiter=1)
        pr = S.polish(p, p.value, c)  # raised ValueError before the fix (free case)
        assert pr.x[i] == edge and np.isfinite(pr.f)

    cfg = copy.deepcopy(base)
    cfg["parameters"]["teff"].update(lo=20000.0, hi=40000.0, value=19999.7)
    p = Problem(cfg)
    assert (p.lo[p.iteff], p.hi[p.iteff], p.value[p.iteff]) == (20000.0, 40000.0, 20000.0)
    assert len(parameter_space(p)["limit_notes"]) == 1


def test_v1q_jacobian_is_one_forward_call(problems, monkeypatch):
    """The LM Jacobian takes the noise scales from row 0 of its own batch (x itself), so it is
    one forward call of 2n + 1 rows; before, `residual_scales` evaluated x once more (2n + 2 rows in two calls). The
    columns stay those of V1q above, which uses explicit scales."""
    p = problems["log"]
    idx = np.flatnonzero(p.free)
    x = np.array(p.value, float)
    want = _jac(p, x, idx, float(p.cfg["search"]["fd_step"]))
    calls = []
    spectra = p.spectra
    monkeypatch.setattr(p, "spectra", lambda X: calls.append(np.atleast_2d(X).shape[0]) or spectra(X))
    monkeypatch.setattr(p, "residual_scales", lambda x: pytest.fail("residual_scales called"))
    counter = {"n": 0}
    r, J = S._jacobian_batched(p, x, idx, np.asarray(p.step, float)[idx], float(p.cfg["search"]["fd_step"]), counter)
    assert calls == [2 * idx.size + 1] and counter["n"] == 2 * idx.size + 1
    np.testing.assert_allclose(J, want, rtol=1e-9, atol=1e-9 * np.max(np.abs(want)))
