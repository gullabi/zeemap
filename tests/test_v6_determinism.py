"""V6a, V6b (VERIFICATION.md V6; SPEC.md SEARCH-5, ENG-2, ENG-3): the search gives the same answer with 1,
2 and 4 worker processes, a different seed gives different starts, and every worker runs single-threaded.

Set-up: the small synthetic problem of tests/test_v7_cli.py (centred dipole of 400 MG at 60 degrees, i = 60 degrees,
Teff 30 000 K, 1000 Fibonacci patches, two phases, noise-free `inject`), fitted from md = 300 MG along z, u = 0.7 with
4 starts, 2 restarts and one restart round (8 end points). `run_search` is called directly; the Problem built here
sits in problem.py's registry, which the forked workers inherit.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from zeemap import cli, search
from zeemap.config import load_config
from zeemap.problem import Problem
from test_v7_cli import config_text

pytestmark = pytest.mark.tier1

TOL = 1e-12  # VERIFICATION.md V6a
SEED = 1
WORKERS = (1, 2, 4)


@pytest.fixture(scope="module")
def data_root(synth_lib, tmp_path_factory):
    """$ZEEMAP_DATA with the synthetic library and the noise-free injection `inj`."""
    root = tmp_path_factory.mktemp("zeemap_data_v6")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        truth = root / "truth.toml"
        truth.write_text(config_text("inj"))
        assert cli.main(["inject", str(truth), "-o", str(root / "inj"), "--noise", "none"]) == 0
    return root


@pytest.fixture(scope="module")
def problem(data_root):
    root = data_root
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        fit = root / "fit.toml"
        fit.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7))
        cfg = load_config(fit)
        pr = Problem(cfg)
        yield pr, cfg


@pytest.fixture(scope="module")
def results(problem):
    pr, cfg = problem
    return {w: search.run_search(pr, cfg, seed=SEED, workers=w) for w in WORKERS}


def _diff(a, b, path="") -> list[str]:
    """Differences between two JSON-like trees: numbers beyond TOL (relative, absolute below 1), anything else unequal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} vs {sorted(b)}"]
        return [d for k in a for d in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _diff(np.asarray(a).tolist(), np.asarray(b).tolist(), path)
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    if math.isnan(a) and math.isnan(b):
        return []
    return [] if abs(a - b) <= TOL * max(1.0, abs(a), abs(b)) else [f"{path}: {a!r} vs {b!r} (diff {a - b:.3g})"]


# ------------------------------------------------------------------------------------------------------------- V6a


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v6a_same_result_for_1_2_4_workers(results):
    ref = results[1]
    assert ref.stats["n_starts"] == 4 and len(ref.endpoints) >= 4
    print(f"V6a: {len(ref.endpoints)} end points, {len(ref.all_modes)} modes, best -lnL {ref.modes[0].f:.6g}")
    for w in WORKERS[1:]:
        r = results[w]
        d = _diff(ref.to_dict(), r.to_dict())
        assert not d, f"workers={w} differs from workers=1 in {len(d)} places, first: {d[:5]}"
        for m1, m2 in zip(ref.all_modes, r.all_modes):  # the canonical end points are not in to_dict
            assert np.allclose(m1.x_canonical, m2.x_canonical, rtol=TOL, atol=TOL)
        bitwise = all(np.array_equal(e1.x, e2.x) and e1.f == e2.f for e1, e2 in zip(ref.endpoints, r.endpoints))
        print(f"V6a: workers={w}: identical to workers=1 within {TOL:g} (bitwise: {bitwise})")


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v6a_starts_follow_the_seed(problem, results):
    pr, cfg = problem
    ref = results[1]
    x0 = np.array([e.x0 for e in ref.endpoints if e.round == 0])
    assert np.array_equal(search.draw_starts(pr, cfg, SEED, 4), x0)  # the search's starts are the seed's (ENG-3)
    other = search.draw_starts(pr, cfg, SEED + 1, 4)
    assert other.shape == x0.shape
    free = pr.free
    assert np.all(np.any(np.abs(other[:, free] - x0[:, free]) > 1e-6, axis=1)), "seed 2 repeats a start of seed 1"
    assert np.array_equal(search.draw_starts(pr, cfg, SEED + 1, 4), other)


# ------------------------------------------------------------------------------------------------------------- V6b


def test_v6b_workers_single_threaded():
    with search.make_pool(2) as pool:
        states = [f.result() for f in [pool.submit(search.thread_state) for _ in range(8)]]
    pids = {s["pid"] for s in states}
    assert os.getpid() not in pids
    for s in states:
        print(f"V6b: worker {s['pid']}: env {s['env']}, numba {s['numba']}, BLAS {s['blas']}")
        for v in search.THREAD_VARS:
            assert s["env"][v] == "1", (v, s)
        assert s["numba"] == 1, s
        if os.path.exists("/proc/self/maps"):
            assert s["blas"], f"no BLAS library found in the worker: {s}"
        assert all(n == 1 for n in s["blas"].values()), s


def test_v6b_serial_search_single_threaded_and_restored():
    before = search.thread_state()
    with search.single_threaded():
        inside = search.thread_state()
    after = search.thread_state()
    print(f"V6b: main process before {before['numba']}, {before['blas']}; inside {inside['numba']}, {inside['blas']}")
    assert inside["numba"] == 1 and all(n == 1 for n in inside["blas"].values())
    assert after["numba"] == before["numba"] and after["blas"] == before["blas"]


# ------------------------------------------------------------------------------------------------------------- V6c


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v6c_two_fits_same_outputs_apart_from_meta(data_root, tmp_path):
    """V6c (FORMATS.md section 6): `fit` twice with the same configuration, seed and 2 workers (search, a
    capped MCMC on the best mode, systematics on the best mode, profiles off): every output file exists in both, the
    JSON files are equal at 1e-12 outside the top-level `_meta` (run.json, with its start and end times, excepted),
    and report.txt, spectra/ and histograms/ are equal byte for byte."""
    import json

    outs = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(data_root))
        cfg_path = tmp_path / "fit.toml"
        cfg_path.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7,
                                        extra="[uncertainty]\npatches = 1000\nensembles = 2\nprofile_quantities = []\n"))
        for k in range(2):
            out = tmp_path / f"fit{k}"
            assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "4", "--workers", "2", "--starts", "3",
                             "--mcmc-modes", "1", "--mcmc-evals", "2000", "--sys-modes", "1"]) == 0
            outs.append(out)
    files = [sorted(str(p.relative_to(o)) for p in o.rglob("*") if p.is_file()) for o in outs]
    assert files[0] == files[1]
    assert "systematics.json" in files[0] and "intervals.json" in files[0]
    for rel in files[0]:
        a, b = outs[0] / rel, outs[1] / rel
        if rel == "run.json":
            continue
        if rel.endswith(".json"):
            ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
            ja.pop("_meta", None)
            jb.pop("_meta", None)
            assert not _diff(ja, jb, rel), _diff(ja, jb, rel)[:5]
        elif rel.endswith(".npz"):
            za, zb = np.load(a), np.load(b)
            assert sorted(za.files) == sorted(zb.files), rel
            for n in za.files:
                x, y = za[n], zb[n]
                assert np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"), f"{rel}:{n}"
        else:
            assert a.read_bytes() == b.read_bytes(), rel


# ------------------------------------------------------------------------------------------------------------- V6d


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v6d_fit_profiles_same_for_1_and_2_workers(data_root, tmp_path):
    """V6d (DESIGN.md section 4b; profiles.profiles_for_families): `fit --no-mcmc` with two profiled quantities (one
    fixed directly, one held by the penalty) and 1 and 2 workers: the walks run in the search's pool, and profiles.json
    is the same, outside the top-level `_meta`."""
    import json

    outs = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(data_root))
        cfg_path = tmp_path / "fit_prof.toml"
        extra = ('[uncertainty]\npatches = 1000\nprofile_quantities = ["incl_deg", "dipole_MG"]\n'
                 "profile_max_steps = 2\nprofile_families = 1\n")
        cfg_path.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7, extra=extra))
        for w in (1, 2):
            out = tmp_path / f"fit_w{w}"
            assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "1", "--workers", str(w), "--starts", "3",
                             "--no-mcmc", "--no-systematics"]) == 0
            outs.append(out)
    ja, jb = (json.loads((o / "profiles.json").read_text()) for o in outs)
    ja.pop("_meta", None)
    jb.pop("_meta", None)
    assert ja["families"] and all(len(e["walks"]) >= 2 for e in ja["families"]["1"]["quantities"].values())
    # the explicit list is profiled exactly (here, not in V7j, whose fit uses the default list)
    assert set(ja["families"]["1"]["quantities"]) == {"incl_deg", "dipole_MG"}
    assert ja == jb, _diff(ja, jb, "profiles.json")[:5]
    fa, fb = (json.loads((o / "families.json").read_text()) for o in outs)  # link tests in the pool
    for f in (fa, fb):
        f.pop("_meta", None)
        f.pop("nevals", None)
    assert fa == fb, _diff(fa, fb, "families.json")[:5]
