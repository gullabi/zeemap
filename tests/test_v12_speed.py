"""V12 (VERIFICATION.md V12; SPEC.md FWD-12, ENG-6): speed of one evaluation and of a batch of 32, and the
peak memory of each worker in a 4-worker search. Tier 2 (`tools/check.sh 2 V12`); tier 1 deselects it.

Problem: the synthetic library (tests/synth.py, variant a) with Teff 20 000 and 30 000 K selected (Teff free, so each
evaluation interpolates between two slices), 5000 Fibonacci patches, 8 phases (0, 0.125, ... 0.875), 900 pixels per
spectrum (grid 4300-6098 A in 2 A steps, one window over all of it), I and V/I per phase; noise-free data from
`zeemap inject`. The truth: an offset-free dipole of 400 MG, 60 degrees from the rotation axis, i = 60 degrees,
Teff 25 000 K. An evaluation is `Problem.cost` (forward model, likelihood, prior-cut and gauge penalties), timed for
the kinds "full" and "chi2" as the best of several repeats (single row: 15 calls; batch of 32 rows: 5 calls).

Pass rules:
- FWD-12: below 10 ms per single evaluation, gated only on the reference machine (`REFERENCE_CPU`, the CPU model
  string of /proc/cpuinfo; "" = no reference machine named yet, the rule is then reported, not gated).
- On any machine: every time at most 2 x the stored baseline of this machine. The baselines are in
  tests/regression/speed_baselines.json, keyed by "<hostname> | <CPU model> | <cores>"; the first run on a new
  machine writes its entry and passes with a note.
- ENG-6: peak memory (VmHWM of each worker process, polled from the parent; cross-checked with getrusage of the
  reaped children) below 1 GB per worker.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from zeemap import cli, search
from zeemap.config import load_config
from zeemap.problem import Problem
from test_v7_cli import MD_TRUE, U_TRUE

pytestmark = pytest.mark.tier2

REFERENCE_CPU = ""  # CPU model string of the reference machine (DESIGN.md section 7); "" = not named yet
FWD12_MS = 10.0  # SPEC.md FWD-12, ms per evaluation on the reference machine
SLOWDOWN_MAX = 2.0  # VERIFICATION.md V12
MEMORY_MAX_MB = 1024.0  # SPEC.md ENG-6, per worker
BASELINES = Path(__file__).resolve().parent / "regression" / "speed_baselines.json"
N_PATCHES, N_PHASES, N_PIX, BATCH = 5000, 8, 900, 32
TIMED = ("full_single_ms", "full_batch_ms", "chi2_single_ms", "chi2_batch_ms")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine_key() -> str:
    return f"{socket.gethostname()} | {cpu_model()} | {os.cpu_count()}"


def _config(data: str, kind: str, search_block: str = "") -> str:
    phases = "\n".join(
        f'[[data.phase]]\nphase = {k / N_PHASES}\nfile = "ph{k:02d}.dat"\nierr = "ph{k:02d}_ierr.dat"\n'
        f'vierr = "ph{k:02d}_vierr.dat"\n' for k in range(N_PHASES))
    md = ", ".join(repr(float(v)) for v in MD_TRUE)
    return f"""
[library]
path = "synth"
teffs = [20000.0, 30000.0]

[data]
path = "{data}"
windows = [[4300.0, {4300.0 + 2.0 * (N_PIX - 1)}]]
grid = {{ start = 4300.0, step = 2.0, npix = {N_PIX} }}

{phases}
[model]
fwhm = 3.0
surface = {{ scheme = "fibonacci", n = {N_PATCHES} }}

[objective]
kind = "{kind}"

[parameters]
md = {{ value = [{md}], step = 10.0, lo = 150.0, hi = 600.0, free = true }}
bq = {{ value = 0.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }}
aq = {{ value = [0.0, 0.0, 1.0], step = 0.1, lo = 0.5, hi = 2.0, free = false }}
mo = {{ value = [0.0, 0.0, 0.0], step = 10.0, lo = 0.0, hi = 100.0, free = false }}
c = {{ value = [0.0, 0.0, 0.0], step = 0.02, lo = 0.0, hi = 0.3, free = false }}
u = {{ value = {U_TRUE!r}, step = 0.1, lo = -1.0, hi = 1.0, free = true }}
teff = {{ value = 25000.0, step = 1000.0, lo = 20000.0, hi = 30000.0, free = true }}

[search]
{search_block}

[search.starts]
md = [200.0, 600.0]
"""


@pytest.fixture(scope="module")
def setup(synth_lib, tmp_path_factory):
    root = tmp_path_factory.mktemp("zeemap_data_v12")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        truth = root / "truth.toml"
        truth.write_text(_config("inj", "chi2"))
        assert cli.main(["inject", str(truth), "-o", str(root / "inj"), "--noise", "none"]) == 0
        cfgs = {}
        for kind in ("full", "chi2"):
            p = root / f"{kind}.toml"
            p.write_text(_config("inj", kind, "n_starts = 4\nrestarts = 0\nrounds_max = 1\nlbfgs_maxiter = 3\n"
                                              "lm_maxiter = 2"))
            cfgs[kind] = load_config(p)
        yield root, cfgs


def _best_time(fn, repeats: int) -> float:
    fn()  # warm-up (numba dispatch, caches)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


@pytest.fixture(scope="module")
def timings(setup):
    _, cfgs = setup
    out, problems = {}, {}
    with search.single_threaded():  # one core, as in a worker (FWD-12, ENG-2)
        for kind, cfg in cfgs.items():
            t = time.perf_counter()
            pr = Problem(cfg)
            out[f"{kind}_construct_s"] = time.perf_counter() - t
            problems[kind] = pr
            x = pr.value.copy()
            rng = np.random.default_rng(12)
            X = np.repeat(x[None, :], BATCH, axis=0)
            X[:, pr.free] += rng.normal(0.0, 0.3, (BATCH, int(pr.free.sum()))) * pr.step[pr.free]
            X[:, pr.iteff] = rng.uniform(21000.0, 29000.0, BATCH)
            f1 = pr.cost(x[None, :])
            fb = pr.cost(X)
            assert np.all(f1 < search.WORST) and np.all(fb < search.WORST), (kind, f1, fb)
            out[f"{kind}_single_ms"] = 1e3 * _best_time(lambda: pr.cost(x[None, :]), 15)
            out[f"{kind}_batch_ms"] = 1e3 * _best_time(lambda: pr.cost(X), 5) / BATCH
    return out, problems


def _children(pid: int) -> list[int]:
    kids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
                if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
                    kids.append(int(d.name))
            except (OSError, ValueError, IndexError):
                pass
    return kids


def _vmhwm_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


@pytest.fixture(scope="module")
def memory(setup, timings):
    _, cfgs = setup
    pr = timings[1]["full"]
    peaks: dict[int, float] = {}
    stop = threading.Event()

    def poll():
        me = os.getpid()
        while not stop.is_set():
            for k in _children(me):
                m = _vmhwm_mb(k)
                if m is not None:
                    peaks[k] = max(peaks.get(k, 0.0), m)
            stop.wait(0.05)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t = time.perf_counter()
    res = search.run_search(pr, cfgs["full"], seed=3, workers=4)
    wall = time.perf_counter() - t
    stop.set()
    th.join()
    ru_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0  # kB on Linux; largest reaped child
    parent_mb = _vmhwm_mb(os.getpid())
    return {"workers": peaks, "children_maxrss_mb": ru_mb, "parent_vmhwm_mb": parent_mb, "search_s": wall,
            "nevals": res.nevals}


def _load_baselines() -> dict:
    try:
        return json.loads(BASELINES.read_text())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------------------- V12


def test_v12_speed_against_baseline_and_fwd12(timings, record_property):
    t, _ = timings
    key = machine_key()
    line = (f"V12 [{key}]: ms per evaluation: full {t['full_single_ms']:.2f} single, {t['full_batch_ms']:.2f} in a "
            f"batch of {BATCH}; chi2 {t['chi2_single_ms']:.2f} single, {t['chi2_batch_ms']:.2f} in a batch; "
            f"Problem construction {t['full_construct_s']:.2f} s (full), {t['chi2_construct_s']:.2f} s (chi2)")
    print(line)
    for k, v in t.items():
        record_property(f"v12_{k}", round(v, 4))
    on_ref = bool(REFERENCE_CPU) and cpu_model() == REFERENCE_CPU
    fwd12 = (f"FWD-12 (< {FWD12_MS:g} ms single evaluation): {t['full_single_ms']:.2f} ms, "
             + ("gated (reference machine)" if on_ref else "reported only (not the reference machine)"))
    print("V12: " + fwd12)
    record_property("v12_fwd12", fwd12)
    if on_ref:
        assert t["full_single_ms"] < FWD12_MS, line

    base = _load_baselines()
    entry = base.get(key)
    if entry is None:
        base[key] = {"recorded": time.strftime("%Y-%m-%d"), "python": platform.python_version(),
                     **{k: round(t[k], 4) for k in TIMED}}
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(dict(sorted(base.items())), indent=2) + "\n")
        note = f"V12: NOTE: no baseline for this machine; wrote it to {BASELINES.name} and pass ({line})"
        print(note)
        record_property("v12_baseline", "written")
        return
    ratios = {k: t[k] / entry[k] for k in TIMED if entry.get(k)}
    print("V12: ratio to the baseline of " + entry.get("recorded", "?") + ": "
          + ", ".join(f"{k} {r:.2f}" for k, r in ratios.items()))
    record_property("v12_ratios", {k: round(r, 3) for k, r in ratios.items()})
    slow = {k: r for k, r in ratios.items() if r > SLOWDOWN_MAX}
    assert not slow, f"more than {SLOWDOWN_MAX:g} x slower than the stored baseline: {slow}; {line}"


def test_v12_peak_memory_per_worker(memory, record_property):
    m = memory
    w = m["workers"]
    print(f"V12: 4-worker search ({m['nevals']} evaluations, {m['search_s']:.1f} s): peak RSS per worker (VmHWM) "
          + ", ".join(f"{v:.0f} MB" for v in sorted(w.values()))
          + f"; largest reaped child (getrusage) {m['children_maxrss_mb']:.0f} MB; parent {m['parent_vmhwm_mb']:.0f} MB")
    record_property("v12_worker_peak_mb", [round(v, 1) for v in sorted(w.values())])
    record_property("v12_children_maxrss_mb", round(m["children_maxrss_mb"], 1))
    assert len(w) >= 4, f"saw {len(w)} worker processes, expected 4: {w}"
    worst = max(max(w.values()), m["children_maxrss_mb"])
    assert worst < MEMORY_MAX_MB, f"peak memory {worst:.0f} MB per worker, limit {MEMORY_MAX_MB:g} MB (ENG-6): {m}"
