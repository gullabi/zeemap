"""Measure the Levenberg-Marquardt shortfall that sets `search.lm_maxiter` (DESIGN.md section 3, "Polish").

shortfall(cap) = f(the LM end point at `cap` iterations) - f(the same LM continued until a step gains less than 1e-6
nat or the Gauss-Newton model predicts less, `search.lm_stop_nat` = 1e-6, with `search.lm_max_rejections` off, or
1000 iterations), for polishes from starts perturbed around a mode (the restart perturbation of step 4), on the
error-bar patch count (`uncertainty.patches`). Prints the median and the 90th percentile per cap and the smallest cap
of 50, 100, 200, 400 whose median is below 1 nat. The LM stopping earlier than a cap is read as no shortfall at that
cap (every iteration has a trace entry, `search.polish`).

With --variant KEY=VALUE[,KEY=VALUE] (`search` keys, e.g. lm_stop_nat=1e-3,lm_max_rejections=5; repeat the option
for several variants) each start is also polished with the configuration as it is and with each variant: printed are
the LM evaluations and each variant's -ln L above the configuration's end point and above the long run.

  python tools/measure_lm_shortfall.py <config.toml> [--modes <modes.json>] [--starts 20] [--seed 1]
                                       [--variant KEY=VALUE[,KEY=VALUE] ...]

Without --modes the configuration's parameter values are the mode (synthetic tests: the truth).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np

from zeemap import search as S
from zeemap.config import load_config
from zeemap.noise import error_bar_cfg
from zeemap.problem import Problem

CAPS = (20, 50, 100, 200, 400)
LONG = 1000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--modes", default=None, help="modes.json of a fit; its first mode is used")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", action="append", default=[], metavar="KEY=VALUE[,KEY=VALUE]",
                    help="search keys of a variant to compare (repeatable: one variant each)")
    a = ap.parse_args(argv)
    cfg = error_bar_cfg(load_config(a.config))
    p = Problem(cfg)
    if a.modes:
        doc = json.loads(open(a.modes).read())
        x_mode = np.array([doc["modes"][0]["vector"][n] for n in doc["names"]], dtype=float)
    else:
        x_mode = np.array(p.value, dtype=float)
    c = copy.deepcopy(cfg)
    c["search"].update(lm_maxiter=LONG, lm_stop_nat=1e-6, lm_max_rejections=0)
    variants = []
    for spec in a.variant:
        var = copy.deepcopy(cfg)
        for kv in spec.split(","):
            k, v = kv.split("=", 1)
            var["search"][k] = type(cfg["search"][k])(float(v))
        variants.append(var)
    rows = {spec: [] for spec in a.variant}
    rngs = S.restart_streams(a.seed, 1, a.starts)
    short = {cap: [] for cap in CAPS}
    hit = {cap: 0 for cap in CAPS}
    for k in range(a.starts):
        x0 = S.perturb(p, x_mode, cfg, rngs[k])
        trace: list = []
        r = S.polish(p, x0, c, trace=trace)
        f_end = r.f
        fs = dict(trace)
        last = max(fs) if fs else 0
        for cap in CAPS:
            f_cap = fs.get(min(cap, last), f_end) if fs else f_end
            short[cap].append(f_cap - f_end)
            hit[cap] += int(last > cap)
        print(f"start {k}: LM {last} iterations, f {f_end:.4f}, stop {r.lm_stop}", file=sys.stderr)
        if variants:
            ref = S.polish(p, x0, cfg)
            print(f"  configured: LM {ref.lm_iterations} it, {ref.lm_nevals} of {ref.nevals} evals, stop "
                  f"{ref.lm_stop}, f {ref.f:.4f}", file=sys.stderr)
            for spec, var in zip(a.variant, variants):
                alt = S.polish(p, x0, var)
                rows[spec].append((ref.lm_nevals, alt.lm_nevals, alt.f - ref.f, alt.f - f_end, ref.f - f_end,
                                   ref.nevals))
                print(f"  {spec}: LM {alt.lm_iterations} it, {alt.lm_nevals} of {alt.nevals} evals, stop "
                      f"{alt.lm_stop}, f {alt.f:.4f}", file=sys.stderr)
    best = None
    for cap in CAPS:
        v = np.array(short[cap])
        print(f"lm_maxiter {cap:4d}: median shortfall {np.median(v):.3f} nats, p90 {np.percentile(v, 90):.3f}, "
              f"{hit[cap]} of {a.starts} polishes needed more")
        if best is None and cap >= 50 and np.median(v) < 1.0:
            best = cap
    print(f"smallest cap of 50, 100, 200, 400 with a median shortfall below 1 nat: {best}")
    for spec, rs in rows.items():
        v = np.array(rs)
        print(f"variant {spec}: LM evaluations {int(v[:, 1].sum())} against {int(v[:, 0].sum())} "
              f"({100.0 * (1.0 - v[:, 1].sum() / max(1.0, v[:, 0].sum())):.0f} % fewer; the configured polishes made "
              f"{int(v[:, 5].sum())} evaluations in all); -ln L above the configured "
              f"end point: median {np.median(v[:, 2]):.2e}, max {v[:, 2].max():.2e} nats; above the long run: median "
              f"{np.median(v[:, 3]):.2e}, max {v[:, 3].max():.2e} (configured: median {np.median(v[:, 4]):.2e}, max "
              f"{v[:, 4].max():.2e}) nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
