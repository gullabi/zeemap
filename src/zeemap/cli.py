"""Command line: inject | eval | fit (SPEC.md CLI-2 to CLI-4).

    zeemap inject <config> -o <dir> [--seed N] [--noise none|white|full] [--sigma-i S] [--sigma-vi S]
    zeemap eval <config> -o <dir>
    zeemap fit <config> -o <dir> [--seed N] [--workers W] [--starts N] [--mcmc | --no-mcmc] [--mcmc-modes N]
               [--mcmc-evals N] [--systematics | --no-systematics] [--sys-modes N] [--resume | --from DIR]

Heavy modules are imported after the arguments are parsed: with `fit --workers W > 1` the thread variables of numba,
BLAS and OpenMP are set to 1 first (unless already set), so that W single-threaded processes share the cores (ENG-2).
Errors in the configuration, the library or the data stop with a one-line message and exit status 2.

`fit` runs, in order: the search for modes (search.py) and the families and table (table.py); the error bars by MCMC
inside the best modes (DESIGN.md section 4, SPEC.md UNC-1 to UNC-3, UNC-6) only when asked (`--mcmc`,
`--mcmc-modes` or `--mcmc-evals`; `--no-mcmc` wins; decision #42); the profiles from each family's best node
(profiles.py); the systematic swaps and the leave-one-phase-out jackknife (SPEC.md SYS-1 to SYS-5, UNC-7; on by
default). Seeds are derived from `--seed` per mode and per swap, so the result does not depend on `--workers`.
Hand-backs of the MCMC can change -ln L; the table is then re-sorted and, when a mode moved, the families recomputed.

Checkpoints: every stage writes its result to `<out>/checkpoint/` (pickles, written atomically) as soon as it ends.
`--resume` reads them from `-o`, `--from DIR` from another run's directory (and copies those it uses into `-o`); a
stage is taken from its checkpoint when its inputs are unchanged, else it runs again. A resume refuses (exit status 2,
one line naming what differs) when the sections library, data, model, parameters, objective or search, the bytes of
the library and data files, the zeemap version and commit, the seed or the number of starts differ from the
checkpointed search's. Checkpoints are read with a restricted unpickler (`_CheckpointUnpickler`) that refuses anything
but zeemap's classes and numpy's arrays; still, resume only from run directories you created or trust.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

from . import __version__

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeemap", description="magnetic white dwarf field-geometry fits")
    parser.add_argument("--version", action="version", version=f"zeemap {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")
    p = sub.add_parser("inject", help="simulate an observation from the configuration's parameter values")
    p.add_argument("config")
    p.add_argument("-o", "--out", required=True, help="output directory (the data set)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--noise", choices=("none", "white", "full"), default="none")
    p.add_argument("--sigma-i", type=float, default=0.01, help="sigma of I relative to the median model I")
    p.add_argument("--sigma-vi", type=float, default=0.003, help="sigma of V/I (absolute)")
    p = sub.add_parser("eval", help="evaluate the configuration's parameter values")
    p.add_argument("config")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p = sub.add_parser("fit", help="search for modes, error bars (MCMC) and systematic swaps")
    p.add_argument("config")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--starts", type=int, default=None, help="number of starts (default: search.n_starts)")
    p.add_argument("--mcmc", action=argparse.BooleanOptionalAction, default=None,
                   help="error bars by MCMC inside the modes (default: off, on with --mcmc-modes or --mcmc-evals)")
    p.add_argument("--mcmc-modes", type=int, default=None,
                   help="MCMC for the best N modes only (default: uncertainty.modes_max; 0 = all)")
    p.add_argument("--mcmc-evals", type=int, default=None,
                   help="cost evaluations per MCMC run of a mode (default: uncertainty.max_evals)")
    p.add_argument("--systematics", action=argparse.BooleanOptionalAction, default=True,
                   help="refit under each [[systematics.swap]] and each left-out phase (default: on)")
    p.add_argument("--sys-modes", type=int, default=None,
                   help="systematics for the best N modes only (default: all; 0 = all)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true",
                   help="continue from the checkpoints in the output directory; only resume from run directories you "
                        "created or trust")
    g.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                   help="continue from the checkpoints of the run in DIR, writing to -o; only resume from run "
                        "directories you created or trust")
    return parser


def _mcmc_on(args) -> bool:
    """MCMC only when asked (decision #42): --mcmc, or an MCMC budget (--mcmc-modes, --mcmc-evals); --no-mcmc wins."""
    if args.mcmc is not None:
        return bool(args.mcmc)
    return args.mcmc_modes is not None or args.mcmc_evals is not None


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


GUARD = ("library", "data", "model", "parameters", "objective", "search")  # a resume refuses changes to these


def _key(*parts) -> str:
    """Checkpoint key: a hash of JSON-able parts."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _ck_put(args, name: str, obj) -> None:
    """Write checkpoint `name` into <out>/checkpoint/ (atomic: a crash keeps the previous one)."""
    from .atomic import replacing

    d = Path(args.out) / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    with replacing(d / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# The globals a checkpoint may name, exactly: the zeemap classes that checkpoints hold, and numpy's arrays, dtypes and
# scalars (the names numpy's own pickles use). Anything else (os.system, or a dotted name such as
# zeemap.report / subprocess.Popen that would walk through a module's imports) is refused before it is called, so a
# crafted checkpoint cannot run code. A new class stored in a checkpoint must be added here.
CHECKPOINT_GLOBALS = {("zeemap.convergence", "V14Report"), ("zeemap.search", "PolishResult"),
                      ("zeemap.search", "Fingerprint"), ("zeemap.search", "EndPoint"), ("zeemap.search", "Mode"),
                      ("zeemap.systematics", "SwapResult"), ("zeemap.uncertainty", "McmcResult"),
                      ("zeemap.uncertainty", "UncertaintyResult"),
                      ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy._core.multiarray", "_reconstruct"),
                      ("numpy._core.multiarray", "scalar"), ("numpy._core.numeric", "_frombuffer"),
                      ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
                      ("numpy.core.numeric", "_frombuffer")}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in CHECKPOINT_GLOBALS:
            return super().find_class(module, name)
        from .config import ConfigError

        raise ConfigError(f"checkpoint refused: it names {module}.{name}, which zeemap checkpoints do not hold")


def _ck_get(args, name: str):
    """Checkpoint `name` of the run resumed from (--resume: -o; --from: that directory), else None; one read from
    another directory is copied into -o, so that -o can be resumed in turn."""
    if args.ck_src is None:
        return None
    path = Path(args.ck_src) / "checkpoint" / f"{name}.pkl"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        obj = _CheckpointUnpickler(f).load()
    args.ck_used.append(name)
    if Path(args.ck_src).resolve() != Path(args.out).resolve():
        _ck_put(args, name, obj)
    return obj


def _mcmc_log(k: int, msg: str) -> None:
    """Log of the MCMC of mode k (1-based): "mcmc: mode k: ..." (also from worker processes)."""
    msg = msg.removeprefix("mcmc: ")
    if msg.startswith("mode 0: "):  # uncertainty_for_modes numbers the single mode it was given 0
        msg = msg[len("mode 0: "):]
    _say(f"mcmc: mode {k}: {msg}")


def _mcmc_task(task, pool=None):
    """One mode's error bars (uncertainty_for_modes on a one-mode table); run in-process (its cost calls spread over
    `pool` when given) or in a pool worker."""
    from functools import partial

    from .uncertainty import uncertainty_for_modes

    problem, mode, cfg, seed, value_tol, k = task
    return uncertainty_for_modes(problem, [mode], cfg, seed, value_tol=value_tol, log=partial(_mcmc_log, k),
                                 pool=pool)


def _per_mode(n: int, w: int, c: int) -> bool:
    """One process per mode when the n modes keep at least as many of the w workers busy as one mode's cost calls
    spread over the pool would (c = `uncertainty.pool_width`); else the modes in turn, each over the pool."""
    return n >= min(w, c)


def _uncertainty(problem, cfg: dict, res, args, key: str) -> tuple:
    """Error bars for the best modes of res.modes (see the module docstring). Replaces res.modes by the final table and
    returns (problem used for the table, intervals.json dict, {mode id: McmcResult}, wall seconds). `key`: the table
    stage's checkpoint key."""
    import copy
    import math
    from concurrent.futures import as_completed

    import numpy as np

    from .problem import Problem
    from .search import fingerprint_thresholds, make_pool, mirror_statistics, same_mode, settle_mirror, single_threaded
    from .uncertainty import PER_MODE_LABEL, UncertaintyResult, pool_width

    t0 = time.perf_counter()
    uc = cfg["uncertainty"]
    surf = cfg["model"]["surface"]
    patches = int(uc.get("patches", 0) or 0)
    uproblem = problem
    modes = [copy.copy(m) for m in res.modes]  # all_modes in diagnostics.json keep the search's values
    if getattr(res, "table_problem", None) is not None:  # the table stage re-polished and re-ranked them already
        uproblem = res.table_problem
    elif surf.get("scheme") == "fibonacci" and patches > 0 and patches != int(surf["n"]):
        cfg_u = copy.deepcopy(cfg)
        cfg_u["model"]["surface"] = {"scheme": "fibonacci", "n": patches}
        uproblem = Problem(cfg_u)
        # each mode's point and its mirror image on the new patches (search.settle_mirror): the better one becomes
        # the mode's point, and the canonical point written to modes.json gets its own cost
        mirror = settle_mirror(uproblem, modes, reevaluate=True)
        res.stats["mirror_uncertainty_patches"] = {"patches": patches, **mirror_statistics(modes, modes),
                                                   "n_mirror_better": mirror["n_mirror_better"]}
        order = sorted(range(len(modes)), key=lambda i: (modes[i].f, i))
        modes = [modes[i] for i in order]
        moved = "" if order == list(range(len(order))) else (
            f"; re-ranked, new order of the search's mode ids: {[i + 1 for i in order]}")
        _say(f"mcmc: {len(modes)} mode(s) re-evaluated with {patches} patches (uncertainty.patches; the search used "
             f"{surf['n']}){moved}")
    nmax = int(args.mcmc_modes if args.mcmc_modes is not None else uc.get("modes_max", 0) or 0)
    n = len(modes) if nmax <= 0 else min(nmax, len(modes))
    value_tol = float(res.stats.get("value_tolerance", 2.0))
    tasks = [(uproblem, modes[k], cfg if uproblem is problem else uproblem.cfg,
              int(np.random.SeedSequence([int(args.seed), k]).generate_state(1)[0]), value_tol, k + 1)
             for k in range(n)]
    w = max(1, int(args.workers))
    c = pool_width(uproblem, tasks[0][2]) if tasks else 1
    ucfg = {a: b for a, b in uc.items() if a != "modes_max" and not a.startswith("profile")}
    keys = [_key("mcmc", key, ucfg, k) for k in range(n)]  # checkpoint keys
    urs = [_ck_get(args, f"mcmc-{h}") for h in keys]
    todo = [k for k in range(n) if urs[k] is None]
    per_mode = _per_mode(len(todo), w, c)
    _say(f"mcmc: {n} of {len(modes)} mode(s) ({n - len(todo)} from checkpoints), {w} process(es) "
         f"({'one per mode' if per_mode else f'the modes in turn, cost calls over up to {min(w, c)}'}), max_evals "
         f"{uc.get('max_evals')} (0 = the default 600000), {uproblem.surface.n} patches")

    def finished(k, ur):
        urs[k] = ur
        _ck_put(args, f"mcmc-{keys[k]}", ur)

    uproblem.objective.bind_forward(uproblem.forward)  # once, before the workers fork (they inherit it)
    if w > 1 and per_mode:
        with make_pool(w) as pool:  # the search's pool: single-threaded workers (ENG-2)
            futs = {pool.submit(_mcmc_task, tasks[k]): k for k in todo}
            for f in as_completed(futs):
                finished(futs[f], f.result())
    elif w > 1:  # chunks of COST_CHUNK rows in either case, so the chains do not depend on this choice (V6)
        with make_pool(w) as pool, single_threaded():
            for k in todo:
                finished(k, _mcmc_task(tasks[k], pool))
    else:
        with single_threaded():  # the same thread counts as a pool worker, so results do not depend on --workers
            for k in todo:
                finished(k, _mcmc_task(tasks[k]))

    entries = []  # (mode, McmcResult | None, hand-backs)
    for k, ur in enumerate(urs):
        r = ur.results[0]
        rep = r.report
        verdict = "converged" if r.converged else f"not converged ({', '.join(rep.failed) or 'budget'})"
        if not r.converged:  # UNC-2 per quantity (decision #17)
            n_ok = sum(1 for q in r.intervals.values() if q.get("median") is not None)
            verdict += f"; {n_ok} of {len(r.intervals)} quantities pass on their own"
        tau = f"{rep.tau:.1f}" if math.isfinite(rep.tau) else "n/a"
        _say(f"mcmc: mode {k + 1}: {r.samples.shape[0]} ensembles x {r.nwalkers} walkers, {r.nevals} evaluations, "
             f"tau_max {tau}, {verdict}; {len(ur.handbacks[0])} hand-back(s)")
        entries.append((ur.modes[0], r, ur.handbacks[0]))
    entries += [(m, None, []) for m in modes[n:]]
    moved = [e[0] for e in entries if e[2]]  # hand-backs moved these modes: settle their mirror images again
    if moved:
        settle_mirror(uproblem, moved)
    order = sorted(range(len(entries)), key=lambda i: (entries[i][0].f, i))
    if order != list(range(len(entries))):
        _say(f"mcmc: mode table re-sorted after hand-backs (UNC-3) and re-numbered; new order of the old ids: "
             f"{[i + 1 for i in order]}")
    entries = [entries[i] for i in order]
    thr = fingerprint_thresholds(cfg)
    notes = []
    for a, (ma, _, hba) in enumerate(entries):
        for b, (mb, _, hbb) in enumerate(entries[:a]):
            if (hba or hbb) and same_mode(ma.fingerprint, mb.fingerprint, thr):
                notes.append(f"after hand-backs modes {b + 1} and {a + 1} have the same fingerprint")
                _say("mcmc: WARNING: " + notes[-1])
    res.modes = [e[0] for e in entries]
    done = [(k + 1, e) for k, e in enumerate(entries) if e[1] is not None]
    ur = UncertaintyResult([e[0] for _, e in done], [e[1] for _, e in done], [e[2] for _, e in done],
                           sum(u.nevals for u in urs), urs[0].calibration if urs else "uncalibrated")
    out = ur.to_dict()
    for (kid, _), d in zip(done, out["modes"]):
        d["mode"] = kid
    wall = time.perf_counter() - t0
    out.update({"label": PER_MODE_LABEL,
                "patches": int(uproblem.surface.n), "modes_without_mcmc": [k + 1 for k, e in enumerate(entries)
                                                                          if e[1] is None],
                "processes": w, "notes": notes, "_meta": {"wall_time_s": wall}})  # wall time: FORMATS.md section 6
    return uproblem, out, {kid: e[1] for kid, e in done}, wall


def _systematics(table_problem, cfg: dict, res, unc: dict | None, args) -> dict:
    """Systematic swaps and the jackknife for the best --sys-modes modes (see the module docstring); returns the
    systematics.json dict. The jackknife sigma is written into unc["intervals"] (only its jackknife_sigma fields)."""
    from .systematics import attach_jackknife, run_systematics

    n_all = len(res.modes)
    nmax = int(args.sys_modes or 0)
    n = n_all if nmax <= 0 else min(nmax, n_all)
    scfg = cfg if table_problem.cfg is cfg else table_problem.cfg
    from .report import sha256

    alts = [[str(f), sha256(f)] for a in cfg.get("systematics", {}).get("library_alternatives") or []
            for f in sorted(Path(a).glob("*")) if f.is_file()]  # the library_<k> swaps read these bytes
    ck = (lambda inp: _ck_get(args, f"sys-{_key(inp, alts)}"),
          lambda inp, r: _ck_put(args, f"sys-{_key(inp, alts)}", r))
    sr = run_systematics(table_problem, scfg, res.modes[:n], unc, args.seed, args.workers, log=_say, checkpoint=ck)
    d = sr.to_dict()
    d["modes_without_systematics"] = list(range(n + 1, n_all + 1))
    if unc is not None:
        d["jackknife_in_intervals"] = attach_jackknife(unc["intervals"], d)
    if sr.swaps:
        _say(f"systematics: {len(sr.swaps)} swap(s) x {n} mode(s); {sr.n_mode_changes} refit(s) changed mode; "
             f"{sr.nevals:.3g} evaluations, {sr.wall_time_s:.1f} s")
    else:
        _say("systematics: " + "; ".join(sr.notes))
    return d


def _fit(args, argv) -> int:
    from .config import ConfigError, load_config
    from .problem import Problem
    from .report import input_files, now, versions, write_fit
    from .search import run_search

    cfg = load_config(args.config)
    if args.mcmc_evals is not None:
        cfg["uncertainty"]["max_evals"] = int(args.mcmc_evals)
    if args.mcmc_modes is not None:
        cfg["uncertainty"]["modes_max"] = int(args.mcmc_modes)
    args.ck_src = args.out if args.resume else args.from_dir
    args.ck_used = []
    t0 = time.perf_counter()
    t_start = now()
    problem = Problem(cfg)
    _say(f"problem: {problem.dim} parameters ({int(problem.free.sum())} free: "
         f"{', '.join(n for n, f in zip(problem.names, problem.free) if f)}), {problem.surface.n} patches, "
         f"{len(problem.dataset.phases)} phase(s), objective {problem.objective.kind}")
    inputs = input_files(cfg, problem)  # also run.json's input_sha256
    lib = os.path.join(str(cfg["library"]["path"]), "")
    shas = {"library": {f: h for f, h in inputs.items() if f.startswith(lib)},  # the configuration file is not
            "data": {f: h for f, h in inputs.items()
                     if not f.startswith(lib) and f != cfg.get("_meta", {}).get("path")}}
    guard = {sec: _key(cfg.get(sec), shas.get(sec)) for sec in GUARD}  # sections and the bytes of their files
    v = versions()
    guard.update(code=_key(v["zeemap"], v["zeemap_commit"]), seed=int(args.seed),
                 starts=int(cfg["search"]["n_starts"] if args.starts is None else args.starts))
    state = _ck_get(args, "search")
    if state is not None and state["guard"] != guard:
        diff = [k for k in guard if state["guard"].get(k) != guard[k]]
        raise ConfigError(f"cannot resume from {args.ck_src}: {', '.join(diff)} differ(s) from the checkpointed run")
    if state is None and args.from_dir is not None:
        raise ConfigError(f"cannot resume: no checkpoint in {args.from_dir}")
    if state is None and args.resume:
        _say(f"resume: no checkpoint in {args.out}; starting from the beginning")
    res = run_search(problem, cfg, seed=args.seed, n_starts=args.starts, workers=args.workers, log=_say,
                     make_problem=Problem, save=lambda st: _ck_put(args, "search", {"guard": guard, **st}),
                     state=state)
    problem, cfg = res.problem, res.cfg  # with the frozen kernel when step 2b ran (LIKE-2)
    wall = time.perf_counter() - t0
    st = res.stats
    per_eval = "" if state is not None else (  # a resumed search's wall time covers only its last part
        f" ({1e3 * wall * args.workers / max(1, res.nevals):.2f} ms per evaluation per worker)")
    _say(f"search: {st['n_starts']} starts + {st['rounds']} restart round(s), {res.nevals:.3g} evaluations, "
         f"{wall:.1f} s{per_eval}")
    once = f"{st['n_band_once']} reached by one start only"
    if st["warning"]:
        once += ": the search cannot bound the chance that a better mode was missed"
    _say(f"modes within {st['band']:g} nats of the best: {st['n_band_modes']} ({once})")
    n_invalid = problem.objective.n_invalid
    from .table import table_problem, table_stage  # SEARCH-8 re-polish, the value tolerance at the error-bar count

    ktable = _key("table", guard, cfg["uncertainty"].get("patches"))
    ts = _ck_get(args, f"table-{ktable}")
    if ts is None:
        ts = table_stage(problem, cfg, res, workers=args.workers, make_problem=Problem, say=_say)
        _ck_put(args, f"table-{ktable}", {k: v for k, v in ts.items() if k != "problem"})
    else:
        ts["problem"] = table_problem(problem, cfg, Problem)[1]
        _say(f"table: {len(ts['modes'])} mode(s) from the checkpoint")
    res.stats["value_tolerance_search"] = res.stats.get("value_tolerance")
    res.stats.update(ts["stats"])
    res.stats["value_tolerance"] = ts["stats"]["value_tolerance_table"]
    res.modes, res.table_problem, res.families = ts["modes"], ts["problem"], ts["families"]
    unc = None
    table_problem = ts["problem"]
    if _mcmc_on(args):
        table_problem, intervals, results, mwall = _uncertainty(problem, cfg, res, args, ktable)
        n_conv = sum(1 for r in results.values() if r.converged)
        _say(f"mcmc: {len(results)} mode(s) x {cfg['uncertainty']['ensembles']} ensembles, {intervals['nevals']:.3g} "
             f"evaluations, {mwall:.1f} s; {n_conv} of {len(results)} converged")
        unc = {"intervals": intervals, "results": results, "wall": mwall}
        if res.families is not None:  # hand-backs that moved a mode make the families stale
            from .table import refresh_families

            got = refresh_families(table_problem, cfg, res.modes, float(res.stats["value_tolerance"]),
                                   float(cfg["search"]["band"]), _say, args.workers)
            if got is not None:
                res.families = got[0]
                res.stats.update(got[1])
    res.profiles = None
    pq = cfg["uncertainty"].get("profile_quantities", "default")
    if res.families and res.families["families"] and pq != []:  # UNC-5, UNC-8 (decision #29)
        from .profiles import profile_entries, profile_walks

        t_p = time.perf_counter()
        try:  # a failure here must not discard the finished search and MCMC: noted, not raised
            prof = {}
            ivm = {e["mode"]: e for e in (unc["intervals"]["modes"] if unc else [])}
            nfam = int(cfg["uncertainty"].get("profile_families", 0) or 0)
            fams = res.families["families"][: nfam if nfam > 0 else None]
            mids = [next(n for n in fam["nodes"] if n["node"] == fam["best"])["mode"] for fam in fams]
            best = [ivm.get(mid) for mid in mids]
            # the walks read the table's problem, the families' nodes (hand-backs can move them), h and the profile
            # settings, not the MCMC: their checkpoint serves a later --mcmc
            nodes = [[[n.get("node"), n.get("mode"), [float(a) for a in n["x"]], float(n["neg_log_l"])]
                      for n in fam["nodes"]] for fam in fams]
            kprof = _key("profiles", ktable, nodes, float(res.families["h"]),
                         {a: b for a, b in cfg["uncertainty"].items() if a.startswith("profile")})
            plans = _ck_get(args, f"profiles-{kprof}")
            if plans is None:
                plans = profile_walks(table_problem, table_problem.cfg, fams, res.families["h"],
                                      workers=args.workers)  # every walk direction a task of the search's pool
                _ck_put(args, f"profiles-{kprof}", plans)
            qs = profile_entries(plans, len(fams), res.families["level"], res.families["h"],
                                 mode_intervals=[None if e is None else e["quantities"] for e in best])
            for fam, mid, e, q in zip(fams, mids, best, qs):
                prof[fam["id"]] = {"best_mode": mid, "converged": None if e is None else bool(e.get("converged")),
                                   "quantities": q}
            res.profiles = prof
            nq = sum(len(p["quantities"]) for p in prof.values())
            ne = sum(q["nevals"] for p in prof.values() for q in p["quantities"].values())
            _say(f"profiles: {len(prof)} family(ies) x {nq // max(1, len(prof))} quantities, {ne} evaluations, "
                 f"{time.perf_counter() - t_p:.1f} s")
        except Exception as exc:  # noqa: BLE001 (diagnostics.json and report.txt carry the reason)
            res.profiles = None
            res.stats["profiles_error"] = f"{type(exc).__name__}: {exc}"
            _say(f"profiles: FAILED ({res.stats['profiles_error']}); the other outputs are written")
    if getattr(problem.objective, "kernel_state", None) is not None:  # the table's problem carries the same kernel
        table_problem.objective.kernel_state = problem.objective.kernel_state
    if table_problem.objective.kind == "full" and res.modes:  # LIKE-10 at the best mode, on the table's patches
        from .noise import FAIL_FLAG, whiteness

        wh = whiteness(table_problem, res.modes[0].x)
        res.stats["whiteness"] = wh
        res.stats["whiteness_pass"] = all(v["pass"] for v in wh.values())
        bad = [k for k, v in wh.items() if not v["pass"]]
        _say("noise model: whitened residuals white in "
             f"{len(wh) - len(bad)} of {len(wh)} spectra" + (f"; {FAIL_FLAG}: {', '.join(bad)}" if bad else ""))
        # a failure puts the flag and the failing spectra into every output file and table (report.write_fit)
    from .noise import capped_kernels

    capped = capped_kernels((getattr(table_problem.objective, "kernel_state", None) or {}).get("frozen") or [])
    if capped:  # named in report.txt; every kernel's at_bound is in run.json
        res.stats["kernel_at_bound"] = capped
    if res.modes:  # DIAG-1 (floor chi2) and DIAG-3 (per-window residuals across library versions), decision #23
        from .diagnostics import family_best_points, floor_chi2, window_residuals
        from .systematics import apply_swap

        bx = res.modes[0].x
        try:
            res.stats["floor_chi2"] = floor_chi2(table_problem, bx)
        except Exception as exc:  # noqa: BLE001 (a diagnostic must not stop the run; the reason is written)
            res.stats["floor_chi2_error"] = f"{type(exc).__name__}: {exc}"
        probs = {"run": table_problem}
        for k, path in enumerate(cfg.get("systematics", {}).get("library_alternatives") or [], start=1):
            try:
                probs[f"library_{k}"] = Problem(apply_swap(table_problem.cfg, {"name": f"library_{k}",
                                                                               "set": {"library": {"path": path}}}))
            except Exception as exc:  # noqa: BLE001
                res.stats.setdefault("window_residuals_notes", []).append(f"library_{k} ({path}): {exc}")
        res.stats["window_residuals"] = [{"family": fid, "mode": mid, **r}  # DIAG-3: per family's best mode
                                         for fid, mid, fx in family_best_points(res.families, res.modes)
                                         for r in window_residuals(probs, fx)]
        if getattr(table_problem.forward, "NMU", 1) > 1:  # LIB-8: area below the smallest library mu
            res.stats["mu_below_fraction"] = table_problem.forward.mu_below_fraction(table_problem.theta(bx)).tolist()
    sysd = None
    if args.systematics:
        sysd = _systematics(table_problem, cfg, res, unc, args)
    out = write_fit(args.out, table_problem, cfg, res, seed=args.seed, workers=args.workers, t_start=t_start,
                    t_end=now(), wall=time.perf_counter() - t0, n_invalid_main=n_invalid, argv=argv,
                    uncertainty=unc, search_wall=wall, systematics=sysd, inputs=inputs,
                    resumed=None if args.ck_src is None else {
                        "from": os.path.relpath(args.ck_src, args.out), "search": state is not None,
                        "checkpoints": args.ck_used})
    extra = (" intervals.json samples/" if unc else "") + (" systematics.json" if sysd is not None else "")
    _say(f"wrote {args.out}/run.json modes.json diagnostics.json report.txt{extra} spectra/ histograms/ "
         f"({len(out['modes'])} mode(s))")
    return 0


def _eval(args) -> int:
    from .config import load_config
    from .problem import Problem
    from .report import write_eval

    cfg = load_config(args.config)
    problem = Problem(cfg)
    ev = write_eval(args.out, problem, cfg)
    chi = ", ".join(f"{k} {v['chi2']:.1f}/{v['n']}" for k, v in ev["per_spectrum"].items())
    _say(f"eval: -lnL {ev['neg_log_l']:.3f} (+ penalty {ev['penalty']:.3g}), chi2 {chi}; wrote {args.out}/eval.json")
    if ev.get("scale_ratio"):
        from .likelihood import SCALE_RATIO_RANGE

        lo, hi = SCALE_RATIO_RANGE
        ratios = ", ".join(f"{k} {v:.3g}" for k, v in ev["scale_ratio"].items())
        odd = [k for k, v in ev["scale_ratio"].items() if not lo <= v <= hi]
        warn = (f"; WARNING: outside [{lo:g}, {hi:g}] for {', '.join(odd)}: library and data in different units? "
                "(-ln L is unaffected except a constant, but check the units)") if odd else ""
        _say(f"eval: scale median |model I| / median |data I|: {ratios}{warn}")
    return 0


def _inject(args) -> int:
    from .config import load_config
    from .inject import inject

    cfg = load_config(args.config)
    inject(cfg, args.out, seed=args.seed, noise=args.noise, sigma_i=args.sigma_i, sigma_vi=args.sigma_vi)
    _say(f"inject: wrote {args.out} (noise {args.noise}, seed {args.seed})")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "fit" and args.workers > 1:
        for v in THREAD_VARS:
            os.environ.setdefault(v, "1")
    from .config import ConfigError
    from .data import DataError
    from .library import LibraryError

    try:
        if args.command == "fit":
            return _fit(args, argv)
        if args.command == "eval":
            return _eval(args)
        return _inject(args)
    except (ConfigError, DataError, LibraryError) as exc:
        _say(f"zeemap {args.command}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
