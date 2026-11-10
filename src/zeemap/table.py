"""The table stage between the search and the error bars (SPEC SEARCH-8, SEARCH-9; DESIGN.md section 3, steps 6 and 6b;
decisions #25, #29).

1. The problem on the error-bar patch count (`uncertainty.patches`, Fibonacci surfaces).
2. Per mode in the band, up to `search.tol_polishes` of its end points, best first, are re-polished on that problem
   (when the patch count differs from the search's; otherwise the end points are used as they are). The mode takes
   the best of them.
3. The optimiser scatter and the value tolerance are measured from these (modes with at least 3), at this patch count.
4. The equator mirror is settled again (search.settle_mirror), and the modes re-ranked.
5. The re-polished end points are the nodes of the families (families.py); each mode keeps its own in `mode.nodes`.
6. After the MCMC (cli.py), `refresh_families` recomputes the families when a hand-back has moved a mode by more than
   the value tolerance, with the new point as an extra node.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from . import search as S
from .families import Node, find_families
from .noise import error_bar_cfg


def table_problem(problem, cfg: dict, make_problem: Callable) -> tuple:
    """(re-polish?, the error-bar problem): a new problem on `uncertainty.patches` when the search's Fibonacci surface
    has another patch count, else the search's problem."""
    surf = cfg["model"]["surface"]
    patches = int((cfg.get("uncertainty") or {}).get("patches", 0) or 0)
    repolish = surf.get("scheme") == "fibonacci" and patches > 0 and patches != int(surf["n"])
    return repolish, (make_problem(error_bar_cfg(cfg)) if repolish else problem)


def table_stage(problem, cfg: dict, res, *, workers: int, make_problem: Callable, say: Callable[[str], None]) -> dict:
    """Returns {"problem": the error-bar problem, "modes": the re-ranked band modes (copies), "families": families.json
    content (or None), "stats": {...}} and leaves `res` unchanged except `res.stats` entries named below."""
    sc = S._search_cfg(cfg)
    repolish, uproblem = table_problem(problem, cfg, make_problem)
    modes = [copy.copy(m) for m in res.modes]
    k_pol = int(sc.get("tol_polishes", 5))
    outside_max = float(sc["starts"].get("outside_max", float("inf"))) if "starts" in sc else float("inf")
    groups = []  # per mode: list of (x, f)
    jobs, owner = [], []
    for mi, m in enumerate(modes):
        mem = [i for i in m.members if res.endpoints[i].valid]
        mem.sort(key=lambda i: res.endpoints[i].f)
        mem = mem[:k_pol] or [None]
        for i in mem:
            jobs.append(m.x if i is None else res.endpoints[i].x)
            owner.append(mi)
        groups.append([])
    nevals = 0
    if repolish:
        prs = S._polish_many(uproblem, jobs, cfg, workers)
        nevals = sum(pr.nevals for pr in prs)
        for mi, pr in zip(owner, prs):
            ok = pr.f < S.WORST and S._outside_ok(uproblem, pr.x, outside_max)
            if ok:
                groups[mi].append((np.asarray(pr.x, float), float(pr.f)))
    else:
        for mi, x in zip(owner, jobs):
            groups[mi].append((np.asarray(x, float), float(uproblem.cost(np.asarray(x, float)[None])[0])))
    keep = []
    for mi, m in enumerate(modes):
        g = groups[mi]
        if not g:
            continue
        x, f = min(g, key=lambda t: t[1])
        m.x, m.f = x, f
        m.x_canonical, m.f_canonical = None, None
        keep.append(mi)
    modes = [modes[mi] for mi in keep]
    groups = [groups[mi] for mi in keep]
    # value tolerance at this patch count (DESIGN.md section 3: pooled std over modes with >= 3 re-polished points)
    devs, dof = 0.0, 0
    for g in groups:
        if len(g) >= 3:
            f = np.array([t[1] for t in g])
            devs += float(np.sum((f - f.mean()) ** 2))
            dof += len(f) - 1
    scatter = float(np.sqrt(devs / dof)) if dof > 0 else 5.0
    tol = S.value_tolerance(scatter)
    S.settle_mirror(uproblem, modes)
    order = sorted(range(len(modes)), key=lambda i: (modes[i].f, i))
    modes = [modes[i] for i in order]
    groups = [groups[i] for i in order]
    best = modes[0].f if modes else 0.0
    band = float(sc["band"])
    in_band = [i for i, m in enumerate(modes) if m.f <= best + band]
    modes = [modes[i] for i in in_band]
    groups = [groups[i] for i in in_band]
    for m, g in zip(modes, groups):
        m.nodes = list(g)
    stats = {"table_patches": int(uproblem.surface.n), "table_repolished": bool(repolish),
             "optimiser_scatter_table": scatter, "value_tolerance_table": tol, "table_nevals": nevals}
    fams = None
    if modes:
        fams, fst = families_for(uproblem, cfg, modes, tol, band, say, workers=workers)
        stats.update(fst)
    say(f"table: {len(modes)} mode(s) on {uproblem.surface.n} patches"
        + (f", {len(jobs)} end points re-polished" if repolish else "")
        + f"; optimiser scatter {scatter:.2f} nats, value tolerance {tol:.2f} nats")
    return {"problem": uproblem, "modes": modes, "families": fams, "stats": stats}


def families_for(uproblem, cfg: dict, modes: list, tol: float, band: float, say: Callable[[str], None],
                 extra: dict | None = None, workers: int = 1) -> tuple[dict, dict]:
    """Families of `modes` (best first) from their nodes (`mode.nodes`, the re-polished end points) plus `extra`
    ({mode index: (x, f)}, the points of hand-backs); sets each mode's `family`, `families_all` and `band_component`.
    Returns (families.json content, statistics for diagnostics.json). `workers`: the link tests' pool."""
    extra = extra or {}
    nodes = [Node(mi, np.asarray(uproblem.canonical(x), float), f) for mi, m in enumerate(modes)
             for x, f in list(m.nodes) + ([extra[mi]] if mi in extra else [])]
    fams = find_families(uproblem, nodes, tol, band, cfg, workers)
    stats = {"n_families": len(fams["families"]), "n_band_components": len(fams["band_components"])}
    for k, m in enumerate(modes, start=1):
        m.family = fams["mode_family"].get(k)
        m.families_all = fams["mode_families_all"].get(k, [])
        m.band_component = fams["mode_band_component"].get(k)
    best_fam = fams["families"][0] if fams["families"] else None
    stats["family_statistics"] = [{"id": fm["id"], "modes": fm["modes"], "n_nodes": len(fm["nodes"]),
                                   "n_starts": int(sum(modes[mi - 1].hits for mi in fm["modes"])),
                                   "best": fm["neg_log_l"]} for fm in fams["families"]]
    stats["best_family_reached_once"] = bool(best_fam and sum(modes[mi - 1].hits for mi in best_fam["modes"]) <= 1)
    say(f"families at the {fams['h']:.1f}-nat level: {len(fams['families'])} "
        f"({sum(1 for m in modes if m.family is not None)} of the {len(modes)} modes; "
        f"{len(fams['band_components'])} component(s) at the band level; {fams['pairs_tested']} pairs tested, "
        f"{fams['nevals']} evaluations)")
    return fams, stats


def refresh_families(uproblem, cfg: dict, modes: list, tol: float, band: float,
                     say: Callable[[str], None], workers: int = 1) -> tuple[dict, dict] | None:
    """After the MCMC: when a hand-back (`uncertainty_for_modes`, UNC-3) has moved a mode more
    than the value tolerance below its best node, the families and their level are stale. Recompute them with each
    moved mode's new point as an extra node, so a family's best node, where its profiles start, is that point.
    `modes` is the final table (best first). Returns None when no mode moved, else (families.json content with
    `recomputed_after_handbacks` = the moved modes' ids, statistics)."""
    moved = {mi: (np.asarray(m.x, float), float(m.f)) for mi, m in enumerate(modes)
             if getattr(m, "nodes", None) and m.f < min(f for _, f in m.nodes) - tol}
    if not moved:
        return None
    say(f"families: recomputed after hand-backs moved mode(s) {[mi + 1 for mi in sorted(moved)]} by more than the "
        f"value tolerance ({tol:.2f} nats)")
    fams, stats = families_for(uproblem, cfg, modes, tol, band, say, extra=moved, workers=workers)
    fams["recomputed_after_handbacks"] = [mi + 1 for mi in sorted(moved)]
    return fams, stats
