"""Systematic swaps and the leave-one-phase-out jackknife (SPEC.md SYS-1 to SYS-5, UNC-7; DESIGN.md
section 3, "left a mode"; FORMATS.md section 3 `[[systematics.swap]]` and section 6 `systematics.json`).

A **swap** replaces one input we do not fully trust by an alternative (SYS-1). It is a configuration block
`{name, set = {key = value, ...}}`; `set` holds configuration keys, as nested tables (TOML `set = {model.ld_mode = 1}`
is the table {"model": {"ld_mode": 1}}) or, from Python, as dotted keys ("model.ld_mode"). `apply_swap` deep-copies
the configuration, merges the keys in (a table merges into a table, so `parameters.u.value` changes only the value of
u; `model.surface` is replaced whole) and checks the result with `config.from_dict`, so unknown keys and bad values stop
with a ConfigError naming the swap.

**Default swaps** (SPEC.md SYS-1, decision #21 of DECISIONS.md, was D3): with `systematics.defaults = true` (the
default) `default_swaps(cfg, problem)` makes, at run time, after the configured swaps:
- "teff_<T>" (T as %g, K) for each selected library slice T (`problem.library.teffs`): `parameters.teff` fixed
  (`free = false`) at `value = T`; skipped, with a note, when the configuration has no teff parameter, for a slice
  outside the parameter's [lo, hi], and for the slice a fixed Teff already has;
- "vi_scale_0.9" and "vi_scale_1.1": `data.vi_scale` = 0.9 and 1.1 (the data reader multiplies the observed V/I and
  its error by it); skipped when no phase fits V/I;
- "library_<k>" (k = 1, 2, ...) for each entry of `systematics.library_alternatives`: `library.path` set to it (other
  library versions cannot be guessed, so none is made without that list);
- "continuum_deg0" (`objective.continuum.degree = 0`; not made when the degree is already 0) and "continuum_off"
  (`objective.continuum = false`: no Legendre continuum and no free constant, decision #23); the mode's entry notes the
  scale ratio of each I spectrum next to "off", which means something only when the I scales agree; neither is made
  without I, or when the continuum is already off.
A configured swap with the same name replaces the default one. The leave-one-phase-out jackknife below is always made.

The **jackknife** (UNC-7) is its own mechanism, not a configuration key: `jackknife_swaps(cfg)` makes one swap per
`[[data.phase]]` block, named "drop_phase_<phase>" (phase as %g), whose configuration lacks that block. With one phase
there is no jackknife (a note says so).

Per mode and swap (`refit`, SYS-2): the swapped configuration is built into its own Problem (always rebuilt, so that a
swap of the library, the Teff list, the data or the parameters takes effect). Starting points: the mode itself plus
n - 1 restart perturbations (`search.perturb`, `[search.perturb]` sizes) made in the table's problem from the stream
SeedSequence([seed, 2, k]) (k = 0-based rank of the mode), so every swap of a mode starts from the same points (SYS-2);
n = 4. The points are mapped into the swapped problem's parameters by name and its fixed parameters set to their
configured values. A start that the swapped problem rejects (cost 1e30 or the prior cut) is
replaced by a perturbation of the mapped mode drawn from the stream SeedSequence([seed, 2, k, s]) (s = swap index), at
most 100 tries. Each start is polished (`search.polish`) and the best end point kept. Every stream depends only on
(seed, k, s), and the pool keeps the task order, so results do not depend on the number of workers.

Quantities: the table problem's free parameters in canonical form plus every derived quantity of
`problem.derived_batch` (array entries as name[j]), the same keys as intervals.json. They and the fingerprint of a
refit are evaluated on the TABLE's problem, with the refit's best point mapped back into its parameter space: the
shifts are shifts of the geometry, measured with the analysis's own definitions (e.g. B50 with the table's limb
darkening, and B50 at a phase that a jackknife refit left out). shift = refit value - mode value.

**Left a mode** (SYS-4; DESIGN.md section 3, step 3): d = `search.fingerprint_distance(mode fingerprint, refit
fingerprint)` with the configured thresholds. With MCMC samples of the mode: "mode change" when d exceeds d99, the 99th
percentile of the same distance over the mode's kept MCMC samples (at most `N_FP_SAMPLES` of them, evenly thinned).
Without samples (fit --no-mcmc, or a mode beyond --mcmc-modes): the search's rule, "mode change" when d > 1
(`search.same_mode` false); the output names the rule used per mode. A mode change replaces every shift of that swap.
The components a swap sets (Teff, for `teff_<T>`) are left out of d and of d99 (`swap_thresholds`).

**Limited by model systematics** (SYS-5): a quantity whose |shift| under any swap exceeds 3 statistical sigma, sigma =
(p84 - p16) / 2 of intervals.json. No sigma (no MCMC, or "n.c."): no flag, and a note says why. Mode changes are listed
apart, not as shifts.

**Jackknife sigma** (UNC-7): over the n leave-one-phase-out refits of a mode, sqrt((n - 1) / n x sum (x_i - mean)^2).
When any of them changed mode the sigma is not given (a note names the phases). It is written to systematics.json
always, and into the `jackknife_sigma` field of intervals.json for the modes that have intervals (`attach_jackknife`,
which touches nothing else; SYS-3: shifts never enter any interval).
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import search as S
from .config import ConfigError, from_dict, spectrum_labels
from .library import on_slice

N_START = 4  # SYS-2
N_FP_SAMPLES = 200  # MCMC samples used for the 99th percentile of the fingerprint distance
P_LEFT = 99.0  # DESIGN.md section 3: percentile of the samples' fingerprint distance
SIGMA_LIMIT = 3.0  # SYS-5
MAX_TRIES = 100
MODE_CHANGE = "mode change"
FAILED = "failed"
RULE_MCMC = "mcmc_p99"
RULE_SEARCH = "search_thresholds"
RULE_TEXT = {RULE_MCMC: "fingerprint distance above the 99th percentile of the mode's MCMC samples (DESIGN.md "
                        "section 3)",
             RULE_SEARCH: "fallback without MCMC samples: fingerprint distance above 1 (the search's thresholds, "
                          "search.same_mode)"}
LIMITED_LABEL = "limited by model systematics"


# ---------------------------------------------------------------------------------------------------------------------
# 1. Swaps


def _flatten(d: dict, prefix: str = "") -> dict:
    """Nested table and/or dotted keys -> {dotted key: value} (vectors and other lists are values)."""
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, p))
        else:
            out[p] = v
    return out


def swap_keys(swap: dict) -> list[str]:
    """The dotted configuration keys a swap sets."""
    return sorted(_flatten(swap.get("set") or {}))


def _set_dotted(raw: dict, key: str, value: Any, name: str) -> None:
    parts = key.split(".")
    if parts[:2] == ["model", "surface"] and len(parts) == 2:
        raw["model"]["surface"] = copy.deepcopy(value)
        return
    t = raw
    for i, p in enumerate(parts[:-1]):
        if p not in t:
            t[p] = {}  # a new [parameters] entry: from_dict fills or refuses it
        if not isinstance(t[p], dict):
            raise ConfigError(f"systematics swap {name!r}: {'.'.join(parts[:i + 1])} is not a table")
        t = t[p]
    t[parts[-1]] = copy.deepcopy(value)


def _raw(cfg: dict) -> dict:
    return copy.deepcopy({k: v for k, v in cfg.items() if k != "_meta"})


def _reload(raw: dict, cfg: dict, name: str) -> dict:
    path = (cfg.get("_meta") or {}).get("path") or None
    try:
        out = from_dict(raw, path)
    except ConfigError as exc:
        raise ConfigError(f"systematics swap {name!r}: {exc}") from None
    return out


def apply_swap(cfg: dict, swap: dict) -> dict:
    """A deep copy of the loaded configuration `cfg` with the keys of `swap["set"]` set, checked by config.from_dict
    (unknown keys and bad values raise ConfigError naming the swap). A swap made by `jackknife_swaps` (it carries
    `drop_phase` = the index of a [[data.phase]] block) instead removes that block. The inherited [[objective.kernel]]
    blocks (the frozen kernels, LIKE-2) of spectra the swap removes are dropped, since config.from_dict refuses a
    label that matches no spectrum; a swap that sets objective.kernel itself is checked as written. A jackknife swap
    also drops the profile names and steps of the left-out phase (`B50_MG[i]`) and moves those of later phases down one
    index, so that each still names the same phase."""
    name = str(swap.get("name", "?"))
    raw = _raw(cfg)
    sets_kernel = False
    if "drop_phase" in swap:
        i = int(swap["drop_phase"])
        del raw["data"]["phase"][i]
        fp = raw.get("search", {}).get("fingerprint")
        if isinstance(fp, dict) and int(fp.get("ref_phase", 0)) >= len(raw["data"]["phase"]):
            fp["ref_phase"] = 0  # the swapped problem's own fingerprint is not used (module docstring)
        uc = raw.get("uncertainty")  # profile names of the left-out phase go, later phases move down one index
        if isinstance(uc, dict):  # (refits do not profile)

            def renumber(q):
                m = re.fullmatch(r"(\w+)\[(\d+)\]", str(q))
                if m is None:
                    return q
                k = int(m.group(2))
                return None if k == i else (q if k < i else f"{m.group(1)}[{k - 1}]")

            if isinstance(uc.get("profile_quantities"), list):
                uc["profile_quantities"] = [renumber(q) for q in uc["profile_quantities"] if renumber(q) is not None]
            if isinstance(uc.get("profile_steps"), dict):
                uc["profile_steps"] = {renumber(q): v for q, v in uc["profile_steps"].items()
                                       if renumber(q) is not None}
    else:
        s = swap.get("set")
        if not isinstance(s, dict) or not s:
            raise ConfigError(f"systematics swap {name!r}: set must be a non-empty table of configuration keys")
        for k, v in _flatten(s).items():
            _set_dotted(raw, k, v, name)
            sets_kernel = sets_kernel or k == "objective.kernel" or k.startswith("objective.kernel.")
    obj = raw.get("objective")
    if not sets_kernel and isinstance(obj, dict) and isinstance(obj.get("kernel"), list) and \
            isinstance(raw.get("data", {}).get("phase"), list):
        labels = spectrum_labels(raw["data"]["phase"])
        obj["kernel"] = [e for e in obj["kernel"] if not isinstance(e, dict) or e.get("spectrum") in labels]
    return _reload(raw, cfg, name)


def jackknife_swaps(cfg: dict) -> tuple[list[dict], str | None]:
    """One swap per [[data.phase]] block, "drop_phase_<phase>", when there are at least 2 phases; else ([], note)."""
    ph = cfg["data"]["phase"]
    if len(ph) < 2:
        return [], f"jackknife skipped: {len(ph)} phase(s); leaving one out needs at least 2 (UNC-7)"
    return [{"name": f"drop_phase_{float(p['phase']):g}", "drop_phase": i, "phase": float(p["phase"])}
            for i, p in enumerate(ph)], None


def configured_swaps(cfg: dict) -> list[dict]:
    return [dict(s) for s in ((cfg.get("systematics") or {}).get("swap") or [])]


VI_SCALES = (0.9, 1.1)  # SYS-1, decision #21


def default_swaps(cfg: dict, problem=None) -> tuple[list[dict], list[str]]:
    """The default swaps of SYS-1 (module docstring; decision #21) as ([{name, set}], notes). `problem` supplies the
    selected library slices (`problem.library.teffs`); without it the Teff swaps use a list of `library.teffs`."""
    sc = cfg.get("systematics") or {}
    if not sc.get("defaults", True):
        return [], ["default swaps off (systematics.defaults = false)"]
    out: list[dict] = []
    notes: list[str] = []
    tp = (cfg.get("parameters") or {}).get("teff")
    lib = getattr(problem, "library", None)
    slices = [float(t) for t in lib.teffs] if lib is not None else (
        [float(t) for t in cfg["library"]["teffs"]] if isinstance(cfg["library"].get("teffs"), list) else [])
    if tp is None:
        notes.append("no teff swaps: the configuration has no teff parameter")
    elif not slices:
        notes.append("no teff swaps: the library slices are not known")
    else:
        lo, hi = float(tp["lo"]), float(tp["hi"])
        for t in slices:
            if not lo <= t <= hi:
                notes.append(f"swap teff_{t:g} skipped: {t:g} K is outside parameters.teff [lo, hi] = [{lo:g}, {hi:g}]")
                continue
            if not tp.get("free", False) and on_slice(float(tp["value"]), t):  # the library's snap rule (library.on_slice)
                notes.append(f"swap teff_{t:g} skipped: teff is already fixed at {t:g} K")
                continue
            out.append({"name": f"teff_{t:g}", "set": {"parameters": {"teff": {"free": False, "value": t}}}})
    if any(bool(p.get("use_VI", True)) for p in cfg["data"]["phase"]):
        base = float(cfg["data"].get("vi_scale", 1.0))
        for f in VI_SCALES:
            out.append({"name": f"vi_scale_{f:g}", "set": {"data": {"vi_scale": base * f}}})
    else:
        notes.append("no V/I scale swaps: no phase fits V/I")
    for k, path in enumerate(sc.get("library_alternatives") or [], start=1):
        out.append({"name": f"library_{k}", "set": {"library": {"path": str(path)}}})
    # decision #23: the continuum treatment of I as a reported choice; degree 0 and off as default swaps
    obj = cfg.get("objective") or {}
    cont = obj.get("continuum", {"degree": 3})
    if not any(bool(p.get("use_I", True)) for p in cfg["data"]["phase"]):
        notes.append("no continuum swaps: no phase fits I")
    elif cont is False:
        notes.append("no continuum swaps: the continuum is already off")
    else:
        if int((cont or {}).get("degree", 3)) != 0:
            out.append({"name": "continuum_deg0", "set": {"objective": {"continuum": {"degree": 0}}}})
        out.append({"name": "continuum_off", "set": {"objective": {"continuum": False}}})
    return out, notes


# ---------------------------------------------------------------------------------------------------------------------
# 2. Parameter spaces


def map_point(src, x: np.ndarray, dst, fix: bool = True) -> np.ndarray:
    """x in `src`'s parameters -> `dst`'s, by name; names `dst` has and `src` lacks take dst.value. With `fix`, dst's
    fixed parameters are set to their configured values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(dst.value, dtype=float).copy()
    sv = dict(zip(src.names, x))
    for i, n in enumerate(dst.names):
        if n in sv:
            y[i] = sv[n]
    if fix:
        free = np.asarray(dst.free, dtype=bool)
        y[~free] = np.asarray(dst.value, dtype=float)[~free]
    return y


def quantities(problem, X: np.ndarray) -> dict[str, np.ndarray]:
    """{name: (K,)}: free parameters in canonical form, then derived_batch (array entries as name[j]); the keys of
    intervals.json."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Xc = np.array([problem.canonical(x) for x in X])
    out = {n: Xc[:, i] for i, n in enumerate(problem.names) if problem.free[i]}
    fb = getattr(problem, "derived_batch", None)
    if fb is not None:
        for k, v in fb(X).items():
            v = np.asarray(v, dtype=float)
            if v.ndim == 1:
                out[k] = v
            else:
                for j in range(v.shape[1]):
                    out[f"{k}[{j}]"] = v[:, j]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 3. One refit


@dataclass
class SwapResult:
    name: str
    mode: int  # 1-based mode id
    keys: list[str]
    x: np.ndarray | None  # best end point, in the TABLE problem's parameter space
    x_swapped: np.ndarray | None  # the same in the swapped problem's space
    neg_log_l: float  # -ln L of the refit (swapped problem)
    start_neg_log_l: list[float]
    n_starts: int
    nevals: int
    redraws: int
    error: str | None = None
    fingerprint: Any = None
    distance: float | None = None
    mode_change: bool | None = None

    def to_dict(self) -> dict:
        return {"keys": self.keys, "neg_log_l": None if self.error else float(self.neg_log_l),
                "start_neg_log_l": [float(v) for v in self.start_neg_log_l], "n_starts": self.n_starts,
                "nevals": self.nevals, "redraws": self.redraws,
                "x": None if self.x is None else [float(v) for v in self.x],
                "fingerprint": None if self.fingerprint is None else self.fingerprint.to_dict(),
                "fingerprint_distance": self.distance, "mode_change": self.mode_change, "error": self.error}


def mode_starts(problem, x_mode: np.ndarray, cfg: dict, seed: int, k: int) -> np.ndarray:
    """The mode itself and N_START - 1 perturbations from SeedSequence([seed, 2, k]) (the same for every swap of mode
    k)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2, int(k)]))
    x0 = np.asarray(x_mode, dtype=float)
    return np.array([x0] + [S.perturb(problem, x0, cfg, rng) for _ in range(N_START - 1)])


def _valid(problem, x: np.ndarray, outside_max: float) -> bool:
    ok, _ = S._start_ok(problem, x, outside_max)
    return ok


def refit(problem_swapped, table_problem, x_mode: np.ndarray, starts: np.ndarray, cfg_swapped: dict,
          rng: np.random.Generator, name: str = "", mode: int = 0, keys: list[str] | None = None) -> SwapResult:
    """SYS-2 for one mode and swap (module docstring): polish n starts in the swapped problem, keep the best; the best
    point is also returned mapped back into the table problem's space (not fixed there)."""
    ps = problem_swapped
    outside_max = float(cfg_swapped["search"]["starts"].get("outside_max", math.inf))
    xm = map_point(table_problem, x_mode, ps)
    X0, redraws, nevals = [], 0, 0
    for x in starts:
        y = map_point(table_problem, x, ps)
        nevals += 1
        if not _valid(ps, y, outside_max):
            for _ in range(MAX_TRIES):
                redraws += 1
                y = S.perturb(ps, xm, cfg_swapped, rng)
                nevals += 1
                if _valid(ps, y, outside_max):
                    break
        X0.append(y)
    res = [S.polish(ps, y, cfg_swapped) for y in X0]
    nevals += sum(r.nevals for r in res)
    best = min(range(len(res)), key=lambda i: (res[i].f, i))
    xb = np.asarray(res[best].x, dtype=float)
    return SwapResult(name, mode, list(keys or []), map_point(ps, xb, table_problem, fix=False), xb,
                      float(res[best].f), [float(r.f) for r in res], len(res), nevals, redraws)


def _refit_task(task) -> SwapResult:
    """Picklable unit of work: build the swapped Problem and refit (run in-process or in a pool worker)."""
    from .problem import Problem

    table_problem, cfg2, x_mode, starts, seed, k, s, name, mid, keys = task
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2, int(k), int(s)]))
    try:
        ps = Problem(cfg2)
        return refit(ps, table_problem, x_mode, starts, cfg2, rng, name, mid, keys)
    except (ConfigError, ValueError) as exc:  # e.g. a swap the problem cannot be built with: recorded, not fatal
        return SwapResult(name, mid, list(keys), None, None, math.nan, [], 0, 0, 0, error=str(exc))


# ---------------------------------------------------------------------------------------------------------------------
# 4. The whole step


@dataclass
class SystematicsResult:
    swaps: list[dict]  # {name, kind, keys | phase}
    modes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_refits: int = 0
    n_mode_changes: int = 0
    nevals: int = 0
    wall_time_s: float = 0.0
    processes: int = 1

    def to_dict(self) -> dict:
        """systematics.json (FORMATS.md section 6): per mode `shifts[quantity][swap]` = shift of the best value or
        "mode change" ("failed" when the swapped problem could not be built), plus per swap the refit's -ln L, number
        of starts and the swapped keys."""
        return {"swaps": self.swaps, "rules": dict(RULE_TEXT), "sigma_limit": SIGMA_LIMIT,
                "note_sys3": "shifts are reported separately and never added to any interval (SYS-3)",
                "modes": self.modes, "notes": self.notes, "n_refits": self.n_refits,
                "n_mode_changes": self.n_mode_changes, "nevals": self.nevals, "wall_time_s": self.wall_time_s,
                "processes": self.processes}


def _mode_samples_d99(problem, mode, mcmc, thr: dict) -> float | None:
    """99th percentile of the fingerprint distance of the mode's kept MCMC samples to the mode; None without samples."""
    if mcmc is None:
        return None
    Z = mcmc.kept()
    if Z.shape[0] == 0:
        return None
    idx = np.unique(np.linspace(0, Z.shape[0] - 1, min(N_FP_SAMPLES, Z.shape[0])).astype(int))
    free = np.flatnonzero(np.asarray(problem.free, dtype=bool))
    base = np.asarray(problem.value, dtype=float)
    d = []
    for z in Z[idx]:
        x = base.copy()
        x[free] = z
        d.append(S.fingerprint_distance(mode.fingerprint, S.fingerprint(problem, x), thr))
    return float(np.percentile(d, P_LEFT))


def swap_thresholds(thr: dict, entry: dict) -> dict:
    """The fingerprint thresholds for judging a swap's refit (SYS-4): the components the swap
    sets are left out, so a swap that fixes Teff at a slice 1000 K or more away is not a "mode change" by construction.
    The fingerprint's non-field components are the free non-field parameters (`extra`); a swap that sets one of them
    in `parameters` (its dotted `keys`, e.g. "parameters.teff.value", or a `set` table) removes it."""
    set_params = set(((entry.get("set") or {}).get("parameters") or {}))
    set_params |= {k.split(".")[1] for k in entry.get("keys") or [] if k.startswith("parameters.") and k.count(".") >= 1}
    out = dict(thr)
    out["extra"] = {k: v for k, v in (thr.get("extra") or {}).items() if k not in set_params}
    return out


def _interval_sigma(entry: dict | None) -> float | None:
    if not entry or entry.get("p16") is None or entry.get("p84") is None:
        return None
    return 0.5 * (float(entry["p84"]) - float(entry["p16"]))


def limited_by_systematics(shifts: dict, iv_mode: dict | None) -> tuple[list[str], str | None]:
    """SYS-5: the quantities whose |shift| under any swap exceeds SIGMA_LIMIT x sigma, sigma = (p84 - p16) / 2 of the
    mode's intervals.json entry; "mode change" and "failed" entries are not shifts. Returns (quantities, note): the
    note says what could not be assessed (no intervals for the mode; quantities without a sigma, e.g. "n.c.")."""
    if iv_mode is None:
        return [], "not assessed: no intervals for this mode (no MCMC)"
    limited, nc = [], []
    for q, per_swap in shifts.items():
        sig = _interval_sigma(iv_mode["quantities"].get(q))
        if sig is None:
            nc.append(q)
            continue
        moved = [abs(v) for v in per_swap.values() if isinstance(v, (int, float))]
        if moved and max(moved) > SIGMA_LIMIT * sig:
            limited.append(q)
    note = None
    if nc:
        note = (f"not assessed for {len(nc)} quantit{'y' if len(nc) == 1 else 'ies'} without a statistical sigma "
                "(n.c. or not in intervals.json)")
    return limited, note


def run_systematics(problem, cfg: dict, modes: list, uncertainty: dict | None, seed: int, workers: int = 1,
                    log: Callable[[str], None] | None = None, swaps: list[dict] | None = None,
                    checkpoint: tuple[Callable, Callable] | None = None) -> SystematicsResult:
    """Every configured swap and every jackknife swap, for every mode of `modes` (the final table, best first; ids
    1-based). `problem` is the table's problem (the one whose -ln L the modes carry) and `cfg` its configuration;
    `uncertainty` is fit's {"intervals": intervals.json dict, "results": {mode id: McmcResult}} or None (--no-mcmc).
    `swaps` overrides the configured list (tests). `checkpoint` = (get, put): `get(inputs)` returns the
    SwapResult of an earlier refit with these inputs (a JSON-able list) or None, `put(inputs, result)` stores each
    refit as it ends."""
    import time

    t0 = time.perf_counter()
    say = log or (lambda _m: None)
    sw = configured_swaps(cfg) if swaps is None else [dict(s) for s in swaps]
    dflt, dnotes = default_swaps(cfg, problem)
    taken = {str(s["name"]) for s in sw}
    dflt = [s for s in dflt if s["name"] not in taken]  # a configured swap of the same name replaces the default
    jk, jk_note = jackknife_swaps(cfg)
    entries = [{"name": str(s["name"]), "kind": "config", "keys": swap_keys(s)} for s in sw]
    entries += [{"name": s["name"], "kind": "default", "keys": swap_keys(s)} for s in dflt]
    entries += [{"name": s["name"], "kind": "jackknife", "phase": s["phase"], "keys": [f"data.phase[{s['drop_phase']}]"]}
                for s in jk]
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise ConfigError(f"systematics: swap names must be distinct, got {names}")
    out = SystematicsResult(entries)
    out.notes.extend(dnotes)
    if jk_note:
        out.notes.append(jk_note)
    if not entries:
        out.notes.append("no swaps: the configuration lists no [[systematics.swap]], makes no default swap and the "
                         "data have one phase")
        return out
    cfgs = [apply_swap(cfg, s) for s in sw + dflt + jk]  # configuration errors stop here, before any refit
    keys = [e["keys"] for e in entries]
    thr = S.fingerprint_thresholds(cfg)
    results = (uncertainty or {}).get("results") or {}
    iv_by_id = {e["mode"]: e for e in ((uncertainty or {}).get("intervals") or {}).get("modes", [])}

    tasks, starts_by_mode = [], []
    for k, m in enumerate(modes):
        st = mode_starts(problem, m.x, cfg, seed, k)
        starts_by_mode.append(st)
        for s, c2 in enumerate(cfgs):
            tasks.append((problem, c2, np.asarray(m.x, dtype=float), st, int(seed), k, s, names[s], k + 1, keys[s]))
    w = max(1, min(int(workers), len(tasks)))
    say(f"systematics: {len(entries)} swap(s) ({len(sw)} configured, {len(dflt)} default, {len(jk)} jackknife) x "
        f"{len(modes)} mode(s), "
        f"{w} process(es)")
    get, put = checkpoint or (lambda _i: None, lambda _i, _r: None)
    def read(c: dict) -> dict:  # the sections a refit reads (not uncertainty, not the swap list)
        return {k: v for k, v in _raw(c).items() if k not in ("uncertainty", "systematics")}

    inputs = [[read(t[0].cfg), read(t[1]), t[2].tolist(), t[3].tolist(), *t[4:8]] for t in tasks]
    refits = [get(i) for i in inputs]
    todo = [j for j, r in enumerate(refits) if r is None]
    if len(todo) < len(tasks):
        say(f"systematics: {len(tasks) - len(todo)} of {len(tasks)} refit(s) from checkpoints")

    def done(j: int, r: SwapResult) -> None:
        refits[j] = r
        put(inputs[j], r)

    if w > 1:
        from concurrent.futures import as_completed

        with S.make_pool(w) as pool:
            futs = {pool.submit(_refit_task, tasks[j]): j for j in todo}
            for f in as_completed(futs):
                done(futs[f], f.result())
    else:
        with S.single_threaded():
            for j in todo:
                done(j, _refit_task(tasks[j]))
    out.processes = w
    out.n_refits = len(refits)
    out.nevals = sum(r.nevals for r in refits)

    for k, m in enumerate(modes):
        mid = k + 1
        rs = refits[k * len(entries):(k + 1) * len(entries)]
        mcmc = results.get(mid)
        d99 = _mode_samples_d99(problem, m, mcmc, thr)
        rule = RULE_MCMC if d99 is not None else RULE_SEARCH
        limit = d99 if d99 is not None else 1.0
        d99_by: dict = {}  # left-a-mode limit per set of components a swap leaves out (swap_thresholds)
        ok = [r for r in rs if r.error is None]
        X = np.array([np.asarray(m.x, dtype=float)] + [r.x for r in ok])
        Q = quantities(problem, X)
        qnames = list(Q)
        vals = {r.name: {q: float(Q[q][1 + i]) for q in qnames} for i, r in enumerate(ok)}
        mode_vals = {q: float(Q[q][0]) for q in qnames}
        mnotes = []
        if mcmc is not None and not getattr(mcmc, "converged", True):
            mnotes.append("left-a-mode percentile from the samples of an MCMC run that did not converge")
        entry_of = {r.name: e for r, e in zip(rs, entries)}
        for r in ok:
            thr_r = swap_thresholds(thr, entry_of[r.name])
            key = tuple(sorted(thr_r["extra"]))
            if key not in d99_by:
                d99_by[key] = d99 if key == tuple(sorted(thr.get("extra") or {})) else _mode_samples_d99(
                    problem, m, mcmc, thr_r)
            lim_r = d99_by[key] if d99_by[key] is not None else 1.0
            r.fingerprint = S.fingerprint(problem, r.x)
            r.distance = float(S.fingerprint_distance(m.fingerprint, r.fingerprint, thr_r))
            r.mode_change = bool(r.distance > lim_r)
            out.n_mode_changes += int(r.mode_change)
        shifts: dict[str, dict] = {q: {} for q in qnames}
        for r in rs:
            for q in qnames:
                if r.error is not None:
                    shifts[q][r.name] = FAILED
                elif r.mode_change:
                    shifts[q][r.name] = MODE_CHANGE
                else:
                    shifts[q][r.name] = vals[r.name][q] - mode_vals[q]
            if r.error is not None:
                mnotes.append(f"swap {r.name}: failed: {r.error}")

        limited, sys5_note = limited_by_systematics(shifts, iv_by_id.get(mid))

        # UNC-7
        jk_sigma, jk_mnote = None, None
        jrs = [r for r, e in zip(rs, entries) if e["kind"] == "jackknife"]
        if jrs:
            bad = [r.name for r in jrs if r.error is not None or r.mode_change]
            if bad:
                jk_mnote = f"jackknife sigma not given: {', '.join(bad)} changed mode or failed"
            else:
                n = len(jrs)
                jk_sigma = {}
                for q in qnames:
                    xi = np.array([vals[r.name][q] for r in jrs])
                    jk_sigma[q] = float(math.sqrt((n - 1) / n * float(np.sum((xi - xi.mean()) ** 2))))
        changed = [r.name for r in ok if r.mode_change]
        scale_ratio = None
        if any(r.name == "continuum_off" for r in rs):  # "off" means something only when the I scales agree (SYS-1)
            I_m, _ = problem.model_spectra(np.asarray(m.x, dtype=float))
            scale_ratio = problem.objective.check_scale(I_m)
        out.modes.append({"scale_ratio_continuum_off": scale_ratio,
            "mode": mid, "neg_log_l": float(m.f), "left_mode_rule": rule, "left_mode_rule_text": RULE_TEXT[rule],
            "d99": d99, "left_mode_limit": float(limit), "n_starts": N_START if not ok else max(
                r.n_starts for r in ok),
            "values": mode_vals, "swaps": {r.name: r.to_dict() for r in rs}, "shifts": shifts,
            "mode_changes": changed, "limited_by_model_systematics": limited, "sys5_note": sys5_note,
            "jackknife_sigma": jk_sigma, "jackknife_n": len(jrs), "jackknife_note": jk_mnote, "notes": mnotes})
        say(f"systematics: mode {mid}: {len(rs)} refit(s), rule {rule} (limit {limit:.3g}), "
            f"{len(changed)} mode change(s){': ' + ', '.join(changed) if changed else ''}")
    out.wall_time_s = time.perf_counter() - t0
    return out


def attach_jackknife(intervals: dict, sysres: dict) -> list[int]:
    """Write each mode's jackknife sigma into intervals.json's `jackknife_sigma` fields (in place; nothing else is
    touched). Returns the mode ids written."""
    by_id = {e["mode"]: e for e in sysres.get("modes", [])}
    done = []
    for e in intervals.get("modes", []):
        s = by_id.get(e["mode"])
        if not s or not s.get("jackknife_sigma"):
            continue
        for q, entry in e["quantities"].items():
            if q in s["jackknife_sigma"]:
                entry["jackknife_sigma"] = s["jackknife_sigma"][q]
        done.append(e["mode"])
    return done
