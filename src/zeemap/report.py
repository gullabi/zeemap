"""Output files of `eval` and `fit` (FORMATS.md section 6; SPEC.md REP-1, SEARCH-3, SEARCH-4, DIAG-2, PAR-4).

- `spectra/<prefix>_ph<phase>.dat` (phase as %.3f): columns wavelength, model I (times the best continuum, as fitted
  inside the likelihood; NaN outside the span of the fitted I pixels), model V/I, data I, data V/I, residual of I and
  of V/I in units of the errors ((data - model) / sigma; NaN where the pixel is not fitted), raw model I (no
  continuum). A `#` header line names the columns.
- `histograms/mode<k>_ph<phase>.dat`: B (MG), cos psi, brightness-weighted visible area (area x mu x LD(mu)) on the
  library's (B, cos psi) nodes, from forward.node_weights (the bilinear shares of the lookup); one row per node.
- `run.json`, `modes.json`, `diagnostics.json`, `report.txt`: see `write_fit`.
- with error bars (`fit` without `--no-mcmc`): `intervals.json` (uncertainty.UncertaintyResult.to_dict with 1-based
  mode ids matching modes.json, plus label, patches, modes_without_mcmc, processes, notes, `_meta.wall_time_s`) and
  `samples/mode<k>.npz` (McmcResult.save_npz: the whole chain incl. burn-in, -ln L per sample, names, burn). report.txt
  then has a second table: per mode the 68 % intervals "median +a -b" of incl, |dipole|, |c|, B50 at the first phase
  and teff (plus beta and the offset along the dipole axis), the labels of UNC-1 and UNC-4, the convergence summary,
  the flags and the hand-back counts. No interval is printed for a quantity that fails the V14 rules on its own
  (UNC-2 per quantity, decision #17: "n.c."; the mode-level verdict is printed as before) or for an orientation
  quantity of a single-phase fit ("orient.", declared nuisance, decision #18); no evidence weights (REP-2).
- with systematics (`fit` without `--no-systematics`; systematics.py): `systematics.json` (SystematicsResult.to_dict
  plus modes_without_systematics and jackknife_in_intervals), the jackknife sigma in intervals.json's
  `jackknife_sigma` fields (only there; SYS-3), a "jk" row under each mode of the error-bar table when the mode has a
  jackknife sigma, and, after the error-bar table and apart from it (SYS-3, V4f), one line per mode and swap
  "systematic shifts of mode k: <swap>: incl +x; |dipole| +y; ..." (or "mode change"), the jackknife sigma per mode,
  and per mode "limited by model systematics (a swap moves it by more than 3 sigma): <quantities>" (SYS-5; SPEC.md
  section 4).
- a rerun into the same directory: `write_fit` first removes every file a fit writes (FIT_FILES, and `mode<k>_*` in
  spectra/, histograms/ and samples/), so no file of an earlier fit survives next to the new run.json; other files
  (eval.json, `eval_*` spectra, the user's own) are kept.
- when the whiteness check (LIKE-10) failed for any spectrum: `noise_model_flag` and `noise_model_failing_spectra` at
  the top level of modes.json, families.json, profiles.json, systematics.json, diagnostics.json and intervals.json
  (whichever are written; intervals.json then also says "uncalibrated", UNC-4), and a line "[<flag>: <spectra>]" on
  each table of report.txt (`noise_flag`).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import atomic
from . import __version__
from .config import dump, portable

SPECTRA_COLUMNS = ["wavelength", "model_I", "model_VI", "data_I", "data_VI", "resid_I", "resid_VI", "model_I_raw"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path, obj) -> None:
    atomic.write_text(path, json.dumps(_jsonable(obj), indent=2) + "\n")


def phase_tag(phase: float) -> str:
    return f"{phase:.3f}"


def write_spectra(outdir, problem, x, prefix: str) -> list[str]:
    """spectra/<prefix>_ph<phase>.dat for every phase; returns the file names."""
    d = Path(outdir) / "spectra"
    d.mkdir(parents=True, exist_ok=True)
    ds = problem.dataset
    I, VI = problem.model_spectra(x)
    Ic = problem.continuum_models(I, VI)
    lam = ds.grid.wavelengths
    names = []
    for p, ph in enumerate(ds.phases):
        rI = np.full(lam.size, np.nan)
        rV = np.full(lam.size, np.nan)
        for s in ds.spectra:
            if s.phase_index != p:
                continue
            model = Ic[p] if s.kind == "I" else VI[p]
            r = rI if s.kind == "I" else rV
            r[s.pixels] = (np.asarray(s.data) - model[s.pixels]) / np.asarray(s.sigma)
        cols = np.column_stack([lam, Ic[p], VI[p], ds.I[p], ds.VI[p], rI, rV, I[p]])
        f = d / f"{prefix}_ph{phase_tag(ph)}.dat"
        atomic.savetxt(f, cols, fmt="%.8g", header=" ".join(SPECTRA_COLUMNS))
        names.append(f"spectra/{f.name}")
    return names


def write_histograms(outdir, problem, x, prefix: str) -> list[str]:
    d = Path(outdir) / "histograms"
    d.mkdir(parents=True, exist_ok=True)
    fw = problem.forward
    W = fw.node_weights(problem.theta(x))  # (P, NB, NC)
    bb, cc = np.meshgrid(fw.bnodes, fw.cnodes, indexing="ij")
    names = []
    for p, ph in enumerate(fw.phases - fw.phase_offset):
        f = d / f"{prefix}_ph{phase_tag(ph)}.dat"
        atomic.savetxt(f, np.column_stack([bb.ravel(), cc.ravel(), W[p].ravel()]), fmt="%.8g",
                   header="B_MG cos_psi area_weighted")
        names.append(f"histograms/{f.name}")
    return names


# ------------------------------------------------------------------------------------------------- run information


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(cfg: dict, problem) -> dict:
    """{path: sha256} of the configuration, the library files in use and the data files."""
    files = []
    if cfg.get("_meta", {}).get("path"):
        files.append(cfg["_meta"]["path"])
    lib = Path(cfg["library"]["path"])
    files.append(lib / "info.dat")
    from .library import slice_path

    for t in problem.library.teffs:
        files.append(slice_path(lib, problem.library.info, float(t)))
    for pc in cfg["data"]["phase"]:
        files += [pc[k] for k in ("file", "ierr", "vierr") if pc.get(k)]
    jf = Path(cfg["data"]["path"]) / "dataset.json"
    if jf.is_file():
        files.append(jf)
    return {str(f): sha256(f) for f in files if Path(f).is_file()}


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():  # an installed zeemap: never record the commit of some enclosing repository
        return None
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        commit = r.stdout.strip()
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10)
        return commit + ("-dirty" if d.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def versions() -> dict:
    out = {"python": sys.version.split()[0], "zeemap": __version__, "zeemap_commit": _git_commit()}
    for mod in ("numpy", "scipy", "numba", "llvmlite"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------- eval


def parameter_space(problem) -> dict:
    """The parameters as the problem uses them, for run.json: names, free flags, limits, steps and values after the
    clamps made at load, the length ranges, and `limit_notes` (one sentence per parameter whose value or limits were
    clamped at load, e.g. Teff beyond an edge slice; [] when none)."""
    return {"names": problem.names,
            "free": [bool(v) for v in problem.free], "lo": problem.lo, "hi": problem.hi, "step": problem.step,
            "value": problem.value, "length_range": problem.length_range,
            "limit_notes": list(getattr(problem, "limit_notes", []))}


def write_eval(outdir, problem, cfg: dict) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    x = problem.value.copy()
    ev = problem.evaluation(x)
    ev["spectra"] = write_spectra(out, problem, x, "eval")
    ev["spectra_columns"] = SPECTRA_COLUMNS
    ev["config"] = dump(cfg)
    ev["versions"] = versions()
    ev["time"] = now()
    write_json(out / "eval.json", ev)
    return ev


# ------------------------------------------------------------------------------------------------- fit


def _written_value(m) -> float:
    """The cost at the vector written for mode m (search.Mode.f_written; m.f for mode-like objects without it)."""
    fc = getattr(m, "f_canonical", None)
    return float(m.f if fc is None else fc)


def mode_entry(problem, m, k: int, best: float) -> dict:
    """One modes.json entry. `vector` is the canonical point and `neg_log_l` the cost there (search.settle_mirror);
    `neg_log_l_mode` is the mode's value, the better of the end point and its equator-mirror image, which ranks the
    modes and gives `delta`; `image` says which of the two that is ("mirror image": the mirror of the polished end
    point); `mirror_asymmetry` = f(mirror image) - f(end point), the patch grid's quadrature noise."""
    x = np.asarray(m.x_canonical if m.x_canonical is not None else m.x, dtype=float)
    asym = getattr(m, "mirror_asymmetry", None)
    return {"id": k, "parameters": problem.grouped(x), "vector": {n: float(v) for n, v in zip(problem.names, x)},
            "derived": problem.derived(x), "neg_log_l": _written_value(m), "neg_log_l_mode": float(m.f),
            "delta": float(m.f - best), "image": getattr(m, "image", "end point"),
            "mirror_asymmetry": None if asym is None else float(asym),
            "n_starts": int(m.hits), "n_endpoints": len(m.members), "patches": int(problem.surface.n),
            "touched_limit": bool(m.touched_limit), "fingerprint": m.fingerprint.to_dict(),
            "family": getattr(m, "family", None), "families_all": list(getattr(m, "families_all", []) or []),
            "band_component": getattr(m, "band_component", None)}


ERRBAR_COLUMNS = [("incl deg", "incl_deg", ".1f"), ("|dipole| MG", "dipole_MG", ".3f"),
                  ("beta deg", "beta_deg", ".1f"), ("|c|", "c_abs", ".3f"), ("c along axis", "c_along_axis", ".3f"),
                  ("B50 ph0 MG", "B50_MG[0]", ".3f"), ("teff K", "teff", ".0f")]
FLAG_CODES = {"touches a range limit": "L", "not constrained by the data": "U", "u reached 0": "0"}
ORIENTATION = "orientation not determined by one phase"  # uncertainty.ORIENTATION (decision #18)
NUISANCE = "nuisance, not required"


def interval_text(q: dict | None, fmt: str) -> str:
    """"median +a -b [flags]" of one intervals.json quantity (68 %); "n.c." when withheld because the quantity failed
    the V14 rules (UNC-2, per quantity, decision #17); "orient." for a declared-nuisance orientation quantity
    (decision #18); "-" if absent."""
    if q is None:
        return "-"
    codes = ",".join(FLAG_CODES[f] for f in q.get("flags", []) if f in FLAG_CODES)
    tail = f" [{codes}]" if codes else ""
    if q.get("median") is None:
        return ("orient." if ORIENTATION in q.get("flags", []) else "n.c.") + tail
    m, lo, hi = q["median"], q["p16"], q["p84"]
    if hi - lo == 0.0:
        return f"{m:{fmt}} (fixed)" + tail
    return f"{m:{fmt}} +{hi - m:{fmt}} -{m - lo:{fmt}}" + tail


def jk_text(q: dict | None, fmt: str) -> str:
    v = None if q is None else q.get("jackknife_sigma")
    return "-" if v is None else f"{v:{fmt}}"


SHIFT_COLUMNS = [("incl", "incl_deg", ".2f"), ("|dipole|", "dipole_MG", ".3f"), ("|c|", "c_abs", ".3f"),
                 ("B50 ph0", "B50_MG[0]", ".3f"), ("teff", "teff", ".0f"), ("beta", "beta_deg", ".2f"),
                 ("c along axis", "c_along_axis", ".3f")]


def _shift_text(v, fmt: str) -> str:
    return f"{v:+{fmt}}" if isinstance(v, (int, float)) else str(v)


def systematics_text(sysd: dict | None, reason: str | None = None) -> list[str]:
    """The systematics part of report.txt (module docstring); always after the error-bar table (SYS-3)."""
    if sysd is None:
        return [f"systematics: not computed ({reason or 'fit --no-systematics'})"]
    if not sysd.get("swaps"):
        return ["systematics: none run: " + "; ".join(sysd.get("notes") or ["no swaps"])]
    L = []
    nconf = sum(1 for s in sysd["swaps"] if s["kind"] == "config")
    ndef = sum(1 for s in sysd["swaps"] if s["kind"] == "default")
    njk = len(sysd["swaps"]) - nconf - ndef
    dtext = f"{ndef} default (SYS-1), " if ndef else ""
    L.append(f"systematics: {len(sysd['swaps'])} swap(s) ({nconf} configured, {dtext}{njk} leave-one-phase-out) x "
             f"{len(sysd['modes'])} mode(s); {sysd['n_mode_changes']} refit(s) changed mode. Shifts = refit value - "
             "mode value; listed apart from the error bars and never added to them (SYS-3)")
    for m in sysd["modes"]:
        k = m["mode"]
        rule = "MCMC samples, 99th percentile" if m["left_mode_rule"] == "mcmc_p99" else \
            "no MCMC samples: the search's fingerprint thresholds"
        L.append(f"  mode {k}: 'mode change' rule: fingerprint distance > {m['left_mode_limit']:.3g} ({rule})")
        for s in sysd["swaps"]:
            name = s["name"]
            parts = []
            for lab, key, fmt in SHIFT_COLUMNS:
                v = m["shifts"].get(key, {}).get(name)
                if v is None:
                    continue
                if v in ("mode change", "failed"):
                    parts = [v]
                    break
                parts.append(f"{lab} {_shift_text(v, fmt)}")
            extra = ""
            if name == "continuum_off" and m.get("scale_ratio_continuum_off"):  # SYS-1: the I scales of model and data
                extra = " (scale ratio median |model I| / median |data I|: " + ", ".join(
                    f"{lab} {v:.3g}" for lab, v in m["scale_ratio_continuum_off"].items()) + ")"
            L.append(f"systematic shifts of mode {k}: {name}: {'; '.join(parts) if parts else '-'}{extra}")
        jk = m.get("jackknife_sigma")
        if jk:
            parts = [f"{lab} {jk[key]:{fmt}}" for lab, key, fmt in SHIFT_COLUMNS if key in jk]
            L.append(f"jackknife sigma of mode {k} ({m['jackknife_n']} phases left out in turn, UNC-7): "
                     + "; ".join(parts))
        elif m.get("jackknife_note"):
            L.append(f"jackknife sigma of mode {k}: {m['jackknife_note']}")
        lim = m.get("limited_by_model_systematics") or []
        if m.get("sys5_note") and not lim:
            L.append(f"limited by model systematics, mode {k}: {m['sys5_note']}")
        else:
            L.append(f"limited by model systematics (a swap moves it by more than 3 sigma), mode {k}: "
                     f"{', '.join(lim) if lim else 'none'}" + (f" ({m['sys5_note']})" if m.get("sys5_note") else ""))
        if m.get("mode_changes"):
            L.append(f"  mode {k}: swaps that left the mode (SYS-4): {', '.join(m['mode_changes'])}")
        for note in m.get("notes", []):
            L.append(f"  note: {note}")
    if sysd.get("modes_without_systematics"):
        L.append(f"  no systematics (--sys-modes): mode(s) {', '.join(str(k) for k in sysd['modes_without_systematics'])}")
    for note in sysd.get("notes", []):
        L.append(f"  note: {note}")
    L.append("  every quantity, swap and start: systematics.json")
    return L


def _finite_vals(d: dict) -> list[float]:
    return [float(v) for v in d.values() if v is not None and math.isfinite(float(v))]


def errorbar_text(intervals: dict) -> list[str]:
    """The error-bar part of report.txt (see the module docstring)."""
    L = []
    by_id = {e["mode"]: e for e in intervals["modes"]}
    L.append("error bars (68 % intervals: posterior median +(p84 - median) -(median - p16); n.c. = this quantity "
             "fails the V14 rules, no interval reported (UNC-2); orient. = orientation not determined by one phase, "
             "a declared nuisance)")
    widths = [max(24, len(h)) for h, _, _ in ERRBAR_COLUMNS]
    L.append(f"{'mode':>4}  " + "  ".join(f"{h:<{w}}" for (h, _, _), w in zip(ERRBAR_COLUMNS, widths)))
    for kid in sorted(by_id):
        q = by_id[kid]["quantities"]
        L.append(f"{kid:>4}  " + "  ".join(f"{interval_text(q.get(key), fmt):<{w}}"
                                           for (_, key, fmt), w in zip(ERRBAR_COLUMNS, widths)))
        if any((q.get(key) or {}).get("jackknife_sigma") is not None for _, key, _ in ERRBAR_COLUMNS):
            L.append(f"{'jk':>4}  " + "  ".join(f"{jk_text(q.get(key), fmt):<{w}}"
                                               for (_, key, fmt), w in zip(ERRBAR_COLUMNS, widths)))
    if intervals.get("modes_without_mcmc"):
        L.append(f"  no MCMC (--mcmc-modes / uncertainty.modes_max): mode(s) "
                 f"{', '.join(str(k) for k in intervals['modes_without_mcmc'])}")
    L.append("  flags: L = touches a range limit (REP-3), U = not constrained by the data (REP-4), 0 = u reached 0")
    if any(e.get("jackknife_sigma") is not None for m in intervals["modes"] for e in m["quantities"].values()):
        L.append("  jk = jackknife sigma over the leave-one-phase-out refits (UNC-7); not part of the intervals")
    cal = intervals.get("calibration", "uncalibrated")
    cal_text = "uncalibrated: no coverage run named in the configuration" if cal == "uncalibrated" else cal
    L.append(f"intervals: statistical (68 %, per mode: conditional on this basin), {cal_text}")
    L.append(f"  MCMC on {intervals.get('patches')} patches, {intervals.get('nevals')} evaluations in all; 95 % "
             "intervals and every other quantity: intervals.json")
    L.append("")
    L.append("convergence (V14) per mode:")
    for kid in sorted(by_id):
        e = by_id[kid]
        c = e["convergence"]
        verdict = "converged" if e["converged"] else (
            "NOT CONVERGED (" + "; ".join(c.get("failed_text") or e["failed_rules"] or ["evaluation budget"]) + ")")
        tau = "n/a" if c.get("tau") is None else f"{c['tau']:.1f}"
        ess = _finite_vals(c.get("ess_bulk", {}))
        rh = _finite_vals(c.get("rhat", {}))
        acc = c.get("acceptance") or []
        L.append(f"  mode {kid}: {verdict}")
        L.append(f"    tau {tau} (needed: chain >= tau_factor x tau), ESS bulk min "
                 f"{'n/a' if not ess else f'{min(ess):.0f}'}, R-hat max {'n/a' if not rh else f'{max(rh):.3f}'}, "
                 f"acceptance {'n/a' if not acc else f'{min(acc):.2f}-{max(acc):.2f}'}; {c['ensembles']} ensembles x "
                 f"{c['nwalkers']} walkers x {c['steps']} steps (burn {c['burn']}), {c['nevals']} evaluations "
                 f"(max {c['max_evals']}); hand-backs {len(e.get('handbacks', []))}")
        nuis = c.get("nuisance") or []
        if nuis:
            stats = []
            for n in nuis:
                r = (c.get("rhat") or {}).get(n)
                b = (c.get("ess_bulk") or {}).get(n)
                if r is not None or b is not None:
                    stats.append(f"{n} R-hat {'n/a' if r is None else f'{r:.3f}'} ESS "
                                 f"{'n/a' if b is None else f'{b:.0f}'}")
            L.append("    nuisance, not required (left out of the verdict): " + ", ".join(nuis)
                     + (f" ({'; '.join(stats)})" if stats else ""))
        flags = list(e.get("flags", []))
        per: dict[str, list[str]] = {}
        nq = len(e["quantities"])
        for name, q in e["quantities"].items():
            for f in q.get("flags", []):
                if f == NUISANCE:
                    continue
                if f not in flags:  # mode-level flags are listed once
                    per.setdefault(f, []).append(name)
        for f, names in per.items():
            if f == "not converged":
                if len(names) == nq:
                    continue  # every quantity: the mode-level verdict says it
                f = "not converged (no interval; UNC-2 per quantity)"
            flags.append(f"{f}: {', '.join(names)}")
        L.append("    flags: " + ("; ".join(flags) if flags else "none"))
        for note in e.get("notes", []):
            L.append(f"    note: {note}")
    for note in intervals.get("notes", []):
        L.append(f"  WARNING: {note}")
    return L


def likelihood_settings(objective) -> dict:
    """The likelihood as used, for run.json and the table headers (LIKE-3, decision #23; LIKE-2, decision #22): kind,
    continuum ("off", or the Legendre degree with the free constant per I spectrum), and the kernel."""
    st = getattr(objective, "settings", {}) or {}
    kind = getattr(objective, "kind", st.get("kind", "full"))
    cont = st.get("continuum", False)
    kernel = getattr(objective, "kernel_state", None)
    if kernel is None and st.get("kernel"):
        kernel = {"state": "given per spectrum", "kernels": st["kernel"]}
    if kernel is None:
        kernel = {"state": "initial (not fitted)", "corr": st.get("corr"), "disc": st.get("disc")}
    out = {"kind": kind, "continuum": "off" if cont is False else {"degree": int(cont["degree"]), "basis": "normalised"},
           "kernel": kernel}
    return out


def likelihood_header(objective) -> str:
    """One line naming the likelihood, the kernel and the continuum setting, for the head of every table."""
    s = likelihood_settings(objective)
    cont = ("off (no Legendre continuum, no free constant)" if s["continuum"] == "off" else
            f"Legendre degree {s['continuum']['degree']} on I, marginalised (normalised basis)")
    if s["kind"] == "chi2":
        return f"likelihood: plain chi2; kernel: none (diagonal errors); continuum: {cont}"
    k = s["kernel"]
    ktext = k.get("state", "initial (not fitted)") if isinstance(k, dict) else str(k)
    return f"likelihood: full; kernel: {ktext}; continuum: {cont}"


def side_by_side_text(profiles: dict, families: dict, header: str | None) -> list[str]:
    """UNC-8: per family and quantity the profile intervals, the envelope, the best mode's 68 % interval, the two
    ratios (decisions #29, #30) and the entry's flags ("constraint not met": a closed end; "polish stalled": a grid
    value left out, the side open), then the two sentences per view."""
    from .profiles import sentences

    def iv(d):
        if d is None or d.get("lo") is None:
            return "-"
        lo = ("<" if d.get("open_lo") else "") + f"{d['lo']:.4g}"
        hi = f"{d['hi']:.4g}" + (">" if d.get("open_hi") else "")
        return f"{lo}..{hi}"

    h = families["h"]
    L = []
    if header:
        L.append(header)
    L.append(f"families side by side with the per-mode error bars (UNC-8): profile intervals (delta < 0.5, < 2 nats of "
             f"the family's best; below the family level, {h:.2f} nats), envelope below the level, the best mode's MCMC "
             f"68 %, R1 = envelope / 68 % width (sqrt(2 h) for one Gaussian basin, not flagged), R2 = R1 / sqrt(2 h) "
             f"(1 for one Gaussian basin; flag '!' above 1.5: a ridge or a second minimum); '<' or '>' = the walk ended "
             f"inside the interval or at a grid value left out (flag 'polish stalled': a polish missed its value twice "
             f"with no limit in sight); "
             f"an end of the range (0 or 90 degrees, a Teff limit, a value the model cannot pass: flag 'constraint not "
             f"met') is closed; flag 'B quantile is a step function: provisional' on every B10/B50/B90 row: "
             f"the quantile jumps between patches, so these profile points can lie above the true profile and the "
             f"intervals be too narrow")
    L.append(f"{'family':>6}  {'quantity':<16}  {'d<0.5':>18}  {'d<2':>18}  {'below level':>20}  {'envelope':>18}  "
             f"{'mode 68 %':>18}  {'R1':>6}  {'R2':>6}  flags")
    texts = []
    for fid, p in profiles.items():
        for name, e in p["quantities"].items():
            m = e.get("mode_68")
            mtxt = "-" if not m else f"{m['p16']:.4g}..{m['p84']:.4g}"
            r1 = "-" if e.get("R1") is None else f"{e['R1']:.2f}"
            r2 = "-" if e.get("R2") is None else f"{e['R2']:.2f}" + ("!" if e["R2"] > 1.5 else "")
            L.append(f"{fid:>6}  {name:<16}  {iv(e['intervals']['delta_0.5']):>18}  {iv(e['intervals']['delta_2']):>18}  "
                     f"{iv(e['intervals']['family_level']):>20}  {iv(e['envelope']):>18}  {mtxt:>18}  {r1:>6}  {r2:>6}  "
                     f"{'; '.join(e.get('flags') or []) or '-'}")
            texts.append(sentences(name, int(fid), p["best_mode"], e, h, p.get("converged")))
    L.append("the same in words, per view:")
    for s1, s2 in texts:
        L.append(f"  {s1}")
        L.append(f"  {s2}")
    return L


def noise_flag(stats: dict) -> dict:
    """LIKE-10: {"noise_model_flag": "noise model fails the whiteness check", "noise_model_failing_spectra": [labels]}
    when the whiteness check failed for any spectrum, else {} (no check run, or all spectra white). write_fit puts it at
    the top level of every output that holds -ln L differences or intervals."""
    from .noise import FAIL_FLAG

    bad = sorted(k for k, v in (stats.get("whiteness") or {}).items() if not v["pass"])
    return {"noise_model_flag": FAIL_FLAG, "noise_model_failing_spectra": bad} if bad else {}


def _flag_line(flag: dict) -> list[str]:
    """The flag line on each table of report.txt whose numbers are -ln L differences or intervals (LIKE-10)."""
    if not flag:
        return []
    return [f"[{flag['noise_model_flag']}: {', '.join(flag['noise_model_failing_spectra'])}]"]


def best_value(modes: list[dict]) -> float:
    """The best mode's -ln L (the reference of every delta), from modes.json entries."""
    return min((m["neg_log_l_mode"] for m in modes), default=0.0)


def report_text(modes: list[dict], stats: dict, wall: float, workers: int, seed: int, kind: str,
                intervals: dict | None = None, mcmc_wall: float | None = None,
                systematics: dict | None = None, header: str | None = None, families: dict | None = None,
                profiles: dict | None = None) -> str:
    L = []
    flag = noise_flag(stats)
    # no wall times here (they are in run.json and the log): two runs with the same seed and workers give the same text
    L.append(f"zeemap fit: objective {kind}, seed {seed}, {stats['n_starts']} starts, {stats['rounds']} restart "
             f"round(s), {stats['nevals']} evaluations on {workers} worker(s)")
    L.append("")
    if header:
        L.append(header)
    L += _flag_line(flag)
    L.append(f"mode table: every mode within {stats['band']:g} nats of the best (delta -lnL known to within the value "
             f"tolerance {stats['value_tolerance']:.2f} nats)")
    if intervals is not None:
        L.append("(best point per mode: the search's end point or a polished hand-back point (UNC-3), evaluated on the "
                 "MCMC's patches; posterior intervals in the next table)")
    L.append("(delta -lnL: the mode's value, the better of its end point and equator-mirror image, minus the best "
             "mode's; -lnL vector: at the written (canonical) vector, which differs from the mode's value by the "
             "mirror asymmetry, diagnostics.json)")
    L.append(f"{'mode':>4}  {'family':>6}  {'delta -lnL':>10}  {'starts':>6}  {'incl':>6}  {'|dipole| MG':>11}  "
             f"{'|c|':>6}  {'B50 ph0 MG':>10}  {'teff K':>8}  {'-lnL vector':>12}  limit")
    for m in modes:
        d = m["derived"]
        teff = "-" if d.get("teff_K") is None else f"{d['teff_K']:.0f}"
        fam = "-" if m.get("family") is None else str(m["family"])
        L.append(f"{m['id']:>4}  {fam:>6}  {m['delta']:>10.2f}  {m['n_starts']:>6}  {d['incl_deg']:>6.2f}  "
                 f"{d['dipole_MG']:>11.3f}  "
                 f"{d['c_abs']:>6.3f}  {d['B50_MG'][0]:>10.3f}  {teff:>8}  {m['neg_log_l']:>12.3f}  "
                 f"{'TOUCHED' if m['touched_limit'] else ''}")
    if families is not None:
        L.append("")
        L.append(f"family table (SEARCH-9): end points joined by paths below the family level, best + "
                 f"{families['h']:.2f} nats (one level for all families); a mode with family '-' lies above it")
        L += _flag_line(flag)
        L.append(f"{'family':>6}  {'delta -lnL':>10}  {'modes':<20}  {'end points':>10}  band component")
        for fm in families["families"]:
            bc = sorted({families["mode_band_component"].get(mi) for mi in fm["modes"]} - {None})
            L.append(f"{fm['id']:>6}  {fm['neg_log_l'] - best_value(modes):>10.2f}  "
                     f"{','.join(str(v) for v in fm['modes']):<20}  {len(fm['nodes']):>10}  "
                     f"{','.join(str(v) for v in bc) or '-'}")
        L.append(f"  components at the band level ({families['band_level'] - best_value(modes):.0f} nats): "
                 f"{len(families['band_components'])}" + ("; pair cap reached" if families.get("capped") else ""))
    L.append("")
    L.append("search statistics (SEARCH-4):")
    L.append(f"  starts {stats['n_starts']}, distinct modes {stats['n_modes']}, modes in the band {stats['n_band_modes']}, "
             f"hits on the best mode {stats['hits_best']}, band modes reached by at most one start "
             f"{stats['n_band_once']} ({100 * stats['frac_band_once']:.0f} %)")
    L.append(f"  optimiser scatter {stats['optimiser_scatter']:.3f} nats, value tolerance {stats['value_tolerance']:.2f} "
             f"nats, rounds {stats['rounds']} (gains {', '.join(f'{g:.2f}' for g in stats['round_gains']) or '-'}), "
             f"end points {stats['n_endpoints']} ({stats['n_invalid_endpoints']} invalid)")
    if stats.get("warning"):
        L.append("  WARNING: " + stats.get("warning_text", "the search cannot bound the chance that a better mode was "
                                           "missed"))
    if stats.get("floor_chi2"):
        L.append("floor chi2 (DIAG-1; the best non-negative mixture of library records, plain chi2, I and V rows) "
                 "against the best mode's plain chi2 (its I with the continuum fitted inside the likelihood, as in "
                 "spectra/; under a fitted kernel that continuum does not minimise the plain chi2): " + ", ".join(
                     f"phase {v['phase']:g}: {v['chi2_floor']:.4g} / {v['chi2_mode']:.4g}"
                     for v in stats["floor_chi2"].values()))
    if stats.get("window_residuals"):
        libs = sorted({r["library"] for r in stats["window_residuals"]})
        nfam = len({(r.get("family"), r.get("mode")) for r in stats["window_residuals"]})
        L.append(f"per-window residuals (DIAG-3): {len(stats['window_residuals'])} rows for {', '.join(libs)} at the "
                 f"best mode of each family ({nfam}), whitened chi2 under the run's continuum and at degree 0: "
                 "diagnostics.json")
    if stats.get("whiteness"):
        from .noise import FAIL_FLAG

        wh = stats["whiteness"]
        bad = [k for k, v in wh.items() if not v["pass"]]
        L.append(f"noise model (LIKE-10): whitened residuals at the best mode white in {len(wh) - len(bad)} of {len(wh)} "
                 "spectra (max |ACF| at lags 1-10 against 3 / sqrt(n) + 0.05: " +
                 ", ".join(f"{k} {v['max_abs_acf']:.3f}/{v['threshold']:.3f}" for k, v in wh.items()) + ")")
        if bad:
            L.append(f"  {FAIL_FLAG.upper()}: {', '.join(bad)}. Every -ln L difference and interval of the run carries "
                     "this flag (each table here and each output file), and no interval is calibrated (UNC-4)")
    if stats.get("kernel_at_bound"):
        from .noise import A_MAX

        L.append(f"noise kernel (LIKE-2): the frozen kernel's long (discrepancy) part at its cap a = {A_MAX:g} absorbs "
                 "much of these spectra, which then count little in the fit; their parameters at a bound of the range "
                 "(values in run.json): " + "; ".join(
                     f"{lab} " + ", ".join(f"{n} {side}" for n, side in b.items())
                     for lab, b in stats["kernel_at_bound"].items()))
    L.append("")
    if intervals is None:
        L.append("error bars: not computed (MCMC is off; add --mcmc). The parameters above are search end points, not "
                 "estimates with uncertainties.")
    else:
        if header:
            L.append(header)
        L += _flag_line(flag)
        L += errorbar_text(intervals)
    if stats.get("profiles_error"):  # the profile stage failed; everything else was written (cli._fit)
        L.append("")
        L.append(f"profiles (UNC-5): failed, none written: {stats['profiles_error']}")
    if profiles:
        L.append("")
        sbs = side_by_side_text(profiles, families, header)
        L += sbs[:1] + _flag_line(flag) + sbs[1:] if header else _flag_line(flag) + sbs
    L.append("")
    if header and systematics is not None and systematics.get("swaps"):
        L.append(header)
    if systematics is not None and systematics.get("swaps"):
        L += _flag_line(flag)
    L += systematics_text(systematics)
    L.append("")
    L.append("Modes are ranked by -ln L only (REP-2).")
    return "\n".join(L) + "\n"


#: the files of `fit` at the top of its output directory, and its directories of per-mode files (`mode<k>_*`)
FIT_FILES = ("run.json", "modes.json", "families.json", "profiles.json", "diagnostics.json", "intervals.json",
             "systematics.json", "report.txt")
FIT_MODE_DIRS = ("spectra", "histograms", "samples")


def clear_fit_outputs(out: Path) -> list[str]:
    """Remove the files an earlier `fit` wrote into `out` (FIT_FILES; `mode<k>_*` in FIT_MODE_DIRS, and samples/ when
    it is left empty); returns the names removed. Files of other names are kept."""
    import re

    gone = []
    for f in FIT_FILES:
        if (out / f).is_file():
            (out / f).unlink()
            gone.append(f)
    for d in FIT_MODE_DIRS:
        if not (out / d).is_dir():
            continue
        for f in sorted((out / d).iterdir()):
            if f.is_file() and re.fullmatch(r"mode\d+[_.].*", f.name):
                f.unlink()
                gone.append(f"{d}/{f.name}")
    if (out / "samples").is_dir() and not any((out / "samples").iterdir()):
        (out / "samples").rmdir()
    return gone


def write_fit(outdir, problem, cfg: dict, result, *, seed: int, workers: int, t_start: str, t_end: str, wall: float,
              n_invalid_main: int, argv: list[str], uncertainty: dict | None = None,
              search_wall: float | None = None, systematics: dict | None = None, inputs: dict | None = None,
              resumed: dict | None = None) -> dict:
    """All output files of `fit`. `uncertainty`: {"intervals": intervals.json dict, "results": {mode id: McmcResult},
    "wall": seconds} or None (no MCMC); `systematics`: the systematics.json dict (cli._systematics) or None
    (--no-systematics); `wall` is the whole run, `search_wall` the search alone (default `wall`); `inputs`: the
    input_sha256 dict when already computed; `resumed`: run.json's `resumed` (cli.py) or None. A resumed
    search (resumed["search"]) has no meaningful time per evaluation or count of invalid models in this process:
    both null."""
    swall = wall if search_wall is None else search_wall
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    clear_fit_outputs(out)  # no file of an earlier fit into the same directory survives
    best = result.modes[0].f
    flag = noise_flag(result.stats)  # LIKE-10: on every file with -ln L differences or intervals
    modes = [mode_entry(problem, m, k + 1, best) for k, m in enumerate(result.modes)]
    files = []
    for k, m in enumerate(result.modes):
        x = m.x_canonical if m.x_canonical is not None else m.x
        files += write_spectra(out, problem, x, f"mode{k + 1}")
        files += write_histograms(out, problem, x, f"mode{k + 1}")
    write_json(out / "modes.json", {"band_nats": float(cfg["search"]["band"]), "objective": problem.objective.kind,
                                    "names": problem.names, "modes": modes, **flag})
    fams = getattr(result, "families", None)
    if fams is not None:  # SEARCH-9 (decision #29)
        write_json(out / "families.json", {"names": problem.names, **fams, **flag})
    profs = getattr(result, "profiles", None)
    if profs is not None:  # UNC-5, UNC-8
        write_json(out / "profiles.json", {"level": fams["level"], "h": fams["h"],
                                           "header": likelihood_header(problem.objective),
                                           "families": {str(k): v for k, v in profs.items()}, **flag})
    stats = dict(result.stats)
    stats["wall_time_s"] = swall
    stats["workers"] = workers
    again = bool(resumed and resumed.get("search"))
    stats["ms_per_eval_per_worker"] = None if again else 1e3 * swall * workers / max(1, result.nevals)
    stats["n_invalid_main_process"] = None if again else n_invalid_main
    all_modes = [{"id": k + 1, "neg_log_l": _written_value(m), "neg_log_l_mode": float(m.f),
                  "image": getattr(m, "image", "end point"), "mirror_asymmetry": getattr(m, "mirror_asymmetry", None),
                  "delta": float(m.f - best), "n_starts": int(m.hits),
                  "n_endpoints": len(m.members), "in_band": k < len(result.modes),
                  "parameters": problem.grouped(m.x_canonical if m.x_canonical is not None else m.x),
                  "fingerprint": m.fingerprint.to_dict()} for k, m in enumerate(result.all_modes)]
    endpoints = [e.to_dict() for e in result.endpoints]
    for e in endpoints:  # 1-based mode ids, as in modes.json
        e["mode"] = None if e["mode"] is None else e["mode"] + 1
    # wall times only under "_meta" (FORMATS.md section 6), which a tolerance comparison of two runs skips
    volatile = ("wall_time_s", "ms_per_eval_per_worker")
    write_json(out / "diagnostics.json", {"statistics": {k: v for k, v in stats.items() if k not in volatile},
                                          "all_modes": all_modes, "names": problem.names, "endpoints": endpoints,
                                          **flag,
                                          "_meta": {k: stats[k] for k in volatile}})
    run = {"command": argv, "config": dump(cfg), "config_path": cfg.get("_meta", {}).get("path"),
           "versions": versions(), "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "seed": int(seed),
           "workers": int(workers), "input_sha256": input_files(cfg, problem) if inputs is None else inputs,
           "start_time": t_start, "end_time": t_end, "wall_time_s": wall,
           "start_distribution": result.start_distribution,
           **parameter_space(problem), "data_notes": problem.dataset.notes, "mcmc": uncertainty is not None,
           "likelihood": likelihood_settings(problem.objective)}
    if resumed is not None:
        run["resumed"] = resumed
    intervals = None
    if uncertainty is not None:
        intervals = uncertainty["intervals"]
        if flag:  # and no interval is calibrated (UNC-4)
            intervals.update(flag)
            intervals["calibration"] = "uncalibrated"
        write_json(out / "intervals.json", portable(intervals, cfg.get("_meta", {}).get("dir")))
        sd = out / "samples"
        sd.mkdir(parents=True, exist_ok=True)
        for kid, r in sorted(uncertainty["results"].items()):
            r.save_npz(sd / f"mode{kid}.npz")
            files.append(f"samples/mode{kid}.npz")
        run["mcmc_wall_time_s"] = uncertainty.get("wall")
    run["systematics"] = systematics is not None
    if systematics is not None:
        sysd = {k: v for k, v in systematics.items() if k != "wall_time_s"}
        sysd.update(flag)
        sysd["_meta"] = {"wall_time_s": systematics.get("wall_time_s")}
        write_json(out / "systematics.json", portable(sysd, cfg.get("_meta", {}).get("dir")))
        run["systematics_wall_time_s"] = systematics.get("wall_time_s")
    write_json(out / "run.json", portable(run, cfg.get("_meta", {}).get("dir")))  # ENG-4: no absolute paths
    text = report_text(modes, stats, swall, workers, seed, problem.objective.kind, intervals,
                       None if uncertainty is None else uncertainty.get("wall"), systematics,
                       header=likelihood_header(problem.objective), families=fams, profiles=profs)
    atomic.write_text(out / "report.txt", text)
    return {"modes": modes, "stats": stats, "report": text, "files": files}
