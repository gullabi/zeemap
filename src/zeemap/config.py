"""Configuration reader (SPEC.md CFG-1, CFG-2, CFG-3; FORMATS.md section 3).

One TOML file holds the whole configuration. `load_config` merges it over `defaults.toml`, stops on an unknown key
(the message names its dotted path, e.g. "unknown key model.fhwm" or "unknown key data.phase[0].fiel"), stops on a
missing required key, checks types where that is cheap, and resolves paths:
- `library.path`, `data.path`: relative to `$ZEEMAP_DATA` (an error if the variable is unset and the path is relative);
- the files of `[[data.phase]]`: relative to `data.path`;
- every other path (`uncertainty.calibration`): relative to the configuration file's directory.
Absolute paths are kept. The result carries `cfg["_meta"] = {"path": ..., "dir": ...}` (the configuration file and its
directory); `dump` drops it.

Conventions of `defaults.toml` (see its header):
- the text value "REQUIRED" marks a key without a default;
- an array of tables holds one template entry, applied to every entry the configuration writes;
- `[parameters."*"]` is the template of every `[parameters]` entry; the allowed names are `PARAMETER_NAMES`.
Merging: tables merge key by key; any other value (lists included) replaces the default. `objective.corr`,
`objective.disc` and `objective.continuum` may be `false`.

Deviation from FORMATS.md: the number of starts is `search.n_starts`, because TOML cannot hold both the integer
`search.starts` and the table `[search.starts]`.
"""

from __future__ import annotations

import copy
import functools
import math
import os
import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Callable

REQUIRED = "REQUIRED"
WILDCARD = "*"
DATA_ENV = "ZEEMAP_DATA"

#: keys that may be `false` instead of a table
FALSE_OK = {"objective.corr", "objective.disc", "objective.continuum"}
#: vector parameters (value = 3 numbers; lo, hi bound the length) and scalar parameters (FORMATS.md section 3)
VECTOR_PARAMETERS = ("md", "mo", "aq", "c")
#: the l >= 2 Gauss coefficients of the field model "harm4", in parameter order (g20, g21, h21, g22, h22, g30, ..., h44)
HARM_NAMES = tuple(f"{k}{l}{m}" for l in range(2, 5) for m in range(l + 1) for k in "gh" if not (k == "h" and m == 0))
SCALAR_PARAMETERS = ("bq", "bo", "teff", "shift", "u") + HARM_NAMES
PARAMETER_NAMES = VECTOR_PARAMETERS + SCALAR_PARAMETERS
#: minimum number of entries of the arrays of tables
MIN_ENTRIES = {"data.phase": 1}
#: names of the profiled quantities (FORMATS.md section 7): one value per model, and one per phase ("B50_MG[k]")
PROFILE_SCALARS = ("incl_deg", "beta_deg", "dipole_MG", "c_abs", "c_along_axis", "c_across_axis", "teff_K")
PROFILE_PER_PHASE = ("B10_MG", "B50_MG", "B90_MG")
_PERCENT = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%")


class ConfigError(ValueError):
    """A configuration that cannot be used; the message names the key."""


# ----------------------------------------------------------------------------------------------------------- defaults
@functools.lru_cache(maxsize=1)
def _defaults_text() -> str:
    return resources.files("zeemap").joinpath("defaults.toml").read_text(encoding="utf-8")


def defaults() -> dict:
    """The parsed `defaults.toml`, as written (with the "REQUIRED" markers and the templates)."""
    return tomllib.loads(_defaults_text())


# ------------------------------------------------------------------------------------------------------------- checks
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(v, key):
    if not _is_num(v):
        raise ConfigError(f"{key} must be a number, got {v!r}")
    return float(v)


def _pos(v, key):
    v = _num(v, key)
    if v <= 0:
        raise ConfigError(f"{key} must be > 0, got {v!r}")
    return v


def _nonneg(v, key):
    v = _num(v, key)
    if v < 0:
        raise ConfigError(f"{key} must be >= 0, got {v!r}")
    return v


def _int(v, key):
    if not isinstance(v, int) or isinstance(v, bool):
        raise ConfigError(f"{key} must be an integer, got {v!r}")
    return v


def _pos_int(v, key):
    v = _int(v, key)
    if v <= 0:
        raise ConfigError(f"{key} must be a positive integer, got {v!r}")
    return v


def _nonneg_int(v, key):
    v = _int(v, key)
    if v < 0:
        raise ConfigError(f"{key} must be an integer >= 0, got {v!r}")
    return v


def _bool(v, key):
    if not isinstance(v, bool):
        raise ConfigError(f"{key} must be true or false, got {v!r}")
    return v


def _str(v, key):
    if not isinstance(v, str):
        raise ConfigError(f"{key} must be a text, got {v!r}")
    return v


def _nonempty_str(v, key):
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{key} must be a non-empty text, got {v!r}")
    return v


def _choice(*options):
    def check(v, key):
        if isinstance(v, bool) or v not in options:
            raise ConfigError(f"{key} must be one of {', '.join(repr(o) for o in options)}; got {v!r}")
        return v
    return check


def _range(v, key, what="[lo, hi]"):
    if not isinstance(v, list) or len(v) != 2 or not all(_is_num(x) for x in v):
        raise ConfigError(f"{key} must be {what} (2 numbers), got {v!r}")
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        raise ConfigError(f"{key}: lo > hi in {v!r}")
    return [lo, hi]


def _list_of(item: Callable, min_len: int = 0):
    def check(v, key):
        if not isinstance(v, list) or len(v) < min_len:
            need = f"a list with at least {min_len} entries" if min_len else "a list"
            raise ConfigError(f"{key} must be {need}, got {v!r}")
        return [item(x, f"{key}[{i}]") for i, x in enumerate(v)]
    return check


def _window(v, key):
    if not isinstance(v, list) or len(v) not in (2, 3) or not all(_is_num(x) for x in v):
        raise ConfigError(f"{key} must be [lo, hi] or [lo, hi, weight] in A, got {v!r}")
    out = [float(x) for x in v]
    if out[0] >= out[1]:
        raise ConfigError(f"{key}: lo >= hi in {v!r}")
    if len(out) == 3 and out[2] < 0:
        raise ConfigError(f"{key}: negative weight in {v!r}")
    return out


def _exclude(v, key):
    ok = isinstance(v, list) and len(v) in (2, 3) and all(_is_num(x) for x in v[:2])
    if ok and len(v) == 3 and v[2] not in ("I", "VI"):
        ok = False
    if not ok:
        raise ConfigError(f'{key} must be [lo, hi] or [lo, hi, "I"/"VI"] in A, got {v!r}')
    out = [float(v[0]), float(v[1]), *v[2:]]
    if out[0] >= out[1]:
        raise ConfigError(f"{key}: lo >= hi in {v!r}")
    return out


def _int_from(lo):
    def check(v, key):
        v = _int(v, key)
        if v < lo:
            raise ConfigError(f"{key} must be an integer >= {lo}, got {v!r}")
        return v
    return check


def percent(v) -> float | None:
    """p of a profile step written "<p>%" (a plain positive decimal number, then "%"), else None: "two%", "-5%",
    "0%", "nan%" and a number without "%" give None."""
    if not isinstance(v, str) or not _PERCENT.fullmatch(v):
        return None
    p = float(v[:-1])
    return p if math.isfinite(p) and p > 0 else None


def spectrum_labels(phases: list[dict]) -> list[str]:
    """The labels of the spectra the [[data.phase]] blocks `phases` give (types.Spectrum.label: the phase as %.3f, then
    "_I" or "_VI", for the kinds the block uses). A block without a numeric phase gives none (from_dict refuses it)."""
    out = []
    for p in phases:
        if not isinstance(p, dict) or not _is_num(p.get("phase")):
            continue
        for kind in ("I", "VI"):
            if p.get(f"use_{kind}", True):
                out.append(f"{float(p['phase']):.3f}_{kind}")
    return out


def _all_or(check):
    def f(v, key):
        return v if v == "all" else check(v, key)
    return f



#: checks by generic key path ([] for an entry of an array of tables, "*" for a parameter name). Keys not listed are
#: checked against the type of their default (bool, integer, number, text, list).
CHECKS: dict[str, Callable] = {
    "library.path": _nonempty_str,
    "library.teffs": _all_or(_list_of(_pos, 1)),
    "library.outside": _choice("clamp", "reject"),
    "data.path": _nonempty_str,
    "data.windows": _list_of(_window, 1),
    "data.exclude": _list_of(_exclude),
    "data.vi_scale": _pos,
    "data.grid.start": _pos,
    "data.grid.step": _pos,
    "data.grid.npix": _pos_int,
    "data.errors.floor_I": _nonneg,
    "data.errors.floor_VI": _nonneg,
    "data.phase[].phase": _num,
    "data.phase[].file": _nonempty_str,
    "data.phase[].ierr": _nonempty_str,
    "data.phase[].vierr": _str,
    "data.weight[].stokes": _choice("I", "VI"),
    "data.weight[].lo": _num,
    "data.weight[].hi": _num,
    "data.weight[].phases": _all_or(_list_of(_num, 1)),
    "data.weight[].w": _nonneg,
    "model.field": _nonempty_str,
    "model.interp_b": _choice("linear", "cubic", "pchip"),
    "model.interp_teff": _choice("planck", "log"),
    "model.shift_unit": _choice("kms", "A"),
    "model.shift_nodes": _int_from(2),
    "model.ld_mode": _choice(0, 1, 3),
    "model.fwhm": _pos,
    "model.phase_offset": _num,
    "model.surface.scheme": _choice("fibonacci"),
    "model.surface.n": _pos_int,
    "objective.kind": _choice("full", "chi2"),
    "search.fingerprint.angle": _choice("auto", "incl", "beta"),
    "objective.corr.rho": _pos,
    "objective.corr.jitter": _nonneg,
    "objective.disc.a_I": _nonneg,
    "objective.disc.rho_I": _pos,
    "objective.disc.a_VI": _nonneg,
    "objective.disc.rho_VI": _pos,
    "objective.continuum.degree": _nonneg_int,
    "search.n_starts": _pos_int,
    "search.top": _pos_int,
    "search.restarts": _nonneg_int,
    "search.rounds_max": _pos_int,
    "search.fd_step": _pos,
    "search.lbfgs_maxiter": _pos_int,
    "search.lm_maxiter": _pos_int,
    "search.lm_stop_nat": _nonneg,
    "search.lm_max_rejections": _nonneg_int,
    "search.tol_polishes": _pos_int,
    "search.family.min_level": _pos,
    "search.family.max_pairs": _pos_int,
    "search.family.mid_iter": _pos_int,
    "search.family.interior": _pos_int,
    "search.starts.md": _range,
    "search.starts.u": _range,
    "search.starts.bq": _nonneg,
    "search.starts.mo": _nonneg,
    "search.starts.c": _nonneg,
    "search.starts.outside_max": _nonneg,
    "search.starts.rules": _list_of(lambda v, k: v),
    "uncertainty.patches": _pos_int,
    "uncertainty.ensembles": _pos_int,
    "uncertainty.max_evals": _nonneg_int,
    "uncertainty.calibration": _str,
    "uncertainty.profile_max_steps": _pos_int,
    "uncertainty.profile_families": _nonneg_int,
    # uncertainty.profile_quantities and profile_steps: checked by _check_profiles (names need the phase count)
    "systematics.library_alternatives": _list_of(_nonempty_str),
    "systematics.swap[].name": _nonempty_str,
    "objective.kernel[].spectrum": _nonempty_str,
    "objective.kernel[].rho_short": _pos,
    "objective.kernel[].a": _nonneg,
    "objective.kernel[].rho_long": _pos,
    "objective.kernel[].j": _pos,
    # systematics.swap[].set: checked by _check_swap_set (keys must be configuration keys)
}


def _by_default_type(default, v, key):
    if isinstance(default, bool):
        return _bool(v, key)
    if isinstance(default, int):
        return _int(v, key)
    if isinstance(default, float):
        return _num(v, key)
    if isinstance(default, str):
        return _str(v, key)
    if isinstance(default, list):
        if not isinstance(v, list):
            raise ConfigError(f"{key} must be a list, got {v!r}")
        return v
    raise AssertionError(f"defaults.toml: no check for {key}")  # pragma: no cover


# ---------------------------------------------------------------------------------------------------------- merging
def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _is_template_array(d) -> bool:
    return isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict)


def _merge_table(d: dict, u: Any, path: str, gpath: str) -> dict:
    """Merge the user table u over the defaults table d. gpath is the generic path (for CHECKS)."""
    if not isinstance(u, dict):
        raise ConfigError(f"{path or 'the configuration'} must be a table, got {u!r}")
    out = {}
    for k in u:
        if k not in d and WILDCARD not in d:
            raise ConfigError(f"unknown key {_join(path, k)}")
        if k not in d and gpath == "parameters" and k not in PARAMETER_NAMES:
            raise ConfigError(
                f"unknown key {_join(path, k)} (parameters are: {', '.join(PARAMETER_NAMES)})"
            )
    for k, dv in d.items():
        if k == WILDCARD:
            continue
        p, gp = _join(path, k), _join(gpath, k)
        out[k] = _merge_value(dv, u[k], p, gp) if k in u else _fill(dv)
    if WILDCARD in d:
        for k, uv in u.items():
            if k not in d:
                out[k] = _merge_table(d[WILDCARD], uv, _join(path, k), _join(gpath, WILDCARD))
    return out


def _merge_value(d, u, path: str, gpath: str):
    if _is_template_array(d):
        if not isinstance(u, list) or not all(isinstance(e, dict) for e in u):
            raise ConfigError(f"{path} must be an array of tables ([[{path}]] blocks), got {u!r}")
        return [_merge_table(d[0], e, f"{path}[{i}]", f"{gpath}[]") for i, e in enumerate(u)]
    if gpath == "uncertainty.profile_steps":  # a free table: quantity name -> grid step, checked by _check_profiles
        if not isinstance(u, dict):
            raise ConfigError(f"{path} must be a table of quantity name -> positive step or \"<p>%\", got {u!r}")
        return dict(u)
    if gpath == "uncertainty.profile_quantities":
        if not (u == "default" or (isinstance(u, list) and all(isinstance(x, str) for x in u))):
            raise ConfigError(f"{path} must be \"default\" or a list of quantity names, got {u!r}")
        return u if u == "default" else list(u)
    if gpath == "search.fingerprint.extra":  # a free table: parameter name -> threshold (search.py)
        if not isinstance(u, dict) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in u.values()):
            raise ConfigError(f"{path} must be a table of parameter name -> positive threshold, got {u!r}")
        return {k: float(x) for k, x in u.items()}
    if isinstance(d, dict):
        if u is False and gpath in FALSE_OK:
            return False
        if not isinstance(u, dict):
            alt = " or false" if gpath in FALSE_OK else ""
            raise ConfigError(f"{path} must be a table{alt}, got {u!r}")
        return _merge_table(d, u, path, gpath)
    if gpath == "systematics.swap[].set":  # a free table of configuration keys, checked by _check_swap_set
        return copy.deepcopy(u)
    if isinstance(u, dict):
        raise ConfigError(f"{path} must not be a table (unknown keys {', '.join(_join(path, k) for k in u)})")
    return copy.deepcopy(u)


def _fill(d):
    """The default value for a key the configuration does not set (template arrays -> no entries)."""
    if _is_template_array(d):
        return []
    if isinstance(d, dict):
        return {k: _fill(v) for k, v in d.items() if k != WILDCARD}
    return copy.deepcopy(d)


# ------------------------------------------------------------------------------------------------------- validation
def _check_tree(cfg: dict, dflt: dict, path: str = "", gpath: str = "") -> None:
    """Required keys, then types (coercing integers to floats where the key is a number), in place."""
    for k in list(cfg):
        v = cfg[k]
        p, gp = _join(path, k), _join(gpath, k)
        dv = dflt.get(k, dflt.get(WILDCARD))
        if gpath == "parameters":
            _check_parameter(v, p, k)
            continue
        if _is_template_array(dv):
            n_min = MIN_ENTRIES.get(gp, 0)
            if len(v) < n_min:
                raise ConfigError(f"missing required key {p}: at least {n_min} [[{p}]] block(s) needed")
            for i, e in enumerate(v):
                _check_tree(e, dv[0], f"{p}[{i}]", f"{gp}[]")
            continue
        if isinstance(v, dict) and gp not in ("systematics.swap[].set", "search.fingerprint.extra",
                                              "uncertainty.profile_steps"):
            _check_tree(v, dv if isinstance(dv, dict) else {}, p, gp)
            continue
        if v is False and gp in FALSE_OK:
            continue
        if isinstance(v, str) and v == REQUIRED:
            raise ConfigError(f"missing required key {p}")
        if gp == "systematics.swap[].set":
            _check_swap_set(v, p)
            continue
        if gp in ("search.fingerprint.extra", "uncertainty.profile_steps", "uncertainty.profile_quantities"):
            continue
        check = CHECKS.get(gp)
        cfg[k] = check(v, p) if check else _by_default_type(dv, v, p)


def _check_parameter(e: dict, path: str, name: str) -> None:
    for k in ("value", "step", "lo", "hi"):
        if isinstance(e[k], str) and e[k] == REQUIRED:
            raise ConfigError(f"missing required key {path}.{k}")
    e["free"] = _bool(e["free"], f"{path}.free")
    e["step"] = _pos(e["step"], f"{path}.step")
    lo, hi = _num(e["lo"], f"{path}.lo"), _num(e["hi"], f"{path}.hi")
    if lo > hi:
        raise ConfigError(f"{path}: lo > hi ({lo} > {hi})")
    e["lo"], e["hi"] = lo, hi
    if name in VECTOR_PARAMETERS:
        v = e["value"]
        if not isinstance(v, list) or len(v) != 3 or not all(_is_num(x) for x in v):
            raise ConfigError(f"{path}.value must be a vector of 3 numbers, got {v!r}")
        v = [float(x) for x in v]
        if lo < 0:
            raise ConfigError(f"{path}.lo bounds a length and must be >= 0, got {lo}")
        length = math.sqrt(sum(x * x for x in v))
        if not lo <= length <= hi:
            raise ConfigError(f"{path}: the length of value, {length:g}, is outside [lo, hi] = [{lo:g}, {hi:g}]")
    else:
        v = _num(e["value"], f"{path}.value")
        if not lo <= v <= hi:
            raise ConfigError(f"{path}: value {v:g} is outside [lo, hi] = [{lo:g}, {hi:g}]")
    e["value"] = v


def _check_swap_set(s, path: str) -> None:
    """The keys a swap sets must be configuration keys (values are checked when the swap is applied)."""
    if not isinstance(s, dict) or not s:
        raise ConfigError(f"{path} must be a non-empty table of configuration keys, got {s!r}")
    tree = defaults()

    def walk(u: dict, d: dict, p: str, gp: str) -> None:
        for k, v in u.items():
            pk = _join(p, k)
            if gp == "parameters":
                if k not in PARAMETER_NAMES:
                    raise ConfigError(f"unknown key {pk}")
                dk = d[WILDCARD]
            elif k in d:
                dk = d[k]
            else:
                raise ConfigError(f"unknown key {pk}")
            if isinstance(v, dict) and isinstance(dk, dict):
                walk(v, dk, pk, _join(gp, k))

    walk(s, tree, path, "")


def _profile_name(n, n_phases: int, key: str, base_ok: bool) -> None:
    """A profiled quantity's name (FORMATS.md section 7): a scalar name, or a per-phase name with a phase index
    below `n_phases` ("B50_MG[0]"); `base_ok`: a per-phase name may also stand alone (a step for every phase)."""
    m = re.fullmatch(r"(\w+?)(?:\[(\d+)\])?", n) if isinstance(n, str) else None
    base, k = (m.group(1), m.group(2)) if m else (None, None)
    ok = (base in PROFILE_SCALARS and k is None) or (
        base in PROFILE_PER_PHASE and (int(k) < n_phases if k is not None else base_ok))
    if not ok:
        names = ", ".join(PROFILE_SCALARS + tuple(f"{b}[k]" for b in PROFILE_PER_PHASE))
        raise ConfigError(f"{key}: {n!r} is not a profiled quantity (names: {names}; k = 0 ... {n_phases - 1}, the "
                          "phase index)")


def _check_profiles(cfg: dict) -> None:
    """uncertainty.profile_quantities and profile_steps: known names, steps > 0 or "<p>%" with p > 0.
    Checked here, at load, so that a bad setting cannot stop a run after its search and MCMC."""
    uc, n_phases = cfg["uncertainty"], len(cfg["data"]["phase"])
    if uc["profile_quantities"] != "default":
        for i, n in enumerate(uc["profile_quantities"]):
            _profile_name(n, n_phases, f"uncertainty.profile_quantities[{i}]", base_ok=False)
    for n, v in uc["profile_steps"].items():
        key = f"uncertainty.profile_steps.{n}"
        _profile_name(n, n_phases, key, base_ok=True)
        if isinstance(v, str):
            if percent(v) is None:
                raise ConfigError(f'{key} must be a positive number or "<p>%" with p a positive number, got {v!r}')
        else:
            _pos(v, key)


def _check_kernel_labels(cfg: dict) -> None:
    """Each [[objective.kernel]] block must name a spectrum of the configuration (an unmatched label would be ignored,
    and that spectrum would silently take the initial kernel)."""
    labels = spectrum_labels(cfg["data"]["phase"])
    for i, e in enumerate(cfg["objective"]["kernel"]):
        if e["spectrum"] not in labels:
            raise ConfigError(f"objective.kernel[{i}].spectrum = {e['spectrum']!r} matches no spectrum of the "
                              f"configuration (spectra: {', '.join(labels) or 'none'})")


# ---------------------------------------------------------------------------------------------------------- paths
def _data_path(p: str, key: str) -> str:
    if os.path.isabs(p):
        return os.path.normpath(p)
    root = os.environ.get(DATA_ENV)
    if not root:
        raise ConfigError(f"{key} = {p!r} is relative to ${DATA_ENV}, but the environment variable {DATA_ENV} is not set")
    return os.path.normpath(os.path.join(os.path.abspath(root), p))


def _rel(p: str, base: str) -> str:
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))


def portable_path(p: str, cfg_dir: str | None) -> str:
    """An absolute path as it is written into run.json and the other outputs (SPEC ENG-4): `$ZEEMAP_DATA/<rest>` when it
    lies under $ZEEMAP_DATA, else relative to the configuration's directory `cfg_dir` (possibly with `..`). Other
    strings are returned unchanged."""
    if not isinstance(p, str) or not os.path.isabs(p):
        return p
    q = os.path.normpath(p)
    root = os.environ.get(DATA_ENV)
    if root:
        r = os.path.normpath(os.path.abspath(root))
        if q == r or q.startswith(r + os.sep):
            return "$" + DATA_ENV + ("/" + os.path.relpath(q, r) if q != r else "")
    if cfg_dir:
        return os.path.relpath(q, cfg_dir)
    return os.path.relpath(q)


def portable(obj, cfg_dir: str | None):
    """`obj` (JSON-like: dicts, lists, strings) with every absolute path replaced by `portable_path`, also inside TOML
    text and command lines (each white-space separated token, and quoted strings)."""
    if isinstance(obj, dict):
        return {portable(k, cfg_dir): portable(v, cfg_dir) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v, cfg_dir) for v in obj]
    if isinstance(obj, str):
        if os.path.isabs(obj) and "\n" not in obj and " " not in obj:
            return portable_path(obj, cfg_dir)
        if os.sep in obj:
            import re

            return re.sub(r'(?<![\w$.~-])(/[^\s"\',\]\[}{]+)', lambda m: portable_path(m.group(1), cfg_dir), obj)
    return obj


def _resolve_paths(cfg: dict, cfg_dir: str) -> None:
    cfg["library"]["path"] = _data_path(cfg["library"]["path"], "library.path")
    cfg["data"]["path"] = _data_path(cfg["data"]["path"], "data.path")
    for e in cfg["data"]["phase"]:
        for k in ("file", "ierr", "vierr"):
            if e[k]:
                e[k] = _rel(e[k], cfg["data"]["path"])
    if cfg["uncertainty"]["calibration"]:
        cfg["uncertainty"]["calibration"] = _rel(cfg["uncertainty"]["calibration"], cfg_dir)


# ------------------------------------------------------------------------------------------------------- public API
def from_dict(raw: dict, path: str | os.PathLike | None = None) -> dict:
    """Merge, check and resolve a configuration given as a dict; `path` is the file it came from (for relative paths;
    None = the current directory)."""
    if "_meta" in raw:
        raise ConfigError("unknown key _meta (reserved for the reader)")
    dflt = defaults()
    cfg = _merge_table(dflt, raw, "", "")
    _check_tree(cfg, dflt)
    _check_profiles(cfg)
    _check_kernel_labels(cfg)
    for i, w in enumerate(cfg["data"]["weight"]):
        if w["lo"] >= w["hi"]:
            raise ConfigError(f"data.weight[{i}]: lo >= hi ({w['lo']} >= {w['hi']})")
    if path is None:
        cfg_dir, cfg_path = os.path.abspath("."), ""
    else:
        cfg_path = os.path.abspath(os.fspath(path))
        cfg_dir = os.path.dirname(cfg_path)
    _resolve_paths(cfg, cfg_dir)
    cfg["_meta"] = {"path": cfg_path, "dir": cfg_dir}
    return cfg


def load_config(path: str | os.PathLike) -> dict:
    """Read the TOML configuration at `path`; see the module docstring."""
    p = Path(path)
    try:
        with p.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {p}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: not valid TOML: {exc}") from None
    return from_dict(raw, p)


def dump(cfg: dict) -> str:
    """TOML text of a loaded configuration, without `_meta` (for run.json and V1k)."""
    import tomli_w

    return tomli_w.dumps({k: v for k, v in cfg.items() if k != "_meta"})
