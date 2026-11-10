"""The fitting problem: configuration -> library, surface, field, data, forward model, objective, and the search's
`CostProblem` protocol (search.py) on top of them (DESIGN.md section 1; DEV_INTERFACES.md "Problem and search";
SPEC.md PAR-1 to PAR-4).

Parameter vector (`names`, `free`, `lo`, `hi`, `step`, `value`, all of length `dim`):

- **Vector parameters (PAR-2).** The field model's numbers in `field.names` order (dqo: md_x, md_y, md_z, bq,
  aq_x, aq_y, aq_z, mo_x, mo_y, mo_z, c_x, c_y, c_z, u; dqo_aligned: md_x, md_y, md_z, bq, bo, c_x, c_y, c_z, u), then
  `teff` when `[parameters]` has it. `field_slice = slice(0, field.dim)`. A configuration vector entry (md, aq, mo, c:
  value = 3 numbers, one step, lo/hi = a range of the LENGTH) gives its 3 components the same step, the bounds
  -hi .. +hi (search.polish_bounds uses these as its safety limits), and its length range is kept in
  `length_range[name]`; the starts must satisfy it (`accept_start`). u takes its configured lo/hi. A field group the
  configuration does not name is fixed at zero (aq: the unit vector z), so a centred dipole needs only md and u.
- `shift` (FWD-13): a fixed shift is built into the forward model, a free one is a parameter.

`cost(X)` = objective.neg_log_l(forward.spectra(...)) + prior-cut penalty + gauge penalty, rows (K, dim):
- prior cut (DESIGN.md section 3): 2e4 x max(0, f - search.starts.outside_max), f = the fraction of the WHOLE
  surface (all patches, area-weighted, not only the visible disc) with |B| outside the library's B range, smoothed with
  logistic steps at b_min and b_max of scale 2 MG / 4 (so the step rises from 12 % to 88 % over 2 MG). The same
  fraction without smoothing is `outside_fraction(x)`, which the search uses to redraw starts and drop end points.
  `cost` takes f from the forward model's node stage (forward.py, `node_weights_batch` / `spectra_with_outside`),
  which has |B| at every patch centre already: no extra field evaluation. `penalty`, `smooth_outside` and
  `outside_fraction` (callers outside `cost`) delegate to `forward.surface_outside` (one field evaluation).
- gauge penalty (dqo, when aq is free): field.aq_length_penalty (DESIGN.md section 4, step 0).
- invalid rows (forward model or objective) stay at 1e30.

`residuals(x)`: the objective's whitened residuals with the noise scales held FIXED (likelihood.py `residuals`):
the scales are computed at an anchor point and reused while the calls follow the Levenberg-Marquardt Jacobian
pattern of search.polish (the anchor plus `search.fd_step * step` in exactly one coordinate); any other point becomes
the new anchor. This keeps the result independent of which worker ran which start (SEARCH-5).

Pickling: a Problem pickles as its configuration and is rebuilt once per process (or found in the process's registry,
which forked workers inherit), so a process pool does not ship the forward model's records with every task.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref

import numpy as np
from numpy.polynomial import legendre

from .config import HARM_NAMES, ConfigError
from .data import exclusion_masks, load_data, window_index, _windows
from .field import aq_length_penalty, get_model
from .forward import SMOOTH_WIDTH_MG, Forward  # noqa: F401 (SMOOTH_WIDTH_MG kept importable from here)
from .library import Library
from .likelihood import WORST, Objective, legendre_x
from .search import INVALID_RESIDUAL  # residual value returned for an invalid model
from .surface import make_surface
from .types import DataSet, Grid, Spectrum

PRIOR_CUT_WEIGHT = 2e4  # nats per unit of outside fraction above outside_max (DESIGN.md section 3)

_REGISTRY: "weakref.WeakValueDictionary[str, Problem]" = weakref.WeakValueDictionary()


def centre_frame(md: np.ndarray, c: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """The centre c in the frame of the dipole axis and the observer, per phase: (P, 3) rows (c . m, c . e, |c . f|),
    m = md / |md|, e = the part of the observer direction n perpendicular to m, normalised, f = m x e. A rotation of
    md, c and n together (in particular about n, and a change of the inclination with md co-rotating) leaves it
    unchanged, and so does a mirror image (the sign of c . f, which a mirror flips, is dropped). For n parallel to m
    (or md = 0) the second entry is the length of the part of c perpendicular to m and the third 0.
    search.fingerprint_distance compares it when the angle component is beta (decision #16)."""
    md = np.asarray(md, dtype=float)
    c = np.asarray(c, dtype=float)
    obs = np.atleast_2d(np.asarray(obs, dtype=float))
    out = np.zeros((obs.shape[0], 3))
    nmd = float(np.linalg.norm(md))
    m = md / nmd if nmd > 0.0 else np.array([0.0, 0.0, 1.0])
    along = float(c @ m)
    perp = c - along * m
    for p, n in enumerate(obs):
        e = n - float(n @ m) * m
        s = float(np.linalg.norm(e))
        if nmd > 0.0 and s > 1e-9:
            e = e / s
            out[p] = (along, float(c @ e), abs(float(c @ np.cross(m, e))))
        else:
            out[p] = (along, float(np.linalg.norm(perp)), 0.0)
    return out


def _restore(key: str, cfg_json: str) -> "Problem":
    p = _REGISTRY.get(key)
    if p is None:
        p = Problem(json.loads(cfg_json))
    return p


def _groups(names: list[str]) -> list[tuple[str, list[int]]]:
    """Field names -> ordered groups: md_x, md_y, md_z -> ("md", [0, 1, 2]); scalars alone."""
    out: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        g = n[:-2] if len(n) > 2 and n[-2] == "_" and n[-1] in "xyz" else n
        out.setdefault(g, []).append(i)
    return list(out.items())


def blank_dataset(cfg: dict) -> DataSet:
    """A data set with the configured grid, phases and windows but no observations (I = 1, V/I = 0, errors 1); for
    `inject`, which makes the data."""
    d = cfg["data"]
    g = d["grid"]
    grid = Grid(float(g["start"]), float(g["step"]), int(g["npix"]))
    win_idx, win_wt = window_index(grid, _windows(d))
    excl = exclusion_masks(grid, d.get("exclude", []))
    P, M = len(d["phase"]), grid.npix
    phases = np.array([float(p["phase"]) for p in d["phase"]])
    I, VI = np.ones((P, M)), np.zeros((P, M))
    spectra = []
    for p, pc in enumerate(d["phase"]):
        for kind, on in (("I", pc.get("use_I", True)), ("VI", pc.get("use_VI", True))):
            if not on:
                continue
            pix = np.flatnonzero((win_idx >= 0) & ~excl[kind])
            src = I if kind == "I" else VI
            spectra.append(Spectrum(p, float(phases[p]), kind, pix, src[p, pix].copy(), np.ones(pix.size),
                                    win_wt[pix].copy(), win_idx[pix].copy(),
                                    bool(pc.get("i_offset", True)) if kind == "I" else False))
    return DataSet(grid, phases, spectra, I, VI, np.ones((P, M)), np.ones((P, M)))


class Problem:
    """See the module docstring. `Problem(cfg)` with cfg from config.load_config; `dataset` replaces the observations
    (e.g. `blank_dataset(cfg)` for inject)."""

    def __init__(self, cfg: dict, dataset: DataSet | None = None):
        self.cfg = cfg
        lib_teffs = cfg["library"]["teffs"]
        self.library = Library(cfg["library"]["path"], None if lib_teffs == "all" else lib_teffs)
        self.surface = make_surface(cfg["model"]["surface"])
        self.field = get_model(cfg["model"]["field"])
        self.blank = dataset is not None
        self.dataset = dataset if dataset is not None else load_data(cfg)
        self.forward = Forward(self.library, self.surface, self.field, self.dataset, cfg)
        self.objective = Objective(self.dataset, cfg)
        self.outside_max = float(cfg["search"]["starts"]["outside_max"])
        self.fd_step = float(cfg["search"]["fd_step"])
        self.length_range: dict[str, tuple[float, float]] = {}
        self.limit_notes: list[str] = []  # values or limits clamped at load (run.json `limit_notes`)
        self._build_parameters(cfg.get("parameters", {}) or {})
        self._centre_areas = self.surface.areas / self.surface.areas.sum()
        self._anchor = None
        self._anchor_scales = None
        if not self.blank:
            self._key = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
            _REGISTRY[self._key] = self

    # ------------------------------------------------------------------------------------------------- parameters

    def _build_parameters(self, params: dict) -> None:
        fld = self.field
        groups = _groups(fld.names)
        gnames = [g for g, _ in groups]
        unknown = set(params) - set(gnames) - {"teff", "shift"}
        if unknown:
            raise ConfigError(f"parameters {', '.join(sorted(unknown))}: not parameters of field model {fld.name!r} "
                              f"(names {', '.join(gnames)})")

        names, free, lo, hi, step, value = [], [], [], [], [], []
        for g, idx in groups:
            e = params.get(g)
            if e is None:
                if g in ("md", "u"):
                    raise ConfigError(f"missing required key parameters.{g}")
                v = [0.0, 0.0, 1.0] if g == "aq" else ([0.0] * 3 if len(idx) == 3 else 0.0)
                e = {"value": v, "step": 0.1 if g == "aq" else 1.0, "free": False,
                     "lo": 1.0 if g == "aq" else 0.0, "hi": 1.0 if g == "aq" else 0.0}
            if len(idx) == 3:
                if not isinstance(e["value"], list):
                    raise ConfigError(f"parameters.{g}.value must be a vector of 3 numbers")
                self._add(names, free, lo, hi, step, value, g, idx, fld.names, e["value"], e["step"], e["free"],
                          None, None, (e["lo"], e["hi"]))
            else:
                if isinstance(e["value"], list):
                    raise ConfigError(f"parameters.{g}.value must be a number")
                self._add(names, free, lo, hi, step, value, g, idx, fld.names, e["value"], e["step"], e["free"],
                          e["lo"], e["hi"], None)

        nf = len(names)
        self.iteff = None
        if "teff" in params:
            e = params["teff"]
            tlo, thi = self.library.teff_min, self.library.teff_max
            if e["lo"] < tlo - 0.5 or e["hi"] > thi + 0.5:
                raise ConfigError(f"parameters.teff: range [{e['lo']:g}, {e['hi']:g}] K is not inside the selected "
                                  f"library temperatures [{tlo:g}, {thi:g}] K (library.teffs)")
            self.iteff = nf
            # the value and limits up to 0.5 K beyond an edge slice (check_teff's tolerance) are clamped to it: the
            # model is that slice there, constant in Teff, so a polish limit there would give a zero Teff derivative;
            # all three together, so that a range wholly beyond the edge becomes lo = hi = value = the edge slice and
            # never lo > hi; a note says so in run.json
            given = (float(e["lo"]), float(e["hi"]), float(e["value"]))
            t_lo, t_hi, t_v = (min(max(t, tlo), thi) for t in given)
            if (t_lo, t_hi, t_v) != given:
                self.limit_notes.append(
                    f"parameters.teff: lo, hi, value = {given[0]:g}, {given[1]:g}, {given[2]:g} K clamped to "
                    f"{t_lo:g}, {t_hi:g}, {t_v:g} K, the selected slice range [{tlo:g}, {thi:g}] K (the model up "
                    f"to 0.5 K beyond an edge slice is that slice)")
            names.append("teff"), free.append(bool(e["free"]))
            lo.append(t_lo), hi.append(t_hi)
            step.append(e["step"]), value.append(t_v)
        elif len(self.library.teffs) != 1:
            raise ConfigError(f"parameters.teff is missing, but {len(self.library.teffs)} library temperatures are "
                              f"selected ({self.library.teffs.tolist()}); add teff or select one in library.teffs")
        self.ishift = None
        if "shift" in params:  # FWD-13: a fixed shift is built into the forward model; a free one is a parameter
            e = params["shift"]
            self.ishift = len(names)
            names.append("shift"), free.append(bool(e["free"])), lo.append(e["lo"]), hi.append(e["hi"])
            step.append(e["step"]), value.append(e["value"])

        self.names = names
        self.dim = len(names)
        self.free = np.array(free, dtype=bool)
        self.lo = np.array(lo, dtype=float)
        self.hi = np.array(hi, dtype=float)
        self.step = np.array(step, dtype=float)
        self.value = np.array(value, dtype=float)
        self.field_slice = slice(0, fld.dim)
        self._aq_free = fld.name == "dqo" and bool(
            np.any(self.free[[fld.names.index(n) for n in ("aq_x", "aq_y", "aq_z")]]))
        # the start distribution must reach the length range of md
        st = self.cfg["search"]["starts"]
        if "md" in self.length_range and np.any(self.free[0:3]):
            a, b = self.length_range["md"]
            if st["md"][1] < a or st["md"][0] > b:
                raise ConfigError(f"search.starts.md = {st['md']} MG does not overlap the length range of "
                                  f"parameters.md, [{a:g}, {b:g}] MG")

    def _add(self, names, free, lo, hi, step, value, g, idx, fnames, v, st, fr, l, h, rng):
        if len(idx) == 3:
            v = [float(x) for x in np.asarray(v, dtype=float).reshape(3)] if v is not None else (
                [0.0, 0.0, 1.0] if g == "aq" else [0.0, 0.0, 0.0])
            a, b = rng
            self.length_range[g] = (float(a), float(b))
            for k, i in enumerate(idx):
                names.append(fnames[i]), free.append(bool(fr)), lo.append(-float(b)), hi.append(float(b))
                step.append(float(st)), value.append(v[k])
        else:
            names.append(fnames[idx[0]]), free.append(bool(fr)), lo.append(float(l)), hi.append(float(h))
            step.append(float(st)), value.append(float(v))

    # ------------------------------------------------------------------------------------------------- helpers

    def __reduce__(self):
        if self.blank:
            raise TypeError("a Problem on a blank data set (inject) cannot be pickled")
        return (_restore, (self._key, json.dumps(self.cfg, sort_keys=True, default=str)))

    def thetas(self, X) -> np.ndarray:
        """(K, dim) parameter rows -> (K, field.dim) field vectors."""
        return np.atleast_2d(np.asarray(X, dtype=float))[:, : self.field.dim]

    def theta(self, x) -> np.ndarray:
        return self.thetas(np.asarray(x, dtype=float)[None, :])[0]

    def _teff(self, X):
        return None if self.iteff is None else X[:, self.iteff]

    def _shift(self, X):
        """The shift per row when it is free (FWD-13); a fixed shift is built into the forward model."""
        if self.ishift is None or not self.free[self.ishift]:
            return None
        return X[:, self.ishift]

    def spectra(self, X):
        """(K, dim) -> I (K, P, M), VI (K, P, M), invalid (K,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.forward.spectra(self.thetas(X), teff=self._teff(X), shift=self._shift(X))

    def smooth_outside(self, Th) -> np.ndarray:
        """(K,) whole-surface area fraction with |B| outside the library, logistic steps of width SMOOTH_WIDTH_MG
        (forward.surface_outside; `cost` gets the same numbers from the node stage)."""
        return self.forward.surface_outside(Th, smooth=True)

    def _penalty(self, Th, fsurf) -> np.ndarray:
        """(K,) prior-cut penalty of the smoothed whole-surface outside fraction fsurf (K,) + gauge penalty."""
        pen = PRIOR_CUT_WEIGHT * np.maximum(0.0, fsurf - self.outside_max)
        if self._aq_free:
            pen = pen + np.array([aq_length_penalty(t) for t in Th])
        return pen

    def penalty(self, X) -> np.ndarray:
        """(K,) prior-cut penalty + gauge penalty."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Th = self.thetas(X)
        return self._penalty(Th, self.smooth_outside(Th))

    # ------------------------------------------------------------------------------------------------- protocol

    def cost(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Th = self.thetas(X)
        if self.objective.bind_forward(self.forward) == "on":  # pre-whitened records (likelihood.py), built once
            Wn, sw, bad, _, fsurf = self.forward.node_weights_batch(Th, teff=self._teff(X), shift=self._shift(X))
            f = self.objective.neg_log_l_nodes(Wn, sw, bad)
        else:
            I, VI, bad, fsurf = self.forward.spectra_with_outside(Th, teff=self._teff(X), shift=self._shift(X))
            f = self.objective.neg_log_l(I, VI, bad)
        ok = f < WORST
        if np.any(ok):  # the outside fraction comes from the node stage: no second field evaluation
            f[ok] = f[ok] + self._penalty(Th[ok], fsurf[ok])
        f[~np.isfinite(f)] = WORST
        return np.minimum(f, WORST)

    def outside_fraction(self, x) -> float:
        """Whole-surface area fraction with |B| outside the library's B range (hard edges; the prior cut)."""
        return float(self.forward.surface_outside(self.theta(x)[None, :], smooth=False)[0])

    def accept_start(self, x) -> bool:
        """Starts must respect the configured length ranges of the free vector parameters, and lo/hi of u."""
        x = np.asarray(x, dtype=float)
        for g, idx in _groups(self.field.names):
            if len(idx) == 3 and g in self.length_range and np.any(self.free[idx]):
                a, b = self.length_range[g]
                if not a <= float(np.linalg.norm(x[idx])) <= b:
                    return False
        iu = self.field.names.index("u")
        return bool(self.lo[iu] <= x[iu] <= self.hi[iu]) if self.free[iu] else True

    def _is_jacobian_point(self, x) -> bool:
        a = self._anchor
        if a is None:
            return False
        d = x - a
        nz = np.flatnonzero(d != 0.0)
        if nz.size != 1:
            return False
        j = int(nz[0])
        h = self.fd_step * self.step[j]
        return abs(d[j] - h) <= 1e-9 * max(1.0, abs(h))

    def residuals(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        I, VI, bad = self.spectra(x[None, :])
        if bad[0] or not self._finite_on_pixels(I[0], VI[0]):
            return np.full(sum(len(s.pixels) for s in self.dataset.spectra), INVALID_RESIDUAL)
        if not self._is_jacobian_point(x):
            self._anchor = x.copy()
            self._anchor_scales = self.objective.noise_scales(I[0], VI[0])
        return np.concatenate(self.objective.residuals(I[0], VI[0], scales=self._anchor_scales))

    def residual_scales(self, x) -> np.ndarray | None:
        """The noise-scale estimates at x (held fixed by a least-squares step, `residuals`); None if x is invalid."""
        I, VI, bad = self.spectra(np.asarray(x, dtype=float)[None, :])
        if bad[0] or not self._finite_on_pixels(I[0], VI[0]):
            return None
        return self.objective.noise_scales(I[0], VI[0])

    def residuals_batch(self, X, scales) -> np.ndarray:
        """Whitened residuals (K, R) of a batch, each spectrum divided by the given noise scale (one forward call for
        all rows; search._lm builds its central-difference Jacobian from it, decision #25), or with scales None by the
        noise scales of row 0 (`residual_scales` of X[0], from the same call). Invalid rows are filled with
        INVALID_RESIDUAL (all rows when scales is None and row 0 is invalid)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        I, VI, bad = self.spectra(X)
        nres = sum(len(s.pixels) for s in self.dataset.spectra)
        out = np.full((X.shape[0], nres), INVALID_RESIDUAL)
        ok = np.flatnonzero(~bad)
        if ok.size == 0 or (scales is None and ok[0] != 0):
            return out
        parts = []
        ob = self.objective
        for j, s in enumerate(ob._specs):
            res, _ = ob._fit(s, ob._rows(s, I[ok], VI[ok]))
            if scales is None:  # row 0 is valid here (ok[0] == 0): the scales of `objective.noise_scales` at X[0]
                sc = math.sqrt(float(res[0] @ res[0]) / (s.n - s.k)) if ob.scale else 1.0
            else:
                sc = scales[j]
            parts.append(res / sc)
        R = np.concatenate(parts, axis=1)
        fin = np.all(np.isfinite(R), axis=1)
        out[ok[fin]] = R[fin]
        return out

    def _finite_on_pixels(self, I, VI) -> bool:
        for s in self.dataset.spectra:
            src = I if s.kind == "I" else VI
            if not np.all(np.isfinite(src[s.phase_index, s.pixels])):
                return False
        return True

    def fingerprint_quantities(self, x) -> dict:
        th = self.theta(x)
        u = float(th[self.field.names.index("u")])
        names = self.field.names
        c = th[[names.index(n) for n in ("c_x", "c_y", "c_z")]] if "c_x" in names else np.zeros(3)
        q = self.forward.b_quantiles(th, (0.5, 0.9))
        xx = np.asarray(x, dtype=float)
        extra = {self.names[i]: float(xx[i]) for i in range(self.field.dim, self.dim) if self.free[i]}  # e.g. teff (A8)
        # beta per phase: the angle between the dipole axis md and the observer direction (forward.observers), 0-180
        # degrees, unchanged by the equator mirror; used by search.fingerprint with [search.fingerprint] angle = "beta"
        md = th[[names.index(n) for n in ("md_x", "md_y", "md_z")]]
        nmd = float(np.linalg.norm(md))
        obs = self.forward.observers(th[None, :])[0][0]  # (P, 3)
        beta = (np.degrees(np.arccos(np.clip(obs @ md / nmd, -1.0, 1.0))) if nmd > 0.0
                else np.full(obs.shape[0], 90.0))
        return {"incl": math.degrees(math.acos(min(1.0, abs(u)))), "c": c, "b50": q[:, 0], "b90": q[:, 1],
                "beta": beta, "c_frame": centre_frame(md, c, obs), "extra": extra}

    def canonical(self, x) -> np.ndarray:
        out = np.array(x, dtype=float)
        out[self.field_slice] = self.field.canonical(out[self.field_slice])
        return out

    def mirror(self, x) -> np.ndarray:
        """The equator-mirror image of x (DESIGN.md section 2; field.py `canonical`): the field model's `mirror` (u and
        the z components of md, aq, mo and c negated, bq and bo unchanged; for "harm4" the signs (-1)^(l+m)), the
        non-field parameters unchanged; canonical(mirror(x)) = canonical(x). The
        mirror is a symmetry of the exact problem but not of the Fibonacci patch grid, so cost(mirror(x)) - cost(x) is
        quadrature noise (search.settle_mirror)."""
        out = np.array(x, dtype=float)
        out[self.field_slice] = self.field.mirror(out[self.field_slice])
        return out

    # ------------------------------------------------------------------------------------------------- outputs

    def model_spectra(self, x):
        """Raw model I (P, M) and V/I (P, M) for one parameter set (NaN if invalid)."""
        I, VI, _ = self.spectra(np.asarray(x, dtype=float)[None, :])
        return I[0], VI[0]

    def continuum_models(self, I, VI) -> np.ndarray:
        """Model I times its best continuum (as fitted inside the likelihood: the same whitening, Legendre degree and
        offset), on the grid pixels from the first to the last fitted pixel of each I spectrum; NaN elsewhere and for
        phases without a fitted I. Returns (P, M)."""
        out = np.full(I.shape, np.nan)
        cont = self.objective.settings["continuum"]
        lam_all = self.dataset.grid.wavelengths
        for j, s in enumerate(self.dataset.spectra):
            if s.kind != "I" or len(s.pixels) == 0:
                continue
            p, pix = s.phase_index, s.pixels
            m = I[p, pix]
            if cont is False:
                out[p, pix[0]: pix[-1] + 1] = I[p, pix[0]: pix[-1] + 1]
                continue
            deg = int(cont["degree"])
            lam = lam_all[pix]
            basis = legendre.legvander(legendre_x(lam), deg)  # (n, deg+1)
            cols = [m[:, None] * basis]
            if s.i_offset:
                cols.append(np.ones((pix.size, 1)))
            A = np.concatenate(cols, axis=1)
            WA = self.objective.whiten(j, A)
            wy = self.objective.whiten(j, np.asarray(s.data, dtype=float))
            # columns scaled to unit length first: with the model in physical flux units (1e15 x the data) the offset
            # column would otherwise fall below lstsq's rcond and be dropped
            d = np.linalg.norm(WA, axis=0)
            d[~(d > 0)] = 1.0
            coef, *_ = np.linalg.lstsq(WA / d, wy, rcond=None)
            coef = coef / d
            span = np.arange(pix[0], pix[-1] + 1)
            ls = lam_all[span]
            xs = 2.0 * (ls - lam[0]) / (lam[-1] - lam[0]) - 1.0 if lam[-1] > lam[0] else np.zeros_like(ls)
            full = I[p, span][:, None] * legendre.legvander(xs, deg)
            if s.i_offset:
                full = np.concatenate([full, np.ones((span.size, 1))], axis=1)
            out[p, span] = full @ coef
        return out

    def derived(self, x) -> dict:
        """Derived quantities of FORMATS.md section 7 (and a few more) for one parameter set."""
        x = np.asarray(x, dtype=float)
        th = self.theta(x)
        names = self.field.names
        u = float(th[names.index("u")])
        md = th[0:3]
        q = self.forward.b_quantiles(th, (0.1, 0.5, 0.9))
        obs, _ = self.forward.observers(th[None, :])
        nmd = float(np.linalg.norm(md))
        ang = [math.degrees(math.acos(max(-1.0, min(1.0, float(md @ n) / nmd)))) if nmd > 0 else float("nan")
               for n in obs[0]]
        c = th[[names.index(n) for n in ("c_x", "c_y", "c_z")]] if "c_x" in names else np.zeros(3)
        out = {
            "incl_deg": math.degrees(math.acos(min(1.0, abs(u)))),
            "i_deg_signed": math.degrees(math.acos(max(-1.0, min(1.0, u)))),
            "dipole_MG": nmd,
            "c_abs": float(np.linalg.norm(c)),
            "phases": [float(v) for v in self.forward.phases],
            "B10_MG": q[:, 0].tolist(), "B50_MG": q[:, 1].tolist(), "B90_MG": q[:, 2].tolist(),
            "outside_fraction_visible": self.forward.outside_fraction(th).tolist(),
            "outside_fraction_surface": self.outside_fraction(x),
            "angle_dipole_observer_deg": ang,
            "beta_deg": ang[0] if ang else float("nan"),
            "c_along_axis": float(c @ md) / nmd if nmd > 0 else float("nan"),
            "c_across_axis": (float(np.linalg.norm(c - (float(c @ md) / nmd ** 2) * md)) if nmd > 0
                              else float("nan")),
            "teff_K": None if self.iteff is None else float(x[self.iteff]),
        }
        return out

    @property
    def nuisance_coordinates(self) -> tuple[str, ...]:
        """Parameters whose values one phase cannot determine (decision #18): with one phase and the field in vectors,
        the orientation of the whole geometry about the line of sight and the inclination are free (only |md|, beta,
        the centre relative to md and the observer, the B distribution and the non-field parameters are determined),
        so the components of every field vector, the harm4 coefficients and u are declared nuisance; uncertainty.py
        samples them but leaves them out of the V14 verdict and reports no interval for them. () otherwise."""
        if len(self.forward.phases) != 1:
            return ()
        return tuple(n for n in self.field.names if n == "u" or n[-2:] in ("_x", "_y", "_z") or n in HARM_NAMES)

    @property
    def aq_penalty_in_cost(self) -> bool:
        """uncertainty.py: `cost` already holds the gauge penalty on ln|aq| (dqo with a free aq)."""
        return bool(self._aq_free)

    def derived_batch(self, X, chunk: int = 256) -> dict:
        """The scalar and per-phase quantities of `derived` for many rows at once (uncertainty.py intervals):
        {incl_deg, dipole_MG, c_abs, beta_deg (at the first phase), c_along_axis, c_across_axis: (K,); B10_MG, B50_MG,
        B90_MG, angle_dipole_observer_deg: (K, P)}. teff is a
        parameter and not repeated here. The B quantiles are those of `forward.b_quantiles` (brightness-weighted |B|
        over the visible patch centres, each patch at the midpoint of its share of the cumulative weight, linear in
        between, clamped at the ends), computed for `chunk` rows at a time."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Th = self.thetas(X)
        K = Th.shape[0]
        names = self.field.names
        fw = self.forward
        u = Th[:, names.index("u")]
        md = Th[:, 0:3]
        nmd = np.linalg.norm(md, axis=1)
        c = Th[:, [names.index(n) for n in ("c_x", "c_y", "c_z")]] if "c_x" in names else np.zeros((K, 3))
        obs, _ = fw.observers(Th)  # (K, P, 3)
        with np.errstate(all="ignore"):
            cosang = np.einsum("ki,kpi->kp", md, obs) / nmd[:, None]
        P = obs.shape[1]
        qs = np.array([0.1, 0.5, 0.9])
        bq = np.empty((K, P, 3))
        cen, areas = fw.centres, fw.areas
        for a in range(0, K, chunk):
            sl = slice(a, min(K, a + chunk))
            with np.errstate(all="ignore"):
                B = self.field.evaluate_batch(cen, Th[sl])  # (k, N, 3)
            bmag = np.sqrt(np.einsum("kni,kni->kn", B, B))
            for p in range(P):
                mu = obs[sl, p] @ cen.T  # (k, N)
                vis = mu > 0.0
                w = np.where(vis, areas[None, :] * mu * (fw.ld_a + fw.ld_b * mu), 0.0)
                b = np.where(vis, bmag, np.inf)
                o = np.argsort(b, axis=1, kind="stable")
                b = np.take_along_axis(b, o, axis=1)
                w = np.take_along_axis(w, o, axis=1)
                nvis = vis.sum(axis=1)
                cdf = (np.cumsum(w, axis=1) - 0.5 * w) / w.sum(axis=1, keepdims=True)
                cdf = np.where(np.isfinite(b), cdf, np.inf)
                rows = np.arange(b.shape[0])
                for iq, q in enumerate(qs):
                    j = np.sum(cdf <= q, axis=1) - 1
                    last = nvis - 1
                    jj = np.clip(j, 0, np.maximum(last - 1, 0))
                    x0, x1 = cdf[rows, jj], cdf[rows, np.minimum(jj + 1, b.shape[1] - 1)]
                    y0, y1 = b[rows, jj], b[rows, np.minimum(jj + 1, b.shape[1] - 1)]
                    with np.errstate(all="ignore"):
                        mid = y0 + (q - x0) / (x1 - x0) * (y1 - y0)
                    val = np.where(j < 0, b[:, 0], np.where(j >= last, b[rows, np.maximum(last, 0)], mid))
                    bq[sl, p, iq] = np.where(nvis > 0, val, np.nan)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        with np.errstate(all="ignore"):
            along = np.where(nmd > 0, np.einsum("ki,ki->k", c, md) / nmd, np.nan)
            across = np.where(nmd > 0, np.linalg.norm(c - (along / nmd)[:, None] * md, axis=1), np.nan)
        return {"incl_deg": np.degrees(np.arccos(np.minimum(1.0, np.abs(u)))), "dipole_MG": nmd,
                "c_abs": np.linalg.norm(c, axis=1), "B10_MG": bq[:, :, 0], "B50_MG": bq[:, :, 1],
                "B90_MG": bq[:, :, 2], "angle_dipole_observer_deg": ang, "beta_deg": ang[:, 0],
                "c_along_axis": along, "c_across_axis": across}

    def grouped(self, x) -> dict:
        """{configuration name: value} (vectors as 3 numbers) for one parameter row."""
        x = np.asarray(x, dtype=float)
        out: dict = {}
        for g, idx in _groups(self.field.names):
            out[g] = [float(v) for v in x[idx]] if len(idx) == 3 else float(x[idx[0]])
        if self.iteff is not None:
            out["teff"] = float(x[self.iteff])
        return out

    def evaluation(self, x) -> dict:
        """Everything `eval` reports for one parameter set."""
        x = np.asarray(x, dtype=float)
        I, VI, bad = self.spectra(x[None, :])
        nll = float(self.objective.neg_log_l(I, VI, bad)[0])
        pen = float(self.penalty(x)[0]) if nll < WORST else 0.0
        per = {}
        if nll < WORST:
            per = self.objective.per_spectrum(I[0], VI[0])
            for lab, d in per.items():
                d["chi2_reduced"] = d["chi2"] / max(1, d["n"] - d["k"])
            scales = self.objective.noise_scales(I[0], VI[0])
            for lab, sc in zip(self.objective.labels, scales):
                per[lab]["noise_scale"] = float(sc)
        scale = {}
        if nll < WORST:  # median |model I| / median |data I| (likelihood.py)
            scale = self.objective.check_scale(I[0])
        return {"neg_log_l": nll, "constant": float(self.objective.constant), "penalty": pen,
                "cost": min(WORST, nll + pen), "valid": bool(nll < WORST), "kind": self.objective.kind,
                "per_spectrum": per, "scale_ratio": scale, "parameters": self.grouped(x), "derived": self.derived(x)}

