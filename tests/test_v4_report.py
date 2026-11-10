"""V4e and V4f (report wording; SPEC.md REP-1, REP-2, UNC-4, UNC-6, SYS-3, SYS-5): report.txt built by
`report.report_text` from hand-made mode, interval and systematics tables (no fit needed): the error-bar
part and the systematics part."""

from __future__ import annotations

import re

import pytest

from zeemap.report import ERRBAR_COLUMNS, report_text

pytestmark = pytest.mark.tier1

STATS = {"n_starts": 8, "rounds": 1, "nevals": 12345, "band": 45.0, "value_tolerance": 2.0, "n_modes": 2,
         "n_band_modes": 2, "hits_best": 5, "n_band_once": 1, "frac_band_once": 0.5, "optimiser_scatter": 0.3,
         "round_gains": [0.1], "n_endpoints": 12, "n_invalid_endpoints": 0, "warning": False}
QUANTS = ["md_x", "u", "incl_deg", "dipole_MG", "c_abs", "B50_MG[0]"]


def _modes():
    return [{"id": k, "delta": d, "n_starts": n, "neg_log_l": 100.0 + d, "touched_limit": False,
             "derived": {"incl_deg": incl, "dipole_MG": 400.0, "c_abs": 0.0, "B50_MG": [250.0, 260.0],
                         "teff_K": None}}
            for k, d, n, incl in ((1, 0.0, 5, 61.234), (2, 4.0, 3, 29.876))]


def _q(med, lo, hi, flags=()):
    return {"median": med, "p16": lo, "p84": hi, "p2_5": lo - 1, "p97_5": hi + 1, "half68": 0.5 * (hi - lo),
            "flags": list(flags), "jackknife_sigma": None}


def _intervals(converged=True):
    conv = {"tau": 12.0, "ess_bulk": {"u": 500.0}, "rhat": {"u": 1.001}, "acceptance": [0.3], "ensembles": 4,
            "nwalkers": 16, "steps": 2000, "burn": 200, "nevals": 128000, "max_evals": 200000}
    q = {"md_x": _q(346.0, 345.0, 347.0), "u": _q(0.5, 0.49, 0.51),
         "incl_deg": _q(60.0, 59.4, 60.6, ["touches a range limit"]), "dipole_MG": _q(400.0, 399.0, 401.0),
         "c_abs": _q(0.0, 0.0, 0.01, ["not constrained by the data"]), "B50_MG[0]": _q(250.0, 249.8, 250.2)}
    if not converged:
        q = {k: dict(v, median=None, p16=None, p84=None, flags=["not converged"]) for k, v in q.items()}
    return {"calibration": "uncalibrated", "label": "per mode: conditional on this basin",
            "patches": 20000, "nevals": 128000, "modes_without_mcmc": [2], "notes": [],
            "modes": [{"mode": 1, "quantities": q, "convergence": conv, "converged": converged,
                       "failed_rules": [] if converged else ["V14a"], "flags": [], "handbacks": [], "notes": []}]}


def _systematics(jk=True):
    swaps = [{"name": "ld1", "kind": "config", "keys": ["model.ld_mode"]},
             {"name": "u_far", "kind": "config", "keys": ["parameters.u.free", "parameters.u.value"]},
             {"name": "drop_phase_0", "kind": "jackknife", "phase": 0.0, "keys": ["data.phase[0]"]},
             {"name": "drop_phase_0.5", "kind": "jackknife", "phase": 0.5, "keys": ["data.phase[1]"]}]
    shifts = {q: {"ld1": 5.0, "u_far": "mode change", "drop_phase_0": 0.1, "drop_phase_0.5": -0.1} for q in QUANTS}
    shifts["dipole_MG"]["ld1"] = 0.5  # 0.5 < 3 x 1 MG: not limited
    m = {"mode": 1, "left_mode_rule": "mcmc_p99", "left_mode_limit": 1.7, "d99": 1.7, "shifts": shifts,
         "mode_changes": ["u_far"], "limited_by_model_systematics": ["incl_deg", "B50_MG[0]"], "sys5_note": None,
         "jackknife_sigma": {q: 0.1 for q in QUANTS} if jk else None, "jackknife_n": 2, "jackknife_note": None,
         "notes": []}
    return {"swaps": swaps, "modes": [m], "notes": [], "n_mode_changes": 1, "modes_without_systematics": [2]}


def _report(converged=True, sysd="default"):
    iv = _intervals(converged)
    s = _systematics() if sysd == "default" else sysd
    if s:
        for q, e in iv["modes"][0]["quantities"].items():  # what attach_jackknife does
            e["jackknife_sigma"] = s["modes"][0]["jackknife_sigma"][q]
    return report_text(_modes(), STATS, 10.0, 1, 1, "full", iv, 5.0, s)


# ------------------------------------------------------------------------------------------------------------- V4e


@pytest.mark.parametrize("variant", ["full", "no_mcmc", "not_converged", "no_systematics"])
def test_v4e_no_single_best_fit_without_mode_table_and_no_evidence(variant):
    """REP-1: the mode table header comes before any parameter value; REP-2: no evidence or weights anywhere."""
    if variant == "no_mcmc":
        text = report_text(_modes(), STATS, 10.0, 1, 1, "full", None, None, _systematics())
    elif variant == "not_converged":
        text = _report(converged=False)
    elif variant == "no_systematics":
        text = report_text(_modes(), STATS, 10.0, 1, 1, "full", _intervals(), 5.0, None)
    else:
        text = _report()
    head = text.index("mode table")
    hdr = text.index("-lnL vector")  # the table header (the line above it explains the columns)
    assert head < hdr
    first_value = min(text.index(s) for s in ("61.23", "400.0", "29.88"))
    assert hdr < first_value, "a parameter value appears before the mode table header"
    assert text.count("\n   1  ") >= 1 and text.count("\n   2  ") >= 1  # both modes in the table, not only the best
    low = text.lower()
    assert "evidence" not in low and "weight" not in low and "probability" not in low


# ------------------------------------------------------------------------------------------------------------- V4f


def test_v4f_labels_per_mode_uncalibrated_limits_and_unconstrained():
    text = _report()
    assert ("intervals: statistical (68 %, per mode: conditional on this basin), uncalibrated: no coverage run named "
            "in the configuration") in text
    assert "tempered" not in text  # decision #24
    assert "touches a range limit (REP-3)" in text and "60.0 +0.6 -0.6 [L]" in text  # REP-3 on incl
    assert "flags: touches a range limit: incl_deg; not constrained by the data: c_abs" in text
    assert "not constrained by the data (REP-4)" in text and "[U]" in text
    assert "n.c." in _report(converged=False)


def test_v4f_limited_by_model_systematics_wording():
    text = _report()
    assert ("limited by model systematics (a swap moves it by more than 3 sigma), mode 1: incl_deg, B50_MG[0]"
            in text)
    s = _systematics()
    s["modes"][0]["limited_by_model_systematics"] = []
    s["modes"][0]["sys5_note"] = "not assessed: no intervals for this mode (no MCMC)"
    text = report_text(_modes(), STATS, 10.0, 1, 1, "full", None, None, s)
    assert "limited by model systematics, mode 1: not assessed: no intervals for this mode (no MCMC)" in text


def test_v4f_shifts_listed_apart_from_error_bars():
    """SYS-3: the shift lines come after the interval table and never inside it; the interval cells hold only the
    statistical numbers (the jackknife sigma is its own "jk" row, not added in)."""
    text = _report()
    lines = text.splitlines()
    i_tab = next(i for i, l in enumerate(lines) if l.startswith("error bars (68 % intervals"))
    i_end = next(i for i, l in enumerate(lines) if l.startswith("intervals: statistical"))
    shift_idx = [i for i, l in enumerate(lines) if l.startswith("systematic shifts of mode")]
    assert shift_idx and min(shift_idx) > i_end > i_tab
    table = lines[i_tab:i_end]
    assert not any("shift" in l or "mode change" in l or "ld1" in l for l in table)
    row = next(l for l in table if l.startswith("   1  "))
    assert "60.0 +0.6 -0.6" in row and "400.000 +1.000 -1.000" in row  # unchanged by the 5.0 shifts
    jk = next(l for l in table if l.startswith("  jk  "))
    assert "0.1" in jk
    assert "systematic shifts of mode 1: u_far: mode change" in text
    assert re.search(r"systematic shifts of mode 1: ld1: incl \+5\.00; \|dipole\| \+0\.500", text)
    # the configuration lists no swaps and one phase: a line says so
    none = {"swaps": [], "modes": [], "notes": ["no swaps: the configuration lists no [[systematics.swap]] and the "
                                                "data have one phase"], "n_mode_changes": 0}
    t2 = report_text(_modes(), STATS, 10.0, 1, 1, "full", None, None, none)
    assert "systematics: none run: no swaps" in t2
    assert "systematics: not computed (fit --no-systematics)" in report_text(_modes(), STATS, 10.0, 1, 1, "full")
    assert ERRBAR_COLUMNS[0][1] == "incl_deg"


def test_v4f_per_quantity_intervals_and_orientation_nuisance():
    """Decision #17 (UNC-2 per quantity): a mode that fails V14 still prints the quantities that pass on their own,
    "n.c." for those that fail, and the mode-level verdict; decision #18: orientation quantities of a single-phase fit
    print "orient." and the nuisance names are listed as not required."""
    iv = _intervals(converged=False)
    q = iv["modes"][0]["quantities"]
    q["dipole_MG"] = _q(400.0, 399.0, 401.0)
    q["beta_deg"] = _q(116.6, 114.7, 118.4)
    q["c_along_axis"] = _q(-0.221, -0.228, -0.214)
    q["incl_deg"] = dict(q["incl_deg"], flags=["orientation not determined by one phase", "nuisance, not required"])
    q["u"] = dict(q["u"], flags=["orientation not determined by one phase", "nuisance, not required"])
    iv["modes"][0]["convergence"].update(nuisance=["md_p", "md_q", "u"], rhat={"u": 1.63}, ess_bulk={"u": 157.0})
    text = report_text(_modes(), STATS, 10.0, 1, 1, "full", iv, 5.0, None)
    lines = text.splitlines()
    i_tab = next(i for i, l in enumerate(lines) if l.startswith("error bars (68 % intervals"))
    row = next(l for l in lines[i_tab:] if l.startswith("   1  "))
    assert row.split()[1] == "orient."  # incl
    assert "400.000 +1.000 -1.000" in row and "116.6 +1.8 -1.9" in row and "-0.221 +0.007 -0.007" in row
    assert "n.c." in row  # |c| and B50 failed on their own
    assert "  mode 1: NOT CONVERGED" in text  # the mode-level verdict is printed as before
    assert "nuisance, not required (left out of the verdict): md_p, md_q, u (u R-hat 1.630 ESS 157)" in text
    assert "not converged (no interval; UNC-2 per quantity): md_x, c_abs, B50_MG[0]" in text
    assert "orientation not determined by one phase: u, incl_deg" in text
    assert [h for h, _, _ in ERRBAR_COLUMNS] == ["incl deg", "|dipole| MG", "beta deg", "|c|", "c along axis",
                                                 "B50 ph0 MG", "teff K"]
