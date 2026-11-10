"""End to end: the kernel stage of the search (LIKE-2, DESIGN.md section 3 step 2b), the whiteness check in the
report (V4h), the floor chi2 (V2i) and the per-window residual table (V7i), on the synthetic library with the full
likelihood and injected white noise. Narrow fit windows (400 pixels per spectrum) keep the kernel fits within tier 1."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zeemap import cli
from zeemap.config import load_config
from zeemap.diagnostics import floor_chi2
from zeemap.problem import Problem

from test_v7_cli import config_text

pytestmark = pytest.mark.tier1


def _text(data, kind="full", extra=""):
    return config_text(data, kind=kind, extra=extra).replace("windows = [[4300.0, 6700.0]]",
                                                             "windows = [[4400.0, 5200.0]]")


@pytest.fixture(scope="module")
def full_run(synth_lib, tmp_path_factory):
    root = tmp_path_factory.mktemp("noise_data")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        truth = root / "truth.toml"
        truth.write_text(_text("inj", extra='[systematics]\nlibrary_alternatives = ["synth"]\n'))
        assert cli.main(["inject", str(truth), "-o", str(root / "inj"), "--noise", "white", "--seed", "5",
                         "--sigma-i", "0.01", "--sigma-vi", "0.002"]) == 0
        assert cli.main(["inject", str(truth), "-o", str(root / "clean"), "--noise", "none"]) == 0
        fit = root / "fit.toml"
        fit.write_text(_text("inj", extra='[systematics]\nlibrary_alternatives = ["synth"]\n'))
        out = root / "fit"
        assert cli.main(["fit", str(fit), "-o", str(out), "--seed", "2", "--starts", "3", "--no-mcmc",
                         "--no-systematics"]) == 0
    return root, out


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v4h_kernel_fitted_frozen_in_run_json_and_report(full_run):
    root, out = full_run
    run = json.loads((out / "run.json").read_text())
    k = run["likelihood"]["kernel"]
    assert k["state"] == "fitted, frozen" and len(k["frozen"]) == 4
    for e in k["frozen"]:
        for key in ("spectrum", "rho_short", "a", "rho_long", "j", "share_short", "gain", "at_bound"):
            assert key in e
        assert 0.5 * 2.0 <= e["rho_short"] <= 3.0 + 1e-9  # [0.5 pixel, FWHM]
    # the frozen kernel is in the configuration as used, so workers and swaps carry it
    assert "kernel_fit = false" in run["config"] and "[[objective.kernel]]" in run["config"]
    text = (out / "report.txt").read_text()
    assert "likelihood: full; kernel: fitted, frozen; continuum: Legendre degree 3 on I" in text
    assert "noise model (LIKE-10): whitened residuals at the best mode white in" in text
    diag = json.loads((out / "diagnostics.json").read_text())
    wh = diag["statistics"]["whiteness"]
    assert len(wh) == 4 and all({"max_abs_acf", "threshold", "pass"} <= set(v) for v in wh.values())
    # white noise injected, kernel fitted: the residuals are white
    assert diag["statistics"]["whiteness_pass"] is True


def test_v4h_failed_whiteness_flags_the_report():
    from zeemap.report import report_text

    stats = {"n_starts": 1, "rounds": 0, "nevals": 1, "band": 45.0, "value_tolerance": 2.0, "n_modes": 1,
             "n_band_modes": 1, "hits_best": 1, "n_band_once": 1, "frac_band_once": 1.0, "optimiser_scatter": 0.0,
             "round_gains": [], "n_endpoints": 1, "n_invalid_endpoints": 0, "warning": False,
             "whiteness": {"0.000_I": {"max_abs_acf": 0.3, "lag": 1, "threshold": 0.15, "n": 900, "pass": False}},
             "kernel_at_bound": {"0.000_I": {"a": "upper", "j": "lower"}}}
    text = report_text([], stats, 1.0, 1, 1, "full", header="likelihood: full; kernel: fitted, frozen; continuum: off")
    assert "NOISE MODEL FAILS THE WHITENESS CHECK: 0.000_I" in text and "white in 0 of 1 spectra" in text
    assert "at its cap a = 10" in text and "0.000_I a upper, j lower" in text  # the capped kernel is named


@pytest.fixture(scope="module")
def red_runs(synth_lib, tmp_path_factory):
    """Correlated noise injected (kind full, the default initial kernel: a short Matern term of 4 A and a long one of
    85 A / 205 A), fitted with `objective.kernel_fit = false` and corr and disc off (an identity kernel), so the
    whitened residuals are the correlated noise itself and the whiteness check fails. One phase. Run "nomcmc": --no-mcmc
    with systematics on the best mode and a profile of incl_deg on the best family; run "mcmc": a capped MCMC on the best
    mode with a coverage run named in `uncertainty.calibration`, no systematics, no profiles."""
    root = tmp_path_factory.mktemp("red_data")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    white = 'kind = "full"\nkernel_fit = false\ncorr = false\ndisc = false\n'
    outs = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        truth = root / "truth.toml"
        truth.write_text(config_text("inj", kind="full", phases=(0.0,)).replace("windows = [[4300.0, 6700.0]]",
                                                                                  "windows = [[4400.0, 5200.0]]"))
        assert cli.main(["inject", str(truth), "-o", str(root / "inj"), "--noise", "full", "--seed", "7",
                         "--sigma-i", "0.01", "--sigma-vi", "0.002"]) == 0
        for name, unc, flags in (
                ("nomcmc", '[uncertainty]\npatches = 1000\nprofile_quantities = ["incl_deg"]\nprofile_max_steps = 4\n'
                           "profile_families = 1\n", ["--no-mcmc", "--sys-modes", "1"]),
                ("mcmc", '[uncertainty]\npatches = 1000\nensembles = 2\nprofile_quantities = []\n'
                         'calibration = "coverage_run"\n',
                 ["--mcmc-modes", "1", "--mcmc-evals", "2000", "--no-systematics"])):
            fit = root / f"fit_{name}.toml"
            txt = config_text("inj", kind="full", phases=(0.0,), extra=unc)
            fit.write_text(txt.replace("windows = [[4300.0, 6700.0]]", "windows = [[4400.0, 5200.0]]")
                           .replace('kind = "full"\n', white))
            outs[name] = root / f"out_{name}"
            assert cli.main(["fit", str(fit), "-o", str(outs[name]), "--seed", "2", "--starts", "3", *flags]) == 0
    return outs


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v4h_failed_whiteness_flags_every_output(red_runs):
    """V4h end to end (LIKE-10, UNC-4): with the whiteness check failed, every output that holds -ln L differences or
    intervals carries the flag and the failing spectra at its top level: modes.json, families.json, profiles.json,
    systematics.json, diagnostics.json and intervals.json, with and without the MCMC; a named coverage run gives
    "uncalibrated"; the report puts the flag on each table. Before the fix only intervals.json carried it, and only
    when the MCMC ran."""
    from zeemap.noise import FAIL_FLAG

    want = {"nomcmc": ["modes", "families", "profiles", "systematics", "diagnostics"],
            "mcmc": ["modes", "families", "diagnostics", "intervals"]}
    for name, out in red_runs.items():
        diag = json.loads((out / "diagnostics.json").read_text())
        assert diag["statistics"]["whiteness_pass"] is False
        bad = sorted(k for k, v in diag["statistics"]["whiteness"].items() if not v["pass"])
        assert bad
        for f in want[name]:
            d = json.loads((out / f"{f}.json").read_text())
            assert d.get("noise_model_flag") == FAIL_FLAG, (name, f)
            assert d.get("noise_model_failing_spectra") == bad, (name, f)
        assert (out / "intervals.json").exists() == (name == "mcmc")
        text = (out / "report.txt").read_text()
        assert text.count(f"[{FAIL_FLAG}: {', '.join(bad)}]") >= (4 if name == "nomcmc" else 3), text
    iv = json.loads((red_runs["mcmc"] / "intervals.json").read_text())
    assert iv["calibration"] == "uncalibrated"  # a named coverage run cannot give "calibrated" (UNC-4)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v4h_passed_whiteness_writes_no_flag(full_run):
    _, out = full_run
    for f in ("modes", "families", "diagnostics"):
        d = json.loads((out / f"{f}.json").read_text())
        assert "noise_model_flag" not in d and "noise_model_failing_spectra" not in d, f


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v2i_floor_chi2_of_a_mixture_is_zero(full_run):
    """Noise-free data from the forward model are a non-negative mixture of records: floor below 1e-8 x the chi2 of the
    zero model; the noisy data of the same truth are not, and their floor is above 0."""
    root, _ = full_run
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        clean = root / "clean.toml"
        clean.write_text(_text("clean"))
        p = Problem(load_config(clean))
        p_noisy = Problem(load_config(root / "truth.toml"))
    x = np.array(p.value, dtype=float)
    zero = sum(float(np.sum((np.asarray(s.data) / np.asarray(s.sigma)) ** 2)) for s in p.dataset.spectra)
    for v in floor_chi2(p, x).values():
        assert v["chi2_floor"] < 1e-8 * zero, v
        assert v["n_I"] == 401 and v["n_V"] == 401 and v["slices"] == [30000.0]
    for v in floor_chi2(p_noisy, x).values():
        assert v["chi2_floor"] > 1e-6 * zero, v  # (the mode's chi2 carries its continuum, the floor none)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7i_window_residual_table(full_run):
    root, out = full_run
    diag = json.loads((out / "diagnostics.json").read_text())
    rows = diag["statistics"]["window_residuals"]
    libs = {r["library"] for r in rows}
    assert libs == {"run", "library_1"}
    spectra = {r["spectrum"] for r in rows}
    # DIAG-3: the best mode of each family, each row tagged with its family and mode (families.json ids)
    fams = json.loads((out / "families.json").read_text())["families"]
    want = {(f["id"], next(n["mode"] for n in f["nodes"] if n["node"] == f["best"])) for f in fams}
    assert fams and {(r["family"], r["mode"]) for r in rows} == want
    assert len(spectra) == 4 and len(rows) == 2 * 4 * len(fams)  # one window per spectrum, two libraries
    for r in rows:
        assert r["n"] == 401 and r["chi2"] > 0
        if r["spectrum"].endswith("_I"):
            assert r["chi2_deg0"] is not None and r["chi2_deg0"] >= r["chi2"] * (1 - 1e-9)
        else:
            assert r["chi2_deg0"] == pytest.approx(r["chi2"], rel=1e-9)  # V/I has no continuum
    # the same library under two names gives the same numbers
    by = {(r["family"], r["library"], r["spectrum"]): r["chi2"] for r in rows}
    for f, _ in want:
        for s in spectra:
            assert by[(f, "run", s)] == pytest.approx(by[(f, "library_1", s)], rel=1e-9)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7i_window_residuals_of_noise_free_data_are_zero(full_run):
    """V7i, noise-free part: data made by the run's library at the truth give a whitened chi2 under the run's setting
    of 0 in every window (1e-8 relative to the data's sum of squares in units of the errors), for the run's library and an
    alternative that is the same library."""
    from zeemap.diagnostics import window_residuals

    root, _ = full_run
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        clean = root / "clean.toml"
        clean.write_text(_text("clean"))
        p = Problem(load_config(clean))
        q = Problem(load_config(clean))
    x = np.array(p.value, dtype=float)
    zero = {s.label: float(np.sum((np.asarray(s.data) / np.asarray(s.sigma)) ** 2)) for s in p.dataset.spectra}
    rows = window_residuals({"run": p, "library_1": q}, x)
    assert len(rows) == 2 * 4
    for r in rows:
        assert r["n"] == 401 and r["chi2"] <= 1e-8 * zero[r["spectrum"]], r


def test_v7i_family_best_points():
    """One parameter set per family, at its best node, with the family id and the node's mode id; no families: the
    best mode alone."""
    from types import SimpleNamespace

    from zeemap.diagnostics import family_best_points

    fams = {"families": [{"id": 1, "best": 2, "nodes": [{"node": 0, "mode": 1, "x": [0.0]},
                                                        {"node": 2, "mode": 3, "x": [2.0]}]},
                         {"id": 2, "best": 1, "nodes": [{"node": 1, "mode": 2, "x": [1.0]}]}]}
    got = family_best_points(fams, [])
    assert [(f, m, list(x)) for f, m, x in got] == [(1, 3, [2.0]), (2, 2, [1.0])]
    got = family_best_points(None, [SimpleNamespace(x=[5.0])])
    assert [(f, m, list(x)) for f, m, x in got] == [(None, 1, [5.0])]
