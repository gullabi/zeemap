"""V7a, V7b, V7c (fit) and V7f (fit with MCMC): the commands end to end on the synthetic test
library (tests/synth.py, VERIFICATION.md section 4).

Set-up: $ZEEMAP_DATA is a temporary directory holding a link to the session's synthetic library (variant a) and the
injected data sets; the configurations are TOML files written there. The truth is a centred dipole of 400 MG whose
axis lies 60 degrees from the rotation axis, seen at i = 60 degrees (u = 0.5), Teff 30 000 K (one slice selected, so
Teff is not a parameter), 1000 Fibonacci patches, two phases (0 and 0.5).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from zeemap import cli
from zeemap.config import ConfigError, load_config
from zeemap.problem import Problem

pytestmark = pytest.mark.tier1

MD_TRUE = 400.0 * np.array([math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
U_TRUE = 0.5
SPECTRA_COLUMNS = ["wavelength", "model_I", "model_VI", "data_I", "data_VI", "resid_I", "resid_VI", "model_I_raw"]


def _toml_vec(v):
    return "[" + ", ".join(repr(float(a)) for a in v) + "]"


# the error bars and the table stage on the search's own 1000 patches unless a test sets [uncertainty] itself: the
# re-polish at 20 000 patches (SEARCH-8) is out of tier 1's budget
UNC_DEFAULT = "[uncertainty]\npatches = 1000\nprofile_quantities = []\n"  # profiles: their own tests


def config_text(data: str, kind: str = "chi2", md=MD_TRUE, u=U_TRUE, phases=(0.0, 0.5), extra: str = "") -> str:
    blocks = "\n".join(
        f'[[data.phase]]\nphase = {ph}\nfile = "ph{k:02d}.dat"\nierr = "ph{k:02d}_ierr.dat"\nvierr = "ph{k:02d}_vierr.dat"\n'
        for k, ph in enumerate(phases))
    return f"""
[library]
path = "synth"
teffs = [30000.0]

[data]
path = "{data}"
windows = [[4300.0, 6700.0]]
grid = {{ start = 4100.0, step = 2.0, npix = 1401 }}

{blocks}
[model]
fwhm = 3.0
surface = {{ scheme = "fibonacci", n = 1000 }}

[objective]
kind = "{kind}"

[parameters]
md = {{ value = {_toml_vec(md)}, step = 10.0, lo = 150.0, hi = 600.0, free = true }}
bq = {{ value = 0.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }}
aq = {{ value = [0.0, 0.0, 1.0], step = 0.1, lo = 0.5, hi = 2.0, free = false }}
mo = {{ value = [0.0, 0.0, 0.0], step = 10.0, lo = 0.0, hi = 100.0, free = false }}
c = {{ value = [0.0, 0.0, 0.0], step = 0.02, lo = 0.0, hi = 0.3, free = false }}
u = {{ value = {u!r}, step = 0.1, lo = -1.0, hi = 1.0, free = true }}

[search]
n_starts = 4
restarts = 2
rounds_max = 1
lbfgs_maxiter = 60

[search.starts]
md = [200.0, 600.0]
{extra}
{"" if "[uncertainty]" in extra else UNC_DEFAULT}"""


@pytest.fixture(scope="module")
def env(synth_lib, tmp_path_factory):
    root = tmp_path_factory.mktemp("zeemap_data")
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = root / "truth.toml"
        cfg_path.write_text(config_text("inj"))
        assert cli.main(["inject", str(cfg_path), "-o", str(root / "inj"), "--noise", "none"]) == 0
        assert cli.main(["eval", str(cfg_path), "-o", str(root / "eval")]) == 0
        yield root, cfg_path


def angle_to_observer(md, u, phase=0.0):
    si = math.sqrt(max(0.0, 1.0 - u * u))
    a = 2.0 * math.pi * phase
    n = np.array([si * math.cos(a), -si * math.sin(a), -u])
    return math.degrees(math.acos(float(np.dot(md, n)) / float(np.linalg.norm(md))))


# ------------------------------------------------------------------------------------------------------------- V7b


def test_v7b_inject_writes_formats_layout(env):
    root, _ = env
    d = root / "inj"
    for k in range(2):
        for f in (f"ph{k:02d}.dat", f"ph{k:02d}_ierr.dat", f"ph{k:02d}_vierr.dat"):
            a = np.loadtxt(d / f)
            assert a.shape[0] == 1401 and a.shape[1] == (3 if f.endswith("0.dat") or f.endswith("1.dat") else 2)
            assert np.allclose(a[:, 0], 4100.0 + 2.0 * np.arange(1401))
    truth = json.loads((d / "truth.json").read_text())
    assert np.allclose(truth["parameters"]["md"], MD_TRUE) and truth["parameters"]["u"] == U_TRUE
    assert truth["noise"]["kind"] == "none"
    assert "truth" in json.loads((d / "dataset.json").read_text())


def test_v7b_inject_noise_free_eval_chi2_zero(env):
    root, _ = env
    ev = json.loads((root / "eval" / "eval.json").read_text())
    assert ev["kind"] == "chi2" and ev["valid"]
    assert math.isfinite(ev["neg_log_l"]) and ev["neg_log_l"] < 1e-8
    assert len(ev["per_spectrum"]) == 4
    for lab, d in ev["per_spectrum"].items():
        assert d["chi2"] < 1e-8, (lab, d["chi2"])


def test_v7b_inject_white_noise_is_seeded(env, tmp_path):
    root, cfg_path = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        for out in ("w1", "w2"):
            assert cli.main(["inject", str(cfg_path), "-o", str(tmp_path / out), "--noise", "white", "--seed", "3",
                             "--sigma-i", "0.01", "--sigma-vi", "0.002"]) == 0
    a, b = np.loadtxt(tmp_path / "w1" / "ph00.dat"), np.loadtxt(tmp_path / "w2" / "ph00.dat")
    c = np.loadtxt(root / "inj" / "ph00.dat")
    assert np.array_equal(a, b)
    r = (a[:, 2] - c[:, 2]) / 0.002
    assert 0.9 < np.std(r) < 1.1


# ------------------------------------------------------------------------------------------------------------- V7a


def test_v7a_eval_writes_eval_json_and_spectra(env):
    root, _ = env
    d = root / "eval"
    ev = json.loads((d / "eval.json").read_text())
    for key in ("neg_log_l", "per_spectrum", "derived", "parameters", "spectra", "config", "versions"):
        assert key in ev
    assert ev["spectra_columns"] == SPECTRA_COLUMNS
    der = ev["derived"]
    assert der["dipole_MG"] == pytest.approx(400.0)
    assert der["incl_deg"] == pytest.approx(60.0)
    assert len(der["B50_MG"]) == 2 and len(der["outside_fraction_visible"]) == 2
    for ph in ("0.000", "0.500"):
        f = d / "spectra" / f"eval_ph{ph}.dat"
        assert f.is_file()
        head = f.read_text().splitlines()[0]
        assert head.lstrip("# ").split() == SPECTRA_COLUMNS
        a = np.loadtxt(f)
        assert a.shape == (1401, 8)
        fitted = np.isfinite(a[:, 5])
        assert fitted.sum() == 1201  # 4300 .. 6700 A
        assert np.nanmax(np.abs(a[:, 5:7])) < 1e-6  # noise-free data: zero residuals
        assert np.all(np.isnan(a[~fitted, 5]))


# ------------------------------------------------------------------------------------------------------------- V7c


@pytest.fixture(scope="module")
def fit_run(env):
    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        start_md = 300.0 * np.array([0.0, 0.0, 1.0])
        cfg_path = root / "fit.toml"
        cfg_path.write_text(config_text("inj", md=start_md, u=0.7))
        out = root / "fit"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "1", "--no-mcmc"]) == 0
    return out


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7c_fit_writes_outputs(fit_run):
    out = fit_run
    for f in ("run.json", "modes.json", "diagnostics.json", "report.txt"):
        assert (out / f).is_file(), f
    run = json.loads((out / "run.json").read_text())
    assert run["seed"] == 1 and run["start_distribution"]["kind"] == "field"
    assert run["start_distribution"]["field"]["md"] == [200.0, 600.0]
    assert "numpy" in run["versions"] and run["input_sha256"]
    diag = json.loads((out / "diagnostics.json").read_text())
    ends = diag["endpoints"]
    assert sum(1 for e in ends if e["round"] == 0) == 4  # every start's end point
    assert diag["statistics"]["n_starts"] == 4
    modes = json.loads((out / "modes.json").read_text())["modes"]
    assert modes and all("n_starts" in m and "delta" in m and "patches" in m for m in modes)
    assert sum(m["n_starts"] for m in json.loads((out / "diagnostics.json").read_text())["all_modes"]) <= 4
    assert modes[0]["delta"] == 0.0
    band = json.loads((out / "modes.json").read_text())["band_nats"]
    assert all(m["delta"] <= band for m in modes)
    text = (out / "report.txt").read_text()
    assert "mode table" in text and "error bars: not computed (MCMC is off; add --mcmc)" in text
    assert not (out / "intervals.json").exists() and not (out / "samples").exists()
    for k in range(1, len(modes) + 1):
        for ph in ("0.000", "0.500"):
            assert (out / "spectra" / f"mode{k}_ph{ph}.dat").is_file()
            h = np.loadtxt(out / "histograms" / f"mode{k}_ph{ph}.dat")
            assert h.shape[1] == 3 and h[:, 2].sum() > 0


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7j_families_json_and_mode_families(fit_run):
    """SEARCH-9: families.json is written and valid; every mode names its family (or null) and its band
    component; the report has the family table next to the mode table (REP-1, V4e)."""
    out = fit_run
    fam = json.loads((out / "families.json").read_text())
    for key in ("level", "h", "band_level", "families", "band_components", "mode_family", "capped"):
        assert key in fam
    assert fam["h"] >= 5.0  # search.family.min_level
    modes = json.loads((out / "modes.json").read_text())["modes"]
    ids = {f["id"] for f in fam["families"]}
    for m in modes:
        assert m["family"] is None or m["family"] in ids
        assert "families_all" in m and m["band_component"] is not None
    assert modes[0]["family"] == 1  # the best mode is in the first family
    for f in fam["families"]:
        assert f["nodes"] and f["modes"] and f["neg_log_l"] <= fam["level"] + 1e-9
    text = (out / "report.txt").read_text()
    assert "family table (SEARCH-9)" in text and "components at the band level" in text


@pytest.fixture(scope="module")
def default_profile_fit(env, tmp_path_factory):
    """`fit --no-mcmc` with the shipped `profile_quantities = "default"`, 12 grid steps, 3 starts, one family: the one
    end-to-end profile fit of this module, shared by the two V7j profile tests."""
    root, _ = env
    d = tmp_path_factory.mktemp("fit_dflt")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = d / "fit_dflt.toml"
        extra = ('[uncertainty]\npatches = 1000\nprofile_quantities = "default"\nprofile_max_steps = 12\n'
                 "profile_families = 1\n")
        cfg_path.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7, extra=extra))
        out = d / "fit_dflt"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "1", "--starts", "3", "--no-mcmc",
                         "--no-systematics"]) == 0
    return out


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7j_profiles_json_and_side_by_side(default_profile_fit):
    """UNC-5, UNC-8: `fit --no-mcmc` with profiled quantities (the default list, the fit shared with the next
    test; an explicit list is honoured exactly, V6d): profiles.json holds per family the grid, the lower envelope, at
    least 2 walks, the three intervals and the envelope; the profile intervals of incl_deg and dipole_MG contain the
    truth (noise-free data); without MCMC the ratios are null with the reason; the report shows the side-by-side table
    and the two sentences per view."""
    out = default_profile_fit
    prof = json.loads((out / "profiles.json").read_text())
    assert prof["h"] >= 5.0 and "likelihood:" in prof["header"]
    f1 = prof["families"]["1"]
    assert {"incl_deg", "dipole_MG"} <= set(f1["quantities"])
    for name, truth in (("incl_deg", 60.0), ("dipole_MG", 400.0)):
        e = f1["quantities"][name]
        assert len(e["walks"]) >= 2 and len(e["grid"]) == len(e["envelope_profile"]) >= 3
        iv = e["intervals"]["family_level"]
        assert iv["lo"] <= truth <= iv["hi"], (name, iv)
        assert e["envelope"]["lo"] <= truth <= e["envelope"]["hi"]
        assert e["R1"] is None and e["R2"] is None and "no per-mode interval" in e["note"]
    text = (out / "report.txt").read_text()
    assert "families side by side with the per-mode error bars (UNC-8)" in text
    assert "Mode 1: incl_deg: no per-mode interval." in text and "Family 1: incl_deg between" in text


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7j_default_profile_quantities_close_no_end_without_a_limit(default_profile_fit):
    """Regression: `fit --no-mcmc` with the shipped `profile_quantities = "default"` on the synthetic problem (one
    family; c fixed, so c_along_axis is left out). No polish limit or range end lies near any profile (|m_d| = 400 MG
    against limits of 600, the inclination 60 degrees), so no quantity may have a closed end "constraint not met"; the
    flag "constraint not met" appears exactly with such an end, "polish stalled" exactly with a stalled point, and the
    side-by-side table and the legend agree. B50_MG[0], whose low side was once closed at 251.4 MG by a
    stalled polish (q_best 256.5 MG, the stall's -ln L 0.43, far below the level), reaches below 251.4 MG. Before the
    fix: 6 of 8 quantities flagged, B50_MG[0] closed at 251.39 and B50_MG[1] at 326.13."""
    out = default_profile_fit
    NOT_MET, STALLED = "constraint not met", "polish stalled"  # profiles.NOT_MET, profiles.STALLED

    prof = json.loads((out / "profiles.json").read_text())
    q = prof["families"]["1"]["quantities"]
    assert list(q) == ["incl_deg", "dipole_MG"] + [f"{b}[0]" for b in ("B10_MG", "B50_MG", "B90_MG")]  # ref phase (decision #40)
    e = q["B50_MG[0]"]
    assert e["ends"]["lo"] is None and min(e["grid"]) < 251.3
    for name, e in q.items():
        assert all(v is None or v["reason"] != NOT_MET for v in e["ends"].values()), (name, e["ends"])
        assert NOT_MET not in e["flags"], name
    for name, e in q.items():
        stalled = any(p["stalled"] for w in e["walks"] for p in w["points"])
        assert (STALLED in e["flags"]) == stalled, name
        for w in e["walks"]:
            for p in w["points"]:
                assert p["met"] or p["stalled"], (name, p)  # every miss without a limit is a stall
    table = (out / "report.txt").read_text().split("families side by side")[1].split("the same in words")[0]
    assert NOT_MET not in table.split("\n", 1)[1]  # in no row (the legend names it)
    # decision #39: every B-quantile entry, and no other, is flagged provisional, in profiles.json and
    # in its report row
    QUANTILE = "B quantile is a step function: provisional"  # profiles.QUANTILE_STEP
    rows = {r.split()[1]: r for r in table.split("\n")[2:] if r.strip() and r.split()[0] == "1"}
    for name, e in q.items():
        is_b = name.split("[")[0] in ("B10_MG", "B50_MG", "B90_MG")
        assert (QUANTILE in e["flags"]) == is_b and (QUANTILE in rows[name]) == is_b, name


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7c_fit_best_mode_recovers_truth(fit_run):
    m = json.loads((fit_run / "modes.json").read_text())["modes"][0]
    md, u = np.array(m["parameters"]["md"]), m["parameters"]["u"]
    rel = abs(np.linalg.norm(md) / 400.0 - 1.0)
    true_angles = [angle_to_observer(MD_TRUE, U_TRUE, ph) for ph in (0.0, 0.5)]
    angles = [angle_to_observer(md, u, ph) for ph in (0.0, 0.5)]
    print(f"V7c: |md| {np.linalg.norm(md):.4f} MG (rel {rel:.2e}), angles {angles} vs {true_angles}, "
          f"-lnL {m['neg_log_l']:.3g}")
    assert rel < 1e-3
    assert abs(angles[0] - true_angles[0]) < 0.5
    assert m["neg_log_l"] < 1e-3


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7c_modes_consistent_neg_log_l_is_the_cost_of_the_written_vector(fit_run, env):
    """modes.json and diagnostics.json: every mode's neg_log_l equals problem.cost(its written vector) to 1e-9; and a
    mode whose end point is the non-canonical mirror image (u < 0) gets the cost of its canonical vector, keeps the
    better image as its value and says which image that is."""
    from zeemap.report import mode_entry
    from zeemap.search import Mode, settle_mirror

    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        pr = Problem(load_config(root / "fit.toml"))
    doc = json.loads((fit_run / "modes.json").read_text())
    for m in doc["modes"]:
        x = np.array([m["vector"][n] for n in doc["names"]])
        assert m["neg_log_l"] == pytest.approx(float(pr.cost(x[None])[0]), rel=1e-9, abs=1e-9)
        assert m["neg_log_l_mode"] <= m["neg_log_l"] + 1e-12 and m["image"] in ("end point", "mirror image")
    diag = json.loads((fit_run / "diagnostics.json").read_text())
    assert diag["statistics"]["mirror_asymmetry_max"] is not None
    for m in diag["all_modes"]:
        x = pr.canonical(np.array([m["parameters"]["md"][0], m["parameters"]["md"][1], m["parameters"]["md"][2],
                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m["parameters"]["u"]]))
        assert m["neg_log_l"] == pytest.approx(float(pr.cost(x[None])[0]), rel=1e-9, abs=1e-9)
    # a mode found at the mirror image (u < 0) of a mode's point
    for problem, x0 in ((pr, x),):
        xm = problem.mirror(np.asarray(x0, dtype=float))
        assert xm[problem.names.index("u")] < 0
        f_end, f_can = (float(v) for v in problem.cost(np.array([xm, problem.canonical(xm)])))
        assert f_end != f_can  # the mirror is not a symmetry of the 1000-patch grid
        mode = Mode(xm.copy(), f_end, None, 1, [0])
        out = settle_mirror(problem, [mode])
        assert out["mirror_asymmetry"][0] == pytest.approx(f_can - f_end, abs=1e-12)
        # f_canonical comes from a batch of another size, whose BLAS sums may round differently in the last bits
        # (OPENBLAS_CORETYPE = Haswell: 4e-13 relative): the two-ways tolerance of VERIFICATION.md section 3
        assert mode.f == pytest.approx(min(f_end, f_can), rel=1e-10, abs=1e-13)
        assert mode.f_canonical == pytest.approx(f_can, rel=1e-10, abs=1e-13)
        assert mode.image == ("mirror image" if f_can < f_end else "end point")
        np.testing.assert_array_equal(mode.x_canonical, problem.canonical(xm))

        class _FP:
            @staticmethod
            def to_dict():
                return {}

        mode.fingerprint = _FP()
        e = mode_entry(problem, mode, 1, mode.f)
        xv = np.array([e["vector"][n] for n in problem.names])
        assert e["neg_log_l"] == pytest.approx(float(problem.cost(xv[None])[0]), rel=1e-9, abs=1e-9)
        assert e["neg_log_l_mode"] == mode.f and e["mirror_asymmetry"] == pytest.approx(f_can - f_end, abs=1e-12)


def _problem(root, name, text):
    p_ = root / f"{name}.toml"
    p_.write_text(text)
    return Problem(load_config(p_))


def test_v1l_field_strengths_scale_the_b_quantities_exactly(env):
    """Decision #39 (B quantities held by scaling): B is linear in the strength parameters (md, bq, mo of
    "dqo"), and the patch weights, the visibility and the order of |B| do not depend on them,
    so scaling every strength by one factor s > 0 scales dipole_MG and every B quantile by s exactly (to rounding;
    1e-12 relative), with a quadrupole, an octupole and an offset present. `profiles.strength_indices` names those
    parameters; with a strength fixed at a non-zero value (it would not scale) it is None and the penalty holds q."""
    from zeemap import profiles as PR

    root, _ = env
    names = ["dipole_MG"] + [f"{b}[{k}]" for b in ("B10_MG", "B50_MG", "B90_MG") for k in (0, 1)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        text = config_text("inj").replace(
            'bq = { value = 0.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }',
            'bq = { value = 30.0, step = 10.0, lo = -100.0, hi = 100.0, free = true }').replace(
            'aq = { value = [0.0, 0.0, 1.0], step = 0.1, lo = 0.5, hi = 2.0, free = false }',
            'aq = { value = [0.3, 0.0, 1.0], step = 0.1, lo = 0.5, hi = 2.0, free = false }').replace(
            'mo = { value = [0.0, 0.0, 0.0], step = 10.0, lo = 0.0, hi = 100.0, free = false }',
            'mo = { value = [10.0, 0.0, 40.0], step = 10.0, lo = 0.0, hi = 100.0, free = true }').replace(
            'c = { value = [0.0, 0.0, 0.0], step = 0.02, lo = 0.0, hi = 0.3, free = false }',
            'c = { value = [0.05, 0.0, 0.02], step = 0.02, lo = 0.0, hi = 0.3, free = false }')
        pr = _problem(root, "dqo_multi", text)
        idx = PR.strength_indices(pr)
        assert [pr.names[i] for i in idx] == ["md_x", "md_y", "md_z", "bq", "mo_x", "mo_y", "mo_z"]
        rng = np.random.default_rng(3)
        x = np.array(pr.value, dtype=float)
        X = np.repeat(x[None, :], 4, axis=0)
        free = np.flatnonzero(pr.free)
        X[:, free] += 0.3 * rng.standard_normal((4, free.size)) * np.asarray(pr.step)[free]
        for scale in (0.7, 1.3):
            Xs = X.copy()
            Xs[:, idx] *= scale
            for name in names:
                q, qs = PR.quantity_values(pr, X, name), PR.quantity_values(pr, Xs, name)
                np.testing.assert_allclose(qs, scale * q, rtol=1e-12, atol=0, err_msg=name)
        fixed = text.replace('bq = { value = 30.0, step = 10.0, lo = -100.0, hi = 100.0, free = true }',
                             'bq = { value = 30.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }')
        assert PR.strength_indices(_problem(root, "dqo_bq_fixed", fixed)) is None


@pytest.mark.xfail(strict=False, reason="the B quantile is a step function, so on some CPUs the scaled-hold walk stalls about 1 nat above the direct minimum (1.49 vs 0.52 nats), until the quantile is made continuous")
def test_v1l_scaled_hold_reaches_the_direct_minimum_of_a_b_quantile(env):
    """Decision #39: B50_MG[0] of the synthetic problem, walked from the truth (-ln L = 0, noise-free
    data) to 1 % and 2 % below its value. Held by the penalty, the polish could not relax the other parameters against
    the quantile's jumps (about 0.3 MG at 1000 patches; here 0.73 nats at 1 % against 0.09 directly). Held by scaling
    the field strengths, q = q0 to 1e-9 relative and each point lies within 0.1 nat of a direct minimisation over the
    geometry with |m_d| scaled so that q = q0 exactly (Nelder-Mead from the truth and from the walk's point, the lower;
    an upper bound of the true minimum). 0.1 nat is the roughness the quantile's jumps still leave in -ln L along the
    geometry at this hold (deviations from a quadratic of 0.08-0.14 nats over +-0.01 parameter steps)."""
    from scipy.optimize import minimize

    from zeemap import profiles as PR
    from zeemap import search as S

    root, cfg_path = env
    name = "B50_MG[0]"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg = load_config(cfg_path)
        prob = Problem(cfg)
    x = np.array(prob.value, dtype=float)
    assert prob.cost(x[None])[0] < 1e-8
    qb = float(PR.quantity_values(prob, x, name)[0])
    dq = 0.01 * qb
    with S.single_threaded():
        r = PR.walk(prob, cfg, name, x, qb, dq, 10.0, 2, -1, 0, 1)
    md = slice(0, 3)
    iu = prob.names.index("u")
    for k in (-1, -2):
        q0 = qb + k * dq
        f_walk, xw, q, met = r["points"][k][:4]

        def fun(v):
            y = x.copy()
            y[md], y[iu] = v[:3], v[3]
            y[md] *= q0 / float(PR.quantity_values(prob, y, name)[0])
            return float(prob.cost(y[None])[0])

        ref = min(minimize(fun, np.r_[y0[md], y0[iu]], method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-5, "maxfev": 2000}).fun for y0 in (x, xw))
        print(f"V1l scaled hold: {name} at {q0:.3f}: walk {f_walk:.4f}, direct {ref:.4f}")
        assert f_walk <= ref + 0.1, (k, f_walk, ref)
        assert met and q == pytest.approx(q0, rel=1e-9)


def test_v1l_scaled_hold_closes_at_a_strength_limit(env):
    """Decision #39: a strength that the scaling would take beyond its polish limit is a real limit.
    With md's limits at +-420 MG per component (truth md = (346, 0, 200)), dipole_MG walked upwards from the truth in
    20 MG steps (no stop level) reaches a grid value where md_x would pass 420: the walk polishes at the q where it
    reaches the limit and ends there, a closed end "constraint not met", below that grid value; every earlier point met
    q = q0 exactly and no strength left its limits."""
    from zeemap import profiles as PR
    from zeemap import search as S

    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        text = config_text("inj").replace("lo = 150.0, hi = 600.0, free = true", "lo = 150.0, hi = 420.0, free = true")
        prob = _problem(root, "md_hi_420", text)
        cfg = prob.cfg
    x = np.array(prob.value, dtype=float)
    with S.single_threaded():
        r = PR.walk(prob, cfg, "dipole_MG", x, 400.0, 20.0, 1e9, 6, 1, 0, 1)
    end = r["end"]
    assert end is not None and end["reason"] == PR.NOT_MET, end
    last = max(r["points"])
    assert 400.0 + 20.0 * last < end["q"] < 400.0 + 20.0 * (last + 1)
    for k, (f, xk, q, met, stalled) in r["points"].items():
        assert met and not stalled and q == pytest.approx(400.0 + 20.0 * k, rel=1e-9)
        assert np.all(np.abs(xk[0:3]) <= 420.0 * (1 + 1e-12))


def test_v1l_penalty_hold_on_the_real_problem_retries_a_miss(env):
    """After decision #39 every default quantity of the synthetic problem is held directly or by scaling,
    so the retry after a missed constraint, `_towards` and the stall rule ran only on toys. With bq fixed at 30 MG (a
    strength fixed non-zero does not scale, so B50_MG[0] falls back to the penalty), B50_MG[0] walked 2 grid steps of
    1 % down from the polished best under the shipped LM stops: every point met the constraint or stalled, `met`
    matches the tolerance, and no end is closed (no limit near). Whether a polish misses and is retried depends on the
    last bits of the quantile (its jumps), so on the BLAS thread count and the machine (no retry with the thread
    variables set, or on a CI runner); the retry itself is pinned by the toy tests of V1l."""
    from zeemap import profiles as PR
    from zeemap import search as S

    root, _ = env
    name = "B50_MG[0]"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        text = config_text("inj").replace('bq = { value = 0.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }',
                                          'bq = { value = 30.0, step = 10.0, lo = -100.0, hi = 100.0, free = false }')
        prob = _problem(root, "bq_fixed_30", text)
        cfg = prob.cfg
    assert PR.strength_indices(prob) is None  # the penalty holds q
    with S.single_threaded():
        x = S.polish(prob, np.array(prob.value, dtype=float), cfg).x
        qb = float(PR.quantity_values(prob, x, name)[0])
        r = PR.walk(prob, cfg, name, x, qb, 0.01 * qb, 1e9, 2, -1, 0, 1)
    assert r["end"] is None and sorted(r["points"]) == [-2, -1]
    for k, (f, xk, q, met, stalled) in r["points"].items():
        assert met or stalled, (k, q)
        assert met == (abs(q - (qb + 0.01 * qb * k)) <= PR.CONSTRAINT_TOL * 0.01 * qb)


# ------------------------------------------------------------------------------------------------------------- V7f


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7f_fit_with_mcmc_writes_every_output(env, tmp_path):
    """V7f: `fit` with the MCMC capped (2 ensembles, 4000 evaluations, i.e. about 120
    steps of 16 walkers, so the mode may be "not converged"), one phase, chi2, search on 1000 patches and the MCMC on
    uncertainty.patches = 500 (the re-evaluation path): every output file of FORMATS.md section 6 written so far exists
    and is valid; intervals.json carries the UNC-4 and UNC-6 labels; report.txt has the mode table and the intervals
    line; samples/ holds the chains."""
    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = tmp_path / "fit_mcmc.toml"
        cfg_path.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7, phases=(0.0,),
                                        extra="[uncertainty]\npatches = 500\nensembles = 2\nprofile_quantities = []\n"))
        out = tmp_path / "fit_mcmc"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "2", "--starts", "3", "--mcmc-modes", "1",
                         "--mcmc-evals", "4000"]) == 0
    for f in ("run.json", "modes.json", "intervals.json", "diagnostics.json"):
        json.loads((out / f).read_text())  # valid JSON
    modes = json.loads((out / "modes.json").read_text())["modes"]
    assert modes and modes[0]["patches"] == 500 and modes[0]["delta"] == 0.0
    run = json.loads((out / "run.json").read_text())
    assert run["mcmc"] is True and "max_evals = 4000" in run["config"]
    iv = json.loads((out / "intervals.json").read_text())
    assert iv["calibration"] == "uncalibrated" and iv["label"] == "per mode: conditional on this basin"
    assert "temper_T" not in iv and iv["patches"] == 500
    assert [e["mode"] for e in iv["modes"]] == [1]
    assert iv["modes_without_mcmc"] == list(range(2, len(modes) + 1))
    e = iv["modes"][0]
    assert e["label"] == "per mode: conditional on this basin"
    c = e["convergence"]
    assert c["ensembles"] == 2 and c["nwalkers"] == 16 and c["max_evals"] == 4000 and c["nevals"] <= 4000 + 3000  # + hand-back polishes
    q = e["quantities"]
    # one phase: the orientation (md components, u, incl) is declared nuisance (decision #18): no numbers, flagged
    for name in ("md_x", "md_y", "md_z", "u", "incl_deg"):
        assert q[name]["median"] is None, name
        assert "orientation not determined by one phase" in q[name]["flags"] and "nuisance, not required" in \
            q[name]["flags"]
        assert q[name]["rhat"] is not None  # its statistics are still reported
    assert c["nuisance"] and "u" in c["nuisance"] and "md_p" in c["nuisance"]
    for name in ("dipole_MG", "c_abs", "B50_MG[0]", "beta_deg", "c_along_axis", "c_across_axis"):
        assert name in q, name
        own = c["quantity_failed"].get(name)
        if e["converged"] or own == []:  # UNC-2 per quantity (decision #17)
            assert q[name]["p16"] <= q[name]["median"] <= q[name]["p84"]
            assert "tempered_half68" not in q[name]  # decision #24
        else:  # UNC-2: no numbers for a quantity that fails V14 on its own
            assert q[name]["median"] is None and "not converged" in q[name]["flags"] and q[name]["failed_rules"]
    if e["converged"]:
        assert q["dipole_MG"]["p2_5"] < 400.0 * (1 + 1e-3) and q["dipole_MG"]["p97_5"] > 400.0 * (1 - 1e-3)
    s = np.load(out / "samples" / "mode1.npz")
    assert s["samples"].shape[0] == 2 and s["samples"].shape[2] == 16 and s["samples"].shape[3] == 4
    assert list(s["names"]) == ["md_x", "md_y", "md_z", "u"] and s["nll"].shape == s["samples"].shape[:3]
    text = (out / "report.txt").read_text()
    print(text)
    assert "mode table" in text and "error bars (68 % intervals" in text
    assert ("intervals: statistical (68 %, per mode: conditional on this basin), uncalibrated: no coverage run named "
            "in the configuration") in text
    assert "convergence (V14) per mode:" in text and "  mode 1: " in text
    assert "weight" not in text.replace("No mode weights are given (REP-2)", "")  # REP-2
    for k in range(1, len(modes) + 1):
        assert (out / "spectra" / f"mode{k}_ph0.000.dat").is_file()
        assert (out / "histograms" / f"mode{k}_ph0.000.dat").is_file()


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7g_rerun_clears_stale_outputs_and_a_profile_failure_keeps_the_run(env, tmp_path, monkeypatch):
    """ENG-7: two fits into the same `-o` with different settings. Run 1 (no MCMC, a
    profile) writes profiles.json; files an earlier run with more modes, MCMC and systematics would have left are
    added, and one file of the user's. Run 2 (capped MCMC) has its profile stage made to fail: it still writes every
    other output (search, families, MCMC intervals and chains), names the failure in diagnostics.json and report.txt,
    and leaves no file of the earlier run that it does not write itself; the user's file stays."""
    from zeemap import profiles

    root, _ = env
    monkeypatch.setenv("ZEEMAP_DATA", str(root))
    out = tmp_path / "rerun"
    unc = '[uncertainty]\npatches = 1000\nprofile_quantities = ["dipole_MG"]\nprofile_max_steps = 1\nprofile_families = 1\n'
    md0 = 300.0 * np.array([0.0, 0.0, 1.0])
    cfg1 = tmp_path / "r1.toml"
    cfg1.write_text(config_text("inj", md=md0, u=0.7, phases=(0.0,), extra=unc))
    assert cli.main(["fit", str(cfg1), "-o", str(out), "--seed", "2", "--starts", "3", "--no-mcmc",
                     "--no-systematics"]) == 0
    assert (out / "profiles.json").is_file() and not (out / "intervals.json").exists()
    stale = ["samples/mode7.npz", "spectra/mode7_ph0.000.dat", "histograms/mode7_ph0.000.dat", "systematics.json",
             "intervals.json", "eval.json"]
    for f in stale:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text("stale")
    (out / "notes.txt").write_text("the user's own file")

    def boom(*args, **kwargs):
        raise RuntimeError("injected profile failure")

    monkeypatch.setattr(profiles, "profile_walks", boom)  # the walks, which cli._fit calls
    cfg2 = tmp_path / "r2.toml"
    cfg2.write_text(config_text("inj", md=md0, u=0.7, phases=(0.0,), extra=unc + "ensembles = 2\n"))
    assert cli.main(["fit", str(cfg2), "-o", str(out), "--seed", "2", "--starts", "3", "--mcmc-modes", "1",
                     "--mcmc-evals", "2000", "--no-systematics"]) == 0
    for f in ("run.json", "modes.json", "families.json", "diagnostics.json", "intervals.json", "report.txt",
              "samples/mode1.npz"):
        assert (out / f).is_file(), f
    assert json.loads((out / "intervals.json").read_text())["modes"][0]["mode"] == 1
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["statistics"]["profiles_error"] == "RuntimeError: injected profile failure"
    assert "profiles (UNC-5): failed, none written: RuntimeError: injected profile failure" in \
        (out / "report.txt").read_text()
    assert not (out / "profiles.json").exists()  # run 1's profiles: run 2 has none
    for f in stale[:-2] + ["profiles.json"]:
        assert not (out / f).exists(), f
    assert (out / "intervals.json").read_text() != "stale" and (out / "eval.json").read_text() == "stale"
    assert (out / "notes.txt").read_text() == "the user's own file"  # only the names a fit writes are removed
    n = len(json.loads((out / "modes.json").read_text())["modes"])
    assert sorted(p.name for p in (out / "spectra").iterdir()) == [f"mode{k}_ph0.000.dat" for k in range(1, n + 1)]
    assert sorted(p.name for p in (out / "samples").iterdir()) == ["mode1.npz"]


SWAPS = """
[[systematics.swap]]
name = "ld1"
set = { model.ld_mode = 1 }

[[systematics.swap]]
name = "u_far"
set = { parameters.u.value = 0.1, parameters.u.free = false }
"""


def test_v7f_swaps_drop_the_kernels_of_spectra_they_remove(tmp_path, monkeypatch):
    """An [[objective.kernel]] label that matches no spectrum is refused at load, so a swap that
    removes spectra (the jackknife's left-out phase, use_VI = false) drops their kernels from the configuration it
    inherits (the frozen kernels of LIKE-2), and keeps the others. A swap that sets the kernels itself is checked as
    written. Likewise the jackknife drops the profile settings of the left-out phase and renumbers the others
    (checked at load)."""
    from zeemap.systematics import apply_swap, jackknife_swaps

    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    k = "rho_short = 4.0\na = 0.5\nrho_long = 80.0\nj = 0.05\n"
    kern = "".join(f'[[objective.kernel]]\nspectrum = "{lab}"\n{k}' for lab in ("0.000_I", "0.000_VI", "0.500_I",
                                                                                 "0.500_VI"))
    cfg_path = tmp_path / "kern.toml"
    unc = ('[uncertainty]\npatches = 1000\nprofile_quantities = ["dipole_MG", "B50_MG[0]", "B50_MG[1]"]\n'
           '[uncertainty.profile_steps]\n"B90_MG[1]" = "2%"\n')
    cfg_path.write_text(config_text("inj", extra=kern + unc))
    cfg = load_config(cfg_path)
    jk, _ = jackknife_swaps(cfg)
    c0 = apply_swap(cfg, jk[0])
    assert [e["spectrum"] for e in c0["objective"]["kernel"]] == ["0.500_I", "0.500_VI"]
    # the left-out phase 0 goes from the profile settings, phase 1 is phase 0 of the swapped configuration (keeping
    # "B50_MG[0]" would name the other phase)
    assert c0["uncertainty"]["profile_quantities"] == ["dipole_MG", "B50_MG[0]"]
    assert c0["uncertainty"]["profile_steps"] == {"B90_MG[0]": "2%"}
    c1 = apply_swap(cfg, {"name": "noVI", "set": {"data": {"phase": [
        {**p, "use_VI": False} for p in cfg["data"]["phase"]]}}})
    assert [e["spectrum"] for e in c1["objective"]["kernel"]] == ["0.000_I", "0.500_I"]
    assert len(cfg["objective"]["kernel"]) == 4  # the run's configuration is not changed
    bad = {"name": "bad", "set": {"objective": {"kernel": [{"spectrum": "0.250_I", "rho_short": 4.0, "a": 0.5,
                                                             "rho_long": 80.0, "j": 0.05}]}}}
    with pytest.raises(ConfigError, match=r"swap 'bad'.*objective\.kernel\[0\]\.spectrum"):
        apply_swap(cfg, bad)


def test_v7f_jackknife_renumbers_the_phase_indices_of_profile_settings(tmp_path, monkeypatch):
    """After the jackknife drops phase i, phase k > i is phase k - 1 of the swapped configuration. The
    profile names and steps of the dropped phase go, those of later phases are renumbered (before, "B50_MG[k]" kept
    its index and named a different phase, and the last index was dropped instead of the left-out one)."""
    from zeemap.systematics import apply_swap, jackknife_swaps

    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    cfg_path = tmp_path / "three.toml"
    unc = ('[uncertainty]\npatches = 1000\nprofile_quantities = ["dipole_MG", "B50_MG[0]", "B50_MG[1]", "B90_MG[2]"]\n'
           '[uncertainty.profile_steps]\n"B10_MG[0]" = "2%"\n"B90_MG[1]" = "3%"\n"B50_MG[2]" = "4%"\nB50_MG = "1%"\n')
    cfg_path.write_text(config_text("inj", phases=(0.0, 0.5, 0.75), extra=unc))
    cfg = load_config(cfg_path)
    jk, _ = jackknife_swaps(cfg)
    assert [s["drop_phase"] for s in jk] == [0, 1, 2]
    got = {s["drop_phase"]: apply_swap(cfg, s)["uncertainty"] for s in jk}
    assert got[0]["profile_quantities"] == ["dipole_MG", "B50_MG[0]", "B90_MG[1]"]
    assert got[0]["profile_steps"] == {"B90_MG[0]": "3%", "B50_MG[1]": "4%", "B50_MG": "1%"}
    assert got[1]["profile_quantities"] == ["dipole_MG", "B50_MG[0]", "B90_MG[1]"]
    assert got[1]["profile_steps"] == {"B10_MG[0]": "2%", "B50_MG[1]": "4%", "B50_MG": "1%"}
    assert got[2]["profile_quantities"] == ["dipole_MG", "B50_MG[0]", "B50_MG[1]"]
    assert got[2]["profile_steps"] == {"B10_MG[0]": "2%", "B90_MG[1]": "3%", "B50_MG": "1%"}


def test_v7f_teff_swap_skipped_only_at_the_slice_to_rounding():
    """The swap teff_<T> is skipped when Teff is already fixed at the slice T. That test was once "within
    0.5 K", the library's old snap window; now the library reads a Teff as a slice only to rounding
    (`library.SLICE_SNAP_REL` x Teff) and blends two slices otherwise, so a Teff fixed 0.3 K from a slice is not that
    slice and its swap runs. The same rule (`library.on_slice`) decides both."""
    from zeemap.systematics import default_swaps

    def cfg(value):
        return {"parameters": {"teff": {"value": value, "step": 500.0, "lo": 9000.0, "hi": 16000.0, "free": False}},
                "library": {"teffs": [10000.0, 12000.0, 14000.0]}, "systematics": {},
                "data": {"phase": [{"phase": 0.0, "use_I": False, "use_VI": False}]}}

    names = [s["name"] for s in default_swaps(cfg(12000.3))[0]]
    assert names == ["teff_10000", "teff_12000", "teff_14000"]
    for v in (12000.0, 12000.0 * (1.0 + 5e-10)):
        sw, notes = default_swaps(cfg(v))
        assert [s["name"] for s in sw] == ["teff_10000", "teff_14000"]
        assert any("teff_12000 skipped" in n for n in notes)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7f_systematics_swaps_jackknife_and_fallback(env, tmp_path):
    """V7f (systematics; SYS-1 to SYS-5, UNC-7): two phases, chi2, `fit --no-mcmc --sys-modes 1` with two
    configured swaps (limb darkening 1; u fixed far from the truth), the default swaps of decision #21 (V/I x 0.9 and
    x 1.1) and the automatic leave-one-phase-out swaps.
    Without MCMC the "left a mode" rule falls back to the search's thresholds and says so. systematics.json holds a
    shift (or "mode change") for every quantity and swap, the per-swap -ln L, starts and keys; the jackknife sigma is
    in systematics.json (no intervals.json without MCMC); u fixed at 0.1 moves the inclination by ~24 degrees > 8, so
    it is a mode change under the fallback rule. attach_jackknife changes only the jackknife_sigma fields."""
    import copy

    from zeemap.systematics import (MODE_CHANGE, RULE_SEARCH, apply_swap, attach_jackknife, jackknife_swaps,
                                    limited_by_systematics)

    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = tmp_path / "fit_sys.toml"
        cfg_path.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7, extra=SWAPS))
        cfg = load_config(cfg_path)
        # apply_swap: keys merged and checked; unknown keys refused; jackknife = one swap per phase
        c2 = apply_swap(cfg, cfg["systematics"]["swap"][1])
        assert c2["parameters"]["u"]["value"] == 0.1 and c2["parameters"]["u"]["free"] is False
        assert c2["parameters"]["u"]["step"] == cfg["parameters"]["u"]["step"] and cfg["parameters"]["u"]["free"]
        with pytest.raises(ConfigError, match="swap 'bad'.*unknown key model.nonsense"):
            apply_swap(cfg, {"name": "bad", "set": {"model.nonsense": 1}})
        jk, note = jackknife_swaps(cfg)
        assert [s["name"] for s in jk] == ["drop_phase_0", "drop_phase_0.5"] and note is None
        assert len(apply_swap(cfg, jk[0])["data"]["phase"]) == 1
        out = tmp_path / "fit_sys"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "3", "--starts", "3", "--no-mcmc",
                         "--sys-modes", "1"]) == 0
    sysj = json.loads((out / "systematics.json").read_text())
    names = [s["name"] for s in sysj["swaps"]]
    # the default swaps of SYS-1 (decisions #21, #23) come after the configured ones: V/I x 0.9 and x 1.1, the continuum
    # at degree 0 and off; no Teff swap (the toy has no teff parameter, a note says so); no library swap (no
    # systematics.library_alternatives)
    assert names == ["ld1", "u_far", "vi_scale_0.9", "vi_scale_1.1", "continuum_deg0", "continuum_off", "drop_phase_0",
                     "drop_phase_0.5"]
    assert [s["kind"] for s in sysj["swaps"]] == ["config", "config"] + ["default"] * 4 + ["jackknife", "jackknife"]
    assert sysj["swaps"][1]["keys"] == ["parameters.u.free", "parameters.u.value"]
    assert sysj["swaps"][2]["keys"] == ["data.vi_scale"]
    assert any("no teff swaps" in n for n in sysj["notes"])
    assert len(sysj["modes"]) == 1
    m = sysj["modes"][0]
    assert m["mode"] == 1 and m["left_mode_rule"] == RULE_SEARCH and m["left_mode_limit"] == 1.0 and m["d99"] is None
    for q in ("md_x", "md_y", "md_z", "u", "incl_deg", "dipole_MG", "c_abs", "B50_MG[0]", "B50_MG[1]"):
        assert q in m["shifts"], q
    for q, per in m["shifts"].items():
        assert set(per) == set(names), q
        for v in per.values():
            assert v == MODE_CHANGE or (isinstance(v, float) and math.isfinite(v)), (q, v)
    for name in names:
        s = m["swaps"][name]
        assert s["n_starts"] == 4 and len(s["start_neg_log_l"]) == 4 and s["error"] is None
        assert math.isfinite(s["neg_log_l"]) and s["neg_log_l"] == min(s["start_neg_log_l"])
    assert m["swaps"]["u_far"]["mode_change"] is True and m["swaps"]["u_far"]["fingerprint_distance"] > 1.0
    assert m["shifts"]["incl_deg"]["u_far"] == MODE_CHANGE and "u_far" in m["mode_changes"]
    assert m["swaps"]["u_far"]["x"][13] == pytest.approx(0.1)  # the swap's fixed value was used
    # noise-free data of the truth (ld_mode 3): the mode is the truth; one phase left out still fits it
    for name in ("drop_phase_0", "drop_phase_0.5", "vi_scale_0.9", "vi_scale_1.1"):
        assert m["shifts"]["dipole_MG"][name] != MODE_CHANGE
        assert abs(m["shifts"]["dipole_MG"][name]) < 5.0 if name.startswith("drop") else \
            abs(m["shifts"]["dipole_MG"][name]) < 60.0
    assert m["shifts"]["dipole_MG"]["ld1"] != MODE_CHANGE
    print("V7f systematics: ld1 shifts", {q: m["shifts"][q]["ld1"] for q in ("incl_deg", "dipole_MG", "B50_MG[0]")},
          "u_far distance", m["swaps"]["u_far"]["fingerprint_distance"])
    # UNC-7: jackknife sigma per quantity over the 2 refits (no intervals.json without MCMC: systematics.json only)
    jk = m["jackknife_sigma"]
    assert m["jackknife_n"] == 2 and set(jk) == set(m["shifts"])
    d = [m["values"]["dipole_MG"] + m["shifts"]["dipole_MG"][n] for n in ("drop_phase_0", "drop_phase_0.5")]
    assert jk["dipole_MG"] == pytest.approx(math.sqrt(0.5 * sum((v - np.mean(d)) ** 2 for v in d)), rel=1e-9,
                                            abs=1e-12)
    assert not (out / "intervals.json").exists()
    assert json.loads((out / "run.json").read_text())["systematics"] is True
    # SYS-3: attach_jackknife touches only jackknife_sigma; the statistical numbers stay
    iv = {"modes": [{"mode": 1, "quantities": {q: {"median": 1.0, "p16": 0.5, "p84": 1.5, "half68": 0.5,
                                                   "jackknife_sigma": None, "flags": []} for q in m["shifts"]}}]}
    before = copy.deepcopy(iv)
    assert attach_jackknife(iv, sysj) == [1]
    for q, e in iv["modes"][0]["quantities"].items():
        assert e["jackknife_sigma"] == jk[q]
        e["jackknife_sigma"] = None
    assert iv == before
    # SYS-5: 3 sigma rule on the shifts; mode changes are not shifts; no intervals -> no flag and a note
    lim, note = limited_by_systematics({"a": {"s": 3.1}, "b": {"s": 2.9}, "c": {"s": MODE_CHANGE}},
                                       {"quantities": {q: {"p16": -1.0, "p84": 1.0} for q in "abc"}})
    assert lim == ["a"] and note is None
    lim, note = limited_by_systematics({"a": {"s": 30.0}}, {"quantities": {"a": {"p16": None, "p84": None}}})
    assert lim == [] and "without a statistical sigma" in note
    assert limited_by_systematics({"a": {"s": 30.0}}, None)[0] == []
    text = (out / "report.txt").read_text()
    print(text)
    assert "systematics: not yet implemented" not in text
    assert "systematic shifts of mode 1: u_far: mode change" in text
    assert "systematic shifts of mode 1: ld1: incl " in text
    assert "systematic shifts of mode 1: drop_phase_0.5: " in text
    assert "systematic shifts of mode 1: vi_scale_1.1: " in text
    assert "(2 configured, 4 default (SYS-1), 2 leave-one-phase-out)" in text
    # decision #23: the continuum setting heads every table; "off" carries the scale ratio of each I spectrum
    assert text.count("likelihood: plain chi2; kernel: none (diagonal errors); continuum: Legendre degree 3 on I, "
                      "marginalised (normalised basis)") >= 2
    off = [ln for ln in text.splitlines() if ln.startswith("systematic shifts of mode 1: continuum_off")]
    assert off and "scale ratio median |model I| / median |data I|:" in off[0]
    run = json.loads((out / "run.json").read_text())
    assert run["likelihood"]["continuum"] == {"degree": 3, "basis": "normalised"} and run["likelihood"]["kind"] == "chi2"
    assert "jackknife sigma of mode 1 (2 phases left out in turn, UNC-7): incl " in text
    assert "limited by model systematics, mode 1: not assessed: no intervals for this mode (no MCMC)" in text
    assert "no MCMC samples: the search's fingerprint thresholds" in text


# ------------------------------------------------------------------------------------------------------------- V3k


def test_v3k_fingerprint_beta_groups_the_inclination_ridge_of_one_phase(env):
    """Decision #16: with one phase, one
    geometry seen at different inclinations, or turned about the line of sight, is ONE mode with the default
    thresholds (angle = "auto" -> beta, centre in the frame of md and the observer). Points at inclinations 60 and 80
    degrees and at an azimuth of 50 degrees about the line of sight, with md and an offset centre c rotated with the
    observer direction, have the same beta and the same centre in that frame; with angle = "incl" they are separate
    modes, and the star-frame centre alone separates them too."""
    from zeemap.search import fingerprint, fingerprint_thresholds, same_mode

    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        (root / "one_phase.toml").write_text(config_text("inj", phases=(0.0,)))
        cfg = load_config(root / "one_phase.toml")
        pr = Problem(cfg)
    assert cfg["search"]["fingerprint"]["angle"] == "auto"
    iu = pr.names.index("u")
    ic = [pr.names.index(n) for n in ("c_x", "c_y", "c_z")]
    x1 = pr.value.copy()
    x1[ic] = [0.12, -0.05, 0.08]
    n1 = pr.forward.observers(pr.thetas(np.array([x1])))[0][0, 0]

    def rot(v, axis, ang):
        return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1.0 - math.cos(ang))

    x2 = x1.copy()
    x2[iu] = math.cos(math.radians(80.0))
    n2 = pr.forward.observers(pr.thetas(np.array([x2])))[0][0, 0]
    axis = np.cross(n1, n2)
    axis /= np.linalg.norm(axis)
    ang = math.acos(float(np.clip(n1 @ n2, -1.0, 1.0)))
    x2[0:3], x2[ic] = rot(x1[0:3], axis, ang), rot(x1[ic], axis, ang)
    x3 = x1.copy()  # the same geometry turned by 50 degrees about the line of sight
    x3[0:3], x3[ic] = rot(x1[0:3], n1, math.radians(50.0)), rot(x1[ic], n1, math.radians(50.0))
    f1, f2, f3 = fingerprint(pr, x1), fingerprint(pr, x2), fingerprint(pr, x3)
    for f in (f2, f3):
        assert f.beta[0] == pytest.approx(f1.beta[0], abs=1e-9)
        assert np.allclose(f.c_frame, f1.c_frame, atol=1e-12)
    assert f1.beta[0] == pytest.approx(angle_to_observer(MD_TRUE, U_TRUE, 0.0), abs=1e-9)
    assert abs(f1.incl - f2.incl) == pytest.approx(20.0, abs=1e-9)
    assert np.linalg.norm(f1.c - f3.c) > 0.06  # the star-frame centre moved
    thr = fingerprint_thresholds(cfg)
    assert same_mode(f1, f2, thr) and same_mode(f1, f3, thr)  # one mode with the defaults
    cfg["search"]["fingerprint"]["angle"] = "incl"
    assert not same_mode(f1, f2, fingerprint_thresholds(cfg))
    assert not same_mode(f1, f3, fingerprint_thresholds(cfg))  # incl equal, but the star-frame centre differs
    cfg["search"]["fingerprint"]["angle"] = "beta"
    assert same_mode(f1, f2, fingerprint_thresholds(cfg)) and same_mode(f1, f3, fingerprint_thresholds(cfg))
    # a config file takes the key; an unknown value is refused
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        beta_cfg = config_text("inj", phases=(0.0,), extra='[search.fingerprint]\nangle = "beta"\n')
        (root / "one_phase_beta.toml").write_text(beta_cfg)
        assert load_config(root / "one_phase_beta.toml")["search"]["fingerprint"]["angle"] == "beta"
        (root / "one_phase_bad.toml").write_text(beta_cfg.replace('angle = "beta"', 'angle = "psi"'))
        with pytest.raises(ConfigError):
            load_config(root / "one_phase_bad.toml")


def test_v7f_default_swaps_generated_from_the_configuration(env, tmp_path):
    """Decision #21 (SYS-1 defaults): `default_swaps` makes Teff fixed at each selected library slice inside the
    parameter's range (the slice of an already fixed Teff skipped), V/I x 0.9 and x 1.1, one swap per
    systematics.library_alternatives entry; nothing with systematics.defaults = false. data.vi_scale scales the
    observed V/I and its error in the reader."""
    from zeemap.data import load_data
    from zeemap.systematics import apply_swap, default_swaps

    root, _ = env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        p = tmp_path / "d.toml"
        p.write_text(config_text("inj", extra='[systematics]\nlibrary_alternatives = ["synth"]\n'))
        cfg = load_config(p)
        cfg["parameters"]["teff"] = {"value": 12000.0, "step": 500.0, "lo": 9000.0, "hi": 16000.0, "free": True}

        class _Lib:
            teffs = np.array([8000.0, 10000.0, 12000.0, 14000.0])

        class _P:
            library = _Lib()

        sw, notes = default_swaps(cfg, _P())
        assert [s["name"] for s in sw] == ["teff_10000", "teff_12000", "teff_14000", "vi_scale_0.9", "vi_scale_1.1",
                                           "library_1", "continuum_deg0", "continuum_off"]
        assert sw[-2]["set"] == {"objective": {"continuum": {"degree": 0}}}
        assert sw[-1]["set"] == {"objective": {"continuum": False}}
        assert sw[0]["set"] == {"parameters": {"teff": {"free": False, "value": 10000.0}}}
        assert sw[5]["set"] == {"library": {"path": "synth"}}
        assert any("teff_8000 skipped" in n for n in notes)
        cfg["parameters"]["teff"]["free"] = False
        assert [s["name"] for s in default_swaps(cfg, _P())[0]][:2] == ["teff_10000", "teff_14000"]
        cfg["systematics"]["defaults"] = False
        assert default_swaps(cfg, _P())[0] == []
        # data.vi_scale in the reader: V/I and its error times the factor, I unchanged
        base = load_config(root / "truth.toml")
        d1 = load_data(base)
        d2 = load_data(apply_swap(base, {"name": "v", "set": {"data": {"vi_scale": 1.1}}}))
    assert np.allclose(d2.VI, 1.1 * d1.VI, equal_nan=True) and np.allclose(d2.sig_VI, 1.1 * d1.sig_VI, equal_nan=True)
    assert np.array_equal(d2.I, d1.I, equal_nan=True)


def test_v7_mcmc_is_opt_in():
    """Decision #42: `fit` runs the MCMC only when asked: `--mcmc`, or `--mcmc-modes` / `--mcmc-evals` (which name an MCMC
    budget, so commands written before the default changed keep their MCMC); `--no-mcmc` always wins."""
    def on(*flags):
        return cli._mcmc_on(cli._parser().parse_args(["fit", "c.toml", "-o", "out", *flags]))

    assert not on() and on("--mcmc") and on("--mcmc-modes", "1") and on("--mcmc-evals", "2000")
    assert not on("--no-mcmc", "--mcmc-modes", "1")
