"""V1k: the configuration is written and read back unchanged; every key in defaults.toml has a comment
(SPEC.md CFG-1, CFG-3)."""

import re
from pathlib import Path

import pytest
import tomli_w

from zeemap import config

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "src" / "zeemap" / "defaults.toml"
EXAMPLE = ROOT / "examples" / "synth_single.toml"


def minimal():
    """A small valid configuration (relative paths)."""
    return {
        "library": {"path": "lib"},
        "data": {
            "path": "obs",
            "windows": [[3900, 6800], [7000, 8000, 0.5]],
            "exclude": [[4000, 4010], [5000, 5010, "VI"]],
            "grid": {"start": 3800, "step": 2.0, "npix": 1500},
            "phase": [
                {"phase": 0.0, "file": "ph00.dat", "ierr": "ph00_ierr.dat", "vierr": "ph00_vierr.dat"},
                {"phase": 0.5, "file": "ph05.dat", "ierr": "ph05_ierr.dat", "use_VI": False},
            ],
            "weight": [{"stokes": "I", "lo": 4100, "hi": 4200, "w": 2.0}],
        },
        "model": {"fwhm": 3, "surface": {"n": 3000}},
        "objective": {"corr": False, "continuum": {"degree": 2}},
        "parameters": {
            "md": {"value": [0, 0, 10], "step": 1.0, "lo": 1, "hi": 40, "free": True},
            "u": {"value": 0.5, "step": 0.1, "lo": -1, "hi": 1},
        },
        "uncertainty": {"calibration": "cal/run.json"},
        "systematics": {"swap": [{"name": "ld1", "set": {"model": {"ld_mode": 1}}}]},
    }


def strip(cfg):
    return {k: v for k, v in cfg.items() if k != "_meta"}


def test_v1k_config_written_and_read_back_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path / "data"))
    p = tmp_path / "run" / "a.toml"
    p.parent.mkdir()
    p.write_text(tomli_w.dumps(minimal()))
    cfg = config.load_config(p)
    assert cfg["_meta"]["dir"] == str(p.parent)
    q = tmp_path / "b.toml"
    q.write_text(config.dump(cfg))
    assert "_meta" not in config.dump(cfg)
    cfg2 = config.load_config(q)
    assert strip(cfg2) == strip(cfg)
    # and a third pass is a fixed point of dump
    assert config.dump(cfg2) == config.dump(cfg)


def test_v1k_defaults_filled_and_values_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    p = tmp_path / "a.toml"
    p.write_text(tomli_w.dumps(minimal()))
    cfg = config.load_config(p)
    assert cfg["objective"]["corr"] is False
    assert cfg["objective"]["disc"] == {"a_I": 0.71, "rho_I": 85.0, "a_VI": 0.42, "rho_VI": 205.0}
    assert cfg["objective"]["continuum"] == {"degree": 2}
    assert cfg["model"]["surface"] == {"scheme": "fibonacci", "n": 3000}
    assert cfg["model"]["fwhm"] == 3.0 and isinstance(cfg["model"]["fwhm"], float)
    assert cfg["model"]["ld_mode"] == 3 and cfg["library"]["teffs"] == "all"
    assert cfg["data"]["errors"] == {"absolute": True, "floor_I": 1e-6, "floor_VI": 1e-5, "derive_VI": False}
    ph1 = cfg["data"]["phase"][1]
    assert ph1["use_I"] is True and ph1["use_VI"] is False and ph1["i_offset"] is True and ph1["vierr"] == ""
    assert cfg["data"]["weight"][0]["phases"] == "all"
    assert cfg["parameters"]["u"]["free"] is False
    assert cfg["parameters"]["md"]["value"] == [0.0, 0.0, 10.0]
    assert cfg["search"]["n_starts"] == 250 and cfg["search"]["starts"]["md"] == [200.0, 600.0]
    # paths: data relative to $ZEEMAP_DATA, phase files to data.path, others to the configuration's directory
    assert cfg["library"]["path"] == str(tmp_path / "lib")
    assert cfg["data"]["phase"][0]["file"] == str(tmp_path / "obs" / "ph00.dat")
    assert cfg["uncertainty"]["calibration"] == str(tmp_path / "cal" / "run.json")


def test_v1k_absolute_paths_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEEMAP_DATA", raising=False)
    raw = minimal()
    raw["library"]["path"] = str(tmp_path / "L")
    raw["data"]["path"] = str(tmp_path / "D")
    p = tmp_path / "a.toml"
    p.write_text(tomli_w.dumps(raw))
    cfg = config.load_config(p)
    assert cfg["library"]["path"] == str(tmp_path / "L")
    assert cfg["data"]["phase"][1]["ierr"] == str(tmp_path / "D" / "ph05_ierr.dat")


def test_v1k_every_default_key_has_a_comment():
    lines = DEFAULTS.read_text().splitlines()
    key_re = re.compile(r'^\s*("[^"]*"|[A-Za-z0-9_-]+)\s*=')
    missing = []
    n = 0
    for i, line in enumerate(lines):
        if not key_re.match(line):
            continue
        n += 1
        same_line = "#" in line.split("=", 1)[1]
        above = i > 0 and lines[i - 1].lstrip().startswith("#")
        if not (same_line or above):
            missing.append(f"line {i + 1}: {line}")
    assert n > 60  # every key of FORMATS.md section 3
    assert not missing, "\n".join(missing)


def test_v1k_defaults_cover_the_key_table():
    d = config.defaults()
    for path in [
        "library.path", "library.teffs", "library.outside", "data.path", "data.grid", "data.windows", "data.exclude",
        "data.errors", "data.phase", "data.weight", "model.field",
        "model.interp_b", "model.surface", "model.ld_mode", "model.fwhm", "model.phase_offset",
        "objective.kind", "objective.scale", "objective.corr", "objective.disc",
        "objective.continuum", "parameters", "search.top", "search.restarts",
        "search.rounds_max", "search.round_gain", "search.band", "search.lbfgs_maxiter", "search.fd_step",
        "search.lm_maxiter", "search.starts.md", "search.starts.bq", "search.starts.mo", "search.starts.c",
        "search.starts.u", "search.starts.outside_max", "search.starts.rules", "search.perturb.b_frac",
        "search.perturb.angle", "search.perturb.c", "search.perturb.incl", "uncertainty.patches",
        "uncertainty.ensembles", "uncertainty.tau_factor", "uncertainty.max_evals",
        "uncertainty.calibration", "systematics.swap", "search.family.min_level", "search.family.max_pairs",
        "search.family.mid_iter", "search.family.interior",
    ]:
        node = d
        for k in path.split("."):
            assert k in node, path
            node = node[k]


def test_v1k_search_family_read_back_and_defaults_match_the_code(tmp_path, monkeypatch):
    """`[search.family]` (FORMATS.md section 3) is a known table: its defaults are the ones `families.family_settings`
    used before the table existed, a value set in the configuration survives the round trip and
    reaches the family code."""
    from zeemap import families

    assert config.defaults()["search"]["family"] == families.DEFAULTS
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    raw = minimal()
    raw["search"] = {"family": {"min_level": 7.0, "max_pairs": 100}}
    p = tmp_path / "a.toml"
    p.write_text(tomli_w.dumps(raw))
    cfg = config.load_config(p)
    want = {**families.DEFAULTS, "min_level": 7.0, "max_pairs": 100}
    assert cfg["search"]["family"] == want
    assert families.family_settings(cfg) == want
    q = tmp_path / "b.toml"
    q.write_text(config.dump(cfg))
    assert strip(config.load_config(q)) == strip(cfg)


def test_v1k_example_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    cfg = config.load_config(EXAMPLE)
    assert cfg["library"]["path"] == str(tmp_path / "synth_zeeman")
    assert cfg["data"]["path"] == str(tmp_path / "obs_dah12")
    ph = cfg["data"]["phase"]
    assert len(ph) == 1 and ph[0]["phase"] == 0.0
    assert ph[0]["file"] == str(tmp_path / "obs_dah12" / "ph00.dat")
    assert ph[0]["vierr"] == str(tmp_path / "obs_dah12" / "ph00_vierr.dat")
    assert cfg["data"]["grid"] == {"start": 3800.0, "step": 2.0, "npix": 1501}
    assert cfg["data"]["windows"] == [[3850.0, 6750.0]]
    assert cfg["model"]["fwhm"] == 3.0
    assert cfg["model"]["surface"] == {"scheme": "fibonacci", "n": 5000}
    assert set(cfg["parameters"]) == {"md", "bq", "aq", "mo", "c", "u", "teff"}
    assert cfg["_meta"]["dir"] == str(EXAMPLE.parent)
