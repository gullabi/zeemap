"""V4: errors that name what is wrong. Here: V4c, the configuration cases (SPEC.md CFG-2)."""

import pytest
import tomli_w

from zeemap.config import ConfigError, load_config

pytestmark = pytest.mark.tier1


def base():
    return {
        "library": {"path": "lib"},
        "data": {
            "path": "obs",
            "windows": [[3900, 6800]],
            "grid": {"start": 3800, "step": 2.0, "npix": 1500},
            "phase": [{"phase": 0.0, "file": "ph00.dat", "ierr": "ph00_ierr.dat"}],
        },
        "model": {"fwhm": 3.0},
    }


def write(tmp_path, raw):
    p = tmp_path / "c.toml"
    p.write_text(tomli_w.dumps(raw))
    return p


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))


def test_v4c_base_config_is_valid(tmp_path, data_env):
    load_config(write(tmp_path, base()))


def test_v4c_unknown_key_named(tmp_path, data_env):
    raw = base()
    raw["model"]["fhwm"] = 3.0
    with pytest.raises(ConfigError, match=r"model\.fhwm"):
        load_config(write(tmp_path, raw))


def test_v4c_unknown_top_level_and_deep_keys_named(tmp_path, data_env):
    for mutate, key in [
        (lambda r: r.update(modle={"fwhm": 1.0}), "modle"),
        (lambda r: r["data"].setdefault("errors", {}).update(floor_V=1e-3), "data.errors.floor_V"),
        (lambda r: r.setdefault("objective", {}).update(disc={"a_I": 0.5, "rho": 3.0}), "objective.disc.rho"),
        (lambda r: r.setdefault("parameters", {}).update(bq={"value": 1, "step": 1, "lo": 0, "hi": 2, "fre": True}),
         "parameters.bq.fre"),
        (lambda r: r.setdefault("parameters", {}).update(bdip={"value": 1, "step": 1, "lo": 0, "hi": 2}),
         "parameters.bdip"),
        (lambda r: r["model"].update(surface={"scheme": "fibonacci", "nint": 4}), "model.surface.nint"),
        (lambda r: r.setdefault("systematics", {}).update(swap=[{"name": "x", "set": {"model": {"fhwm": 1.0}}}]),
         "systematics.swap[0].set.model.fhwm"),
    ]:
        raw = base()
        mutate(raw)
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert key in str(exc.value), (key, str(exc.value))


def test_v4c_unknown_key_in_phase_entry_named(tmp_path, data_env):
    raw = base()
    raw["data"]["phase"].append({"phase": 0.5, "file": "ph05.dat", "ierr": "e.dat", "use_V": False})
    with pytest.raises(ConfigError, match=r"data\.phase\[1\]\.use_V"):
        load_config(write(tmp_path, raw))


def test_v4c_missing_required_key_named(tmp_path, data_env):
    for path in ["model.fwhm", "library.path", "data.windows", "data.grid.npix", "data.phase[0].ierr"]:
        raw = base()
        node, keys = raw, path.replace("[0]", ".0").split(".")
        for k in keys[:-1]:
            node = node[int(k)] if k.isdigit() else node[k]
        del node[keys[-1]]
        with pytest.raises(ConfigError, match="missing required key " + path.replace(".", r"\.").replace("[", r"\[")
                           .replace("]", r"\]")):
            load_config(write(tmp_path, raw))
    raw = base()
    raw["data"]["phase"] = []
    with pytest.raises(ConfigError, match=r"data\.phase"):
        load_config(write(tmp_path, raw))
    raw = base()
    raw["parameters"] = {"u": {"value": 0.5, "lo": -1, "hi": 1}}
    with pytest.raises(ConfigError, match=r"missing required key parameters\.u\.step"):
        load_config(write(tmp_path, raw))


def test_v4c_relative_data_path_needs_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEEMAP_DATA", raising=False)
    with pytest.raises(ConfigError, match="ZEEMAP_DATA") as exc:
        load_config(write(tmp_path, base()))
    assert "library.path" in str(exc.value)


def test_v4c_wrong_types_named(tmp_path, data_env):
    for mutate, key in [
        (lambda r: r["data"].update(windows=[[3900, 6800, 1.0, 2.0]]), "data.windows[0]"),
        (lambda r: r["data"].update(windows=[[6800, 3900]]), "data.windows[0]"),
        (lambda r: r["data"].update(exclude=[[4000, 4100, "Q"]]), "data.exclude[0]"),
        (lambda r: r["model"].update(ld_mode=2), "model.ld_mode"),
        (lambda r: r["data"]["grid"].update(npix=1500.5), "data.grid.npix"),
        (lambda r: r.setdefault("objective", {}).update(corr=True), "objective.corr"),
        (lambda r: r.setdefault("parameters", {}).update(u={"value": 2.0, "step": 0.1, "lo": -1, "hi": 1}),
         "parameters.u"),
        (lambda r: r.setdefault("parameters", {}).update(md={"value": [0, 0, 50], "step": 1, "lo": 1, "hi": 40}),
         "parameters.md"),
        (lambda r: r.setdefault("parameters", {}).update(md={"value": 10.0, "step": 1, "lo": 1, "hi": 40}),
         "parameters.md.value"),
        (lambda r: r.setdefault("parameters", {}).update(u={"value": 0.5, "step": 0.0, "lo": -1, "hi": 1}),
         "parameters.u.step"),
    ]:
        raw = base()
        mutate(raw)
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert key in str(exc.value), (key, str(exc.value))


def test_v4c_profile_settings_named(tmp_path, data_env):
    """Bad profile settings stop at load, naming the key, not after the search and the MCMC. Quantity
    names are those of FORMATS.md section 7, a B quantity with a phase index below the number of phases; a step is a
    positive number or a positive "<p>%"."""
    for mutate, key in [
        (lambda u: u.update(profile_quantities=["inclination"]), "uncertainty.profile_quantities[0]"),
        (lambda u: u.update(profile_quantities=["incl_deg", "B50_MG[1]"]), "uncertainty.profile_quantities[1]"),
        (lambda u: u.update(profile_quantities=["B50_MG"]), "uncertainty.profile_quantities[0]"),
        (lambda u: u.update(profile_quantities=["dipole_MG[0]"]), "uncertainty.profile_quantities[0]"),
        (lambda u: u.update(profile_quantities="all"), "uncertainty.profile_quantities"),
        (lambda u: u.update(profile_steps={"dipole_MG": "two%"}), "uncertainty.profile_steps.dipole_MG"),
        (lambda u: u.update(profile_steps={"dipole_MG": "-5%"}), "uncertainty.profile_steps.dipole_MG"),
        (lambda u: u.update(profile_steps={"dipole_MG": "0%"}), "uncertainty.profile_steps.dipole_MG"),
        (lambda u: u.update(profile_steps={"dipole_MG": "nan%"}), "uncertainty.profile_steps.dipole_MG"),
        (lambda u: u.update(profile_steps={"dipole_MG": "5"}), "uncertainty.profile_steps.dipole_MG"),
        (lambda u: u.update(profile_steps={"incl_deg": 0.0}), "uncertainty.profile_steps.incl_deg"),
        (lambda u: u.update(profile_steps={"incl_deg": -2.5}), "uncertainty.profile_steps.incl_deg"),
        (lambda u: u.update(profile_steps={"incl_deg": True}), "uncertainty.profile_steps.incl_deg"),
        (lambda u: u.update(profile_steps={"inclination": 2.5}), "uncertainty.profile_steps.inclination"),
        (lambda u: u.update(profile_steps={"B50_MG[3]": "1%"}), "uncertainty.profile_steps.B50_MG[3]"),
        (lambda u: u.update(profile_max_steps=0), "uncertainty.profile_max_steps"),
        (lambda u: u.update(profile_families=-2), "uncertainty.profile_families"),
    ]:
        raw = base()
        mutate(raw.setdefault("uncertainty", {}))
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert key in str(exc.value), (key, str(exc.value))
    raw = base()  # accepted forms
    raw["uncertainty"] = {"profile_quantities": ["incl_deg", "beta_deg", "dipole_MG", "c_abs", "c_along_axis",
                                                 "c_across_axis", "B10_MG[0]", "B50_MG[0]", "B90_MG[0]", "teff_K"],
                          "profile_steps": {"dipole_MG": "2.5%", "B50_MG": "1e-1%", "B90_MG[0]": 5, "teff_K": 50.0},
                          "profile_max_steps": 1, "profile_families": 0}
    cfg = load_config(write(tmp_path, raw))
    assert cfg["uncertainty"]["profile_steps"]["dipole_MG"] == "2.5%"


def test_v4c_family_and_shift_node_ranges_named(tmp_path, data_env):
    """The integer and number ranges of search.family and model.shift_nodes are checked at load."""
    for mutate, key in [
        (lambda r: r.setdefault("search", {}).update(family={"min_level": -3.0}), "search.family.min_level"),
        (lambda r: r.setdefault("search", {}).update(family={"min_level": 0.0}), "search.family.min_level"),
        (lambda r: r.setdefault("search", {}).update(family={"max_pairs": 0}), "search.family.max_pairs"),
        (lambda r: r.setdefault("search", {}).update(family={"mid_iter": -1}), "search.family.mid_iter"),
        (lambda r: r.setdefault("search", {}).update(family={"mid_iter": 0}), "search.family.mid_iter"),
        (lambda r: r.setdefault("search", {}).update(family={"interior": 0}), "search.family.interior"),
        (lambda r: r.setdefault("search", {}).update(family={"interior": 2.5}), "search.family.interior"),
        (lambda r: r["model"].update(shift_nodes=1), "model.shift_nodes"),
    ]:
        raw = base()
        mutate(raw)
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert key in str(exc.value), (key, str(exc.value))
    raw = base()
    raw["search"] = {"family": {"min_level": 0.5, "max_pairs": 1, "mid_iter": 1, "interior": 1}}
    raw["model"]["shift_nodes"] = 2
    load_config(write(tmp_path, raw))


def test_v4c_unmatched_kernel_label_named(tmp_path, data_env):
    """An [[objective.kernel]] block whose spectrum label matches no spectrum of the configuration
    (phase as %.3f, "_I" or "_VI", and only the kinds the phase uses) stops at load, naming the block; it was ignored."""
    k = {"rho_short": 4.0, "a": 0.5, "rho_long": 80.0, "j": 0.05}
    for spec, phase_extra in (("0.250_I", {}), ("0.000_V", {}), ("0_I", {}), ("0.000_VI", {"use_VI": False})):
        raw = base()
        raw["data"]["phase"][0].update(phase_extra)
        raw["objective"] = {"kernel_fit": False, "kernel": [{"spectrum": "0.000_I", **k}, {"spectrum": spec, **k}]}
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert "objective.kernel[1].spectrum" in str(exc.value) and spec in str(exc.value), str(exc.value)
    raw = base()
    raw["objective"] = {"kernel_fit": False, "kernel": [{"spectrum": "0.000_I", **k}, {"spectrum": "0.000_VI", **k}]}
    load_config(write(tmp_path, raw))


def test_v4c_lm_stop_settings_named(tmp_path, data_env):
    """search.lm_stop_nat (a number >= 0, 0 = off) and search.lm_max_rejections (an integer >= 0, 0 = off) are checked at
    load, naming the key."""
    for mutate, key in [
        (lambda s: s.update(lm_stop_nat=-0.01), "search.lm_stop_nat"),
        (lambda s: s.update(lm_stop_nat="0.01"), "search.lm_stop_nat"),
        (lambda s: s.update(lm_stop_nat=True), "search.lm_stop_nat"),
        (lambda s: s.update(lm_max_rejections=-1), "search.lm_max_rejections"),
        (lambda s: s.update(lm_max_rejections=2.5), "search.lm_max_rejections"),
    ]:
        raw = base()
        mutate(raw.setdefault("search", {}))
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, raw))
        assert key in str(exc.value), (key, str(exc.value))
    raw = base()
    raw["search"] = {"lm_stop_nat": 0, "lm_max_rejections": 0}
    cfg = load_config(write(tmp_path, raw))
    assert cfg["search"]["lm_stop_nat"] == 0.0 and isinstance(cfg["search"]["lm_stop_nat"], float)
    raw["search"] = {"lm_stop_nat": 0.01, "lm_max_rejections": 3}
    load_config(write(tmp_path, raw))
