"""V7g, V7h, and a further V3k item.

- V3k: a geometry and its field-reversed copy (V of opposite sign, same folded inclination) are two modes.
- V7g: outputs are written atomically (ENG-7); run.json holds no absolute path (ENG-4).
- V7h: a default Teff swap whose refit keeps the geometry is a shift, not a "mode change" (SYS-4).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from zeemap import atomic, cli
from zeemap import search as S

from test_v7_cli import config_text

pytestmark = pytest.mark.tier1


# ------------------------------------------------------------------------------------------------------------- V3k


def test_v3k_field_reversed_copy_is_another_mode():
    b50, b90 = np.array([200.0, 210.0]), np.array([400.0, 380.0])
    a = S.Fingerprint(30.0, np.zeros(3), b50, b90, beta=np.array([40.0, 60.0]))
    rev = S.Fingerprint(30.0, np.zeros(3), b50, b90, beta=np.array([140.0, 120.0]))  # beta -> 180 - beta
    assert S.fingerprint_distance(a, rev) == pytest.approx(100.0 / 8.0)
    assert not S.same_mode(a, rev)
    near = S.Fingerprint(30.0, np.zeros(3), b50, b90, beta=np.array([47.0, 60.0]))
    assert S.same_mode(a, near)
    # without beta (a problem that does not supply it) the rule is the folded inclination alone, as before
    assert S.same_mode(S.Fingerprint(30.0, np.zeros(3), b50, b90), S.Fingerprint(30.0, np.zeros(3), b50, b90))


def test_v3k_field_reversed_copy_on_the_real_problem(synth_lib, tmp_path):
    """On the synthetic library: reversing every field vector flips beta at every phase and keeps the inclination."""
    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = root / "truth.toml"
        cfg_path.write_text(config_text("inj"))
        assert cli.main(["inject", str(cfg_path), "-o", str(root / "inj"), "--noise", "none"]) == 0
        from zeemap.config import load_config
        from zeemap.problem import Problem

        p = Problem(load_config(cfg_path))
        x = np.array(p.value, dtype=float)
        xr = x.copy()
        for i, n in enumerate(p.names):
            if n.startswith(("md_", "mo_")) or n == "bq":
                xr[i] = -xr[i]
        f1, f2 = S.fingerprint(p, x), S.fingerprint(p, xr)
    assert f1.incl == pytest.approx(f2.incl, abs=1e-9)
    assert np.allclose(f1.beta + f2.beta, 180.0, atol=1e-9)
    assert not S.same_mode(f1, f2, S.fingerprint_thresholds(None))


# ------------------------------------------------------------------------------------------------------------- V7g


def test_v7g_interrupted_write_leaves_no_final_file(tmp_path):
    f = tmp_path / "out.json"
    with pytest.raises(RuntimeError):
        with atomic.replacing(f) as h:
            h.write('{"half": ')
            raise RuntimeError("interrupted")
    assert not f.exists() and list(tmp_path.iterdir()) == []
    f.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic.replacing(f) as h:
            h.write("new, half")
            raise RuntimeError("interrupted")
    assert f.read_text() == "old" and list(tmp_path.iterdir()) == [f]
    atomic.savetxt(f, np.arange(3.0))
    assert np.allclose(np.loadtxt(f), [0, 1, 2]) and list(tmp_path.iterdir()) == [f]


ABS_PATH = re.compile(r'(?:^|[\s"=\[,(])(/(?:home|tmp|Users|root|data|var|private)/)')


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7g_run_json_has_no_absolute_paths(synth_lib, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = root / "truth.toml"
        cfg_path.write_text(config_text("inj"))
        assert cli.main(["inject", str(cfg_path), "-o", str(root / "inj"), "--noise", "none"]) == 0
        out = tmp_path / "fit"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "1", "--starts", "2", "--no-mcmc",
                         "--no-systematics"]) == 0
    run = json.loads((out / "run.json").read_text())
    for s in _strings(run):
        assert str(root) not in s and str(tmp_path) not in s, s[:200]
        assert not os.path.isabs(s) and not ABS_PATH.search(s), s[:200]
    keys = list(run["input_sha256"])
    assert keys and all(k.startswith("$ZEEMAP_DATA/") for k in keys), keys
    assert any(k.endswith("synth/info.dat") for k in keys)
    assert run["config_path"] == "$ZEEMAP_DATA/truth.toml"
    assert not list(out.rglob(".*.tmp*"))  # no temporary files left


# ------------------------------------------------------------------------------------------------------------- V7h


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7h_teff_swap_that_keeps_the_geometry_is_a_shift(synth_lib, tmp_path):
    """Noise-free data at 30 000 K from the synthetic library, whose spectra scale with Teff only through the continuum
    (VERIFICATION.md section 4): fixing Teff at the neighbouring slices leaves the geometry where it is, so the default
    swaps teff_20000 and teff_40000 are shifts. Before the fix they were "mode change" by construction, because the
    1000 K Teff component of the fingerprint was compared."""
    root = tmp_path / "data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    teff = 'teff = { value = 30000.0, step = 1000.0, lo = 20000.0, hi = 40000.0, free = true }\n'
    txt = config_text("inj", phases=(0.0,)).replace("teffs = [30000.0]", "teffs = [20000.0, 30000.0, 40000.0]")
    txt = txt.replace("[search]\n", teff + "\n[search]\n")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(root))
        cfg_path = root / "truth.toml"
        cfg_path.write_text(txt)
        assert cli.main(["inject", str(cfg_path), "-o", str(root / "inj"), "--noise", "none"]) == 0
        out = tmp_path / "fit"
        assert cli.main(["fit", str(cfg_path), "-o", str(out), "--seed", "4", "--starts", "3", "--no-mcmc",
                         "--sys-modes", "1"]) == 0
    sysj = json.loads((out / "systematics.json").read_text())
    m = sysj["modes"][0]
    names = list(m["swaps"])
    assert "teff_20000" in names and "teff_40000" in names, names
    for n in ("teff_20000", "teff_40000"):
        sw = m["swaps"][n]
        assert sw["mode_change"] is False, (n, sw)
        assert isinstance(m["shifts"]["dipole_MG"][n], float)
        assert abs(m["shifts"]["dipole_MG"][n]) < 5.0
    assert "teff_20000" not in m["mode_changes"] and "teff_40000" not in m["mode_changes"]
