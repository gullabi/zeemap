"""V7k (ENG-8): `fit` writes each stage's result to <out>/checkpoint/ as it ends, and `--resume` / `--from` continue
from them. On the small synthetic problem of tests/test_v6_determinism.py (fitted from md = 300 MG along z, u = 0.7,
3 starts, one restart round, a capped MCMC on the best mode, systematics on the best mode, one profiled quantity with
2 grid steps):
- a run that crashes right after the search's round-0 checkpoint and is resumed writes the same modes.json,
  families.json, profiles.json, intervals.json and systematics.json as an uninterrupted run (V6 tolerance); resumed
  once more with another MCMC budget, the table stage, the profile walks and every refit come from their checkpoints;
- `--from` a search-only run (`--no-mcmc --no-systematics`) with the MCMC and the systematics on does not search again
  or walk the profiles again, and gives the same outputs;
- a resume with a changed data path, or with changed bytes of a data file at the same path, stops with a one-line
  message naming `data` and exit status 2;
- `run_search` resumed from each state it saved (round 0, the kernel stage with the full likelihood, a restart round
  that met the stop rule with `rounds_max` = 3) gives the uninterrupted search's result;
- a checkpoint that names a global outside zeemap's classes and numpy's arrays (here os.system) is refused when it is
  read, before anything it names is called.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest

from zeemap import cli, search
from zeemap.config import load_config
from zeemap.problem import Problem
from test_v6_determinism import _diff, data_root  # noqa: F401 (data_root: the module fixture with the injection)
from test_v7_cli import config_text
from test_v7_noise_diag import _text

pytestmark = pytest.mark.tier1

FILES = ("modes.json", "families.json", "profiles.json", "intervals.json", "systematics.json")
LATER = ["--mcmc-modes", "1", "--mcmc-evals", "2000", "--sys-modes", "1"]
UNC = '[uncertainty]\npatches = 1000\nensembles = 2\nprofile_quantities = ["dipole_MG"]\nprofile_max_steps = 2\n'


class Crash(Exception):
    pass


def _fit(cfg, out, *extra):
    return cli.main(["fit", str(cfg), "-o", str(out), "--seed", "4", "--workers", "2", "--starts", "3", *extra])


def _same(a, b):
    for name in FILES:
        ja, jb = (json.loads((d / name).read_text()) for d in (a, b))
        ja.pop("_meta", None)
        jb.pop("_meta", None)
        assert not _diff(ja, jb, name), _diff(ja, jb, name)[:5]


def _used(out) -> list[str]:
    """Kinds of the checkpoints a resumed run took (run.json `resumed`)."""
    return [c.split("-")[0] for c in json.loads((out / "run.json").read_text())["resumed"]["checkpoints"]]


@pytest.fixture(scope="module")
def runs(data_root, tmp_path_factory):  # noqa: F811
    tmp = tmp_path_factory.mktemp("resume")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZEEMAP_DATA", str(data_root))
        cfg = tmp / "fit.toml"
        cfg.write_text(config_text("inj", md=300.0 * np.array([0.0, 0.0, 1.0]), u=0.7, extra=UNC))
        assert _fit(cfg, tmp / "whole", *LATER) == 0
        assert _fit(cfg, tmp / "search", "--no-mcmc", "--no-systematics") == 0
        yield tmp, cfg


@pytest.mark.slow  # end-to-end fits (tools/check.sh 1 --fast leaves them out)
def test_v7k_crash_after_round_0_then_resume_gives_the_same_outputs(runs, monkeypatch, capsys):
    tmp, cfg = runs
    put = cli._ck_put

    def put_then_crash(args, name, obj):
        put(args, name, obj)
        if name == "search":
            raise Crash

    monkeypatch.setattr(cli, "_ck_put", put_then_crash)
    with pytest.raises(Crash):
        _fit(cfg, tmp / "crashed", *LATER)
    assert not (tmp / "crashed" / "modes.json").exists()
    monkeypatch.setattr(cli, "_ck_put", put)
    capsys.readouterr()
    assert _fit(cfg, tmp / "crashed", *LATER, "--resume") == 0
    assert "resumed after round 0" in capsys.readouterr().err
    _same(tmp / "whole", tmp / "crashed")
    assert json.loads((tmp / "crashed" / "diagnostics.json").read_text())["_meta"]["ms_per_eval_per_worker"] is None
    n_sys = len(json.loads((tmp / "whole" / "systematics.json").read_text())["modes"][0]["swaps"])
    # another MCMC budget reruns the MCMC only: the walks and the refits do not read it
    assert _fit(cfg, tmp / "crashed", *LATER[:2], "--mcmc-evals", "3000", "--sys-modes", "1", "--resume") == 0
    used = _used(tmp / "crashed")
    assert used.count("table") == used.count("profiles") == 1 and used.count("sys") == n_sys and "mcmc" not in used


@pytest.mark.slow
def test_v7k_from_a_search_only_run_does_not_search_again(runs, capsys):
    tmp, cfg = runs
    capsys.readouterr()
    assert _fit(cfg, tmp / "later", *LATER, "--from", str(tmp / "search")) == 0
    err = capsys.readouterr().err
    assert "starts drawn" not in err and "resumed after round 1" in err
    assert "profiles" in _used(tmp / "later")
    _same(tmp / "whole", tmp / "later")


@pytest.mark.slow  # needs the search-only run of the fixture
def test_v7k_resume_refuses_changed_data(runs, capsys):
    tmp, cfg = runs
    root = os.environ["ZEEMAP_DATA"]
    if not os.path.exists(os.path.join(root, "inj2")):
        os.symlink(os.path.join(root, "inj"), os.path.join(root, "inj2"), target_is_directory=True)
    cfg2 = tmp / "fit2.toml"
    cfg2.write_text(cfg.read_text().replace('path = "inj"', 'path = "inj2"'))
    dat = os.path.join(root, "inj", "ph00.dat")
    original = open(dat, "rb").read()
    for c, change in ((cfg2, None), (cfg, b"\n")):  # another path; the same path with other bytes
        capsys.readouterr()
        try:
            if change:
                open(dat, "ab").write(change)
            assert _fit(c, tmp / "refused", "--from", str(tmp / "search")) == 2
        finally:
            open(dat, "wb").write(original)
        err = capsys.readouterr().err.strip().splitlines()
        assert len(err) == 2 and "cannot resume" in err[1] and "data" in err[1], err  # err[0]: the problem line
        assert not (tmp / "refused" / "modes.json").exists()


def test_v7k_checkpoint_naming_other_code_is_refused(tmp_path):
    from types import SimpleNamespace

    from zeemap.config import ConfigError

    class Evil:
        def __reduce__(self):
            return os.system, (f"touch {tmp_path / 'ran'}",)

    args = SimpleNamespace(ck_src=str(tmp_path), out=str(tmp_path), ck_used=[])
    cli._ck_put(args, "fine", {"x": np.arange(3.0), "fp": search.Fingerprint(10.0, np.zeros(3), np.ones(1), np.ones(1))})
    got = cli._ck_get(args, "fine")
    assert np.array_equal(got["x"], np.arange(3.0)) and isinstance(got["fp"], search.Fingerprint)
    cli._ck_put(args, "search", Evil())
    with pytest.raises(ConfigError, match="posix.system|os.system"):
        cli._ck_get(args, "search")
    # protocol 4 resolves dotted names: module zeemap.report, name subprocess.Popen would reach Popen through report's
    # imports (STACK_GLOBAL, then STOP; refused when looked up, before anything is called)
    (tmp_path / "checkpoint" / "search.pkl").write_bytes(
        b"\x80\x04\x8c\x0dzeemap.report\x8c\x10subprocess.Popen\x93.")
    with pytest.raises(ConfigError, match="subprocess.Popen"):
        cli._ck_get(args, "search")
    assert not (tmp_path / "ran").exists()


@pytest.mark.slow  # a search with the kernel stage, resumed three times
def test_v7k_run_search_resumes_from_each_saved_state(synth_lib, tmp_path, monkeypatch):
    os.symlink(synth_lib, tmp_path / "synth", target_is_directory=True)
    monkeypatch.setenv("ZEEMAP_DATA", str(tmp_path))
    (tmp_path / "truth.toml").write_text(_text("inj"))
    assert cli.main(["inject", str(tmp_path / "truth.toml"), "-o", str(tmp_path / "inj"), "--noise", "white",
                     "--seed", "5", "--sigma-i", "0.01", "--sigma-vi", "0.002"]) == 0
    (tmp_path / "fit.toml").write_text(_text("inj").replace("rounds_max = 1", "rounds_max = 3\nround_gain = 1e9"))
    cfg = load_config(tmp_path / "fit.toml")
    states = []
    whole = search.run_search(Problem(cfg), cfg, seed=2, n_starts=3, make_problem=Problem,
                              save=lambda st: states.append(pickle.loads(pickle.dumps(st))))
    assert [s["step"] for s in states] == ["round 0", "the kernel stage", "round 1"]
    for st in states:
        again = search.run_search(Problem(cfg), cfg, seed=2, n_starts=3, make_problem=Problem, state=st)
        assert not _diff(whole.to_dict(), again.to_dict(), st["step"]), _diff(whole.to_dict(), again.to_dict())[:5]
