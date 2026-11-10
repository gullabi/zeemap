"""V7e: `fit` with a toy field model registered from this test file (not in `src/`) runs without any change to the
search or error code (SPEC.md FIELD-4).

Two toy models, both registered here with `field.register`:

- "toy_dipole_l4" names its parameters like the built-in models (md_x, md_y, md_z, u): a centred dipole md plus an
  l = 4 component on the same axis whose polar strength is a fixed fraction L4_RATIO of |md|. It is not one of the
  built-in models, so the forward model, the search, the table stage, the systematics and the
  outputs all meet a field that only this file knows. `inject` writes noise-free data from the truth below, and `fit`
  (with `--no-mcmc`, as the error-bar stage is checked by V7f) must finish, write its outputs and find the truth.
- "toy_axis_dipole" has a parameter name of its own: a centred dipole on a fixed axis (60 degrees from the rotation
  axis, in the x-z plane) with a free polar strength `bd`, plus u. Such a model cannot be configured or fitted
  without changes in `src/` (FIELD-4 is a SHOULD): the test is marked xfail(strict) and names the
  places that block it, so the gap stays visible and the mark has to go when they are fixed. With those four places
  patched in the test process (a probe, not kept), the same `fit --no-mcmc` ran to the end, systematics included, and
  found bd = 400 MG and u = 0.5: the search, the table stage, the families and the outputs handle such a model already.

Set-up as in test_v7_cli.py: $ZEEMAP_DATA is a temporary directory holding a link to the session's synthetic library
(variant a), the configurations and the injected data. Truth: |md| = 400 MG with the axis 60 degrees from the
rotation axis, seen at i = 60 degrees (u = 0.5), Teff 30 000 K (one slice), 1000 Fibonacci patches, phases 0 and 0.5.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from zeemap import cli, field

pytestmark = pytest.mark.tier1

AXIS = np.array([math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
BD_TRUE = 400.0
U_TRUE = 0.5
L4_RATIO = 0.25  # polar strength of the l = 4 component / |md|


@field.register("toy_dipole_l4")
class ToyDipoleL4(field.FieldModel):
    """Centred dipole md (MG) plus an l = 4 component on md's axis of polar strength L4_RATIO |md|; u = cos i."""

    names = ["md_x", "md_y", "md_z", "u"]
    dim = 4

    def evaluate(self, points, theta):
        t = self._theta(theta)
        md = t[0:3]
        s = float(np.linalg.norm(md))
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        if s == 0.0:
            return np.zeros_like(p)
        n = md / s
        return field.component_field(p, n, s, 1) + field.component_field(p, n, L4_RATIO * s, 4)

    def valid(self, theta):
        t = np.asarray(theta, dtype=float).reshape(-1)
        return t.shape[0] == self.dim and bool(np.all(np.isfinite(t))) and float(np.linalg.norm(t[0:3])) > 0.0

    def canonical(self, theta):
        """u >= 0 via the equator mirror (md_z and u negated): an axisymmetric component turns with its axis."""
        t = np.array(self._theta(theta), dtype=float)
        if t[3] < 0.0:
            t[2], t[3] = -t[2], -t[3]
        return t


@field.register("toy_axis_dipole")
class ToyAxisDipole(field.FieldModel):
    """Centred dipole on the fixed axis AXIS with free polar strength bd (MG); u = cos i."""

    names = ["bd", "u"]
    dim = 2

    def evaluate(self, points, theta):
        t = self._theta(theta)
        return field.component_field(np.asarray(points, dtype=float).reshape(-1, 3), AXIS, t[0], 1)

    def valid(self, theta):
        t = np.asarray(theta, dtype=float).reshape(-1)
        return t.shape[0] == self.dim and bool(np.all(np.isfinite(t)))

    def canonical(self, theta):
        return np.array(self._theta(theta), dtype=float)  # the fixed axis breaks the equator mirror: no folding


def _vec(v):
    return "[" + ", ".join(repr(float(a)) for a in v) + "]"


def config_text(model: str, parameters: str, phases=(0.0, 0.5), extra: str = "") -> str:
    blocks = "\n".join(
        f'[[data.phase]]\nphase = {ph}\nfile = "ph{k:02d}.dat"\nierr = "ph{k:02d}_ierr.dat"\nvierr = "ph{k:02d}_vierr.dat"\n'
        for k, ph in enumerate(phases))
    return f"""
[library]
path = "synth"
teffs = [30000.0]

[data]
path = "inj"
windows = [[4300.0, 6700.0]]
grid = {{ start = 4100.0, step = 2.0, npix = 1401 }}

{blocks}
[model]
field = "{model}"
fwhm = 3.0
surface = {{ scheme = "fibonacci", n = 1000 }}

[objective]
kind = "chi2"

[parameters]
{parameters}

[search]
n_starts = 4
restarts = 2
rounds_max = 1
lbfgs_maxiter = 60

[search.starts]
md = [200.0, 600.0]

[uncertainty]
patches = 1000
profile_quantities = []
{extra}"""


def dipole_l4_parameters(md, u):
    return (f"md = {{ value = {_vec(md)}, step = 10.0, lo = 150.0, hi = 600.0, free = true }}\n"
            f"u = {{ value = {u!r}, step = 0.1, lo = -1.0, hi = 1.0, free = true }}")


def axis_dipole_parameters(bd, u):
    return (f"bd = {{ value = {bd!r}, step = 10.0, lo = 150.0, hi = 600.0, free = true }}\n"
            f"u = {{ value = {u!r}, step = 0.1, lo = -1.0, hi = 1.0, free = true }}")


def angle_to_observer(md, u, phase=0.0):
    """Angle (degrees) between md and the observer direction at `phase` (forward.observers' convention)."""
    si = math.sqrt(max(0.0, 1.0 - u * u))
    a = 2.0 * math.pi * phase
    n = np.array([si * math.cos(a), -si * math.sin(a), -u])
    return math.degrees(math.acos(float(np.dot(md, n)) / float(np.linalg.norm(md))))


def _inject_and_fit(root, model: str, truth: str, start: str, out_name: str) -> int:
    """`inject` noise-free data from `truth`, then `fit` from `start`; returns fit's exit code."""
    (root / f"{out_name}_truth.toml").write_text(config_text(model, truth))
    (root / f"{out_name}_fit.toml").write_text(config_text(model, start))
    assert cli.main(["inject", str(root / f"{out_name}_truth.toml"), "-o", str(root / "inj"), "--noise", "none"]) == 0
    return cli.main(["fit", str(root / f"{out_name}_fit.toml"), "-o", str(root / out_name), "--seed", "1",
                     "--starts", "4", "--no-mcmc"])


@pytest.fixture()
def data_root(synth_lib, tmp_path, monkeypatch):
    root = tmp_path / "zeemap_data"
    root.mkdir()
    os.symlink(synth_lib, root / "synth", target_is_directory=True)
    monkeypatch.setenv("ZEEMAP_DATA", str(root))
    return root


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7e_fit_with_a_toy_field_model_registered_in_the_test_file(data_root):
    assert "toy_dipole_l4" in field.MODELS and ToyDipoleL4.__module__ == __name__  # registered here, not in src/
    root = data_root
    md_true = BD_TRUE * AXIS
    rc = _inject_and_fit(root, "toy_dipole_l4", dipole_l4_parameters(md_true, U_TRUE),
                         dipole_l4_parameters(300.0 * np.array([0.0, 0.0, 1.0]), 0.7), "fit")
    assert rc == 0
    out = root / "fit"
    for f in ("run.json", "modes.json", "diagnostics.json", "report.txt"):
        assert (out / f).is_file(), f
    run = json.loads((out / "run.json").read_text())
    assert 'field = "toy_dipole_l4"' in run["config"]
    doc = json.loads((out / "modes.json").read_text())
    assert doc["names"] == ["md_x", "md_y", "md_z", "u"]
    modes = doc["modes"]
    assert modes and modes[0]["delta"] == 0.0
    for k in range(1, len(modes) + 1):
        for ph in ("0.000", "0.500"):
            assert (out / "spectra" / f"mode{k}_ph{ph}.dat").is_file()
    # the truth, within a loose tolerance (noise-free data; V7c holds the built-in dipole to 1e-3 and 0.5 degrees)
    m = modes[0]
    md, u = np.array(m["parameters"]["md"]), m["parameters"]["u"]
    rel = abs(np.linalg.norm(md) / BD_TRUE - 1.0)
    angles = [angle_to_observer(md, u, ph) for ph in (0.0, 0.5)]
    true_angles = [angle_to_observer(md_true, U_TRUE, ph) for ph in (0.0, 0.5)]
    print(f"V7e: |md| {np.linalg.norm(md):.4f} MG (rel {rel:.2e}), angles {angles} vs {true_angles}, "
          f"-lnL {m['neg_log_l']:.3g}")
    assert rel < 1e-2
    assert max(abs(a - b) for a, b in zip(angles, true_angles)) < 2.0
    assert m["neg_log_l"] < 1e-2


BLOCKERS = (
    "a toy model with parameter names of its own cannot be configured or fitted without changes in src/: "
    "config.py PARAMETER_NAMES (lines 47-52; checked at line 358) rejects [parameters.bd]; "
    "problem.py Problem.derived (line 557, md = th[0:3]) and Problem.derived_batch (line 615) take the first three "
    "field numbers as the dipole; problem.py Problem.fingerprint_quantities (line 478) looks up md_x, md_y, md_z"
)


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
@pytest.mark.xfail(strict=True, reason=BLOCKERS)
def test_v7e_fit_with_a_toy_field_model_with_its_own_parameter_names(data_root):
    root = data_root
    rc = _inject_and_fit(root, "toy_axis_dipole", axis_dipole_parameters(BD_TRUE, U_TRUE),
                         axis_dipole_parameters(300.0, 0.7), "fit_axis")
    assert rc == 0
    m = json.loads((root / "fit_axis" / "modes.json").read_text())["modes"][0]
    assert abs(m["parameters"]["bd"] / BD_TRUE - 1.0) < 1e-2
    assert abs(math.degrees(math.acos(m["parameters"]["u"])) - 60.0) < 2.0


@pytest.mark.slow  # an end-to-end fit (tools/check.sh 1 --fast leaves it out)
def test_v7e_fit_with_harm4(data_root):
    # the built-in l <= 4 expansion with its own coefficient names (config, problem, search, outputs): a dipole plus
    # g20 and h32, the other coefficients fixed at 0; noise-free data must be fitted to the noise floor. The dipole
    # points at the observer of phase 0 so that V/I fixes the sign of B (with AXIS the net B . n is ~0 and 4 starts
    # found -B at -ln L 0.15). The fit holds md at its true value and searches g20, h32 and u: with md free too, the
    # two phases leave a second basin 0.02 nats above the truth that 15 of 16 starts reach, so 4 starts found the
    # truth only by the luck of one start, and on some CI runners not at all (-ln L 0.0199). With md held, 16 of 16
    # starts reach it. The dipole search is V7e's toy-model test.
    root = data_root
    md_true = BD_TRUE * np.array([math.sqrt(1.0 - U_TRUE**2), 0.0, -U_TRUE])

    def params(md, g20, h32, u):
        return (dipole_l4_parameters(md, u)
                + f"\ng20 = {{ value = {g20!r}, step = 5.0, lo = -100.0, hi = 100.0, free = true }}"
                + f"\nh32 = {{ value = {h32!r}, step = 5.0, lo = -100.0, hi = 100.0, free = true }}")

    rc = _inject_and_fit(root, "harm4", params(md_true, 60.0, 40.0, U_TRUE),
                         params(md_true, 0.0, 0.0, 0.7).replace("lo = 150.0, hi = 600.0, free = true",
                                                                "lo = 150.0, hi = 600.0, free = false"), "fit_harm4")
    assert rc == 0
    m = json.loads((root / "fit_harm4" / "modes.json").read_text())["modes"][0]
    print(f"V7e harm4: -lnL {m['neg_log_l']:.3g}, md {m['parameters']['md']}, g20 {m['parameters']['g20']:.3f}, "
          f"h32 {m['parameters']['h32']:.3f}, u {m['parameters']['u']:.4f}")
    assert m["neg_log_l"] < 1e-2
    assert abs(m["parameters"]["g20"] - 60.0) < 1.0 and abs(m["parameters"]["h32"] - 40.0) < 1.0
