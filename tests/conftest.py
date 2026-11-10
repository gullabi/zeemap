"""Test configuration: tier markers, item selection for tools/check.sh, shared synthetic libraries.

Rules (TASKS.md, "The check runner"): every test carries exactly one of the markers tier1, tier2, tier3, and its
name starts with test_v<k><letter>_ where check V<k> belongs to that tier; a test whose marker and name disagree fails.
The marker `slow` (end-to-end tier-1 fits, left out by `tools/check.sh 1 --fast`) comes in addition to the tier marker.
With `--items`, each named item must select at least one test, or the run stops with a usage error naming the items
that selected none (otherwise an item without tests among others would go unnoticed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/synth.py importable as `synth`

TIER_OF_CHECK = {**{k: 1 for k in range(1, 9)}, 14: 1, **{k: 2 for k in range(9, 14)}, **{k: 3 for k in range(15, 20)}}
NAME_RE = re.compile(r"^test_v(\d+)([a-z]?)_")


def pytest_addoption(parser):
    parser.addoption(
        "--items",
        nargs="*",
        default=None,
        metavar="V<k>[<letter>]",
        help="run only the tests of these check items (by name prefix test_v2c_, test_v2*_ ...)",
    )
    parser.addoption("--seeds", type=int, default=None, help="number of seeds for the search campaigns (V17)")


def _item_check(item) -> tuple[int, str] | None:
    m = NAME_RE.match(item.name)
    return (int(m.group(1)), m.group(2)) if m else None


def pytest_collection_modifyitems(config, items):
    wanted = config.getoption("--items")
    if not wanted:
        return
    pats = []
    for w in wanted:
        m = re.fullmatch(r"V(\d+)([a-z]?)", w)
        if not m:
            raise pytest.UsageError(f"--items: {w!r} is not a check item (V2, V2c, ...)")
        pats.append((int(m.group(1)), m.group(2)))
    keep, drop, hit = [], [], set()
    for it in items:
        c = _item_check(it)
        got = [p for p in pats if c is not None and c[0] == p[0] and (not p[1] or c[1] == p[1])]
        hit.update(got)
        (keep if got else drop).append(it)
    missing = [w for w, p in zip(wanted, pats) if p not in hit]
    if missing:
        raise pytest.UsageError(f"--items: no test for {', '.join(missing)} (each named item must select a test)")
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_runtest_setup(item):
    tiers = [int(m.name[4]) for m in item.iter_markers() if re.fullmatch(r"tier[123]", m.name)]
    if len(tiers) != 1:
        pytest.fail(f"{item.nodeid}: needs exactly one marker tier1/tier2/tier3, has {tiers}")
    if item.get_closest_marker("slow") is not None and tiers[0] != 1:
        pytest.fail(f"{item.nodeid}: the marker slow is for tier-1 tests only (tools/check.sh 1 --fast)")
    c = _item_check(item)
    if c is None:
        pytest.fail(f"{item.nodeid}: test name must start with test_v<k><letter>_")
    expected = TIER_OF_CHECK.get(c[0])
    if expected is None:
        pytest.fail(f"{item.nodeid}: no check V{c[0]} exists")
    if expected != tiers[0]:
        pytest.fail(f"{item.nodeid}: check V{c[0]} belongs to tier {expected}, but the test is marked tier{tiers[0]}")


@pytest.fixture(scope="session")
def synth_lib(tmp_path_factory):
    """Directory of the synthetic library, variant a (VERIFICATION.md section 4), written once per session."""
    import synth

    d = tmp_path_factory.mktemp("synth_a")
    synth.write_library(d, "a")
    return d


@pytest.fixture(scope="session")
def synth_lib_b(tmp_path_factory):
    """Variant b (line shifts 0.022 instead of 0.02), the stand-in for another library version (V18a)."""
    import synth

    d = tmp_path_factory.mktemp("synth_b")
    synth.write_library(d, "b")
    return d
