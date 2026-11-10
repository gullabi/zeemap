"""V8e: the check runner (tools/check.sh with tests/conftest.py; TASKS.md, "The check runner"). A named item that selects
no test stops the run with its name, even next to items that do; `--fast` leaves out exactly the
tier-1 tests marked `slow` and labels its SUMMARY line. The runs only collect tests (pytest
--collect-only), so they cost seconds."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, **env):
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True, timeout=600)


def _collected(expr: str) -> set[str]:
    r = _run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", expr])
    assert r.returncode in (0, 5), r.stdout + r.stderr
    return {line.strip() for line in r.stdout.splitlines() if "::" in line}


def test_v8e_named_item_without_tests_fails_and_is_named():
    r = _run(["bash", "tools/check.sh", "1", "V8a", "V8z", "--", "--collect-only"], ZEEMAP_PY=sys.executable)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "--items: no test for V8z" in r.stdout + r.stderr
    r = _run(["bash", "tools/check.sh", "1", "V8a", "V8b", "--", "--collect-only"], ZEEMAP_PY=sys.executable)
    assert r.returncode == 0 and "SUMMARY tier 1: 0 passed, 0 failed, 0 errors, 0 skipped" in r.stdout, r.stdout


def test_v8e_fast_deselects_exactly_the_slow_tests(tmp_path):
    log = tmp_path / "args"
    py = tmp_path / "py"  # records the arguments check.sh passes to pytest
    py.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{log}"\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    r = _run(["bash", "tools/check.sh", "1", "--fast", "--", "--collect-only"], ZEEMAP_PY=str(py))
    assert r.returncode == 0 and "SUMMARY tier 1 (fast): 0 passed, 0 failed, 0 errors, 0 skipped" in r.stdout, \
        r.stdout + r.stderr
    args = log.read_text().splitlines()  # python -m pytest ... -m <marker expression> ...
    assert [args[i + 1] for i, a in enumerate(args[:-1]) if a == "-m"] == ["pytest", "tier1 and not slow"]
    every, fast, slow = _collected("tier1"), _collected("tier1 and not slow"), _collected("slow")
    assert slow and slow <= every and fast == every - slow
    assert all("test_v8e_" not in t for t in slow)
