"""V8: repository hygiene, through tools/scan.py."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "tools" / "scan.py"


def run_scan(*args):
    return subprocess.run([sys.executable, str(SCAN), *args], capture_output=True, text=True)


def test_v8a_no_absolute_paths_in_tracked_files():
    r = run_scan()
    bad = [line for line in r.stdout.splitlines() if line.startswith("V8a ")]
    assert not bad, "\n".join(bad)


def test_v8b_no_data_files_or_large_files():
    r = run_scan()
    bad = [line for line in r.stdout.splitlines() if line.startswith("V8b ")]
    assert not bad, "\n".join(bad)


def test_v8c_no_forbidden_identifiers():
    r = run_scan()
    bad = [line for line in r.stdout.splitlines() if line.startswith("V8c ")]
    assert not bad, "\n".join(bad)
    assert r.returncode == 0, r.stdout + r.stderr


def test_v8d_scan_fails_on_a_planted_absolute_path(tmp_path):
    copy = tmp_path / "library.py"
    shutil.copy(ROOT / "src" / "zeemap" / "library.py", copy)
    planted = "/" + "home/someone/data/lib"  # built at run time so this test file itself stays clean
    copy.write_text(copy.read_text() + f'\nDEFAULT = "{planted}"\n')
    r = run_scan("--root", str(ROOT), str(copy))
    assert r.returncode == 1, r.stdout
    assert "V8a" in r.stdout and "library.py" in r.stdout


def test_v8d_scan_fails_on_a_planted_forbidden_identifier(tmp_path):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "forbidden_identifiers.txt").write_text("plantedname\n")
    src = root / "x.py"
    src.write_text("def plantedname(p):\n    return p\n")
    r = run_scan("--root", str(root), str(src))
    assert r.returncode == 1 and "V8c" in r.stdout, r.stdout
    src.write_text("def plantednames(p):\n    return p\n")  # not a whole word
    assert run_scan("--root", str(root), str(src)).returncode == 0
