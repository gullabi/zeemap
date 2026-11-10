#!/usr/bin/env python3
"""Repository hygiene scan (VERIFICATION.md V8; SPEC.md ENG-4, ENG-5).

Scans the files tracked by git (or the files given on the command line) for:
  V8a  absolute paths: the patterns in PATTERNS, except lines allowed in tools/scan_allow.txt (`path:text` per line:
       a line of that file containing that text is allowed);
  V8b  files with the extensions .bin, .dat, .fits, .npz outside tests/fixtures/ and tests/regression/, and any
       file over 1 MB;
  V8c  identifiers listed in tools/forbidden_identifiers.txt (one per line; whole words, case-sensitive).
The two list files themselves are not scanned for V8a/V8c (they hold the patterns).
Usage: tools/scan.py [--root DIR] [FILE ...]      exit 1 if anything is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = ["/home/", "/Users/", "/tmp/", "C:\\"]
DATA_EXT = {".bin", ".dat", ".fits", ".npz"}
DATA_DIRS = ("tests/fixtures/", "tests/regression/")
MAX_BYTES = 1_000_000
LIST_FILES = {"tools/scan_allow.txt", "tools/forbidden_identifiers.txt"}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [root / f for f in out.decode().split("\0") if f]


def load_allow(root: Path) -> list[tuple[str, str]]:
    p = root / "tools" / "scan_allow.txt"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        path, _, text = line.partition(":")
        out.append((path.strip(), text))
    return out


def load_forbidden(root: Path) -> list[str]:
    p = root / "tools" / "forbidden_identifiers.txt"
    if not p.is_file():
        return []
    return [w.strip() for w in p.read_text().splitlines() if w.strip() and not w.startswith("#")]


def scan(root: Path, files: list[Path]) -> list[str]:
    allow = load_allow(root)
    forbidden = load_forbidden(root)
    forbidden_re = re.compile(r"\b(" + "|".join(re.escape(w) for w in forbidden) + r")\b") if forbidden else None
    findings = []
    for f in files:
        if not f.is_file():
            continue
        try:
            rel = f.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = f.as_posix()
        size = f.stat().st_size
        if size > MAX_BYTES:
            findings.append(f"V8b {rel}: {size} bytes, over {MAX_BYTES}")
        if f.suffix.lower() in DATA_EXT and not rel.startswith(DATA_DIRS):
            findings.append(f"V8b {rel}: data file extension {f.suffix} outside {' or '.join(DATA_DIRS)}")
        if rel in LIST_FILES:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary: V8b covers it
        for n, line in enumerate(text.splitlines(), 1):
            for pat in PATTERNS:
                if pat in line and not any(rel == ap and at in line for ap, at in allow):
                    findings.append(f"V8a {rel}:{n}: absolute path pattern {pat!r}")
            if forbidden_re and (m := forbidden_re.search(line)):
                findings.append(f"V8c {rel}:{n}: forbidden identifier {m.group(1)!r}")
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="repository root (default: the parent of tools/)")
    ap.add_argument("files", nargs="*", help="files to scan instead of the tracked files")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else Path(__file__).resolve().parents[1]
    files = [Path(f) for f in a.files] if a.files else tracked_files(root)
    findings = scan(root, files)
    for line in findings:
        print(line)
    print(f"SUMMARY scan: {len(files)} files, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
