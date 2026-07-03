#!/usr/bin/env python3
"""Docs drift gate: fail CI when documentation resurrects removed features.

Part of the post-2026-07 doc-drift cleanup (RFC 0001, D17/D20). The banned
tokens below name modules, commands, and endpoints that were removed or never
built: the ``ml_server`` module spelling, the ``cli_v2`` module, the ``ainit``
command, the ``/readyz`` and ``/startupz`` probes, the ``/batch_predict``
endpoint, and the ``max_concurrent_requests`` setting. When one of them
reappears in user-facing docs or examples it almost always means stale text
was copy-pasted back in, so this gate blocks the drift at CI time.

Historical material is exempt: ``docs/archive/`` (frozen history) and
``docs/rfcs/`` (decision records that deliberately name these tokens --
RFC 0001's D17 row defines this very gate).

Usage: python3 scripts/check_docs_drift.py [--root PATH]

Exit status: 0 when clean; 1 with one ``file:line: pattern`` line per hit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (label, compiled regex) pairs. Case-sensitive. \b guards tokens that are
# substrings of legitimate words, e.g. `mlserver` must never match
# `ml_server`, and `maintain`-style words must never match `ainit`.
BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ml_server", re.compile(r"\bml_server\b")),
    ("cli_v2", re.compile(r"cli_v2")),
    ("ainit", re.compile(r"\bainit\b")),
    ("/readyz", re.compile(r"/readyz")),
    ("/startupz", re.compile(r"/startupz")),
    ("batch_predict", re.compile(r"\bbatch_predict\b")),
    ("max_concurrent_requests", re.compile(r"max_concurrent_requests")),
]

# Documentation surfaces to scan, relative to the repo root.
SCAN_FILES = ["README.md", "CLAUDE.md"]
SCAN_DIRS = ["docs", "examples"]
# Historical material that legitimately mentions the banned tokens.
EXCLUDED_DIRS = ["docs/archive", "docs/rfcs"]
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".txt", ".ipynb"}


def iter_candidate_files(root: Path):
    """Yield the text files that make up the scanned documentation surfaces."""
    for name in SCAN_FILES:
        path = root / name
        if path.is_file():
            yield path
    excluded = [root / d for d in EXCLUDED_DIRS]
    for dirname in SCAN_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if any(excl in path.parents for excl in excluded):
                continue
            yield path


def scan_file(path: Path, root: Path) -> list[str]:
    """Return ``file:line: pattern`` hit strings for one file."""
    hits = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, pattern in BANNED_PATTERNS:
            if pattern.search(line):
                hits.append(f"{path.relative_to(root)}:{lineno}: {label}")
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when docs mention removed/never-built features."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root to scan (default: parent of scripts/).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    hits: list[str] = []
    for path in iter_candidate_files(root):
        hits.extend(scan_file(path, root))

    if hits:
        print(f"Docs drift detected ({len(hits)} hit(s)):", file=sys.stderr)
        for hit in hits:
            print(hit)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
