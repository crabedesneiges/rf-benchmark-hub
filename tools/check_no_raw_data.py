#!/usr/bin/env python3
"""Repo-hygiene guard: no raw data ever enters git (CLAUDE.md, IMPLEMENTATION_PLAN §10).

RF-Benchmark-Hub never redistributes datasets. Only split *indices* and *checksums*
are versioned (``leaderboard/splits/``); the raw signals are downloaded on demand via
``rfbench data prepare``. This script is the CI guard-rail behind that rule: it lists
every git-tracked file and fails (nonzero exit) if any of them either

* carries a forbidden raw-data / weights extension, or
* is unreasonably large (> ``MAX_TRACKED_BYTES``),

printing each offender with the reason. It is intentionally dependency-free and typed
so it runs anywhere ``python`` and ``git`` exist (``python tools/check_no_raw_data.py``),
and its core (:func:`find_violations`) is importable for the hygiene test.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Raw signal captures, packed arrays, and model weights: never versioned.
# Mirrors the git-ignored extensions in .gitignore and the CLAUDE.md rule.
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".h5",
        ".hdf5",
        ".npy",
        ".npz",
        ".bin",
        ".sigmf-data",
        ".pt",
        ".ckpt",
        ".pth",
    }
)

# Hard size ceiling for any single tracked file. Split-index JSON + checksums are tiny;
# anything above this is almost certainly a dataset or checkpoint that slipped in.
MAX_TRACKED_BYTES: int = 5 * 1024 * 1024  # 5 MiB


@dataclass(frozen=True)
class Violation:
    """A single tracked file that breaks the no-raw-data rule."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _matched_forbidden_extension(path: str) -> str | None:
    """Return the forbidden extension a path ends with (case-insensitive), else None.

    Matches on the full lowercased name so multi-dot suffixes like ``.sigmf-data``
    are caught even though :attr:`pathlib.PurePath.suffix` would only see ``-data``.
    """
    lowered = path.lower()
    for ext in FORBIDDEN_EXTENSIONS:
        if lowered.endswith(ext):
            return ext
    return None


def list_tracked_files(repo_root: Path) -> list[str]:
    """Return repo-relative paths of all git-tracked files under ``repo_root``.

    Uses ``git ls-files -z`` so paths with spaces/newlines are handled safely.
    Raises :class:`RuntimeError` if the directory is not a git working tree.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - git always present in CI
        raise RuntimeError("git executable not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"`git ls-files` failed: {stderr}") from exc

    raw = completed.stdout.decode("utf-8", "surrogateescape")
    return [entry for entry in raw.split("\0") if entry]


def find_violations(repo_root: Path) -> list[Violation]:
    """Scan tracked files under ``repo_root`` and return every hygiene violation.

    A file is flagged if it ends with a :data:`FORBIDDEN_EXTENSIONS` suffix or its
    on-disk size exceeds :data:`MAX_TRACKED_BYTES`. Results are sorted by path for
    deterministic output. Tracked files missing from the worktree (e.g. deleted but
    not yet committed) are skipped for the size check.
    """
    violations: list[Violation] = []
    for rel_path in list_tracked_files(repo_root):
        ext = _matched_forbidden_extension(rel_path)
        if ext is not None:
            violations.append(Violation(rel_path, f"forbidden raw-data extension '{ext}'"))

        absolute = repo_root / rel_path
        try:
            size = absolute.stat().st_size
        except OSError:
            # Tracked-but-absent (staged deletion, sparse checkout): nothing to size.
            continue
        if size > MAX_TRACKED_BYTES:
            violations.append(
                Violation(
                    rel_path,
                    f"file too large ({size} bytes > {MAX_TRACKED_BYTES} byte limit)",
                )
            )

    return sorted(violations, key=lambda v: (v.path, v.reason))


def main(argv: list[str] | None = None) -> int:
    """Entry point: scan the repo and report. Return 0 if clean, 1 otherwise."""
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path(args[0]).resolve() if args else Path.cwd()

    try:
        violations = find_violations(repo_root)
    except RuntimeError as exc:
        print(f"check_no_raw_data: {exc}", file=sys.stderr)
        return 2

    if violations:
        print(
            "check_no_raw_data: raw data / oversized files must not be tracked in git",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            f"check_no_raw_data: {len(violations)} offending file(s); "
            "see CLAUDE.md 'no raw data in git'.",
            file=sys.stderr,
        )
        return 1

    print("check_no_raw_data: OK - no raw-data or oversized tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
