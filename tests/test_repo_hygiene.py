"""Repo-hygiene tests: enforce the CLAUDE.md 'no raw data in git' rule in CI.

These tests exercise the dependency-free checker in ``tools/check_no_raw_data.py``
against the live repository (no network access): the current tree must be clean, and
the detection logic must actually flag forbidden extensions and oversized files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "tools" / "check_no_raw_data.py"


def _load_checker() -> ModuleType:
    """Import ``tools/check_no_raw_data.py`` as a module without needing it on sys.path."""
    spec = importlib.util.spec_from_file_location("check_no_raw_data", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_checker_script_exists() -> None:
    assert CHECKER_PATH.is_file(), "tools/check_no_raw_data.py must exist"


def test_repo_has_no_tracked_raw_data() -> None:
    """The real repo must not track any forbidden or oversized file."""
    violations = checker.find_violations(REPO_ROOT)
    assert violations == [], "tracked raw-data/oversized files: " + ", ".join(
        str(v) for v in violations
    )


def test_main_returns_zero_on_clean_repo() -> None:
    """The CLI entry point exits 0 when the repo is clean."""
    assert checker.main([str(REPO_ROOT)]) == 0


def test_detects_forbidden_extension(tmp_path: Path) -> None:
    """A tracked forbidden extension is flagged with a clear reason."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    offender = tmp_path / "weights.pth"
    offender.write_bytes(b"\x00\x01\x02")
    subprocess.run(["git", "add", "weights.pth"], cwd=tmp_path, check=True)

    violations = checker.find_violations(tmp_path)
    paths = {v.path for v in violations}
    assert "weights.pth" in paths
    assert checker.main([str(tmp_path)]) == 1


def test_detects_oversized_file(tmp_path: Path) -> None:
    """A tracked file above the size ceiling is flagged even with a benign extension."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    big = tmp_path / "huge.json"
    big.write_bytes(b"0" * (checker.MAX_TRACKED_BYTES + 1))
    subprocess.run(["git", "add", "huge.json"], cwd=tmp_path, check=True)

    violations = checker.find_violations(tmp_path)
    assert any(v.path == "huge.json" and "too large" in v.reason for v in violations)


def test_matched_forbidden_extension_handles_multidot() -> None:
    """SigMF data files use a hyphenated multi-dot suffix that plain .suffix would miss."""
    assert checker._matched_forbidden_extension("capture.sigmf-data") == ".sigmf-data"
    assert checker._matched_forbidden_extension("model.CKPT") == ".ckpt"
    assert checker._matched_forbidden_extension("leaderboard/splits/amc.idx.json") is None
