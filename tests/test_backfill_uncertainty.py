"""Tests for ``scripts/backfill_uncertainty.py`` -- Wilson-score CI backfill.

The script is loaded by path (``importlib``) like the other standalone scripts under
``tests/`` so it needs no package install. These tests cover the Wilson formula on a known
case, eligibility filtering (proportion metric + harness tier only), idempotence and the
strict exclusion of literature (``from_paper*``) rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_uncertainty.py"


def _load_backfill() -> ModuleType:
    """Import ``scripts/backfill_uncertainty.py`` by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location("backfill_uncertainty", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load_backfill()


# --------------------------------------------------------------------------------------------------
# Fixtures: minimal, schema-valid rows written to tmp_path.
# --------------------------------------------------------------------------------------------------
def _amc_row(
    *,
    model_name: str,
    accuracy: float,
    n_samples: int,
    status: str,
    schema_version: str = "1.0.0",
    primary: str = "accuracy_overall",
    include_eval: bool = True,
) -> dict[str, Any]:
    """Build a schema-valid AMC-shaped result row (primary metric = a proportion)."""
    verification: dict[str, Any] = {"status": status}
    if status == "verified":
        verification.update(
            verified_by="tester",
            verified_date="2026-06-30",
            verified_hardware="4x NVIDIA GB200",
            method="eval_only",
        )
    row: dict[str, Any] = {
        "schema_version": schema_version,
        "task": {"name": "amc", "version": "v1"},
        "model": {"name": model_name, "family": "baseline", "n_params": 100000},
        "regime": {"name": "from_scratch"},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "amc-strat-snr-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": "sha256:" + "ab" * 32,
        },
        "metrics": {"primary": primary, "values": {primary: accuracy, "macro_f1": accuracy}},
        "environment": {"seed": 42, "rfbench_version": "0.1.0", "python_version": "3.11.0"},
        "verification": verification,
    }
    if include_eval:
        row["eval"] = {"conditions": {}, "n_samples": n_samples}
    return row


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# Wilson formula
# --------------------------------------------------------------------------------------------------
def test_wilson_interval_known_case() -> None:
    """Wilson interval for p=0.5, n=100 matches the textbook 95% value (~[0.404, 0.596])."""
    lo, hi = backfill.wilson_interval(0.5, 100)
    # Centre stays at 0.5 by symmetry; half-width ~0.096 for this classic case.
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert lo < 0.5 < hi


def test_wilson_interval_is_bounded_in_unit_range() -> None:
    """Even at extreme proportions the interval never leaves [0, 1]."""
    lo, hi = backfill.wilson_interval(0.9987, 750)
    assert 0.0 <= lo <= hi <= 1.0
    # Near the upper boundary the interval is asymmetric (Wilson, not the naive normal).
    assert hi - 0.9987 < 0.9987 - lo


def test_wilson_interval_narrows_with_n() -> None:
    """More samples -> a tighter interval for the same proportion."""
    lo_small, hi_small = backfill.wilson_interval(0.6, 100)
    lo_big, hi_big = backfill.wilson_interval(0.6, 10000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_rejects_bad_inputs() -> None:
    """Non-positive n or out-of-range p raise ``ValueError``."""
    with pytest.raises(ValueError):
        backfill.wilson_interval(0.5, 0)
    with pytest.raises(ValueError):
        backfill.wilson_interval(1.5, 100)


# --------------------------------------------------------------------------------------------------
# Eligibility + document mutation
# --------------------------------------------------------------------------------------------------
def test_eligible_self_reported_proportion_row_is_backfilled() -> None:
    """A self_reported accuracy row gains a wilson_backfill CI and is bumped to 1.2.0."""
    row = _amc_row(model_name="cldnn", accuracy=0.5805, n_samples=22000, status="self_reported")
    assert backfill._eligibility_reason(row) is None
    assert backfill.backfill_document(row) is True

    entry = row["metrics"]["uncertainty"]["accuracy_overall"]
    assert entry["method"] == "wilson_backfill"
    assert entry["confidence"] == 0.95
    assert entry["ci_low"] <= 0.5805 <= entry["ci_high"]
    assert "note" in entry and "backfill" in entry["note"]
    assert row["schema_version"] == "1.2.0"


def test_from_paper_rows_are_excluded() -> None:
    """``from_paper`` / ``from_paper_uncertain`` rows are NEVER backfilled."""
    for status in ("from_paper", "from_paper_uncertain"):
        row = _amc_row(
            model_name="paper",
            accuracy=0.381,
            n_samples=0,
            status=status,
            schema_version="1.1.0",
            include_eval=False,  # literature rows carry no eval block
        )
        reason = backfill._eligibility_reason(row)
        assert reason is not None and "harness tier" in reason
        assert backfill.backfill_document(row) is False
        assert "uncertainty" not in row["metrics"]


def test_non_proportion_metric_is_excluded() -> None:
    """A non-proportion primary (e.g. mAP) is not Wilson-backfillable."""
    row = _amc_row(
        model_name="detector", accuracy=0.42, n_samples=5000, status="self_reported", primary="mAP"
    )
    reason = backfill._eligibility_reason(row)
    assert reason is not None and "binomial proportion" in reason


def test_missing_n_samples_is_excluded() -> None:
    """No ``eval.n_samples`` -> nothing to reconstruct a Wilson interval from."""
    row = _amc_row(
        model_name="cldnn", accuracy=0.58, n_samples=0, status="self_reported", include_eval=False
    )
    reason = backfill._eligibility_reason(row)
    assert reason is not None and "n_samples" in reason


def test_verified_rank1_accuracy_is_eligible() -> None:
    """``rank1_accuracy`` (the SEI closed-set proportion) is also backfillable when verified."""
    row = _amc_row(
        model_name="sei", accuracy=0.83, n_samples=3000, status="verified", primary="rank1_accuracy"
    )
    assert backfill._eligibility_reason(row) is None
    assert backfill.backfill_document(row) is True
    assert "rank1_accuracy" in row["metrics"]["uncertainty"]


# --------------------------------------------------------------------------------------------------
# Tree walk: writing, idempotence, validation
# --------------------------------------------------------------------------------------------------
def test_backfill_tree_writes_and_revalidates(tmp_path: Path) -> None:
    """The tree walk mutates eligible files on disk, keeping them schema-valid."""
    results = tmp_path / "results" / "amc"
    _write(
        results / "cldnn.json",
        _amc_row(model_name="cldnn", accuracy=0.5805, n_samples=22000, status="self_reported"),
    )
    changed = backfill.backfill_tree(tmp_path / "results")
    assert changed == 1

    on_disk = json.loads((results / "cldnn.json").read_text(encoding="utf-8"))
    assert on_disk["metrics"]["uncertainty"]["accuracy_overall"]["method"] == "wilson_backfill"
    assert on_disk["schema_version"] == "1.2.0"
    # Re-validating the mutated document against the repo schema must pass.
    backfill._validate_or_raise(on_disk)


def test_backfill_tree_is_idempotent(tmp_path: Path) -> None:
    """A second pass changes nothing (already-backfilled rows are skipped)."""
    results = tmp_path / "results" / "amc"
    _write(
        results / "cldnn.json",
        _amc_row(model_name="cldnn", accuracy=0.5805, n_samples=22000, status="self_reported"),
    )
    assert backfill.backfill_tree(tmp_path / "results") == 1
    before = (results / "cldnn.json").read_text(encoding="utf-8")
    assert backfill.backfill_tree(tmp_path / "results") == 0
    after = (results / "cldnn.json").read_text(encoding="utf-8")
    assert before == after  # byte-identical: no spurious rewrite


def test_backfill_tree_skips_from_paper_files(tmp_path: Path) -> None:
    """A results tree of only literature rows yields zero changes."""
    results = tmp_path / "results" / "amc"
    _write(
        results / "iqfm_paper.json",
        _amc_row(
            model_name="iqfm",
            accuracy=0.381,
            n_samples=0,
            status="from_paper",
            schema_version="1.1.0",
            include_eval=False,
        ),
    )
    assert backfill.backfill_tree(tmp_path / "results") == 0
    on_disk = json.loads((results / "iqfm_paper.json").read_text(encoding="utf-8"))
    assert "uncertainty" not in on_disk["metrics"]
    assert on_disk["schema_version"] == "1.1.0"  # untouched


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``dry_run=True`` reports the change count without touching disk."""
    results = tmp_path / "results" / "amc"
    _write(
        results / "cldnn.json",
        _amc_row(model_name="cldnn", accuracy=0.5805, n_samples=22000, status="self_reported"),
    )
    before = (results / "cldnn.json").read_text(encoding="utf-8")
    assert backfill.backfill_tree(tmp_path / "results", dry_run=True) == 1
    after = (results / "cldnn.json").read_text(encoding="utf-8")
    assert before == after
