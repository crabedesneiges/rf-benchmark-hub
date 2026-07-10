"""WP-53 tests for the maintainer verification pipeline (:mod:`rfbench.verify`).

Pure stdlib + jsonschema only: no numpy/torch/torchsig, no network, no real model. The re-run's
recomputed metrics are injected directly (``rerun_metrics``), so the whole self_reported ->
verified flip is exercised without any heavy dependency.

Covers, per WP-53 acceptance ("a demo run flips a score from self_reported to verified"):
  * within-tolerance re-run -> ``verified`` with verified_by/verified_date/verified_hardware/method
    stamped and the output schema-valid;
  * out-of-tolerance re-run -> stays ``self_reported`` (original returned untouched);
  * incomplete manifest -> rejected (stays ``self_reported``);
  * tolerance resolution (absolute / relative / per_metric);
  * ``import rfbench.verify`` pulls in no heavy dependency.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from rfbench.verify import (
    MetricCheck,
    VerificationError,
    VerificationReport,
    rerun_metrics_from_result,
    verify_result,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- fixtures (a self_reported result + a complete manifest describing the same run) ------


def _self_reported_result() -> dict[str, Any]:
    """A minimal, schema-valid AMC result in the self_reported state."""
    return {
        "schema_version": "1.0.0",
        "task": {"name": "amc", "version": "v1"},
        "model": {"name": "mcldnn"},
        "regime": {"name": "full_finetune"},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "amc-radioml2016-strat-snr-8010-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": "sha256:" + "3b" * 32,
        },
        "metrics": {
            "primary": "accuracy_overall",
            "values": {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        },
        "verification": {"status": "self_reported"},
    }


def _complete_manifest() -> dict[str, Any]:
    """A complete Tier-2 manifest (valid against submission.schema.json) for the result above."""
    return {
        "schema_version": "1.0.0",
        "result_path": "leaderboard/results/amc/mcldnn.json",
        "task": {"name": "amc", "version": "v1"},
        "regime": {"name": "full_finetune"},
        "code_commit": "git@1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
        "command": "rfbench eval amc --model mcldnn --regime full_finetune --seed 42",
        "artifacts": {
            "weights_url": "https://zenodo.org/records/0/files/mcldnn.pt",
        },
        "hardware": "1x NVIDIA A100 80GB",
        "expected_metrics": {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        "tolerance": {"absolute": 0.01},
    }


@pytest.fixture
def result() -> dict[str, Any]:
    return _self_reported_result()


@pytest.fixture
def manifest() -> dict[str, Any]:
    return _complete_manifest()


# --- within tolerance -> verified with stamps ----------------------------------------


def test_within_tolerance_flips_to_verified(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """A re-run within the manifest's absolute tolerance verifies with full provenance."""
    rerun = {"accuracy_overall": 0.6100, "macro_f1": 0.5990}  # |delta| 0.0023 <= 0.01
    report = verify_result(
        result,
        manifest,
        rerun,
        verified_by="rf-bench-maintainers",
        verified_hardware="4x NVIDIA GB200",
        verified_date=date(2026, 6, 20),
        method="eval_only",
    )
    assert report.verified is True
    verification = report.result["verification"]
    assert verification["status"] == "verified"
    assert verification["verified_by"] == "rf-bench-maintainers"
    assert verification["verified_date"] == "2026-06-20"
    assert verification["verified_hardware"] == "4x NVIDIA GB200"
    assert verification["method"] == "eval_only"
    assert verification["tolerance"] == pytest.approx(0.01)
    assert verification["observed_primary"] == pytest.approx(0.6100)


def test_verified_output_is_schema_valid(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """The stamped verified result validates against result.schema.json (no missing provenance)."""
    from jsonschema import Draft202012Validator

    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123},
        verified_by="maint",
        verified_hardware="4x GB200",
    )
    assert report.verified is True
    schema = _load_schema("result.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(report.result))
    assert errors == [], [e.message for e in errors]


def test_verify_does_not_mutate_input(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """The submitted result is never mutated; the verified doc is a fresh copy."""
    before = copy.deepcopy(result)
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123},
        verified_by="maint",
        verified_hardware="hw",
    )
    assert result == before  # untouched
    assert report.result is not result
    assert result["verification"]["status"] == "self_reported"


def test_verified_date_defaults_to_today(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Omitting verified_date stamps today's ISO date."""
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123},
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.result["verification"]["verified_date"] == date.today().isoformat()


def test_method_defaults_to_manifest_rerun_mode(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """When method is not passed, it defaults to the manifest's rerun_mode."""
    manifest["rerun_mode"] = "full_retrain"
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123},
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.result["verification"]["method"] == "full_retrain"


# --- out of tolerance -> stays self_reported -----------------------------------------


def test_out_of_tolerance_stays_self_reported(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """A primary metric outside tolerance keeps the row self_reported and returns the original."""
    rerun = {"accuracy_overall": 0.55}  # |delta| 0.0623 > 0.01
    report = verify_result(
        result,
        manifest,
        rerun,
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.verified is False
    assert report.result is result
    assert report.result["verification"]["status"] == "self_reported"
    assert report.primary_check is not None
    assert report.primary_check.within is False
    assert "out of tolerance" in report.summary()


def test_secondary_metric_out_of_tolerance_vetoes(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """An in-tolerance primary but out-of-tolerance secondary metric still fails."""
    rerun = {"accuracy_overall": 0.6123, "macro_f1": 0.10}  # macro_f1 far off
    report = verify_result(
        result,
        manifest,
        rerun,
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.verified is False
    assert report.result["verification"]["status"] == "self_reported"
    offending = [c for c in report.checks if not c.within]
    assert [c.name for c in offending] == ["macro_f1"]


# --- incomplete manifest -> rejected --------------------------------------------------


def test_incomplete_manifest_rejected(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """A manifest missing required repro fields cannot verify (stays self_reported)."""
    del manifest["expected_metrics"]  # drop a required field
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123},
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.verified is False
    assert report.result["verification"]["status"] == "self_reported"
    assert any("manifest" in e for e in report.errors)


def test_manifest_missing_primary_metric_rejected(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """A complete manifest whose expected_metrics lacks the primary metric cannot verify."""
    manifest["expected_metrics"] = {"macro_f1": 0.5987}  # no accuracy_overall (the primary)
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.verified is False
    assert any("primary metric" in e for e in report.errors)


def test_rerun_missing_primary_metric_rejected(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """If the re-run never measured the primary metric, the row cannot verify."""
    report = verify_result(
        result,
        manifest,
        {"macro_f1": 0.5987},  # no accuracy_overall in the re-run
        verified_by="maint",
        verified_hardware="hw",
    )
    assert report.verified is False
    assert any("primary metric" in e for e in report.errors)


# --- guard rails ----------------------------------------------------------------------


def test_empty_verified_by_raises(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """A blank maintainer handle is a hard error (can never produce an unsigned verified row)."""
    with pytest.raises(VerificationError, match="verified_by"):
        verify_result(
            result,
            manifest,
            {"accuracy_overall": 0.6123},
            verified_by="",
            verified_hardware="hw",
        )


def test_empty_hardware_raises(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    with pytest.raises(VerificationError, match="verified_hardware"):
        verify_result(
            result,
            manifest,
            {"accuracy_overall": 0.6123},
            verified_by="maint",
            verified_hardware="",
        )


# --- tolerance resolution -------------------------------------------------------------


def test_relative_tolerance(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    """A relative tolerance is turned into an absolute bound via relative * abs(expected)."""
    manifest["tolerance"] = {"relative": 0.02}  # 0.02 * 0.6123 ~= 0.01225
    within = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123 - 0.012},
        verified_by="m",
        verified_hardware="h",
    )
    assert within.verified is True
    outside = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123 - 0.02},
        verified_by="m",
        verified_hardware="h",
    )
    assert outside.verified is False


def test_per_metric_tolerance_overrides_global(
    result: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """A per_metric bound overrides the global one for that metric."""
    manifest["tolerance"] = {"absolute": 0.001, "per_metric": {"accuracy_overall": 0.05}}
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123 - 0.04, "macro_f1": 0.5987},
        verified_by="m",
        verified_hardware="h",
    )
    assert report.verified is True
    assert report.primary_check is not None
    assert report.primary_check.tolerance == pytest.approx(0.05)


# --- helper API -----------------------------------------------------------------------


def test_rerun_metrics_from_result_extracts_values() -> None:
    """metrics.values of a re-run result.json is pulled into the rerun_metrics mapping."""
    rerun_doc = {"metrics": {"primary": "accuracy_overall", "values": {"accuracy_overall": 0.61}}}
    assert rerun_metrics_from_result(rerun_doc) == {"accuracy_overall": 0.61}


def test_rerun_metrics_from_result_requires_values() -> None:
    with pytest.raises(VerificationError, match="metrics.values"):
        rerun_metrics_from_result({"metrics": {"primary": "x"}})


def test_metric_check_delta() -> None:
    """MetricCheck.delta is the signed observed - expected difference."""
    check = MetricCheck(name="m", expected=0.6, observed=0.62, tolerance=0.05, within=True)
    assert check.delta == pytest.approx(0.02)


def test_report_status_property(result: dict[str, Any]) -> None:
    """VerificationReport.status reflects the wrapped result's verification.status."""
    report = VerificationReport(verified=False, result=result)
    assert report.status == "self_reported"


# --- import purity --------------------------------------------------------------------


def test_importing_verify_pulls_no_heavy_deps() -> None:
    """Importing rfbench.verify in a fresh interpreter loads no numpy/torch/jsonschema/etc."""
    code = (
        "import sys, rfbench.verify;"
        "heavy={'numpy','h5py','torch','torchsig','requests','jsonschema'};"
        "leaked=sorted(heavy & set(sys.modules));"
        "print(leaked);"
        "sys.exit(1 if leaked else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, f"heavy deps leaked on import: {proc.stdout.strip()}"


# --- shared helper --------------------------------------------------------------------


def _load_schema(name: str) -> dict[str, Any]:
    import json

    return json.loads((_REPO_ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_source_only_artifact_flips_to_verified() -> None:
    """A schema-1.1.0 manifest with `artifacts.source_only` (no weights/image) flips to verified.

    The honest artifact form for a deterministic from-source seed baseline: reproducible from
    code_commit + the exact command + the committed splits + uv.lock, with no external artifact.
    verify_result validates the manifest against submission.schema.json first, so a successful
    flip proves the source_only manifest is schema-valid AND accepted by the pipeline.
    """
    result = _self_reported_result()
    manifest = _complete_manifest()
    manifest["schema_version"] = "1.1.0"
    manifest["artifacts"] = {"source_only": True}

    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        verified_by="rf-bench-maintainers",
        verified_hardware="1x Dalia defq node (ARM Neoverse V2, CPU-only)",
        method="full_retrain",
    )
    assert report.verified is True
    assert report.result["verification"]["status"] == "verified"
    assert report.result["verification"]["method"] == "full_retrain"


def test_empty_artifacts_manifest_is_rejected() -> None:
    """An artifacts block with none of weights_url/docker_image/source_only stays self_reported."""
    result = _self_reported_result()
    manifest = _complete_manifest()
    manifest["artifacts"] = {}  # violates artifacts.anyOf + minProperties
    report = verify_result(
        result,
        manifest,
        {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        verified_by="rf-bench-maintainers",
        verified_hardware="1x Dalia defq node",
    )
    assert report.verified is False
    assert any("manifest" in e for e in report.errors)
