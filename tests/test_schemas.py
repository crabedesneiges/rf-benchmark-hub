"""WP-01 verification tests for the RF-Benchmark-Hub JSON schemas.

Acceptance criterion (IMPLEMENTATION_PLAN.md §8, WP-01): a valid and an invalid
example are tested with ``jsonschema`` in CI. These tests treat
``schemas/result.schema.json`` and ``schemas/submission.schema.json`` as the
frozen source of truth and assert that:

* both schemas are themselves valid JSON Schema Draft 2020-12,
* ``schemas/examples/result.valid.json`` validates against the result schema,
* ``schemas/examples/result.invalid.json`` is rejected by it,
* ``schemas/examples/submission.valid.json`` validates against the submission
  schema.

The schemas and example fixtures are owned elsewhere and MUST NOT be edited from
here; this module is read-only with respect to them.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

# Repository layout: this file lives at <repo>/tests/test_schemas.py, so the
# schemas directory is two levels up. Resolve everything from here so the tests
# are independent of the current working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SCHEMAS_DIR: Path = REPO_ROOT / "schemas"
EXAMPLES_DIR: Path = SCHEMAS_DIR / "examples"

RESULT_SCHEMA_PATH: Path = SCHEMAS_DIR / "result.schema.json"
SUBMISSION_SCHEMA_PATH: Path = SCHEMAS_DIR / "submission.schema.json"

RESULT_VALID_PATH: Path = EXAMPLES_DIR / "result.valid.json"
RESULT_INVALID_PATH: Path = EXAMPLES_DIR / "result.invalid.json"
SUBMISSION_VALID_PATH: Path = EXAMPLES_DIR / "submission.valid.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a UTF-8 JSON document, returning its top-level object."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"expected a JSON object at {path}, got {type(data).__name__}"
    return data


def _result_schema() -> dict[str, Any]:
    return _load_json(RESULT_SCHEMA_PATH)


def _submission_schema() -> dict[str, Any]:
    return _load_json(SUBMISSION_SCHEMA_PATH)


def test_schema_files_exist() -> None:
    """The frozen schemas and their example fixtures must be present on disk."""
    for path in (
        RESULT_SCHEMA_PATH,
        SUBMISSION_SCHEMA_PATH,
        RESULT_VALID_PATH,
        RESULT_INVALID_PATH,
        SUBMISSION_VALID_PATH,
    ):
        assert path.is_file(), f"missing schema/example file: {path}"


def test_result_schema_is_valid_draft_2020_12() -> None:
    """result.schema.json must itself be a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(_result_schema())


def test_submission_schema_is_valid_draft_2020_12() -> None:
    """submission.schema.json must itself be a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(_submission_schema())


def test_result_valid_example_validates() -> None:
    """The valid result fixture must satisfy the result schema."""
    validator = Draft202012Validator(_result_schema())
    instance = _load_json(RESULT_VALID_PATH)
    # `validate` raises on the first error; assert no errors for a clear message.
    errors = sorted(validator.iter_errors(instance), key=lambda err: err.json_path)
    assert errors == [], "result.valid.json should validate but reported: " + "; ".join(
        f"{err.json_path}: {err.message}" for err in errors
    )


def test_result_invalid_example_fails() -> None:
    """The invalid result fixture must be rejected by the result schema."""
    validator = Draft202012Validator(_result_schema())
    instance = _load_json(RESULT_INVALID_PATH)
    with pytest.raises(ValidationError):
        validator.validate(instance)


def test_submission_valid_example_validates() -> None:
    """The valid submission fixture must satisfy the submission schema."""
    validator = Draft202012Validator(_submission_schema())
    instance = _load_json(SUBMISSION_VALID_PATH)
    errors = sorted(validator.iter_errors(instance), key=lambda err: err.json_path)
    assert errors == [], "submission.valid.json should validate but reported: " + "; ".join(
        f"{err.json_path}: {err.message}" for err in errors
    )


# --- schema 1.2.0: additive fields + $id org fix (PR-1) ------------------------


def _assert_valid(validator: Draft202012Validator, instance: dict[str, Any]) -> None:
    """Fail with all validation messages if ``instance`` does not validate."""
    errors = sorted(validator.iter_errors(instance), key=lambda err: err.json_path)
    assert errors == [], "; ".join(f"{err.json_path}: {err.message}" for err in errors)


def test_schema_ids_point_to_crabedesneiges_org() -> None:
    """Both schema $ids must point to the real 'crabedesneiges' GitHub org."""
    for schema in (_result_schema(), _submission_schema()):
        schema_id = schema["$id"]
        assert "crabedesneiges/rf-benchmark-hub" in schema_id, schema_id
        assert "rf-benchmark-hub/rf-benchmark-hub" not in schema_id, schema_id


def test_result_schema_version_enum_is_additive() -> None:
    """1.2.0 is accepted while the older 1.0.0/1.1.0 rows stay valid (additive)."""
    enum = _result_schema()["properties"]["schema_version"]["enum"]
    assert enum == ["1.0.0", "1.1.0", "1.2.0"]


def test_result_1_0_0_without_new_fields_still_validates() -> None:
    """Non-regression: a 1.0.0 row lacking every 1.2.0 field still validates."""
    validator = Draft202012Validator(_result_schema())
    instance = _load_json(RESULT_VALID_PATH)
    assert instance["schema_version"] == "1.0.0"
    for absent in ("pretraining", "transfer", "efficiency"):
        assert absent not in instance
    assert "uncertainty" not in instance["metrics"]
    _assert_valid(validator, instance)


def test_result_1_2_0_all_new_fields_round_trip() -> None:
    """A 1.2.0 row populating all four additive blocks validates."""
    validator = Draft202012Validator(_result_schema())
    instance = _load_json(RESULT_VALID_PATH)
    instance["schema_version"] = "1.2.0"
    instance["metrics"]["uncertainty"] = {
        "accuracy_overall": {
            "ci_low": 0.6021,
            "ci_high": 0.6225,
            "method": "bootstrap_percentile",
            "confidence": 0.95,
            "n_resamples": 2000,
        },
        "macro_f1": {
            "ci_low": 0.58,
            "ci_high": 0.62,
            "method": "multi_seed_std",
            "n_seeds": 5,
            "note": "std across 5 seeds",
        },
    }
    instance["pretraining"] = {
        "pretrain_datasets": ["sig53", "radioml_2018_01a"],
        "overlap_with_eval": "none",
        "disclosure_note": "backbone pretrained on disjoint synthetic corpus",
    }
    instance["transfer"] = {
        "source_dataset": "radioml_2018_01a",
        "source_domain": "synthetic",
    }
    instance["efficiency"] = {
        "inference_latency_ms": 1.2,
        "throughput_samples_per_sec": 8300.0,
        "n_flops": 1.2e9,
        "memory_mb": 512.0,
        "training_gpu_hours": 4.5,
    }
    _assert_valid(validator, instance)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda i: i["metrics"]["uncertainty"]["accuracy_overall"].pop("ci_high"),
            id="uncertainty-missing-required-ci_high",
        ),
        pytest.param(
            lambda i: i["metrics"]["uncertainty"]["accuracy_overall"].__setitem__(
                "method", "jackknife"
            ),
            id="uncertainty-unknown-method",
        ),
        pytest.param(
            lambda i: i.__setitem__("pretraining", {}),
            id="pretraining-empty-object",
        ),
        pytest.param(
            lambda i: i.__setitem__("transfer", {"source_domain": "synthetic"}),
            id="transfer-missing-source_dataset",
        ),
        pytest.param(
            lambda i: i.__setitem__("efficiency", {}),
            id="efficiency-empty-object",
        ),
        pytest.param(
            lambda i: i.__setitem__("efficiency", {"inference_latency_ms": -1.0}),
            id="efficiency-negative-latency",
        ),
    ],
)
def test_result_1_2_0_rejects_malformed_new_fields(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    """Each 1.2.0 block enforces its own constraints (required/enum/min bounds)."""
    validator = Draft202012Validator(_result_schema())
    instance = _load_json(RESULT_VALID_PATH)
    instance["schema_version"] = "1.2.0"
    instance["metrics"]["uncertainty"] = {
        "accuracy_overall": {
            "ci_low": 0.60,
            "ci_high": 0.62,
            "method": "bootstrap_percentile",
        }
    }
    instance["pretraining"] = {"overlap_with_eval": "none"}
    instance["transfer"] = {"source_dataset": "radioml_2018_01a"}
    instance["efficiency"] = {"n_flops": 1.0e9}
    mutate(instance)
    with pytest.raises(ValidationError):
        validator.validate(copy.deepcopy(instance))
