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

import json
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
