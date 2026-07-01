"""WP-51/52 -- structural tests for the M5 CI workflows and issue forms.

These are pure-stdlib+PyYAML checks: every workflow / issue-form YAML must parse and
carry the required triggers, steps and keys. They do NOT spin up GitHub Actions; they
guard against the classic footguns (the ``on:`` -> ``True`` YAML coercion, a dropped
Pages permission, a renamed step) so a broken workflow is caught in the same ``pytest``
run as the rest of the harness. Only ``pytest`` + ``PyYAML`` are needed (no torch/numpy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ISSUE_TEMPLATES = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"

VALIDATE = WORKFLOWS / "validate-submission.yml"
BUILD = WORKFLOWS / "build-leaderboard.yml"
SUBMISSION_FORM = ISSUE_TEMPLATES / "submission.yml"
TASK_PROPOSAL_FORM = ISSUE_TEMPLATES / "task_proposal.yml"


def _load(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict, asserting it exists and is a mapping."""
    assert path.is_file(), f"missing YAML file: {path}"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path} did not parse into a mapping"
    return doc


def _on_block(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow ``on`` trigger block as a mapping.

    PyYAML coerces an unquoted ``on:`` key to the boolean ``True``; the workflows quote
    it (``"on":``) so the key stays the string ``"on"``. Accept either so the test does
    not silently pass on the coerced form.
    """
    # ``True`` (bool) is the coerced key. Iterate the raw items (typed loosely so mypy
    # does not narrow keys to ``str``) and match either the string ``"on"`` or ``True``.
    items: list[tuple[object, object]] = list(doc.items())
    block: object = None
    for key, value in items:
        if key == "on" or key is True:
            block = value
            break
    assert isinstance(block, dict), "workflow has no 'on:' trigger mapping"
    return block


def _all_run_steps(doc: dict[str, Any]) -> str:
    """Concatenate every job step's ``run`` and ``uses`` into one searchable blob."""
    chunks: list[str] = []
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                chunks.append(str(step["run"]))
            if "uses" in step:
                chunks.append(str(step["uses"]))
    return "\n".join(chunks)


# --------------------------------------------------------------------------------------------------
# All YAML parses
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [VALIDATE, BUILD, SUBMISSION_FORM, TASK_PROPOSAL_FORM],
    ids=lambda p: p.name,
)
def test_yaml_parses(path: Path) -> None:
    _load(path)


# --------------------------------------------------------------------------------------------------
# validate-submission.yml (WP-51)
# --------------------------------------------------------------------------------------------------
def test_validate_submission_triggers_on_result_pr() -> None:
    doc = _load(VALIDATE)
    on = _on_block(doc)
    assert "pull_request" in on, "validate-submission must trigger on pull_request"
    paths = on["pull_request"]["paths"]
    assert any(
        "leaderboard/results/**" in p for p in paths
    ), "validate-submission must filter on leaderboard/results/**"


def test_validate_submission_runs_check_and_hygiene() -> None:
    doc = _load(VALIDATE)
    blob = _all_run_steps(doc)
    assert 'pip install -e ".[dev]"' in blob, "must install the dev extras"
    assert "rfbench submit --check" in blob, "must run rfbench submit --check on results"
    assert "tools/check_no_raw_data.py" in blob, "must run the no-raw-data guard"


def test_validate_submission_is_read_only() -> None:
    doc = _load(VALIDATE)
    assert doc["permissions"] == {"contents": "read"}


# --------------------------------------------------------------------------------------------------
# build-leaderboard.yml (WP-52)
# --------------------------------------------------------------------------------------------------
def test_build_leaderboard_triggers_on_main_and_dispatch() -> None:
    doc = _load(BUILD)
    on = _on_block(doc)
    assert "workflow_dispatch" in on, "build-leaderboard must be manually dispatchable"
    branches = on["push"]["branches"]
    assert "main" in branches, "build-leaderboard must trigger on push to main"


def test_build_leaderboard_builds_and_deploys() -> None:
    doc = _load(BUILD)
    blob = _all_run_steps(doc)
    assert (
        "rfbench leaderboard build" in blob or "leaderboard/site/generate.py" in blob
    ), "must build the site via the CLI or the generator"
    assert "actions/upload-pages-artifact" in blob, "must upload a Pages artifact"
    assert "actions/deploy-pages" in blob, "must deploy to GitHub Pages"


def test_build_leaderboard_has_pages_permissions_and_environment() -> None:
    doc = _load(BUILD)
    perms = doc["permissions"]
    assert perms.get("pages") == "write", "Pages deploy needs pages: write"
    assert perms.get("id-token") == "write", "Pages deploy needs id-token: write"
    # The deploy job must bind the github-pages environment.
    envs = [job.get("environment") for job in doc["jobs"].values() if job.get("environment")]
    assert any(
        (isinstance(e, dict) and e.get("name") == "github-pages") or e == "github-pages"
        for e in envs
    ), "the deploy job must target the github-pages environment"


# --------------------------------------------------------------------------------------------------
# Issue forms
# --------------------------------------------------------------------------------------------------
def _form_field_ids(doc: dict[str, Any]) -> set[str]:
    """Return the set of ``id``s of the form's input elements."""
    return {el["id"] for el in doc.get("body", []) if "id" in el}


def test_submission_form_has_required_fields() -> None:
    doc = _load(SUBMISSION_FORM)
    assert doc.get("name"), "issue form needs a name"
    ids = _form_field_ids(doc)
    required = {"task", "model", "regime", "metrics", "repro_manifest"}
    missing = required - ids
    assert not missing, f"submission form missing fields: {sorted(missing)}"


def test_task_proposal_form_has_required_fields() -> None:
    doc = _load(TASK_PROPOSAL_FORM)
    assert doc.get("name"), "issue form needs a name"
    ids = _form_field_ids(doc)
    required = {"task_name", "datasets", "metric", "split", "licence"}
    missing = required - ids
    assert not missing, f"task proposal form missing fields: {sorted(missing)}"
