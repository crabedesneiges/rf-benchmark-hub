"""WP-42 acceptance tests for the wired ``rfbench`` CLI.

Pure stdlib: no numpy/h5py/torch/torchsig/requests, no network. The dataset download /
generation functions are DEFINED but never called here; only the split-GENERATION path is
exercised, fed synthetic labels/records/samples through ``--labels-file`` so it runs
without any heavy dependency. ``$RFBENCH_CACHE`` is pinned to ``tmp_path`` and the prepare
output tree is redirected under ``tmp_path`` via ``--out`` so nothing touches the repo.

Covers, per WP-42 acceptance:
  * ``rfbench --help`` still lists every subcommand (data/eval/submit/leaderboard/verify);
  * ``data list`` reflects prepared status; ``data prepare`` on synthetic labels reaches the
    real ``prepare_*`` and writes split indices for AMC / SEI / detection;
  * ``data verify`` recomputes checksums (OK) and flags a corrupted index (exit 1);
  * ``eval`` emits a schema-valid ``result.json`` and enforces the ``--k-shot`` <-> few_shot
    coupling (exit 2 either way it is violated), with the regime routed through
    ``rfbench.regimes``;
  * ``leaderboard build`` renders the sample results into ``tmp_path``;
  * ``submit --check`` passes on a valid ``result.json`` and fails on an invalid one and on
    the all-zero placeholder checksum;
  * importing the CLI pulls in no heavy dependency (checked in a clean subprocess).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rfbench.cli import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    main,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_RESULTS = _REPO_ROOT / "leaderboard" / "results"
_VALID_RESULT = _SAMPLE_RESULTS / "amc" / "mcldnn-radioml2016-full_finetune.json"


# --- synthetic fixtures (pure stdlib; no numpy) -------------------------------------


def _amc_labels_file(path: Path) -> Path:
    """Write a synthetic RadioML ``--labels-file`` (mod x snr tuples) and return its path."""
    labels = [[mod, snr] for mod in ("QPSK", "BPSK") for snr in (0, 10) for _ in range(6)]
    dest = path / "amc_labels.json"
    dest.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return dest


def _sei_records_file(path: Path) -> Path:
    """Write a synthetic WiSig ``--labels-file`` ((tx, rx, day) records) and return its path."""
    records = [
        [tx, rx, day]
        for tx in ("tx0", "tx1", "tx2", "tx3")
        for rx in ("rxA", "rxB")
        for day in ("d1", "d2")
        for _ in range(3)
    ]
    dest = path / "sei_records.json"
    dest.write_text(
        json.dumps({"records": records, "conditions": ["closed_set"]}), encoding="utf-8"
    )
    return dest


def _detection_samples_file(path: Path) -> Path:
    """Write a synthetic WBSig53 ``--labels-file`` (per-sample T-F boxes) and return its path."""
    samples = [
        {"boxes": [{"class": "wifi", "t_start": 0.1, "t_stop": 0.4, "f_low": 0.2, "f_high": 0.5}]}
        for _ in range(20)
    ]
    dest = path / "det_samples.json"
    dest.write_text(json.dumps({"samples": samples, "track": "detection"}), encoding="utf-8")
    return dest


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``$RFBENCH_CACHE`` to a tmp dir so no test ever resolves a repo/absolute cache path."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path / "cache"))


# --- --help surface ------------------------------------------------------------------


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """``rfbench --help`` still advertises all five top-level subcommands."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("data", "eval", "submit", "leaderboard", "verify"):
        assert command in out


def test_help_lists_data_verbs(capsys: pytest.CaptureFixture[str]) -> None:
    """The ``data`` group still exposes download/prepare/list/verify."""
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for verb in ("download", "prepare", "list", "verify"):
        assert verb in out


def test_parser_builds_without_heavy_imports() -> None:
    """Constructing the parser (help surface) needs no runtime dependency."""
    parser = build_parser()
    assert parser.prog == "rfbench"


# --- data list -----------------------------------------------------------------------


def test_data_list_reports_not_prepared(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An empty splits tree lists every known dataset as not prepared."""
    rc = main(["data", "list", "--out", str(tmp_path / "lb")])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "radioml_2016_10a: not prepared" in out
    assert "wisig" in out and "wbsig53" in out


def test_data_list_reflects_prepared(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """After ``prepare`` the dataset is reported as prepared with its canonical id."""
    out_tree = tmp_path / "lb"
    labels = _amc_labels_file(tmp_path)
    assert (
        main(
            [
                "data",
                "prepare",
                "radioml_2016_10a",
                "--labels-file",
                str(labels),
                "--out",
                str(out_tree),
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()
    assert main(["data", "list", "--out", str(out_tree)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "radioml_2016_10a: prepared (amc-radioml2016-strat-snr-8010-seed42-v1)" in out


# --- data prepare (reaches the real prepare_* and writes indices) --------------------


def test_data_prepare_amc_writes_index(tmp_path: Path) -> None:
    """``data prepare`` on synthetic AMC labels writes the idx.json + manifest sidecar."""
    out_tree = tmp_path / "lb"
    labels = _amc_labels_file(tmp_path)
    rc = main(
        [
            "data",
            "prepare",
            "radioml_2016_10a",
            "--labels-file",
            str(labels),
            "--out",
            str(out_tree),
        ]
    )
    assert rc == EXIT_OK
    split_dir = out_tree / "splits" / "radioml_2016_10a"
    idx = split_dir / "amc-radioml2016-strat-snr-8010-seed42-v1.idx.json"
    manifest = split_dir / "amc-radioml2016-strat-snr-8010-seed42-v1.manifest.json"
    assert idx.is_file() and manifest.is_file()
    doc = json.loads(idx.read_text(encoding="utf-8"))
    assert set(doc["indices"]) == {"train", "val", "test"}
    assert doc["checksum"].startswith("sha256:")


def test_data_prepare_by_task_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A task target narrowed by ``--dataset`` prepares just that dataset."""
    out_tree = tmp_path / "lb"
    labels = _amc_labels_file(tmp_path)
    rc = main(
        [
            "data",
            "prepare",
            "amc",
            "--dataset",
            "radioml_2016_10a",
            "--labels-file",
            str(labels),
            "--out",
            str(out_tree),
        ]
    )
    assert rc == EXIT_OK
    assert (out_tree / "splits" / "radioml_2016_10a").is_dir()
    # Only the requested dataset was written.
    assert not (out_tree / "splits" / "sig53").exists()


def test_data_prepare_sei_writes_index(tmp_path: Path) -> None:
    """SEI closed_set prepare on synthetic (tx, rx, day) records writes its split index."""
    out_tree = tmp_path / "lb"
    records = _sei_records_file(tmp_path)
    rc = main(["data", "prepare", "wisig", "--labels-file", str(records), "--out", str(out_tree)])
    assert rc == EXIT_OK
    idx = out_tree / "splits" / "wisig" / "sei-wisig-closedset-strat-tx-8010-seed42-v1.idx.json"
    assert idx.is_file()


def test_data_prepare_detection_writes_annotations(tmp_path: Path) -> None:
    """Detection prepare writes the idx + manifest + annotations sidecar trio."""
    out_tree = tmp_path / "lb"
    samples = _detection_samples_file(tmp_path)
    rc = main(["data", "prepare", "wbsig53", "--labels-file", str(samples), "--out", str(out_tree)])
    assert rc == EXIT_OK
    split_dir = out_tree / "splits" / "wbsig53"
    stem = "detect-wbsig53-detection-8010-seed42-v1"
    assert (split_dir / f"{stem}.idx.json").is_file()
    assert (split_dir / f"{stem}.annotations.json").is_file()


def test_data_prepare_idempotent_skip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A second prepare without ``--force`` is skipped (idempotent)."""
    out_tree = tmp_path / "lb"
    labels = _amc_labels_file(tmp_path)
    args = [
        "data",
        "prepare",
        "radioml_2016_10a",
        "--labels-file",
        str(labels),
        "--out",
        str(out_tree),
    ]
    assert main(args) == EXIT_OK
    capsys.readouterr()
    assert main(args) == EXIT_OK
    assert "already present" in capsys.readouterr().out


def test_data_prepare_unknown_target_is_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown prepare target is a usage error (exit 2), not a crash."""
    rc = main(["data", "prepare", "nonesuch", "--out", str(tmp_path / "lb")])
    assert rc == EXIT_USAGE
    assert "unknown prepare target" in capsys.readouterr().err


# --- data verify ---------------------------------------------------------------------


def test_data_verify_ok_then_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``data verify`` reports OK for a fresh index and exit 1 after corruption."""
    out_tree = tmp_path / "lb"
    labels = _amc_labels_file(tmp_path)
    main(
        [
            "data",
            "prepare",
            "radioml_2016_10a",
            "--labels-file",
            str(labels),
            "--out",
            str(out_tree),
        ]
    )
    capsys.readouterr()

    assert main(["data", "verify", "--out", str(out_tree)]) == EXIT_OK
    assert "OK" in capsys.readouterr().out

    idx = next((out_tree / "splits" / "radioml_2016_10a").glob("*.idx.json"))
    doc = json.loads(idx.read_text(encoding="utf-8"))
    doc["checksum"] = "sha256:" + "0" * 64
    idx.write_text(json.dumps(doc), encoding="utf-8")

    assert main(["data", "verify", "--out", str(out_tree)]) == EXIT_FAILURE
    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert "failed checksum verification" in captured.err


def test_data_verify_no_index_is_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifying an empty tree is a no-op success (nothing to diff)."""
    assert main(["data", "verify", "--out", str(tmp_path / "lb")]) == EXIT_OK
    assert "no split index" in capsys.readouterr().out


# --- eval ----------------------------------------------------------------------------


def test_eval_emits_schema_valid_result(tmp_path: Path) -> None:
    """``eval`` writes a schema-valid result.json with the declared regime verbatim."""
    emit = tmp_path / "result.json"
    rc = main(["eval", "amc", "--model", "mcldnn", "--regime", "linear_probe", "--emit", str(emit)])
    assert rc == EXIT_OK
    doc = json.loads(emit.read_text(encoding="utf-8"))
    assert doc["regime"] == {"name": "linear_probe"}
    assert doc["metrics"]["primary"] == "accuracy_overall"
    assert doc["verification"]["status"] == "self_reported"


def test_eval_few_shot_writes_k_shot(tmp_path: Path) -> None:
    """few_shot eval carries ``regime.k_shot`` (routed through the regimes adapter)."""
    emit = tmp_path / "fs.json"
    rc = main(
        [
            "eval",
            "sei",
            "--model",
            "iqfm",
            "--regime",
            "few_shot",
            "--k-shot",
            "5",
            "--emit",
            str(emit),
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(emit.read_text(encoding="utf-8"))
    assert doc["regime"] == {"name": "few_shot", "k_shot": 5}


def test_eval_few_shot_requires_k(capsys: pytest.CaptureFixture[str]) -> None:
    """``--regime few_shot`` without ``--k-shot`` is a usage error (exit 2)."""
    rc = main(["eval", "amc", "--model", "m", "--regime", "few_shot"])
    assert rc == EXIT_USAGE
    assert "--k-shot is REQUIRED" in capsys.readouterr().err


def test_eval_k_forbidden_outside_few_shot(capsys: pytest.CaptureFixture[str]) -> None:
    """``--k-shot`` with a non-few_shot regime is a usage error (exit 2)."""
    rc = main(["eval", "amc", "--model", "m", "--regime", "linear_probe", "--k-shot", "3"])
    assert rc == EXIT_USAGE
    assert "--k-shot is FORBIDDEN" in capsys.readouterr().err


def test_eval_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--dry-run`` validates but does not write the emit path."""
    emit = tmp_path / "nope.json"
    rc = main(
        [
            "eval",
            "amc",
            "--model",
            "m",
            "--regime",
            "from_scratch",
            "--emit",
            str(emit),
            "--dry-run",
        ]
    )
    assert rc == EXIT_OK
    assert not emit.exists()
    assert "--dry-run" in capsys.readouterr().out


# --- leaderboard build ---------------------------------------------------------------


def test_leaderboard_build_renders_sample_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``leaderboard build`` renders the committed sample results into tmp_path."""
    out_site = tmp_path / "site"
    rc = main(["leaderboard", "build", "--results", str(_SAMPLE_RESULTS), "--out", str(out_site)])
    assert rc == EXIT_OK
    index = out_site / "index.html"
    assert index.is_file()
    assert (out_site / "amc.html").is_file()
    assert (out_site / "sei.html").is_file()
    assert "wrote static site" in capsys.readouterr().out


def test_leaderboard_validate_passes_on_samples(capsys: pytest.CaptureFixture[str]) -> None:
    """``leaderboard validate`` accepts the committed schema-valid sample rows."""
    rc = main(["leaderboard", "validate", "--results", str(_SAMPLE_RESULTS)])
    assert rc == EXIT_OK
    assert "rows valid" in capsys.readouterr().out


# --- submit --check ------------------------------------------------------------------


def test_submit_check_passes_on_valid_result(capsys: pytest.CaptureFixture[str]) -> None:
    """A committed valid+verified result passes the pre-flight (exit 0)."""
    rc = main(["submit", "--check", str(_VALID_RESULT)])
    assert rc == EXIT_OK
    assert "PR-ready" in capsys.readouterr().out


def test_submit_check_fails_on_invalid_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A schema-incomplete result fails the pre-flight (exit 1) and reports why."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    rc = main(["submit", "--check", str(bad)])
    assert rc == EXIT_FAILURE
    assert "NOT PR-ready" in capsys.readouterr().err


def test_submit_check_rejects_placeholder_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An eval-emitted row (all-zero placeholder split checksum) fails the manifest gate."""
    emit = tmp_path / "stub.json"
    main(["eval", "amc", "--model", "stub", "--regime", "linear_probe", "--emit", str(emit)])
    capsys.readouterr()
    rc = main(["submit", "--check", str(emit)])
    assert rc == EXIT_FAILURE
    assert "placeholder" in capsys.readouterr().err


def test_submit_check_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing result path fails cleanly (exit 1), not with a traceback."""
    rc = main(["submit", "--check", str(tmp_path / "absent.json")])
    assert rc == EXIT_FAILURE
    assert "could not read" in capsys.readouterr().err


# --- verify (maintainer pipeline, WP-53) ---------------------------------------------


def _self_reported_result_file(path: Path) -> Path:
    """Write a minimal schema-valid self_reported AMC result and return its path."""
    doc = {
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
    dest = path / "result.json"
    dest.write_text(json.dumps(doc), encoding="utf-8")
    return dest


def _manifest_file(path: Path, **overrides: object) -> Path:
    """Write a complete Tier-2 manifest for the result above; overrides patch top-level keys."""
    doc: dict[str, object] = {
        "schema_version": "1.0.0",
        "result_path": "leaderboard/results/amc/mcldnn.json",
        "task": {"name": "amc", "version": "v1"},
        "regime": {"name": "full_finetune"},
        "code_commit": "git@1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
        "command": "rfbench eval amc --model mcldnn --regime full_finetune --seed 42",
        "artifacts": {"weights_url": "https://zenodo.org/records/0/files/mcldnn.pt"},
        "hardware": "1x NVIDIA A100 80GB",
        "expected_metrics": {"accuracy_overall": 0.6123, "macro_f1": 0.5987},
        "tolerance": {"absolute": 0.01},
    }
    doc.update(overrides)
    dest = path / "manifest.json"
    dest.write_text(json.dumps(doc), encoding="utf-8")
    return dest


def _rerun_result_file(path: Path, accuracy: float) -> Path:
    """Write a re-run result.json carrying a single recomputed accuracy value."""
    doc = {
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
        "metrics": {"primary": "accuracy_overall", "values": {"accuracy_overall": accuracy}},
        "verification": {"status": "self_reported"},
    }
    dest = path / "rerun.json"
    dest.write_text(json.dumps(doc), encoding="utf-8")
    return dest


def test_verify_flips_to_verified_within_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A re-run within tolerance flips the row to verified and writes the stamped result."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path)
    rerun = _rerun_result_file(tmp_path, accuracy=0.6100)
    out = tmp_path / "verified.json"
    rc = main(
        [
            "verify",
            str(result),
            "--manifest",
            str(manifest),
            "--rerun",
            str(rerun),
            "--by",
            "rf-bench-maintainers",
            "--hardware",
            "4x NVIDIA GB200",
            "--out",
            str(out),
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["verification"]["status"] == "verified"
    assert doc["verification"]["verified_by"] == "rf-bench-maintainers"
    assert doc["verification"]["verified_hardware"] == "4x NVIDIA GB200"
    assert doc["verification"]["method"] == "eval_only"
    # The submitted result is left untouched (verified doc went to --out).
    original = json.loads(result.read_text(encoding="utf-8"))
    assert original["verification"]["status"] == "self_reported"
    assert "verified" in capsys.readouterr().out


def test_verify_out_of_tolerance_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A re-run outside tolerance fails (exit 1), writes nothing, keeps the row self_reported."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path)
    rerun = _rerun_result_file(tmp_path, accuracy=0.40)
    out = tmp_path / "verified.json"
    rc = main(
        [
            "verify",
            str(result),
            "--manifest",
            str(manifest),
            "--rerun",
            str(rerun),
            "--by",
            "maint",
            "--hardware",
            "hw",
            "--out",
            str(out),
        ]
    )
    assert rc == EXIT_FAILURE
    assert not out.exists()
    assert "out of tolerance" in capsys.readouterr().err


def test_verify_incomplete_manifest_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An incomplete manifest is rejected (exit 1) and nothing is written."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path, tolerance={})  # invalid: tolerance needs a bound
    out = tmp_path / "verified.json"
    rc = main(
        [
            "verify",
            str(result),
            "--manifest",
            str(manifest),
            "--by",
            "maint",
            "--hardware",
            "hw",
            "--out",
            str(out),
        ]
    )
    assert rc == EXIT_FAILURE
    assert not out.exists()
    assert "manifest" in capsys.readouterr().err


def test_verify_requires_by(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Omitting --by is a usage error (an unsigned flip must never happen)."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path)
    rc = main(["verify", str(result), "--manifest", str(manifest), "--hardware", "hw"])
    assert rc == EXIT_USAGE
    assert "--by" in capsys.readouterr().err


def test_verify_smoke_check_without_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --rerun the result is compared to itself (smoke check) and verifies in place."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path)
    rc = main(
        [
            "verify",
            str(result),
            "--manifest",
            str(manifest),
            "--by",
            "maint",
            "--hardware",
            "hw",
        ]
    )
    assert rc == EXIT_OK
    # Overwrote the input in place (no --out).
    doc = json.loads(result.read_text(encoding="utf-8"))
    assert doc["verification"]["status"] == "verified"
    out = capsys.readouterr().out
    assert "smoke check only" in out


# --- submit --check manifest completeness (WP-53 strengthening) -----------------------


def test_submit_check_with_complete_manifest_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid result + complete manifest passes submit --check."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path)
    rc = main(["submit", "--check", str(result), "--manifest", str(manifest)])
    assert rc == EXIT_OK
    assert "PR-ready" in capsys.readouterr().out


def test_submit_check_incomplete_manifest_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An incomplete manifest fails submit --check with a clear manifest error."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(tmp_path, code_commit="not a sha")  # violates pattern
    rc = main(["submit", "--check", str(result), "--manifest", str(manifest)])
    assert rc == EXIT_FAILURE
    err = capsys.readouterr().err
    assert "NOT PR-ready" in err
    assert "manifest" in err


def test_submit_check_manifest_task_mismatch_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete manifest describing a different task is rejected as inconsistent."""
    result = _self_reported_result_file(tmp_path)
    manifest = _manifest_file(
        tmp_path,
        task={"name": "sei", "version": "v1"},
        result_path="leaderboard/results/sei/mcldnn.json",
        expected_metrics={"accuracy_overall": 0.6123},
    )
    rc = main(["submit", "--check", str(result), "--manifest", str(manifest)])
    assert rc == EXIT_FAILURE
    assert "does not match result.task" in capsys.readouterr().err


# --- import purity (clean subprocess, no heavy deps) ---------------------------------


def test_importing_cli_pulls_no_heavy_deps() -> None:
    """Importing ``rfbench.cli`` in a fresh interpreter loads no numpy/torch/jsonschema/etc."""
    code = (
        "import sys, rfbench.cli;"
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
