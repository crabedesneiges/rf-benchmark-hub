"""Tests for ``scripts/aggregate_multiseed.py`` -- multi-seed aggregation.

The script is loaded by path (``importlib``) so no package install is needed.
All tests use ``tmp_path`` fixtures; no torch/numpy dependency.

Coverage:
- Correct mean and stdev on a known case.
- Schema validity of the output document.
- Refusal on missing staging file.
- Refusal on divergent split.checksum.
- Refusal on inconsistent metric key sets.
- Idempotence (second pass produces no change).
- Reference seed selection (seed 42 wins; fallback to first if absent).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_multiseed.py"


def _load_aggregate() -> ModuleType:
    """Import ``scripts/aggregate_multiseed.py`` by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location("aggregate_multiseed", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agg = _load_aggregate()

# ---------------------------------------------------------------------------
# Helpers: minimal valid staging result.json
# ---------------------------------------------------------------------------

_CHECKSUM_A = "sha256:" + "ab" * 32
_CHECKSUM_B = "sha256:" + "cd" * 32


def _make_staging_doc(
    *,
    seed: int,
    accuracy: float,
    macro_f1: float,
    checksum: str = _CHECKSUM_A,
    task_name: str = "amc",
    model_name: str = "cldnn",
    regime_name: str = "from_scratch",
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a minimal schema-valid per-seed staging result."""
    values: dict[str, float] = {"accuracy_overall": accuracy, "macro_f1": macro_f1}
    if extra_metrics:
        values.update(extra_metrics)
    return {
        "schema_version": "1.2.0",
        "task": {"name": task_name, "version": "v1"},
        "model": {"name": model_name, "family": "baseline", "n_params": 100_000},
        "regime": {"name": regime_name},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "amc-strat-snr-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": checksum,
        },
        "metrics": {
            "primary": "accuracy_overall",
            "values": values,
            "curves": {"accuracy_vs_snr": [{"x": 0.0, "y": accuracy}]},
        },
        "environment": {"seed": seed, "rfbench_version": "0.1.0", "python_version": "3.11.0"},
        "eval": {"conditions": {}, "n_samples": 22_000},
        "verification": {"status": "self_reported"},
    }


def _write_staging(
    staging_dir: Path, task: str, model: str, seed: int, doc: dict[str, Any]
) -> Path:
    path = staging_dir / task / f"{model}-seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    return tmp_path / "staging"


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


# ---------------------------------------------------------------------------
# Mean / stdev correctness
# ---------------------------------------------------------------------------


def test_mean_and_stdev_exact(staging_dir: Path, out_dir: Path) -> None:
    """Verify the aggregated values match hand-computed mean and stdev."""
    accuracies = [0.580, 0.590, 0.600]
    f1s = [0.570, 0.580, 0.590]
    seeds = [42, 43, 44]

    for seed, acc, f1 in zip(seeds, accuracies, f1s, strict=True):
        _write_staging(
            staging_dir,
            "amc",
            "cldnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=acc,
                macro_f1=f1,
            ),
        )

    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)

    expected_acc_mean = mean(accuracies)
    expected_acc_std = stdev(accuracies)  # ddof=1

    acc_val = result["metrics"]["values"]["accuracy_overall"]
    assert acc_val == pytest.approx(expected_acc_mean, abs=1e-9)
    f1_val = result["metrics"]["values"]["macro_f1"]
    assert f1_val == pytest.approx(mean(f1s), abs=1e-9)

    unc = result["metrics"]["uncertainty"]["accuracy_overall"]
    assert unc["ci_low"] == pytest.approx(expected_acc_mean - expected_acc_std, abs=1e-9)
    assert unc["ci_high"] == pytest.approx(expected_acc_mean + expected_acc_std, abs=1e-9)
    assert unc["method"] == "multi_seed_std"
    assert unc["n_seeds"] == 3
    assert "note" in unc
    # Note must mention it is NOT a 95% CI
    assert "95%" in unc["note"] or "confiance" in unc["note"].lower() or "DESCRIPTIF" in unc["note"]


def test_all_metrics_get_uncertainty(staging_dir: Path, out_dir: Path) -> None:
    """Every scalar in metrics.values gets an uncertainty entry."""
    seeds = [42, 43, 44]
    for seed in seeds:
        _write_staging(
            staging_dir,
            "amc",
            "cldnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=0.58 + seed * 0.001,
                macro_f1=0.57 + seed * 0.001,
            ),
        )
    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)

    for metric in result["metrics"]["values"]:
        assert metric in result["metrics"]["uncertainty"], f"uncertainty missing for {metric}"


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------


def test_output_validates_schema(staging_dir: Path, out_dir: Path) -> None:
    """The aggregated document must pass the repo JSON schema."""
    seeds = [42, 43, 44]
    for seed in seeds:
        _write_staging(
            staging_dir,
            "amc",
            "cldnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=0.58 + seed * 0.001,
                macro_f1=0.57 + seed * 0.001,
            ),
        )
    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)
    # Should not raise
    agg._validate_or_raise(result)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_staging_file_raises(staging_dir: Path, out_dir: Path) -> None:
    """A missing staging file causes a FileNotFoundError with a clear message."""
    seeds = [42, 43, 44]
    # Only write 2 of the 3 seeds
    for seed in [42, 43]:
        _write_staging(
            staging_dir,
            "amc",
            "cldnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=0.58,
                macro_f1=0.57,
            ),
        )
    out = out_dir / "amc" / "cldnn.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)
    assert "44" in str(exc_info.value)


def test_divergent_checksum_raises(staging_dir: Path, out_dir: Path) -> None:
    """Different split.checksum values cause a ValueError about incompatibility."""
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        42,
        _make_staging_doc(
            seed=42,
            accuracy=0.58,
            macro_f1=0.57,
            checksum=_CHECKSUM_A,
        ),
    )
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        43,
        _make_staging_doc(
            seed=43,
            accuracy=0.59,
            macro_f1=0.58,
            checksum=_CHECKSUM_B,
        ),
    )
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        44,
        _make_staging_doc(
            seed=44,
            accuracy=0.60,
            macro_f1=0.59,
            checksum=_CHECKSUM_A,
        ),
    )
    out = out_dir / "amc" / "cldnn.json"
    with pytest.raises(ValueError) as exc_info:
        agg.aggregate(staging_dir, "amc", "cldnn", [42, 43, 44], out)
    assert "checksum" in str(exc_info.value).lower() or "incompatib" in str(exc_info.value).lower()


def test_incompatible_model_name_raises(staging_dir: Path, out_dir: Path) -> None:
    """Different model.name values cause a ValueError about incompatibility."""
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        42,
        _make_staging_doc(
            seed=42,
            accuracy=0.58,
            macro_f1=0.57,
            model_name="cldnn",
        ),
    )
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        43,
        _make_staging_doc(
            seed=43,
            accuracy=0.59,
            macro_f1=0.58,
            model_name="mcldnn",  # different!
        ),
    )
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        44,
        _make_staging_doc(
            seed=44,
            accuracy=0.60,
            macro_f1=0.59,
            model_name="cldnn",
        ),
    )
    out = out_dir / "amc" / "cldnn.json"
    with pytest.raises(ValueError) as exc_info:
        agg.aggregate(staging_dir, "amc", "cldnn", [42, 43, 44], out)
    assert "model.name" in str(exc_info.value)


def test_inconsistent_metric_keys_raises(staging_dir: Path, out_dir: Path) -> None:
    """A seed with extra/missing metric keys causes a ValueError."""
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        42,
        _make_staging_doc(
            seed=42,
            accuracy=0.58,
            macro_f1=0.57,
        ),
    )
    # seed 43 has an extra metric
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        43,
        _make_staging_doc(
            seed=43,
            accuracy=0.59,
            macro_f1=0.58,
            extra_metrics={"extra_metric": 0.5},
        ),
    )
    _write_staging(
        staging_dir,
        "amc",
        "cldnn",
        44,
        _make_staging_doc(
            seed=44,
            accuracy=0.60,
            macro_f1=0.59,
        ),
    )
    out = out_dir / "amc" / "cldnn.json"
    with pytest.raises(ValueError) as exc_info:
        agg.aggregate(staging_dir, "amc", "cldnn", [42, 43, 44], out)
    assert "mismatch" in str(exc_info.value).lower() or "metric" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_idempotent(staging_dir: Path, out_dir: Path) -> None:
    """Running aggregate twice leaves the output file byte-identical."""
    seeds = [42, 43, 44]
    for seed in seeds:
        _write_staging(
            staging_dir,
            "amc",
            "cldnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=0.58 + seed * 0.001,
                macro_f1=0.57 + seed * 0.001,
            ),
        )
    out = out_dir / "amc" / "cldnn.json"
    agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)
    first_content = out.read_text(encoding="utf-8")

    agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)
    second_content = out.read_text(encoding="utf-8")

    assert first_content == second_content


# ---------------------------------------------------------------------------
# Reference seed selection
# ---------------------------------------------------------------------------


def test_reference_seed_is_42_when_present(staging_dir: Path, out_dir: Path) -> None:
    """When seed 42 is in the list, curves come from seed 42 (reference)."""
    seeds = [42, 43, 44]
    for seed in seeds:
        doc = _make_staging_doc(
            seed=seed,
            accuracy=0.58 + (seed - 42) * 0.01,
            macro_f1=0.57,
        )
        # Give seed 42 a distinctive curve y value
        doc["metrics"]["curves"] = {"accuracy_vs_snr": [{"x": 0.0, "y": float(seed)}]}
        _write_staging(staging_dir, "amc", "cldnn", seed, doc)

    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)

    # Curves should come from seed 42 (y=42.0)
    curve_y = result["metrics"]["curves"]["accuracy_vs_snr"][0]["y"]
    assert curve_y == pytest.approx(42.0)


def test_reference_seed_fallback_to_first_when_42_absent(staging_dir: Path, out_dir: Path) -> None:
    """When seed 42 is not in the list, the first seed is used as reference."""
    seeds = [43, 44, 45]
    for seed in seeds:
        doc = _make_staging_doc(
            seed=seed,
            accuracy=0.58 + (seed - 43) * 0.01,
            macro_f1=0.57,
        )
        doc["metrics"]["curves"] = {"accuracy_vs_snr": [{"x": 0.0, "y": float(seed)}]}
        _write_staging(staging_dir, "amc", "cldnn", seed, doc)

    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)

    # Curves should come from seed 43 (first in list)
    curve_y = result["metrics"]["curves"]["accuracy_vs_snr"][0]["y"]
    assert curve_y == pytest.approx(43.0)


# ---------------------------------------------------------------------------
# schema_version is always 1.2.0 in output
# ---------------------------------------------------------------------------


def test_schema_version_is_bumped_to_1_2_0(staging_dir: Path, out_dir: Path) -> None:
    """Output always declares schema_version=1.2.0 (uncertainty block requires it)."""
    seeds = [42, 43, 44]
    for seed in seeds:
        doc = _make_staging_doc(seed=seed, accuracy=0.58, macro_f1=0.57)
        doc["schema_version"] = "1.0.0"  # force an older version
        _write_staging(staging_dir, "amc", "cldnn", seed, doc)

    out = out_dir / "amc" / "cldnn.json"
    result = agg.aggregate(staging_dir, "amc", "cldnn", seeds, out)
    assert result["schema_version"] == "1.2.0"


# ---------------------------------------------------------------------------
# CLI: parse_seeds helper
# ---------------------------------------------------------------------------


def test_parse_seeds_valid() -> None:
    assert agg._parse_seeds("42,43,44") == [42, 43, 44]
    assert agg._parse_seeds("42") == [42]
    assert agg._parse_seeds(" 42 , 43 ") == [42, 43]


def test_parse_seeds_invalid() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        agg._parse_seeds("42,abc")


def test_proportion_ci_clamped_to_unit_interval(staging_dir: Path, out_dir: Path) -> None:
    """A near-perfect proportion metric cannot report ci_high above 1 (or ci_low below 0).

    Mirrors the interf_cnn case that motivated the clamp: mean 0.9996 with stdev ~0.0008 gives a
    raw mean + stdev of 1.0003, which is meaningless for an accuracy. The clamp keeps the interval
    inside the metric's domain without ever widening it; non-proportion metrics pass through.
    """
    accuracies = [1.0, 0.9987, 1.0]
    f1s = [1.0, 0.9985, 1.0]
    seeds = [42, 43, 44]

    for seed, acc, f1 in zip(seeds, accuracies, f1s, strict=True):
        _write_staging(
            staging_dir,
            "interference_id",
            "interf_cnn",
            seed,
            _make_staging_doc(
                seed=seed,
                accuracy=acc,
                macro_f1=f1,
                task_name="interference_id",
                model_name="interf_cnn",
            ),
        )

    out = out_dir / "interference_id" / "interf_cnn.json"
    result = agg.aggregate(staging_dir, "interference_id", "interf_cnn", seeds, out)

    raw_high = mean(accuracies) + stdev(accuracies)
    assert raw_high > 1.0  # the fixture genuinely exercises the clamp
    unc = result["metrics"]["uncertainty"]["accuracy_overall"]
    assert unc["ci_high"] == 1.0
    assert 0.0 <= unc["ci_low"] <= 1.0
    # ci_low is untouched by the clamp (it is inside [0, 1] already)
    assert unc["ci_low"] == pytest.approx(mean(accuracies) - stdev(accuracies), abs=1e-9)
