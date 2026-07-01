"""WP-40 acceptance tests for :func:`rfbench.core.evaluate`.

These tests use pure-Python dummy ``Task`` / ``Dataset`` / ``Metric`` / ``Model``
implementations (NO torch/numpy) driving a tiny in-memory split, and assert that
:func:`evaluate` emits a dict that independently validates against
``schemas/result.schema.json`` via ``jsonschema``. They must pass with only
``pytest`` + ``jsonschema`` installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.dataset import Dataset
from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.metric import Metric
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.splits import SplitManifest
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track

# A byte-stable checksum satisfying the schema pattern ``^sha256:[0-9a-f]{64}$``.
_CHECKSUM = "sha256:" + "ab" * 32


# --------------------------------------------------------------------------------------------------
# Pure-Python dummies (no torch/numpy)
# --------------------------------------------------------------------------------------------------
class _InMemorySplit:
    """A tiny map-style dataset: a list of per-sample ``Batch`` dicts."""

    def __init__(self, samples: list[Batch]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._samples)


class DummyDataset(Dataset):
    """A dataset variant backed by an in-memory ``{iq,label,snr_db}`` split."""

    name = "radioml_2016_10a"
    canonical_split_id = "amc-strat-snr-seed42-v1"
    checksum = _CHECKSUM

    def __init__(self, samples: list[Batch]) -> None:
        self._samples = samples

    def download(self, cache: Path | None = None) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def prepare(self, seed: int = 42) -> SplitManifest:  # pragma: no cover - unused
        raise NotImplementedError

    def load(self, split: SplitName, track: Track | None = None) -> _InMemorySplit:
        return _InMemorySplit(self._samples)


class DummyAccuracy(Metric):
    """A streaming top-1 accuracy plus a constant curve, in pure Python."""

    name = "accuracy"
    primary_key = "accuracy_overall"

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        for predicted, expected in zip(pred, target, strict=True):
            self._total += 1
            if predicted == expected:
                self._correct += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        accuracy = self._correct / self._total if self._total else 0.0
        return {
            "accuracy_overall": accuracy,
            "macro_f1": accuracy,
            "accuracy_vs_snr": [{"x": 0.0, "y": accuracy}],
        }

    def eval_conditions(self) -> dict[str, Any]:
        return {"snr_db_min": -20, "snr_db_max": 18, "full_snr_range": True}


class DummyTask(Task):
    """An AMC-shaped task over a single in-memory dataset."""

    name = "amc"
    version = "v1"

    def __init__(self, samples: list[Batch]) -> None:
        self._dataset = DummyDataset(samples)

    def datasets(self) -> list[Dataset]:
        return [self._dataset]

    def metrics(self) -> list[Metric]:
        return [DummyAccuracy()]

    def default_split(self) -> SplitName:
        return "test"

    def tracks(self) -> list[Track]:
        return ["closed_set"]

    def build_targets(self, batch: Batch) -> Tensor:
        return batch["label"]


class DummyModel(Model):
    """A deterministic model: predicts the ground-truth label (perfect classifier)."""

    name = "dummy-perfect"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        # ``x`` is the collated batch; echo the labels back as predictions.
        return list(x["label"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 12345


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------
def _make_samples() -> list[Batch]:
    return [
        {"iq": [0.1, 0.2], "label": 0, "snr_db": -20},
        {"iq": [0.3, 0.4], "label": 1, "snr_db": -10},
        {"iq": [0.5, 0.6], "label": 2, "snr_db": 0},
        {"iq": [0.7, 0.8], "label": 1, "snr_db": 10},
        {"iq": [0.9, 1.0], "label": 0, "snr_db": 18},
    ]


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    return schema


def _run(
    *,
    model: Model | None = None,
    task: Task | None = None,
    split: SplitName = "test",
    regime: RegimeSpec | None = None,
    dataset: str | None = None,
    track: Track | None = None,
    seed: int = 42,
    batch_size: int = 2,
    out_path: Path | None = None,
) -> dict[str, Any]:
    return evaluate(
        model if model is not None else DummyModel(),
        task if task is not None else DummyTask(_make_samples()),
        split,
        regime if regime is not None else RegimeSpec(Regime.LINEAR_PROBE),
        dataset=dataset,
        track=track,
        seed=seed,
        batch_size=batch_size,
        out_path=out_path,
    )


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------
def test_result_validates_against_schema() -> None:
    """The emitted dict independently validates against result.schema.json."""
    from jsonschema import Draft202012Validator

    result = _run()
    Draft202012Validator(_load_schema()).validate(result)


def test_regime_written_verbatim() -> None:
    """``regime.name`` equals the passed regime and is never inferred."""
    result = _run(regime=RegimeSpec(Regime.FULL_FINETUNE))
    assert result["regime"]["name"] == "full_finetune"
    assert "k_shot" not in result["regime"]


def test_few_shot_regime_carries_k_shot() -> None:
    """``few_shot`` writes ``k_shot`` (schema allOf); other regimes never do."""
    result = _run(regime=RegimeSpec(Regime.FEW_SHOT, k_shot=5))
    assert result["regime"] == {"name": "few_shot", "k_shot": 5}


def test_primary_metric_present_in_values() -> None:
    """``metrics.primary`` is the task primary key AND a key of ``metrics.values``."""
    result = _run()
    assert result["metrics"]["primary"] == "accuracy_overall"
    assert "accuracy_overall" in result["metrics"]["values"]


def test_scalars_and_curves_are_partitioned() -> None:
    """Scalar metrics land in ``values``; list payloads land in ``curves``."""
    result = _run()
    assert result["metrics"]["values"] == {"accuracy_overall": 1.0, "macro_f1": 1.0}
    assert "accuracy_vs_snr" in result["metrics"]["curves"]
    # a scalar name never leaks into curves and vice-versa
    assert "accuracy_vs_snr" not in result["metrics"]["values"]


def test_split_identity_and_conditions() -> None:
    """Split identity is copied from the dataset; full-protocol conditions recorded."""
    result = _run()
    assert result["split"]["canonical_split_id"] == "amc-strat-snr-seed42-v1"
    assert result["split"]["checksum"] == _CHECKSUM
    assert result["split"]["seed"] == 42
    assert result["eval"]["conditions"]["full_snr_range"] is True
    assert result["eval"]["n_samples"] == 5


def test_track_recorded_when_given() -> None:
    """``split.track`` is present iff a track is passed."""
    with_track = _run(track="closed_set")
    assert with_track["split"]["track"] == "closed_set"
    without_track = _run()
    assert "track" not in without_track["split"]


def test_verification_starts_self_reported() -> None:
    """``verification.status`` is always initialised to ``self_reported``."""
    result = _run()
    assert result["verification"] == {"status": "self_reported"}


def test_environment_is_deterministic() -> None:
    """The environment fingerprint carries the seed and stays stable across calls."""
    first = _run(seed=42)
    second = _run(seed=42)
    assert first["environment"]["seed"] == 42
    assert first["environment"] == second["environment"]


def test_call_is_deterministic() -> None:
    """Two identical calls produce byte-identical result dicts."""
    assert _run() == _run()


def test_writes_atomically_and_validates(tmp_path: Path) -> None:
    """When ``out_path`` is given the file is written with sorted keys and re-validates."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "nested" / "result.json"
    result = _run(out_path=out_path)
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)
    # sort_keys=True: top-level keys are alphabetically ordered on disk.
    text = out_path.read_text(encoding="utf-8")
    assert list(json.loads(text).keys()) == sorted(json.loads(text).keys())


def test_schema_invalid_path_raises() -> None:
    """A result that violates the schema raises ``jsonschema.ValidationError``.

    ``split='train'`` is a valid ``SplitName`` for fitting but is FORBIDDEN by the
    schema's ``split.name`` enum (``{test, val}``), so the assembled dict fails
    validation before it can leave the harness.
    """
    from jsonschema import ValidationError

    with pytest.raises(ValidationError):
        _run(split="train")


def test_unknown_dataset_raises() -> None:
    """Passing a dataset name the task does not declare raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown dataset"):
        _run(dataset="sig53")
