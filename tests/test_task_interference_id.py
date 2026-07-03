"""Acceptance tests for the interference-ID (GNSS jamming classification) task adapter.

Everything here runs on PURE-PYTHON synthetic predictions/targets -- no numpy, no torch, no
network -- so the suite passes with only ``pytest`` + ``jsonschema`` installed:

* the two interference-ID metrics (:class:`AccuracyOverall`, :class:`MacroF1`) computed on
  fixtures with hand-checked expected values (overall accuracy, macro-F1);
* the task registers under ``"interference_id"`` and exposes the protocol datasets/metrics/split;
* an end-to-end :func:`rfbench.core.evaluate.evaluate` over a synthetic in-memory
  :class:`InterferenceDataset` yields a ``result.json`` that validates against
  ``schemas/result.schema.json`` with ``metrics.primary == "accuracy_overall"`` present in
  ``metrics.values``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import TASKS, get_task
from rfbench.core.types import Batch, Tensor
from rfbench.tasks.interference_id import (
    AccuracyOverall,
    InterferenceDataset,
    InterferenceIdTask,
    MacroF1,
)

# --------------------------------------------------------------------------------------------------
# Hand-checked fixtures (3 classes, 6 samples)
# --------------------------------------------------------------------------------------------------
# targets:  [0, 1, 2, 0, 1, 2]
# preds:    [0, 1, 2, 0, 2, 2]  -> a single error at index 4 (true=1, pred=2)
_TARGETS = [0, 1, 2, 0, 1, 2]
_PREDS = [0, 1, 2, 0, 2, 2]

# accuracy_overall = 5/6
_EXPECTED_ACC = 5 / 6
# macro_f1 = mean(f1(0)=1.0, f1(1)=2/3, f1(2)=0.8)
_EXPECTED_MACRO_F1 = (1.0 + (2 / 3) + 0.8) / 3


# --------------------------------------------------------------------------------------------------
# Metric compute on synthetic data (known expected values)
# --------------------------------------------------------------------------------------------------
def test_accuracy_overall_known_value() -> None:
    """``accuracy_overall`` matches the hand-computed 5/6 over the whole split."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    computed = metric.compute()
    assert metric.primary_key == "accuracy_overall"
    assert computed["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)


def test_accuracy_overall_also_emits_macro_f1() -> None:
    """The primary metric object emits ``macro_f1`` as a second scalar with the known value."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    assert metric.compute()["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_standalone_macro_f1_matches() -> None:
    """The standalone :class:`MacroF1` agrees with the value from :class:`AccuracyOverall`."""
    metric = MacroF1()
    metric.update(_PREDS, _TARGETS)
    assert metric.compute()["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_metrics_are_streaming_across_batches() -> None:
    """Splitting the batch in two and updating twice yields the same result as one update."""
    whole = AccuracyOverall()
    whole.update(_PREDS, _TARGETS)

    streamed = AccuracyOverall()
    streamed.update(_PREDS[:3], _TARGETS[:3])
    streamed.update(_PREDS[3:], _TARGETS[3:])

    assert streamed.compute() == whole.compute()


def test_reset_clears_state() -> None:
    """After :meth:`reset` the metric behaves as freshly constructed."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    metric.reset()
    metric.update([0, 0], [0, 0])
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_perfect_predictions_are_one() -> None:
    """A perfect classifier scores 1.0 on both accuracy and macro-F1."""
    metric = AccuracyOverall()
    metric.update(_TARGETS, _TARGETS)
    computed = metric.compute()
    assert computed["accuracy_overall"] == pytest.approx(1.0)
    assert computed["macro_f1"] == pytest.approx(1.0)


def test_argmax_logits_path() -> None:
    """A batch of per-class score vectors is argmax-decoded (no numpy) before scoring."""
    logits = [
        [9.0, 0.0, 0.0],  # -> 0
        [0.0, 9.0, 0.0],  # -> 1
        [0.0, 0.0, 9.0],  # -> 2
    ]
    metric = AccuracyOverall()
    metric.update(logits, [0, 1, 2])
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_accuracy_overall_has_no_snr_conditions() -> None:
    """Interference-ID's primary metric declares NO extra eval conditions (no SNR grid)."""
    # The AMC AccuracyOverall exposes eval_conditions(); the interference one must not, so
    # result.json.eval.conditions stays clean for this task.
    assert not hasattr(AccuracyOverall(), "eval_conditions")


# --------------------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------------------
def test_task_registered_under_interference_id() -> None:
    """``InterferenceIdTask`` resolves by name through the registry."""
    assert "interference_id" in TASKS
    assert TASKS.get("interference_id") is InterferenceIdTask
    assert isinstance(get_task("interference_id"), InterferenceIdTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metrics/split/tracks match EVALUATION_PROTOCOL.md §interference_id."""
    task = InterferenceIdTask()
    assert task.name == "interference_id"
    assert task.version == "v1"
    dataset_names = [ds.name for ds in task.datasets()]
    assert dataset_names == ["interf_gnss6"]
    assert task.default_split() == "test"
    assert task.tracks() == ["closed_set"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "accuracy_overall"  # primary is first
    assert set(metric_keys) == {"accuracy_overall", "macro_f1"}


def test_canonical_split_id_version_matches_task_version() -> None:
    """The dataset's canonical_split_id -v<N> suffix must equal the task version."""
    task = InterferenceIdTask()
    dataset = task.datasets()[0]
    assert dataset.canonical_split_id.endswith(f"-{task.version}")


def test_build_targets_extracts_labels() -> None:
    """``build_targets`` returns the per-sample class labels."""
    batch: Batch = {"iq": [[0.0], [0.0]], "label": [3, 5]}
    assert InterferenceIdTask().build_targets(batch) == [3, 5]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing an :class:`InterferenceDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown interference dataset"):
        InterferenceDataset("not_a_dataset")


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory InterferenceDataset
# --------------------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "ab" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in prediction (no torch).

    Each synthetic sample carries a ``pred`` field, so ``forward`` echoes predictions per-row
    and stays correct under any batch size (order preserved by the collate).
    """

    name = "interference-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["pred"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_samples(preds: list[int]) -> list[Batch]:
    return [{"iq": [0.1, 0.2], "label": t, "pred": p} for t, p in zip(_TARGETS, preds, strict=True)]


def _task_with_samples(preds: list[int]) -> InterferenceIdTask:
    dataset = InterferenceDataset(
        "interf_gnss6", samples=_synthetic_samples(preds), checksum=_CHECKSUM
    )
    return InterferenceIdTask(datasets=[dataset])


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return schema


def test_end_to_end_evaluate_validates_against_schema() -> None:
    """``evaluate`` over a synthetic interference dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 samples
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "interference_id"
    assert result["metrics"]["primary"] == "accuracy_overall"
    values = result["metrics"]["values"]
    assert "accuracy_overall" in values
    assert values["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)
    assert values["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_end_to_end_perfect_model_scores_one() -> None:
    """A perfect model yields accuracy_overall == 1.0 through the full harness."""
    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_TARGETS),
        "test",
        RegimeSpec(Regime.FULL_FINETUNE),
        batch_size=2,
    )
    assert result["metrics"]["values"]["accuracy_overall"] == pytest.approx(1.0)
    assert math.isclose(result["metrics"]["values"]["macro_f1"], 1.0)


def test_end_to_end_writes_valid_json(tmp_path: Path) -> None:
    """With ``out_path`` set the on-disk result re-validates against the schema."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "interference_id" / "result.json"
    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.LINEAR_PROBE),
        batch_size=3,
        out_path=out_path,
    )
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


# --- on-disk loader: index alignment (regression guard) --------------------------------


def test_interference_arrays_align_with_prepare_labels(tmp_path: Path) -> None:
    """The on-disk IQ flatten order MUST equal prepare's label order (else indices corrupt).

    numpy-guarded: skips in the dep-free venv, runs on the cluster [tasks]/[data] venv. Uses a
    synthetic per-class ``.npy`` tree matching the loader's class-subdir + sorted-file walk.
    """
    np = pytest.importorskip("numpy")

    from rfbench.data.prepare.interference import (
        INTERFERENCE_CLASSES,
        _load_interference_gnss6_labels,
    )
    from rfbench.tasks.interference_id.dataset import _load_interference_arrays

    ds_dir = tmp_path / "interf_gnss6"
    # Two files for the first class, one for the second, to exercise the flatten order.
    layout = {"DME": 2, "narrowband": 1, "no_jamming": 1}
    for class_name, n in layout.items():
        class_dir = ds_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            np.save(class_dir / f"sample_{i}.npy", np.zeros((2, 8), dtype=np.float32))

    # Point the cache at tmp_path so the loaders find the synthetic tree.
    import os

    os.environ["RFBENCH_CACHE"] = str(tmp_path)
    try:
        labels = _load_interference_gnss6_labels(tmp_path)
        iq, class_names = _load_interference_arrays("interf_gnss6")
    finally:
        os.environ.pop("RFBENCH_CACHE", None)

    assert len(iq) == len(labels) == 4
    assert class_names == labels  # identical order == index alignment
    # Class order follows INTERFERENCE_CLASSES (DME, narrowband, no_jamming, ...).
    assert labels[0] == "DME"
    assert set(labels) <= set(INTERFERENCE_CLASSES)
    assert iq[0].shape == (2, 8)
