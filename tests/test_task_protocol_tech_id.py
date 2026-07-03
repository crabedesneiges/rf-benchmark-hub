"""Acceptance tests for the protocol-tech-ID (WiFi 802.11 standard recognition) task adapter.

Everything here runs on PURE-PYTHON synthetic predictions/targets -- no numpy, no torch, no
network -- so the suite passes with only ``pytest`` + ``jsonschema`` installed:

* the two protocol-tech-ID metrics (:class:`AccuracyOverall`, :class:`MacroF1`) computed on
  fixtures with hand-checked expected values (overall accuracy, macro-F1);
* the task registers under ``"protocol_tech_id"`` and exposes the protocol datasets/metrics/split;
* an end-to-end :func:`rfbench.core.evaluate.evaluate` over a synthetic in-memory
  :class:`ProtocolDataset` yields a ``result.json`` that validates against
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
from rfbench.tasks.protocol_tech_id import (
    AccuracyOverall,
    MacroF1,
    ProtocolDataset,
    ProtocolTechIdTask,
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
        [9.0, 0.0, 0.0, 0.0],  # -> 0
        [0.0, 9.0, 0.0, 0.0],  # -> 1
        [0.0, 0.0, 9.0, 0.0],  # -> 2
    ]
    metric = AccuracyOverall()
    metric.update(logits, [0, 1, 2])
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_accuracy_overall_has_no_snr_conditions() -> None:
    """Protocol-tech-ID's primary metric declares NO extra eval conditions (no SNR grid)."""
    # The AMC AccuracyOverall exposes eval_conditions(); the protocol one must not, so
    # result.json.eval.conditions stays clean for this task.
    assert not hasattr(AccuracyOverall(), "eval_conditions")


# --------------------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------------------
def test_task_registered_under_protocol_tech_id() -> None:
    """``ProtocolTechIdTask`` resolves by name through the registry."""
    assert "protocol_tech_id" in TASKS
    assert TASKS.get("protocol_tech_id") is ProtocolTechIdTask
    assert isinstance(get_task("protocol_tech_id"), ProtocolTechIdTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metrics/split/tracks match EVALUATION_PROTOCOL.md §protocol_tech_id."""
    task = ProtocolTechIdTask()
    assert task.name == "protocol_tech_id"
    assert task.version == "v1"
    dataset_names = [ds.name for ds in task.datasets()]
    assert dataset_names == ["tprime_wifi4"]
    assert task.default_split() == "test"
    assert task.tracks() == ["closed_set"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "accuracy_overall"  # primary is first
    assert set(metric_keys) == {"accuracy_overall", "macro_f1"}


def test_canonical_split_id_version_matches_task_version() -> None:
    """The dataset's canonical_split_id -v<N> suffix must equal the task version."""
    task = ProtocolTechIdTask()
    dataset = task.datasets()[0]
    assert dataset.canonical_split_id.endswith(f"-{task.version}")


def test_build_targets_extracts_labels() -> None:
    """``build_targets`` returns the per-sample class labels."""
    batch: Batch = {"iq": [[0.0], [0.0]], "label": [3, 1]}
    assert ProtocolTechIdTask().build_targets(batch) == [3, 1]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing a :class:`ProtocolDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown protocol dataset"):
        ProtocolDataset("not_a_dataset")


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory ProtocolDataset
# --------------------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "ab" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in prediction (no torch).

    Each synthetic sample carries a ``pred`` field, so ``forward`` echoes predictions per-row
    and stays correct under any batch size (order preserved by the collate).
    """

    name = "protocol-dummy"
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


def _task_with_samples(preds: list[int]) -> ProtocolTechIdTask:
    dataset = ProtocolDataset("tprime_wifi4", samples=_synthetic_samples(preds), checksum=_CHECKSUM)
    return ProtocolTechIdTask(datasets=[dataset])


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return schema


def test_end_to_end_evaluate_validates_against_schema() -> None:
    """``evaluate`` over a synthetic protocol dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 samples
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "protocol_tech_id"
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

    out_path = tmp_path / "protocol_tech_id" / "result.json"
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


def test_protocol_arrays_align_with_prepare_labels(tmp_path: Path) -> None:
    """The on-disk IQ flatten order MUST equal prepare's label order (else indices corrupt).

    numpy-guarded: skips in the dep-free venv, runs on the cluster [tasks]/[data] venv. Uses a
    synthetic per-class ``.npy`` tree matching the loader's class-subdir + sorted-file walk.
    """
    np = pytest.importorskip("numpy")

    from rfbench.data.prepare.protocol import (
        PROTOCOL_CLASSES,
        _load_tprime_wifi4_labels,
    )
    from rfbench.tasks.protocol_tech_id.dataset import _load_protocol_arrays

    ds_dir = tmp_path / "tprime_wifi4"
    # Two files for the first class, one each for the next two, to exercise the flatten order.
    # Uses the primary on-disk folder spelling for each class (802_11<x>).
    layout = {"802_11b": 2, "802_11g": 1, "802_11n": 1}
    for class_dir_name, n in layout.items():
        class_dir = ds_dir / class_dir_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            np.save(class_dir / f"sample_{i}.npy", np.zeros((2, 8), dtype=np.float32))

    # Point the cache at tmp_path so the loaders find the synthetic tree.
    import os

    os.environ["RFBENCH_CACHE"] = str(tmp_path)
    try:
        labels = _load_tprime_wifi4_labels(tmp_path)
        iq, class_names = _load_protocol_arrays("tprime_wifi4")
    finally:
        os.environ.pop("RFBENCH_CACHE", None)

    assert len(iq) == len(labels) == 4
    assert class_names == labels  # identical order == index alignment
    # Class order follows PROTOCOL_CLASSES (802.11b, 802.11g, 802.11n, 802.11ax).
    assert labels[0] == "802.11b"
    assert set(labels) <= set(PROTOCOL_CLASSES)
    assert iq[0].shape == (2, 8)
