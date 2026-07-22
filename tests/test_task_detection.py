"""WP-22 acceptance tests for the wideband-detection task adapter.

Pure stdlib: NO torchmetrics / numpy / torch imported here. The metric ``compute()`` runs
on plain-Python predicted vs ground-truth time-frequency boxes, so the whole file passes
with only ``pytest`` + ``jsonschema`` installed.

Covers, per WP-22 acceptance:
  * IoU on known T-F box pairs (identical -> 1, disjoint -> 0, half-overlap -> a
    hand-computed value);
  * mAP / mAR on a tiny synthetic set of predicted vs ground-truth boxes with a
    hand-computable expected AP;
  * the task is registered under ``"wideband_detection"`` and keeps the detection vs
    recognition tracks distinct;
  * an ``evaluate()`` end-to-end producing a schema-valid ``result.json`` with
    ``metrics.primary == "mAP"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import TASKS, get_task
from rfbench.core.types import Batch, Tensor
from rfbench.tasks.wideband_detection.task import (
    DEFAULT_IOU_THRESHOLD,
    DETECTION_TRACK,
    RECOGNITION_TRACK,
    TRACKS,
    DetectionMetric,
    TFBox,
    WidebandDetectionTask,
    average_precision,
    iou,
)

_ONE_THIRD = 1.0 / 3.0


# ==================================================================================================
# IoU on known box pairs
# ==================================================================================================
def test_iou_identical_boxes_is_one() -> None:
    """Two identical T-F boxes have IoU exactly 1.0."""
    box = TFBox(0.0, 0.5, 0.0, 0.5)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    """Boxes that share no time-frequency area have IoU 0.0."""
    a = TFBox(0.0, 0.4, 0.0, 0.4)
    b = TFBox(0.6, 1.0, 0.6, 1.0)
    assert iou(a, b) == 0.0


def test_iou_time_disjoint_only_is_zero() -> None:
    """Overlapping in frequency but disjoint in time still yields IoU 0.0."""
    a = TFBox(0.0, 0.3, 0.0, 1.0)
    b = TFBox(0.5, 0.8, 0.0, 1.0)
    assert iou(a, b) == 0.0


def test_iou_half_overlap_known_value() -> None:
    """A half-overlap in frequency gives a hand-computed IoU of 1/3.

    A = time[0,1] x freq[0,0.5], B = time[0,1] x freq[0.25,0.75]. Each area = 0.5;
    intersection = 1.0 * 0.25 = 0.25; union = 0.5 + 0.5 - 0.25 = 0.75; IoU = 1/3.
    """
    a = TFBox(0.0, 1.0, 0.0, 0.5)
    b = TFBox(0.0, 1.0, 0.25, 0.75)
    assert iou(a, b) == pytest.approx(_ONE_THIRD)


def test_iou_is_symmetric() -> None:
    """IoU does not depend on argument order."""
    a = TFBox(0.0, 1.0, 0.0, 0.5)
    b = TFBox(0.0, 1.0, 0.25, 0.75)
    assert iou(a, b) == iou(b, a)


def test_iou_accepts_plain_mappings() -> None:
    """IoU also accepts the annotation-sidecar mapping shape, not just ``TFBox``."""
    a = {"t_start": 0.0, "t_stop": 0.5, "f_low": 0.0, "f_high": 0.5, "class": "bpsk"}
    b = {"t_start": 0.0, "t_stop": 0.5, "f_low": 0.0, "f_high": 0.5, "class": "bpsk"}
    assert iou(a, b) == 1.0


# ==================================================================================================
# average_precision on a hand-computable set
# ==================================================================================================
def test_average_precision_perfect_is_one() -> None:
    """Two true positives and no false positives over two GTs give AP 1.0."""
    flags = [(0.9, True), (0.8, True)]
    assert average_precision(flags, n_gt=2) == pytest.approx(1.0)


def test_average_precision_no_ground_truth_is_zero() -> None:
    """AP is 0.0 when there is no ground truth to recover."""
    assert average_precision([(0.9, True)], n_gt=0) == 0.0


def test_average_precision_hand_computed() -> None:
    """A pooled TP/FP/TP sequence over 2 GTs yields the hand-computed AP 5/6.

    Sorted by score: 0.9 (TP), 0.8 (FP), 0.7 (TP), n_gt = 2.
      after 0.9: recall 0.5, precision 1.0
      after 0.8: recall 0.5, precision 0.5
      after 0.7: recall 1.0, precision 2/3
    Precision envelope -> [1.0, 2/3, 2/3]; integrating the step function over recall:
      0.5 * 1.0 + 0.5 * (2/3) = 0.5 + 1/3 = 5/6.
    """
    flags = [(0.9, True), (0.8, False), (0.7, True)]
    assert average_precision(flags, n_gt=2) == pytest.approx(5.0 / 6.0)


# ==================================================================================================
# DetectionMetric: mAP / mAR / IoU on a tiny synthetic set
# ==================================================================================================
def _gt() -> TFBox:
    return TFBox(0.0, 0.5, 0.0, 0.5)


def test_metric_perfect_detection() -> None:
    """A perfect detector (boxes echoed back) scores mAP = mAR = IoU = 1.0."""
    metric = DetectionMetric(track=DETECTION_TRACK)
    gt = _gt()
    metric.update(
        pred=[[TFBox(0.0, 0.5, 0.0, 0.5, score=1.0)]],
        target=[[gt]],
    )
    out = metric.compute()
    assert out == {"mAP": 1.0, "mAR": 1.0, "IoU": 1.0}


def test_metric_map_and_mar_hand_computed() -> None:
    """mAP / mAR on two images match the hand-computed 5/6 and 1.0.

    Image 0: 1 GT; preds = perfect match (score 0.9, TP) + a disjoint FP (score 0.8).
    Image 1: 1 GT; pred = perfect match (score 0.7, TP).
    Pooled TP/FP by score over 2 GTs -> AP = 5/6 (see test_average_precision_hand_computed);
    detection track has a single class-agnostic pool, so mAP == that AP. Both GTs are
    matched -> mAR = 2/2 = 1.0. Matched IoUs are both 1.0 -> IoU = 1.0.
    """
    gt = _gt()
    img0_pred = [
        TFBox(0.0, 0.5, 0.0, 0.5, score=0.9),  # perfect match -> TP
        TFBox(0.6, 1.0, 0.6, 1.0, score=0.8),  # disjoint -> FP
    ]
    img1_pred = [TFBox(0.0, 0.5, 0.0, 0.5, score=0.7)]  # perfect match -> TP

    metric = DetectionMetric(track=DETECTION_TRACK)
    metric.update(pred=[img0_pred, img1_pred], target=[[gt], [gt]])
    out = metric.compute()
    assert out["mAP"] == pytest.approx(5.0 / 6.0)
    assert out["mAR"] == pytest.approx(1.0)
    assert out["IoU"] == pytest.approx(1.0)


def test_metric_below_threshold_is_not_matched() -> None:
    """A predicted box whose IoU is under the threshold counts as a miss (mAP 0, mAR 0).

    Pred = time[0,1] x freq[0,0.5] vs GT = time[0,1] x freq[0.25,0.75] -> IoU 1/3 < 0.5.
    """
    metric = DetectionMetric(track=DETECTION_TRACK, iou_threshold=0.5)
    pred = TFBox(0.0, 1.0, 0.0, 0.5, score=0.9)
    gt = TFBox(0.0, 1.0, 0.25, 0.75)
    metric.update(pred=[[pred]], target=[[gt]])
    out = metric.compute()
    assert out["mAP"] == 0.0
    assert out["mAR"] == 0.0


def test_metric_reset_clears_state() -> None:
    """``reset()`` drops accumulated images so the metric can be reused."""
    metric = DetectionMetric(track=DETECTION_TRACK)
    metric.update(pred=[[_gt()]], target=[[_gt()]])
    metric.reset()
    out = metric.compute()
    assert out == {"mAP": 0.0, "mAR": 0.0, "IoU": 0.0}


def test_metric_primary_key_is_map() -> None:
    """The primary ranking key is ``mAP`` and is a key of ``compute()``."""
    metric = DetectionMetric(track=DETECTION_TRACK)
    metric.update(pred=[[_gt()]], target=[[_gt()]])
    assert metric.primary_key == "mAP"
    assert "mAP" in metric.compute()


def test_metric_eval_conditions_records_threshold_and_track() -> None:
    """The metric reports the IoU threshold + track for ``eval.conditions``."""
    metric = DetectionMetric(track=RECOGNITION_TRACK, iou_threshold=0.5)
    assert metric.eval_conditions() == {"iou_threshold": 0.5, "track": RECOGNITION_TRACK}


def test_metric_recognition_track_separates_classes() -> None:
    """On the recognition track a right-place-wrong-class prediction is a false positive.

    One GT of class 1. A perfectly-localised prediction of class 2 must NOT match, so
    mAP = 0 on the recognition (per-class) track even though the box overlaps exactly.
    """
    gt = TFBox(0.0, 0.5, 0.0, 0.5, label=1)
    wrong_class = TFBox(0.0, 0.5, 0.0, 0.5, label=2, score=0.9)
    metric = DetectionMetric(track=RECOGNITION_TRACK)
    metric.update(pred=[[wrong_class]], target=[[gt]])
    out = metric.compute()
    assert out["mAP"] == 0.0
    assert out["mAR"] == 0.0


def test_metric_detection_track_is_class_agnostic_with_real_labels() -> None:
    """On the detection track a right-place-wrong-class prediction still matches (mAP = 1.0).

    Real RadDet boxes carry a (hashed) class id, not the ``-1`` sentinel, so this guards that
    the detection track scores LOCALISATION ONLY -- a pred of class 7 recovers a GT of class 3
    when the box overlaps exactly. The torchmetrics production path collapses labels to a single
    class for the same reason (see ``DetectionMetric._compute_torchmetrics``); this locks the
    stdlib path's matching semantics the two must agree on.
    """
    gt = TFBox(0.0, 0.5, 0.0, 0.5, label=3)
    other_class = TFBox(0.0, 0.5, 0.0, 0.5, label=7, score=0.9)
    metric = DetectionMetric(track=DETECTION_TRACK)
    metric.update(pred=[[other_class]], target=[[gt]])
    out = metric.compute()
    assert out["mAP"] == pytest.approx(1.0)
    assert out["mAR"] == pytest.approx(1.0)


def test_metric_unknown_track_raises() -> None:
    """Constructing the metric with an unknown track raises a clear error."""
    with pytest.raises(ValueError, match="unknown detection track"):
        DetectionMetric(track="bogus")


# ==================================================================================================
# Task registration + tracks
# ==================================================================================================
def test_task_registered_under_name() -> None:
    """The task resolves by name from the registry to ``WidebandDetectionTask``."""
    assert "wideband_detection" in TASKS.names()
    assert TASKS.get("wideband_detection") is WidebandDetectionTask
    assert isinstance(get_task("wideband_detection"), WidebandDetectionTask)


def test_task_metadata() -> None:
    """Name, version, default split and primary metric match the protocol."""
    task = WidebandDetectionTask()
    assert task.name == "wideband_detection"
    assert task.version == "v1"
    assert task.default_split() == "test"
    assert task.metrics()[0].primary_key == "mAP"
    assert task.datasets()[0].name == "raddet"


def test_task_tracks_kept_distinct() -> None:
    """Detection and recognition are distinct tracks, never blended (WP-22)."""
    task = WidebandDetectionTask()
    assert task.tracks() == [DETECTION_TRACK, RECOGNITION_TRACK]
    assert DETECTION_TRACK != RECOGNITION_TRACK
    assert set(task.tracks()) == set(TRACKS)


def test_task_build_targets_extracts_boxes() -> None:
    """``build_targets`` returns the per-image list of T-F box targets."""
    gt = _gt()
    batch: Batch = {"iq": [[0.0], [0.1]], "boxes": [[gt], [gt, gt]], "meta": [{}, {}]}
    targets = WidebandDetectionTask().build_targets(batch)
    assert [len(image) for image in targets] == [1, 2]
    assert all(isinstance(box, TFBox) for image in targets for box in image)


def test_task_build_targets_accepts_mapping_boxes() -> None:
    """``build_targets`` also normalises boxes given as annotation mappings."""
    box = {"t_start": 0.0, "t_stop": 0.5, "f_low": 0.0, "f_high": 0.5, "class": "bpsk"}
    batch: Batch = {"boxes": [[box]]}
    targets = WidebandDetectionTask().build_targets(batch)
    assert isinstance(targets[0][0], TFBox)


# ==================================================================================================
# Dataset adapter (synthetic in-memory path)
# ==================================================================================================
def test_dataset_load_yields_iq_boxes_meta() -> None:
    """The dataset load() serves synthetic ``(iq, boxes, meta)`` samples verbatim."""
    gt = _gt()
    samples: list[Batch] = [
        {"iq": [0.0], "boxes": [gt], "meta": {"id": 0}},
        {"iq": [0.1], "boxes": [gt], "meta": {"id": 1}},
    ]
    dataset = WidebandDetectionTask(samples=samples).datasets()[0]
    loaded = list(dataset.load("test"))
    assert len(loaded) == 2
    assert loaded[0]["iq"] == [0.0]
    assert loaded[0]["boxes"] == [gt]
    assert loaded[1]["meta"] == {"id": 1}


def test_dataset_canonical_split_id_is_track_specific() -> None:
    """The canonical split id encodes the track and matches the schema id pattern."""
    det = WidebandDetectionTask(track=DETECTION_TRACK).datasets()[0]
    rec = WidebandDetectionTask(track=RECOGNITION_TRACK).datasets()[0]
    assert det.canonical_split_id == "detect-raddet-detection-8010-seed42-v1"
    assert rec.canonical_split_id == "detect-raddet-recognition-8010-seed42-v1"
    assert det.canonical_split_id != rec.canonical_split_id


# ==================================================================================================
# evaluate() end-to-end -> schema-valid result.json with metrics.primary == "mAP"
# ==================================================================================================
class _PerfectDetector(Model):
    """A deterministic detector that echoes each image's GT boxes back as predictions."""

    name = "perfect-detector"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        preds: list[list[TFBox]] = []
        for image_boxes in x["boxes"]:
            preds.append(
                [
                    TFBox(b.t_start, b.t_stop, b.f_low, b.f_high, label=b.label, score=1.0)
                    for b in image_boxes
                ]
            )
        return preds

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 4242


def _synthetic_batch() -> list[Batch]:
    gt = _gt()
    other = TFBox(0.6, 1.0, 0.6, 1.0)
    return [
        {"iq": [0.0], "boxes": [gt], "meta": {"id": 0}},
        {"iq": [0.1], "boxes": [gt, other], "meta": {"id": 1}},
        {"iq": [0.2], "boxes": [other], "meta": {"id": 2}},
    ]


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    return schema


def _run(track: str = DETECTION_TRACK, out_path: Path | None = None) -> dict[str, Any]:
    task = WidebandDetectionTask(track=track, samples=_synthetic_batch())
    return evaluate(
        _PerfectDetector(),
        task,
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        track=track,
        batch_size=2,
        out_path=out_path,
    )


def test_evaluate_end_to_end_validates_against_schema() -> None:
    """evaluate() emits a dict that independently validates against result.schema.json."""
    from jsonschema import Draft202012Validator

    result = _run()
    Draft202012Validator(_load_schema()).validate(result)


def test_evaluate_primary_is_map_and_in_values() -> None:
    """``metrics.primary`` is ``mAP`` and appears as a key of ``metrics.values``."""
    result = _run()
    assert result["metrics"]["primary"] == "mAP"
    values = result["metrics"]["values"]
    assert "mAP" in values
    assert set(values) == {"mAP", "mAR", "IoU"}
    # Perfect detector on this synthetic set recovers every GT.
    assert values["mAP"] == pytest.approx(1.0)
    assert values["mAR"] == pytest.approx(1.0)
    assert values["IoU"] == pytest.approx(1.0)


def test_evaluate_records_track_and_conditions() -> None:
    """The scored track and IoU threshold are recorded on the result."""
    result = _run(track=DETECTION_TRACK)
    assert result["split"]["track"] == DETECTION_TRACK
    assert result["eval"]["conditions"]["iou_threshold"] == DEFAULT_IOU_THRESHOLD
    assert result["eval"]["conditions"]["track"] == DETECTION_TRACK
    assert result["task"]["name"] == "wideband_detection"
    assert result["task"]["version"] == "v1"


def test_evaluate_writes_valid_file(tmp_path: Path) -> None:
    """When ``out_path`` is given the on-disk result re-validates against the schema."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "wideband" / "result.json"
    result = _run(out_path=out_path)
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


def test_metric_torchmetrics_falls_back_when_extra_absent() -> None:
    """The lazy torchmetrics path degrades gracefully to the stdlib path (extra absent).

    ``rfbench[detection]`` is not installed in the light env, so requesting the production
    path must warn and fall back rather than raise, keeping ``compute()`` total.
    """
    metric = DetectionMetric(track=DETECTION_TRACK, use_torchmetrics=True)
    metric.update(pred=[[_gt()]], target=[[_gt()]])
    with pytest.warns(RuntimeWarning, match="rfbench\\[detection\\]"):
        out = metric.compute()
    assert out == {"mAP": 1.0, "mAR": 1.0, "IoU": 1.0}


def test_no_heavy_deps_imported() -> None:
    """The task path must not pull torchmetrics/numpy/torch into ``sys.modules``.

    Guards the HARD CONSTRAINT that ``import rfbench.tasks.wideband_detection`` and the
    pure-stdlib metric path stay dependency-free.
    """
    import sys

    for heavy in ("torchmetrics", "numpy", "torch"):
        assert heavy not in sys.modules, f"{heavy} must not be imported by the stdlib path"
