"""Acceptance tests for the RadDet YOLOv3 wideband-detection baseline.

Pure stdlib: NO ultralytics / torch / numpy imported here. The detector's ultralytics backend
is replaced by an injected ``predict_fn`` seam, so the whole model<->metric BRIDGE (forward ->
per-image predicted boxes -> ``DetectionMetric`` mAP -> schema-valid ``result.json``) is
exercised on plain-Python boxes with only ``pytest`` + ``jsonschema`` installed.

Covers:
  * the raw-detection -> T-F box conversion (axis mapping, clamping, class names);
  * ``forward`` returns per-image box lists aligned with ``build_targets``;
  * an ``evaluate()`` end-to-end with the real model class (injected backend) validating
    against the result schema with ``metrics.primary == "mAP"``;
  * the RadDet split loader now carries ``image_path`` and the task's cache-load path filters
    by split (so eval scores only the requested split's images);
  * the training driver's pure helpers (data.yaml writer, committed-checksum lookup);
  * the module top stays dependency-free (no ultralytics import).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Regime, RegimeSpec
from rfbench.core.registry import MODELS
from rfbench.core.types import Batch
from rfbench.models.baselines.raddet_detector import (
    PredictFn,
    RadDetDetector,
    RawImageDetections,
    _class_name,
    tfboxes_from_detections,
)
from rfbench.tasks.wideband_detection.task import (
    DETECTION_TRACK,
    TFBox,
    WidebandDetectionTask,
)
from rfbench.training_detection import official_split_checksum, write_raddet_data_yaml

_RADDET_NAMES = ("Rect", "Barker", "Frank")


# ==================================================================================================
# Raw-detection -> T-F box conversion (pure, no torch)
# ==================================================================================================
def test_tfboxes_axis_mapping_and_score() -> None:
    """xyxyn maps x->time, y->frequency (no flip); confidence rides along as score."""
    raw = RawImageDetections(boxes_xyxyn=[[0.1, 0.2, 0.5, 0.6]], scores=[0.87], class_ids=[0])
    boxes = tfboxes_from_detections(raw, _RADDET_NAMES)
    assert len(boxes) == 1
    box = boxes[0]
    assert box["class"] == "Rect"
    assert box["t_start"] == pytest.approx(0.1)
    assert box["t_stop"] == pytest.approx(0.5)
    assert box["f_low"] == pytest.approx(0.2)
    assert box["f_high"] == pytest.approx(0.6)
    assert box["score"] == pytest.approx(0.87)


def test_tfboxes_clamps_and_orders() -> None:
    """Out-of-range / inverted coordinates are clamped to [0,1] and ordered lo<=hi."""
    raw = RawImageDetections(boxes_xyxyn=[[0.6, 1.2, 0.2, -0.3]], scores=[0.5], class_ids=[1])
    box = tfboxes_from_detections(raw, _RADDET_NAMES)[0]
    assert box["t_start"] == pytest.approx(0.2)  # ordered from (0.6, 0.2)
    assert box["t_stop"] == pytest.approx(0.6)
    assert box["f_low"] == 0.0  # -0.3 clamped
    assert box["f_high"] == 1.0  # 1.2 clamped


def test_tfboxes_empty_frame() -> None:
    """A background frame (no detections) yields an empty box list."""
    assert tfboxes_from_detections(RawImageDetections(), _RADDET_NAMES) == []


def test_class_name_in_range_and_fallback() -> None:
    """In-range ids map to RadDet names; an out-of-range id degrades to ``class_<id>``."""
    assert _class_name(2, _RADDET_NAMES) == "Frank"
    assert _class_name(99, _RADDET_NAMES) == "class_99"


# ==================================================================================================
# forward() bridge with an injected backend (no ultralytics)
# ==================================================================================================
def _perfect_predictor(refs_to_boxes: dict[str, RawImageDetections]) -> PredictFn:
    """Return a predict_fn that echoes a per-image detection keyed by its image path."""

    def predict_fn(image_refs: Sequence[str]) -> list[RawImageDetections]:
        return [refs_to_boxes[str(ref)] for ref in image_refs]

    return predict_fn


def test_forward_returns_per_image_box_lists() -> None:
    """forward maps a collated batch (image_path list) to aligned per-image predicted boxes."""
    refs = {
        "img0": RawImageDetections([[0.0, 0.0, 0.5, 0.5]], [0.9], [0]),
        "img1": RawImageDetections(
            [[0.6, 0.6, 1.0, 1.0], [0.0, 0.0, 0.4, 0.4]], [0.8, 0.7], [1, 0]
        ),
    }
    model = RadDetDetector(predict_fn=_perfect_predictor(refs), class_names=_RADDET_NAMES)
    batch: Batch = {"image_path": ["img0", "img1"], "boxes": [[], []]}
    preds = model.forward(batch)
    assert [len(image) for image in preds] == [1, 2]
    assert preds[0][0]["t_stop"] == pytest.approx(0.5)
    assert preds[1][1]["class"] == "Rect"


def test_forward_requires_image_reference() -> None:
    """A batch with neither image_path nor sample_id fails loudly."""
    model = RadDetDetector(predict_fn=_perfect_predictor({}), class_names=_RADDET_NAMES)
    with pytest.raises(KeyError, match="neither 'image_path' nor 'sample_id'"):
        model.forward({"boxes": [[]]})


def test_model_registered() -> None:
    """The detector is registered under ``raddet_yolov3``."""
    assert "raddet_yolov3" in MODELS.names()
    assert MODELS.get("raddet_yolov3") is RadDetDetector


def test_n_params_declared_without_backend() -> None:
    """With an injected backend (no torch model) n_params reports the declared count."""
    model = RadDetDetector(
        predict_fn=_perfect_predictor({}), n_params=1234, class_names=_RADDET_NAMES
    )
    assert model.n_params == 1234


# ==================================================================================================
# evaluate() end-to-end through the real model class -> schema-valid result.json
# ==================================================================================================
def _gt() -> TFBox:
    return TFBox(0.0, 0.5, 0.0, 0.5)


def _other() -> TFBox:
    return TFBox(0.6, 1.0, 0.6, 1.0)


def _synthetic_samples() -> list[Batch]:
    """Three captures with known GT boxes + image paths keying the perfect predictor."""
    return [
        {"image_path": "img0", "boxes": [_gt()], "meta": {"id": 0}},
        {"image_path": "img1", "boxes": [_gt(), _other()], "meta": {"id": 1}},
        {"image_path": "img2", "boxes": [_other()], "meta": {"id": 2}},
    ]


def _perfect_detections() -> dict[str, RawImageDetections]:
    """Predictions that exactly cover each image's GT (xyxyn == [t0, f0, t1, f1])."""
    return {
        "img0": RawImageDetections([[0.0, 0.0, 0.5, 0.5]], [0.95], [0]),
        "img1": RawImageDetections(
            [[0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 1.0, 1.0]], [0.95, 0.9], [0, 1]
        ),
        "img2": RawImageDetections([[0.6, 0.6, 1.0, 1.0]], [0.9], [1]),
    }


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _run(out_path: Path | None = None) -> dict[str, Any]:
    task = WidebandDetectionTask(track=DETECTION_TRACK, samples=_synthetic_samples())
    model = RadDetDetector(
        predict_fn=_perfect_predictor(_perfect_detections()),
        class_names=_RADDET_NAMES,
        n_params=42,
    )
    return evaluate(
        model,
        task,
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        track=DETECTION_TRACK,
        batch_size=2,
        out_path=out_path,
        compute_bootstrap_ci=False,
    )


def test_evaluate_end_to_end_validates_against_schema() -> None:
    """evaluate() with the real detector class emits a schema-valid result with mAP primary."""
    from jsonschema import Draft202012Validator

    result = _run()
    Draft202012Validator(_load_schema()).validate(result)
    assert result["metrics"]["primary"] == "mAP"
    values = result["metrics"]["values"]
    assert set(values) == {"mAP", "mAR", "IoU"}
    # A perfect detector recovers every GT box on this synthetic set.
    assert values["mAP"] == pytest.approx(1.0)
    assert values["mAR"] == pytest.approx(1.0)
    assert values["IoU"] == pytest.approx(1.0)
    assert result["model"]["name"] == "raddet_yolov3"
    assert result["model"]["n_params"] == 42


def test_evaluate_writes_valid_file(tmp_path: Path) -> None:
    """The on-disk result re-validates and round-trips exactly."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "raddet" / "result.json"
    result = _run(out_path=out_path)
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


# ==================================================================================================
# Harness bridge: RadDet loader image_path + split-filtered cache load
# ==================================================================================================
def _write_raddet_tree(root: Path, tree: dict[str, dict[str, str]]) -> None:
    """Materialise a synthetic RadDet tree: ``images/<split>/<stem>.png`` + sibling ``.txt``."""
    for split, items in tree.items():
        split_dir = root / "raddet" / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem, label_text in items.items():
            (split_dir / f"{stem}.png").write_bytes(b"")
            (split_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def test_loader_emits_image_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_raddet_annotations now tags each sample with its spectrogram image_path."""
    from rfbench.data.download.detection_wbsig53 import load_raddet_annotations

    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    _write_raddet_tree(tmp_path, {"train": {"000000000000": "0 0.5 0.5 0.2 0.4\n"}})
    sample = load_raddet_annotations()[0]
    assert sample["sample_id"] == "train/000000000000"
    assert str(sample["image_path"]).endswith("images/train/000000000000.png")


def test_cache_load_filters_by_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The task's cache-load path serves ONLY the requested split's captures (no leakage)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    _write_raddet_tree(
        tmp_path,
        {
            "train": {f"{i:012d}": "0 0.5 0.5 0.2 0.2\n" for i in range(3)},
            "val": {"000000000100": "1 0.3 0.3 0.1 0.1\n"},
            "test": {"000000000200": "2 0.5 0.5 0.5 0.5\n", "000000000201": ""},
        },
    )
    dataset = WidebandDetectionTask(track=DETECTION_TRACK, official=True).datasets()[0]
    assert dataset.canonical_split_id == "detect-raddet-detection-official-v1"

    test_samples = list(dataset.load("test", DETECTION_TRACK))
    assert {s["sample_id"] for s in test_samples} == {"test/000000000200", "test/000000000201"}
    assert all("image_path" in s for s in test_samples)

    train_samples = list(dataset.load("train", DETECTION_TRACK))
    assert len(train_samples) == 3
    assert all(str(s["sample_id"]).startswith("train/") for s in train_samples)


# ==================================================================================================
# Training driver: pure helpers
# ==================================================================================================
def test_write_raddet_data_yaml(tmp_path: Path) -> None:
    """The data.yaml writer emits ultralytics-shaped paths + a class-id name map (no pyyaml)."""
    out = write_raddet_data_yaml(tmp_path / "variant", ["Rect", "Barker"], tmp_path / "data.yaml")
    text = out.read_text(encoding="utf-8")
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert "  0: Rect" in text
    assert "  1: Barker" in text
    assert str((tmp_path / "variant").resolve()) in text


def test_official_split_checksum_reads_committed_index(tmp_path: Path) -> None:
    """official_split_checksum returns the committed checksum, else None."""
    splits = tmp_path / "splits" / "raddet"
    splits.mkdir(parents=True)
    (splits / "detect-raddet-detection-official-v1.idx.json").write_text(
        json.dumps({"checksum": "sha256:deadbeef"}), encoding="utf-8"
    )
    got = official_split_checksum(
        "detect-raddet-detection-official-v1", splits_dir=tmp_path / "splits"
    )
    assert got == "sha256:deadbeef"
    assert official_split_checksum("nonexistent-split", splits_dir=tmp_path / "splits") is None


# ==================================================================================================
# Dependency discipline
# ==================================================================================================
def test_model_module_import_is_dependency_free() -> None:
    """Importing + constructing the detector must not pull ultralytics into sys.modules."""
    import sys

    RadDetDetector(predict_fn=_perfect_predictor({}), class_names=_RADDET_NAMES)
    assert "ultralytics" not in sys.modules
