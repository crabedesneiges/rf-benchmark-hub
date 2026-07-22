"""RadDet wideband-detection baseline -- a YOLOv3 T-F box detector (ICASSP 2025).

The wideband-detection track (``docs/EVALUATION_PROTOCOL.md`` §Wideband detection) scores
**object detection** over spectrograms: predict a set of time-frequency bounding boxes per
capture, matched to ground truth by IoU (``mAP`` primary). This is a WHOLLY different stack
from the classification baselines -- there is no ``(B, n_classes)`` logit tensor; the model
emits, per image, a *variable-length list of boxes*.

Model choice -- **YOLOv3 via ultralytics**. The RadDet paper (Huang et al., arXiv:2501.10407)
reports RT-DETR-L / YOLOv3-L as its top detectors over RadDet spectrograms with YOLO-format
T-F boxes. We wrap **YOLOv3** because (1) it is the simpler, one-stage anchor detector -- far
cheaper to train and converge than the transformer-decoder RT-DETR; (2) RadDet ships
YOLO-format labels + ``data.yaml``, which ultralytics trains on directly (zero glue); (3)
the single ``ultralytics`` dependency also exposes RT-DETR, so ``arch`` is just a constructor
argument -- picking YOLOv3 does not lock RT-DETR out. It seeds the detection board the same
way MCLDNN/WiSig-CNN seed the classification boards.

Contract bridge (read ``rfbench/core/model.py`` + ``rfbench/tasks/wideband_detection/task.py``).
``forward`` receives the COLLATED detection batch that
:func:`rfbench.core.evaluate.evaluate` builds -- ``x["image_path"]`` is a *list* of spectrogram
PNG paths (one per capture; supplied by
:meth:`~rfbench.tasks.wideband_detection.task.WidebandDetectionDataset.load`) and ``x["boxes"]``
the ground truth. ``forward`` runs the detector on those images and returns a **per-image list
of predicted boxes**, each a plain mapping ``{class, t_start, t_stop, f_low, f_high, score}``
in the exact shape :class:`~rfbench.tasks.wideband_detection.task.DetectionMetric` consumes
(it ``zip``s pred vs GT image-by-image and matches by IoU). The axis convention mirrors the
GT loader (``rfbench/data/download/detection_wbsig53.py``): ultralytics' normalised ``xyxyn``
maps ``x -> time`` (``t_start = x1``, ``t_stop = x2``) and ``y -> frequency``
(``f_low = y1``, ``f_high = y2``), no flip -- so predictions and targets share one frame and
IoU is consistent.

HARD CONSTRAINT (deps discipline, see the task module): ``import
rfbench.models.baselines.raddet_detector`` must stay dependency-free. UNLIKE the classification
baselines (which import ``torch`` at module top), this module keeps EVERY heavy import
(``ultralytics``/``torch``) LAZY inside the method that needs it. That makes the model<->metric
bridge unit-testable with no DL stack: tests inject a ``predict_fn`` seam and drive
:func:`evaluate` end-to-end on plain-Python boxes. The ``@register_model`` entry is created on
an explicit ``import rfbench.models.baselines.raddet_detector``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor

#: A predicted box mapping in the canonical detection shape the metric consumes.
TFBoxDict = dict[str, object]

#: Default detector architecture (ultralytics model spec). ``"yolov3.yaml"`` is a RANDOM-init
#: YOLOv3 (genuine from-scratch training, the board regime for a specialised baseline);
#: ``"yolov3u.pt"`` / ``"rtdetr-l.pt"`` would start from published weights instead.
DEFAULT_ARCH = "yolov3.yaml"

#: Default square inference size (px). RadDet's canonical ``512_9T`` variant is 512x512.
DEFAULT_IMGSZ = 512

#: Confidence floor for kept detections. Deliberately LOW: mAP integrates precision over the
#: full recall range, so a detector should emit its whole ranked box list (a high conf floor
#: silently truncates recall and depresses AP).
DEFAULT_CONF_THRESHOLD = 0.001

#: IoU threshold for ultralytics' internal NMS (dedupe overlapping predictions before scoring).
DEFAULT_NMS_IOU = 0.7


@dataclass(frozen=True, slots=True)
class RawImageDetections:
    """One image's raw detector output, framework-agnostic (plain Python, no torch).

    ``boxes_xyxyn`` is a list of ``[x1, y1, x2, y2]`` rows normalised to ``[0, 1]`` (ultralytics
    ``Results.boxes.xyxyn``); ``scores`` the per-box confidences; ``class_ids`` the per-box
    integer class ids. The three lists are aligned and equal length (empty for a background
    frame). Keeping this a plain dataclass lets the ultralytics adapter and the test seam both
    produce it, so :func:`tfboxes_from_detections` is exercised with no DL dependency.
    """

    boxes_xyxyn: Sequence[Sequence[float]] = field(default_factory=list)
    scores: Sequence[float] = field(default_factory=list)
    class_ids: Sequence[int] = field(default_factory=list)


#: A pluggable inference backend: image paths -> one :class:`RawImageDetections` per image.
#: The real backend calls ultralytics; tests inject a deterministic stub.
PredictFn = Callable[[Sequence[str]], "list[RawImageDetections]"]
#: A pluggable embedding backend (image paths -> a feature tensor); optional.
EmbedFn = Callable[[Sequence[str]], Tensor]


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    """Map a detector class id to its RadDet class name, or a ``class_<id>`` fallback.

    Mirrors :func:`rfbench.data.download.detection_wbsig53._yolo_to_tf_box`: an in-range id
    resolves through ``class_names`` (the published ``RADDET_CLASSES`` order), an out-of-range
    id degrades to a raw ``"class_<id>"`` string rather than raising, so an unexpected class
    never aborts an eval. The metric coerces either form to a stable label.
    """
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the normalised ``[0, 1]`` box range."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def tfboxes_from_detections(
    raw: RawImageDetections,
    class_names: Sequence[str],
) -> list[TFBoxDict]:
    """Convert one image's raw detections into canonical T-F box mappings (pure, no torch).

    Each ``[x1, y1, x2, y2]`` normalised row becomes a box with ``x -> time``
    (``t_start = x1``, ``t_stop = x2``) and ``y -> frequency`` (``f_low = y1``, ``f_high = y2``)
    -- the SAME axis mapping the RadDet ground-truth loader uses, so predictions and targets
    live in one frame. Coordinates are clamped to ``[0, 1]`` and ordered ``lo <= hi`` (guards
    the sub-pixel overflow / degenerate boxes a detector can emit) so every box satisfies the
    metric's box contract. The class id resolves to a RadDet name; the confidence rides along
    as ``score`` (the metric ranks predictions by it for AP).
    """
    boxes: list[TFBoxDict] = []
    for row, score, class_id in zip(raw.boxes_xyxyn, raw.scores, raw.class_ids, strict=True):
        x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        t_start, t_stop = sorted((_clamp01(x1), _clamp01(x2)))
        f_low, f_high = sorted((_clamp01(y1), _clamp01(y2)))
        boxes.append(
            {
                "class": _class_name(int(class_id), class_names),
                "t_start": t_start,
                "t_stop": t_stop,
                "f_low": f_low,
                "f_high": f_high,
                "score": float(score),
            }
        )
    return boxes


def _default_class_names() -> tuple[str, ...]:
    """Return the published RadDet class order (lazy import keeps the module top dep-free)."""
    from rfbench.data.download.detection_wbsig53 import RADDET_CLASSES  # noqa: PLC0415

    return RADDET_CLASSES


def _image_refs(batch: Batch) -> list[str]:
    """Resolve the per-image spectrogram paths from a collated detection batch.

    Prefers the ``image_path`` field the RadDet loader now emits (the real pixels). Falls back
    to reconstructing ``<variant_root>/images/<split>/<stem>.png`` from the ``sample_id``
    (``"<split>/<stem>"``) + ``$RFBENCH_CACHE`` / ``$RFBENCH_RADDET_VARIANT`` when only the id is
    present. Raises a clear error if neither field is available (e.g. a raw synthetic batch with
    no image), so a misconfigured run fails loudly rather than silently predicting nothing.
    """
    if "image_path" in batch:
        return [str(path) for path in batch["image_path"]]
    if "sample_id" in batch:
        return [_image_path_from_sample_id(str(sid)) for sid in batch["sample_id"]]
    raise KeyError(
        "detection batch carries neither 'image_path' nor 'sample_id'; the RadDet detector "
        "needs the spectrogram pixels. Load the split via WidebandDetectionDataset.load (it "
        "emits image_path) or pass image_path per sample."
    )


def _image_path_from_sample_id(sample_id: str, cache: str | None = None) -> str:
    """Reconstruct a spectrogram path from ``"<split>/<stem>"`` + the RadDet cache root."""
    from rfbench.data.download.detection_wbsig53 import (  # noqa: PLC0415
        _RADDET_SUBDIR,
        _raddet_variant_root,
    )
    from rfbench.data.prepare._common import resolve_cache_dir  # noqa: PLC0415

    split, _, stem = sample_id.partition("/")
    variant_root = _raddet_variant_root(resolve_cache_dir(cache) / _RADDET_SUBDIR)
    return str(variant_root / "images" / split / f"{stem}.png")


@register_model("raddet_yolov3")
class RadDetDetector(Model):
    """The RadDet YOLOv3 wideband-detection baseline as a :class:`~rfbench.core.model.Model`.

    Registered ``"raddet_yolov3"``. Wraps an ultralytics detector to satisfy the frozen
    ``Model`` contract for the DETECTION bridge:

    * :meth:`forward` maps the collated detection batch (``x["image_path"]`` a list of
      spectrogram PNGs) to a per-image list of predicted T-F box mappings -- consumed directly,
      image-by-image, by :class:`DetectionMetric` (``mAP`` primary).
    * :meth:`embed` returns a backbone feature vector (``linear_probe`` / ``few_shot``); it is
      OPTIONAL for a baseline and best-effort here (ultralytics ``embed``), raising a clear
      error when no inference backend is available.
    * :attr:`n_params` reports the detector's parameter count; :attr:`family` is ``"baseline"``.

    Heavy imports are lazy: constructing the model with ``weights=`` loads the ultralytics
    network on first use, while tests pass ``predict_fn=`` to drive the bridge with no DL stack.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "raddet_yolov3",
        weights: str | None = None,
        arch: str = DEFAULT_ARCH,
        imgsz: int = DEFAULT_IMGSZ,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_iou: float = DEFAULT_NMS_IOU,
        device: str | None = None,
        class_names: Sequence[str] | None = None,
        predict_fn: PredictFn | None = None,
        embed_fn: EmbedFn | None = None,
        n_params: int | None = None,
    ) -> None:
        """Configure the detector.

        ``weights`` is a trained ultralytics checkpoint (``best.pt``); with it the real backend
        is loaded lazily. ``arch`` names the architecture for a fresh build. ``predict_fn`` /
        ``embed_fn`` inject a backend (tests / custom inference) and, when given, are used
        INSTEAD of ultralytics -- so the whole class runs dependency-free. ``class_names``
        overrides the default RadDet class order. ``n_params`` pre-declares the param count for
        the injected-backend path (where no torch model exists).
        """
        if not name:
            raise ValueError("RadDetDetector needs a non-empty name")
        self.name = name
        self._weights = weights
        self._arch = arch
        self._imgsz = imgsz
        self._conf = conf_threshold
        self._nms_iou = nms_iou
        self._device = device
        self._class_names = (
            tuple(class_names) if class_names is not None else _default_class_names()
        )
        self._predict_fn = predict_fn
        self._embed_fn = embed_fn
        self._declared_n_params = n_params
        self._model: object | None = None  # lazily-loaded ultralytics YOLO

    def forward(self, x: Batch) -> list[list[TFBoxDict]]:
        """Return per-image predicted T-F boxes for the collated detection batch ``x``.

        Resolves each capture's spectrogram path, runs the detector once over the batch, and
        converts every image's raw detections into the canonical box mappings the metric
        consumes. The returned outer list is aligned image-by-image with
        ``task.build_targets(x)``, exactly as :func:`rfbench.core.evaluate.evaluate` expects.
        """
        image_refs = _image_refs(x)
        raw_per_image = self._predict(image_refs)
        return [tfboxes_from_detections(raw, self._class_names) for raw in raw_per_image]

    def embed(self, x: Batch) -> Tensor:
        """Return a per-image backbone embedding (``linear_probe`` / ``few_shot``; optional).

        Uses the injected ``embed_fn`` when present, else ultralytics' ``embed`` on the loaded
        weights. Raises :class:`NotImplementedError` when neither is available -- embedding is
        not required for a detection baseline and is provided best-effort.
        """
        image_refs = _image_refs(x)
        if self._embed_fn is not None:
            return self._embed_fn(image_refs)
        if self._weights is None:
            raise NotImplementedError(
                "RadDetDetector.embed needs trained weights or an injected embed_fn; "
                "the detection baseline is scored on forward()/mAP, not embeddings."
            )
        model = self._ensure_model()
        return model.embed(  # type: ignore[attr-defined]
            source=list(image_refs),
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

    @property
    def n_params(self) -> int:
        """Total detector parameter count (written to ``result.json.model.n_params``)."""
        if self._model is not None:
            return self._count_params(self._model)
        if self._weights is not None:
            return self._count_params(self._ensure_model())
        return int(self._declared_n_params or 0)

    # --- inference backends ---------------------------------------------------------------------
    def _predict(self, image_refs: Sequence[str]) -> list[RawImageDetections]:
        """Dispatch to the injected ``predict_fn`` (tests) or the lazy ultralytics backend."""
        if self._predict_fn is not None:
            return list(self._predict_fn(image_refs))
        return self._ultralytics_predict(image_refs)

    def _ultralytics_predict(self, image_refs: Sequence[str]) -> list[RawImageDetections]:
        """Run the trained ultralytics detector and normalise its output (lazy torch path)."""
        model = self._ensure_model()
        results = model.predict(  # type: ignore[attr-defined]
            source=list(image_refs),
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._nms_iou,
            device=self._device,
            verbose=False,
            stream=False,
        )
        raw_per_image: list[RawImageDetections] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                raw_per_image.append(RawImageDetections())
                continue
            raw_per_image.append(
                RawImageDetections(
                    boxes_xyxyn=boxes.xyxyn.tolist(),
                    scores=boxes.conf.tolist(),
                    class_ids=[int(c) for c in boxes.cls.tolist()],
                )
            )
        return raw_per_image

    def _ensure_model(self) -> object:
        """Lazily build/load the ultralytics detector (``ultralytics`` imported here only)."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on the cluster
            raise RuntimeError(
                "the RadDet detector needs ultralytics (+ torch); install it with "
                "`pip install rfbench[raddet]`."
            ) from exc
        self._model = YOLO(self._weights if self._weights is not None else self._arch)
        return self._model

    @staticmethod
    def _count_params(model: object) -> int:
        """Best-effort parameter count of an ultralytics model (0 if not introspectable)."""
        inner = getattr(model, "model", None)
        params = getattr(inner, "parameters", None)
        if not callable(params):
            return 0
        return int(sum(p.numel() for p in params()))


__all__ = [
    "RadDetDetector",
    "RawImageDetections",
    "PredictFn",
    "EmbedFn",
    "TFBoxDict",
    "tfboxes_from_detections",
    "DEFAULT_ARCH",
    "DEFAULT_IMGSZ",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_NMS_IOU",
]
