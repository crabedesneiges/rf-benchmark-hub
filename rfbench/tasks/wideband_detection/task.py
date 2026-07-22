"""WP-22 -- the wideband-detection task adapter (WBSig53).

Delivers, per ``docs/EVALUATION_PROTOCOL.md`` §Wideband detection:

* :class:`WidebandDetectionTask` -- a :class:`~rfbench.core.task.Task` registered as
  ``"wideband_detection"``. Its :meth:`~WidebandDetectionTask.tracks` keeps the
  **detection** (boxes only) and **recognition** (+per-box class) tracks distinct: the
  board never blends the two in one column. :meth:`~WidebandDetectionTask.build_targets`
  extracts the per-sample list of time-frequency box targets from a canonical batch.
* :class:`DetectionMetric` -- streaming ``mAP`` (**primary**), ``mAR`` and ``IoU`` over
  time-frequency boxes. ``IoU`` is computed on normalised ``(t_start, t_stop, f_low,
  f_high)`` extents in ``[0, 1]``. A light **pure-stdlib** IoU + AP-by-IoU-threshold path
  makes ``compute()`` exercisable on plain-Python predictions/targets (no numpy) so unit
  tests run dependency-free; a **lazy torchmetrics** path (``rfbench[detection]``) is used
  for the production computation and selected at runtime.
* :class:`WidebandDetectionDataset` -- a :class:`~rfbench.core.dataset.Dataset` whose
  :meth:`~WidebandDetectionDataset.load` yields ``(iq, boxes, meta)`` samples. Real loading
  is lazy/cluster-only (numpy/h5py/torchsig behind ``rfbench[detection]``); tests drive it
  with synthetic in-memory boxes.

HARD CONSTRAINT: ``import rfbench.tasks.wideband_detection`` must stay dependency-free.
Only stdlib + the frozen ``rfbench.core`` contracts are imported at module top; every
numpy/torch/torchmetrics import is LAZY (inside the function that needs it) and guarded
with a clear ``rfbench[...]`` install hint.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.splits import SplitManifest
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track

#: Task protocol version (``EVALUATION_PROTOCOL.md`` §Wideband detection). Any metric or
#: split change bumps it; matches the ``-v<N>`` suffix of the canonical split id.
TASK_VERSION = "v1"

#: The real published wideband-detection dataset used by the board: RadDet (ICASSP 2025).
#: WBSig53 is generation-only (no static release) and, per policy, is not synthesised here.
DATASET_NAME = "raddet"

#: The two reporting tracks, kept distinct and never mixed in one board column:
#: ``detection`` scores presence/localisation (boxes only), ``recognition`` additionally
#: scores each box's signal class.
DETECTION_TRACK: Track = "detection"
RECOGNITION_TRACK: Track = "recognition"
TRACKS: tuple[Track, Track] = (DETECTION_TRACK, RECOGNITION_TRACK)

#: Default IoU threshold at which a predicted box is matched to a ground-truth box for
#: the ``mAP``/``mAR`` primary computation (COCO's canonical AP@0.5 lower anchor). Baked
#: into ``eval.conditions.iou_threshold`` for reproducibility.
DEFAULT_IOU_THRESHOLD = 0.5

#: Install hint surfaced when the (optional) torchmetrics production path is requested.
_DETECTION_EXTRA_HINT = (
    "the torchmetrics mAP path needs torch + torchmetrics; "
    "install it with `pip install rfbench[detection]` (else the pure-stdlib path is used)."
)


# ==================================================================================================
# Time-frequency boxes + geometry (pure stdlib)
# ==================================================================================================
@dataclass(frozen=True, slots=True)
class TFBox:
    """A normalised time-frequency detection box.

    Localises a signal in time (``t_start`` <= ``t_stop``) and frequency (``f_low`` <=
    ``f_high``), both in ``[0, 1]``, and carries a signal ``label`` (an integer class id;
    ``-1`` means "class-agnostic", used by the detection track where only localisation is
    scored). ``score`` is the detector's confidence for a *predicted* box and is ignored
    for a ground-truth box.
    """

    t_start: float
    t_stop: float
    f_low: float
    f_high: float
    label: int = -1
    score: float = 1.0

    @property
    def area(self) -> float:
        """Area of the box in the normalised time-frequency plane."""
        return max(0.0, self.t_stop - self.t_start) * max(0.0, self.f_high - self.f_low)


def _box_from_mapping(box: Mapping[str, Any]) -> TFBox:
    """Build a :class:`TFBox` from a plain mapping (the annotation-sidecar shape).

    Accepts the canonical annotation fields ``t_start/t_stop/f_low/f_high`` plus an
    optional ``class``/``label`` and ``score``. Bounds are validated so a malformed box
    fails loudly rather than silently skewing the metric.
    """
    t_start = float(box["t_start"])
    t_stop = float(box["t_stop"])
    f_low = float(box["f_low"])
    f_high = float(box["f_high"])
    label_raw = box.get("label", box.get("class", -1))
    label = _coerce_label(label_raw)
    score = float(box.get("score", 1.0))
    for name, lo, hi in (("time", t_start, t_stop), ("freq", f_low, f_high)):
        if lo > hi:
            raise ValueError(f"box {name} extent is inverted ({lo} > {hi}) in {dict(box)!r}")
    for name, value in (
        ("t_start", t_start),
        ("t_stop", t_stop),
        ("f_low", f_low),
        ("f_high", f_high),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"box {name}={value} outside normalised [0, 1] in {dict(box)!r}")
    return TFBox(t_start, t_stop, f_low, f_high, label=label, score=score)


def _coerce_label(raw: object) -> int:
    """Coerce a raw class label (int or str name) into a stable non-negative id.

    Integer labels pass through; string class names hash to a stable non-negative id so a
    recognition-track box keeps a comparable class identity without a shared vocabulary.
    ``-1`` (class-agnostic) is preserved.
    """
    if isinstance(raw, bool):  # bool is an int subclass; treat as class-agnostic sentinel
        return -1
    if isinstance(raw, int):
        return raw
    text = str(raw)
    # Deterministic, positive, small-ish id (only used for equality, never printed).
    return abs(hash(text)) % (10**9)


def _as_box(box: TFBox | Mapping[str, Any]) -> TFBox:
    """Normalise a box given either as a :class:`TFBox` or a plain mapping."""
    return box if isinstance(box, TFBox) else _box_from_mapping(box)


def iou(box_a: TFBox | Mapping[str, Any], box_b: TFBox | Mapping[str, Any]) -> float:
    """Intersection-over-union of two time-frequency boxes (pure stdlib).

    Standard 2-D box IoU with the axes being *time* (``t_start``..``t_stop``) and
    *frequency* (``f_low``..``f_high``), both normalised to ``[0, 1]``. Identical boxes
    give ``1.0``, disjoint boxes give ``0.0``. If the union degenerates to zero area the
    IoU is ``0.0``.
    """
    a = _as_box(box_a)
    b = _as_box(box_b)
    inter_t = max(0.0, min(a.t_stop, b.t_stop) - max(a.t_start, b.t_start))
    inter_f = max(0.0, min(a.f_high, b.f_high) - max(a.f_low, b.f_low))
    intersection = inter_t * inter_f
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


# ==================================================================================================
# Pure-stdlib average precision / recall by IoU threshold
# ==================================================================================================
def _match_image(
    preds: Sequence[TFBox],
    gts: Sequence[TFBox],
    *,
    iou_threshold: float,
    class_agnostic: bool,
) -> tuple[list[tuple[float, bool]], int]:
    """Greedily match one image's predictions to its ground-truth boxes.

    Returns ``(scored_flags, n_gt)`` where ``scored_flags`` is a list of
    ``(confidence, is_true_positive)`` for every prediction (highest-confidence first,
    each GT matched at most once) and ``n_gt`` is the ground-truth count. When
    ``class_agnostic`` is ``False`` a prediction may only match a GT of the same class
    (recognition track); otherwise class is ignored (detection track).
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    matched: set[int] = set()
    flags: list[tuple[float, bool]] = []
    for pi in order:
        pred = preds[pi]
        best_iou = iou_threshold
        best_gt = -1
        for gi, gt in enumerate(gts):
            if gi in matched:
                continue
            if not class_agnostic and pred.label != gt.label:
                continue
            overlap = iou(pred, gt)
            if overlap >= best_iou:
                best_iou = overlap
                best_gt = gi
        is_tp = best_gt >= 0
        if is_tp:
            matched.add(best_gt)
        flags.append((pred.score, is_tp))
    return flags, len(gts)


def average_precision(
    flags: Sequence[tuple[float, bool]],
    n_gt: int,
) -> float:
    """All-point (VOC-style) average precision from scored TP/FP flags (pure stdlib).

    ``flags`` are ``(confidence, is_tp)`` pairs pooled over all images for one IoU
    threshold; ``n_gt`` is the total ground-truth count. Precision is made monotonically
    non-increasing (the standard AP envelope) and integrated over the full recall range,
    so it matches the value a hand computation yields on a tiny set. Returns ``0.0`` when
    there is no ground truth.
    """
    if n_gt == 0:
        return 0.0
    ordered = sorted(flags, key=lambda f: f[0], reverse=True)
    tp = 0
    fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _score, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))

    # Precision envelope: p_interp(r) = max precision at recall >= r.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate the step function over recall, anchored at recall 0.
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in zip(recalls, precisions, strict=True):
        if recall > prev_recall:
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return ap


def average_recall(n_true_positives: int, n_gt: int) -> float:
    """Recall = matched ground-truth boxes / total ground-truth boxes.

    Used for ``mAR`` at the operating IoU threshold; returns ``0.0`` when there is no
    ground truth.
    """
    if n_gt == 0:
        return 0.0
    return n_true_positives / n_gt


# ==================================================================================================
# The detection metric (mAP primary, mAR, IoU)
# ==================================================================================================
class DetectionMetric(Metric):
    """Streaming wideband-detection metric: ``mAP`` (primary), ``mAR`` and ``IoU``.

    Accumulates per-image ``(predicted_boxes, ground_truth_boxes)`` and computes, at the
    operating IoU threshold:

    * ``mAP`` -- mean average precision, averaged over classes (recognition track) or a
      single class-agnostic pool (detection track). **Primary** ranking metric.
    * ``mAR`` -- mean average recall (matched GT / total GT) at the threshold.
    * ``IoU`` -- mean IoU of the greedily-matched (prediction, GT) pairs, a localisation
      quality summary.

    The default computation is a **pure-stdlib** AP-by-IoU-threshold path so ``compute()``
    runs on plain-Python boxes with no numpy (unit tests). Passing ``use_torchmetrics=True``
    selects a **lazy torchmetrics** path (``rfbench[detection]``) for the production
    computation; the stdlib path is always the fallback.
    """

    name = "detection"
    primary_key = "mAP"

    def __init__(
        self,
        *,
        track: Track = DETECTION_TRACK,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_torchmetrics: bool = False,
    ) -> None:
        """Configure the metric for a track + IoU threshold.

        ``track`` selects class-agnostic matching (``"detection"``) vs per-class matching
        (``"recognition"``). ``use_torchmetrics`` opts into the lazy production path.
        """
        if track not in TRACKS:
            raise ValueError(f"unknown detection track {track!r}; expected one of {list(TRACKS)}")
        self._track = track
        self._class_agnostic = track == DETECTION_TRACK
        self._iou_threshold = float(iou_threshold)
        self._use_torchmetrics = use_torchmetrics
        # Per-image (preds, gts) accumulator; kept as boxes so both paths can consume it.
        self._images: list[tuple[list[TFBox], list[TFBox]]] = []
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated per-image boxes."""
        self._images = []

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate one batch of predicted vs ground-truth box lists.

        ``pred`` and ``target`` are per-image sequences of boxes (each box a
        :class:`TFBox` or a mapping); the batch axis lines them up image-by-image, matching
        the ``(iq, boxes, meta)`` collation. ``meta`` is unused (detection has no per-sample
        stratification), accepted only to satisfy the :class:`Metric` contract.
        """
        del meta  # detection metrics are not stratified by conditioning
        for pred_boxes, gt_boxes in zip(pred, target, strict=True):
            self._images.append(
                (
                    [_as_box(box) for box in pred_boxes],
                    [_as_box(box) for box in gt_boxes],
                )
            )

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{"mAP", "mAR", "IoU"}`` at the operating IoU threshold.

        Dispatches to the lazy torchmetrics path when requested and importable, else to the
        pure-stdlib path. The stdlib path is what the dependency-free unit tests exercise.
        """
        if self._use_torchmetrics:
            torch_result = self._compute_torchmetrics()
            if torch_result is not None:
                return torch_result
        return self._compute_stdlib()

    # --- eval.conditions hook (consumed by evaluate()) ------------------------------------------
    def eval_conditions(self) -> dict[str, Any]:
        """Record the full-protocol detection conditions in ``eval.conditions``.

        The IoU threshold and the reporting track guard comparability, exactly as AMC
        records its full SNR range.
        """
        return {"iou_threshold": self._iou_threshold, "track": self._track}

    # --- pure-stdlib path -----------------------------------------------------------------------
    def _compute_stdlib(self) -> dict[str, float | list[dict[str, float]]]:
        """Pure-Python mAP / mAR / IoU (no numpy) -- the dependency-free default."""
        # Group predictions/GTs per class (or a single class-agnostic pool).
        classes = self._classes()
        aps: list[float] = []
        total_tp = 0
        total_gt = 0
        matched_ious: list[float] = []

        for cls in classes:
            flags: list[tuple[float, bool]] = []
            n_gt_class = 0
            for preds, gts in self._images:
                cls_preds = self._select(preds, cls)
                cls_gts = self._select(gts, cls)
                image_flags, n_gt = _match_image(
                    cls_preds,
                    cls_gts,
                    iou_threshold=self._iou_threshold,
                    class_agnostic=self._class_agnostic,
                )
                flags.extend(image_flags)
                n_gt_class += n_gt
            aps.append(average_precision(flags, n_gt_class))
            total_tp += sum(1 for _score, is_tp in flags if is_tp)
            total_gt += n_gt_class

        # IoU summary: mean IoU over greedily-matched (pred, gt) pairs across all images.
        matched_ious = self._matched_ious()

        mean_ap = sum(aps) / len(aps) if aps else 0.0
        mean_ar = average_recall(total_tp, total_gt)
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        return {"mAP": mean_ap, "mAR": mean_ar, "IoU": mean_iou}

    def _classes(self) -> list[int]:
        """Classes to average over: a single agnostic pool, or the sorted GT+pred labels."""
        if self._class_agnostic:
            return [-1]
        labels: set[int] = set()
        for preds, gts in self._images:
            labels.update(box.label for box in preds)
            labels.update(box.label for box in gts)
        return sorted(labels) if labels else [-1]

    def _select(self, boxes: Sequence[TFBox], cls: int) -> list[TFBox]:
        """Filter ``boxes`` to class ``cls`` (all boxes when class-agnostic)."""
        if self._class_agnostic:
            return list(boxes)
        return [box for box in boxes if box.label == cls]

    def _matched_ious(self) -> list[float]:
        """Greedily match preds->GTs per image and return each matched pair's IoU."""
        ious: list[float] = []
        for preds, gts in self._images:
            order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
            matched: set[int] = set()
            for pi in order:
                pred = preds[pi]
                best_iou = 0.0
                best_gt = -1
                for gi, gt in enumerate(gts):
                    if gi in matched:
                        continue
                    if not self._class_agnostic and pred.label != gt.label:
                        continue
                    overlap = iou(pred, gt)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_gt = gi
                if best_gt >= 0:
                    matched.add(best_gt)
                    ious.append(best_iou)
        return ious

    # --- lazy torchmetrics production path ------------------------------------------------------
    def _compute_torchmetrics(self) -> dict[str, float | list[dict[str, float]]] | None:
        """Production mAP via torchmetrics; ``None`` if the extra is not installed.

        ``torch`` + ``torchmetrics`` are imported LAZILY here (guarded with a clear
        ``rfbench[detection]`` hint) so the module import stays dependency-free. Boxes are
        converted to ``[t_start, f_low, t_stop, f_high]`` xyxy tensors. On any import
        failure this returns ``None`` and the caller falls back to the stdlib path.
        """
        try:
            import torch  # noqa: PLC0415
            from torchmetrics.detection.mean_ap import (  # noqa: PLC0415
                MeanAveragePrecision,
            )
        except ModuleNotFoundError:
            # Guarded, clear signal; the stdlib fallback keeps compute() total.
            import warnings  # noqa: PLC0415

            warnings.warn(_DETECTION_EXTRA_HINT, RuntimeWarning, stacklevel=2)
            return None

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[self._iou_threshold],
            class_metrics=False,
        )

        def _label(box: TFBox) -> int:
            # DETECTION track is class-AGNOSTIC: collapse every box to a single class so
            # torchmetrics matches on localisation ALONE, exactly like the stdlib path. Real
            # RadDet GT/pred boxes carry a (hashed) class id, so WITHOUT this collapse
            # torchmetrics would require class agreement even on the detection track -- silently
            # scoring recognition instead. Only the RECOGNITION track keeps per-box classes.
            return 0 if self._class_agnostic else max(box.label, 0)

        preds_payload: list[dict[str, Any]] = []
        target_payload: list[dict[str, Any]] = []
        for preds, gts in self._images:
            preds_payload.append(
                {
                    "boxes": torch.tensor(
                        [[b.t_start, b.f_low, b.t_stop, b.f_high] for b in preds],
                        dtype=torch.float32,
                    ).reshape(-1, 4),
                    "scores": torch.tensor([b.score for b in preds], dtype=torch.float32),
                    "labels": torch.tensor([_label(b) for b in preds], dtype=torch.int64),
                }
            )
            target_payload.append(
                {
                    "boxes": torch.tensor(
                        [[b.t_start, b.f_low, b.t_stop, b.f_high] for b in gts],
                        dtype=torch.float32,
                    ).reshape(-1, 4),
                    "labels": torch.tensor([_label(b) for b in gts], dtype=torch.int64),
                }
            )
        metric.update(preds_payload, target_payload)
        out = metric.compute()
        stdlib = self._compute_stdlib()  # for the IoU summary torchmetrics does not report
        return {
            "mAP": float(out["map"]),
            "mAR": float(out.get("mar_100", out.get("mar_1", 0.0))),
            "IoU": stdlib["IoU"],
        }


# ==================================================================================================
# The dataset adapter
# ==================================================================================================
class _InMemoryDetectionSplit:
    """A tiny map-style dataset over ``(iq, boxes, meta)`` samples (test/synthetic use)."""

    def __init__(self, samples: Sequence[Batch]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> Batch:
        return self._samples[index]


class WidebandDetectionDataset(Dataset):
    """RadDet wideband-detection dataset variant for one ``(split, track)``.

    :meth:`load` yields canonical ``{"iq", "boxes", "meta"}`` samples where ``boxes`` is a
    per-sample list of time-frequency boxes. Real loading (reading the RadDet YOLO boxes +
    the ``.annotations.json`` sidecar) is lazy/cluster-only behind ``rfbench[detection]``;
    unit tests pass ``samples=`` to drive an in-memory split with no heavy dependency.
    """

    name = DATASET_NAME

    def __init__(
        self,
        *,
        track: Track = DETECTION_TRACK,
        samples: Sequence[Batch] | None = None,
        official: bool = False,
    ) -> None:
        """Bind the dataset to a ``track`` and (optionally) an in-memory sample list.

        When ``samples`` is given the dataset serves them directly (synthetic/tests). The
        ``canonical_split_id`` / ``checksum`` mirror the ids produced by
        :mod:`rfbench.data.prepare.detection`; the checksum is a placeholder until the split
        is prepared on the cluster (it is overwritten by the prepared manifest's checksum).
        """
        if track not in TRACKS:
            raise ValueError(f"unknown detection track {track!r}; expected one of {list(TRACKS)}")
        self._track = track
        self._samples = list(samples) if samples is not None else None
        stem = f"detect-{DATASET_NAME}-{track}"
        self.canonical_split_id = f"{stem}-official-v1" if official else f"{stem}-8010-seed42-v1"
        # Placeholder integrity anchor; the prepared split manifest supplies the real one.
        self.checksum = "sha256:" + "0" * 64

    def download(self, cache: Path | None = None) -> None:
        """Fetch the real RadDet dataset into ``$RFBENCH_CACHE`` (lazy, cluster-only).

        Delegates to :func:`rfbench.data.download.detection_wbsig53.download_raddet`; heavy
        deps + Kaggle credentials are handled there and NEVER exercised in unit tests.
        """
        from rfbench.data.download.detection_wbsig53 import download_raddet  # noqa: PLC0415

        download_raddet(cache=cache)

    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build the canonical detection split + T-F annotations (lazy, cluster-only).

        Extracts per-sample boxes from the RadDet YOLO annotations (lazy) and hands them to
        :func:`rfbench.data.prepare.detection.prepare_detection`. Updates
        :attr:`canonical_split_id` / :attr:`checksum` from the returned manifest.
        """
        from rfbench.data.download.detection_wbsig53 import (  # noqa: PLC0415
            load_raddet_annotations,
        )
        from rfbench.data.prepare.detection import prepare_detection  # noqa: PLC0415

        samples = load_raddet_annotations()
        split, _manifest, _ann = prepare_detection(
            DATASET_NAME,
            out_dir="leaderboard",
            samples=samples,
            track=self._track,
            seed=seed,
        )
        self.canonical_split_id = split.canonical_split_id
        self.checksum = split.checksum
        return split

    def load(
        self,
        split: SplitName,
        track: Track | None = None,
    ) -> _InMemoryDetectionSplit:
        """Return a map-style dataset of ``{"iq", "boxes", "meta"}`` samples for ``split``.

        With in-memory ``samples`` (tests) it serves them verbatim. Otherwise it reads the
        prepared annotations sidecar + the generated IQ from ``$RFBENCH_CACHE`` via the lazy
        cluster-only loader. ``track`` overrides the dataset's configured track when given.
        """
        active_track = track if track is not None else self._track
        if active_track not in TRACKS:
            raise ValueError(
                f"unknown detection track {active_track!r}; expected one of {list(TRACKS)}"
            )
        if self._samples is not None:
            return _InMemoryDetectionSplit(self._samples)
        return self._load_from_cache(split, active_track)

    def _load_from_cache(self, split: SplitName, track: Track) -> _InMemoryDetectionSplit:
        """Lazy cluster-only path: read the RadDet split's ``(image_path, boxes)`` samples.

        :func:`~rfbench.data.download.detection_wbsig53.load_raddet_annotations` returns EVERY
        RadDet capture across all YOLO splits, each tagged ``sample_id = "<split>/<stem>"`` and
        carrying its spectrogram ``image_path`` (the real pixels a detector's ``forward`` loads)
        + normalised T-F ``boxes``. We adopt RadDet's OFFICIAL partition verbatim (the committed
        ``detect-raddet-detection-official-v1`` split), so filtering by the ``<split>`` prefix
        here yields exactly that split's images: ``evaluate("test", ...)`` therefore scores ONLY
        the official test set (no train/val leakage into the mAP), directly comparable to a
        paper's RadDet test number. ``track`` does not change WHICH samples are served (both
        tracks share the same captures; the metric decides class-agnostic vs per-class matching).
        """
        from rfbench.data.download.detection_wbsig53 import (  # noqa: PLC0415
            load_raddet_annotations,
        )

        del track  # both tracks share the same captures; track only steers the metric
        wanted = str(split)
        samples = [
            sample
            for sample in load_raddet_annotations()
            if str(sample.get("sample_id", "")).split("/", 1)[0] == wanted
        ]
        return _InMemoryDetectionSplit(cast("list[Batch]", samples))


# ==================================================================================================
# The task
# ==================================================================================================
@register_task("wideband_detection")
class WidebandDetectionTask(Task):
    """Wideband detection over WBSig53 (WP-22).

    Registered as ``"wideband_detection"``. Reports the **detection** and **recognition**
    tracks separately (never mixed in one column). Metrics are ``mAP`` (primary), ``mAR``
    and ``IoU`` over normalised time-frequency boxes; the primary metric ranks the board.
    """

    name = "wideband_detection"
    version = TASK_VERSION

    def __init__(
        self,
        *,
        track: Track = DETECTION_TRACK,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_torchmetrics: bool = False,
        samples: Sequence[Batch] | None = None,
        official: bool = False,
    ) -> None:
        """Configure the task for a track + IoU threshold.

        ``samples`` (synthetic/tests) is threaded into the dataset adapter so
        :func:`rfbench.core.evaluate.evaluate` can drive an end-to-end run with no heavy
        dependency. ``use_torchmetrics`` opts the metric into the lazy production path.
        ``official=True`` binds the dataset to RadDet's adopted split id
        (``detect-<dataset>-<track>-official-v1``) so a cluster eval on the real tree reports
        the same canonical split the leaderboard commits (see
        :meth:`WidebandDetectionDataset.__init__`); the synthetic ``samples`` path leaves it
        ``False``.
        """
        if track not in TRACKS:
            raise ValueError(f"unknown detection track {track!r}; expected one of {list(TRACKS)}")
        self._track = track
        self._iou_threshold = iou_threshold
        self._use_torchmetrics = use_torchmetrics
        self._dataset = WidebandDetectionDataset(track=track, samples=samples, official=official)

    def datasets(self) -> list[Dataset]:
        """Return the single WBSig53 dataset variant."""
        return [self._dataset]

    def metrics(self) -> list[Metric]:
        """Return the detection metric (``mAP`` primary, then ``mAR`` and ``IoU``)."""
        return [
            DetectionMetric(
                track=self._track,
                iou_threshold=self._iou_threshold,
                use_torchmetrics=self._use_torchmetrics,
            )
        ]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``)."""
        return "test"

    def tracks(self) -> list[Track]:
        """Return the two reporting tracks, detection and recognition, kept distinct."""
        return list(TRACKS)

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the per-sample list of ground-truth time-frequency boxes.

        The canonical detection batch carries a ``"boxes"`` field: a per-image list of
        boxes (each a :class:`TFBox` or a mapping). Returns that per-image list of
        :class:`TFBox` targets, which the :class:`DetectionMetric` consumes directly.
        """
        raw_per_image = batch["boxes"]
        return [[_as_box(box) for box in image_boxes] for image_boxes in raw_per_image]


__all__ = [
    "TASK_VERSION",
    "DATASET_NAME",
    "TRACKS",
    "DETECTION_TRACK",
    "RECOGNITION_TRACK",
    "DEFAULT_IOU_THRESHOLD",
    "TFBox",
    "iou",
    "average_precision",
    "average_recall",
    "DetectionMetric",
    "WidebandDetectionDataset",
    "WidebandDetectionTask",
]
