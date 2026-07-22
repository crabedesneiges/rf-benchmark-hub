"""RadDet wideband-detection training + eval driver -- YOLOv3 via ultralytics.

Two responsibilities, deliberately decoupled so eval never re-trains:

1. :func:`train_raddet_detector` -- fit a YOLOv3 detector on the RadDet YOLO tree
   (``<variant_root>/{images,labels}/{train,val}``) with the ultralytics trainer, returning the
   ``best.pt`` checkpoint. RadDet ships YOLO-format labels + a class map, so ultralytics trains
   on it directly; :func:`write_raddet_data_yaml` synthesises the ``data.yaml`` it needs.
2. :func:`evaluate_raddet_detector` -- wrap the trained weights in
   :class:`~rfbench.models.baselines.raddet_detector.RadDetDetector` and score them through the
   ONE canonical writer :func:`rfbench.core.evaluate.evaluate` on RadDet's OFFICIAL **test**
   split, emitting a schema-valid ``result.json``. The number is OUR ``mAP`` (the task's
   :class:`DetectionMetric`) on OUR committed split (``detect-raddet-detection-official-v1``),
   so it is board-consistent -- not ultralytics' internal COCO val.

HARD CONSTRAINT: ``import rfbench.training_detection`` stays dependency-free -- ``ultralytics`` /
``torch`` are imported LAZILY inside the training function. The eval path pulls only the
dep-free task/model/evaluate contracts, so this module imports (and is importable) with no DL
stack; the heavy work happens only when a function actually trains/infers on the cluster.

Regime: a from-scratch YOLOv3 (``arch="yolov3.yaml"``, random init) trained specifically on
RadDet is declared ``from_scratch`` -- the board regime for a specialised baseline, exactly as
the classification CNNs are. Result is written ``self_reported`` (only ``rfbench verify`` flips
it to ``verified``) and is NOT committed to the leaderboard pending review.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rfbench.core.evaluate import evaluate
from rfbench.core.model import Regime, RegimeSpec
from rfbench.core.types import Track
from rfbench.models.baselines.raddet_detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_NMS_IOU,
    RadDetDetector,
)
from rfbench.tasks.wideband_detection.task import DETECTION_TRACK, WidebandDetectionTask

logger = logging.getLogger(__name__)

#: Default from-scratch YOLOv3 training schedule (tune per compute budget on the cluster).
DEFAULT_EPOCHS = 100
#: Default per-step image batch for training (GB200 has headroom; raise if memory allows).
DEFAULT_TRAIN_BATCH = 16
#: Default per-step image batch for EVAL (ultralytics predict over the batch); modest so a
#: 20k-image test split streams without a memory spike.
DEFAULT_EVAL_BATCH = 32
#: Where the committed canonical split indices live (checksum provenance for the result row).
DEFAULT_SPLITS_DIR = "leaderboard/splits"


def write_raddet_data_yaml(
    variant_root: str | Path,
    class_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """Write the ultralytics ``data.yaml`` for a RadDet variant tree (pure stdlib).

    Points ultralytics at ``<variant_root>`` with ``images/{train,val,test}`` subdirs (it
    infers the sibling ``labels/`` tree by the usual ``images`` -> ``labels`` swap) and lists
    the class ``names`` in id order. Serialised as plain YAML by hand so this helper needs no
    ``pyyaml`` and is exercisable without any heavy dependency. Returns the written path.
    """
    root = Path(variant_root).resolve()
    lines = [
        f"path: {root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(class_names)],
        "",
    ]
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def train_raddet_detector(
    *,
    variant_root: str | Path,
    data_yaml: str | Path | None = None,
    arch: str = "yolov3.yaml",
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_TRAIN_BATCH,
    seed: int = 42,
    device: str | None = None,
    project: str | Path = "runs/raddet",
    run_name: str = "yolov3_from_scratch",
) -> Path:
    """Train a YOLOv3 detector on the RadDet tree; return the ``best.pt`` checkpoint path.

    ``ultralytics`` is imported LAZILY here (this is the only heavy path). A ``data.yaml`` is
    synthesised under the run dir when not supplied. Training uses the ultralytics defaults
    otherwise (SGD/AdamW auto, cosine schedule); ``seed`` is threaded for reproducibility.
    Returns ``<project>/<run_name>/weights/best.pt``. NEVER exercised in unit tests (cluster
    GPU only).
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "training the RadDet detector needs ultralytics (+ torch); install it with "
            "`pip install rfbench[raddet]`."
        ) from exc

    from rfbench.data.download.detection_wbsig53 import RADDET_CLASSES  # noqa: PLC0415

    root = Path(variant_root)
    if data_yaml is None:
        data_yaml = write_raddet_data_yaml(
            root, RADDET_CLASSES, Path(project) / run_name / "raddet_data.yaml"
        )
    logger.info("training %s on RadDet (%s) for %d epochs at imgsz=%d", arch, root, epochs, imgsz)
    model = YOLO(arch)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        device=device,
        project=str(project),
        name=run_name,
        exist_ok=True,
    )
    best = Path(project) / run_name / "weights" / "best.pt"
    logger.info("training done; best checkpoint at %s", best)
    return best


def official_split_checksum(
    split_id: str,
    *,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    dataset: str = "raddet",
) -> str | None:
    """Return the committed integrity checksum for ``split_id`` (or ``None`` if absent).

    Reads ``<splits_dir>/<dataset>/<split_id>.idx.json`` -- the committed canonical split index
    -- so the emitted ``result.json`` records the SAME ``split.checksum`` the leaderboard pins,
    tying the self-reported row to the exact adopted RadDet partition.
    """
    idx_path = Path(splits_dir) / dataset / f"{split_id}.idx.json"
    if not idx_path.is_file():
        return None
    doc = json.loads(idx_path.read_text(encoding="utf-8"))
    checksum = doc.get("checksum")
    return str(checksum) if checksum else None


def evaluate_raddet_detector(
    *,
    weights: str | Path,
    track: Track = DETECTION_TRACK,
    variant: str | None = None,
    out_path: str | Path | None = None,
    use_torchmetrics: bool = True,
    seed: int = 42,
    batch_size: int = DEFAULT_EVAL_BATCH,
    device: str = "cuda",
    imgsz: int = DEFAULT_IMGSZ,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    compute_bootstrap_ci: bool = False,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> dict[str, Any]:
    """Score a trained RadDet detector on the OFFICIAL test split -> schema-valid result dict.

    Binds :class:`WidebandDetectionTask` to RadDet's adopted split (``official=True``), pins the
    committed split checksum, wraps ``weights`` in
    :class:`~rfbench.models.baselines.raddet_detector.RadDetDetector`, and calls
    :func:`rfbench.core.evaluate.evaluate` on the ``test`` split with the declared
    ``from_scratch`` regime. ``use_torchmetrics`` selects the production ``mAP`` path
    (``rfbench[detection]``'s torchmetrics), falling back to the stdlib AP if absent.

    ``variant`` overrides ``$RFBENCH_RADDET_VARIANT`` (the RadDet resolution/density subtree,
    default ``512_9T``) so the loader resolves the same tree the detector was trained on.
    ``compute_bootstrap_ci`` defaults to ``False``: the detection metric's per-image greedy
    matching makes a 1000x resample over ~20k test images very slow, so per-run CIs are left as
    future work (flagged honestly rather than silently emitting a noisy interval). Writes
    ``out_path`` if given (``self_reported`` -- NOT committed to the board pending review) and
    returns the result dict.
    """
    if variant is not None:
        os.environ["RFBENCH_RADDET_VARIANT"] = variant

    task = WidebandDetectionTask(track=track, use_torchmetrics=use_torchmetrics, official=True)
    dataset = task.datasets()[0]
    checksum = official_split_checksum(dataset.canonical_split_id, splits_dir=splits_dir)
    if checksum is not None:
        dataset.checksum = checksum
    else:
        logger.warning(
            "no committed split index for %s; result.split.checksum stays the placeholder.",
            dataset.canonical_split_id,
        )

    model = RadDetDetector(
        weights=str(weights),
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
        device=device,
    )
    logger.info(
        "evaluating RadDet detector (%s track) on the official test split via %s mAP",
        track,
        "torchmetrics" if use_torchmetrics else "stdlib",
    )
    return evaluate(
        model,
        task,
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        dataset="raddet",
        track=track,
        seed=seed,
        batch_size=batch_size,
        device=device,
        out_path=Path(out_path) if out_path is not None else None,
        compute_bootstrap_ci=compute_bootstrap_ci,
    )


__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_TRAIN_BATCH",
    "DEFAULT_EVAL_BATCH",
    "DEFAULT_SPLITS_DIR",
    "write_raddet_data_yaml",
    "train_raddet_detector",
    "official_split_checksum",
    "evaluate_raddet_detector",
]
