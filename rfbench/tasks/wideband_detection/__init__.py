"""Wideband-detection task package (WP-22).

Importing this package registers :class:`WidebandDetectionTask` under
``"wideband_detection"`` in :data:`rfbench.core.registry.TASKS` (the ``@register_task``
decorator runs at import). Only stdlib + the frozen ``rfbench.core`` contracts are pulled
in: every numpy/torch/torchmetrics import in :mod:`.task` is lazy, so
``import rfbench.tasks.wideband_detection`` stays dependency-free.
"""

from __future__ import annotations

from rfbench.tasks.wideband_detection.task import (
    DATASET_NAME,
    DEFAULT_IOU_THRESHOLD,
    DETECTION_TRACK,
    RECOGNITION_TRACK,
    TASK_VERSION,
    TRACKS,
    DetectionMetric,
    TFBox,
    WidebandDetectionDataset,
    WidebandDetectionTask,
    average_precision,
    average_recall,
    iou,
)

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
