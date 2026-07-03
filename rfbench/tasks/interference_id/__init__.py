"""Interference-ID (GNSS jamming classification) task package.

Importing this package registers
:class:`~rfbench.tasks.interference_id.task.InterferenceIdTask` under the name
``"interference_id"`` in ``rfbench.core.registry.TASKS`` (registration is a side effect of
importing ``.task``), so ``rfbench.core.registry.get_task("interference_id")`` resolves it.

Re-exports the public interference-ID surface: the task, the dataset adapter and the two
classification metrics. ``import rfbench.tasks.interference_id`` stays dependency-free --
stdlib + the frozen core contracts only; numpy is imported lazily inside the dataset loaders.
"""

from __future__ import annotations

from rfbench.tasks.interference_id.dataset import InterferenceDataset
from rfbench.tasks.interference_id.metrics import AccuracyOverall, MacroF1
from rfbench.tasks.interference_id.task import (
    INTERFERENCE_DATASET_NAMES,
    InterferenceIdTask,
)

__all__ = [
    "InterferenceIdTask",
    "INTERFERENCE_DATASET_NAMES",
    "InterferenceDataset",
    "AccuracyOverall",
    "MacroF1",
]
