"""AMC (automatic modulation classification) task package (WP-20).

Importing this package registers :class:`~rfbench.tasks.amc.task.AmcTask` under the name
``"amc"`` in ``rfbench.core.registry.TASKS`` (registration is a side effect of importing
``.task``), so ``rfbench.core.registry.get_task("amc")`` resolves it.

Re-exports the public AMC surface: the task, the dataset adapter and the three metrics.
``import rfbench.tasks.amc`` stays dependency-free -- stdlib + the frozen core contracts
only; numpy/h5py/torch are imported lazily inside the dataset loaders.
"""

from __future__ import annotations

from rfbench.tasks.amc.dataset import AmcDataset
from rfbench.tasks.amc.metrics import AccuracyOverall, AccuracyVsSnr, MacroF1
from rfbench.tasks.amc.task import AMC_DATASET_NAMES, AmcTask

__all__ = [
    "AmcTask",
    "AMC_DATASET_NAMES",
    "AmcDataset",
    "AccuracyOverall",
    "AccuracyVsSnr",
    "MacroF1",
]
