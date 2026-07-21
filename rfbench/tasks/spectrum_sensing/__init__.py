"""Spectrum-sensing (binary occupancy detection) task package.

Importing this package registers
:class:`~rfbench.tasks.spectrum_sensing.task.SpectrumSensingTask` under the name
``"spectrum_sensing"`` in ``rfbench.core.registry.TASKS`` (registration is a side effect of
importing ``.task``), so ``rfbench.core.registry.get_task("spectrum_sensing")`` resolves it.

Re-exports the public spectrum-sensing surface: the task, the dataset adapter and the
Pd@Pfa metric. ``import rfbench.tasks.spectrum_sensing`` stays dependency-free -- stdlib + the
frozen core contracts only; numpy is imported lazily inside the dataset loaders.
"""

from __future__ import annotations

from rfbench.tasks.spectrum_sensing.dataset import SpectrumSensingDataset
from rfbench.tasks.spectrum_sensing.metrics import PdAtPfa
from rfbench.tasks.spectrum_sensing.task import (
    SENSING_DATASET_NAMES,
    SpectrumSensingTask,
)

__all__ = [
    "SpectrumSensingTask",
    "SENSING_DATASET_NAMES",
    "SpectrumSensingDataset",
    "PdAtPfa",
]
