"""Protocol-tech-ID (WiFi 802.11 standard recognition) task package.

Importing this package registers
:class:`~rfbench.tasks.protocol_tech_id.task.ProtocolTechIdTask` under the name
``"protocol_tech_id"`` in ``rfbench.core.registry.TASKS`` (registration is a side effect of
importing ``.task``), so ``rfbench.core.registry.get_task("protocol_tech_id")`` resolves it.

Re-exports the public protocol-tech-ID surface: the task, the dataset adapter and the two
classification metrics. ``import rfbench.tasks.protocol_tech_id`` stays dependency-free --
stdlib + the frozen core contracts only; numpy is imported lazily inside the dataset loaders.
"""

from __future__ import annotations

from rfbench.tasks.protocol_tech_id.dataset import ProtocolDataset
from rfbench.tasks.protocol_tech_id.metrics import AccuracyOverall, MacroF1
from rfbench.tasks.protocol_tech_id.task import (
    PROTOCOL_DATASET_NAMES,
    ProtocolTechIdTask,
)

__all__ = [
    "ProtocolTechIdTask",
    "PROTOCOL_DATASET_NAMES",
    "ProtocolDataset",
    "AccuracyOverall",
    "MacroF1",
]
