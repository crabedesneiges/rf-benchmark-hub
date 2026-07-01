"""SEI (RF fingerprinting) task package -- WP-21.

Importing this package registers :class:`~rfbench.tasks.sei.task.SeiTask` under ``"sei"``
in ``rfbench.core.registry.TASKS`` (the ``@register_task`` decorator fires as a side
effect of importing :mod:`rfbench.tasks.sei.task`). Only stdlib + the frozen core
contracts are pulled in at import time, so ``import rfbench.tasks.sei`` stays
dependency-free (numpy/torch stay lazy inside the dataset loader).
"""

from __future__ import annotations

from rfbench.tasks.sei.dataset import SeiDataset
from rfbench.tasks.sei.metrics import OpenSetMetric, Rank1Accuracy
from rfbench.tasks.sei.task import SEI_TRACKS, SeiTask

__all__ = ["SeiTask", "SEI_TRACKS", "SeiDataset", "Rank1Accuracy", "OpenSetMetric"]
