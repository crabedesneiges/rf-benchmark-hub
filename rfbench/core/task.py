"""The ``Task`` contract: a canonical benchmark task.

A :class:`Task` binds together its datasets, metrics, default scored split, evaluation
tracks, and the rule for extracting supervision targets from a canonical batch. Its
``version`` is the protocol version from ``EVALUATION_PROTOCOL.md``: any metric or
split change bumps it and must match ``result.json.task.version`` (e.g. ``"v1"``).

No ``torch`` import at module top: tensors/batches are typed via
:data:`rfbench.core.types` so ``import rfbench.core`` stays dependency-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.types import Batch, SplitName, TaskName, Tensor, Track


class Task(ABC):
    """A canonical benchmark task.

    ``version`` is the protocol version from ``EVALUATION_PROTOCOL.md``; any metric or
    split change bumps it and must match ``result.json.task.version`` (schema pattern
    ``^v[0-9]+$``, e.g. ``"v1"``).
    """

    #: Registered task id, e.g. ``"amc"`` (matches ``rfbench/tasks/<name>/``).
    name: TaskName
    #: Protocol version, e.g. ``"v1"``; bumps on any metric/split change.
    version: str

    @abstractmethod
    def datasets(self) -> list[Dataset]:
        """Return the dataset variants this task can be evaluated on."""

    @abstractmethod
    def metrics(self) -> list[Metric]:
        """Return the task's metrics; the primary one ranks the board."""

    @abstractmethod
    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``).

        ``result.json.split.name`` is one of ``{test, val}``.
        """

    @abstractmethod
    def tracks(self) -> list[Track]:
        """Return the per-task evaluation tracks / conditions.

        e.g. SEI -> ``["closed_set", "cross_receiver", "cross_day", "open_set"]``;
        AMC -> ``["closed_set"]``. Each track yields its own ``result.json`` row and
        the board never blends tracks.
        """

    @abstractmethod
    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the supervision target tensor from a canonical ``batch``."""


__all__ = ["Task"]
