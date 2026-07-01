"""WP-21 -- the SEI (RF fingerprinting) task.

:class:`SeiTask` binds the SEI datasets (WiSig / ORACLE), the SEI metrics and the
per-condition tracks per ``docs/EVALUATION_PROTOCOL.md`` §SEI. It is registered as
``"sei"`` so ``rfbench.core.registry.TASKS.get("sei")`` (and ``get_task("sei")``) resolve
it.

The task is **track-aware**: an instance carries the active ``track`` (defaulting to
``closed_set`` so the zero-arg registry construction is valid) and reports the correct,
*separate* metric family for it:

* ``closed_set`` / ``cross_receiver`` / ``cross_day`` -> closed-set identification, scored
  by :class:`~rfbench.tasks.sei.metrics.Rank1Accuracy` (PRIMARY ``rank1_accuracy``);
* ``open_set`` -> open-set verification, scored by
  :class:`~rfbench.tasks.sei.metrics.OpenSetMetric` (``auroc`` + ``eer``).

Closed-set and open-set metrics are never conflated: a given instance emits exactly one
family, and :meth:`build_targets` extracts the matching supervision target (transmitter id
for closed-set, the binary genuine/impostor label for open-set).

Module-top imports are stdlib + the frozen core contracts + this task's own
(dependency-free) submodules; no numpy/torch at import time so
``import rfbench.tasks.sei`` stays dependency-free.
"""

from __future__ import annotations

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.sei.dataset import SeiDataset
from rfbench.tasks.sei.metrics import OpenSetMetric, Rank1Accuracy

#: The SEI tracks/conditions reported *separately* on the board (protocol §SEI). The first
#: three are closed-set identification conditions; ``open_set`` is the open-set
#: verification protocol. WiSig carries all four; ORACLE is closed-set only.
SEI_TRACKS: tuple[Track, ...] = ("closed_set", "cross_receiver", "cross_day", "open_set")

#: Tracks scored with the open-set metric family (AUROC + EER); the rest are closed-set.
_OPEN_SET_TRACKS: frozenset[Track] = frozenset({"open_set"})


@register_task("sei")
class SeiTask(Task):
    """The SEI / RF-fingerprinting benchmark task (registered as ``"sei"``).

    ``track`` selects the active condition and hence the metric family and default dataset
    variant; it defaults to ``closed_set`` so the registry's zero-arg construction yields
    the primary (rank-1) closed-set configuration. ``dataset`` selects the WiSig / ORACLE
    variant (default WiSig, the only one carrying the cross-receiver / cross-day
    conditions).
    """

    name = "sei"
    version = "v1"

    def __init__(self, track: Track = "closed_set", *, dataset: str = "wisig") -> None:
        """Bind the task to a ``(track, dataset)`` (defaults: closed-set WiSig)."""
        if track not in SEI_TRACKS:
            raise ValueError(f"unknown SEI track {track!r}; expected one of {list(SEI_TRACKS)}")
        self._track = track
        self._dataset_name = dataset

    @property
    def track(self) -> Track:
        """The active track/condition this instance scores."""
        return self._track

    def datasets(self) -> list[Dataset]:
        """Return the dataset variant bound to the active track."""
        return [SeiDataset(self._dataset_name, track=self._track)]

    def metrics(self) -> list[Metric]:
        """Return the *single* metric family for the active track (never conflated).

        Closed-set tracks -> :class:`Rank1Accuracy` (primary ``rank1_accuracy``); the
        ``open_set`` track -> :class:`OpenSetMetric` (``auroc`` + ``eer``). The two are
        kept strictly separate so a closed-set row never carries open-set scalars and
        vice-versa.
        """
        if self._track in _OPEN_SET_TRACKS:
            return [OpenSetMetric()]
        return [Rank1Accuracy()]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``).

        The default *track* is ``closed_set``, whose canonical split id is exposed via the
        dataset's ``canonical_split_id`` (``sei-wisig-closedset-...``).
        """
        return "test"

    def tracks(self) -> list[Track]:
        """Return all SEI tracks: closed-set conditions + the open-set protocol.

        ``closed_set`` / ``cross_receiver`` / ``cross_day`` are the distinct closed-set
        conditions; ``open_set`` is the open-set reporting track. Each is scored and
        reported as its own ``result.json`` row -- the board never blends them.
        """
        return list(SEI_TRACKS)

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the supervision target for the active track from a canonical ``batch``.

        Closed-set tracks -> the transmitter-identity labels (``batch["label"]``). The
        ``open_set`` track -> the binary genuine/impostor labels; these live under
        ``batch["genuine"]`` when present, else fall back to ``batch["label"]`` (already
        binarised upstream). The score itself is produced by the model's ``forward`` and
        consumed by :class:`OpenSetMetric`.
        """
        if self._track in _OPEN_SET_TRACKS and "genuine" in batch:
            return batch["genuine"]
        return batch["label"]


__all__ = ["SeiTask", "SEI_TRACKS"]
