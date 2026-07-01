"""The AMC :class:`~rfbench.core.task.Task` (WP-20), registered as ``"amc"``.

Automatic modulation classification: a single-label closed-set classification of an IQ
window into a modulation class. This binds the AMC datasets, metrics, canonical split and
target-extraction rule per ``docs/EVALUATION_PROTOCOL.md`` §AMC.

* ``datasets()`` -> the AMC dataset variants (RadioML 2016.10a / 2018.01a / Sig53), each an
  :class:`~rfbench.tasks.amc.dataset.AmcDataset`. Tests inject a synthetic in-memory dataset.
* ``metrics()`` -> ``[AccuracyOverall, AccuracyVsSnr, MacroF1]``; the FIRST is primary, so
  ``result.json.metrics.primary == "accuracy_overall"``.
* ``default_split()`` -> ``"test"``; ``tracks()`` -> ``["closed_set"]`` (single track).
* ``build_targets(batch)`` -> the per-sample modulation-class targets (the ``label`` field).

No numpy/torch import: ``import rfbench.tasks.amc`` stays dependency-free (stdlib + core).
"""

from __future__ import annotations

from collections.abc import Sequence

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.amc.dataset import AmcDataset
from rfbench.tasks.amc.metrics import AccuracyOverall, AccuracyVsSnr, MacroF1

#: The AMC dataset ids in the order ``datasets()`` returns them (first == default).
AMC_DATASET_NAMES: tuple[str, ...] = ("radioml_2016_10a", "radioml_2018_01a", "sig53")


@register_task("amc")
class AmcTask(Task):
    """Automatic modulation classification (AMC), registered as ``"amc"``.

    Instantiated with no arguments by ``rfbench.core.registry.get_task("amc")`` for the
    cluster path (real :class:`AmcDataset` variants). Tests pass ``datasets=`` to inject a
    synthetic in-memory dataset so the whole metric/evaluate path runs on pure-Python
    fixtures. ``version`` is the AMC protocol version and must match the ``-v<N>`` suffix of
    each ``canonical_split_id``.
    """

    name = "amc"
    version = "v1"

    def __init__(self, datasets: Sequence[Dataset] | None = None) -> None:
        """Bind the AMC datasets; default to the three real variants when none injected."""
        if datasets is None:
            self._datasets: list[Dataset] = [AmcDataset(name) for name in AMC_DATASET_NAMES]
        else:
            self._datasets = list(datasets)

    def datasets(self) -> list[Dataset]:
        """Return the AMC dataset variants (first is the default scored one)."""
        return list(self._datasets)

    def metrics(self) -> list[Metric]:
        """Return ``[AccuracyOverall (primary), AccuracyVsSnr, MacroF1]``.

        The primary metric is first, so ``evaluate`` sets ``metrics.primary`` to
        ``"accuracy_overall"``. ``AccuracyOverall`` also emits ``macro_f1`` and
        ``AccuracyVsSnr`` emits the ``accuracy_vs_snr`` curve, covering the AMC protocol.
        """
        return [AccuracyOverall(), AccuracyVsSnr(), MacroF1()]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``)."""
        return "test"

    def tracks(self) -> list[Track]:
        """Return the single closed-set track (AMC never blends tracks)."""
        return ["closed_set"]

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the modulation-class targets from a collated batch (the ``label`` field).

        ``batch`` is the collated batch-of-lists produced by ``evaluate``; the supervision
        target for AMC is the per-sample integer modulation-class id under ``"label"``.
        """
        return batch["label"]


__all__ = ["AmcTask", "AMC_DATASET_NAMES"]
