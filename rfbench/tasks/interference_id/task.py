"""The interference-ID :class:`~rfbench.core.task.Task`, registered as ``"interference_id"``.

GNSS jamming classification: a single-label closed-set classification of a raw-IQ window into
one of six interference classes (``DME``, ``narrowband``, ``no_jamming``, ``single_am``,
``single_chirp``, ``single_fm``). This binds the interference-ID datasets, metrics, canonical
split and target-extraction rule per ``docs/EVALUATION_PROTOCOL.md`` §interference_id.

* ``datasets()`` -> the interference-ID dataset variants (``interf_gnss6``), each an
  :class:`~rfbench.tasks.interference_id.dataset.InterferenceDataset`. Tests inject a synthetic
  in-memory dataset.
* ``metrics()`` -> ``[AccuracyOverall, MacroF1]``; the FIRST is primary, so
  ``result.json.metrics.primary == "accuracy_overall"``. Both are single-label classification
  metrics reused from / mirroring the AMC ones (no SNR curve for this task).
* ``default_split()`` -> ``"test"``; ``tracks()`` -> ``["closed_set"]`` (single track).
* ``build_targets(batch)`` -> the per-sample class targets (the ``label`` field).

No numpy/torch import: ``import rfbench.tasks.interference_id`` stays dependency-free (stdlib +
core).
"""

from __future__ import annotations

from collections.abc import Sequence

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.interference_id.dataset import InterferenceDataset
from rfbench.tasks.interference_id.metrics import AccuracyOverall, MacroF1

#: The interference-ID dataset ids in the order ``datasets()`` returns them (first == default).
INTERFERENCE_DATASET_NAMES: tuple[str, ...] = ("interf_gnss6",)


@register_task("interference_id")
class InterferenceIdTask(Task):
    """GNSS jamming classification (interference-ID), registered as ``"interference_id"``.

    Instantiated with no arguments by ``rfbench.core.registry.get_task("interference_id")`` for
    the cluster path (real :class:`InterferenceDataset` variants). Tests pass ``datasets=`` to
    inject a synthetic in-memory dataset so the whole metric/evaluate path runs on pure-Python
    fixtures. ``version`` is the interference-ID protocol version and must match the ``-v<N>``
    suffix of each ``canonical_split_id``.
    """

    name = "interference_id"
    version = "v1"

    def __init__(self, datasets: Sequence[Dataset] | None = None) -> None:
        """Bind the interference datasets; default to the real variant when none injected."""
        if datasets is None:
            self._datasets: list[Dataset] = [
                InterferenceDataset(name) for name in INTERFERENCE_DATASET_NAMES
            ]
        else:
            self._datasets = list(datasets)

    def datasets(self) -> list[Dataset]:
        """Return the interference-ID dataset variants (first is the default scored one)."""
        return list(self._datasets)

    def metrics(self) -> list[Metric]:
        """Return ``[AccuracyOverall (primary), MacroF1]``.

        The primary metric is first, so ``evaluate`` sets ``metrics.primary`` to
        ``"accuracy_overall"``. ``AccuracyOverall`` also emits ``macro_f1`` in the same pass;
        ``MacroF1`` reports it standalone (``evaluate`` merges the identical scalar idempotently).
        """
        return [AccuracyOverall(), MacroF1()]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``)."""
        return "test"

    def tracks(self) -> list[Track]:
        """Return the single closed-set track (interference-ID never blends tracks)."""
        return ["closed_set"]

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the class targets from a collated batch (the ``label`` field).

        ``batch`` is the collated batch-of-lists produced by ``evaluate``; the supervision
        target for interference-ID is the per-sample integer jamming-class id under ``"label"``.
        """
        return batch["label"]


__all__ = ["InterferenceIdTask", "INTERFERENCE_DATASET_NAMES"]
