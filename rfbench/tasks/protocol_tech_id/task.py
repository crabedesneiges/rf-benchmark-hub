"""The protocol-tech-ID :class:`~rfbench.core.task.Task`, registered as ``"protocol_tech_id"``.

WiFi 802.11 standard recognition: a single-label closed-set classification of a raw-IQ window
into one of four 802.11 standards (``802.11b``, ``802.11g``, ``802.11n``, ``802.11ax``). This
binds the protocol-tech-ID datasets, metrics, canonical split and target-extraction rule per
``docs/EVALUATION_PROTOCOL.md`` §protocol_tech_id. Distinct from AMC (recognises the WiFi
*standard*, not the modulation scheme).

* ``datasets()`` -> the protocol-tech-ID dataset variants (``tprime_wifi4``), each a
  :class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset`. Tests inject a synthetic
  in-memory dataset.
* ``metrics()`` -> ``[AccuracyOverall, MacroF1]``; the FIRST is primary, so
  ``result.json.metrics.primary == "accuracy_overall"``. Both are single-label classification
  metrics reused from / mirroring the AMC ones (no SNR curve for this task).
* ``default_split()`` -> ``"test"``; ``tracks()`` -> ``["closed_set"]`` (single track).
* ``build_targets(batch)`` -> the per-sample class targets (the ``label`` field).

No numpy/torch import: ``import rfbench.tasks.protocol_tech_id`` stays dependency-free (stdlib +
core).
"""

from __future__ import annotations

from collections.abc import Sequence

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.protocol_tech_id.dataset import ProtocolDataset
from rfbench.tasks.protocol_tech_id.metrics import AccuracyOverall, MacroF1

#: The protocol-tech-ID dataset ids in the order ``datasets()`` returns them (first == default).
PROTOCOL_DATASET_NAMES: tuple[str, ...] = ("tprime_wifi4",)


@register_task("protocol_tech_id")
class ProtocolTechIdTask(Task):
    """WiFi 802.11 standard recognition (protocol-tech-ID), registered ``"protocol_tech_id"``.

    Instantiated with no arguments by ``rfbench.core.registry.get_task("protocol_tech_id")`` for
    the cluster path (real :class:`ProtocolDataset` variants). Tests pass ``datasets=`` to
    inject a synthetic in-memory dataset so the whole metric/evaluate path runs on pure-Python
    fixtures. ``version`` is the protocol-tech-ID protocol version and must match the ``-v<N>``
    suffix of each ``canonical_split_id``.
    """

    name = "protocol_tech_id"
    version = "v1"

    def __init__(self, datasets: Sequence[Dataset] | None = None) -> None:
        """Bind the protocol datasets; default to the real variant when none injected."""
        if datasets is None:
            self._datasets: list[Dataset] = [
                ProtocolDataset(name) for name in PROTOCOL_DATASET_NAMES
            ]
        else:
            self._datasets = list(datasets)

    def datasets(self) -> list[Dataset]:
        """Return the protocol-tech-ID dataset variants (first is the default scored one)."""
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
        """Return the evaluation tracks, never blended: ``closed_set`` (within-distribution, rooms
        mixed) and ``cross_room`` (the paper's scenario-split: leave-one-location-out)."""
        return ["closed_set", "cross_room"]

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the class targets from a collated batch (the ``label`` field).

        ``batch`` is the collated batch-of-lists produced by ``evaluate``; the supervision
        target for protocol-tech-ID is the per-sample integer standard id under ``"label"``.
        """
        return batch["label"]


__all__ = ["ProtocolTechIdTask", "PROTOCOL_DATASET_NAMES"]
