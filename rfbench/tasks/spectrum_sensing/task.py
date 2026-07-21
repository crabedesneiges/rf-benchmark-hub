"""The spectrum-sensing :class:`~rfbench.core.task.Task`, registered as ``"spectrum_sensing"``.

Binary spectrum-occupancy detection: each raw-IQ window is classified as occupied (target ``1``)
or vacant (target ``0``). This binds the spectrum-sensing datasets, metric, canonical split and
target-extraction rule per ``docs/EVALUATION_PROTOCOL.md`` §"Spectrum sensing".

* ``datasets()`` -> the spectrum-sensing dataset variants (``deepsense``), each a
  :class:`~rfbench.tasks.spectrum_sensing.dataset.SpectrumSensingDataset`. Tests inject a
  synthetic in-memory dataset.
* ``metrics()`` -> ``[PdAtPfa]``; it is primary, so ``result.json.metrics.primary ==
  "pd@pfa=0.1"``. The one metric object emits ``pd@pfa=0.1`` (primary), ``auroc``, ``pfa_achieved``
  and the ``roc`` curve in a single pass.
* ``default_split()`` -> ``"test"``; ``tracks()`` -> ``["occupancy"]`` (single track).
* ``build_targets(batch)`` -> the per-sample binary occupancy targets (the ``label`` field).

No numpy/torch import: ``import rfbench.tasks.spectrum_sensing`` stays dependency-free (stdlib +
core).
"""

from __future__ import annotations

from collections.abc import Sequence

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.spectrum_sensing.dataset import SpectrumSensingDataset
from rfbench.tasks.spectrum_sensing.metrics import PdAtPfa

#: The spectrum-sensing dataset ids in the order ``datasets()`` returns them (first == default).
SENSING_DATASET_NAMES: tuple[str, ...] = ("deepsense",)


@register_task("spectrum_sensing")
class SpectrumSensingTask(Task):
    """Binary spectrum-occupancy detection, registered as ``"spectrum_sensing"``.

    Instantiated with no arguments by ``rfbench.core.registry.get_task("spectrum_sensing")`` for
    the cluster path (real :class:`SpectrumSensingDataset` variants). Tests pass ``datasets=`` to
    inject a synthetic in-memory dataset so the whole metric/evaluate path runs on pure-Python
    fixtures. ``version`` is the spectrum-sensing protocol version and must match the ``-v<N>``
    suffix of each ``canonical_split_id``.
    """

    name = "spectrum_sensing"
    version = "v1"

    def __init__(self, datasets: Sequence[Dataset] | None = None) -> None:
        """Bind the sensing datasets; default to the real variant when none injected."""
        if datasets is None:
            self._datasets: list[Dataset] = [
                SpectrumSensingDataset(name) for name in SENSING_DATASET_NAMES
            ]
        else:
            self._datasets = list(datasets)

    def datasets(self) -> list[Dataset]:
        """Return the spectrum-sensing dataset variants (first is the default scored one)."""
        return list(self._datasets)

    def metrics(self) -> list[Metric]:
        """Return ``[PdAtPfa (primary)]``.

        The single metric object is primary, so ``evaluate`` sets ``metrics.primary`` to
        ``"pd@pfa=0.1"``. It emits ``auroc``, ``pfa_achieved`` and the ``roc`` curve alongside the
        primary Pd in the same pass.
        """
        return [PdAtPfa()]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``)."""
        return "test"

    def tracks(self) -> list[Track]:
        """Return the single occupancy track (spectrum sensing never blends tracks)."""
        return ["occupancy"]

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the binary occupancy targets from a collated batch (the ``label`` field).

        ``batch`` is the collated batch-of-lists produced by ``evaluate``; the supervision target
        for spectrum sensing is the per-sample binary occupancy label (``0`` vacant / ``1``
        occupied) under ``"label"``.
        """
        return batch["label"]


__all__ = ["SpectrumSensingTask", "SENSING_DATASET_NAMES"]
