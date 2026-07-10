"""The SNR-estimation :class:`~rfbench.core.task.Task` (J4), registered as ``"snr_estimation"``.

SNR estimation: a scalar **regression** of a raw-IQ window to its signal-to-noise ratio (dB).
Unlike every other board task (which are classification), the target is a continuous float and
the metrics are error metrics (lower is better). This binds the SNR dataset, metrics, canonical
split and target-extraction rule per ``docs/EVALUATION_PROTOCOL.md`` "Statistical rigor &
uncertainty" -> "Regression metric (snr_estimation)".

* ``datasets()`` -> the SNR dataset variant(s) (RadioML 2016.10a), each a
  :class:`~rfbench.tasks.snr_estimation.dataset.SnrDataset`. Tests inject a synthetic
  in-memory dataset.
* ``metrics()`` -> ``[Rmse, Mae]``; the FIRST is primary, so
  ``result.json.metrics.primary == "rmse_db"``.
* ``default_split()`` -> ``"test"``; ``tracks()`` -> ``["all_snr"]`` (single track over the
  full SNR range -- no cherry-picking).
* ``build_targets(batch)`` -> the per-sample SNR (dB) targets (the ``snr_db`` field), the same
  field AMC carries as ``meta`` but which is the supervision target here.

No numpy/torch import: ``import rfbench.tasks.snr_estimation`` stays dependency-free (stdlib +
core).
"""

from __future__ import annotations

from collections.abc import Sequence

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.registry import register_task
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.snr_estimation.dataset import SnrDataset
from rfbench.tasks.snr_estimation.metrics import Mae, Rmse

#: The SNR-estimation dataset ids in the order ``datasets()`` returns them (first == default).
SNR_DATASET_NAMES: tuple[str, ...] = ("radioml_2016_10a",)


@register_task("snr_estimation")
class SnrEstimationTask(Task):
    """SNR estimation (raw-IQ -> SNR in dB regression), registered as ``"snr_estimation"``.

    Instantiated with no arguments by ``rfbench.core.registry.get_task("snr_estimation")`` for
    the cluster path (real :class:`SnrDataset` variant). Tests pass ``datasets=`` to inject a
    synthetic in-memory dataset so the whole metric/evaluate path runs on pure-Python fixtures.
    ``version`` is the SNR protocol version and must match the ``-v<N>`` suffix of each
    ``canonical_split_id``.
    """

    name = "snr_estimation"
    version = "v1"

    def __init__(self, datasets: Sequence[Dataset] | None = None) -> None:
        """Bind the SNR datasets; default to the RadioML 2016.10a variant when none injected."""
        if datasets is None:
            self._datasets: list[Dataset] = [SnrDataset(name) for name in SNR_DATASET_NAMES]
        else:
            self._datasets = list(datasets)

    def datasets(self) -> list[Dataset]:
        """Return the SNR dataset variants (first is the default scored one)."""
        return list(self._datasets)

    def metrics(self) -> list[Metric]:
        """Return ``[Rmse (primary), Mae]``.

        The primary metric is first, so ``evaluate`` sets ``metrics.primary`` to ``"rmse_db"``.
        Both are error metrics in dB (lower is better) over the full SNR range.
        """
        return [Rmse(), Mae()]

    def default_split(self) -> SplitName:
        """Return the partition scored by default (``"test"``)."""
        return "test"

    def tracks(self) -> list[Track]:
        """Return the single full-SNR-range track (SNR estimation never blends tracks)."""
        return ["all_snr"]

    def build_targets(self, batch: Batch) -> Tensor:
        """Extract the SNR (dB) regression targets from a collated batch (the ``snr_db`` field).

        ``batch`` is the collated batch-of-lists produced by ``evaluate``; the supervision
        target for SNR estimation is the per-sample SNR in dB under ``"snr_db"`` -- the very
        field AMC carries only as conditioning ``meta`` but which is the label here.
        """
        return batch["snr_db"]


__all__ = ["SnrEstimationTask", "SNR_DATASET_NAMES"]
