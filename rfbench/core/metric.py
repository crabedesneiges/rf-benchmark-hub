"""The ``Metric`` contract: streaming, JSON-serialisable evaluation metrics.

A :class:`Metric` accumulates predictions over batches and then computes a
JSON-serialisable dict. Scalars land in ``result.json.metrics.values``; curves
(lists of ``{x, y[, label]}``) land in ``result.json.metrics.curves``.
:attr:`Metric.primary_key` is the ranking-metric name and MUST appear as a key of the
scalar dict returned by :meth:`compute` (and therefore of ``metrics.values``).

No ``torch`` import at module top: tensors are typed via
:data:`rfbench.core.types.Tensor` so ``import rfbench.core`` stays dependency-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rfbench.core.types import Tensor


class Metric(ABC):
    """A streaming metric.

    Accumulate over batches with :meth:`update`, then produce a JSON-serialisable
    dict with :meth:`compute`. :attr:`primary_key` is the ranking metric name and
    MUST appear as a key of the scalar dict / ``metrics.values``.
    """

    #: Human-readable metric name.
    name: str
    #: Ranking-metric key, e.g. ``"accuracy_overall"``, ``"rank1_accuracy"``,
    #: ``"auroc"``, ``"mAP"``, ``"pd@pfa=0.1"``. MUST be a key of :meth:`compute`.
    primary_key: str

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state so the metric can be reused."""

    @abstractmethod
    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate one batch of predictions and targets.

        ``meta`` carries per-sample conditioning for stratified curves/conditions,
        e.g. ``{"snr_db": ..., "receiver": ..., "day": ...}``.
        """

    @abstractmethod
    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return the computed metrics.

        The mapping is ``{scalar_name: float, curve_name: [{"x": .., "y": ..}, ...]}``
        and MUST include :attr:`primary_key` among its scalar keys.
        """


__all__ = ["Metric"]
