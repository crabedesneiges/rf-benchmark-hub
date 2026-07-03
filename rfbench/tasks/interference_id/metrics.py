"""Pure-stdlib interference-ID metrics.

Interference-ID (GNSS jamming classification) is a single-label closed-set classification
task with no SNR grid, so it reuses the task-agnostic AMC classification machinery but WITHOUT
the AMC-specific ``full_snr_range`` eval condition. Per ``docs/EVALUATION_PROTOCOL.md``
§interference_id the reported metrics are:

* ``accuracy_overall`` -- **primary**, top-1 accuracy over the whole test split. Its metric
  object also emits ``macro_f1`` (unweighted mean of per-class F1) as a second scalar so both
  land in ``result.json.metrics.values`` in one pass.
* ``macro_f1`` -- reported standalone via the reused :class:`rfbench.tasks.amc.metrics.MacroF1`.

The confusion-count helpers (:func:`_macro_f1`, :func:`_iter_pairs`, :func:`_as_class_index`)
are REUSED from :mod:`rfbench.tasks.amc.metrics`; only the primary metric is re-declared here so
its ``eval_conditions`` stays empty (no SNR leakage into ``result.json.eval.conditions``). No
numpy/torch import anywhere in this module, so the metric path runs with only ``pytest``
installed.
"""

from __future__ import annotations

from typing import Any

from rfbench.core.metric import Metric
from rfbench.core.types import Tensor
from rfbench.tasks.amc.metrics import MacroF1, _iter_pairs, _macro_f1

__all__ = ["AccuracyOverall", "MacroF1"]


class AccuracyOverall(Metric):
    """Top-1 accuracy over the whole test split (**primary**), plus macro-F1.

    Streams ``(pred, target)`` batches, accumulating the confusion counts needed for both
    overall accuracy and the per-class F1 used by the macro average. :meth:`compute` returns
    both ``accuracy_overall`` (the primary key) and ``macro_f1`` as scalars, so a single metric
    object fills the two mandatory interference-ID scalars.

    Unlike :class:`rfbench.tasks.amc.metrics.AccuracyOverall`, this metric declares NO
    ``eval_conditions``: interference-ID has no SNR grid, so nothing extra is written to
    ``result.json.eval.conditions``.
    """

    name = "accuracy_overall"
    primary_key = "accuracy_overall"

    def __init__(self) -> None:
        """Initialise empty confusion accumulators."""
        self._tp: dict[int, int] = {}
        self._fp: dict[int, int] = {}
        self._fn: dict[int, int] = {}
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        """Clear all accumulated confusion counts."""
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()
        self._correct = 0
        self._total = 0

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate overall-accuracy and per-class confusion counts for one batch."""
        for predicted, expected in _iter_pairs(pred, target):
            self._total += 1
            self._tp.setdefault(expected, 0)
            if predicted == expected:
                self._correct += 1
                self._tp[expected] += 1
            else:
                self._fn[expected] = self._fn.get(expected, 0) + 1
                self._fp[predicted] = self._fp.get(predicted, 0) + 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{accuracy_overall, macro_f1}`` as scalars over the whole split."""
        accuracy = self._correct / self._total if self._total else 0.0
        return {
            "accuracy_overall": accuracy,
            "macro_f1": _macro_f1(self._tp, self._fp, self._fn),
        }
