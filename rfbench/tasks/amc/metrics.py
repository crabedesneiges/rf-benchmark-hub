"""Pure-stdlib AMC metrics (WP-20).

Automatic modulation classification is a single-label closed-set classification task.
Per ``docs/EVALUATION_PROTOCOL.md`` §AMC the reported metrics are:

* ``accuracy_overall`` -- **primary**, top-1 accuracy over the **full SNR range** (never
  cherry-picked). Its metric object also emits ``macro_f1`` (unweighted mean of per-class
  F1) as a second scalar so both land in ``result.json.metrics.values`` in one pass.
* ``accuracy_vs_snr`` -- a curve of top-1 accuracy per SNR bin, grouped by the per-sample
  ``snr_db`` carried in ``meta``. Lands in ``result.json.metrics.curves``.

Every ``compute()`` here is exercisable on pure-Python synthetic predictions/targets: no
numpy/torch import anywhere in this module. A batch of predictions may be either a list of
class indices (the pure-Python test path) or a list of per-class score vectors (a real
model's logits); :func:`_as_class_index` collapses both to an ``int`` class id with a lazy,
dependency-free ``argmax`` so the metric path runs with only ``pytest`` installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from rfbench.core.metric import Metric
from rfbench.core.types import Tensor

#: Full SNR range recorded in ``result.json.eval.conditions`` (RadioML convention:
#: -20..+18 dB in 2 dB steps). Guards comparability -- AMC is scored over the WHOLE range.
DEFAULT_SNR_DB_MIN = -20
DEFAULT_SNR_DB_MAX = 18


def _as_class_index(pred: Tensor) -> int:
    """Collapse one per-sample prediction to an integer class id.

    Accepts either an already-decoded class index (``int``/``bool``-free numeric) or a
    per-class score vector (``list``/``tuple`` of numbers, or anything iterable-with-len),
    in which case the ``argmax`` is taken. Ties go to the lowest index. Pure stdlib -- no
    numpy -- so the metric runs on synthetic Python predictions in tests.
    """
    if isinstance(pred, bool):  # bool is an int subclass; treat as the 0/1 class id
        return int(pred)
    # torch tensor / numpy array (duck-typed via ``ndim`` + ``argmax``, no import needed): a
    # 0-d value is already a class id; a 1-D per-class score vector -> its argmax.
    ndim = getattr(pred, "ndim", None)
    if ndim is not None and hasattr(pred, "argmax"):
        return int(pred) if ndim == 0 else int(pred.argmax())
    if isinstance(pred, (list, tuple)):
        scores = list(pred)
        if not scores:
            raise ValueError("prediction score vector is empty; cannot take argmax")
        best_i = 0
        best_v = scores[0]
        for i in range(1, len(scores)):
            if scores[i] > best_v:
                best_v = scores[i]
                best_i = i
        return best_i
    return int(pred)


def _iter_pairs(pred: Tensor, target: Tensor) -> Iterable[tuple[int, int]]:
    """Yield ``(predicted_class, true_class)`` integer pairs for one batch.

    ``pred`` and ``target`` are batch-of-samples sequences; predictions are decoded via
    :func:`_as_class_index` so both the class-index and the logits layouts are accepted.
    """
    for predicted, expected in zip(pred, target, strict=True):
        yield _as_class_index(predicted), int(expected)


def _snr_of(meta: dict[str, Any] | None, index: int) -> int:
    """Extract the SNR (dB) of sample ``index`` from a collated ``meta`` batch.

    The canonical batch field is ``snr_db`` (falling back to ``snr``); its value is the
    per-sample list produced by ``evaluate``'s collate. Raises ``KeyError`` if neither
    field is present, because ``accuracy_vs_snr`` cannot be grouped without the SNR.
    """
    if meta is None:
        raise KeyError("accuracy_vs_snr needs per-sample 'snr_db' in meta, got None")
    field = meta.get("snr_db", meta.get("snr"))
    if field is None:
        raise KeyError("accuracy_vs_snr needs a 'snr_db' (or 'snr') field in meta")
    return int(field[index])


def _macro_f1(
    per_class_tp: dict[int, int],
    per_class_fp: dict[int, int],
    per_class_fn: dict[int, int],
) -> float:
    """Unweighted mean of per-class F1 over every class that appears in the targets.

    A class present only as a false positive (predicted but never a true label) is not a
    ground-truth class and is excluded from the macro average, matching sklearn's default
    ``macro`` behaviour over the label set of the targets. Returns ``0.0`` when no class is
    present.
    """
    classes = sorted(per_class_tp)
    if not classes:
        return 0.0
    total = 0.0
    for cls in classes:
        tp = per_class_tp[cls]
        fp = per_class_fp.get(cls, 0)
        fn = per_class_fn.get(cls, 0)
        denom = 2 * tp + fp + fn
        total += (2 * tp / denom) if denom else 0.0
    return total / len(classes)


class AccuracyOverall(Metric):
    """Top-1 accuracy over the full SNR range (**primary**), plus macro-F1.

    Streams ``(pred, target)`` batches, accumulating the confusion counts needed for both
    overall accuracy and the per-class F1 used by the macro average. :meth:`compute` returns
    both ``accuracy_overall`` (the primary key) and ``macro_f1`` as scalars, so a single
    metric object fills the two mandatory AMC scalars. :meth:`eval_conditions` records the
    full SNR range so ``result.json.eval.conditions`` attests no cherry-picking.
    """

    name = "accuracy_overall"
    primary_key = "accuracy_overall"

    def __init__(
        self,
        snr_db_min: int = DEFAULT_SNR_DB_MIN,
        snr_db_max: int = DEFAULT_SNR_DB_MAX,
    ) -> None:
        """Record the full SNR range guard-rail and initialise empty accumulators."""
        self._snr_db_min = snr_db_min
        self._snr_db_max = snr_db_max
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
        """Return ``{accuracy_overall, macro_f1}`` as scalars over the full SNR range."""
        accuracy = self._correct / self._total if self._total else 0.0
        return {
            "accuracy_overall": accuracy,
            "macro_f1": _macro_f1(self._tp, self._fp, self._fn),
        }

    def eval_conditions(self) -> dict[str, Any]:
        """Record the full SNR range so the row attests no cherry-picking (AMC rule)."""
        return {
            "snr_db_min": self._snr_db_min,
            "snr_db_max": self._snr_db_max,
            "full_snr_range": True,
        }


class MacroF1(Metric):
    """Standalone macro-F1 metric (unweighted mean of per-class F1).

    Emits the ``macro_f1`` scalar on its own so the metric is unit-testable in isolation and
    reusable outside the AMC task. In the AMC task the value is identical to the ``macro_f1``
    emitted by :class:`AccuracyOverall`; ``evaluate`` merges the two identical scalars
    idempotently.
    """

    name = "macro_f1"
    primary_key = "macro_f1"

    def __init__(self) -> None:
        """Initialise empty per-class confusion accumulators."""
        self._tp: dict[int, int] = {}
        self._fp: dict[int, int] = {}
        self._fn: dict[int, int] = {}

    def reset(self) -> None:
        """Clear all accumulated per-class counts."""
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate per-class TP/FP/FN counts for one batch."""
        for predicted, expected in _iter_pairs(pred, target):
            self._tp.setdefault(expected, 0)
            if predicted == expected:
                self._tp[expected] += 1
            else:
                self._fn[expected] = self._fn.get(expected, 0) + 1
                self._fp[predicted] = self._fp.get(predicted, 0) + 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{macro_f1}``."""
        return {"macro_f1": _macro_f1(self._tp, self._fp, self._fn)}


class AccuracyVsSnr(Metric):
    """Top-1 accuracy grouped by SNR bin -> the ``accuracy_vs_snr`` curve.

    Groups every sample by its per-sample ``snr_db`` (read from ``meta``) and computes a
    top-1 accuracy per bin. :meth:`compute` returns a single curve key ``accuracy_vs_snr``
    whose value is a list of ``{"x": snr_db, "y": accuracy}`` points sorted by SNR, landing
    in ``result.json.metrics.curves``. ``primary_key`` points at the curve name only so the
    contract's "primary is a compute() key" invariant holds; the AMC task never uses this
    metric as its primary.
    """

    name = "accuracy_vs_snr"
    primary_key = "accuracy_vs_snr"

    def __init__(self) -> None:
        """Initialise the empty per-SNR-bin (correct, total) accumulators."""
        self._correct: dict[int, int] = {}
        self._total: dict[int, int] = {}

    def reset(self) -> None:
        """Clear all per-SNR-bin counters."""
        self._correct.clear()
        self._total.clear()

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate per-SNR-bin correct/total counts for one batch."""
        pairs = list(_iter_pairs(pred, target))
        for index, (predicted, expected) in enumerate(pairs):
            snr = _snr_of(meta, index)
            self._total[snr] = self._total.get(snr, 0) + 1
            if predicted == expected:
                self._correct[snr] = self._correct.get(snr, 0) + 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return the ``accuracy_vs_snr`` curve as SNR-sorted ``{x, y}`` points."""
        curve: list[dict[str, float]] = []
        for snr in sorted(self._total):
            total = self._total[snr]
            correct = self._correct.get(snr, 0)
            curve.append({"x": float(snr), "y": (correct / total) if total else 0.0})
        return {"accuracy_vs_snr": curve}

    def sample_bins(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> list[float]:
        """Per-sample SNR bin key (the curve's ``x``) for the per-bin bootstrap band.

        Returns one bin key per accumulated sample, aligned with ``pred``/``target``, so
        ``evaluate``'s per-bin percentile bootstrap can resample WITHIN each SNR bin and
        attach a ``y_low``/``y_high`` envelope to the ``accuracy_vs_snr`` points. The key is
        the same ``snr_db`` this metric groups on in :meth:`update`, read from the collated
        ``meta`` batch. Raises ``KeyError`` if the SNR field is absent (the curve cannot be
        binned without it).
        """
        n = len(target)
        return [float(_snr_of(meta, index)) for index in range(n)]


def accuracy(pred: Sequence[Any], target: Sequence[int]) -> float:
    """Convenience top-1 accuracy over two aligned sequences (used by tests).

    Decodes predictions via :func:`_as_class_index`; returns ``0.0`` for an empty input.
    """
    pairs = list(_iter_pairs(pred, target))
    if not pairs:
        return 0.0
    correct = sum(1 for predicted, expected in pairs if predicted == expected)
    return correct / len(pairs)


__all__ = [
    "DEFAULT_SNR_DB_MIN",
    "DEFAULT_SNR_DB_MAX",
    "AccuracyOverall",
    "MacroF1",
    "AccuracyVsSnr",
    "accuracy",
]
