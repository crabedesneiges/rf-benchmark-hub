"""SEI metrics -- pure-stdlib streaming implementations (WP-21).

Two *separate*, never-conflated metric families per ``docs/EVALUATION_PROTOCOL.md`` §SEI:

* **closed-set** -> :class:`Rank1Accuracy` (PRIMARY, ``rank1_accuracy``): the fraction of
  test emitters whose predicted identity matches the ground-truth transmitter label
  (rank-1 / top-1 identification accuracy);
* **open-set** -> :class:`OpenSetMetric` (``auroc`` + ``eer``): treats fingerprinting as
  verification -- each probe carries a real-valued *match score* and a binary label
  (``1`` genuine / in-gallery, ``0`` impostor / novel). ``auroc`` is the probability a
  random genuine score outranks a random impostor score (computed from rank statistics);
  ``eer`` is the error rate at the operating point where the false-accept rate equals the
  false-reject rate.

All computation is stdlib-only so the metrics are exercisable on pure-Python synthetic
predictions/scores with no numpy: :meth:`update` accepts plain Python sequences/scalars
and :meth:`compute` returns JSON-serialisable floats. The metrics implement the frozen
:class:`rfbench.core.metric.Metric` contract (``reset`` / ``update`` / ``compute`` /
``name`` / ``primary_key``).
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from rfbench.core.metric import Metric
from rfbench.core.types import Tensor


class Rank1Accuracy(Metric):
    """Closed-set rank-1 (top-1) identification accuracy -- the SEI PRIMARY metric.

    Streams predicted transmitter ids against ground-truth transmitter labels and reports
    the fraction that match. ``pred`` may be either a batch of already-argmaxed integer
    ids (one per sample) or a batch of per-class score rows (a nested sequence), in which
    case the argmax is taken here (highest score wins, ties broken by lowest index). No
    numpy: iteration and comparison are pure Python.
    """

    name = "rank1_accuracy"
    primary_key = "rank1_accuracy"

    def __init__(self) -> None:
        """Start with an empty accumulator."""
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        """Clear the running correct/total counts."""
        self._correct = 0
        self._total = 0

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate one batch of predicted vs ground-truth transmitter ids.

        ``pred[i]`` is either the predicted id (a scalar) or a per-class score row; in the
        latter case the argmax over the row is used. ``target[i]`` is the true id.
        """
        for predicted, expected in zip(pred, target, strict=True):
            self._total += 1
            if _as_label(predicted) == expected:
                self._correct += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{"rank1_accuracy": correct / total}`` (0.0 on an empty stream)."""
        accuracy = self._correct / self._total if self._total else 0.0
        return {"rank1_accuracy": accuracy}


class BalancedAccuracy(Metric):
    """Closed-set balanced accuracy -- the SEI SECONDARY metric (WiSig parity).

    The unweighted mean of the per-class rank-1 recalls: for each ground-truth transmitter
    class present in the stream, the fraction of its probes classified correctly, averaged
    with equal weight across classes. This is the class-balanced counterpart of
    :class:`Rank1Accuracy` -- on an imbalanced closed set (WiSig ManyTx is built with
    ``p=0.9``) it down-weights over-represented emitters, matching the balanced accuracy the
    WiSig paper reports (``docs/EVALUATION_PROTOCOL.md`` SEI). Reported ALONGSIDE
    ``rank1_accuracy`` (never as the primary/ranking key). ``pred`` accepts either argmaxed
    ids or per-class score rows (argmaxed here); pure stdlib, no numpy.
    """

    name = "balanced_accuracy"
    primary_key = "balanced_accuracy"

    def __init__(self) -> None:
        """Start with empty per-class correct/total accumulators."""
        self._correct: dict[object, int] = {}
        self._total: dict[object, int] = {}

    def reset(self) -> None:
        """Clear the per-class correct/total counts."""
        self._correct = {}
        self._total = {}

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate one batch of predicted vs ground-truth transmitter ids, keyed by class."""
        for predicted, expected in zip(pred, target, strict=True):
            self._total[expected] = self._total.get(expected, 0) + 1
            if _as_label(predicted) == expected:
                self._correct[expected] = self._correct.get(expected, 0) + 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{"balanced_accuracy": mean_c(correct_c / total_c)}`` (0.0 on empty)."""
        if not self._total:
            return {"balanced_accuracy": 0.0}
        per_class = [
            self._correct.get(cls, 0) / total for cls, total in self._total.items() if total
        ]
        balanced = sum(per_class) / len(per_class) if per_class else 0.0
        return {"balanced_accuracy": balanced}


class OpenSetMetric(Metric):
    """Open-set verification metric: AUROC + EER over match scores (never conflated).

    Accumulates ``(score, label)`` pairs where ``label == 1`` marks a genuine/in-gallery
    probe and ``label == 0`` an impostor/novel one, then reports:

    * ``auroc`` -- via the Mann-Whitney rank statistic: the probability that a random
      genuine score exceeds a random impostor score, with ties counted at half weight
      (so identical genuine/impostor distributions give ~0.5 and a perfect separation
      gives 1.0);
    * ``eer`` -- the equal-error rate, i.e. the false-accept rate at the decreasing
      threshold where it first meets the false-reject rate (FAR == FRR crossing on the
      ROC), found by sweeping every distinct score as a threshold.

    ``primary_key`` is ``auroc`` (the open-set ranking metric per the protocol). Both
    scalars are always emitted together. Pure stdlib -- exercisable on synthetic score
    lists with no numpy.
    """

    name = "open_set"
    primary_key = "auroc"

    def __init__(self) -> None:
        """Start with empty positive/negative score buffers."""
        self._pos: list[float] = []
        self._neg: list[float] = []

    def reset(self) -> None:
        """Clear the accumulated genuine/impostor score buffers."""
        self._pos = []
        self._neg = []

    def prepare_predictions(self, pred: Tensor) -> list[float]:
        """Reduce a whole batch of per-class rows to scalar MSP scores, ONCE, for bootstrapping.

        Optional hook honoured by :func:`rfbench.core.evaluate._bootstrap_uncertainty`: the
        percentile bootstrap re-runs ``update`` ~1000x, so if ``pred`` stayed a list of
        120-class rows every resample would recompute the softmax over every probe
        (O(resamples x n x classes) -> the open-set eval stalls for ~45 min on 144k probes).
        Reducing to scalars here makes each resample a cheap ``float`` pass. Pure/deterministic,
        so the CI is unchanged; scalars pass through untouched (idempotent).
        """
        return [match_score(row) for row in pred]

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate a batch of match scores (``pred``) with binary labels (``target``).

        ``pred[i]`` is the model's per-class score row (as returned by ``forward``), reduced
        to a scalar **maximum-softmax-probability** (MSP) confidence via :func:`match_score`
        -- the standard open-set score (higher = more likely genuine). A ``pred[i]`` that is
        already a scalar match score is used verbatim (so synthetic score fixtures and the
        pre-reduced :meth:`prepare_predictions` output keep working). ``target[i]`` is ``1``
        for a genuine/in-gallery probe or ``0`` for an impostor/novel one. Labels outside
        ``{0, 1}`` raise :class:`ValueError`.
        """
        for prediction, label in zip(pred, target, strict=True):
            value = match_score(prediction)
            if label == 1:
                self._pos.append(value)
            elif label == 0:
                self._neg.append(value)
            else:
                raise ValueError(
                    f"open-set target must be 0 (impostor) or 1 (genuine), got {label!r}"
                )

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{"auroc": ..., "eer": ...}`` over the accumulated scores."""
        return {
            "auroc": auroc(self._pos, self._neg),
            "eer": eer(self._pos, self._neg),
        }


# --- pure-stdlib scoring primitives (no numpy) --------------------------------------


def _as_label(prediction: object) -> object:
    """Return a scalar predicted id, argmaxing a per-class score row if needed.

    A non-string sequence is treated as a per-class score row and reduced to its argmax
    (highest score, ties broken by the lowest index); any other value is returned as the
    predicted id verbatim.
    """
    if isinstance(prediction, (str, bytes)):
        return prediction
    try:
        row = list(prediction)  # type: ignore[call-overload]
    except TypeError:
        return prediction
    if not row:
        raise ValueError("cannot argmax an empty score row")
    best_index = 0
    best_score = row[0]
    for index in range(1, len(row)):
        if row[index] > best_score:
            best_score = row[index]
            best_index = index
    return best_index


def match_score(prediction: object) -> float:
    """Reduce one prediction to a scalar open-set **match score** (higher = more genuine).

    A per-class score row (a non-string sequence, i.e. the model's ``forward`` logits/scores
    for the sample) is reduced to its **maximum softmax probability** (MSP):
    ``max_i softmax(row)_i = 1 / Σ_j exp(row_j - max(row))``. MSP is the standard confidence
    an open-set verifier thresholds — an in-gallery (genuine) input peaks high on one class,
    a novel/impostor input spreads its mass and peaks lower. A ``0``-d tensor-like exposing
    ``.item()`` or a plain scalar is treated as an already-computed match score and returned
    verbatim (so synthetic score fixtures still work). Pure stdlib (``math.exp``), no numpy.
    """
    if isinstance(prediction, bool):  # avoid bool being read as a 0/1 score row
        return float(prediction)
    if isinstance(prediction, (int, float)):
        return float(prediction)
    # Iterate FIRST: a per-class score row (list / 1-D tensor) yields the class scores. Only a
    # non-iterable prediction (a 0-d tensor / numpy scalar) falls through to ``.item()`` -- calling
    # ``.item()`` eagerly would raise on a multi-element 1-D tensor ("a Tensor with N elements
    # cannot be converted to Scalar"), which is exactly a per-class row.
    try:
        row = [float(x) for x in prediction]  # type: ignore[attr-defined]
    except TypeError:
        item = getattr(prediction, "item", None)
        if callable(item):  # 0-d tensor / numpy scalar -> its Python scalar
            return float(item())
        return float(prediction)  # type: ignore[arg-type]
    if not row:
        raise ValueError("cannot score an empty prediction row")
    if len(row) == 1:
        return row[0]  # a length-1 "row" is a scalar match score, not a class distribution
    peak = max(row)
    denom = sum(math.exp(value - peak) for value in row)
    return 1.0 / denom  # == max(softmax(row)); denom >= 1 so the score is in (0, 1]


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U rank statistic (stdlib only).

    Equals the probability that a random ``positive`` score is ranked above a random
    ``negative`` score, counting ties at half weight. Perfectly separable scores give
    ``1.0``; identical positive/negative distributions give ``0.5``. Returns ``0.5`` when
    either class is empty (an undefined ROC degrades gracefully to chance).
    """
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank all scores (average ranks for ties), then sum the positive ranks (U statistic).
    combined = sorted(
        [(score, 1) for score in positives] + [(score, 0) for score in negatives],
        key=lambda pair: pair[0],
    )
    rank_sum_pos = 0.0
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        # Ranks are 1-based; the average rank of the tied block [i, j) is their mean.
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if combined[k][1] == 1:
                rank_sum_pos += average_rank
        i = j

    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


def eer(positives: list[float], negatives: list[float]) -> float:
    """Equal-error rate: the FAR at the threshold where FAR first meets FRR (stdlib only).

    Sweeps every distinct score as a decreasing decision threshold ``t`` (accept iff
    ``score >= t``), tracking the false-accept rate ``FAR = P(neg >= t)`` and false-reject
    rate ``FRR = P(pos < t)``. Returns the false-accept rate at the operating point where
    ``FAR`` first drops to or below ``FRR`` (the ROC's FAR==FRR crossing), interpolating
    the two bracketing thresholds. Perfectly separable scores give ``0.0``; fully
    overlapping ones give ~``0.5``. Returns ``0.0`` when either class is empty.
    """
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_sorted = sorted(positives)
    neg_sorted = sorted(negatives)
    thresholds = sorted({*positives, *negatives}, reverse=True)
    # Seed the operating point above the highest score (accept nothing: FAR=0, FRR=1); the
    # sweep then lowers the threshold until FAR overtakes FRR and interpolates the crossing.
    # FAR/FRR are read via binary search on the sorted score arrays (O(log n) per threshold), so
    # the whole sweep is O(n log n) -- essential for bootstrap CIs over large open-set test sets
    # (a per-threshold linear scan is O(n^2) and hangs on 100k+ probes).
    prev_far = 0.0
    prev_frr = 1.0
    for threshold in thresholds:
        far = (n_neg - bisect.bisect_left(neg_sorted, threshold)) / n_neg  # P(neg >= t)
        frr = bisect.bisect_left(pos_sorted, threshold) / n_pos  # P(pos < t)
        if far >= frr:
            # Between the previous point (far < frr) and this one (far >= frr), interpolate
            # the FAR at the FAR==FRR crossing for a smooth, threshold-grid-independent EER.
            return _interpolate_eer(prev_far, prev_frr, far, frr)
        prev_far, prev_frr = far, frr
    # FAR never reached FRR (e.g. degenerate scores): fall back to the last FAR/FRR mean.
    return (prev_far + prev_frr) / 2.0


def _interpolate_eer(far0: float, frr0: float, far1: float, frr1: float) -> float:
    """Linearly interpolate the FAR at the ``FAR == FRR`` crossing between two ROC points.

    Point 0 satisfies ``far0 < frr0`` and point 1 satisfies ``far1 >= frr1``; the two
    difference curves ``d = FAR - FRR`` bracket a zero, so the crossing FAR is a linear
    blend of the endpoints. Falls back to the nearer endpoint's FAR when the segment is
    degenerate (both differences equal).
    """
    diff0 = far0 - frr0
    diff1 = far1 - frr1
    if diff1 == diff0:
        return far1
    fraction = diff0 / (diff0 - diff1)
    return far0 + fraction * (far1 - far0)


__all__ = ["Rank1Accuracy", "BalancedAccuracy", "OpenSetMetric", "auroc", "eer"]
