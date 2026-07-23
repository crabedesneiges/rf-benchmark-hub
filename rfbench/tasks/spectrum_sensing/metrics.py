"""Spectrum-sensing metrics -- pure-stdlib streaming F1 (primary) + Pd@Pfa/AUROC.

Spectrum sensing is occupancy detection per ``docs/EVALUATION_PROTOCOL.md`` §"Spectrum sensing".
DeepSense is MULTI-LABEL: each ``(2, 32)`` raw-IQ window carries a length-16 per-subband
occupancy vector, and the model emits ``P(occupied)`` per sub-band. Each ``(window, subband)``
pair is one binary CELL, and every metric MICRO-averages over cells (via
:func:`iter_occupancy_cells`), so a window contributes 16 binary decisions. The same metrics also
accept a scalar ``0/1`` target (one cell per window) so the binary code path stays exercisable on
pure-Python fixtures. The reported metrics are:

* ``f1`` -- **primary** (:class:`OccupancyClassification`), the occupied-class F1 the sensing
  literature actually reports (DeepSense precision 98% / recall 97%; IPFSCNN calls F1 "the primary
  metric for overall model accuracy"), so published baselines are board-comparable. The same metric
  object also emits ``accuracy`` / ``precision`` / ``recall``.
* ``pd@pfa=0.1`` -- **secondary** (:class:`PdAtPfa`), the classical probability of detection at a
  fixed false-alarm rate of ``0.1`` (the ROC operating point). A detector that never misses an
  occupied window while raising few false alarms scores near ``1.0``; a random detector scores near
  the false-alarm target itself (``~0.1``).
* ``auroc`` -- secondary, area under the ROC (reused from :mod:`rfbench.tasks.sei.metrics` to avoid
  duplication); and ``roc`` -- the ROC curve as ``{"x": pfa, "y": pd}`` points, landing in
  ``result.json.metrics.curves``.

PROTOCOL / THRESHOLD CALIBRATION: the normative protocol calibrates the decision threshold on
the **val** split (to hit ``pfa == 0.1`` there) then FREEZES it for **test**, reporting the test
Pd + the test Pfa achieved at that frozen threshold. This metric supports that path via the
optional ``threshold=`` argument: when set, :meth:`compute` reports Pd + achieved Pfa at the
frozen threshold; when ``None`` (the self-contained default) it reads ``pd@pfa`` off the
accumulated stream's own ROC. Full val-calibration wiring inside
:func:`rfbench.core.evaluate.evaluate` (threshold fit on val, carried into the test pass) is a
documented follow-up.

All computation is stdlib-only so the metric is exercisable on pure-Python synthetic
scores/targets with no numpy: :meth:`update` accepts plain Python sequences/scalars,
:func:`pd_at_pfa` uses :mod:`bisect` on sorted arrays for ``O(n log n)`` sweeps (needed for
bootstrap CIs over large test sets), and :meth:`compute` returns JSON-serialisable floats. The
metric implements the frozen :class:`rfbench.core.metric.Metric` contract.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from rfbench.core.metric import Metric
from rfbench.core.types import Tensor
from rfbench.tasks.sei.metrics import auroc

#: Default false-alarm rate the primary metric operates at (``pd@pfa=0.1``).
DEFAULT_PFA_TARGET = 0.1

#: Cap on the number of ROC-curve points emitted so a huge stream does not bloat the curve.
_MAX_ROC_POINTS = 256


class PdAtPfa(Metric):
    """Probability of detection at a fixed false-alarm rate -- the classical ROC SECONDARY metric.

    Accumulates ``(score, target)`` pairs where ``score`` is the model's ``P(occupied)`` for a
    window and ``target`` is ``1`` (occupied) or ``0`` (vacant), then reports the probability of
    detection at ``pfa == pfa_target`` (default ``0.1``), the AUROC, the achieved Pfa, and the
    ROC curve.

    ``primary_key`` is ``"pd@pfa=0.1"`` (formatted from ``pfa_target``). Two operating-point
    modes:

    * ``threshold is None`` (default, self-contained): :meth:`compute` reads ``pd@pfa`` directly
      off the accumulated stream's own ROC via :func:`pd_at_pfa`, and ``pfa_achieved`` equals
      ``pfa_target``.
    * ``threshold`` set (the val-calibrated protocol path): :meth:`compute` detects iff
      ``score >= threshold`` and reports the resulting Pd (recall over occupied windows) and the
      Pfa actually achieved at that frozen threshold. The normative protocol fits ``threshold``
      on val so its val-Pfa is ``pfa_target``, freezes it, and scores test with it; wiring that
      fit into ``evaluate`` is a documented follow-up.

    Pure stdlib -- exercisable on synthetic score/target lists with no numpy.
    """

    def __init__(
        self,
        pfa_target: float = DEFAULT_PFA_TARGET,
        threshold: float | None = None,
    ) -> None:
        """Bind the target false-alarm rate and (optional) frozen decision threshold."""
        self.pfa_target = pfa_target
        self.threshold = threshold
        self.name = f"pd@pfa={pfa_target:g}"
        self.primary_key = self.name
        self._pos: list[float] = []
        self._neg: list[float] = []

    def reset(self) -> None:
        """Clear the accumulated occupied/vacant score buffers."""
        self._pos = []
        self._neg = []

    def prepare_predictions(self, pred: Tensor) -> list[object]:
        """Reduce a whole batch of per-sample model outputs to per-window occupied-scores, ONCE.

        Optional hook honoured by :func:`rfbench.core.evaluate._bootstrap_uncertainty` (mirrors
        :meth:`rfbench.tasks.sei.metrics.OpenSetMetric.prepare_predictions`): the percentile
        bootstrap re-runs :meth:`update` ~1000x, so reducing each raw per-sample output here makes
        every resample a cheap pass. A BINARY output collapses to a scalar ``P(occupied)`` via
        :func:`occupancy_score`; a MULTI-LABEL 16-band row is kept as its length-16 per-subband
        prob list (:func:`reduce_prediction`). Pure/deterministic, so the CI is unchanged;
        already-reduced predictions pass through untouched (idempotent).
        """
        return [reduce_prediction(row) for row in pred]

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate occupied-scores (``pred``) with binary / 16-band targets (``target``).

        Each ``(prediction, target)`` pair is expanded to one or more binary CELLS by
        :func:`iter_occupancy_cells`: a scalar target is one binary window (``P(occupied)`` via
        :func:`occupancy_score`); a length-16 target is 16 window×subband cells (the matching
        per-subband probabilities). Positives accumulate into the occupied buffer, negatives into
        the vacant one, so ``pd@pfa`` is micro-averaged over cells. Targets outside ``{0, 1}``
        raise :class:`ValueError`.
        """
        for value, label in iter_occupancy_cells(pred, target):
            if label == 1:
                self._pos.append(value)
            else:  # iter_occupancy_cells has already validated label in {0, 1}
                self._neg.append(value)

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{pd@pfa, auroc, pfa_achieved, roc}`` over the accumulated scores.

        When ``threshold is None`` the operating point is read off the stream's own ROC at
        ``pfa_target`` (``pfa_achieved == pfa_target``); when a frozen ``threshold`` is set, Pd
        and the achieved Pfa are computed at that fixed threshold. ``roc`` is the ROC curve,
        down-sampled to at most :data:`_MAX_ROC_POINTS` points when the stream is large.
        """
        if self.threshold is None:
            pd_value = pd_at_pfa(self._pos, self._neg, pfa_target=self.pfa_target)
            pfa_achieved = self.pfa_target
        else:
            pd_value, pfa_achieved = _pd_pfa_at_threshold(self._pos, self._neg, self.threshold)
        return {
            self.primary_key: pd_value,
            "auroc": auroc(self._pos, self._neg),
            "pfa_achieved": pfa_achieved,
            "roc": _roc_curve(self._pos, self._neg),
        }


#: Decision threshold on ``P(occupied)`` for the hard-label metrics (occupied iff score >= this).
_OCCUPANCY_THRESHOLD = 0.5


class OccupancyClassification(Metric):
    """Binary occupancy classification metrics -- ``f1`` PRIMARY (+ accuracy/precision/recall).

    The spectrum-sensing literature (DeepSense: precision 98% / recall 97%; IPFSCNN and successors)
    reports occupancy performance with **F1 / precision / recall** (IPFSCNN calls F1 "the primary
    metric for overall model accuracy") -- never plain accuracy -- so ``f1`` over the occupied class
    is the board's primary (the metric published baselines actually report, so they are board-
    comparable), while ``accuracy`` / ``precision`` / ``recall`` ride along and :class:`PdAtPfa`
    keeps the classical ROC point as secondaries. Accumulates ``(score, target)`` pairs where
    ``score`` is ``P(occupied)`` (via :func:`occupancy_score`), thresholds at
    :data:`_OCCUPANCY_THRESHOLD` (occupied iff ``score >= 0.5``). Pure stdlib; ``primary_key`` is
    ``"f1"``.
    """

    name = "f1"
    primary_key = "f1"

    def __init__(self, threshold: float = _OCCUPANCY_THRESHOLD) -> None:
        """Bind the hard-label decision threshold on ``P(occupied)`` and reset counters."""
        self.threshold = threshold
        self._correct = 0
        self._n = 0
        self._tp = 0  # predicted occupied & truly occupied
        self._fp = 0  # predicted occupied & truly vacant
        self._fn = 0  # predicted vacant & truly occupied

    def reset(self) -> None:
        """Clear the accumulated confusion counts."""
        self._correct = self._n = self._tp = self._fp = self._fn = 0

    def prepare_predictions(self, pred: Tensor) -> list[object]:
        """Reduce a batch of per-sample outputs to per-window occupied-scores once (bootstrap hook).

        Binary rows collapse to a scalar ``P(occupied)``; multi-label 16-band rows keep their
        length-16 per-subband prob list (see :func:`reduce_prediction`).
        """
        return [reduce_prediction(row) for row in pred]

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate hard-label confusion counts for one batch, per binary/16-band CELL.

        :func:`iter_occupancy_cells` expands each ``(prediction, target)`` into binary cells (one
        per window for a scalar target; 16 window×subband cells for a length-16 target), so ``f1`` /
        precision / recall are MICRO-averaged over cells. Targets outside ``{0, 1}`` raise.
        """
        for value, label in iter_occupancy_cells(pred, target):
            predicted = 1 if value >= self.threshold else 0
            self._n += 1
            if predicted == label:
                self._correct += 1
            if predicted == 1 and label == 1:
                self._tp += 1
            elif predicted == 1 and label == 0:
                self._fp += 1
            elif predicted == 0 and label == 1:
                self._fn += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{f1 (primary), accuracy, precision, recall}`` over the accumulated stream."""
        accuracy = self._correct / self._n if self._n else 0.0
        precision = self._tp / (self._tp + self._fp) if (self._tp + self._fp) else 0.0
        recall = self._tp / (self._tp + self._fn) if (self._tp + self._fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# --- pure-stdlib scoring primitives (no numpy) --------------------------------------


def occupancy_score(prediction: object) -> float:
    """Reduce one prediction to a scalar occupied-probability ``P(occupied)`` (higher = occupied).

    Mirrors :func:`rfbench.tasks.sei.metrics.match_score`'s robust duck-typing but for the
    *class-1* (occupied) probability:

    * a **length-2** per-class score row (the model's ``[vacant, occupied]`` logits/scores) is
      reduced to ``softmax(row)[1] == 1 / (1 + exp(row[0] - row[1]))`` -- the occupied posterior;
    * a **length-1** row or a plain scalar (``int``/``float``) is treated as an already-computed
      ``P(occupied)`` and returned verbatim (so synthetic score fixtures and a sigmoid head both
      work);
    * a ``0``-d tensor-like exposing ``.item()`` is unwrapped to its Python scalar.

    Longer rows raise :class:`ValueError` -- occupancy is binary, so a length-``k>2`` output is a
    caller bug, not something to silently reduce. Pure stdlib (``math.exp``), no numpy.
    """
    if isinstance(prediction, bool):  # bool is an int subclass; use its 0/1 value.
        return float(prediction)
    if isinstance(prediction, (int, float)):
        return float(prediction)
    # Iterate FIRST: a per-class row (list / 1-D tensor) yields the class scores. Only a
    # non-iterable prediction (a 0-d tensor / numpy scalar) falls through to ``.item()`` --
    # calling ``.item()`` eagerly would raise on a multi-element 1-D tensor.
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
        return row[0]  # a length-1 "row" is an already-computed P(occupied), not a distribution.
    if len(row) != 2:
        raise ValueError(
            "spectrum-sensing occupancy expects a length-1 or length-2 output, "
            f"got length {len(row)}"
        )
    # softmax(row)[1] == 1 / (1 + exp(row[0] - row[1])); numerically stable via the sigmoid form.
    return _sigmoid(row[1] - row[0])


def _is_multilabel_target(label: object) -> bool:
    """Return ``True`` when ``label`` is a per-subband occupancy VECTOR (not a scalar 0/1).

    A binary window carries a scalar target (``int`` / ``float`` / ``bool``); a multi-label
    DeepSense window carries a length-K sequence of ``0/1`` bits (one per LTE-M sub-band). Strings
    are never valid occupancy targets, so they are treated as scalar-ish and fail downstream.
    """
    if isinstance(label, (bool, int, float, str)):
        return False
    try:
        iter(label)  # type: ignore[call-overload]
    except TypeError:
        return False
    return True


def _check_bit(label: object) -> int:
    """Coerce one occupancy target to a strict ``{0, 1}`` cell label; raise otherwise."""
    if label == 1:
        return 1
    if label == 0:
        return 0
    raise ValueError(f"spectrum-sensing target must be 0 (vacant) or 1 (occupied), got {label!r}")


def _as_prob_row(prediction: object) -> list[float]:
    """Coerce a multi-label per-subband prediction row to a plain ``list[float]``.

    ``prediction`` is the model's per-window output over the K sub-bands -- a Python list, a 1-D
    tensor (iterating yields 0-d tensors that ``float`` unwraps), or an already-reduced prob list.
    Each value is an already-computed per-subband ``P(occupied)`` (the DeepSense head is a
    per-band sigmoid), taken verbatim -- mirroring how :func:`occupancy_score` passes a scalar /
    length-1 row through.
    """
    try:
        return [float(x) for x in prediction]  # type: ignore[attr-defined]
    except TypeError as exc:
        raise ValueError(
            "multi-label spectrum-sensing prediction must be a per-subband row"
        ) from exc


def iter_occupancy_cells(pred: Tensor, target: Tensor) -> list[tuple[float, int]]:
    """Expand a batch of ``(prediction, target)`` pairs into binary ``(score, label)`` CELLS.

    The single place the binary and multi-label sensing paths converge. For each per-sample pair
    (strictly zipped, so a length mismatch fails loudly):

    * a **scalar** target -> ONE cell: ``(occupancy_score(prediction), bit)`` -- the binary
      occupancy window (a length-2 row softmaxes to ``P(occupied)``; a scalar / length-1 passes
      through);
    * a **length-K** target (DeepSense's 16 LTE-M sub-bands) -> K cells: the per-subband
      probability paired with its per-subband bit, so the metrics micro-average over
      window×subband cells. The prediction row length MUST match the target length.

    Every cell label is validated to ``{0, 1}`` via :func:`_check_bit`. Returns a materialised list
    (not a generator) so a metric's ``update`` can iterate it directly.
    """
    cells: list[tuple[float, int]] = []
    for prediction, label in zip(pred, target, strict=True):
        if _is_multilabel_target(label):
            row = _as_prob_row(prediction)
            bits = list(label)
            if len(row) != len(bits):
                raise ValueError(
                    "multi-label spectrum-sensing prediction/target length mismatch: "
                    f"{len(row)} sub-bands predicted vs {len(bits)} targeted"
                )
            for value, bit in zip(row, bits, strict=True):
                cells.append((value, _check_bit(bit)))
        else:
            cells.append((occupancy_score(prediction), _check_bit(label)))
    return cells


def reduce_prediction(row: object) -> object:
    """Reduce ONE per-sample model output for the bootstrap ``prepare_predictions`` hook.

    A binary output (scalar / length-1 / length-2 row) collapses to a scalar ``P(occupied)`` via
    :func:`occupancy_score`; a MULTI-LABEL per-subband row (length > 2) is kept as its
    ``list[float]`` of per-subband probabilities so :func:`iter_occupancy_cells` can still expand
    it into cells after a resample gather. Idempotent on already-reduced values.
    """
    if not isinstance(row, (bool, int, float)):
        try:
            length = len(row)  # type: ignore[arg-type]
        except TypeError:
            length = None
        if length is not None and length > 2:
            return _as_prob_row(row)
    return occupancy_score(row)


def _sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-value))`` (stdlib ``math.exp``)."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)  # value < 0: avoids overflow in exp(-value).
    return exp_v / (1.0 + exp_v)


def pd_at_pfa(
    positives: list[float],
    negatives: list[float],
    pfa_target: float = DEFAULT_PFA_TARGET,
) -> float:
    """Probability of detection at a fixed false-alarm rate (stdlib only, ``O(n log n)``).

    Sweeps every distinct score as a DECREASING decision threshold ``t`` (detect iff
    ``score >= t``): as ``t`` falls from above the max score toward the min, both the false-alarm
    rate ``FAR = P(neg >= t)`` and the detection rate ``PD = P(pos >= t)`` rise monotonically from
    ``0`` to ``1``. Finds the operating point where ``FAR`` first reaches ``pfa_target`` and
    LINEARLY INTERPOLATES the ``PD`` between the two bracketing thresholds -- exactly the
    threshold-grid-independent interpolation :func:`rfbench.tasks.sei.metrics.eer` /
    ``_interpolate_eer`` use, so the reported Pd does not jump with the score granularity.

    FAR/PD at each threshold are read via :func:`bisect.bisect_left` on the sorted score arrays
    (``O(log n)`` per threshold), making the whole sweep ``O(n log n)`` -- essential for the
    percentile bootstrap CIs over large sensing test sets (a per-threshold linear scan is
    ``O(n^2)`` and hangs on 100k+ windows).

    Edge cases: returns ``0.0`` when either class is empty (an undefined ROC degrades to no
    detection); if ``FAR`` never reaches ``pfa_target`` (e.g. degenerate/tied scores), returns
    the maximum ``PD`` achieved over the sweep.
    """
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_sorted = sorted(positives)
    neg_sorted = sorted(negatives)
    thresholds = sorted({*positives, *negatives}, reverse=True)

    # Seed the operating point above the highest score (detect nothing: FAR=0, PD=0); the sweep
    # then lowers the threshold until FAR first reaches pfa_target and interpolates the PD.
    prev_far = 0.0
    prev_pd = 0.0
    max_pd = 0.0
    for threshold in thresholds:
        far = (n_neg - bisect.bisect_left(neg_sorted, threshold)) / n_neg  # P(neg >= t)
        pd = (n_pos - bisect.bisect_left(pos_sorted, threshold)) / n_pos  # P(pos >= t)
        if pd > max_pd:
            max_pd = pd
        if far >= pfa_target:
            # Between the previous point (far < pfa_target) and this one (far >= pfa_target),
            # linearly interpolate the PD at the exact pfa_target crossing.
            return _interpolate_pd(prev_far, prev_pd, far, pd, pfa_target)
        prev_far, prev_pd = far, pd
    # FAR never reached pfa_target (degenerate scores): report the best PD attained.
    return max_pd


def _interpolate_pd(
    far0: float,
    pd0: float,
    far1: float,
    pd1: float,
    pfa_target: float,
) -> float:
    """Linearly interpolate PD at ``FAR == pfa_target`` between two ROC points.

    Point 0 satisfies ``far0 < pfa_target`` and point 1 satisfies ``far1 >= pfa_target``, so the
    target FAR lies in ``[far0, far1]`` and the PD is the corresponding linear blend of the
    endpoints. Falls back to ``pd1`` when the FAR segment is degenerate (``far1 == far0``, e.g.
    a tied-score block where both endpoints already sit at or above the target).
    """
    if far1 == far0:
        return pd1
    fraction = (pfa_target - far0) / (far1 - far0)
    return pd0 + fraction * (pd1 - pd0)


def _pd_pfa_at_threshold(
    positives: list[float],
    negatives: list[float],
    threshold: float,
) -> tuple[float, float]:
    """Return ``(pd, pfa)`` at a frozen decision ``threshold`` (detect iff ``score >= threshold``).

    ``pd`` is the recall over occupied windows (``P(pos >= threshold)``) and ``pfa`` the
    false-alarm rate over vacant windows (``P(neg >= threshold)``). Returns ``(0.0, 0.0)`` when
    the corresponding class is empty. Used by the val-calibrated protocol path where the
    threshold was fit on val and frozen for test.
    """
    pos_sorted = sorted(positives)
    neg_sorted = sorted(negatives)
    n_pos = len(pos_sorted)
    n_neg = len(neg_sorted)
    pd = (n_pos - bisect.bisect_left(pos_sorted, threshold)) / n_pos if n_pos else 0.0
    pfa = (n_neg - bisect.bisect_left(neg_sorted, threshold)) / n_neg if n_neg else 0.0
    return pd, pfa


def _roc_curve(positives: list[float], negatives: list[float]) -> list[dict[str, float]]:
    """Build the ROC curve as ``[{"x": pfa, "y": pd}, ...]`` (stdlib; down-sampled if large).

    Sweeps every distinct score as a decreasing threshold, emitting the ``(pfa, pd)`` point at
    each; the curve runs from ``(0, 0)`` up to ``(1, 1)``. Returns an empty list when either
    class is empty (an undefined ROC). When the number of distinct thresholds exceeds
    :data:`_MAX_ROC_POINTS`, the thresholds are evenly down-sampled (endpoints kept) so the curve
    stays modest in ``result.json.metrics.curves`` without distorting its shape.
    """
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return []

    pos_sorted = sorted(positives)
    neg_sorted = sorted(negatives)
    thresholds = sorted({*positives, *negatives}, reverse=True)
    thresholds = _downsample(thresholds, _MAX_ROC_POINTS)

    curve: list[dict[str, float]] = [{"x": 0.0, "y": 0.0}]
    for threshold in thresholds:
        pfa = (n_neg - bisect.bisect_left(neg_sorted, threshold)) / n_neg
        pd = (n_pos - bisect.bisect_left(pos_sorted, threshold)) / n_pos
        curve.append({"x": pfa, "y": pd})
    if curve[-1] != {"x": 1.0, "y": 1.0}:
        curve.append({"x": 1.0, "y": 1.0})
    return curve


def _downsample(values: list[float], max_points: int) -> list[float]:
    """Evenly down-sample ``values`` to at most ``max_points`` items, keeping both endpoints."""
    n = len(values)
    if n <= max_points:
        return values
    step = (n - 1) / (max_points - 1)
    picked = [values[round(i * step)] for i in range(max_points)]
    # ``round`` can repeat an index at the seams; dedup while preserving the descending order.
    deduped: list[float] = []
    for value in picked:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


__all__ = [
    "DEFAULT_PFA_TARGET",
    "OccupancyClassification",
    "PdAtPfa",
    "occupancy_score",
    "iter_occupancy_cells",
    "reduce_prediction",
    "pd_at_pfa",
]
