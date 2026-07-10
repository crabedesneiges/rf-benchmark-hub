"""Pure-stdlib SNR-estimation regression metrics (J4).

SNR estimation is a scalar regression task: predict the per-window SNR (dB) of a raw-IQ
window. Per ``docs/EVALUATION_PROTOCOL.md`` "Statistical rigor & uncertainty" -> "Regression
metric (snr_estimation)" the reported metrics are:

* ``rmse_db`` -- **primary**, root-mean-square error in dB. Chosen as primary because it is
  the standard for this class of benchmark and is more sensitive to outliers -- relevant for
  catching estimation failures at low SNR.
* ``mae_db`` -- **secondary**, mean absolute error in dB.

Both are ``lower_is_better`` (0 dB is a perfect estimate), unlike every classification metric
on the board. The board rendering handles that direction generically off the metric name.

Every ``compute()`` here is exercisable on pure-Python synthetic predictions/targets: no
numpy/torch import anywhere in this module. A per-sample prediction may be a plain ``float``
(the pure-Python test path), a length-1 sequence/vector, or a 0-d tensor (a real regression
head's output); :func:`_as_float` collapses all three to a Python ``float`` with a lazy,
dependency-free unwrap so the metric path runs with only ``pytest`` installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from rfbench.core.metric import Metric
from rfbench.core.types import Tensor

#: Full SNR range recorded in ``result.json.eval.conditions`` (RadioML 2016.10a convention:
#: -20..+18 dB in 2 dB steps). Guards comparability -- SNR estimation is scored over the WHOLE
#: range (no cherry-picking of high-SNR points), mirroring the AMC full-range rule.
DEFAULT_SNR_DB_MIN = -20
DEFAULT_SNR_DB_MAX = 18


def _as_float(value: Tensor) -> float:
    """Collapse one per-sample regression prediction (or target) to a Python ``float``.

    Accepts a plain scalar (``int``/``float``), a length-1 ``list``/``tuple`` (a head that
    emits a ``[y]`` per sample), or a 0-d/1-element tensor/array (duck-typed via ``item()``
    or ``float()``). Longer sequences raise -- SNR estimation is a single-scalar regression,
    so a multi-element prediction is a caller bug, not something to silently reduce. Pure
    stdlib -- no numpy -- so the metric runs on synthetic Python values in tests.
    """
    if isinstance(value, bool):
        # bool is an int subclass; a bool SNR is almost certainly a caller error, but treat
        # it as its numeric value rather than crashing.
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    # torch tensor / numpy scalar: prefer ``.item()`` (0-d or 1-element), no import needed.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            pass  # fall through to the sequence path (e.g. a multi-element tensor)
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) != 1:
            raise ValueError(
                f"SNR regression expects one scalar per sample, got a length-{len(seq)} vector"
            )
        return _as_float(seq[0])
    return float(value)  # last-resort numeric coercion (numpy scalar, Decimal, ...)


def _iter_pairs(pred: Tensor, target: Tensor) -> Iterable[tuple[float, float]]:
    """Yield ``(predicted_snr, true_snr)`` float pairs for one batch.

    ``pred`` and ``target`` are batch-of-samples sequences; both sides are decoded via
    :func:`_as_float` so scalar, length-1-vector and 0-d-tensor layouts are all accepted.
    """
    for predicted, expected in zip(pred, target, strict=True):
        yield _as_float(predicted), _as_float(expected)


class Rmse(Metric):
    """Root-mean-square error in dB over the full SNR range (**primary**).

    Streams ``(pred, target)`` batches, accumulating the running sum of squared errors and
    the sample count; :meth:`compute` returns ``rmse_db`` (the primary key). :meth:`eval_conditions`
    records the full SNR range so ``result.json.eval.conditions`` attests no cherry-picking,
    mirroring the AMC full-range guarantee.

    Lower is better (0 dB == perfect). An empty stream computes to ``0.0`` (no error yet),
    matching the classification metrics' empty-input convention.
    """

    name = "rmse_db"
    primary_key = "rmse_db"

    def __init__(
        self,
        snr_db_min: int = DEFAULT_SNR_DB_MIN,
        snr_db_max: int = DEFAULT_SNR_DB_MAX,
    ) -> None:
        """Record the full SNR range guard-rail and initialise empty accumulators."""
        self._snr_db_min = snr_db_min
        self._snr_db_max = snr_db_max
        self._sse = 0.0
        self._n = 0

    def reset(self) -> None:
        """Clear the accumulated squared-error sum and count."""
        self._sse = 0.0
        self._n = 0

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate the sum of squared errors and the sample count for one batch."""
        for predicted, expected in _iter_pairs(pred, target):
            error = predicted - expected
            self._sse += error * error
            self._n += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{rmse_db}`` = sqrt(mean squared error) over the whole stream."""
        return {"rmse_db": math.sqrt(self._sse / self._n) if self._n else 0.0}

    def eval_conditions(self) -> dict[str, Any]:
        """Record the full SNR range so the row attests no cherry-picking (full-range rule)."""
        return {
            "snr_db_min": self._snr_db_min,
            "snr_db_max": self._snr_db_max,
            "full_snr_range": True,
        }


class Mae(Metric):
    """Mean absolute error in dB (secondary regression metric).

    Streams ``(pred, target)`` batches, accumulating the running sum of absolute errors and
    the sample count; :meth:`compute` returns ``mae_db``. Emitted on its own so the metric is
    unit-testable in isolation and reusable outside the SNR task. Lower is better; an empty
    stream computes to ``0.0``.
    """

    name = "mae_db"
    primary_key = "mae_db"

    def __init__(self) -> None:
        """Initialise the empty absolute-error accumulators."""
        self._sae = 0.0
        self._n = 0

    def reset(self) -> None:
        """Clear the accumulated absolute-error sum and count."""
        self._sae = 0.0
        self._n = 0

    def update(
        self,
        pred: Tensor,
        target: Tensor,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate the sum of absolute errors and the sample count for one batch."""
        for predicted, expected in _iter_pairs(pred, target):
            self._sae += abs(predicted - expected)
            self._n += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        """Return ``{mae_db}`` = mean absolute error over the whole stream."""
        return {"mae_db": self._sae / self._n if self._n else 0.0}


def rmse(pred: Sequence[Any], target: Sequence[Any]) -> float:
    """Convenience RMSE (dB) over two aligned sequences (used by tests).

    Decodes both sides via :func:`_as_float`; returns ``0.0`` for an empty input.
    """
    pairs = list(_iter_pairs(pred, target))
    if not pairs:
        return 0.0
    sse = sum((p - t) ** 2 for p, t in pairs)
    return math.sqrt(sse / len(pairs))


def mae(pred: Sequence[Any], target: Sequence[Any]) -> float:
    """Convenience MAE (dB) over two aligned sequences (used by tests).

    Decodes both sides via :func:`_as_float`; returns ``0.0`` for an empty input.
    """
    pairs = list(_iter_pairs(pred, target))
    if not pairs:
        return 0.0
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


__all__ = [
    "DEFAULT_SNR_DB_MIN",
    "DEFAULT_SNR_DB_MAX",
    "Rmse",
    "Mae",
    "rmse",
    "mae",
]
