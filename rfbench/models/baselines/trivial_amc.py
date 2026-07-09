"""Trivial floor baselines for AMC (J2) -- ``majority_class`` and ``chance``.

The two deliberately-minimal references that anchor the *bottom* of the AMC board, so every real
model (DSP or deep) is read against a meaningful floor:

* ``majority_class`` -- always predicts the single most frequent class of the train split
  (the "zero-rule" / ``sklearn.dummy.DummyClassifier(strategy="most_frequent")`` baseline). Its
  test accuracy is the majority-class prior; on a balanced set (RadioML) that is ~1/C, but on a
  skewed split it is the honest "predict the mode" floor.
* ``chance`` -- predicts a class drawn uniformly at random, with a seed-42 RNG so the row is
  reproducible. Its *expected* accuracy is exactly ``1/C`` (``1/11 ≈ 0.0909`` on RML2016.10a);
  the realised accuracy fluctuates around it by sampling noise, which is why the RNG is seeded.

Both are true :class:`~rfbench.core.model.Model` s in the ``"baseline"`` family and both go
through a *trivial* :meth:`fit` (out of the frozen ``Model`` contract, called by the dedicated CPU
script before :func:`rfbench.core.evaluate.evaluate`): ``majority_class`` learns the modal class
of the train split, ``chance`` learns only the class count ``C`` (so its uniform draw spans the
right label set). :meth:`forward` returns ``list[list[float]]`` one-hot / uniform score vectors
the AMC metrics' pure-stdlib ``argmax`` decodes; :meth:`embed` raises ``NotImplementedError`` (a
constant/uniform predictor has no representation to probe). :attr:`n_params` is ``0``.

HARD CONSTRAINT (mirrors the sibling baselines): ``import rfbench`` stays dependency-free. These
models are pure stdlib -- no numpy, no torch, no sklearn -- so nothing is imported beyond the core
contracts. The ``@register_model`` entries are created only on an explicit
``import rfbench.models.baselines.trivial_amc``.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from typing import Literal

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor

#: Registry name of the majority-class (zero-rule) floor.
MAJORITY_MODEL_NAME = "majority_class"
#: Registry name of the uniform-random floor.
CHANCE_MODEL_NAME = "chance"
#: RNG seed shared with the rest of the harness so the ``chance`` row is reproducible.
_RANDOM_STATE = 42


def _batch_size(x: Batch) -> int:
    """Return the number of samples in a collated batch (the length of any per-sample field).

    Reads ``x["iq"]`` (always present for AMC) and treats a single unbatched ``(2, L)`` window --
    a 2-level nested sequence whose first element is itself a sequence of scalars -- as a batch of
    one, mirroring the torch baselines' unbatched-sample handling.
    """
    iq = x["iq"]
    first = iq[0]
    # A single unbatched (2, L) window: iq is length-2 (I, Q) and iq[0] is a sequence of scalars.
    if len(iq) == 2 and hasattr(first, "__len__") and not hasattr(first[0], "__len__"):
        return 1
    return len(iq)


@register_model(MAJORITY_MODEL_NAME)
class MajorityClass(Model):
    """Always predict the most frequent train-split class (the zero-rule floor), registered
    ``"majority_class"``.

    A :class:`~rfbench.core.model.Model` in the ``"baseline"`` family. :meth:`fit` counts the train
    labels and stores the modal class id and the class count ``C``; :meth:`forward` returns, per
    sample, the one-hot score vector (``1.0`` for the majority class, ``0.0`` elsewhere) whose
    ``argmax`` is that class. Fully deterministic. :attr:`regime` is ``from_scratch`` (the "model"
    is fit from scratch on the train prior).
    """

    family: Literal["baseline"] = "baseline"

    def __init__(self, *, name: str = MAJORITY_MODEL_NAME) -> None:
        """Construct the unfitted floor (the modal class is learned in :meth:`fit`)."""
        if not name:
            raise ValueError("MajorityClass needs a non-empty name")
        self.name = name
        self.regime = RegimeSpec(Regime.FROM_SCRATCH)
        self._majority: int | None = None
        self._num_classes = 0

    def fit(self, samples: Iterable[Batch]) -> MajorityClass:
        """Learn the modal class of the train split (and the class count ``C``).

        ``samples`` is any iterable of per-sample dicts ``{"label": int, ...}`` (e.g.
        ``AmcDataset.load("train")``). Ties on frequency go to the lowest class id
        (``Counter.most_common`` is stable, and the loop below breaks ties deterministically).
        Returns ``self``.
        """
        counts: Counter[int] = Counter()
        for sample in samples:
            counts[int(sample["label"])] += 1
        if not counts:
            raise ValueError("MajorityClass.fit received an empty train split")
        # Deterministic argmax with lowest-id tie-break (Counter.most_common order is undefined
        # among equal counts, so pick explicitly).
        best_count = max(counts.values())
        self._majority = min(c for c, n in counts.items() if n == best_count)
        self._num_classes = max(counts) + 1
        return self

    def forward(self, x: Batch) -> Tensor:
        """Return, per sample, the one-hot score vector for the learned majority class."""
        if self._majority is None:
            raise RuntimeError("MajorityClass.forward called before fit")
        row = [0.0] * self._num_classes
        row[self._majority] = 1.0
        return [list(row) for _ in range(_batch_size(x))]

    def embed(self, x: Batch) -> Tensor:
        """A constant predictor has no representation -- raises, as the contract permits."""
        raise NotImplementedError("majority_class has no embedding to probe")

    @property
    def n_params(self) -> int:
        """No learnable parameters -- the floor stores only the modal class id."""
        return 0


@register_model(CHANCE_MODEL_NAME)
class UniformChance(Model):
    """Predict a uniformly-random class (seed 42), registered ``"chance"``.

    A :class:`~rfbench.core.model.Model` in the ``"baseline"`` family whose per-sample prediction is
    a class drawn uniformly at random from the ``C`` train classes with a seed-42 RNG, so the row is
    reproducible. Its expected accuracy is exactly ``1/C``. :meth:`forward` returns, per sample, a
    one-hot vector at the *drawn* class (so the metrics' ``argmax`` selects that class) -- a flat
    ``[1/C, …]`` vector would make every argmax pick class 0 and mis-report chance as majority-of-0.
    :attr:`regime` is ``from_scratch``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(self, *, name: str = CHANCE_MODEL_NAME, seed: int = _RANDOM_STATE) -> None:
        """Construct the unfitted floor; ``seed`` (default 42) fixes the draw so the row replays."""
        if not name:
            raise ValueError("UniformChance needs a non-empty name")
        self.name = name
        self.seed = seed
        self.regime = RegimeSpec(Regime.FROM_SCRATCH)
        self._num_classes = 0
        self._rng = random.Random(seed)

    def fit(self, samples: Iterable[Batch]) -> UniformChance:
        """Learn only the class count ``C`` from the train split, and re-seed the RNG.

        ``samples`` is any iterable of ``{"label": int, ...}`` dicts. Re-seeding here (rather than
        only in ``__init__``) makes a fit-then-forward run reproducible regardless of any draws a
        caller made on the model between construction and fit. Returns ``self``.
        """
        max_label = -1
        for sample in samples:
            label = int(sample["label"])
            if label > max_label:
                max_label = label
        if max_label < 0:
            raise ValueError("UniformChance.fit received an empty train split")
        self._num_classes = max_label + 1
        self._rng = random.Random(self.seed)
        return self

    def forward(self, x: Batch) -> Tensor:
        """Return, per sample, a one-hot vector at a uniformly-drawn class (seeded RNG)."""
        if self._num_classes < 1:
            raise RuntimeError("UniformChance.forward called before fit")
        rows: list[list[float]] = []
        for _ in range(_batch_size(x)):
            drawn = self._rng.randrange(self._num_classes)
            row = [0.0] * self._num_classes
            row[drawn] = 1.0
            rows.append(row)
        return rows

    def embed(self, x: Batch) -> Tensor:
        """A uniform predictor has no representation -- raises, as the contract permits."""
        raise NotImplementedError("chance has no embedding to probe")

    @property
    def n_params(self) -> int:
        """No learnable parameters -- the floor stores only the class count."""
        return 0


__all__ = [
    "MajorityClass",
    "UniformChance",
    "MAJORITY_MODEL_NAME",
    "CHANCE_MODEL_NAME",
]
