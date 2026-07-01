"""The ``few_shot(k)`` regime: k labelled examples per class, then probe.

``few_shot`` is ``linear_probe`` restricted to a **k-shot support set**: at fit time the
adapter keeps at most ``k`` examples per class (deterministically, seed 42) and fits the
same frozen-backbone head on that support set. Everything downstream -- embedding with the
frozen backbone, the injectable head, predict -- is inherited from
:class:`~rfbench.regimes.probe.LinearProbeAdapter`.

The ``k``-required-iff-``few_shot`` coupling mirrors
:class:`~rfbench.core.model.RegimeSpec` exactly: a ``FewShotAdapter`` cannot be built
without a ``k``, and ``k`` is written to ``result.json.regime.k_shot`` (verbatim, never
inferred). Subsampling is seed-stable: two adapters with the same ``(k, seed)`` select the
byte-identical support set from the same train split.

Pure stdlib -- no ``torch``/``numpy`` import; determinism via :class:`random.Random`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from rfbench.core.model import Regime, RegimeSpec
from rfbench.core.types import Batch
from rfbench.regimes.probe import Head, LinearProbeAdapter

#: Protocol-mandated subsampling seed (mirrors the 42 used across splits / configs).
DEFAULT_FEW_SHOT_SEED = 42


class FewShotAdapter(LinearProbeAdapter):
    """The ``few_shot`` regime: subsample ``k`` per class (seed 42), then linear-probe.

    Reuses the whole probe machinery and only overrides
    :meth:`LinearProbeAdapter._select_train_samples` to reduce the train split to a k-shot
    support set. ``k`` is required (enforced by :class:`RegimeSpec`) and injected into the
    declared regime so it reaches ``result.json.regime.k_shot``.
    """

    #: This adapter carries the ``few_shot`` regime (see the base's guard).
    _expected_regime = Regime.FEW_SHOT

    def __init__(
        self,
        k: int,
        head: Head | None = None,
        *,
        label_field: str = "label",
        seed: int = DEFAULT_FEW_SHOT_SEED,
    ) -> None:
        # RegimeSpec enforces k >= 1 and the few_shot<->k_shot coupling; let it raise.
        super().__init__(
            head,
            label_field=label_field,
            regime=RegimeSpec(Regime.FEW_SHOT, k_shot=k),
        )
        self._k = k
        self._seed = seed

    @property
    def k(self) -> int:
        """The per-class shot count (``== self.regime.k_shot``)."""
        return self._k

    def _select_train_samples(self, train_split: Iterable[Batch]) -> list[Batch]:
        """Keep at most ``k`` examples per class, deterministically (seeded shuffle).

        Groups the (materialised) train split by label preserving first-seen order, then
        for each class -- iterated in ascending label order for a class-order-independent
        result -- shuffles that class's samples with a per-run :class:`random.Random`
        seeded by ``self._seed`` and takes the first ``k``. Classes with fewer than ``k``
        samples contribute all of theirs. The output is flattened in ascending label order,
        so the same ``(k, seed, train_split)`` always yields the byte-identical support set.
        """
        by_label: dict[int, list[Batch]] = {}
        for sample in train_split:
            by_label.setdefault(int(sample[self._label_field]), []).append(sample)

        rng = random.Random(self._seed)
        support: list[Batch] = []
        for label in sorted(by_label):
            members = list(by_label[label])
            rng.shuffle(members)
            support.extend(members[: self._k])
        return support


__all__ = ["FewShotAdapter", "DEFAULT_FEW_SHOT_SEED"]
