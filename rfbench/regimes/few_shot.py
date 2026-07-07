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

:class:`FewShotAdapter` itself stays a **single-episode** adapter (one instance == one
seeded draw == one declared regime row, per the "one row, one seed" contract) --
:func:`run_episodic` is the separate multi-episode ORCHESTRATION primitive that builds and
runs several such adapters across a seed range, for callers that need the per-episode
metric spread (e.g. ``metrics.uncertainty.multi_seed_std`` in schema 1.2.0, wired up by a
sibling PR -- this module only produces the raw per-episode values).

Pure stdlib -- no ``torch``/``numpy`` import; determinism via :class:`random.Random`.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.types import Batch
from rfbench.regimes.base import FittedState, TrainSplit
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


#: Protocol-mandated episode count floor for a multi-seed few-shot report.
MIN_EPISODES = 10


class EpisodeResult:
    """One episode's outcome from :func:`run_episodic`: the seed and its metric value.

    A thin, JSON-serialisable record -- exactly the two fields an aggregator needs to
    compute a multi-seed spread (e.g. ``result.json.metrics.uncertainty.multi_seed_std``,
    schema 1.2.0): the ``seed`` that produced this episode's support set, and the
    ``primary_metric`` value ``predict_fn`` reported for the adapter fitted on it.
    """

    __slots__ = ("seed", "primary_metric")

    def __init__(self, seed: int, primary_metric: float) -> None:
        self.seed = seed
        self.primary_metric = primary_metric

    def __repr__(self) -> str:
        return f"EpisodeResult(seed={self.seed}, primary_metric={self.primary_metric!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EpisodeResult):
            return NotImplemented
        return self.seed == other.seed and self.primary_metric == other.primary_metric


def run_episodic(
    adapter_factory: Callable[[int], FewShotAdapter],
    model: Model,
    train_split: TrainSplit,
    predict_fn: Callable[[Model, FewShotAdapter, FittedState], float],
    *,
    n_episodes: int = MIN_EPISODES,
    base_seed: int = DEFAULT_FEW_SHOT_SEED,
) -> list[EpisodeResult]:
    """Fit + score ``n_episodes`` independent few-shot draws and return one result each.

    ``FewShotAdapter`` stays mono-episode by design (one instance == one seed == one
    declared regime row); this is the orchestration primitive that repeats that single
    episode across a range of seeds so a caller can report an uncertainty estimate over
    the draw, as the evaluation protocol requires for the ``few_shot`` regime
    (``n_episodes >= 10``, seed 42 as the base -- mirrors :data:`DEFAULT_FEW_SHOT_SEED`).

    For each seed in ``range(base_seed, base_seed + n_episodes)``: builds a fresh adapter
    via ``adapter_factory(seed)`` (the caller controls ``k``/head/label_field -- this
    function only varies the seed), fits it on ``train_split`` (materialised once and
    reused across episodes, since each episode reshuffles it independently), and calls
    ``predict_fn(model, adapter, state)`` -- ``state`` is the :class:`~rfbench.regimes.base.
    FittedState` that episode's ``fit`` returned, so the callback can drive
    ``adapter.predict(model, eval_batch, state)`` itself -- to obtain that episode's
    primary-metric value. The caller closes over the eval split/task/metric; this module
    makes no assumption about what "primary metric" means beyond "one float per episode".

    Returns the list of :class:`EpisodeResult`, one per episode, in seed order. This is a
    reusable primitive only -- it does NOT fit into ``result.json`` itself; the caller (or
    a sibling PR) is responsible for aggregating these into
    ``metrics.uncertainty.multi_seed_std``.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    materialised_split = list(train_split)
    results: list[EpisodeResult] = []
    for seed in range(base_seed, base_seed + n_episodes):
        adapter = adapter_factory(seed)
        state = adapter.fit(model, materialised_split)
        results.append(EpisodeResult(seed=seed, primary_metric=predict_fn(model, adapter, state)))
    return results


__all__ = [
    "FewShotAdapter",
    "DEFAULT_FEW_SHOT_SEED",
    "MIN_EPISODES",
    "EpisodeResult",
    "run_episodic",
]
