"""Pass-through regimes: ``from_scratch`` and ``full_finetune``.

Both regimes assume the wrapped :class:`~rfbench.core.model.Model` is *already trained
under that regime* (a from-scratch baseline, or a fully fine-tuned foundation model), so
at eval time the adapter is a thin pass-through to :meth:`Model.forward`. The only thing
that distinguishes the two on the leaderboard is the declared regime name, which
:func:`rfbench.core.evaluate.evaluate` writes verbatim into ``result.json`` (D5).

The *real training loops* (from-scratch training, full fine-tuning) are M3/M6 work: they
do not belong in the eval harness. :meth:`_PassThroughAdapter.fit` is the documented hook
where such a loop would live; here it is a no-op returning an empty
:class:`~rfbench.regimes.base.FittedState`.

Pure stdlib -- no ``torch``/``numpy`` import.
"""

from __future__ import annotations

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.types import Batch, Tensor
from rfbench.regimes.base import FittedState, RegimeAdapter, TrainSplit


class _PassThroughAdapter(RegimeAdapter):
    """Shared base for the two regimes that call :meth:`Model.forward` directly.

    Subclasses only pin :attr:`regime`. ``k_shot`` is rejected for both (mirrors
    :class:`RegimeSpec`: only ``few_shot`` carries ``k_shot``), so a stray ``k`` cannot
    silently ride along on a non-few-shot regime.
    """

    def __init__(self, regime: RegimeSpec) -> None:
        if regime.k_shot is not None:
            raise ValueError(
                f"regime '{regime.name.value}' does not accept k_shot "
                f"(got k_shot={regime.k_shot}); only 'few_shot' is parameterised by k"
            )
        self.regime = regime

    def fit(self, model: Model, train_split: TrainSplit) -> FittedState:
        """No-op fit: the model is assumed already trained under this regime.

        Documented hook for M3/M6: the from-scratch training loop / full fine-tuning loop
        would run here (consuming ``train_split``, mutating ``model``'s weights) and record
        provenance in the returned state. The eval harness deliberately does NOT implement
        heavy training; it returns an empty state so the pass-through path stays pure and
        dependency-free.
        """
        return FittedState(head=None, info={"regime": self.regime.name.value, "trained": False})

    def predict(self, model: Model, inputs: Batch, state: FittedState) -> Tensor:
        """Return ``model.forward(inputs)`` unchanged -- the model owns the task head."""
        return model.forward(inputs)


class FromScratchAdapter(_PassThroughAdapter):
    """The ``from_scratch`` regime: a model trained from random init on the task.

    Thin pass-through to :meth:`Model.forward`; the training that produced those weights
    is out of scope for the harness (M3).
    """

    def __init__(self) -> None:
        super().__init__(RegimeSpec(Regime.FROM_SCRATCH))


class FullFinetuneAdapter(_PassThroughAdapter):
    """The ``full_finetune`` regime: a foundation model fine-tuned end-to-end on the task.

    Thin pass-through to :meth:`Model.forward`; the fine-tuning that produced those weights
    is out of scope for the harness (M6).
    """

    def __init__(self) -> None:
        super().__init__(RegimeSpec(Regime.FULL_FINETUNE))


__all__ = ["FromScratchAdapter", "FullFinetuneAdapter"]
