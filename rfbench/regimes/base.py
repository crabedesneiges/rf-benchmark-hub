"""The :class:`RegimeAdapter` contract: the four locked regimes as adapters (D5).

A **regime is an adapter around a** :class:`~rfbench.core.model.Model`, not a property of
the model (plan §5: "régimes = adaptateurs"). Every adapter exposes the same two-method
surface so the harness / CLI can drive all four uniformly:

* :meth:`RegimeAdapter.fit` consumes the ``train`` split and returns a
  :class:`FittedState` (opaque per-adapter state; empty for the pass-through regimes).
* :meth:`RegimeAdapter.predict` maps a collated inference batch to predictions, in the
  same field-agnostic shape :func:`rfbench.core.evaluate.evaluate` feeds to
  ``model.forward``.

The adapter also surfaces its declared :class:`~rfbench.core.model.RegimeSpec` via
:attr:`RegimeAdapter.regime`, which is what :func:`evaluate` writes VERBATIM into
``result.json.regime`` -- the regime is *declared, never inferred* (D5).

No third-party imports at module top: this subpackage stays pure stdlib so it imports in
the light install (only ``pytest`` + ``jsonschema``). Tensors/batches are typed via
:mod:`rfbench.core.types`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.types import Batch, Tensor

#: A ``train`` split as seen by an adapter: an iterable of per-sample ``Batch`` dicts
#: (exactly what :meth:`rfbench.core.dataset.Dataset.load` yields, before collation).
#: Kept as an ``Iterable`` so an adapter may stream a large split without materialising it
#: (few-shot subsampling is the one place we buffer, and only ``k`` per class).
TrainSplit = Iterable[Batch]


@dataclass(frozen=True, slots=True)
class FittedState:
    """The opaque state produced by :meth:`RegimeAdapter.fit`.

    The pass-through regimes (``from_scratch`` / ``full_finetune``) return an empty state:
    the model is assumed already trained per that regime, so there is nothing to fit here
    (the real training loops are M3 -- see the pass-through adapters' documented hook).

    The probing regimes (``linear_probe`` / ``few_shot``) store the fitted head under
    :attr:`head` and any per-adapter provenance under :attr:`info` (e.g. the number of
    train samples actually used, or the per-class shot count for few-shot). ``info`` is
    diagnostic only -- it never reaches ``result.json``.
    """

    #: The fitted classification head (a :class:`~rfbench.regimes.probe.Head`) or ``None``
    #: for the pass-through regimes. Typed ``Any`` to avoid a base -> probe import cycle.
    head: Any | None = None
    #: Free-form, JSON-serialisable fit provenance (diagnostic only, never scored).
    info: dict[str, Any] = field(default_factory=dict)


class RegimeAdapter(ABC):
    """Uniform adapter surface for one declared regime around any :class:`Model`.

    Concrete adapters (``from_scratch``, ``full_finetune``, ``linear_probe``,
    ``few_shot``) implement :meth:`fit` and :meth:`predict`. The harness calls
    :meth:`fit` once on the ``train`` split, then :meth:`predict` per eval batch, and
    reads :attr:`regime` to tag the row. Regime coupling (``k_shot`` iff ``few_shot``) is
    enforced by :class:`RegimeSpec`, so an ill-formed regime can never reach an adapter.
    """

    #: The regime this adapter declares; drives ``result.json.regime`` verbatim (D5).
    regime: RegimeSpec

    @property
    def name(self) -> Regime:
        """The declared regime enum (``self.regime.name``); shorthand for tests/logs."""
        return self.regime.name

    @abstractmethod
    def fit(self, model: Model, train_split: TrainSplit) -> FittedState:
        """Adapt ``model`` to ``train_split`` and return the :class:`FittedState`.

        Deterministic given a deterministic ``train_split`` and ``seed`` (few-shot
        subsampling and the default centroid head are both seed-stable). Called at most
        once per evaluation, before any :meth:`predict`.
        """

    @abstractmethod
    def predict(self, model: Model, inputs: Batch, state: FittedState) -> Tensor:
        """Map a collated inference ``inputs`` batch to predictions under this regime.

        ``inputs`` is the collated batch (dict of field -> list) exactly as
        :func:`rfbench.core.evaluate.evaluate` passes to ``model.forward``; the returned
        predictions are consumed by the task's metrics. ``state`` is the value returned by
        :meth:`fit` (empty for the pass-through regimes).
        """


__all__ = ["TrainSplit", "FittedState", "RegimeAdapter"]
