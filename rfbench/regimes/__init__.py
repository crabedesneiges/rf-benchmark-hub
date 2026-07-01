"""Regime adapters: the four locked regimes as uniform adapters around a ``Model`` (D5).

A regime is an *adapter*, not a model attribute (plan §5: "régimes = adaptateurs"). Each
adapter shares the :class:`RegimeAdapter` surface (``fit`` + ``predict`` + a declared
:class:`~rfbench.core.model.RegimeSpec`), so the harness and CLI drive all four the same
way and the regime is written VERBATIM into ``result.json`` -- declared, never inferred.

* ``from_scratch`` / ``full_finetune`` -- :class:`FromScratchAdapter` /
  :class:`FullFinetuneAdapter`: thin pass-throughs to :meth:`Model.forward` (the model is
  assumed already trained under that regime; the training loops are M3/M6).
* ``linear_probe`` -- :class:`LinearProbeAdapter`: frozen backbone, an injectable head fit
  on :meth:`Model.embed` features (default: the pure-stdlib :class:`NearestCentroidHead`).
* ``few_shot(k)`` -- :class:`FewShotAdapter`: subsample ``k`` per class (deterministic,
  seed 42), then behave like ``linear_probe``.

:func:`make_adapter` maps a :class:`~rfbench.core.model.RegimeSpec` to the right adapter,
so callers select a regime by spec (as parsed from ``configs/regime/*.yaml``) without a
manual switch. This subpackage is pure stdlib: importing it (or ``rfbench.regimes`` at
large) pulls in no ``torch``/``numpy``/``sklearn``/``jsonschema``.
"""

from __future__ import annotations

from rfbench.core.model import Regime, RegimeSpec
from rfbench.regimes.base import FittedState, RegimeAdapter, TrainSplit
from rfbench.regimes.few_shot import DEFAULT_FEW_SHOT_SEED, FewShotAdapter
from rfbench.regimes.passthrough import FromScratchAdapter, FullFinetuneAdapter
from rfbench.regimes.probe import (
    EmbeddingVector,
    Head,
    LinearProbeAdapter,
    NearestCentroidHead,
)


def make_adapter(regime: RegimeSpec, *, head: Head | None = None) -> RegimeAdapter:
    """Build the :class:`RegimeAdapter` for a declared :class:`RegimeSpec`.

    Dispatches on ``regime.name``; ``head`` is honoured only by the probing regimes
    (``linear_probe`` / ``few_shot``) and ignored by the pass-throughs. ``regime.k_shot``
    is required iff ``few_shot`` (already enforced by :class:`RegimeSpec`), so an
    ill-formed regime never reaches this factory. Raises ``ValueError`` on an unhandled
    regime -- a guard that would only trip if a new regime were added to the enum without
    an adapter here.
    """
    name = regime.name
    if name is Regime.FROM_SCRATCH:
        return FromScratchAdapter()
    if name is Regime.FULL_FINETUNE:
        return FullFinetuneAdapter()
    if name is Regime.LINEAR_PROBE:
        return LinearProbeAdapter(head)
    if name is Regime.FEW_SHOT:
        # RegimeSpec guarantees k_shot is set for few_shot; assert for the type checker.
        assert regime.k_shot is not None  # noqa: S101 - invariant already enforced upstream
        return FewShotAdapter(regime.k_shot, head)
    raise ValueError(f"no adapter registered for regime {name!r}")  # pragma: no cover


__all__ = [
    # Contract
    "RegimeAdapter",
    "FittedState",
    "TrainSplit",
    # Adapters
    "FromScratchAdapter",
    "FullFinetuneAdapter",
    "LinearProbeAdapter",
    "FewShotAdapter",
    # Heads
    "Head",
    "NearestCentroidHead",
    "EmbeddingVector",
    # Factory + constants
    "make_adapter",
    "DEFAULT_FEW_SHOT_SEED",
]
