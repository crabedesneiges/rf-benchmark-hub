"""The ``Model`` contract and the locked adaptation regimes (D5).

A :class:`Model` is anything the harness can evaluate. Foundation models must expose
:meth:`Model.embed` (used by the ``linear_probe`` and ``few_shot`` regimes); plain
baselines may leave it unimplemented.

The regime is *declared, never inferred* (D5). :class:`Regime` is the frozen set of
four regimes and, being a ``str`` enum, serialises straight into
``result.json.regime.name``. :class:`RegimeSpec` couples the regime name with an
optional ``k_shot`` exactly as the JSON schema's ``allOf`` does.

No ``torch`` import at module top: tensors are typed via :data:`rfbench.core.types.Tensor`
(an alias for ``Any``) so ``import rfbench.core`` stays dependency-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from rfbench.core.types import Tensor


class Regime(str, Enum):
    """The four locked adaptation regimes (D5).

    A ``str`` enum so ``regime.value`` serialises directly into the
    ``result.json.regime.name`` field. The values are kept in lockstep with the
    ``regime.name`` enum of ``schemas/result.schema.json``.
    """

    FROM_SCRATCH = "from_scratch"
    FULL_FINETUNE = "full_finetune"
    LINEAR_PROBE = "linear_probe"
    FEW_SHOT = "few_shot"


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    """A declared regime, mirroring the ``regime{}`` object of the result schema.

    ``k_shot`` is set **iff** the regime is :attr:`Regime.FEW_SHOT`. ``__post_init__``
    enforces this coupling in Python exactly as the schema's ``allOf`` does, so an
    ill-formed regime can never be constructed and never reach :func:`evaluate`.
    """

    name: Regime
    k_shot: int | None = None

    def __post_init__(self) -> None:
        """Enforce the ``k_shot`` coupling and positivity."""
        if (self.name is Regime.FEW_SHOT) != (self.k_shot is not None):
            raise ValueError("k_shot must be set iff regime is 'few_shot'")
        if self.k_shot is not None and self.k_shot < 1:
            raise ValueError("k_shot must be >= 1")


class Model(ABC):
    """Any evaluable model.

    Foundation models MUST implement :meth:`embed` (used by the ``linear_probe`` and
    ``few_shot`` regimes). ``name`` and ``family`` are the identity written into
    ``result.json.model``.
    """

    #: Leaderboard / registry name, e.g. ``"mcldnn"``, ``"xcit-nano"``.
    name: str
    #: Coarse board bucket: ``"baseline"`` seeds the board, ``"foundation"`` wraps an
    #: FM exposing :meth:`embed`.
    family: Literal["baseline", "foundation"]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Return the task-head output (logits / boxes / detection scores)."""

    @abstractmethod
    def embed(self, x: Tensor) -> Tensor:
        """Return a frozen representation for ``linear_probe`` / ``few_shot``.

        Foundation models MUST override this; baselines may raise
        :class:`NotImplementedError`.
        """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total parameter count, including any adapter head.

        Written to ``result.json.model.n_params``.
        """


__all__ = ["Regime", "RegimeSpec", "Model"]
