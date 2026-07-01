"""WP-60 -- the generic foundation-model wrapper and the regime bridge.

A **foundation model (FM)** on the board is an arbitrary RF backbone adapted to a task
under one of the four locked regimes (D5). This module provides the two pieces that make
that work without touching the frozen core:

* :class:`FoundationModel` -- a generic :class:`~rfbench.core.model.Model` that wraps a
  backbone and exposes :meth:`~rfbench.core.model.Model.embed` (the representation the
  ``linear_probe`` / ``few_shot`` regime adapters fit a head on) plus
  :meth:`~rfbench.core.model.Model.forward` (the task-head output the pass-through
  ``from_scratch`` / ``full_finetune`` regimes read). A concrete FM is built either by
  subclassing (override :meth:`embed` / :meth:`forward`) or -- for a quick wrapper -- by
  injecting ``embed_fn`` / ``forward_fn`` callables. ``family`` is fixed to ``"foundation"``
  so every wrapped FM lands in the board's foundation bucket.

* :func:`run_regime` -- the bridge from a :class:`FoundationModel` + a
  :class:`~rfbench.core.model.RegimeSpec` to a :class:`~rfbench.core.model.Model` that
  :func:`rfbench.core.evaluate.evaluate` can drive directly. It resolves the adapter via
  :func:`rfbench.regimes.make_adapter`, fits it on the ``train`` split, and returns an
  :class:`_AdaptedModel` whose :meth:`forward` delegates to ``adapter.predict``. This is
  how "the same wrapped FM run via make_adapter for linear_probe / full_finetune /
  few_shot" reaches ``evaluate()`` and produces a schema-valid ``result.json`` with the
  regime declared verbatim.

HARD CONSTRAINT: importing this module pulls in **no** third-party dependency. Tensors are
typed via :mod:`rfbench.core.types` (aliased to ``Any``); ``torch``/``numpy`` are never
imported here. A real backbone loads lazily behind the ``rfbench[torch]`` extra -- see
:func:`require_torch`, the single, clearly-hinted guard a concrete torch wrapper calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import SupportsFloat, cast

from rfbench.core.model import Model, RegimeSpec
from rfbench.core.types import Batch, Tensor
from rfbench.regimes import make_adapter
from rfbench.regimes.base import FittedState, RegimeAdapter, TrainSplit
from rfbench.regimes.probe import Head

#: A backbone hook: maps a collated batch to per-sample outputs (embeddings or logits).
#: Kept structural (``Batch -> Tensor``) so a wrapper can inject a plain Python function,
#: a bound method, or a lazily-loaded torch module's ``__call__`` without this module ever
#: importing a tensor framework.
BackboneFn = Callable[[Batch], Tensor]


def require_torch() -> ModuleType:
    """Import and return ``torch``, or raise with the ``rfbench[torch]`` install hint.

    The single guarded import point for concrete torch-backed FM wrappers: dependency-free
    code (this module, the example dummy, the unit tests) never calls it, so
    ``import rfbench.models.foundation`` stays dependency-free. A real backbone wrapper
    calls this inside its loader/``forward``/``embed`` so the heavy import happens lazily
    and, when the extra is missing, fails with a clear, actionable message rather than a
    bare ``ModuleNotFoundError``.
    """
    try:
        import torch  # noqa: PLC0415 - lazy by design; keeps the package import dependency-free
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "this foundation-model wrapper needs PyTorch; install the optional extra "
            "with `pip install rfbench[torch]` (or add torch to your cluster ARM venv)."
        ) from exc
    return cast("ModuleType", torch)


class FoundationModel(Model):
    """Generic wrapper adapting an arbitrary RF backbone to the :class:`Model` contract.

    Implements the full contract -- :meth:`forward`, :meth:`embed`, :attr:`n_params`,
    :attr:`name`, :attr:`family` -- so the harness can evaluate any wrapped backbone in any
    of the four regimes. :attr:`family` is fixed to ``"foundation"``; :meth:`embed` is the
    frozen representation the ``linear_probe`` / ``few_shot`` adapters fit a head on.

    Two ways to specialise, both without touching the core:

    * **Subclass** and override :meth:`embed` (and :meth:`forward` if the FM ships a task
      head) -- the route real backbones take; the loader can call :func:`require_torch`.
    * **Inject** ``embed_fn`` / ``forward_fn`` callables at construction -- the quick route
      for a thin wrapper or a test double, with no subclass needed.

    ``forward`` defaults to ``embed`` when no task head is provided: the pass-through
    regimes (``from_scratch`` / ``full_finetune``) then read the raw representation, which
    is enough to exercise the whole path; a concrete FM overrides ``forward`` with its real
    head (or is driven through :func:`run_regime` for the probing regimes, where the head is
    the adapter's).
    """

    family = "foundation"

    def __init__(
        self,
        name: str,
        *,
        embed_fn: BackboneFn | None = None,
        forward_fn: BackboneFn | None = None,
        n_params: int = 0,
        backbone: str | None = None,
        pretrained: bool = True,
    ) -> None:
        """Wrap a backbone under ``name``.

        ``embed_fn`` / ``forward_fn`` inject the backbone hooks for the no-subclass route;
        leave them ``None`` in a subclass that overrides :meth:`embed` / :meth:`forward`.
        ``n_params`` is the reported parameter count (written to ``result.json.model``);
        ``backbone`` / ``pretrained`` are provenance a caller may surface on the board.
        """
        if not name:
            raise ValueError("FoundationModel needs a non-empty name")
        self.name = name
        self._embed_fn = embed_fn
        self._forward_fn = forward_fn
        self._n_params = int(n_params)
        #: Optional backbone/checkpoint id (``result.json.model.backbone``); provenance only.
        self.backbone = backbone
        #: Whether the backbone weights were pretrained (``result.json.model.pretrained``).
        self.pretrained = pretrained

    def embed(self, x: Batch) -> Tensor:
        """Return one frozen representation vector per sample in the collated batch ``x``.

        The default implementation delegates to the injected ``embed_fn``. A subclass wraps
        a real encoder here (calling :func:`require_torch` for the lazy import). The output
        is what the ``linear_probe`` / ``few_shot`` adapters normalise to ``list[float]``
        vectors, so it must be one vector per sample (a ``Sequence[Sequence[float]]`` on the
        dependency-free path).
        """
        if self._embed_fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} '{self.name}' provides no embed(): override embed() "
                "in a subclass or pass embed_fn= to FoundationModel(...)."
            )
        return self._embed_fn(x)

    def forward(self, x: Batch) -> Tensor:
        """Return the task-head output for the collated batch ``x``.

        Uses the injected ``forward_fn`` when present; otherwise falls back to :meth:`embed`
        so a wrapper with no task head still runs through the pass-through regimes. Concrete
        FMs that ship a fine-tuned head override this (or are driven via :func:`run_regime`
        for the probing regimes, where the adapter owns the head).
        """
        if self._forward_fn is not None:
            return self._forward_fn(x)
        return self.embed(x)

    @property
    def n_params(self) -> int:
        """Total parameter count, including any adapter head (``result.json.model.n_params``)."""
        return self._n_params


class _AdaptedModel(Model):
    """A :class:`Model` view of ``(base_fm, adapter, state)`` for :func:`evaluate`.

    :func:`rfbench.core.evaluate.evaluate` drives a model purely through
    :meth:`Model.forward`; a probing regime, however, lives in a
    :class:`~rfbench.regimes.base.RegimeAdapter` (``fit`` on ``train`` then ``predict`` per
    batch). This thin bridge closes the gap: its :meth:`forward` delegates to
    ``adapter.predict(base, batch, state)``, so the *adapted* FM -- regardless of regime --
    is a plain :class:`Model` that ``evaluate`` can score. It carries the base FM's identity
    (``name`` / ``family`` / ``n_params``) so the emitted row attributes to the FM, and the
    adapter's declared :class:`RegimeSpec` so the caller can write the regime verbatim.
    """

    def __init__(self, base: FoundationModel, adapter: RegimeAdapter, state: FittedState) -> None:
        self._base = base
        self._adapter = adapter
        self._state = state
        self.name = base.name
        self.family = base.family
        #: The regime this adapted model was fitted under (declared, never inferred).
        self.regime: RegimeSpec = adapter.regime

    def forward(self, x: Batch) -> Tensor:
        """Return the adapter's predictions for the collated batch ``x``."""
        return self._adapter.predict(self._base, x, self._state)

    def embed(self, x: Batch) -> Tensor:
        """Delegate to the wrapped FM's embedding (kept available for the probing regimes)."""
        return self._base.embed(x)

    @property
    def n_params(self) -> int:
        """Report the wrapped FM's parameter count verbatim."""
        return self._base.n_params


def run_regime(
    model: FoundationModel,
    regime: RegimeSpec,
    train_split: TrainSplit,
    *,
    head: Head | None = None,
) -> _AdaptedModel:
    """Adapt ``model`` under ``regime`` and return a :class:`Model` ready for :func:`evaluate`.

    Resolves the adapter with :func:`rfbench.regimes.make_adapter` (so the ``k_shot`` <->
    ``few_shot`` coupling and the regime dispatch are the harness' own), fits it on
    ``train_split`` (a no-op for the pass-through regimes; head fit on frozen ``embed``
    features for the probing ones), and wraps the result in an :class:`_AdaptedModel`. The
    returned model's :meth:`forward` yields the regime's predictions, so::

        adapted = run_regime(fm, RegimeSpec(Regime.LINEAR_PROBE), train_split)
        result = evaluate(adapted, task, "test", adapted.regime)

    runs the *same* wrapped FM through ``evaluate()`` in any of the four regimes and emits a
    schema-valid ``result.json`` with the regime declared verbatim. ``head`` is honoured only
    by the probing regimes (ignored by the pass-throughs), mirroring ``make_adapter``.
    """
    adapter = make_adapter(regime, head=head)
    state = adapter.fit(model, train_split)
    return _AdaptedModel(model, adapter, state)


def as_vectors(embedded: Tensor) -> list[list[float]]:
    """Normalise an ``embed`` output into a list of per-sample float vectors (pure stdlib).

    A convenience mirror of the harness' internal normaliser for wrapper authors and tests:
    accepts a list of per-sample vectors (or a single flat vector treated as one sample) and
    coerces every element to ``list[float]``. Rejects a mapping, which is never a valid
    per-sample embedding batch.
    """
    if isinstance(embedded, dict):
        raise TypeError("embed() must return per-sample vectors, not a mapping")
    rows: Sequence[object] = list(embedded)
    if rows and not isinstance(rows[0], (list, tuple)):
        return [[float(v) for v in _as_iterable(rows)]]
    return [[float(v) for v in _as_iterable(row)] for row in rows]


def _as_iterable(value: object) -> Iterable[SupportsFloat]:
    """Return ``value`` as an iterable of float-coercible scalars, raising otherwise.

    Runtime-narrows an ``object`` (the ``Tensor`` alias is ``Any``, so ``embed`` output is
    untyped) to an iterable ``float()`` accepts; the returned elements are typed
    ``SupportsFloat`` for the static checker (coercion still fails loudly at runtime on a
    genuinely non-numeric element).
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"expected an iterable of scalars, got {type(value).__name__}")
    return cast("Iterable[SupportsFloat]", value)


__all__ = [
    "BackboneFn",
    "FoundationModel",
    "run_regime",
    "require_torch",
    "as_vectors",
]
