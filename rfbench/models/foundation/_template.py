"""WP-61 -- copy-me template for adding a foundation-model wrapper.

Copy this file to ``rfbench/models/foundation/<your_model>.py``, rename the class, fill in
the four TODOs, and register it. You do **not** touch the frozen core
(``rfbench/core/``), the schemas, the regimes, or ``evaluate()`` -- a new model is purely a
new wrapper. See ``docs/ADDING_A_MODEL.md`` for the full walkthrough and
``rfbench/models/foundation/dummy.py`` for a complete, dependency-free worked example.

Checklist (also in the guide):

1. Subclass :class:`~rfbench.models.foundation.base.FoundationModel` (fixes
   ``family = "foundation"``).
2. Implement :meth:`embed` -- one representation vector per sample. This is what the
   ``linear_probe`` / ``few_shot`` regimes fit a head on.
3. Implement :meth:`forward` if your FM ships a task head (needed for ``from_scratch`` /
   ``full_finetune``); otherwise inherit the default, which falls back to :meth:`embed`.
4. Report :attr:`n_params` and register the class with ``@register_model("<name>")``.

Import the heavy stack (torch, your checkpoint loader) **lazily** via
:func:`~rfbench.models.foundation.base.require_torch` inside the method that needs it, so
``import rfbench.models.foundation`` stays dependency-free and the wrapper only pulls torch
when actually run (behind the ``rfbench[torch]`` extra).
"""

from __future__ import annotations

from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel, require_torch


@register_model("my-fm")  # TODO(1): pick a unique leaderboard name for your model.
class MyFoundationModel(FoundationModel):
    """TODO(2): one-line description of the backbone this wraps.

    Subclasses :class:`FoundationModel`, so it is a ``family == "foundation"`` model. It
    must be constructible with no required positional arguments so the registry path
    (``MODELS.get("my-fm")()``) can instantiate it; give any config a default or read it
    from an env/config file.
    """

    def __init__(self, *, name: str = "my-fm", checkpoint: str | None = None) -> None:
        """Wrap the backbone. Keep construction cheap -- load weights lazily in :meth:`embed`."""
        super().__init__(
            name,
            n_params=0,  # TODO(4a): set once the backbone is loaded (see load hook below).
            backbone=checkpoint,  # provenance surfaced on the board (result.json.model.backbone)
            pretrained=True,
        )
        self._checkpoint = checkpoint
        self._backbone: object | None = None  # loaded on first use

    def _load(self) -> object:
        """Lazily load and cache the backbone (the ONLY place the heavy import happens).

        Calling :func:`require_torch` here means ``import rfbench.models.foundation`` never
        pulls torch: the extra is required only when the model is actually run. Replace the
        body with your real checkpoint load.
        """
        if self._backbone is None:
            torch = require_torch()  # noqa: F841 - used by the real load below
            # TODO(3a): load your pretrained encoder, e.g.
            #   self._backbone = torch.load(self._checkpoint, map_location="cpu").eval()
            # and set self._n_params from the loaded module's parameter count.
            raise NotImplementedError("template: implement _load() to load your backbone")
        return self._backbone

    def embed(self, x: Batch) -> Tensor:
        """Return one frozen representation vector per sample in the collated batch ``x``.

        TODO(2): run the backbone's encoder on ``x["iq"]`` and return per-sample vectors.
        This feeds the ``linear_probe`` / ``few_shot`` adapters, so shape it as one vector
        per sample (a tensor/2-D array; the adapters normalise it to ``list[float]``).
        """
        backbone = self._load()  # noqa: F841 - stand-in until the real forward pass is filled in
        raise NotImplementedError("template: implement embed() to encode x['iq']")

    def forward(self, x: Batch) -> Tensor:
        """Return the task-head output for ``x`` (needed for from_scratch / full_finetune).

        TODO(3b): if your FM ships a fine-tuned task head, run it here and return logits /
        boxes / scores. If it does NOT (you only ever probe it), delete this override and
        inherit :meth:`FoundationModel.forward`, which falls back to :meth:`embed`.
        """
        raise NotImplementedError("template: implement forward() or inherit the embed() fallback")


__all__ = ["MyFoundationModel"]
