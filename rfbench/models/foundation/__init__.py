"""Foundation-model wrappers (WP-60) + the copy-me template (WP-61).

A foundation model on the board is an arbitrary RF backbone adapted to a task under one of
the four locked regimes (D5). This package holds the reusable plumbing:

* :class:`~rfbench.models.foundation.base.FoundationModel` -- the generic wrapper
  implementing the :class:`~rfbench.core.model.Model` contract (``forward`` / ``embed`` /
  ``n_params`` + ``name`` / ``family == "foundation"``), so any backbone runs in every
  regime;
* :func:`~rfbench.models.foundation.base.run_regime` -- the bridge that adapts a wrapped FM
  under a :class:`~rfbench.core.model.RegimeSpec` (via :func:`rfbench.regimes.make_adapter`)
  into a :class:`~rfbench.core.model.Model` that :func:`rfbench.core.evaluate.evaluate` can
  score directly;
* :class:`~rfbench.models.foundation.dummy.DummyFoundationModel` -- a concrete,
  dependency-free EXAMPLE FM (deterministic hash embedding) registered as ``"dummy-fm"``, so
  the whole path runs in tests without ``torch``;
* :mod:`rfbench.models.foundation._template` -- the stub a contributor copies (see
  ``docs/ADDING_A_MODEL.md``).

Importing this package **registers the example FM** in
:data:`rfbench.core.registry.MODELS` (the ``@register_model`` decorator on
:class:`DummyFoundationModel` fires as an import side effect), exactly as the task packages
register their tasks. Import stays dependency-free: stdlib + the frozen core + the pure-stdlib
regimes only; ``torch``/``numpy`` load lazily behind ``rfbench[torch]`` via
:func:`~rfbench.models.foundation.base.require_torch`.
"""

from __future__ import annotations

from rfbench.models.foundation.base import (
    BackboneFn,
    FoundationModel,
    as_vectors,
    require_torch,
    run_regime,
)
from rfbench.models.foundation.dummy import (
    DEFAULT_EMBED_DIM,
    DummyFoundationModel,
    build_example_fm,
)

__all__ = [
    # Generic wrapper + bridge
    "FoundationModel",
    "BackboneFn",
    "run_regime",
    "require_torch",
    "as_vectors",
    # Example FM
    "DummyFoundationModel",
    "DEFAULT_EMBED_DIM",
    "build_example_fm",
]
