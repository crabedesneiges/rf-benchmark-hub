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
* :class:`~rfbench.models.foundation.iqfm.IqfmBase` -- the IQFM raw-IQ SSL foundation model
  (ShuffleNetV2-x0.5 backbone) registered as ``"iqfm-base"``; re-exported here so its
  ``@register_model`` fires on package import. Its torch backbone loads lazily, so importing
  this package stays dependency-free;
* :class:`~rfbench.models.foundation.wireless_jepa.WirelessJepa` -- the WirelessJEPA raw-IQ JEPA
  foundation model registered as ``"wireless-jepa"``; shares IQFM's ShuffleNetV2-x0.5 backbone
  ("matched to IQFM"), differing only in the pre-training objective (JEPA vs SimCLR). Re-exported
  here so its ``@register_model`` fires on package import;
* :mod:`rfbench.models.foundation._template` -- the stub a contributor copies (see
  ``docs/ADDING_A_MODEL.md``).

Importing this package **registers the example FM, IQFM and WirelessJEPA** in
:data:`rfbench.core.registry.MODELS` (the ``@register_model`` decorators on
:class:`DummyFoundationModel` / :class:`IqfmBase` / :class:`WirelessJepa` fire as an import side
effect), exactly as the task packages register their tasks. Import stays dependency-free: stdlib
+ the frozen core + the
pure-stdlib regimes only; ``torch``/``numpy`` load lazily behind ``rfbench[torch]`` via
:func:`~rfbench.models.foundation.base.require_torch`. (The LWM-Spectro wrapper is deliberately
NOT re-exported here — it registers only on an explicit
``import rfbench.models.foundation.lwm_spectro``.)
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
from rfbench.models.foundation.iqfm import IqfmBase
from rfbench.models.foundation.shufflenet1d import build_shufflenet1d
from rfbench.models.foundation.wireless_jepa import WirelessJepa

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
    # IQFM raw-IQ SSL FM + its reusable backbone
    "IqfmBase",
    "build_shufflenet1d",
    # WirelessJEPA raw-IQ JEPA FM (shares IQFM's backbone)
    "WirelessJepa",
]
