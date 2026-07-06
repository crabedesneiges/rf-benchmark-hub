"""WirelessJEPA: a raw-IQ JEPA foundation model on the board (registered ``"wireless-jepa"``).

`WirelessJEPA <https://arxiv.org/abs/2601.20190>`_ (arXiv:2601.20190, 2026) is a raw-IQ RF
foundation model built on the **same ShuffleNetV2-x0.5 backbone as IQFM** ("matched to IQFM"),
but pre-trained with **JEPA** — masked-latent prediction with an **EMA teacher** (momentum
0.996 → 1.0) and, crucially, **no data augmentation** (the masking is the whole self-supervised
signal). This module wraps that backbone (the context / target encoder — NOT the JEPA predictor
head) as a frozen :class:`~rfbench.models.foundation.base.FoundationModel`, so the AMC board gets
a second raw-IQ SSL-FM row alongside IQFM (`iqfm-base`).

WHY IT MATTERS. WirelessJEPA's paper number — **74.78% on RML2016.10a** (11 mods, −20…+18 dB,
linear-probe, 500-shot) — is the **single most board-comparable public FM result**: it is on our
exact AMC dataset/protocol and **beats our supervised MCLDNN (61.71%)**. See
`docs/BIBLIOGRAPHY.md` §A.5.

WEIGHTS PROVENANCE — read before trusting any score. WirelessJEPA's **weights are NOT published**,
and the 74.78% is an **out-of-distribution** figure: the encoder is pre-trained on the authors'
own OTA MIMO testbed (the same one IQFM uses — which we do NOT have) and probed on RadioML. This
wrapper reproduces only the *architecture and recipe*: our backbone is (re-)pre-trained IN-REPO
with JEPA on the RadioML 2016.10a **train** split (delabelised), which is a **different,
in-distribution** setting — so a score from these weights is **ours**, not the paper's, and must be
labelled as such on the board (``model.notes`` / the result row), **never** presented as 74.78%.

Backbone sharing. The encoder is built by the reusable
:func:`rfbench.models.foundation.shufflenet1d.build_shufflenet1d` — the *same* function IQFM uses,
so the two FMs are guaranteed architecturally identical (the paper's "matched to IQFM"). They
differ only in the pre-training objective (JEPA here vs SimCLR for IQFM) and hence the loaded
weights. Input normalisation is unit-max ``iq/max(|iq|)`` — adopted as the matched-IQFM-family
convention (WirelessJEPA's exact input norm is unpublished; flagged UNVERIFIED below).

HARD CONSTRAINT: ``import rfbench.models.foundation`` stays dependency-free. ``torch`` is imported
lazily via :func:`~rfbench.models.foundation.base.require_torch` inside the loader / ``embed``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel, require_torch
from rfbench.models.foundation.shufflenet1d import (
    EMBED_DIM_X0_5,
    IN_CHANNELS,
    build_shufflenet1d,
)

_LOG = logging.getLogger(__name__)

#: Frozen embedding width (the shared ShuffleNetV2-x0.5 conv5 output, mean-pooled over time).
EMBED_DIM = EMBED_DIM_X0_5
#: Default cached checkpoint filename for our in-repo JEPA-pre-trained backbone (the EMA target
#: encoder — the representation the paper probes).
BACKBONE_CHECKPOINT = "wireless_jepa_shufflenet1d.pth"
#: Small floor so unit-max normalisation never divides by zero on an all-zero window.
_MAX_EPS = 1e-8


def wireless_jepa_cache_dir() -> Path:
    """Return ``$RFBENCH_CACHE/wireless-jepa`` (backbone cache; ``$RFBENCH_CACHE`` or CWD)."""
    root = Path(os.environ.get("RFBENCH_CACHE", ".")).expanduser()
    return root / "wireless-jepa"


def backbone_checkpoint_path() -> Path:
    """Return the cached backbone-checkpoint path ``$RFBENCH_CACHE/wireless-jepa/<file>``."""
    return wireless_jepa_cache_dir() / BACKBONE_CHECKPOINT


def _unit_max_normalise(iq: Tensor, torch_mod: ModuleType) -> Tensor:
    """Per-sample unit-max normalisation ``iq / max(|iq|)`` over a ``(2, L)`` window.

    The scale is the max complex magnitude over the window (floored by :data:`_MAX_EPS`). Adopted
    as the matched-IQFM-family input convention; WirelessJEPA's exact input norm is unpublished.
    """
    torch = cast("Any", torch_mod)
    magnitude = torch.sqrt(iq[0] ** 2 + iq[1] ** 2)  # (L,) complex magnitude
    scale = magnitude.max().clamp_min(_MAX_EPS)
    return iq / scale


def _iq_batch_to_tensor(iq_batch: object, torch_mod: ModuleType) -> Tensor:
    """Collate ``x["iq"]`` (a list of ``(2, L)`` windows) into a unit-max-normalised ``(B, 2, L)``.

    Each window is coerced to ``float32``, validated as ``(2, L)`` and unit-max normalised per
    sample before stacking. Accepts the nested-list windows the dependency-free synthetic AMC path
    yields and the numpy arrays the cluster path yields (``torch.as_tensor`` handles both).
    """
    torch = cast("Any", torch_mod)
    if isinstance(iq_batch, dict) or not isinstance(iq_batch, Iterable):
        raise TypeError("WirelessJEPA adapter expects x['iq'] as an iterable of IQ windows")
    samples: list[object] = list(iq_batch)
    if not samples:
        raise ValueError("WirelessJEPA adapter expected a non-empty list of IQ samples")

    windows: list[Any] = []
    for sample in samples:
        iq = torch.as_tensor(sample, dtype=torch.float32)
        if iq.ndim != 2 or iq.shape[0] != IN_CHANNELS:
            raise ValueError(
                f"expected an IQ window of shape ({IN_CHANNELS}, L); got {tuple(iq.shape)}"
            )
        windows.append(_unit_max_normalise(iq, torch))
    return torch.stack(windows, dim=0)  # (B, 2, L)


@register_model("wireless-jepa")
class WirelessJepa(FoundationModel):
    """The WirelessJEPA raw-IQ JEPA foundation model as a board ``Model`` (``"wireless-jepa"``).

    Wraps the frozen ShuffleNetV2-x0.5 raw-IQ encoder (the JEPA context / EMA-target encoder,
    shared with IQFM) behind the :class:`~rfbench.core.model.Model` contract:

    * :meth:`embed` -> ``(B, 1024)`` frozen mean-pooled features (unit-max normalised input) for
      the ``linear_probe`` / ``few_shot`` regimes;
    * :meth:`forward` -> inherited from :class:`FoundationModel` (falls back to :meth:`embed`) —
      WirelessJEPA ships **no task head** (only the JEPA predictor, which is discarded), so we only
      ever probe it;
    * :attr:`n_params` -> encoder parameter count (~335k once loaded); :attr:`family` ->
      ``"foundation"``.

    Constructed with no required args (``MODELS.get("wireless-jepa")()`` on the registry path).
    Construction is cheap: torch + the backbone build lazily on first :meth:`embed`. When
    ``checkpoint=None`` the wrapper resolves the cached backbone at
    ``$RFBENCH_CACHE/wireless-jepa/wireless_jepa_shufflenet1d.pth``; if that is absent it runs on a
    randomly-initialised backbone and sets :attr:`pretrained` to ``False`` (never silently claims
    pretrained features). A checkpoint present but not matching the backbone RAISES.
    """

    def __init__(
        self,
        *,
        name: str = "wireless-jepa",
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """Wrap the WirelessJEPA backbone under ``name``; keep construction torch-free and cheap.

        ``checkpoint`` overrides the resolved cache path
        (``$RFBENCH_CACHE/wireless-jepa/wireless_jepa_shufflenet1d.pth``); ``device`` pins the run
        device (defaults to CUDA when available, else CPU). Weights load lazily on first
        :meth:`embed`.
        """
        super().__init__(
            name,
            n_params=0,  # set once the backbone is loaded (see _ensure_loaded)
            backbone="shufflenet_v2_x0_5_1d (WirelessJEPA/JEPA; weights unpublished, retrained)",
            pretrained=True,
        )
        self._checkpoint = checkpoint
        self._device_str = device
        self._backbone: Any = None
        self._device: Any = None

    # -- lazy load ------------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Build the backbone and load real weights on first use (the only heavy step)."""
        if self._backbone is not None:
            return
        torch = require_torch()
        resolved = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved)

        backbone = build_shufflenet1d().to(self._device).eval()
        self._load_weights(backbone, torch)
        self._backbone = backbone
        self._n_params = sum(p.numel() for p in backbone.parameters())

    def _load_weights(self, backbone: Tensor, torch_mod: ModuleType) -> None:
        """Load our JEPA-pre-trained ``state_dict`` into ``backbone`` (``strict=False`` + guard).

        Resolves the checkpoint from ``self._checkpoint`` or the cached default. When absent the
        backbone keeps its random init and :attr:`pretrained` flips to ``False``. When present, any
        MISSING backbone key (the state_dict does not match the reconstruction) RAISES — we refuse
        to run a partially-random backbone and report it as pretrained. The checkpoint stores the
        **EMA target encoder** (the representation the paper probes).
        """
        torch = cast("Any", torch_mod)
        ckpt_path = self._resolve_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            self.pretrained = False
            _LOG.warning(
                "WirelessJEPA checkpoint not found (%s); running on a randomly-initialised "
                "backbone. Pre-train one with `scripts/pretrain/wireless_jepa.py` (SLURM: "
                "`slurm/pretrain_wireless_jepa_arm.sh`) — the paper does not publish weights.",
                ckpt_path,
            )
            return

        raw = torch.load(ckpt_path, map_location=self._device)
        state: dict[str, Any] = raw
        if isinstance(raw, dict):
            for key in ("target_encoder_state_dict", "backbone_state_dict", "model_state_dict",
                        "state_dict"):
                if key in raw:
                    state = raw[key]
                    break
        cleaned = {k.replace("_orig_mod.", "").replace("module.", "", 1): v for k, v in
                   state.items()}
        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        if missing:
            raise RuntimeError(
                f"WirelessJEPA checkpoint {ckpt_path} loaded but {len(missing)} backbone "
                f"parameters are MISSING from the state_dict (e.g. {list(missing)[:6]}). The "
                "backbone does not match the pretrained weights — refusing to run a "
                "partially-random encoder as if it were pretrained."
            )
        _LOG.info(
            "WirelessJEPA weights loaded from %s (missing=0, unexpected=%d).", ckpt_path,
            len(unexpected),
        )

    def _resolve_checkpoint(self) -> Path | None:
        """Return the backbone checkpoint path (explicit arg or the cached default)."""
        if self._checkpoint is not None:
            return Path(self._checkpoint).expanduser()
        return backbone_checkpoint_path()

    # -- Model contract -------------------------------------------------------------------------
    def embed(self, x: Batch) -> Tensor:
        """Return ``(B, 1024)`` frozen mean-pooled features for the collated AMC batch ``x``.

        Reads the canonical ``"iq"`` field, unit-max normalises each ``(2, L)`` window, stacks to
        ``(B, 2, L)``, and runs the frozen encoder under ``no_grad`` — one vector per sample, the
        representation the ``linear_probe`` / ``few_shot`` adapters fit a head on.
        """
        self._ensure_loaded()
        torch = require_torch()
        tokens = _iq_batch_to_tensor(x["iq"], torch).to(self._device)
        self._backbone.eval()
        with torch.no_grad():
            return self._backbone(tokens)

    @property
    def n_params(self) -> int:
        """Total encoder parameter count (~335k); 0 until the first :meth:`embed` load."""
        return self._n_params


__all__ = [
    "WirelessJepa",
    "EMBED_DIM",
    "BACKBONE_CHECKPOINT",
    "wireless_jepa_cache_dir",
    "backbone_checkpoint_path",
]
