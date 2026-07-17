"""IQFM: a raw-IQ contrastive-SSL foundation model on the board (registered ``"iqfm-base"``).

STATUS (2026-07): faithful reproduction is **PAUSED** — IQFM publishes no weights and pre-trains on
the authors' proprietary OTA MIMO testbed (which we do NOT have), so an in-repo retrain can only be
a **HOMEMADE, in-distribution** model, not the paper's, and **no board row is committed** for it
(a prior 48.87% `self_reported` row was pulled from the board). It must **never** be conflated with
the paper's OOD **38.1%**, which lives only as a hand-curated `from_paper` row
(`.../amc/iqfm_paper.json`). The wrapper + shared backbone are kept for future use; further
retraining is on hold.

`IQFM <https://arxiv.org/abs/2506.06718>`_ (Mashaal & Abou-Zeid, arXiv:2506.06718v2, 2025,
**CC-BY 4.0**) is a small RF foundation model: a **ShuffleNetV2-x0.5** backbone (~341k params)
over raw IQ, pre-trained with **contrastive SSL** (SimCLR / InfoNCE) and normalised at the input
with per-sample **unit-max** scaling ``iq / max(|iq|)``. This module wraps that backbone as a
frozen :class:`~rfbench.models.foundation.base.FoundationModel`, so the AMC board gets a raw-IQ
SSL-FM row to sit alongside the spectrogram FM (LWM-Spectro) and the supervised baselines.

WEIGHTS PROVENANCE — read before trusting any score. IQFM's **weights are NOT published** (the
paper releases the recipe and the architecture, not a checkpoint). This wrapper therefore never
claims IQFM's headline number: the paper's **38.1% on RML2016.10a** is a *linear-probe, 50
samples/class, OUT-OF-DISTRIBUTION* result — the encoder was pre-trained on the authors' own OTA
MIMO testbed (which we do NOT have) and probed on RadioML. We reproduce only the *architecture and
recipe*: our backbone is (re-)pre-trained IN-REPO with SimCLR on the RadioML 2016.10a **train**
split (delabelised), which is a **different, in-distribution** setting — so a score from these
weights is **ours**, not the paper's, and must be labelled as such on the board
(``model.notes`` / the result row), never presented as the 38.1% figure.

If no checkpoint is available the wrapper still runs on a randomly-initialised backbone (a
plumbing smoke test) and honestly sets :attr:`pretrained` to ``False``; if a checkpoint IS present
but does not populate the backbone (key mismatch) the loader RAISES rather than silently score a
partially-random encoder as "pretrained" — the same honesty guard the LWM-Spectro wrapper uses.

HARD CONSTRAINT: ``import rfbench.models.foundation`` stays dependency-free. ``torch`` is imported
lazily via :func:`~rfbench.models.foundation.base.require_torch` inside the loader/``embed``; the
backbone comes from the reusable :func:`rfbench.models.foundation.shufflenet1d.build_shufflenet1d`.
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

#: Frozen embedding width (the ShuffleNetV2-x0.5 conv5 output, mean-pooled over time).
EMBED_DIM = EMBED_DIM_X0_5
#: Default cached checkpoint filename for our in-repo SimCLR-pre-trained backbone.
BACKBONE_CHECKPOINT = "iqfm_shufflenet1d_simclr.pth"
#: Small floor so unit-max normalisation never divides by zero on an all-zero window.
_MAX_EPS = 1e-8


def iqfm_cache_dir() -> Path:
    """Return ``$RFBENCH_CACHE/iqfm`` (the backbone-checkpoint cache; ``$RFBENCH_CACHE`` or CWD)."""
    root = Path(os.environ.get("RFBENCH_CACHE", ".")).expanduser()
    return root / "iqfm"


def backbone_checkpoint_path() -> Path:
    """Return the default cached backbone-checkpoint path ``$RFBENCH_CACHE/iqfm/<file>``."""
    return iqfm_cache_dir() / BACKBONE_CHECKPOINT


def _unit_max_normalise(iq: Tensor, torch_mod: ModuleType) -> Tensor:
    """Per-sample unit-max normalisation ``iq / max(|iq|)`` (IQFM's input normalisation).

    ``iq`` is a single ``(2, L)`` window (row 0 = I, row 1 = Q). The scale is the maximum complex
    magnitude ``max_t sqrt(I_t^2 + Q_t^2)`` over the window (floored by :data:`_MAX_EPS`), so both
    channels are divided by the same per-sample constant — matching the paper's ``iq/max(|iq|)``.
    """
    torch = cast("Any", torch_mod)
    magnitude = torch.sqrt(iq[0] ** 2 + iq[1] ** 2)  # (L,) complex magnitude
    scale = magnitude.max().clamp_min(_MAX_EPS)
    return iq / scale


def _iq_batch_to_tensor(iq_batch: object, torch_mod: ModuleType) -> Tensor:
    """Collate ``x["iq"]`` (a list of ``(2, L)`` windows) into a unit-max-normalised ``(B, 2, L)``.

    Each window is coerced to a ``float32`` tensor, validated as ``(2, L)``, and unit-max
    normalised per sample before stacking. Accepts the nested-list windows the dependency-free
    synthetic AMC path yields and the numpy arrays the cluster path yields (``torch.as_tensor``
    handles both).
    """
    torch = cast("Any", torch_mod)
    if isinstance(iq_batch, dict) or not isinstance(iq_batch, Iterable):
        raise TypeError("IQFM adapter expects x['iq'] as an iterable of IQ windows")
    samples: list[object] = list(iq_batch)
    if not samples:
        raise ValueError("IQFM adapter expected a non-empty list of IQ samples")

    windows: list[Any] = []
    for sample in samples:
        iq = torch.as_tensor(sample, dtype=torch.float32)
        if iq.ndim != 2 or iq.shape[0] != IN_CHANNELS:
            raise ValueError(
                f"expected an IQ window of shape ({IN_CHANNELS}, L); got {tuple(iq.shape)}"
            )
        windows.append(_unit_max_normalise(iq, torch))
    return torch.stack(windows, dim=0)  # (B, 2, L)


@register_model("iqfm-base")
class IqfmBase(FoundationModel):
    """The IQFM raw-IQ SSL foundation model as a board ``Model`` (registered ``"iqfm-base"``).

    Wraps the frozen ShuffleNetV2-x0.5 raw-IQ backbone behind the
    :class:`~rfbench.core.model.Model` contract:

    * :meth:`embed` -> ``(B, 1024)`` frozen mean-pooled features (unit-max normalised input) for
      the ``linear_probe`` / ``few_shot`` regimes;
    * :meth:`forward` -> inherited from :class:`FoundationModel` (falls back to :meth:`embed`) —
      IQFM ships **no task head**, so we only ever probe it; ``from_scratch`` / ``full_finetune``
      have no meaningful IQFM row and are refused at the SLURM layer;
    * :attr:`n_params` -> backbone parameter count (~341k once loaded); :attr:`family` ->
      ``"foundation"``.

    Constructed with no required args (``MODELS.get("iqfm-base")()`` on the registry path).
    Construction is cheap: torch + the backbone build lazily on first :meth:`embed`. When
    ``checkpoint=None`` the wrapper resolves the cached backbone at
    ``$RFBENCH_CACHE/iqfm/iqfm_shufflenet1d_simclr.pth``; if that is absent it runs on a
    randomly-initialised backbone and sets :attr:`pretrained` to ``False`` (never silently claims
    pretrained features). A checkpoint present but not matching the backbone RAISES.
    """

    def __init__(
        self,
        *,
        name: str = "iqfm-base",
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """Wrap the IQFM backbone under ``name``; keep construction torch-free and cheap.

        ``checkpoint`` overrides the resolved cache path
        (``$RFBENCH_CACHE/iqfm/iqfm_shufflenet1d_simclr.pth``); ``device`` pins the run device
        (defaults to CUDA when available, else CPU). Weights load lazily on first :meth:`embed`.
        """
        super().__init__(
            name,
            n_params=0,  # set once the backbone is loaded (see _ensure_loaded)
            backbone="shufflenet_v2_x0_5_1d (IQFM recipe; weights unpublished, retrained in-repo)",
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
        """Load our SimCLR-pre-trained ``state_dict`` into ``backbone`` (``strict=False`` + guard).

        Resolves the checkpoint from ``self._checkpoint`` or the cached default. When absent the
        backbone keeps its random init and :attr:`pretrained` flips to ``False``. When present, any
        MISSING backbone key (the state_dict does not match the reconstruction) RAISES — we refuse
        to run a partially-random backbone and report it as pretrained.
        """
        torch = cast("Any", torch_mod)
        ckpt_path = self._resolve_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            self.pretrained = False
            _LOG.warning(
                "IQFM checkpoint not found (%s); running on a randomly-initialised backbone. "
                "Pre-train one with `scripts/pretrain/iqfm_simclr.py` (SLURM: "
                "`slurm/pretrain_iqfm_arm.sh`) — the paper does not publish weights.",
                ckpt_path,
            )
            return

        raw = torch.load(ckpt_path, map_location=self._device)
        state: dict[str, Any] = raw
        if isinstance(raw, dict):
            for key in ("model_state_dict", "backbone_state_dict", "state_dict"):
                if key in raw:
                    state = raw[key]
                    break
        cleaned = {
            k.replace("_orig_mod.", "").replace("module.", "", 1): v for k, v in state.items()
        }
        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        if missing:
            raise RuntimeError(
                f"IQFM checkpoint {ckpt_path} loaded but {len(missing)} backbone parameters are "
                f"MISSING from the state_dict (e.g. {list(missing)[:6]}). The backbone does not "
                "match the pretrained weights — refusing to run a partially-random encoder as if "
                "it were pretrained."
            )
        _LOG.info(
            "IQFM weights loaded from %s (missing=0, unexpected=%d).", ckpt_path, len(unexpected)
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
        ``(B, 2, L)``, and runs the frozen backbone under ``no_grad`` — one vector per sample, the
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
        """Total backbone parameter count (~341k); 0 until the first :meth:`embed` load."""
        return self._n_params


__all__ = [
    "IqfmBase",
    "EMBED_DIM",
    "BACKBONE_CHECKPOINT",
    "iqfm_cache_dir",
    "backbone_checkpoint_path",
]
