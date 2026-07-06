"""WP-62 -- LWM-Spectro: the first *public-weights* RF foundation model on the board.

`LWM-Spectro <https://huggingface.co/wi-lab/lwm-spectro>`_ (wi-lab) is a **public, non-gated**
RF foundation model (License: MIT, no non-commercial restriction -- verified upstream): a
12-layer Transformer (``d_model=128``, ``n_heads=8``) pretrained on 128x128 RF spectrograms with a
masked-reconstruction objective, on top of which a Mixture-of-Experts head routes between
WiFi / LTE / 5G protocol experts. This module wraps the **pretraining backbone** (the encoder that
produces the 128-d representation) as an :class:`FoundationModel`, so the AMC board gets an FM row
to compare against the MCLDNN (60.08%) / CLDNN (58.76%) baselines.

Why the encoder and not the MoE head. The board needs (a) a frozen ``embed`` feeding the
``linear_probe`` / ``few_shot`` head and (b) a ``forward`` producing 11-class AMC logits. The
LWM-Spectro *encoder* gives us (a) directly. For (b) we attach a fresh 11-way linear head on top of
the frozen encoder (trained by a ``full_finetune`` loop -- NOT yet implemented; a raw ``forward``
on the untrained head is a plumbing smoke test only, never a publishable score). We deliberately do
NOT vendor the repo's un-versioned research modules
(``mixture.train_embedding_router.MoEPredictor``, ``pretraining.pretrained_model.PretrainedLWM``):
their protocol classes (WiFi/LTE/5G) are not the AMC label set, and importing arbitrary code from
an HF snapshot at eval time is fragile. Instead we reconstruct the LWM encoder here and load the
*real* pretrained weights into it by matching the checkpoint's ``state_dict`` keys.

WHICH checkpoint holds the encoder (verified 2026-07 by inspecting the real tensors). NOT
``checkpoints/checkpoint.pth`` -- that is the ``snr_mobility`` MoE bundle (router + classifier +
expert list, 14 classes), with NO encoder tensors. The real 12-layer LWM encoders are the
per-protocol **expert** files ``experts/{WiFi,LTE,5G}_expert.pth`` (each a 203-tensor state_dict
with ``module.embedding.proj``/``layers.i...``/``norm.alpha`` keys). The experts are fine-tuned from
a shared pretraining base, so any is a reasonable RF encoder; we load one (default ``WiFi``).

ARCHITECTURE FIDELITY (verified against the shipped expert weights). The reconstruction below
mirrors the real module tree so an expert's ``state_dict`` loads by name (keys are ``module.``-
prefixed and stripped on load):

* Every normalisation is a **custom** ``LayerNormalization`` storing ``.alpha`` / ``.bias`` (NOT
  ``nn.LayerNorm``'s ``.weight`` / ``.bias``) -- this is the load-bearing detail: using
  ``nn.LayerNorm`` silently leaves all 25 norm layers at random init.
* ``MultiHeadAttention`` adds its residual **internally** (``residual + linear(attn)``); the FFN
  uses **ReLU** (not GELU); the block is post-norm: ``norm1(mha(x))`` then
  ``norm2(a + ffn(a))``.
* The downstream representation is the **mean over the sequence** of the raw encoder output --
  BEFORE the top-level ``norm`` / ``linear`` (those run only in the masked-reconstruction branch,
  which we never take). ``self.norm`` / ``self.linear`` are still defined so their checkpoint keys
  load cleanly.
* The ``[CLS]`` token is a data-side **constant 0.2 vector** (``utils.make_sample``), not zeros and
  not a learned parameter.

Input adapter (UNVERIFIED preprocessing -- see :func:`_iq_to_lwm_tokens`). AMC samples are RadioML
2016.10a IQ windows of shape ``(2, 128)``; LWM-Spectro consumes a **128x128 SINGLE-CHANNEL
spectrogram** -- the expert weights prove this: ``embedding.proj`` is ``Linear(16, 128)`` and
``decoder_bias`` is ``(16,)``, so a 4x4 patch is ``4*4 = 16`` values (NOT the ``4*4*2 = 32`` a
complex real/imag layout would give). CRITICALLY, upstream ships **no** IQ->spectrogram code: the
128x128 spectrograms are pre-computed externally (config.json ``input_shape=[128,128]``,
``input_dtype='float16'``) and the exact STFT (only "512-FFT" is stated on the card -- hop, window,
and the resize are unpublished). We therefore implement a *best-effort approximation*: STFT
``n_fft=512`` -> **log-magnitude** (dB; the ``mean_db``/``std_db`` upstream names imply
a dB spectrogram) -> resize to 128x128 -> per-sample ``(x-mean)/std`` -> 4x4 ``patch_maker``
(element_length ``16``, ``32*32 = 1024`` patches + 1 CLS = ``max_len=1025``). The token LAYOUT
(patch order + CLS constant 0.2 + 16-wide width + per-sample normalisation) matches the shipped
weights; the STFT front-end is NOT and CANNOT be reproduced from public artifacts. Any resulting FM
score is therefore **provisional / UNVERIFIED** until the upstream spectrogram-generation config is
obtained; :meth:`embed` warns loudly to that effect.

HARD CONSTRAINT: ``import rfbench.models.foundation`` stays dependency-free. ``torch`` is imported
lazily via :func:`~rfbench.models.foundation.base.require_torch` inside the loader/forward/embed;
this module is NOT imported by ``foundation/__init__`` (only an explicit
``import rfbench.models.foundation.lwm_spectro`` registers ``"lwm-spectro"`` in
:data:`rfbench.core.registry.MODELS`). The real weights are fetched by the guarded
:mod:`rfbench.models.foundation._download_lwm_spectro`, never in unit tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel, require_torch

_LOG = logging.getLogger(__name__)

# --- LWM-Spectro encoder hyper-parameters (verified against the SHIPPED expert checkpoints) ------
#: Per-token feature width fed to the encoder: a 4x4 patch of the SINGLE-CHANNEL 128x128
#: spectrogram -> ``4 * 4 = 16`` values per token. Verified from the real expert weights
#: (``embedding.proj.weight`` has shape ``(128, 16)`` and ``decoder_bias`` shape ``(16,)``), which
#: rules out the real/imag-interleaved 32-wide layout: the spectrogram is single-channel.
ELEMENT_LENGTH = 16
#: Transformer hidden width (also the returned embedding dim).
D_MODEL = 128
#: Number of Transformer encoder layers.
N_LAYERS = 12
#: Self-attention heads per layer.
N_HEADS = 8
#: Max sequence length: ``(128/4) * (128/4) + 1`` CLS token ``= 1024 + 1``.
MAX_LEN = 1025
#: Feed-forward hidden width (``4 * d_model``).
D_FF = D_MODEL * 4
#: The AMC closed set (RadioML 2016.10a): 11 modulation classes. NOTE this is a fresh downstream
#: head chosen by rf-benchmark-hub, NOT an upstream dim (upstream has WiFi/LTE/5G experts).
DEFAULT_NUM_CLASSES = 11
#: Spectrogram side length the model consumes (square, 128x128).
SPEC_SIZE = 128
#: Patch side used by the repo's ``patch_maker`` (4x4 patches).
PATCH = 4
#: STFT size on the model card ("128x128 spectrograms from 512-FFT"). UNVERIFIED: upstream ships no
#: STFT code, so hop/window/magnitude are unknown and this front-end cannot be reproduced exactly.
N_FFT = 512
#: The data-side ``[CLS]`` token value (``utils.make_sample`` uses ``np.full(patch_size, 0.2)``).
CLS_VALUE = 0.2
#: Checkpoint keys present in the pretraining .pth but NOT part of the frozen-encoder representation
#: path (the masked-reconstruction decoder head). Allowed to be "unexpected" on load.
_ALLOWED_UNEXPECTED = frozenset({"decoder.weight", "decoder_bias"})


def _build_encoder(nn_mod: ModuleType) -> Tensor:
    """Construct the LWM-Spectro Transformer encoder with the repo's exact submodule names.

    The attribute names mirror upstream ``pretraining/pretrained_model.py`` so the published
    ``checkpoint.pth`` ``state_dict`` keys load by name. ``nn_mod`` is the lazily imported
    ``torch.nn`` (this function is only reached after :func:`require_torch`). Returns an
    ``nn.Module`` (typed :data:`Tensor` == ``Any`` to keep this module torch-free at import).
    """
    nn = cast("Any", nn_mod)

    class _LayerNormalization(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy)
        """The repo's CUSTOM LayerNorm: parameters ``alpha`` / ``bias`` (NOT ``weight`` / ``bias``).

        Upstream every norm (``embedding.norm``, ``layers.i.norm1/norm2``, top-level ``norm``) is
        this class, so the checkpoint stores ``...norm.alpha`` / ``...norm.bias``. Reconstructing
        it with ``nn.LayerNorm`` (``.weight`` / ``.bias``) would leave ALL norm scales at random
        default init -- the single fatal bug this fixes.
        """

        def __init__(self, d_model: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.alpha = nn.Parameter(_ones(d_model))
            self.bias = nn.Parameter(_zeros(d_model))
            self.eps = eps

        def forward(self, x: Tensor) -> Tensor:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias

    class _Embedding(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """Patch projection + learned positional embedding + LayerNormalization (``Embedding``)."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(ELEMENT_LENGTH, D_MODEL)
            self.pos_embed = nn.Embedding(MAX_LEN, D_MODEL)
            self.norm = _LayerNormalization(D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            seq_len = x.size(1)
            pos = self._arange(x, seq_len)
            out = self.proj(x) + self.pos_embed(pos)
            return self.norm(out)

        @staticmethod
        def _arange(x: Tensor, seq_len: int) -> Tensor:
            import torch as _torch

            pos = _torch.arange(seq_len, dtype=_torch.long, device=x.device)
            return pos.unsqueeze(0).expand(x.size(0), seq_len)

    class _MultiHeadAttention(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy)
        """Self-attention: repo ``W_Q``/``W_K``/``W_V``/``linear`` names + internal residual.

        Separate Q/K/V projections (not ``nn.MultiheadAttention``'s fused ``in_proj_weight``) so the
        checkpoint's ``enc_self_attn.W_Q``/``W_K``/``W_V``/``linear`` load by name. Upstream returns
        ``residual + linear(attn)`` (the residual add lives INSIDE attention).
        """

        def __init__(self) -> None:
            super().__init__()
            self.W_Q = nn.Linear(D_MODEL, D_MODEL)
            self.W_K = nn.Linear(D_MODEL, D_MODEL)
            self.W_V = nn.Linear(D_MODEL, D_MODEL)
            self.linear = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            import torch as _torch

            residual = x
            b, seq, _ = x.shape
            head_dim = D_MODEL // N_HEADS
            q = self.W_Q(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            k = self.W_K(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            v = self.W_V(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            attn = _torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(b, seq, D_MODEL)
            return residual + self.linear(attn)

    class _FeedForward(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """Position-wise FFN (``pos_ffn``): ``fc1`` -> ReLU -> ``fc2`` (upstream ``F.relu``)."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(D_MODEL, D_FF)
            self.fc2 = nn.Linear(D_FF, D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            import torch as _torch

            return self.fc2(_torch.nn.functional.relu(self.fc1(x)))

    class _EncoderLayer(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """One post-norm block: ``norm1(mha(x))`` then ``norm2(a + ffn(a))`` (MHA holds res)."""

        def __init__(self) -> None:
            super().__init__()
            self.enc_self_attn = _MultiHeadAttention()
            self.pos_ffn = _FeedForward()
            self.norm1 = _LayerNormalization(D_MODEL)
            self.norm2 = _LayerNormalization(D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            attn_outputs = self.norm1(self.enc_self_attn(x))  # MHA adds its residual internally
            return self.norm2(attn_outputs + self.pos_ffn(attn_outputs))

    class _LWMEncoder(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """The LWM-Spectro encoder: ``embedding`` + ``layers`` (+ unused ``norm``/``linear``).

        ``forward`` returns the RAW sequence output ``(B, L, D_MODEL)`` with NO top-level
        ``norm``/``linear`` applied (upstream applies those only in the masked-reconstruction
        branch). :meth:`encode` mean-pools that sequence into the ``(B, D_MODEL)`` frozen
        representation the board reads. ``self.norm``/``self.linear`` exist ONLY so their real
        checkpoint keys load; they are deliberately not on the representation path.
        """

        def __init__(self) -> None:
            super().__init__()
            self.embedding = _Embedding()
            self.layers = nn.ModuleList([_EncoderLayer() for _ in range(N_LAYERS)])
            # Present in the checkpoint (masked-recon LM head); kept so their keys load. NOT applied
            # to the representation -- upstream uses them only when ``masked_pos`` is given.
            self.norm = _LayerNormalization(D_MODEL)
            self.linear = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, tokens: Tensor) -> Tensor:
            out = self.embedding(tokens)
            for layer in self.layers:
                out = layer(out)
            return out  # raw sequence; NO top-level norm/linear (recon-only upstream)

        def encode(self, tokens: Tensor) -> Tensor:
            return self.forward(tokens).mean(
                dim=1
            )  # mean-pool over the sequence (upstream pooling)

    return _LWMEncoder()


def _ones(dim: int) -> Tensor:
    import torch as _torch

    return _torch.ones(dim)


def _zeros(dim: int) -> Tensor:
    import torch as _torch

    return _torch.zeros(dim)


def _iq_to_lwm_tokens(iq_batch: object, torch_mod: ModuleType) -> Tensor:
    """Adapt a collated AMC ``x["iq"]`` list into LWM-Spectro token tensors ``(B, 1025, 16)``.

    Pipeline (token LAYOUT verified against the shipped expert weights; STFT front-end is the
    UNVERIFIED approximation -- see the module docstring). For each ``(2, 128)`` IQ window: form the
    complex signal ``I + jQ``; STFT ``n_fft=512``; take **log-magnitude** (single channel --
    ``embedding.proj`` = ``Linear(16, 128)`` and ``mean_db``/``std_db`` upstream names imply
    a single dB spectrogram, NOT complex real/imag); resize to ``(128, 128)``; **per-sample**
    normalise ``(x - mean) / std``; 4x4 ``patch_maker`` -> ``(1024, 16)``; prepend the constant-0.2
    ``[CLS]`` token -> ``(1025, 16)``. Returns a ``float32`` batch tensor.
    """
    torch = cast("Any", torch_mod)
    if isinstance(iq_batch, dict) or not isinstance(iq_batch, Iterable):
        raise TypeError("LWM-Spectro adapter expects x['iq'] as an iterable of IQ windows")
    samples: list[object] = list(iq_batch)
    if not samples:
        raise ValueError("LWM-Spectro adapter expected a non-empty list of IQ samples")

    tokens: list[Any] = []
    window = torch.hann_window(N_FFT)
    for sample in samples:
        iq = torch.as_tensor(sample, dtype=torch.float32)
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValueError(f"expected an IQ window of shape (2, L); got {tuple(iq.shape)}")
        complex_sig = torch.complex(iq[0], iq[1])  # (L,)
        # ``center=True`` reflect-pads by ``n_fft // 2``, impossible for a 128-sample window at
        # n_fft=512 (reflect needs pad < length); AMC windows are short, so zero-pad instead.
        # onesided is forced False for a complex input -> F = n_fft = 512 (two-sided).
        spec = torch.stft(
            complex_sig,
            n_fft=N_FFT,
            hop_length=1,
            win_length=N_FFT,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )  # (F=512, T) complex
        db = 20.0 * torch.log10(spec.abs() + 1e-8)  # (F, T) log-magnitude, single channel
        img = _resize_to_square(db, torch)  # (128, 128) real
        img = _normalise(img, torch)  # per-sample (x-mean)/std
        tokens.append(_patch_maker(img, torch))  # (1024, 16)

    batch = torch.stack(tokens, dim=0)  # (B, 1024, 16)
    cls = torch.full((batch.size(0), 1, ELEMENT_LENGTH), CLS_VALUE, dtype=batch.dtype)
    return torch.cat([cls, batch], dim=1)  # (B, 1025, 16)


def _resize_to_square(plane: Tensor, torch_mod: ModuleType) -> Tensor:
    """Bilinearly resize a real spectrogram ``(F, T)`` to ``(SPEC_SIZE, SPEC_SIZE)``.

    A 512-FFT of a 128-sample AMC window gives ``(512, 129)``, which does not tile into a 128x128
    grid, so we resize (bilinear) to exactly ``(128, 128)``. Part of the UNVERIFIED STFT front-end
    (the real pretraining resize is unpublished); always well-formed for any STFT shape.
    """
    torch = cast("Any", torch_mod)
    resized = torch.nn.functional.interpolate(
        plane[None, None], size=(SPEC_SIZE, SPEC_SIZE), mode="bilinear", align_corners=False
    )
    return resized[0, 0]


def _normalise(img: Tensor, torch_mod: ModuleType) -> Tensor:
    """Per-sample normalisation ``(x - mean) / std`` over the whole ``(128, 128)`` spectrogram.

    Matches upstream ``tokenizer_train`` per-sample norm (``mean_db``/``std_db`` over the
    whole array). ``torch_mod`` is unused (kept for signature symmetry with the sibling helpers).
    """
    del torch_mod  # tensor methods below need no framework handle
    mean = img.mean()
    std = img.std().clamp_min(1e-6)
    return (img - mean) / std


def _patch_maker(img: Tensor, torch_mod: ModuleType) -> Tensor:
    """Split a single-channel ``(128, 128)`` spectrogram into ``(1024, 16)`` 4x4 patch tokens.

    Mirrors upstream ``patch_maker(patch_rows=4, patch_cols=4)``: reshape ``(128, 128) ->
    (32, 4, 32, 4)``, ``transpose(0, 2, 1, 3)``, flatten to ``(1024, 16)`` (C-order) -> ``4*4 = 16``
    = :data:`ELEMENT_LENGTH` per token, ``(128/4)*(128/4) = 1024`` tokens. The 16-wide token matches
    the shipped ``embedding.proj`` = ``Linear(16, 128)``.
    """
    torch = cast("Any", torch_mod)
    h, w = img.shape  # (128, 128)
    if h != SPEC_SIZE or w != SPEC_SIZE:
        raise ValueError(f"expected a spectrogram of shape (128, 128); got {(h, w)}")
    n_r, n_c = h // PATCH, w // PATCH  # 32, 32
    grid = img.contiguous().reshape(n_r, PATCH, n_c, PATCH)
    grid = grid.permute(0, 2, 1, 3).reshape(n_r * n_c, PATCH * PATCH)
    return grid.contiguous().to(torch.float32)


@register_model("lwm-spectro")
class LwmSpectroModel(FoundationModel):
    """The LWM-Spectro RF foundation model as a board ``Model`` (registered ``"lwm-spectro"``).

    Wraps the pretrained 12-layer LWM-Spectro encoder (128-d mean-pooled representation) behind the
    frozen :class:`~rfbench.core.model.Model` contract:

    * :meth:`embed` -> ``(B, 128)`` frozen features for ``linear_probe`` / ``few_shot``;
    * :meth:`forward` -> ``(B, 11)`` AMC logits from a fresh linear head on the frozen encoder
      (the head is UNTRAINED unless a ``full_finetune`` loop has fitted it -- see the module
      docstring; a raw ``forward`` is a smoke test, not a publishable score);
    * :attr:`n_params` -> encoder + head parameter count; :attr:`family` -> ``"foundation"``.

    Constructed with no required args (``MODELS.get("lwm-spectro")()`` on the registry path).
    Construction is cheap: torch + weights load lazily on first :meth:`embed` / :meth:`forward`.
    The real weights come from :func:`rfbench.models.foundation._download_lwm_spectro.
    download_lwm_spectro`; when ``checkpoint=None`` the wrapper resolves the cached backbone under
    ``$RFBENCH_CACHE/lwm-spectro/checkpoints/checkpoint.pth``. If the checkpoint is absent,
    :meth:`embed` / :meth:`forward` still run on the randomly-initialised encoder (a plumbing smoke
    test) and set :attr:`pretrained` to ``False`` -- they never silently claim pretrained features.
    If a checkpoint IS present but does not populate the encoder (key mismatch), the loader RAISES
    rather than run a partially-random backbone.
    """

    def __init__(
        self,
        *,
        name: str = "lwm-spectro",
        num_classes: int = DEFAULT_NUM_CLASSES,
        checkpoint: str | Path | None = None,
        expert: str = "WiFi",
        device: str | None = None,
    ) -> None:
        """Wrap the LWM-Spectro encoder under ``name``; keep construction torch-free and cheap.

        ``expert`` selects which protocol expert's LWM encoder is the frozen backbone
        (``WiFi`` / ``LTE`` / ``5G``); the three are fine-tuned from a shared pretraining base, so
        any is a reasonable generic RF encoder. ``checkpoint`` overrides the resolved expert path.
        """
        super().__init__(
            name,
            n_params=0,  # set once the backbone is loaded (see _ensure_loaded)
            backbone=f"wi-lab/lwm-spectro:experts/{expert}_expert.pth",
            pretrained=True,
        )
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = num_classes
        self._checkpoint = checkpoint
        self._expert = expert
        self._device_str = device
        self._encoder: Any = None
        self._head: Any = None
        self._device: Any = None
        self._warned_unverified = False

    # -- lazy load ------------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Build the encoder + head and load real weights on first use (the only heavy step)."""
        if self._encoder is not None:
            return
        torch = require_torch()
        from torch import nn

        resolved = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved)

        encoder = _build_encoder(nn).to(self._device).eval()
        self._load_weights(encoder, torch)
        head = nn.Linear(D_MODEL, self.num_classes).to(self._device)

        self._encoder = encoder
        self._head = head
        self._n_params = sum(p.numel() for p in encoder.parameters()) + sum(
            p.numel() for p in head.parameters()
        )

    def _load_weights(self, encoder: Tensor, torch_mod: ModuleType) -> None:
        """Load the real pretrained ``state_dict`` into ``encoder`` (``strict=False`` + guard).

        Resolves the checkpoint from ``self._checkpoint`` or the cached expert encoder
        ``$RFBENCH_CACHE/lwm-spectro/experts/<expert>_expert.pth`` (keys ``module.``-prefixed and
        stripped below). When absent the encoder keeps its random init and :attr:`pretrained` flips
        to ``False``. When present, any MISSING encoder key
        (i.e. the reconstruction does not match the weights) RAISES -- we refuse to run a
        partially-random backbone and report it as pretrained.
        """
        torch = cast("Any", torch_mod)
        ckpt_path = self._resolve_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            self.pretrained = False
            _LOG.warning(
                "LWM-Spectro checkpoint not found (%s); running on a randomly-initialised encoder. "
                "Fetch real weights with "
                "`python -m rfbench.models.foundation._download_lwm_spectro`.",
                ckpt_path,
            )
            return

        raw = torch.load(ckpt_path, map_location=self._device)
        state: dict[str, Any] = raw
        if isinstance(raw, dict):
            for key in ("model_state_dict", "state_dict"):
                if key in raw:
                    state = raw[key]
                    break
        cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
        if missing:
            raise RuntimeError(
                f"LWM-Spectro checkpoint {ckpt_path} loaded but {len(missing)} encoder parameters "
                f"are MISSING from the state_dict (e.g. {list(missing)[:6]}). The reconstructed "
                "architecture does not match the pretrained weights -- refusing to run a "
                "partially-random encoder as if it were pretrained."
            )
        unknown = sorted(set(map(str, unexpected)) - _ALLOWED_UNEXPECTED)
        _LOG.info(
            "LWM-Spectro weights loaded from %s (missing=0, unexpected=%d; unknown-unexpected=%s).",
            ckpt_path,
            len(unexpected),
            unknown or "none",
        )

    def _resolve_checkpoint(self) -> Path | None:
        """Return the backbone checkpoint path (explicit arg or the cached default)."""
        if self._checkpoint is not None:
            return Path(self._checkpoint).expanduser()
        from rfbench.models.foundation._download_lwm_spectro import backbone_checkpoint_path

        return backbone_checkpoint_path(expert=self._expert)

    def _warn_unverified_preprocessing(self) -> None:
        """Emit a one-time loud warning: IQ->STFT front-end is UNVERIFIED (provisional score)."""
        if self._warned_unverified:
            return
        self._warned_unverified = True
        _LOG.warning(
            "LWM-Spectro IQ->STFT preprocessing is UNVERIFIED: upstream ships no IQ->spectrogram "
            "code (spectrograms are pre-computed externally; exact 512-FFT hop/window/magnitude "
            "are unpublished). Tokenisation (interleave/patch/CLS/normalisation) is faithful, but "
            "the STFT front-end is a best-effort approximation -- any resulting FM score is "
            "PROVISIONAL and must not be published as a faithful LWM-Spectro figure until the "
            "upstream spectrogram-generation config is confirmed."
        )

    # -- Model contract -------------------------------------------------------------------------
    def embed(self, x: Batch) -> Tensor:
        """Return ``(B, 128)`` frozen mean-pooled features for the collated AMC batch ``x``."""
        self._ensure_loaded()
        self._warn_unverified_preprocessing()
        torch = require_torch()
        tokens = _iq_to_lwm_tokens(x["iq"], torch).to(self._device)
        self._encoder.eval()
        with torch.no_grad():
            return self._encoder.encode(tokens)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` AMC logits: the head over the frozen encoder features.

        The head is UNTRAINED unless a ``full_finetune`` loop has fitted it -- a raw ``forward``
        is a plumbing smoke test, never a publishable ``full_finetune`` score.
        """
        self._ensure_loaded()
        torch = require_torch()
        with torch.no_grad():
            features = self.embed(x)
        return self._head(features)

    @property
    def n_params(self) -> int:
        """Total parameter count (encoder + 11-class head); 0 until first load."""
        return self._n_params


__all__ = [
    "LwmSpectroModel",
    "ELEMENT_LENGTH",
    "D_MODEL",
    "N_LAYERS",
    "N_HEADS",
    "MAX_LEN",
    "DEFAULT_NUM_CLASSES",
    "SPEC_SIZE",
    "N_FFT",
    "CLS_VALUE",
]
