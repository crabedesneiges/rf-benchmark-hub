"""WP-62 -- LWM-Spectro: the first *public-weights* RF foundation model on the board.

`LWM-Spectro <https://huggingface.co/wi-lab/lwm-spectro>`_ (wi-lab) is a **public, non-gated**
RF foundation model: a 12-layer Transformer (``d_model=128``, ``n_heads=8``) pretrained on RF
spectrograms with a masked-reconstruction objective, on top of which a Mixture-of-Experts head
routes between WiFi / LTE / 5G protocol experts. This module wraps the **pretraining backbone**
(the encoder that produces the 128-d representation) as an :class:`FoundationModel`, so the AMC
board gets an FM row to compare against the MCLDNN (60.08%) / CLDNN (58.76%) baselines.

Why the encoder and not the MoE head. The board needs (a) a frozen ``embed`` feeding the
``linear_probe`` / ``few_shot`` head and (b) a ``forward`` producing 11-class AMC logits. The
LWM-Spectro *encoder* gives us (a) directly -- the CLS-token output is the 128-d frozen
representation. For (b) we attach a fresh 11-way linear head on top of the frozen encoder
(trained by the ``full_finetune`` / ``from_scratch`` regime loops, or randomly-initialised for a
plumbing smoke test). We deliberately do NOT vendor the repo's un-versioned research modules
(``mixture.train_embedding_router.MoEPredictor``, ``pretraining.pretrained_model.PretrainedLWM``):
their protocol classes (WiFi/LTE/5G) are not the AMC label set, and importing arbitrary code from
an HF snapshot at eval time is fragile. Instead we reconstruct the **exact** LWM encoder here (a
stable, well-known architecture) and load the *real* pretrained weights into it by matching the
checkpoint's ``state_dict`` keys.

Input adapter (documented honestly -- see :func:`_iq_to_lwm_tokens`). AMC samples are RadioML
2016.10a IQ windows of shape ``(2, 128)``; LWM-Spectro consumes a **128x128 complex spectrogram**,
tokenised into ``element_length=32`` patches. The model card documents only ``n_fft=512`` ("128x128
spectrograms generated from 512-FFT"); the exact hop/window and the pretraining normalisation mode
are NOT published. We therefore implement a *documented approximation*: STFT with ``n_fft=512`` and
``hop=1`` over the length-128 window (zero-padded), take the 128 lowest frequency bins, crop/interp
to 128 time columns, then the repo's real/imag-interleave + 4x4 ``patch_maker`` (element_length
``4*4*2 = 32``, sequence ``32*32 = 1024`` patches + 1 CLS = ``max_len=1025``). This reproduces the
model's *interface* exactly; it does not claim to reproduce the pretraining STFT distribution
bit-for-bit (that would need the unpublished preprocessing config). This is the single honest
adaptation caveat, and it is where the FM-vs-baseline gap should be read with care.

HARD CONSTRAINT: ``import rfbench.models.foundation`` stays dependency-free. ``torch`` is imported
lazily via :func:`~rfbench.models.foundation.base.require_torch` inside the loader/forward/embed;
this module is NOT imported by ``foundation/__init__`` (only an explicit
``import rfbench.models.foundation.lwm_spectro`` registers ``"lwm-spectro"`` in
:data:`rfbench.core.registry.MODELS`). The real weights are fetched by the guarded
:mod:`rfbench.models.foundation._download_lwm_spectro`, never in unit tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel, require_torch

# --- LWM-Spectro encoder hyper-parameters (from the repo's pretraining config) -----------------
#: Per-token feature width fed to the encoder: a 4x4 patch of the real/imag-interleaved
#: spectrogram -> ``4 * 4 * 2 = 32`` values per token.
ELEMENT_LENGTH = 32
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
#: The AMC closed set (RadioML 2016.10a): 11 modulation classes.
DEFAULT_NUM_CLASSES = 11
#: Spectrogram side length the model consumes (square, 128x128).
SPEC_SIZE = 128
#: Patch side used by the repo's ``patch_maker`` (4x4 patches).
PATCH = 4
#: STFT size documented on the model card ("128x128 spectrograms from 512-FFT").
N_FFT = 512


def _build_encoder(nn_mod: ModuleType) -> Tensor:
    """Construct the LWM-Spectro Transformer encoder with the repo's submodule names.

    The attribute names (``embedding.proj``/``pos_embed``/``norm``, ``layers[i].enc_self_attn.
    W_Q``/``W_K``/``W_V``/``linear``, ``layers[i].pos_ffn.fc1``/``fc2``, ``layers[i].norm1``/
    ``norm2``, top-level ``norm``/``linear``) mirror ``pretraining/pretrained_model.py`` so the
    published ``checkpoint.pth`` ``state_dict`` keys load by name. ``nn_mod`` is the lazily
    imported ``torch.nn`` (this function is only reached after :func:`require_torch`). Returns
    an ``nn.Module`` (typed :data:`Tensor` == ``Any`` to keep this module torch-free at import).
    """
    nn = cast("Any", nn_mod)

    class _Embedding(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """Patch projection + learned positional embedding + LayerNorm (LWM ``Embedding``)."""

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(ELEMENT_LENGTH, D_MODEL)
            self.pos_embed = nn.Embedding(MAX_LEN, D_MODEL)
            self.norm = nn.LayerNorm(D_MODEL)

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
        """Self-attention with the repo's ``W_Q``/``W_K``/``W_V``/``linear`` submodule names.

        Uses SEPARATE Q/K/V projections (not ``nn.MultiheadAttention``'s fused
        ``in_proj_weight``) so the published checkpoint's ``enc_self_attn.W_Q``/``W_K``/``W_V``/
        ``linear`` weights load by name under ``strict=False``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.W_Q = nn.Linear(D_MODEL, D_MODEL)
            self.W_K = nn.Linear(D_MODEL, D_MODEL)
            self.W_V = nn.Linear(D_MODEL, D_MODEL)
            self.linear = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            import torch as _torch

            b, seq, _ = x.shape
            head_dim = D_MODEL // N_HEADS
            q = self.W_Q(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            k = self.W_K(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            v = self.W_V(x).view(b, seq, N_HEADS, head_dim).transpose(1, 2)
            attn = _torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(b, seq, D_MODEL)
            return self.linear(attn)

    class _EncoderLayer(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """One pre-norm Transformer block: MHSA (``enc_self_attn``) + FFN (``pos_ffn``)."""

        def __init__(self) -> None:
            super().__init__()
            self.enc_self_attn = _MultiHeadAttention()
            self.pos_ffn = _FeedForward()
            self.norm1 = nn.LayerNorm(D_MODEL)
            self.norm2 = nn.LayerNorm(D_MODEL)

        def forward(self, x: Tensor) -> Tensor:
            attn_out = self.enc_self_attn(x)
            x = self.norm1(x + attn_out)
            return self.norm2(x + self.pos_ffn(x))

    class _FeedForward(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """Position-wise FFN (``pos_ffn``): ``fc1`` -> GELU -> ``fc2``."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(D_MODEL, D_FF)
            self.fc2 = nn.Linear(D_FF, D_MODEL)
            self.act = nn.GELU()

        def forward(self, x: Tensor) -> Tensor:
            return self.fc2(self.act(self.fc1(x)))

    class _LWMEncoder(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """The LWM-Spectro encoder: ``embedding`` + ``layers`` + final ``norm``.

        ``forward`` returns the full sequence output ``(B, L, D_MODEL)``; :meth:`encode`
        returns the CLS-token vector ``(B, D_MODEL)`` used as the frozen representation.
        """

        def __init__(self) -> None:
            super().__init__()
            self.embedding = _Embedding()
            self.layers = nn.ModuleList([_EncoderLayer() for _ in range(N_LAYERS)])
            self.norm = nn.LayerNorm(D_MODEL)
            # Present in the pretraining checkpoint (projection head); kept so its keys load.
            self.linear = nn.Linear(D_MODEL, D_MODEL)

        def forward(self, tokens: Tensor) -> Tensor:
            out = self.embedding(tokens)
            for layer in self.layers:
                out = layer(out)
            return self.norm(out)

        def encode(self, tokens: Tensor) -> Tensor:
            return self.forward(tokens)[:, 0]  # CLS token

    return _LWMEncoder()


def _iq_to_lwm_tokens(iq_batch: object, torch_mod: ModuleType) -> Tensor:
    """Adapt a collated AMC ``x["iq"]`` list into LWM-Spectro token tensors ``(B, 1025, 32)``.

    Pipeline (documented approximation -- see the module docstring). For each ``(2, 128)`` IQ
    window: form the complex signal ``I + jQ``; STFT with ``n_fft=512`` (Hann window, ``hop=1``,
    zero-padded, ``center=True``) -> take the 128 lowest freq bins and 128 time columns -> a
    128x128 **complex** spectrogram; per-sample normalise the magnitude; interleave real/imag
    along the width (``(128, 256)``); 4x4 ``patch_maker`` (element_length ``4*4*2=32``, 1024
    patches); prepend a zero CLS token -> ``(1025, 32)``. Returns a ``float32`` batch tensor.
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
        # ``center=True`` reflect-pads by ``n_fft // 2``, which is impossible for a 128-sample
        # window at n_fft=512 (reflect needs pad < length). AMC windows are short, so we
        # zero-pad instead (``pad_mode="constant"``) to keep the card's n_fft=512 intent.
        spec = torch.stft(
            complex_sig,
            n_fft=N_FFT,
            hop_length=1,
            win_length=N_FFT,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )  # (F, T), F = n_fft for a complex (two-sided) input
        spec = _resize_to_square(spec, torch)  # (128, 128) complex
        spec = _normalise_spec(spec, torch)
        interleaved = _interleave_real_imag(spec, torch)  # (128, 256)
        tokens.append(_patch_maker(interleaved, torch))  # (1024, 32)

    batch = torch.stack(tokens, dim=0)  # (B, 1024, 32)
    cls = torch.zeros(batch.size(0), 1, ELEMENT_LENGTH, dtype=batch.dtype)
    return torch.cat([cls, batch], dim=1)  # (B, 1025, 32)


def _resize_to_square(spec: Tensor, torch_mod: ModuleType) -> Tensor:
    """Bilinearly resize an STFT ``(F, T)`` to a ``(SPEC_SIZE, SPEC_SIZE)`` complex spectrogram.

    A 512-FFT of a 128-sample AMC window gives ``(512, 129)``, which does not tile into a
    128x128 grid. Rather than fragile axis-specific crop/pad we resize the real and imaginary
    parts independently (bilinear) to exactly ``(128, 128)`` -- a documented approximation of
    the pretraining resize (whose exact interpolation mode is unpublished), always well-formed
    for any STFT shape.
    """
    torch = cast("Any", torch_mod)
    fn = torch.nn.functional

    def _resize(plane: Tensor) -> Tensor:
        resized = fn.interpolate(
            plane[None, None], size=(SPEC_SIZE, SPEC_SIZE), mode="bilinear", align_corners=False
        )
        return resized[0, 0]

    return torch.complex(_resize(spec.real), _resize(spec.imag))


def _normalise_spec(spec: Tensor, torch_mod: ModuleType) -> Tensor:
    """Per-sample magnitude normalisation to zero-mean/unit-std (the card's 'internal norm')."""
    torch = cast("Any", torch_mod)
    mag = spec.abs()
    mean = mag.mean()
    std = mag.std().clamp_min(1e-6)
    normed = (mag - mean) / std
    phase = torch.angle(spec)
    return torch.polar(normed.abs(), phase)


def _interleave_real_imag(spec: Tensor, torch_mod: ModuleType) -> Tensor:
    """Interleave real/imag along width: ``(H, W)`` complex -> ``(H, 2W)`` real (repo layout)."""
    torch = cast("Any", torch_mod)
    h, w = spec.shape
    out = torch.zeros(h, 2 * w, dtype=torch.float32)
    out[:, 0::2] = spec.real
    out[:, 1::2] = spec.imag
    return out


def _patch_maker(spec: Tensor, torch_mod: ModuleType) -> Tensor:
    """Split a ``(128, 256)`` interleaved spectrogram into ``(1024, 32)`` 4x4 patch tokens.

    Mirrors the repo's ``patch_maker(patch_rows=4, patch_cols=4, interleaved=True)``: the width
    is ``2*128`` because real/imag are interleaved, so a 4-column patch spans 8 interleaved
    values -> ``4 * 8 = 32`` = :data:`ELEMENT_LENGTH` per token, ``(128/4)*(128/4) = 1024`` tokens.
    """
    torch = cast("Any", torch_mod)
    h, w = spec.shape  # (128, 256)
    n_r, n_c = h // PATCH, (w // 2) // PATCH  # 32, 32 (columns count the ORIGINAL, un-doubled grid)
    # Reshape to (n_r, PATCH, n_c, PATCH*2) then flatten each patch to ELEMENT_LENGTH.
    grid = spec.reshape(n_r, PATCH, n_c, PATCH * 2)
    grid = grid.permute(0, 2, 1, 3).reshape(n_r * n_c, PATCH * PATCH * 2)
    return grid.to(torch.float32)


@register_model("lwm-spectro")
class LwmSpectroModel(FoundationModel):
    """The LWM-Spectro RF foundation model as a board ``Model`` (registered ``"lwm-spectro"``).

    Wraps the pretrained 12-layer LWM-Spectro encoder (128-d CLS representation) behind the
    frozen :class:`~rfbench.core.model.Model` contract:

    * :meth:`embed` -> ``(B, 128)`` frozen CLS features for ``linear_probe`` / ``few_shot``;
    * :meth:`forward` -> ``(B, 11)`` AMC logits from a fresh linear head on the frozen encoder;
    * :attr:`n_params` -> encoder + head parameter count; :attr:`family` -> ``"foundation"``.

    Constructed with no required args (``MODELS.get("lwm-spectro")()`` on the registry path).
    Construction is cheap: torch + weights load lazily on first :meth:`embed` / :meth:`forward`.
    The real weights come from :func:`rfbench.models.foundation._download_lwm_spectro.
    download_lwm_spectro`; when ``checkpoint=None`` the wrapper resolves the cached backbone
    under ``$RFBENCH_CACHE/lwm-spectro``. If the checkpoint is absent, :meth:`embed` /
    :meth:`forward` still run on the randomly-initialised encoder (a plumbing smoke test) and
    warn -- they never silently claim pretrained features.
    """

    def __init__(
        self,
        *,
        name: str = "lwm-spectro",
        num_classes: int = DEFAULT_NUM_CLASSES,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """Wrap the LWM-Spectro encoder under ``name``; keep construction torch-free and cheap."""
        super().__init__(
            name,
            n_params=0,  # set once the backbone is loaded (see _ensure_loaded)
            backbone="wi-lab/lwm-spectro:checkpoints/checkpoint.pth",
            pretrained=True,
        )
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = num_classes
        self._checkpoint = checkpoint
        self._device_str = device
        self._encoder: Any = None
        self._head: Any = None
        self._device: Any = None

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
        """Load the real pretrained ``state_dict`` into ``encoder`` (``strict=False``).

        Resolves the checkpoint from ``self._checkpoint`` or the cached
        ``$RFBENCH_CACHE/lwm-spectro/checkpoints/checkpoint.pth``. When absent the encoder keeps
        its random init and :attr:`pretrained` flips to ``False`` -- the wrapper never pretends
        to hold pretrained weights it did not load.
        """
        import logging

        torch = cast("Any", torch_mod)
        ckpt_path = self._resolve_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            self.pretrained = False
            logging.getLogger(__name__).warning(
                "LWM-Spectro checkpoint not found (%s); running on a randomly-initialised "
                "encoder. Fetch real weights with "
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
        logging.getLogger(__name__).info(
            "LWM-Spectro weights loaded from %s (missing=%d, unexpected=%d).",
            ckpt_path,
            len(missing),
            len(unexpected),
        )

    def _resolve_checkpoint(self) -> Path | None:
        """Return the backbone checkpoint path (explicit arg or the cached default)."""
        if self._checkpoint is not None:
            return Path(self._checkpoint).expanduser()
        from rfbench.models.foundation._download_lwm_spectro import backbone_checkpoint_path

        return backbone_checkpoint_path()

    # -- Model contract -------------------------------------------------------------------------
    def embed(self, x: Batch) -> Tensor:
        """Return ``(B, 128)`` frozen CLS features for the collated AMC batch ``x``."""
        self._ensure_loaded()
        torch = require_torch()
        tokens = _iq_to_lwm_tokens(x["iq"], torch).to(self._device)
        self._encoder.eval()
        with torch.no_grad():
            return self._encoder.encode(tokens)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` AMC logits: the head over the frozen encoder features."""
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
]
