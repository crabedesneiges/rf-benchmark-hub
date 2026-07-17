"""T-PRIME baseline -- a transformer over raw interleaved IQ for WiFi standard recognition.

T-PRIME (Belgiovine et al., "T-PRIME: Transformer-based Protocol Identification for
Machine-learning at the Edge", arXiv:2401.04837, extended IEEE INFOCOM 2024; Genesys Lab /
Northeastern, code at github.com/genesys-neu/t-prime) classifies a raw over-the-air IQ window
into one of four 802.11 standards (``802.11b``, ``802.11g``, ``802.11n``, ``802.11ax``) with a
transformer encoder that uses **no learned input embedding**: the raw IQ values are fed
directly as the token space.

Tokenisation (paper §III, Table II). One window is a ``(2, N)`` sequence (I on row 0, Q on
row 1) that is sliced along time into ``M`` consecutive slices of ``(2, S)`` (``N = M * S``);
each ``(2, S)`` slice is **flattened** into a single ``1 x 2S`` token (I samples then Q
samples). Those ``M`` tokens of width ``2S`` are fed straight into a standard transformer
encoder (a learned positional encoding is added, but the token values themselves are the raw
IQ -- there is no learned linear projection of the input). A mean-pool over the ``M`` token
representations gives the window embedding, and a linear head maps it to the four class logits.

Two published variants (Table II):

* **SM (small, default here)**: ``M = 24`` tokens, ``S = 64`` (``N = 1536``), ``2`` encoder
  layers, ~1.6 M params.
* **LG (large)**: ``M = 64`` tokens, ``S = 128`` (``N = 8192``), ``2`` encoder layers,
  ~6.8 M params. Select it via ``TPrime(variant="LG")`` (Hydra ``model.variant=LG``).

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the COLLATED
batch dict that :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` is a *list* of
per-sample IQ payloads. :class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset` yields
windows of shape ``(2, L)`` (I on row 0, Q on row 1), so the collated ``x["iq"]`` is
``list[ (2, L) ]`` and :func:`_iq_to_tensor` stacks it into a ``(B, 2, N)`` float tensor (each
window is centre-cropped / zero-padded to the variant's ``N``). ``forward`` returns
``(B, num_classes)`` class logits; iterating that tensor yields one per-class score vector per
sample, exactly what the protocol-tech-ID metrics' ``argmax`` decoder consumes. ``embed``
returns the ``(B, d_model)`` pooled encoder feature for the ``linear_probe`` / ``few_shot``
regimes.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module
is a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("tprime")`` entry in :data:`rfbench.core.registry.MODELS` is created only on
an explicit ``import rfbench.models.baselines.tprime``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: The four 802.11 standards the head classifies (802.11 b/g/n/ax).
DEFAULT_NUM_CLASSES = 4
#: Number of IQ channels (I and Q) -- the per-slice row count before flattening to ``2S``.
_IQ_CHANNELS = 2

#: T-PRIME SM (small) variant tokenisation + encoder (paper Table II).
SM_SLICES = 24  # M: number of slices/tokens
SM_SLICE_LEN = 64  # S: samples per slice per channel
SM_LAYERS = 2  # transformer encoder layers
SM_HEADS = 8  # attention heads (d_model = 2S = 128 must be divisible by heads)
SM_FF_DIM = 128  # feed-forward hidden width
#: SM window length N = M * S.
SM_SEQUENCE_LEN = SM_SLICES * SM_SLICE_LEN  # 1536

#: T-PRIME LG (large) variant tokenisation + encoder (paper Table II).
LG_SLICES = 64  # M
LG_SLICE_LEN = 128  # S
LG_LAYERS = 2
LG_HEADS = 8  # d_model = 2S = 256, divisible by 8
LG_FF_DIM = 256
#: LG window length N = M * S.
LG_SEQUENCE_LEN = LG_SLICES * LG_SLICE_LEN  # 8192

#: Transformer variant name -> its (M, S, layers, heads, ff_dim) hyper-parameters.
TPrimeVariant = Literal["SM", "LG"]
_VARIANTS: dict[str, dict[str, int]] = {
    "SM": {
        "slices": SM_SLICES,
        "slice_len": SM_SLICE_LEN,
        "layers": SM_LAYERS,
        "heads": SM_HEADS,
        "ff_dim": SM_FF_DIM,
    },
    "LG": {
        "slices": LG_SLICES,
        "slice_len": LG_SLICE_LEN,
        "layers": LG_LAYERS,
        "heads": LG_HEADS,
        "ff_dim": LG_FF_DIM,
    },
}


class TPrimeNet(nn.Module):
    """The T-PRIME transformer network (encoder over raw-IQ slice tokens, no input embedding).

    Slices a ``(B, 2, N)`` IQ window into ``M`` tokens of width ``2S`` (each token is one
    ``(2, S)`` slice flattened as ``[I_0..I_{S-1}, Q_0..Q_{S-1}]``), adds a learned positional
    encoding, runs a ``num_layers``-deep transformer encoder, mean-pools the ``M`` token
    outputs, and classifies the pooled ``(B, 2S)`` embedding into ``num_classes`` logits.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, d_model)`` pooled encoder embedding the probing regimes fit a head on.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        variant: TPrimeVariant = "SM",
    ) -> None:
        """Build the positional encoding, transformer encoder and classifier for ``variant``."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown T-PRIME variant {variant!r}; expected one of {list(_VARIANTS)}"
            )
        cfg = _VARIANTS[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.slices = cfg["slices"]
        self.slice_len = cfg["slice_len"]
        self.sequence_len = self.slices * self.slice_len
        # No learned input embedding: the token dimension IS the flattened raw-IQ slice (2*S).
        self.d_model = _IQ_CHANNELS * self.slice_len
        self.embed_dim = self.d_model

        # Learned positional encoding over the M token positions (added to the raw tokens).
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.slices, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg["heads"],
            dim_feedforward=cfg["ff_dim"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg["layers"])
        self.classifier = nn.Linear(self.d_model, num_classes)

    def _tokenize(self, x: Tensor) -> Tensor:
        """Slice a ``(B, 2, N)`` IQ window into ``(B, M, 2S)`` flattened slice tokens.

        Each of the ``M`` consecutive ``(2, S)`` slices is flattened to ``[I..., Q...]`` (row-
        major over the channel axis) so token ``m`` is ``[I_{mS}..I_{mS+S-1}, Q_{mS}..]``.
        """
        b = x.shape[0]
        # (B, 2, M, S) -> (B, M, 2, S) -> (B, M, 2S): channel-major flatten per slice.
        sliced = x.reshape(b, _IQ_CHANNELS, self.slices, self.slice_len)
        sliced = sliced.permute(0, 2, 1, 3).contiguous()
        return sliced.reshape(b, self.slices, self.d_model)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, d_model)`` mean-pooled encoder embedding for a ``(B, 2, N)`` batch."""
        tokens = self._tokenize(x) + self.pos_encoding  # (B, M, 2S)
        encoded = self.encoder(tokens)  # (B, M, 2S)
        return cast("Tensor", encoded.mean(dim=1))  # (B, 2S)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` protocol logits for a ``(B, 2, N)`` IQ batch."""
        return cast("Tensor", self.classifier(self.features(x)))


def _unit_variance_normalize(x: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Standardise each ``(2, N)`` IQ window to zero mean and unit variance (per-sample).

    T-PRIME has NO learned input embedding (the raw IQ slices ARE the tokens) and no input
    scaling, and the DS 3.0 over-the-air captures come in at a small raw scale (I/Q std on the
    order of ~7e-2, |val| < ~0.2 in the extracted ``.bin``). Feeding a near-zero-scale signal
    straight into the token space -- summed with the learned positional encoding -- starts
    training from a poorly-conditioned scale, the same failure mode that collapsed the CLDNN /
    ResNet AMC baselines (see ``cldnn._unit_variance_normalize`` /
    ``resnet_amc._unit_variance_normalize``, the transform this mirrors). Standardising each
    window over BOTH channels and the whole time axis (dims ``(1, 2)``) removes the absolute
    capture scale (which carries no protocol information) while preserving the I/Q geometry that
    does. ``eps`` guards an all-constant (zero-variance) window. Matches the T-PRIME reference's
    per-window normalisation. Duplicated here (not imported from a sibling baseline) so this
    module stays standalone, the same rationale as :func:`_iq_to_tensor`.
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
    return cast("Tensor", (x - mean) / (std + eps))


def _iq_to_tensor(iq_batch: object, device: torch.device, sequence_len: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, sequence_len)`` normalised float tensor.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset`: each element is a
    ``(2, L)`` array-like (numpy on the cluster, nested lists in a synthetic fixture) in the
    channel-first layout (I on row 0, Q on row 1). Each window is coerced to ``float32`` and
    fixed to the variant's ``N`` by a centre-crop (if longer) or a right zero-pad (if shorter),
    so the tokeniser always sees exactly ``M * S`` samples, then standardised per-window
    (:func:`_unit_variance_normalize`) so the raw IQ reaches the embedding-free token space at a
    healthy scale. A mis-shaped batch (channel axis not 2) fails loudly rather than silently
    mis-classifying.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, L) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != _IQ_CHANNELS:
        raise ValueError(
            f"expected IQ batch of shape (B, 2, {sequence_len}); got {tuple(tensor.shape)}"
        )
    length = tensor.shape[2]
    if length > sequence_len:  # centre-crop to N
        start = (length - sequence_len) // 2
        tensor = tensor[:, :, start : start + sequence_len]
    elif length < sequence_len:  # right zero-pad to N
        pad = torch.zeros(
            tensor.shape[0], _IQ_CHANNELS, sequence_len - length, dtype=tensor.dtype, device=device
        )
        tensor = torch.cat([tensor, pad], dim=2)
    return _unit_variance_normalize(tensor)


@register_model("tprime")
class TPrime(Model):
    """The T-PRIME baseline as a :class:`~rfbench.core.model.Model` (registered ``"tprime"``).

    Wraps :class:`TPrimeNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, L)`` IQ windows
      from :class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset`) to
      ``(B, num_classes)`` logits -- iterated per-sample by the protocol-tech-ID metrics.
    * :meth:`embed` returns the ``(B, d_model)`` pooled encoder feature for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("tprime")()`` on the registry path (the SM
    variant, 4-class head). Pass ``variant="LG"`` (Hydra ``model.variant=LG``) for the large
    variant. Eval runs in :meth:`eval` mode with gradients disabled; a from-scratch training
    loop (M3) loads weights into :attr:`net` before evaluation.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "tprime",
        num_classes: int = DEFAULT_NUM_CLASSES,
        variant: TPrimeVariant = "SM",
        device: str | None = None,
    ) -> None:
        """Build the network for ``variant`` and move it to ``device`` (auto: CUDA else CPU)."""
        if not name:
            raise ValueError("TPrime needs a non-empty name")
        self.name = name
        self.variant = variant
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = TPrimeNet(num_classes, variant=variant).to(self.device)
        self.sequence_len = self.net.sequence_len

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` protocol logits for the collated batch ``x``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.sequence_len)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, d_model)`` embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.sequence_len)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "TPrime",
    "TPrimeNet",
    "TPrimeVariant",
    "DEFAULT_NUM_CLASSES",
    "SM_SLICES",
    "SM_SLICE_LEN",
    "SM_SEQUENCE_LEN",
    "LG_SLICES",
    "LG_SLICE_LEN",
    "LG_SEQUENCE_LEN",
]
