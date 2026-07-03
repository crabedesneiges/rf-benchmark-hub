"""Interference-CNN baseline -- a compact 1-D IQ CNN for GNSS jamming classification.

A small convolutional network over one raw-IQ window is the in-repo specialised baseline for
the interference-ID task (Swinney & Woods 2021 GNSS-jamming set): a short stack of
conv-BN-ReLU blocks with stride-2 pooling learns the time-domain structure of each jamming
class (DME pulse pairs, chirp sweeps, narrowband tones, AM/FM envelopes) directly from the
``(2, L)`` signal, a global average pool collapses the time axis, and two dense layers
classify the resulting embedding into the six interference classes. It is deliberately compact
(a few hundred k params), so it seeds the interference-ID board rather than acting as a heavy
backbone -- the same role WiSig-CNN plays for SEI.

Literature SOTA (see the interference-ID section of ``docs/EVALUATION_PROTOCOL.md``). This raw-
IQ baseline is a from-scratch reference next to the published transform-domain results:
Morales-Ferre et al. 2019 (IEEE Sensors / IEEE Trans. Aerospace, a CNN and an SVM on STFT
spectrogram images of the jamming signals, reporting 91.36% (SVM) / 94.90% (CNN) accuracy) and
Swinney & Woods 2021 ("GNSS Jamming Classification via CNN, Transfer Learning & the Loss
Curvature Metric", CNN-feature extraction + transfer learning on THIS exact raw-IQ set). Both
operate on time-frequency images; this compact 1-D CNN over raw IQ is the harness' from-scratch
board seed.

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the COLLATED
batch dict that :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` is a *list* of
per-sample IQ payloads, one per sample. ``InterferenceDataset`` (in
``rfbench.tasks.interference_id.dataset``)
yields windows of shape ``(2, L)`` (I on row 0, Q on row 1), so the collated ``x["iq"]`` is
``list[ (2, L) ]`` and :func:`_iq_to_tensor` stacks it into a ``(B, 2, L)`` float tensor.
``forward`` returns ``(B, num_classes)`` class logits; iterating that tensor yields one
per-class score vector per sample, exactly what the interference-ID metrics' ``argmax`` decoder
consumes. ``embed`` returns the ``(B, embed_dim)`` penultimate feature vector for the
``linear_probe`` / ``few_shot`` regimes.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module
is a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("interf_cnn")`` entry in :data:`rfbench.core.registry.MODELS` is created only
on an explicit ``import rfbench.models.baselines.interf_cnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: The six GNSS-jamming classes the head classifies (DME, narrowband, no_jamming, single_am,
#: single_chirp, single_fm).
DEFAULT_NUM_CLASSES = 6
#: Default raw-IQ window length (samples per channel). A nominal value; the global pool makes
#: the network agnostic to the exact window length, so the real per-capture length is fine.
DEFAULT_WINDOW = 1024
#: Number of IQ channels (I and Q) -- the conv stack's input channel count.
_IQ_CHANNELS = 2
#: Convolution feature widths of the three stride-2 conv blocks (compact, board-seeding).
DEFAULT_CONV_FILTERS: tuple[int, int, int] = (32, 64, 128)
#: Convolution kernel length shared by every block (odd, so "same" padding stays symmetric).
DEFAULT_KERNEL = 7
#: Penultimate dense width -- the embedding dimension returned by :meth:`embed`.
DEFAULT_EMBED_DIM = 128


def _same_pad_1d(kernel: int) -> int:
    """Return the symmetric ``padding`` that keeps a 1-D conv's length unchanged (odd kernel)."""
    return (kernel - 1) // 2


class _ConvBlock(nn.Module):
    """One conv block: 1-D conv (same padding) + batch-norm + ReLU, then a stride-2 max-pool.

    Learns local IQ features at a fixed receptive field, then halves the time axis so the stack
    sees progressively coarser structure. Batch-norm stabilises the from-scratch training the
    interference-ID baseline runs under (``regime=from_scratch``).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        """Build the conv + norm + activation + pooling for one block."""
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel, padding=_same_pad_1d(kernel)
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in_channels, L)`` -> ``(B, out_channels, L // 2)``."""
        return cast("Tensor", self.pool(self.act(self.norm(self.conv(x)))))


class InterferenceCNNNet(nn.Module):
    """The interference-CNN network (compact 1-D CNN over one raw-IQ window).

    Three stride-2 :class:`_ConvBlock` s over the ``(2, L)`` IQ signal downsample the time axis
    while widening the channel axis; an adaptive average pool collapses the remaining time steps
    to a single per-channel feature, which a dense layer projects to the ``embed_dim``
    penultimate embedding and a classifier maps to ``num_classes`` interference logits.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, embed_dim)`` penultimate representation the probing regimes fit a head on. The global
    pool makes the network agnostic to the exact window length, so a window other than the
    nominal default still yields a fixed-width embedding.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        conv_filters: tuple[int, int, int] = DEFAULT_CONV_FILTERS,
        kernel: int = DEFAULT_KERNEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        """Build the conv stack, the global pool, the dense embedding and the classifier head."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.num_classes = num_classes
        self.window = window
        self.embed_dim = embed_dim

        channels = (_IQ_CHANNELS, *conv_filters)
        self.blocks = nn.Sequential(
            *(_ConvBlock(channels[i], channels[i + 1], kernel) for i in range(len(conv_filters)))
        )
        # Collapse the (possibly length-varying) time axis to one feature per channel.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Sequential(
            nn.Linear(conv_filters[-1], embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, embed_dim)`` penultimate embedding for a ``(B, 2, L)`` IQ batch."""
        feats = self.blocks(x)  # (B, C, L')
        pooled = self.pool(feats).squeeze(-1)  # (B, C)
        return cast("Tensor", self.fc_embed(pooled))  # (B, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` interference logits for a ``(B, 2, L)`` IQ batch."""
        return cast("Tensor", self.classifier(self.features(x)))


def _iq_to_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, L)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.interference_id.dataset.InterferenceDataset`: each element is a
    ``(2, L)`` array-like (numpy on the cluster, nested lists in a synthetic fixture) in the
    channel-first layout (I on row 0, Q on row 1). ``torch.as_tensor`` handles both; the result
    is coerced to ``float32``, the batch axis is added for a single unbatched sample, and a
    mis-shaped batch (channel axis not 2) fails loudly rather than silently mis-classifying.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, L) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor


@register_model("interf_cnn")
class InterferenceCNN(Model):
    """The interference-CNN baseline as a :class:`~rfbench.core.model.Model` (``"interf_cnn"``).

    Wraps :class:`InterferenceCNNNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, L)`` IQ windows
      from :class:`~rfbench.tasks.interference_id.dataset.InterferenceDataset`) to
      ``(B, num_classes)`` logits -- iterated per-sample by the interference-ID metrics.
    * :meth:`embed` returns the ``(B, embed_dim)`` penultimate feature vector for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("interf_cnn")()`` on the registry path
    (defaulting to the 6-class head). Eval runs in :meth:`eval` mode with gradients disabled; a
    from-scratch training loop (M3) loads weights into :attr:`net` before evaluation.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "interf_cnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("InterferenceCNN needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = InterferenceCNNNet(num_classes, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` interference logits for the collated batch ``x``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, embed_dim)`` embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "InterferenceCNN",
    "InterferenceCNNNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_CONV_FILTERS",
    "DEFAULT_KERNEL",
    "DEFAULT_EMBED_DIM",
]
