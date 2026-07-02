"""WiSig-CNN SEI baseline (Track A) -- a compact 1-D CNN transmitter fingerprinter.

The canonical SEI closed-set baseline on WiSig ManyTx (Hanna et al., IEEE Access 2022) is a
small convolutional network over a single IQ window: a short stack of 1-D convolutions with
batch-norm + ReLU and stride-2 pooling learns per-emitter hardware impairments (I/Q imbalance,
phase noise, PA nonlinearity) directly from the raw ``(2, 256)`` signal, a global average pool
collapses the time axis, and two dense layers classify the resulting embedding into the
``n_tx`` transmitter identities. It is deliberately compact (a few hundred k params), so it
seeds the SEI board rather than acting as a heavy backbone -- the same role MCLDNN plays for
AMC.

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the COLLATED
batch dict that :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` is a *list* of
per-sample IQ payloads, one per sample. :class:`~rfbench.tasks.sei.dataset.SeiDataset` yields
WiSig windows of shape ``(256, 2)`` (256 time steps, I in column 0 / Q in column 1; see the
``(n, 256, 2)`` layout in ``rfbench/data/prepare/sei.py`` ``extract_wisig_records``), so the
collated ``x["iq"]`` is ``list[ (256, 2) ]`` and :func:`_iq_to_tensor` stacks it into a
``(B, 256, 2)`` tensor and transposes it to the ``(B, 2, 256)`` channel-first layout the 1-D
convolutions expect. ``forward`` returns ``(B, n_tx)`` class logits; iterating that tensor
yields one per-class score vector per sample, exactly what the closed-set
:class:`~rfbench.tasks.sei.metrics.Rank1Accuracy` metric's ``argmax`` decoder consumes.
``embed`` returns the ``(B, embed_dim)`` penultimate feature vector for the ``linear_probe`` /
``few_shot`` regimes.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module
is a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("wisig_cnn")`` entry in :data:`rfbench.core.registry.MODELS` is created only
on an explicit ``import rfbench.models.baselines.sei_cnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: WiSig ManyTx transmitter count used as the default closed-set head width. WiSig ships up to
#: this many emitters in the ManyTx subset; the concrete class count is taken from the prepared
#: split on the cluster, so ``n_tx`` is a constructor argument and this is only the fallback.
DEFAULT_NUM_CLASSES = 150
#: Canonical WiSig IQ window length (time steps per signal; see the ``(n, 256, 2)`` layout).
DEFAULT_WINDOW = 256
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
    SEI baseline runs under (``regime=from_scratch``).
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


class WiSigCNNNet(nn.Module):
    """The WiSig-CNN transmitter-fingerprinting network (compact 1-D CNN over one IQ window).

    Three stride-2 :class:`_ConvBlock` s over the ``(2, L)`` IQ signal downsample the time axis
    while widening the channel axis; an adaptive average pool collapses the remaining time steps
    to a single per-channel feature, which a dense layer projects to the ``embed_dim``
    penultimate embedding and a classifier maps to ``num_classes`` transmitter logits.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, embed_dim)`` penultimate representation the probing regimes fit a head on. The global
    pool makes the network agnostic to the exact window length, so a WiSig window other than the
    canonical 256 still yields a fixed-width embedding.
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
        """Return ``(B, num_classes)`` transmitter logits for a ``(B, 2, L)`` IQ batch."""
        return cast("Tensor", self.classifier(self.features(x)))


def _iq_to_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, L)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.sei.dataset.SeiDataset`: each element is a ``(window, 2)`` array-like
    (numpy on the cluster, nested lists in a synthetic fixture) in the WiSig ``(256, 2)``
    time-major layout. ``torch.as_tensor`` handles both; the result is coerced to ``float32``,
    the batch axis is added for a single unbatched sample, and the ``(B, window, 2)`` tensor is
    transposed to the ``(B, 2, window)`` channel-first layout the 1-D convs expect. A mis-shaped
    batch (channel axis not 2) fails loudly rather than silently mis-classifying.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (window, 2) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[2] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, {window}, 2); got {tuple(tensor.shape)}")
    return tensor.transpose(1, 2).contiguous()  # (B, window, 2) -> (B, 2, window)


@register_model("wisig_cnn")
class WiSigCNN(Model):
    """The WiSig-CNN SEI baseline as a :class:`~rfbench.core.model.Model` (``"wisig_cnn"``).

    Wraps :class:`WiSigCNNNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(window, 2)`` WiSig
      windows from :class:`~rfbench.tasks.sei.dataset.SeiDataset`) to ``(B, n_tx)`` transmitter
      logits -- iterated per-sample by the closed-set :class:`Rank1Accuracy` metric.
    * :meth:`embed` returns the ``(B, embed_dim)`` penultimate feature vector for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("wisig_cnn")()`` on the registry path
    (defaulting to the WiSig ManyTx head width). Eval runs in :meth:`eval` mode with gradients
    disabled; a from-scratch training loop (M3) loads weights into :attr:`net` before evaluation
    and should pass ``num_classes=`` the prepared split's transmitter count.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "wisig_cnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("WiSigCNN needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = WiSigCNNNet(num_classes, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, n_tx)`` transmitter logits for the collated SEI batch ``x``."""
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
    "WiSigCNN",
    "WiSigCNNNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_CONV_FILTERS",
    "DEFAULT_KERNEL",
    "DEFAULT_EMBED_DIM",
]
