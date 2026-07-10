"""ResNet-1D SEI baseline -- a modern deep residual fingerprinter on raw IQ.

The de-facto strong deep baseline for SEI: a 1-D residual network over the raw ``(2, L)`` IQ
window, following the ResNet-1D used as the scalable reference in Jian et al., "Deep Learning for
RF Fingerprinting: A Massive Experimental Study" (IEEE IoT Magazine 2020), with the standard
residual design of He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016), adapted
to 1-D signals. It occupies the *depth / capacity* axis of the SEI board that the shallow
WiSig 2-D CNN (5 conv) and ORACLE CNN (2 conv) do not cover, answering "does a deep residual net
help on the cross-receiver / cross-day splits". See ``docs/BIBLIOGRAPHY.md`` C.2.

Architecture (ResNet-18-1D): a 7-wide stem conv (stride 2) + max-pool, then four residual stages
of two ``BasicBlock`` s each (widths 64/128/256/512; the first block of stages 2-4 strides 2 with
a 1x1 projection skip), a global average pool, and a linear classifier. Per-layer BatchNorm keeps
the deep ReLU stack trainable on raw IQ, and a per-signal unit-average-power input normalisation
removes the capture scale (the SEI-standard transform; its absence collapses deep raw-IQ nets, as
seen for the AMC ResNet/CLDNN baselines).

**Input layout convention (shared SEI baselines).** ``forward`` / ``embed`` receive the collated
batch (``x["iq"]`` a list of ``(window, 2)`` IQ windows); the ``nn.Module`` normalises and
transposes ``(B, window, 2)`` -> ``(B, 2, window)`` itself, so the same tensor is fed at eval time
(Model wrapper) and train time (:mod:`rfbench.training_sei`).

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported at THIS module's top; ``@register_model("resnet1d_sei")`` fires only on an explicit
``import rfbench.models.baselines.resnet1d_sei``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: Default transmitter count (overridden by the prepared split's class count on the cluster).
DEFAULT_NUM_CLASSES = 150
#: Nominal IQ window length; the global pool makes the net window-agnostic (WiSig 256, ORACLE 128).
DEFAULT_WINDOW = 256
#: Number of IQ channels (I and Q) -- the stem conv's input channel count.
_IQ_CHANNELS = 2
#: Residual stage widths (ResNet-18: 64/128/256/512).
_STAGE_WIDTHS: tuple[int, int, int, int] = (64, 128, 256, 512)
#: Blocks per stage (ResNet-18: 2/2/2/2).
_STAGE_BLOCKS: tuple[int, int, int, int] = (2, 2, 2, 2)


def _unit_average_power_normalize(x: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Per-signal unit-average-power normalisation of a ``(B, window, 2)`` IQ batch.

    Divides each signal by ``sqrt(mean_t(I^2 + Q^2))``; shared SEI transform (see
    :mod:`rfbench.models.baselines.wisig_cnn_paper`). Load-bearing for a deep raw-IQ net.
    """
    power = (x**2).sum(dim=2).mean(dim=1)
    rms = torch.sqrt(power).clamp_min(eps)
    return cast("Tensor", x / rms[:, None, None])


class _BasicBlock1d(nn.Module):
    """A ResNet 1-D basic block: two 3-wide conv-BN layers with a (projected) identity skip.

    Both convs keep the channel width and (for stride 1) the length; the first conv strides
    ``stride`` to downsample, and a 1x1 conv-BN projection matches the skip's channels/length when
    they change. The residual sum is passed through a final ReLU (canonical Conv-BN-ReLU block).
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        """Build the two conv-BN layers and the (optional) projection skip."""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in, L)`` -> ``(B, out, L')`` with the residual skip added before the ReLU."""
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return cast("Tensor", F.relu(out + identity, inplace=True))


class ResNet1dNet(nn.Module):
    """ResNet-18-1D over ``(B, window, 2)`` raw IQ.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the ``(B, 512)``
    pooled penultimate embedding for the probing regimes. The adaptive pool makes the embedding
    width window-agnostic.
    """

    def __init__(
        self, num_classes: int = DEFAULT_NUM_CLASSES, *, window: int = DEFAULT_WINDOW
    ) -> None:
        """Build the stem, the four residual stages, the global pool and the classifier."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 8:
            raise ValueError(f"window must be >= 8, got {window}")
        self.num_classes = num_classes
        self.window = window

        self.stem = nn.Sequential(
            nn.Conv1d(_IQ_CHANNELS, _STAGE_WIDTHS[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(_STAGE_WIDTHS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_ch = _STAGE_WIDTHS[0]
        for i, (width, n_blocks) in enumerate(zip(_STAGE_WIDTHS, _STAGE_BLOCKS, strict=True)):
            stride = 1 if i == 0 else 2  # first stage keeps length; later stages downsample
            for b in range(n_blocks):
                stages.append(_BasicBlock1d(in_ch, width, stride=stride if b == 0 else 1))
                in_ch = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(_STAGE_WIDTHS[-1], num_classes)
        self.embed_dim = _STAGE_WIDTHS[-1]

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, 512)`` pooled penultimate embedding for a ``(B, window, 2)`` batch."""
        x = _unit_average_power_normalize(x)
        x = x.transpose(1, 2)  # (B, window, 2) -> (B, 2, window)
        x = self.stages(self.stem(x))
        return cast("Tensor", self.pool(x).squeeze(-1))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` transmitter logits for a ``(B, window, 2)`` IQ batch."""
        return cast("Tensor", self.classifier(self.features(x)))


def _iq_to_bt2_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, window, 2)`` float tensor on ``device``."""
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[2] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, {window}, 2); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("resnet1d_sei")
class ResNet1dSEI(Model):
    """The ResNet-1D SEI baseline as a :class:`~rfbench.core.model.Model` (``"resnet1d_sei"``).

    Wraps :class:`ResNet1dNet` to the frozen ``Model`` contract: :meth:`forward` maps the collated
    batch (``x["iq"]`` a list of ``(window, 2)`` windows) to ``(B, n_tx)`` logits; :meth:`embed`
    returns the ``(B, 512)`` pooled feature; :attr:`family` is ``"baseline"``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "resnet1d_sei",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("ResNet1dSEI needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = ResNet1dNet(num_classes, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, n_tx)`` transmitter logits for the collated SEI batch ``x``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, 512)`` pooled embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "ResNet1dSEI",
    "ResNet1dNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
]
