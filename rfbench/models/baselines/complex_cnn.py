"""Complex-valued CNN SEI baseline -- Gopalakrishnan/Cekic/Madhow, faithful reconstruction.

The strongest *architecturally distinct* specialised baseline we add for SEI: a complex-valued
CNN that convolves directly on complex baseband IQ and uses the modReLU activation before taking
the magnitude, from Gopalakrishnan, Cekic & Madhow, "Robust Wireless Fingerprinting via
Complex-Valued Neural Networks" (IEEE GLOBECOM 2019, arXiv:1905.09388) and the follow-up
"Wireless Fingerprinting via Deep Learning: The Impact of Confounding Factors" (Asilomar 2021,
arXiv:2002.10791). It is the biggest inductive-bias contrast to the real-valued WiSig 2-D CNN
and ORACLE 2-conv CNN -- it keeps I/Q phase coupled through the network instead of treating I
and Q as two independent real channels -- and it directly targets the channel/CFO confounders
the SEI cross-receiver / cross-day protocol stresses.

Faithful to the official model ``network_20_modrelu_short`` in ``cxnn/models.py`` of
``github.com/metehancekic/wireless-fingerprinting`` (MIT-licensed; complex layers after
Trabelsi et al., "Deep Complex Networks", ICLR 2018). Verbatim layer sequence::

    ComplexConv1D(filters=100, kernel=20, stride=10, padding='valid', use_bias=False)
    ModReLU
    ComplexConv1D(filters=100, kernel=10, stride=1,  padding='valid', use_bias=False)
    ModReLU
    GetAbs                       # complex -> real magnitude
    GlobalAveragePooling1D
    Dense(100, relu)
    Dense(N, softmax)
    L2 weight decay 1e-4; 'complex_independent' init in the reference.

The reference ships a *simulation-based* dataset; we do NOT use it -- this is the architecture
ported to PyTorch and evaluated on our real WiSig / ORACLE splits (faithful to the papers' own
validation on real WiFi + ADS-B). See ``docs/BIBLIOGRAPHY.md`` C.2.

**Input layout convention (shared SEI baselines).** ``forward`` / ``embed`` receive the collated
batch (``x["iq"]`` a list of ``(window, 2)`` IQ windows). Channel 0 is I (real), channel 1 is Q
(imaginary): the ``nn.Module`` splits ``(B, window, 2)`` into the real/imag streams itself, so
the same tensor is fed at eval time (Model wrapper) and train time (:mod:`rfbench.training_sei`).

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported at THIS module's top; ``@register_model("complex_cnn")`` fires only on an explicit
``import rfbench.models.baselines.complex_cnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
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
#: Number of IQ channels (I -> real, Q -> imaginary).
_IQ_CHANNELS = 2
#: Complex conv feature width (both layers use 100 complex filters).
DEFAULT_FILTERS = 100
#: Penultimate dense width (``Dense(100, relu)``) -- the embedding dimension.
DEFAULT_EMBED_DIM = 100
#: L2 regularisation strength on the conv + dense kernels (reference default 1e-4).
DEFAULT_L2_LAMBDA = 1e-4


class ComplexConv1d(nn.Module):
    """A complex-valued 1-D convolution: ``(yr, yi) = (xr*Wr - xi*Wi, xr*Wi + xi*Wr)``.

    Holds two real ``Conv1d`` kernels (``Wr``, ``Wi``, both bias-free like the reference
    ``use_bias=False``) and applies the complex-multiply convolution to a ``(real, imag)`` pair
    of ``(B, C_in, L)`` tensors, returning the ``(real, imag)`` pair of ``(B, C_out, L')``
    outputs. ``padding='valid'`` (no padding) and the reference strides are honoured.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, *, stride: int) -> None:
        """Build the real/imag kernels (no bias, valid padding, given stride)."""
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Return the complex convolution ``(yr, yi)`` of the complex input ``(xr, xi)``."""
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi

    def kernel_l2(self) -> Tensor:
        """Sum of squared real + imag kernel weights (for the trainer's L2 term)."""
        return self.conv_r.weight.pow(2).sum() + self.conv_i.weight.pow(2).sum()


class ModReLU(nn.Module):
    """modReLU (Trabelsi et al. 2018): ``ReLU(|z| + b) * (z / |z|)`` with a learned per-channel b.

    Scales each complex activation by ``relu(|z| + b) / |z|``, so it thresholds on the *magnitude*
    (shifted by the learned bias ``b``) while preserving the phase ``z/|z|`` -- the phase-aware
    nonlinearity that distinguishes a complex-valued net from a real one. ``b`` is a real bias
    per output channel, initialised to zero as in the reference.
    """

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        """Create the per-channel magnitude bias ``b`` (zero-init)."""
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, xr: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Apply modReLU to a ``(real, imag)`` pair of ``(B, C, L)`` tensors."""
        # ``eps`` inside the sqrt floors the magnitude at ``sqrt(eps)`` (> 0) -- this makes the
        # subsequent division numerically bounded (no /0, no divergence as |z| -> 0) AND keeps the
        # sqrt gradient finite at |z| = 0. ``scale`` is a REAL per-element factor, so multiplying
        # ``(xr, xi)`` by it rescales the magnitude while preserving the phase ``z/|z|`` EXACTLY.
        mag = torch.sqrt(xr * xr + xi * xi + self.eps)  # (B, C, L); mag >= sqrt(eps) > 0
        scale = F.relu(mag + self.b[None, :, None]) / mag  # magnitude threshold, phase preserved
        return xr * scale, xi * scale


class ComplexCNNNet(nn.Module):
    """The ``network_20_modrelu_short`` complex-valued CNN over ``(B, window, 2)`` IQ.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the ``(B, 100)``
    penultimate (ReLU'd ``Dense(100)``) embedding; :meth:`l2_penalty` sums the conv + dense
    kernels for the trainer's L2. The global average pool makes the head width independent of the
    window length.
    """

    def __init__(
        self, num_classes: int = DEFAULT_NUM_CLASSES, *, window: int = DEFAULT_WINDOW
    ) -> None:
        """Build the two complex convs + modReLUs, the magnitude pool and the dense head."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        # conv1 (kernel 20, stride 10, valid) then conv2 (kernel 10, stride 1, valid): the second
        # conv needs conv1's output length >= 10. conv1_out = floor((window - 20) / 10) + 1.
        conv1_out = (window - 20) // 10 + 1
        if window < 20 or conv1_out < 10:
            min_window = 20 + 9 * 10  # smallest window giving conv1_out >= 10
            raise ValueError(
                f"window must be >= {min_window} for the (kernel-20/stride-10, kernel-10) convs, "
                f"got {window} (conv1 output length {max(conv1_out, 0)} < 10). WiSig 256 / "
                "ORACLE 128 both satisfy this."
            )
        self.num_classes = num_classes
        self.window = window

        self.cconv1 = ComplexConv1d(1, DEFAULT_FILTERS, 20, stride=10)
        self.mrelu1 = ModReLU(DEFAULT_FILTERS)
        self.cconv2 = ComplexConv1d(DEFAULT_FILTERS, DEFAULT_FILTERS, 10, stride=1)
        self.mrelu2 = ModReLU(DEFAULT_FILTERS)
        self.dense = nn.Linear(DEFAULT_FILTERS, DEFAULT_EMBED_DIM)
        self.classifier = nn.Linear(DEFAULT_EMBED_DIM, num_classes)
        self.embed_dim = DEFAULT_EMBED_DIM

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, 100)`` penultimate embedding for a ``(B, window, 2)`` IQ batch."""
        xr = x[:, :, 0].unsqueeze(1)  # (B, 1, window) real (I)
        xi = x[:, :, 1].unsqueeze(1)  # (B, 1, window) imag (Q)
        xr, xi = self.cconv1(xr, xi)
        xr, xi = self.mrelu1(xr, xi)
        xr, xi = self.cconv2(xr, xi)
        xr, xi = self.mrelu2(xr, xi)
        mag = torch.sqrt(xr * xr + xi * xi + 1e-12)  # GetAbs -> real magnitude (B, 100, L')
        pooled = mag.mean(dim=2)  # GlobalAveragePooling1D -> (B, 100)
        return cast("Tensor", F.relu(self.dense(pooled)))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` transmitter logits for a ``(B, window, 2)`` IQ batch."""
        return cast("Tensor", self.classifier(self.features(x)))

    def l2_penalty(self) -> Tensor:
        """Sum of squared conv + dense KERNELS -- the L2 term the trainer adds to the loss."""
        return (
            self.cconv1.kernel_l2()
            + self.cconv2.kernel_l2()
            + self.dense.weight.pow(2).sum()
            + self.classifier.weight.pow(2).sum()
        )


def _iq_to_bt2_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, window, 2)`` float tensor on ``device``."""
    # Collate the per-sample list into ONE ndarray first: torch.as_tensor on a list of ndarrays
    # copies element-by-element ("extremely slow" per torch) and dominated SEI eval wall-time.
    # np.asarray stacks in one shot; the resulting tensor is numerically identical.
    if isinstance(iq_batch, (list, tuple)):
        iq_batch = np.asarray(iq_batch)
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[2] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, {window}, 2); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("complex_cnn")
class ComplexCNN(Model):
    """The complex-valued CNN SEI baseline as a :class:`~rfbench.core.model.Model`.

    Wraps :class:`ComplexCNNNet` to the frozen ``Model`` contract: :meth:`forward` maps the
    collated batch (``x["iq"]`` a list of ``(window, 2)`` windows) to ``(B, n_tx)`` logits;
    :meth:`embed` returns the ``(B, 100)`` penultimate feature; :attr:`family` is ``"baseline"``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "complex_cnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("ComplexCNN needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = ComplexCNNNet(num_classes, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, n_tx)`` transmitter logits for the collated SEI batch ``x``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, 100)`` penultimate embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "ComplexCNN",
    "ComplexCNNNet",
    "ComplexConv1d",
    "ModReLU",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_L2_LAMBDA",
]
