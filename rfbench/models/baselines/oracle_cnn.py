"""ORACLE CNN SEI baseline -- the Sankhe et al. 2019 raw-IQ device fingerprinter.

The canonical closed-set fingerprinting CNN for the ORACLE dataset (Sankhe, Rajendran,
Belgiovine, Chowdhury, Ioannidis, "ORACLE: Optimized Radio clAssification through Convolutional
neuraL nEtworks", IEEE INFOCOM 2019, arXiv:1812.01124, DOI 10.1109/INFOCOM.2019.8737463). It
learns per-emitter hardware impairments (I/Q imbalance, DC offset, phase noise, PA nonlinearity)
directly from a short ``2 x 128`` raw-IQ slice: two convolutions -- the first (kernel ``1 x 7``)
filters the I and Q streams independently, the second (kernel ``2 x 7``) mixes them -- then two
dense layers classify into the ``num_classes`` (16 X310 radios in the reference) transmitter
identities. See ``docs/BIBLIOGRAPHY.md`` A.3 / C.2 for the audit. Static same-location accuracy
is ~98.6% (16 tx); the cross-location drop (Fig. 6: 87.13%) motivates a separate ORACLE
cross-location reference track.

Verbatim architecture (paper Fig. 4)::

    input 2 x 128 (I,Q as height=2, time=128)
    Conv2D(50, (1,7), relu)         # per-stream filtering
    Conv2D(50, (2,7), relu)         # I/Q mixing (collapses the height-2 axis)
    Flatten
    Dense(256, relu); Dropout(0.5)
    Dense(80 , relu); Dropout(0.5)
    Dense(N, softmax)
    Adam(lr=1e-4), categorical cross-entropy, L2 1e-4, early stop patience 10 on val acc

**Input layout convention (shared with the other SEI baselines).** ``forward`` / ``embed``
receive the collated batch dict (``x["iq"]`` a list of per-sample ``(window, 2)`` IQ windows).
The ``nn.Module`` accepts the natural ``(B, window, 2)`` tensor and permutes to ``(B, 1, 2,
window)`` itself, so the same tensor is fed at eval time (Model wrapper) and train time
(:mod:`rfbench.training_sei` collate).

Preprocessing note: the ORACLE paper's exact per-example input scaling is not pinned in the
primary sources; ORACLE IQ is stored as (large-scale) ``complex128``. This module applies the
SEI-standard **per-signal unit-average-power** normalisation by default (``input_norm=True``,
the same transform :mod:`~rfbench.models.baselines.wisig_cnn_paper` uses) for a stable, scale-
robust fit; pass ``input_norm=False`` to feed raw IQ for an ablation. Flagged as an open
reproduction question in the bibliography.

Training-recipe note (deferred track): the ORACLE run is deferred (its data is not on the
cluster). The shared SEI loop (:mod:`rfbench.training_sei`) is the WiSig recipe -- best
checkpoint + early stop on **validation LOSS** (patience 5 default). ORACLE's paper uses
**Adam lr=1e-4** (set by the SLURM driver ``slurm/train_sei_arm.sh`` for this model) and
early-stops on **val accuracy, patience 10** -- a minor recipe difference (both early-stop on
convergence) documented as a known deviation in ``docs/BIBLIOGRAPHY.md`` B.4 for the eventual
ORACLE reproduction; not implemented here (the loop is deliberately the val-loss WiSig recipe).

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch``
is imported at THIS module's top; ``@register_model("oracle_cnn")`` fires only on an explicit
``import rfbench.models.baselines.oracle_cnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: ORACLE transmitter count -- 16 bit-identical USRP X310 radios in the reference release.
DEFAULT_NUM_CLASSES = 16
#: ORACLE IQ slice length (time steps). The reference uses 128-sample WiFi-burst slices.
DEFAULT_WINDOW = 128
#: Number of IQ channels (I and Q) -- the conv input HEIGHT axis (kernel (2,7) mixes them).
_IQ_CHANNELS = 2
#: Conv feature width shared by both conv layers (paper: 50 filters each).
DEFAULT_CONV_FILTERS = 50
#: Conv kernel time-width (the "7" in (1,7)/(2,7)).
DEFAULT_KERNEL_TIME = 7
#: Dense head widths (paper: FC 256 then FC 80).
DEFAULT_FC1 = 256
DEFAULT_FC2 = 80
#: Dropout rate at the dense layers (paper: 0.5).
DEFAULT_DROPOUT = 0.5
#: L2 regularisation strength on the conv + dense kernels (paper: 1e-4).
DEFAULT_L2_LAMBDA = 1e-4


def _keras_same_pad_1d(kernel: int) -> tuple[int, int]:
    """Keras ``padding='same'`` ``(before, after)`` for a stride-1 axis (extra pad trails)."""
    total = kernel - 1
    before = total // 2
    return before, total - before


def _unit_average_power_normalize(x: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Per-signal unit-average-power normalisation of a ``(B, window, 2)`` IQ batch.

    Divides each signal by ``sqrt(mean_t(I^2 + Q^2))`` (the SEI-standard scale removal). Shared
    with :mod:`rfbench.models.baselines.wisig_cnn_paper`; see its docstring for the derivation.
    """
    power = (x**2).sum(dim=2).mean(dim=1)
    rms = torch.sqrt(power).clamp_min(eps)
    return cast("Tensor", x / rms[:, None, None])


class OracleCNNNet(nn.Module):
    """The ORACLE 2-conv + 2-FC CNN, operating on ``(B, window, 2)`` raw IQ.

    :meth:`forward` returns ``(B, num_classes)`` transmitter logits; :meth:`features` returns the
    ``(B, 80)`` penultimate embedding (ReLU'd ``Dense(80)``, before its dropout). :meth:`l2_penalty`
    exposes the sum of squared conv + dense kernels for the trainer's L2 term.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        dropout: float = DEFAULT_DROPOUT,
        input_norm: bool = True,
    ) -> None:
        """Build the two convs, the FC(256)->FC(80)->head, and record the input-norm flag."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.num_classes = num_classes
        self.window = window
        self.input_norm = input_norm

        # Conv1 (1,7): 'same' on time, keeps the height-2 (I/Q) axis; filters each stream.
        w_before, w_after = _keras_same_pad_1d(DEFAULT_KERNEL_TIME)
        self._pad1 = (w_before, w_after, 0, 0)  # (W_left, W_right, H_top, H_bottom)
        self.conv1 = nn.Conv2d(1, DEFAULT_CONV_FILTERS, kernel_size=(1, DEFAULT_KERNEL_TIME))
        # Conv2 (2,7): 'same' on time, VALID on height -> collapses the 2-row axis to 1 (mixes I/Q).
        self._pad2 = (w_before, w_after, 0, 0)
        self.conv2 = nn.Conv2d(
            DEFAULT_CONV_FILTERS,
            DEFAULT_CONV_FILTERS,
            kernel_size=(_IQ_CHANNELS, DEFAULT_KERNEL_TIME),
        )
        self.act = nn.ReLU(inplace=True)

        flat_dim = DEFAULT_CONV_FILTERS * 1 * window  # height collapses to 1, time preserved
        self.fc1 = nn.Linear(flat_dim, DEFAULT_FC1)
        self.fc2 = nn.Linear(DEFAULT_FC1, DEFAULT_FC2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(DEFAULT_FC2, num_classes)
        self.embed_dim = DEFAULT_FC2

    def _conv_features(self, x: Tensor) -> Tensor:
        """Run the two convs on a ``(B, window, 2)`` batch -> flattened ``(B, 50*window)``."""
        if self.input_norm:
            x = _unit_average_power_normalize(x)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, window, 2) -> (B, 2, window) -> (B, 1, 2, window)
        x = self.act(self.conv1(F.pad(x, self._pad1)))  # (B, 50, 2, window)
        x = self.act(self.conv2(F.pad(x, self._pad2)))  # (B, 50, 1, window)
        return x.flatten(1)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, 80)`` penultimate embedding for a ``(B, window, 2)`` IQ batch."""
        feats = self.dropout(self.act(self.fc1(self._conv_features(x))))
        return cast("Tensor", self.act(self.fc2(feats)))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` transmitter logits for a ``(B, window, 2)`` IQ batch."""
        feats = self.dropout(self.features(x))
        return cast("Tensor", self.classifier(feats))

    def l2_penalty(self) -> Tensor:
        """Sum of squared conv + dense KERNELS -- the L2 term the trainer adds to the loss."""
        return (
            self.conv1.weight.pow(2).sum()
            + self.conv2.weight.pow(2).sum()
            + self.fc1.weight.pow(2).sum()
            + self.fc2.weight.pow(2).sum()
            + self.classifier.weight.pow(2).sum()
        )


def _iq_to_bt2_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, window, 2)`` float tensor on ``device``."""
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[2] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, {window}, 2); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("oracle_cnn")
class OracleCNN(Model):
    """The ORACLE CNN SEI baseline as a :class:`~rfbench.core.model.Model` (``"oracle_cnn"``).

    Wraps :class:`OracleCNNNet` to the frozen ``Model`` contract: :meth:`forward` maps the
    collated batch (``x["iq"]`` a list of ``(window, 2)`` windows) to ``(B, n_tx)`` logits;
    :meth:`embed` returns the ``(B, 80)`` penultimate feature; :attr:`family` is ``"baseline"``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "oracle_cnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        input_norm: bool = True,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("OracleCNN needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = OracleCNNNet(num_classes, window=window, input_norm=input_norm).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, n_tx)`` transmitter logits for the collated SEI batch ``x``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, 80)`` penultimate embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_bt2_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "OracleCNN",
    "OracleCNNNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_DROPOUT",
    "DEFAULT_L2_LAMBDA",
]
