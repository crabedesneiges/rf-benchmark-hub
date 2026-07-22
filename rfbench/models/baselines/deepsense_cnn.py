"""DeepSense CNN spectrum-sensing baseline -- the Uvaydov et al. 2021 wideband occupancy net.

The canonical multi-label spectrum-occupancy CNN for the DeepSense dataset (Uvaydov, D'Oro,
Restuccia, Melodia, "DeepSense: Fast Wideband Spectrum Sensing Through Real-Time In-the-Loop Deep
Learning", IEEE INFOCOM 2021, DOI 10.1109/INFOCOM42981.2021.9488764; wineslab repo
https://github.com/wineslab/deepsense-spectrum-sensing-datasets). It maps a short ``(2, 32)``
raw-IQ window (2 channels = I/Q, 32 time samples) to ``K = 16`` INDEPENDENT per-subband occupancy
decisions (the 16 LTE-M sub-bands), i.e. a multi-label sigmoid head rather than a single softmax.

Verbatim architecture (``docs/BIBLIOGRAPHY.md`` §A.4)::

    input (2, 32)                     # I/Q as 2 channels, 32-sample window
    Conv1d(16, k=3, same, relu) x2    # two 16-filter k=3 convs over time
    MaxPool1d(2)                      # 32 -> 16
    Conv1d(32, k=5, same, relu) x2    # two 32-filter k=5 convs over time
    MaxPool1d(2)                      # 16 -> 8
    Flatten                           # 32 * 8 = 256
    Dense(64, relu)
    Dense(K)                          # per-subband occupancy LOGITS
    sigmoid                           # per-subband P(occupied)
    Adam(lr=1e-3), batch 256, 150 ep, binary cross-entropy

**Logits vs. probabilities.** The backing :class:`DeepSenseNet` returns raw ``(B, K)`` LOGITS so
the trainer (:mod:`rfbench.training_sensing`) can use the numerically-stable
``BCEWithLogitsLoss``. The :class:`Model` wrapper's :meth:`forward` applies the per-subband
sigmoid, so ``evaluate`` receives ``(B, K)`` per-subband ``P(occupied)`` in ``[0, 1]`` -- exactly
what :class:`rfbench.tasks.spectrum_sensing.metrics.OccupancyClassification` scores per
window×subband cell (it takes each per-subband value verbatim). :meth:`embed` returns the
``(B, 64)`` penultimate feature for the probing regimes.

**Input layout.** ``forward`` / ``embed`` receive the collated batch dict (``x["iq"]`` a list of
per-sample ``(2, window)`` channel-FIRST IQ windows -- the DeepSense ``X`` slice layout). The
``nn.Module`` consumes the natural ``(B, 2, window)`` tensor directly (``Conv1d`` is channel-first),
so the SAME tensor is fed at eval time (this wrapper) and train time
(:mod:`rfbench.training_sensing` collate).

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported at THIS module's top; ``@register_model("deepsense_cnn")`` fires only on an explicit
``import rfbench.models.baselines.deepsense_cnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: Number of occupancy sub-bands (the DeepSense LTE-M release has 16) -- the sigmoid head width.
DEFAULT_NUM_SUBBANDS = 16
#: DeepSense window length (time samples). The published lte_m ``X`` is ``(2, 32, N)`` -> 32.
DEFAULT_WINDOW = 32
#: Number of IQ channels (I and Q) -- the ``Conv1d`` input CHANNEL axis.
_IQ_CHANNELS = 2
#: Filter widths of the two conv stages (paper: 16 then 32).
_CONV1_FILTERS = 16
_CONV2_FILTERS = 32
#: Conv kernel sizes of the two stages (paper: k=3 then k=5).
_KERNEL1 = 3
_KERNEL2 = 5
#: Pooling factor after each conv stage (halves the time axis).
_POOL = 2
#: Penultimate dense width (paper: Dense(64)).
DEFAULT_FC = 64


class DeepSenseNet(nn.Module):
    """The DeepSense 4-conv + 1-FC CNN, operating on ``(B, 2, window)`` raw IQ.

    :meth:`forward` returns ``(B, num_subbands)`` per-subband occupancy LOGITS (no sigmoid -- the
    trainer applies ``BCEWithLogitsLoss``); :meth:`features` returns the ``(B, 64)`` penultimate
    embedding (the ReLU'd ``Dense(64)``). Two ``k=3`` convs (16 filters) + pool, then two ``k=5``
    convs (32 filters) + pool, flatten, ``Dense(64)`` and the ``Dense(K)`` occupancy head.
    """

    def __init__(
        self,
        num_subbands: int = DEFAULT_NUM_SUBBANDS,
        *,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        """Build the two conv stages, the pools, and the ``Dense(64) -> Dense(K)`` head."""
        super().__init__()
        if num_subbands < 1:
            raise ValueError(f"num_subbands must be >= 1, got {num_subbands}")
        if window < _POOL * _POOL:
            raise ValueError(f"window must be >= {_POOL * _POOL} (two /2 pools), got {window}")
        self.num_subbands = num_subbands
        self.window = window

        # Stage 1: two k=3 16-filter convs ('same' time padding), then /2 pool.
        self.conv1a = nn.Conv1d(_IQ_CHANNELS, _CONV1_FILTERS, _KERNEL1, padding="same")
        self.conv1b = nn.Conv1d(_CONV1_FILTERS, _CONV1_FILTERS, _KERNEL1, padding="same")
        # Stage 2: two k=5 32-filter convs ('same' time padding), then /2 pool.
        self.conv2a = nn.Conv1d(_CONV1_FILTERS, _CONV2_FILTERS, _KERNEL2, padding="same")
        self.conv2b = nn.Conv1d(_CONV2_FILTERS, _CONV2_FILTERS, _KERNEL2, padding="same")
        self.pool = nn.MaxPool1d(_POOL)
        self.act = nn.ReLU(inplace=True)

        pooled_time = window // _POOL // _POOL  # two /2 pools ('same' convs preserve time)
        flat_dim = _CONV2_FILTERS * pooled_time
        self.fc = nn.Linear(flat_dim, DEFAULT_FC)
        self.classifier = nn.Linear(DEFAULT_FC, num_subbands)
        self.embed_dim = DEFAULT_FC

    def _conv_features(self, x: Tensor) -> Tensor:
        """Run both conv stages on ``(B, 2, window)`` -> flattened ``(B, 32 * window/4)``."""
        x = self.act(self.conv1a(x))
        x = self.act(self.conv1b(x))
        x = self.pool(x)
        x = self.act(self.conv2a(x))
        x = self.act(self.conv2b(x))
        x = self.pool(x)
        return x.flatten(1)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, 64)`` penultimate embedding for a ``(B, 2, window)`` IQ batch."""
        return cast("Tensor", self.act(self.fc(self._conv_features(x))))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_subbands)`` per-subband occupancy LOGITS (sigmoid downstream)."""
        return cast("Tensor", self.classifier(self.features(x)))


def _iq_to_bcw_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, window)`` channel-first float tensor."""
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single (2, window) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("deepsense_cnn")
class DeepSenseCNN(Model):
    """The DeepSense CNN spectrum-sensing baseline as a :class:`Model` (``"deepsense_cnn"``).

    Wraps :class:`DeepSenseNet` to the frozen ``Model`` contract: :meth:`forward` maps the collated
    batch (``x["iq"]`` a list of ``(2, window)`` windows) to ``(B, num_subbands)`` per-subband
    ``P(occupied)`` (sigmoid over the net's logits), :meth:`embed` returns the ``(B, 64)``
    penultimate feature, and :attr:`family` is ``"baseline"``. The sigmoid lives HERE (not in the
    net) so the trainer keeps the raw logits for ``BCEWithLogitsLoss``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "deepsense_cnn",
        num_subbands: int = DEFAULT_NUM_SUBBANDS,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("DeepSenseCNN needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = DeepSenseNet(num_subbands, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_subbands)`` per-subband ``P(occupied)`` for the collated batch ``x``."""
        iq = _iq_to_bcw_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net.forward(iq))

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, 64)`` penultimate embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_bcw_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "DeepSenseCNN",
    "DeepSenseNet",
    "DEFAULT_NUM_SUBBANDS",
    "DEFAULT_WINDOW",
    "DEFAULT_FC",
]
