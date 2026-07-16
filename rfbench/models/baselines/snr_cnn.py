"""1-D CNN SNR regressor baseline (``snr_cnn``) -- the first *deep* snr_estimation baseline.

The seed SNR baselines (:mod:`rfbench.models.baselines.snr_regressors`) are sklearn/stdlib
regressors: the ``mean_snr`` floor and the ``snr_moment_ridge`` DSP reference (test RMSE
7.64 dB). This adds the deep counterpart the board was missing -- a small 1-D convolutional
network that maps a raw-IQ window directly to a scalar SNR (dB), trained by MSE regression on
the ``snr_db`` target (see :mod:`rfbench.training_snr`). It is meant to BEAT the DSP ridge, the
same way the deep SEI/AMC baselines beat their hand-designed-feature counterparts.

**Input layout convention (AMC / SNR channel-first).** ``forward`` receives the collated batch
(``x["iq"]`` a list of ``(2, window)`` IQ windows: I on row 0, Q on row 1 -- the RadioML
2016.10a layout the AMC/SNR loaders yield, see :func:`rfbench.tasks.amc.dataset._load_amc_arrays`
and :mod:`rfbench.models.baselines.mcldnn`). The ``nn.Module`` consumes ``(B, 2, window)``
directly, so the SAME tensor is fed at eval time (this ``Model`` wrapper) and at train time
(:mod:`rfbench.training_snr`) -- no train/eval layout skew. NOTE this is the ``(2, window)``
channel-first convention, NOT the SEI ``(window, 2)`` time-major one.

The head emits ONE scalar per window (the predicted SNR in dB); :meth:`forward` returns a
``(B,)`` tensor which the regression metrics (``Rmse`` / ``Mae`` in
:mod:`rfbench.tasks.snr_estimation.metrics`) consume per-sample. :attr:`regime` is reported by
the trainer as ``from_scratch`` (weights learned only from the task's own train split).

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported at THIS module's top; ``@register_model("snr_cnn")`` fires only on an explicit
``import rfbench.models.baselines.snr_cnn``.
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

#: Nominal IQ window length (RadioML 2016.10a is 128). The global pool makes the net
#: window-agnostic, so this is only a default for the no-arg registry construction.
DEFAULT_WINDOW = 128
#: Number of IQ channels (I on row 0, Q on row 1) -- the stem conv's input channel count.
_IQ_CHANNELS = 2
#: Convolutional stage widths (three conv blocks, doubling width).
_STAGE_WIDTHS: tuple[int, int, int] = (64, 128, 256)
#: Convolution kernel size (odd, "same"-padded so a block preserves the length before pooling).
_KERNEL = 5
#: Penultimate (regression-head input) width -- the embedding dimension.
DEFAULT_EMBED_DIM = 128


def _unit_average_power_normalize(x: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Per-signal unit-average-power normalisation of a ``(B, 2, window)`` IQ batch.

    Divides each signal by ``sqrt(mean_t(I^2 + Q^2))`` so the absolute capture scale (which
    carries no SNR information on its own) is removed before the deep stack -- the standard
    raw-IQ transform (mirrors :mod:`rfbench.models.baselines.resnet1d_sei`). Load-bearing:
    without it a deep raw-IQ net keys on capture gain rather than the noise structure.
    """
    power = (x**2).sum(dim=1).mean(dim=1)  # mean_t(I^2 + Q^2) over the (2, window) map -> (B,)
    rms = torch.sqrt(power).clamp_min(eps)
    return cast("Tensor", x / rms[:, None, None])


class _ConvBlock(nn.Module):
    """A conv-BN-ReLU block over ``(B, C_in, L)`` followed by a stride-2 max-pool (halves L).

    The convolution is "same"-padded so it keeps the length; the pool downsamples by 2. Standard
    1-D feature-extraction block, sized small so the model trains quickly on the cluster GPU.
    """

    def __init__(self, in_ch: int, out_ch: int, *, kernel: int = _KERNEL) -> None:
        """Build the conv-BN and record the pool (kernel must be odd for symmetric padding)."""
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd for symmetric 'same' padding, got {kernel}")
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in, L)`` -> ``(B, out, L // 2)`` (conv-BN-ReLU then a stride-2 pool)."""
        return cast("Tensor", self.pool(F.relu(self.bn(self.conv(x)), inplace=True)))


class SnrCNNNet(nn.Module):
    """A small 1-D CNN over ``(B, 2, window)`` raw IQ regressing to a scalar SNR (dB).

    :meth:`forward` returns a ``(B,)`` predicted-SNR tensor; :meth:`features` returns the
    ``(B, DEFAULT_EMBED_DIM)`` penultimate embedding (for a ``linear_probe`` regime). The
    adaptive average pool makes the head width independent of the window length.
    """

    def __init__(self, *, window: int = DEFAULT_WINDOW, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        """Build the three conv blocks, the global pool and the two-layer regression head."""
        super().__init__()
        if window < 8:
            raise ValueError(f"window must be >= 8, got {window}")
        self.window = window
        self.embed_dim = embed_dim

        blocks: list[nn.Module] = []
        in_ch = _IQ_CHANNELS
        for width in _STAGE_WIDTHS:
            blocks.append(_ConvBlock(in_ch, width))
            in_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(_STAGE_WIDTHS[-1], embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, embed_dim)`` penultimate embedding for a ``(B, 2, window)`` batch."""
        x = _unit_average_power_normalize(x)
        x = self.blocks(x)
        pooled = self.pool(x).squeeze(-1)  # (B, C)
        return cast("Tensor", F.relu(self.fc(pooled), inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Return the ``(B,)`` predicted SNR (dB) for a ``(B, 2, window)`` IQ batch."""
        return cast("Tensor", self.head(self.features(x)).squeeze(-1))


def _iq_to_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, window)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.snr_estimation.dataset.SnrDataset`: each element is a ``(2, window)``
    array-like (numpy on the cluster, nested lists in a synthetic fixture).
    """
    # Collate the per-sample list into ONE ndarray first: ``torch.as_tensor`` on a list of
    # ndarrays copies element-by-element ("extremely slow" per torch) and would dominate eval
    # wall-time. ``np.asarray`` stacks in one shot; the resulting tensor is numerically identical.
    if isinstance(iq_batch, (list, tuple)):
        iq_batch = np.asarray(iq_batch)
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, window) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("snr_cnn")
class SnrCNN(Model):
    """The 1-D CNN SNR regressor as a :class:`~rfbench.core.model.Model` (``"snr_cnn"``).

    Wraps :class:`SnrCNNNet` to the frozen ``Model`` contract: :meth:`forward` maps the collated
    batch (``x["iq"]`` a list of ``(2, window)`` windows) to a ``(B,)`` predicted-SNR (dB) tensor
    the regression metrics consume per-sample; :meth:`embed` returns the penultimate feature;
    :attr:`family` is ``"baseline"``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "snr_cnn",
        window: int = DEFAULT_WINDOW,
        embed_dim: int = DEFAULT_EMBED_DIM,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("SnrCNN needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = SnrCNNNet(window=window, embed_dim=embed_dim).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return the ``(B,)`` predicted SNR (dB) for the collated SNR batch ``x``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def predict(self, x: Batch) -> Tensor:
        """Alias of :meth:`forward` -- the predicted SNR (dB) per sample (regression output)."""
        return self.forward(x)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, embed_dim)`` penultimate embedding for a probing regime."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "SnrCNN",
    "SnrCNNNet",
    "DEFAULT_WINDOW",
    "DEFAULT_EMBED_DIM",
]
