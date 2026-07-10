"""Paper-faithful WiSig ManyTx SEI baseline -- the exact 2-D CNN of Hanna et al. 2022.

This is the byte-faithful reconstruction of the official WiSig ManyTx transmitter classifier
``create_net`` in ``py/d006_ManyTx_ntx.py`` of ``github.com/WiSig-dataset/wisig-examples``
(default branch ``master``; the shared ``data_utilities.py`` lives at the repo root). Every
layer, kernel, pool, activation, regulariser and the input normalisation were read verbatim
from that file and the paper (Hanna, Karunaratne, Cabric, *IEEE Access* 10:22808-22818, 2022,
arXiv:2112.15363, DOI 10.1109/ACCESS.2022.3154790). See ``docs/BIBLIOGRAPHY.md`` B.4 for the
line-by-line audit. The compact 1-D :mod:`rfbench.models.baselines.sei_cnn` (``wisig_cnn``)
stays as a board-seeding variant; THIS module is the one that reproduces the paper.

Verbatim Keras (``create_net``)::

    Input(shape=(256,2)); Reshape((256,2,1))
    Conv2D(8 ,(3,2),relu,'same'); MaxPool2D((2,1))
    Conv2D(16,(3,2),relu,'same'); MaxPool2D((2,1))
    Conv2D(16,(3,2),relu,'same'); MaxPool2D((2,2))
    Conv2D(32,(3,1),relu,'same'); MaxPool2D((2,1))
    Conv2D(16,(3,1),relu,'same')                     # NB: 5th conv is UNPOOLED
    Flatten()
    Dense(100,relu, l2(1e-4)); Dense(80,relu, l2(1e-4)); Dropout(0.5)
    Dense(N ,softmax, l2(1e-4))
    compile(loss='categorical_crossentropy', optimizer=Adam(5e-4))

Three fidelity points our earlier doc got wrong and this module fixes (confirmed against the
raw master file): (a) **L2 is on the three Dense layers ONLY**, never on any conv -- exposed
here via :meth:`WiSigCNNPaperNet.l2_penalty` so the SEI trainer can add ``l2_lambda * penalty``
to the loss exactly like Keras' ``kernel_regularizer=l2(1e-4)``; (b) there are only **four**
max-pools -- the fifth conv feeds Flatten directly; (c) ``padding='same'`` is reproduced with
the **Keras asymmetric** convention (extra pad on the bottom/right for even kernels) so the
(3,2) convs align sample-for-sample with the reference.

**Input layout convention (shared across the SEI baselines).** ``forward`` / ``embed`` receive
the COLLATED batch dict :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` a *list* of
per-sample ``(256, 2)`` WiSig windows (time-major; see ``rfbench/data/prepare/sei.py``). The
network's ``nn.Module`` accepts the natural ``(B, 256, 2)`` tensor and does the unit-average-
power normalisation + ``(B, 1, 256, 2)`` reshape itself, so the SAME tensor is fed at eval time
(via the Model wrapper) and at train time (via :mod:`rfbench.training_sei`'s collate) -- there
is no layout skew between the two paths.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module
imports ``torch`` at its top and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``@register_model("wisig_cnn_paper")`` fires only on an
explicit ``import rfbench.models.baselines.wisig_cnn_paper``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: WiSig ManyTx transmitter count -- the default closed-set head width (paper: up to 150 Tx).
#: The concrete class count comes from the prepared split on the cluster, so this is a fallback.
DEFAULT_NUM_CLASSES = 150
#: Canonical WiSig Id-signal length: the first 256 IQ samples of the preamble.
DEFAULT_WINDOW = 256
#: Number of IQ channels (I and Q).
_IQ_CHANNELS = 2
#: L2 regularisation strength on the Dense kernels (Keras ``l2(1e-4)`` in ``create_net``).
DEFAULT_L2_LAMBDA = 1e-4
#: Dropout rate between Dense(80) and the softmax output (``Dropout(0.5)``).
DEFAULT_DROPOUT = 0.5

#: The five conv blocks as ``(out_channels, (kernel_h, kernel_w))`` -- filters 8/16/16/32/16,
#: kernels (3,2)x3 then (3,1)x2, ``padding='same'`` on every one (verbatim ``create_net``).
_CONV_SPEC: tuple[tuple[int, tuple[int, int]], ...] = (
    (8, (3, 2)),
    (16, (3, 2)),
    (16, (3, 2)),
    (32, (3, 1)),
    (16, (3, 1)),
)
#: The four max-pool windows ``(pool_h, pool_w)`` applied after convs 1-4; the 5th conv is
#: unpooled (``None``). Verbatim: (2,1),(2,1),(2,2),(2,1).
_POOL_SPEC: tuple[tuple[int, int] | None, ...] = ((2, 1), (2, 1), (2, 2), (2, 1), None)
#: Dense head widths before the classifier (``Dense(100)`` then ``Dense(80)``).
_DENSE_100 = 100
_DENSE_80 = 80


def _keras_same_pad(kernel: int) -> tuple[int, int]:
    """Return ``(before, after)`` zero-padding reproducing Keras ``padding='same'`` (stride 1).

    Keras pads a stride-1 axis by ``k-1`` total, split ``before = (k-1)//2`` and the remainder
    ``after`` -- so an even kernel puts the extra pad on the trailing (bottom/right) edge. This
    matters for the WiSig (3,2) convs: torch's own ``padding='same'`` places the asymmetry on
    the *leading* edge, which would shift the width-2 IQ axis by one relative to the reference.
    """
    total = kernel - 1
    before = total // 2
    return before, total - before


class _SameConv2d(nn.Module):
    """One ``Conv2D(out, kernel, activation='relu', padding='same')`` block, Keras-faithful.

    Applies explicit Keras-style asymmetric zero-padding (:func:`_keras_same_pad` per axis)
    then a ``padding=0`` conv, so the output keeps the input's (H, W) exactly as Keras' ``'same'``
    does -- including the trailing-edge asymmetry for the even (…,2) width kernels. ReLU is the
    conv's activation, matching ``activation='relu'`` on every WiSig conv.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int]) -> None:
        """Build the same-padding + conv + ReLU for one block."""
        super().__init__()
        kh, kw = kernel
        h_before, h_after = _keras_same_pad(kh)
        w_before, w_after = _keras_same_pad(kw)
        # F.pad order is (W_left, W_right, H_top, H_bottom).
        self._pad = (w_before, w_after, h_before, h_after)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in, H, W)`` -> ``(B, out, H, W)`` (length-preserving same conv + ReLU)."""
        return cast("Tensor", self.act(self.conv(F.pad(x, self._pad))))


def _unit_average_power_normalize(x: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Divide each ``(W, 2)`` signal by its RMS power -- WiSig ``norm()``, verbatim.

    Reference (``data_utilities.py``)::

        pwr = np.sqrt(np.mean(np.sum(sig_u**2, axis=-1), axis=-1))   # per-signal scalar
        sig_u = sig_u / pwr[:, None, None]

    For a ``(B, W, 2)`` batch: sum I^2+Q^2 over the IQ axis, mean over the W time steps, sqrt ->
    one scalar RMS power per signal; divide the whole signal by it. This removes the capture's
    absolute scale (which carries no transmitter identity) while preserving the I/Q geometry
    that does. It divides by the scalar power ONLY -- no mean subtraction, no whitening -- exactly
    like the paper ("all signals were normalized to have unit average power"). ``eps`` guards a
    degenerate all-zero signal.
    """
    power = (x**2).sum(dim=2).mean(dim=1)  # (B,)
    rms = torch.sqrt(power).clamp_min(eps)  # (B,)
    return cast("Tensor", x / rms[:, None, None])


class WiSigCNNPaperNet(nn.Module):
    """The paper-exact WiSig ManyTx 2-D CNN (``create_net``), operating on ``(B, 256, 2)`` IQ.

    :meth:`forward` returns ``(B, num_classes)`` transmitter logits (pre-softmax -- the softmax
    is folded into the training cross-entropy and the eval argmax). :meth:`features` returns the
    ``(B, 80)`` penultimate embedding (the ReLU'd ``Dense(80)`` output, before dropout) for the
    ``linear_probe`` / ``few_shot`` regimes. :meth:`l2_penalty` exposes the sum of squared Dense
    kernels so the trainer can reproduce Keras' ``l2(1e-4)`` on those three layers only.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Build the 5-conv stack, the 4 pools and the Dense(100)->Dense(80)->Dropout->head."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.num_classes = num_classes
        self.window = window

        layers: list[nn.Module] = []
        in_ch = 1  # Reshape((256,2,1)) -> single input channel, H=256, W=2
        for (out_ch, kernel), pool in zip(_CONV_SPEC, _POOL_SPEC, strict=True):
            layers.append(_SameConv2d(in_ch, out_ch, kernel))
            if pool is not None:
                layers.append(nn.MaxPool2d(pool))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        flat_dim = self._infer_flat_dim(window)
        self.dense_100 = nn.Linear(flat_dim, _DENSE_100)
        self.dense_80 = nn.Linear(_DENSE_100, _DENSE_80)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(_DENSE_80, num_classes)
        self.embed_dim = _DENSE_80

    def _infer_flat_dim(self, window: int) -> int:
        """Return the flattened conv-output width for a ``(1, window, 2)`` input (shape probe)."""
        with torch.no_grad():
            probe = torch.zeros(1, 1, window, _IQ_CHANNELS)
            return int(self.conv(probe).flatten(1).shape[1])

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, 80)`` penultimate embedding for a ``(B, window, 2)`` IQ batch."""
        x = _unit_average_power_normalize(x)  # per-signal unit average power
        x = x.unsqueeze(1)  # (B, 1, window, 2)  == Keras Reshape((window,2,1)) channels-last
        feats = self.conv(x).flatten(1)
        feats = F.relu(self.dense_100(feats))
        return cast("Tensor", F.relu(self.dense_80(feats)))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` transmitter logits for a ``(B, window, 2)`` IQ batch."""
        feats = self.dropout(self.features(x))
        return cast("Tensor", self.classifier(feats))

    def l2_penalty(self) -> Tensor:
        """Sum of squared Dense KERNELS -- the term Keras' ``l2(1e-4)`` adds to the loss.

        Only the three Dense layers (100, 80, softmax) carry ``kernel_regularizer=l2(1e-4)`` in
        ``create_net``; no conv does. Biases are excluded (Keras regularises the kernel only).
        The trainer adds ``l2_lambda * l2_penalty()`` to the cross-entropy, reproducing the
        reference exactly (rather than torch's coupled ``weight_decay`` on every parameter).
        """
        return (
            self.dense_100.weight.pow(2).sum()
            + self.dense_80.weight.pow(2).sum()
            + self.classifier.weight.pow(2).sum()
        )


def _iq_to_bt2_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, window, 2)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.sei.dataset.SeiDataset`: each element is a ``(window, 2)`` array-like
    (numpy on the cluster, nested lists in a fixture). A single unbatched ``(window, 2)`` sample
    is promoted to a batch of one. A mis-shaped batch (IQ axis != 2) fails loudly.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single (window, 2) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[2] != _IQ_CHANNELS:
        raise ValueError(f"expected IQ batch of shape (B, {window}, 2); got {tuple(tensor.shape)}")
    return tensor.contiguous()


@register_model("wisig_cnn_paper")
class WiSigCNNPaper(Model):
    """The paper-exact WiSig ManyTx CNN as a :class:`~rfbench.core.model.Model`.

    Wraps :class:`WiSigCNNPaperNet` to the frozen ``Model`` contract: :meth:`forward` maps the
    collated SEI batch (``x["iq"]`` a list of ``(256, 2)`` windows) to ``(B, n_tx)`` logits;
    :meth:`embed` returns the ``(B, 80)`` penultimate feature; :attr:`n_params` reports the
    trainable count; :attr:`family` is ``"baseline"``. Eval runs in :meth:`eval` mode with no
    grad; :mod:`rfbench.training_sei` loads trained weights into :attr:`net` first and should
    pass ``num_classes=`` the prepared split's transmitter count.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "wisig_cnn_paper",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("WiSigCNNPaper needs a non-empty name")
        self.name = name
        self.window = window
        resolved = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved)
        self.net = WiSigCNNPaperNet(num_classes, window=window).to(self.device)

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
    "WiSigCNNPaper",
    "WiSigCNNPaperNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_L2_LAMBDA",
    "DEFAULT_DROPOUT",
]
