"""CLDNN AMC baseline (WP-30) -- the Convolutional-LSTM-DNN modulation classifier.

CLDNN ("Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks",
Sainath et al., ICASSP 2015; adapted to RF AMC by West & O'Shea, "Deep Architectures for
Modulation Recognition", IEEE DySPAN 2017) is a single-stream complement to the three-branch
MCLDNN on RadioML 2016.10a. Unlike MCLDNN's spatiotemporal multi-channel fusion, CLDNN runs
ONE view of the IQ window through a short stack of 1-D convolutions (extracting local
time-frequency structure across both channels), feeds the resulting feature sequence to an
LSTM (modelling temporal dependencies), and classifies the LSTM's last hidden state with two
dense layers into the 11 RML2016.10a modulation classes. Like MCLDNN it is deliberately small
(a few hundred k params), which is why it seeds the AMC board rather than a heavy backbone.

Paper-exact fidelity (West & O'Shea 2017, DySPAN, arXiv:1703.09197). Two load-bearing features
of the published CLDNN are reproduced here (see ``docs/BIBLIOGRAPHY.md`` §B.2):

* the **raw-waveform bypass/skip** -- the paper concatenates the *raw* IQ waveform with the
  convolutional feature maps before the recurrent stack, so the LSTM sees both the learned
  local features AND the untouched signal (a DenseNet-style forward bypass), and
* a **three-layer stacked LSTM** (the paper stacks THREE LSTMs; a lighter re-impl using two
  loses ~1-2 pts of overall accuracy on RML2016.10a).

The skip means the LSTM's per-timestep input width is ``conv_filters + 2`` (conv features plus
the two raw IQ channels), not ``conv_filters``.

From-scratch conditioning (chance-collapse fix, ``input_norm``, default on). The paper does not
specify input scaling, and the naive choice makes this exact architecture *fragile* on RML2016.10a:
the IQ is ~unit-power, so raw per-sample values are tiny (~1e-2 RMS), and with no BatchNorm that
near-zero signal -- fed through the conv front end AND, via the skip, straight into a THREE-layer
stacked LSTM -- lets the deep recurrent stack collapse to a class-independent (constant-class, 1/11)
output for *some* weight-init draws under the long-schedule recipe (verified: the un-normalized net
collapsed on the board's unseeded init, yet learns on seed 42). **Per-sample unit-variance input
normalization** (the same transform ResNet uses, which cured ResNet's identical 1/11 collapse; see
:func:`_unit_variance_normalize`) removes the fragility -- with a real input scale the LSTM cannot
ignore the (tiny) input, so it learns robustly regardless of the init draw. A per-epoch diagnostic
(``slurm/diagnose_cldnn.py``) confirmed normalization is necessary AND sufficient (val-acc 0.58 vs
0.09 without it); it also **ruled out** a forget-gate-bias/orthogonal LSTM re-init, which is inert
once the input is normalized and actively *causes* the collapse when applied without it.
MCLDNN/ResNet are untouched.

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the
COLLATED batch dict that :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` is a
*list* of per-sample IQ payloads, one per sample. :class:`~rfbench.tasks.amc.dataset.AmcDataset`
yields RML2016.10a windows of shape ``(2, 128)`` (I on row 0, Q on row 1; see
``rfbench/data/prepare/amc.py`` ``[N, 2, 128]``), so the collated ``x["iq"]`` is
``list[ (2, 128) ]`` and :func:`_iq_to_tensor` stacks it into a ``(B, 2, 128)`` float tensor.
``forward`` returns ``(B, 11)`` class logits; iterating that tensor yields one per-class score
vector per sample, exactly what the AMC metrics' ``argmax`` decoder consumes. ``embed`` returns
the ``(B, 128)`` penultimate feature vector for the ``linear_probe`` / ``few_shot`` regimes.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module
is a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("cldnn")`` entry in :data:`rfbench.core.registry.MODELS` is created only on
an explicit ``import rfbench.models.baselines.cldnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: RadioML 2016.10a modulation classes (the 11-way closed set CLDNN classifies).
DEFAULT_NUM_CLASSES = 11
#: Canonical RML2016.10a IQ window length (samples per channel).
DEFAULT_WINDOW = 128
#: Convolution feature width shared by the conv stack (West & O'Shea use ~50-64 filters).
DEFAULT_CONV_FILTERS = 64
#: Number of stacked 1-D conv layers over the IQ window.
DEFAULT_CONV_LAYERS = 3
#: 1-D conv kernel length along the time axis (odd -> length-preserving with "same" padding).
DEFAULT_KERNEL = 7
#: Number of stacked LSTM layers. West & O'Shea (2017) stack THREE; a two-layer re-impl underfits.
DEFAULT_LSTM_LAYERS = 3
#: Hidden width of each LSTM layer; also the penultimate feature width returned by :meth:`embed`.
DEFAULT_LSTM_HIDDEN = 128
#: Number of raw IQ channels concatenated onto the conv features by the raw-waveform skip (I, Q).
_RAW_IQ_CHANNELS = 2


def _same_pad_1d(kernel: int) -> int:
    """Return the symmetric ``padding`` that keeps a 1-D conv's length unchanged (odd kernel)."""
    return (kernel - 1) // 2


def _unit_variance_normalize(x: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Standardise each ``(2, L)`` IQ window to zero mean and unit variance (per-sample).

    RML2016.10a is distributed with each example normalised to ~unit average POWER, so the raw
    per-sample IQ is tiny (RMS on the order of 1e-2), and CLDNN has neither BatchNorm nor any input
    scaling. Feeding that near-zero-scale signal straight into a 3-layer stacked LSTM (and, via the
    raw-waveform skip, again alongside the equally-tiny conv features) starts the recurrence in a
    dead-gate regime from which the deep stack settles into a constant-class (chance, 1/11) output
    it never leaves under the long-schedule recipe. Standardising each window to unit variance --
    the SAME transform ``resnet_amc._unit_variance_normalize`` applies, which cured ResNet's
    identical 1/11 collapse -- restores a healthy activation scale. The statistics are taken over
    BOTH channels and the whole time axis of each sample independently (dims ``(1, 2)``), so the
    absolute capture scale (which carries no modulation information) is removed while the I/Q
    geometry that does is preserved. ``eps`` guards an all-constant (zero-variance) window.

    Duplicated here (rather than imported from the ResNet baseline) so this module stays standalone
    and never depends on a sibling baseline's import side effects -- the same rationale as
    :func:`_iq_to_tensor`.
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
    return cast("Tensor", (x - mean) / (std + eps))


class CLDNNNet(nn.Module):
    """The CLDNN convolutional-LSTM-DNN network (Sainath et al. 2015; West & O'Shea 2017).

    One IQ window (the two channels as a ``(B, 2, L)`` signal) is run through a short stack of
    length-preserving 1-D convolutions; the **raw waveform is then concatenated back onto the
    conv feature sequence** (the paper's forward bypass/skip) and the combined length-``L``
    sequence is passed through a **three-layer stacked LSTM** (West & O'Shea 2017). The LSTM's
    final hidden state is projected by a dense layer (the penultimate embedding) and a
    classifier to ``num_classes`` logits.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, embed_dim)`` penultimate representation the probing regimes fit a head on.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        conv_filters: int = DEFAULT_CONV_FILTERS,
        conv_layers: int = DEFAULT_CONV_LAYERS,
        kernel: int = DEFAULT_KERNEL,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        lstm_layers: int = DEFAULT_LSTM_LAYERS,
        input_norm: bool = True,
    ) -> None:
        """Build the 1-D conv stack, the raw-waveform skip, the stacked LSTM and the dense head.

        ``input_norm`` (default ``True``, the CLDNN-scoped chance-collapse fix) per-sample
        unit-variance normalises the IQ window before the conv AND the raw-waveform skip (see
        :func:`_unit_variance_normalize`); set it ``False`` to reproduce the earlier fragile
        (raw-IQ) behaviour used by the diagnostic ablation.
        """
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if conv_layers < 1:
            raise ValueError(f"conv_layers must be >= 1, got {conv_layers}")
        if lstm_layers < 1:
            raise ValueError(f"lstm_layers must be >= 1, got {lstm_layers}")
        self.num_classes = num_classes
        self.window = window
        self.conv_filters = conv_filters
        self.conv_layers = conv_layers
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.input_norm = input_norm

        # Convolutional front end: a stack of length-preserving 1-D convs over the (2, L) IQ.
        # The first layer maps the 2 IQ channels to ``conv_filters``; the rest keep that width.
        pad = _same_pad_1d(kernel)
        conv_blocks: list[nn.Module] = []
        in_ch = 2
        for _ in range(conv_layers):
            conv_blocks.append(nn.Conv1d(in_ch, conv_filters, kernel_size=kernel, padding=pad))
            conv_blocks.append(nn.ReLU(inplace=True))
            in_ch = conv_filters
        self.conv = nn.Sequential(*conv_blocks)

        # Raw-waveform bypass/skip (West & O'Shea 2017): the untouched IQ waveform is concatenated
        # onto the conv feature sequence before the LSTM, so each timestep the LSTM consumes is
        # ``conv_filters`` learned features PLUS the two raw IQ channels -> width conv_filters + 2.
        self.lstm_input_size = conv_filters + _RAW_IQ_CHANNELS
        # The combined (B, L, conv_filters + 2) sequence -> a three-layer stacked LSTM (paper: 3).
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Penultimate dense layer (the embedding) + the classifier head.
        self.fc_embed = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.SELU(inplace=True),
        )
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def _conv_sequence(self, x: Tensor) -> Tensor:
        """Run the conv stack + raw-waveform skip, returning the ``(B, L, F+2)`` LSTM sequence.

        ``x`` is a ``(B, 2, L)`` IQ batch (row 0 = I, row 1 = Q). The 1-D conv stack keeps the
        time axis at length ``L`` and produces ``(B, conv_filters, L)``. Following West & O'Shea
        (2017), the **raw IQ waveform is concatenated back onto** these features along the channel
        axis (the forward bypass/skip) before the whole thing is transposed into the
        ``(B, L, conv_filters + 2)`` layout ``nn.LSTM(batch_first=True)`` expects -- so the LSTM
        sees both the learned local features and the untouched signal at every timestep.

        When ``input_norm`` is set the window is first per-sample unit-variance normalised, so BOTH
        the conv front end and the raw-waveform skip see a healthy ~unit-scale IQ rather than the
        raw ~1e-2-RMS signal that stalls the deep LSTM at chance (:func:`_unit_variance_normalize`).
        """
        if self.input_norm:
            x = _unit_variance_normalize(x)
        feat = self.conv(x)  # (B, conv_filters, L)
        # Raw-waveform bypass: concat the (normalised) (B, 2, L) IQ onto the conv channels.
        fused = torch.cat((feat, x), dim=1)  # (B, conv_filters + 2, L)
        return fused.transpose(1, 2)  # (B, L, conv_filters + 2)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, lstm_hidden)`` penultimate embedding for a ``(B, 2, L)`` batch."""
        seq = self._conv_sequence(x)  # (B, L, conv_filters + 2) -- conv features + raw-IQ skip
        _out, (h_n, _c_n) = self.lstm(seq)
        last = h_n[-1]  # (B, lstm_hidden) -- final layer's last hidden state
        embedded = self.fc_embed(last)  # (B, lstm_hidden)
        return cast("Tensor", embedded)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` logits for a ``(B, 2, L)`` IQ batch."""
        logits = self.classifier(self.features(x))
        return cast("Tensor", logits)


def _iq_to_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, window)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample IQ list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.amc.dataset.AmcDataset`: each element is a ``(2, window)`` array-like
    (numpy on the cluster, nested lists in a synthetic fixture). ``torch.as_tensor`` handles
    both; the result is coerced to ``float32`` and validated to the expected ``(B, 2, window)``
    shape so a mis-shaped batch fails loudly rather than silently mis-classifying.

    Mirrors :func:`rfbench.models.baselines.mcldnn._iq_to_tensor` -- the AMC baselines share the
    same collated-batch contract, and the helper is duplicated (rather than imported) so this
    module stands alone and never depends on a sibling baseline's import side effects.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, window) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != 2:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor


@register_model("cldnn")
class CLDNN(Model):
    """The CLDNN AMC baseline as a :class:`~rfbench.core.model.Model` (registered ``"cldnn"``).

    Wraps :class:`CLDNNNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, window)`` IQ
      windows from :class:`~rfbench.tasks.amc.dataset.AmcDataset`) to ``(B, num_classes)``
      logits -- iterated per-sample by the AMC metrics.
    * :meth:`embed` returns the ``(B, lstm_hidden)`` penultimate feature vector for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("cldnn")()`` on the registry path. Eval runs
    in :meth:`eval` mode with gradients disabled; a from-scratch training loop (M3) loads weights
    into :attr:`net` before evaluation.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "cldnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
        input_norm: bool = True,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU).

        ``input_norm`` (default ``True``) is the CLDNN-scoped chance-collapse fix, passed through to
        :class:`CLDNNNet`; the no-arg registry path ``MODELS.get("cldnn")()`` therefore builds the
        fixed model. Pass ``False`` to reproduce the earlier fragile (raw-IQ) configuration (the
        diagnostic does this for its ``broken`` variant).
        """
        if not name:
            raise ValueError("CLDNN needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = CLDNNNet(num_classes, window=window, input_norm=input_norm).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` class logits for the collated AMC batch ``x``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, lstm_hidden)`` embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "CLDNN",
    "CLDNNNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_CONV_FILTERS",
    "DEFAULT_CONV_LAYERS",
    "DEFAULT_KERNEL",
    "DEFAULT_LSTM_LAYERS",
    "DEFAULT_LSTM_HIDDEN",
]
