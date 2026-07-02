"""MCLDNN AMC baseline (WP-30) -- the standard RML2016.10a modulation classifier.

MCLDNN ("A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation
Classification", Xu et al., IEEE Wireless Commun. Lett. 2020) is the canonical from-scratch
baseline on RadioML 2016.10a. It fuses THREE input views of an IQ window:

* a **combined I/Q** view (both channels together) through a 2-D convolution, and
* **separate I** and **separate Q** views, each through its own 1-D convolution,

concatenates the three feature maps, refines them with a further 2-D convolution, then feeds
the resulting sequence to a two-layer LSTM whose last state is classified by two dense layers
into the 11 RML2016.10a modulation classes. The design is small (~a few hundred k params),
which is why it seeds the AMC board rather than a heavy backbone.

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
``@register_model("mcldnn")`` entry in :data:`rfbench.core.registry.MODELS` is created only on
an explicit ``import rfbench.models.baselines.mcldnn``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: RadioML 2016.10a modulation classes (the 11-way closed set MCLDNN classifies).
DEFAULT_NUM_CLASSES = 11
#: Canonical RML2016.10a IQ window length (samples per channel).
DEFAULT_WINDOW = 128
#: Convolution feature width shared by the three input branches (Xu et al. use 50 filters).
DEFAULT_CONV_FILTERS = 50
#: Filter count of the fusion conv (Xu et al. use 100 -- twice the per-branch width); its output
#: channel axis becomes the LSTM ``input_size``. See BIBLIOGRAPHY.md B.1 (fusion = 100 filters).
DEFAULT_FUSE_FILTERS = 100
#: Hidden width of each LSTM layer; also the penultimate feature width returned by :meth:`embed`.
DEFAULT_LSTM_HIDDEN = 128
#: Dropout probability in the dense head (Xu et al. place Dropout(0.5) after each SELU dense).
DEFAULT_HEAD_DROPOUT = 0.5


def _same_pad_1d(kernel: int) -> int:
    """Return the symmetric ``padding`` that keeps a 1-D conv's length unchanged (odd kernel)."""
    return (kernel - 1) // 2


class MCLDNNNet(nn.Module):
    """The MCLDNN spatiotemporal multi-channel network (Xu et al. 2020).

    Three parallel convolutional views of one IQ window -- combined I/Q (2-D conv over the
    ``(2, L)`` map), separate I and separate Q (1-D convs over each ``(1, L)`` row) -- are
    fused, refined by a wider (100-filter) 2-D fusion conv, reshaped into a length-``L`` sequence
    of feature vectors, and passed through a two-layer LSTM. The LSTM's final hidden state feeds
    Xu et al.'s dense head -- two SELU ``Dense(128)`` layers each followed by ``Dropout(0.5)`` (the
    second dropout's output is the penultimate embedding) -- then a classifier to ``num_classes``.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, embed_dim)`` penultimate representation the probing regimes fit a head on.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        conv_filters: int = DEFAULT_CONV_FILTERS,
        fuse_filters: int = DEFAULT_FUSE_FILTERS,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
    ) -> None:
        """Build the three-branch conv stack, the fusion conv, the LSTM and the dense head."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.num_classes = num_classes
        self.window = window
        self.conv_filters = conv_filters
        self.fuse_filters = fuse_filters
        self.lstm_hidden = lstm_hidden
        self.head_dropout = head_dropout

        # padding="same" (stride 1) keeps the TIME axis length identical across all branches even
        # for the EVEN kernel width 8 -- a symmetric integer pad shrinks an even-kernel conv by 1,
        # which desynchronised the branches (feat_iq 127 vs feat_iq_sep 126). Only conv_fuse reduces
        # the 2-row axis to 1 (kernel height 2, no height padding).
        # Branch 1: combined I/Q -- a 2-D conv seeing both channels as one (1, 2, L) image.
        self.conv_iq = nn.Sequential(
            nn.Conv2d(1, conv_filters, kernel_size=(2, 8), padding="same"),
            nn.ReLU(inplace=True),
        )
        # Branch 2/3: separate I and separate Q -- 1-D convs over each (1, L) row.
        self.conv_i = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=8, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.conv_q = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=8, padding="same"),
            nn.ReLU(inplace=True),
        )
        # Fuse the separate-I/Q maps back into a 2-channel image, conv it, then combine with
        # the combined-I/Q branch and refine with a second 2-D conv (Xu et al.'s fusion path).
        self.conv_iq2 = nn.Sequential(
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 8), padding="same"),
            nn.ReLU(inplace=True),
        )
        # Xu et al.'s fusion conv widens to fuse_filters (100 = 2x the per-branch width); its
        # 2-row kernel (no height padding) collapses the channel axis to a single row.
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(conv_filters, fuse_filters, kernel_size=(2, 5), padding=(0, _same_pad_1d(5))),
            nn.ReLU(inplace=True),
        )

        # The fused (B, fuse_filters, 1, L) map -> a length-L sequence of fuse_filters features.
        self.lstm = nn.LSTM(
            input_size=fuse_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
        )
        # Paper dense head (Xu et al. 2020): Dense(selu)->Dropout(0.5)->Dense(selu)->Dropout(0.5).
        # The final Dropout's output is the penultimate embedding features() / embed() return; the
        # classifier maps it to num_classes logits. Restoring the two dropouts + the second dense
        # matches the published head (see BIBLIOGRAPHY.md B.1).
        self.fc_embed = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.SELU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.SELU(inplace=True),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def _fused_sequence(self, x: Tensor) -> Tensor:
        """Run the three conv branches + fusion, returning a ``(B, L, fuse_filters)`` sequence.

        ``x`` is a ``(B, 2, L)`` IQ batch (row 0 = I, row 1 = Q). The combined branch sees it as
        a ``(B, 1, 2, L)`` image; the separate branches see each row as a ``(B, 1, L)`` signal.
        The three branches carry ``conv_filters`` (``F``) channels; the fusion conv widens to
        ``fuse_filters`` (``G``). The fused ``(B, G, 1, L)`` map is squeezed and transposed into
        the ``(B, L, G)`` layout ``nn.LSTM(batch_first=True)`` expects.
        """
        # Combined I/Q branch -> (B, F, 2, L): padding="same" keeps both the 2-row and time axes.
        iq_img = x.unsqueeze(1)  # (B, 1, 2, L)
        feat_iq = self.conv_iq(iq_img)  # (B, F, 2, L)

        # Separate I and Q 1-D branches -> (B, F, L) each; stack into a (B, F, 2, L) image.
        i_sig = x[:, 0:1, :]  # (B, 1, L)
        q_sig = x[:, 1:2, :]  # (B, 1, L)
        feat_i = self.conv_i(i_sig)  # (B, F, L)
        feat_q = self.conv_q(q_sig)  # (B, F, L)
        feat_iq_sep = torch.stack((feat_i, feat_q), dim=2)  # (B, F, 2, L)
        feat_iq_sep = self.conv_iq2(feat_iq_sep)  # (B, F, 2, L) (1-row kernel keeps both rows)

        # Combine the two aligned (B, F, 2, L) feature maps and refine with the fusion conv, whose
        # 2-row kernel (no height padding) collapses the channel axis to a single row and widens
        # the feature axis F -> G (fuse_filters).
        combined = feat_iq_sep + feat_iq  # (B, F, 2, L)
        fused = self.conv_fuse(combined)  # (B, G, 1, L)

        seq = fused.squeeze(2)  # (B, G, L)
        return seq.transpose(1, 2)  # (B, L, G)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, lstm_hidden)`` penultimate embedding for a ``(B, 2, L)`` batch."""
        seq = self._fused_sequence(x)  # (B, L, F)
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
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, window) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != 2:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor


@register_model("mcldnn")
class MCLDNN(Model):
    """The MCLDNN AMC baseline as a :class:`~rfbench.core.model.Model` (registered ``"mcldnn"``).

    Wraps :class:`MCLDNNNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, window)`` IQ
      windows from :class:`~rfbench.tasks.amc.dataset.AmcDataset`) to ``(B, num_classes)``
      logits -- iterated per-sample by the AMC metrics.
    * :meth:`embed` returns the ``(B, lstm_hidden)`` penultimate feature vector for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("mcldnn")()`` on the registry path. Eval runs
    in :meth:`eval` mode with gradients disabled; a from-scratch training loop (M3) loads weights
    into :attr:`net` before evaluation.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "mcldnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("MCLDNN needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = MCLDNNNet(num_classes, window=window).to(self.device)

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
    "MCLDNN",
    "MCLDNNNet",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_CONV_FILTERS",
    "DEFAULT_FUSE_FILTERS",
    "DEFAULT_LSTM_HIDDEN",
    "DEFAULT_HEAD_DROPOUT",
]
