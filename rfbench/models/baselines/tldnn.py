"""TLDNN AMC baseline -- the transformer+LSTM global-feature AMC model (Qu et al. 2024).

TLDNN ("Enhancing Automatic Modulation Recognition through Robust Global Feature Extraction",
Qu, Lu, Zeng, Wang, Wang, IEEE TVT 2024, arXiv:2401.01056) is a hybrid deep framework that
pairs a convolutional feature-embedding stem with a transformer encoder (self-attention for
global correlations) and an LSTM (temporal correlations), then a small dense classifier. On
RadioML 2016.10a it reports **62.83%** full-SNR-range overall accuracy (Table II) -- above our
MCLDNN (61.71%) -- and **63.32%** on 2018.01a, which is why it is reimplementation target #1 in
docs/BIBLIOGRAPHY.md. The `+SS` variant (63.35% on 2016.10a) adds a *segment-substitution* data
augmentation applied at TRAINING time; it is a distinct configuration and NOT reproduced here --
this module is the plain-TLDNN backbone, the 62.83% row.

Architecture (extracted verbatim from arXiv:2401.01056v1, Sec. III + Table I; line refs are into
the PDF's pdftotext dump -- see the PR description for the full quoted passages):

* **Input = normalized A/P** (Sec. III, eqs. 3-6): the model does NOT consume raw I/Q. Each
  complex sample ``r[n] = I[n] + jQ[n]`` is mapped to amplitude ``A[n] = sqrt(I^2 + Q^2)`` and
  phase ``P[n] = atan2(Q, I)``. Amplitude is min-max normalized to ``[0, 1]`` (eq. 5); phase is
  normalized to ``[-1, 1]`` radians/pi. The ``(2, N)`` A/P matrix (row 0 = A, row 1 = P) is the
  network input. This normalization is also what conditions the input scale -- there is no raw
  ~1e-2-RMS-IQ chance-collapse fragility here (contrast MCLDNN/CLDNN/ResNet), because A in [0,1]
  and P in [-1,1] are already healthily scaled.
* **Feature Embedding Layer** (Sec. III-A): ``K`` analogous 1-D conv layers, kernel
  ``Ks = Ls/2 = 4`` (symbol sample size ``Ls = 8``), **stride 2** (each layer halves the length),
  ``d = 64`` output channels. First conv maps 2->d, the rest d->d. ``K = 2`` for 2016.10a
  (N=128 -> L=32) and ``K = 4`` for 2018.01a (N=1024 -> L=64); ``L = floor(N / 2^K)``. A
  **Squeeze-and-Excitation** block (reduction ``r = 4``, eqs. 7-9) recalibrates the d channels,
  followed by dropout. Output embedding tokens ``Xe in R^{L x d}``.
* **Transformer Encoder** (Sec. III-B): ``Mt = 2`` identical layers. Learnable positional
  encodings are ADDED to the token sequence. Each layer = a **talking-heads** multi-head
  attention block (``h = 8`` heads, per-head dim ``dt = d/h = 8``; talking-heads mixes the h
  heads with learned ``Pl, Pw in R^{h x h}`` immediately before and after the softmax, eqs.
  10-12) + a position-wise feed-forward network. The FFN is a **ReGLU**
  ``(relu(X W1) (x) (X W2)) W3`` (eq. 13) of hidden width ``dffn = 2d``. Residual + LayerNorm
  wrap both sub-blocks (standard transformer encoder).
* **LSTM Layer + Classifier** (Sec. III-C): the transformer output sequence is processed by
  ``Ml = 4`` stacked LSTM layers with hidden size ``dl = d = 64``. The **last time step** of the
  LSTM output sequence feeds a classifier of **three fully-connected layers with ReLU** to
  ``num_classes`` logits. (The two classifier hidden widths are unspecified by the paper; we keep
  them at ``dl`` -- a faithful, minimal choice -- and expose the penultimate ``dl``-d feature as
  the ``embed`` representation.)

Training recipe (paper Table I): AdamW, lr 1e-3, ReduceLROnPlateau (factor 0.1, patience 10),
CrossEntropy, batch 128 (2016.10a) / 512 (2018.01a), 150 epochs, 6:2:2 split. **lr caveat**: the
rfbench from-scratch loop uses plain ``Adam`` with **no warmup**, and at ``lr=1e-3`` a
transformer-bearing model collapses under that recipe (documented lesson: tprime went
0.259->0.995 val, i.e. chance/degenerate). Train TLDNN with **``--lr 2e-4``** (see
``slurm/train_tldnn_arm.sh``), not the 1e-3 that works for the pure-CNN/CNN-LSTM baselines.

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the COLLATED
batch dict :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` a *list* of per-sample
``(2, N)`` I/Q windows from :class:`~rfbench.tasks.amc.dataset.AmcDataset`. :func:`_iq_to_tensor`
stacks it into ``(B, 2, N)``; :func:`_iq_to_ap` then derives the ``(B, 2, N)`` A/P input.
``forward`` returns ``(B, num_classes)`` raw logits (rfbench trains with CrossEntropyLoss, no
LogSoftmax); ``embed`` returns the ``(B, dl)`` penultimate feature for ``linear_probe`` /
``few_shot``.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module is
a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("tldnn")`` entry in :data:`rfbench.core.registry.MODELS` is created only on an
explicit ``import rfbench.models.baselines.tldnn``.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: RadioML 2016.10a modulation classes (the 11-way closed set); 24 for 2018.01a.
DEFAULT_NUM_CLASSES = 11
#: Canonical RML2016.10a IQ window length (samples per channel).
DEFAULT_WINDOW = 128
#: Conv output channels / token embedding width ``d`` (Qu et al. set d = 64).
DEFAULT_EMBED_DIM = 64
#: Conv kernel ``Ks = Ls/2`` with symbol sample size ``Ls = 8`` -> 4 (Sec. III-A).
DEFAULT_CONV_KERNEL = 4
#: Squeeze-and-Excitation reduction ratio ``r`` (Sec. III-A, eq. 7).
DEFAULT_SE_REDUCTION = 4
#: Number of transformer encoder layers ``Mt`` (Sec. III-B / IV; depth-2 chosen in the paper).
DEFAULT_TRANSFORMER_LAYERS = 2
#: Attention heads ``h`` (Sec. III-B). Per-head dim is ``d/h`` (= 8 at d=64).
DEFAULT_NUM_HEADS = 8
#: Number of stacked LSTM layers ``Ml`` and hidden size ``dl = d`` (Sec. III-C).
DEFAULT_LSTM_LAYERS = 4
#: Dropout after the SE block (paper mentions a dropout layer but not its rate; modest default).
DEFAULT_DROPOUT = 0.1


def _conv_layers_for_window(window: int) -> int:
    """Number of stride-2 conv layers ``K`` for a signal of length ``window``.

    Qu et al. pick ``K`` per dataset to keep the token sequence length ``L = floor(N / 2^K)``
    small: ``K = 2`` for RadioML 2016.10a (N=128 -> L=32) and ``K = 4`` for 2018.01a
    (N=1024 -> L=64). We reproduce exactly that rule (2 for the short 128-sample windows, 4 for
    the long 1024-sample ones); an explicit ``conv_layers=`` overrides it.
    """
    return 2 if window <= 128 else 4


def _iq_to_ap(iq: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Map a ``(B, 2, N)`` I/Q batch to TLDNN's normalized amplitude/phase input (Sec. III).

    Row 0 of ``iq`` is I, row 1 is Q. The complex sample ``r = I + jQ`` is written in polar form
    (eq. 4); we return the ``(B, 2, N)`` matrix whose row 0 is the min-max-normalized amplitude
    ``A = sqrt(I^2 + Q^2)`` scaled to ``[0, 1]`` per sample (eq. 5) and whose row 1 is the phase
    ``atan2(Q, I)`` normalized to ``[-1, 1]`` (radians / pi, Sec. III). The min-max is taken over
    the whole time axis of each sample independently, so the absolute capture scale (no modulation
    information) is removed while the amplitude *envelope shape* is kept.
    """
    i = iq[:, 0, :]
    q = iq[:, 1, :]
    amplitude = torch.sqrt(i * i + q * q)  # (B, N)
    phase = torch.atan2(q, i) / math.pi  # (B, N) in [-1, 1]
    a_min = amplitude.amin(dim=1, keepdim=True)
    a_max = amplitude.amax(dim=1, keepdim=True)
    amplitude = (amplitude - a_min) / (a_max - a_min + eps)  # (B, N) in [0, 1]
    return torch.stack((amplitude, phase), dim=1)  # (B, 2, N)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration over a ``(B, d, L)`` token map (Sec. III-A).

    Squeezes the sequence axis to a per-channel descriptor ``z in R^d`` (global average over the
    L tokens, eq. 8), passes it through a bottleneck ``sigma(W2 relu(W1 z))`` with reduction
    ratio ``r`` (eq. 7), and rescales each channel of the input by the resulting weights
    ``s in R^d`` (eq. 9).
    """

    def __init__(self, channels: int, reduction: int = DEFAULT_SE_REDUCTION) -> None:
        """Build the squeeze-excite bottleneck for ``channels`` feature channels."""
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Recalibrate ``x`` ``(B, d, L)`` -> ``(B, d, L)`` by learned per-channel weights."""
        z = x.mean(dim=2)  # (B, d) -- squeeze the L tokens
        s = self.fc(z)  # (B, d)
        return cast("Tensor", x * s.unsqueeze(-1))


class FeatureEmbedding(nn.Module):
    """The convolutional embedding stem + SE block (Sec. III-A).

    ``K`` stride-2 conv layers (kernel ``Ks``, padding 1 so each layer halves the length) map the
    ``(B, 2, N)`` A/P input to a ``(B, d, L)`` token map with ``L = N / 2^K``; the first conv is
    2->d, the rest d->d, each followed by ReLU. An SE block recalibrates the d channels, then a
    dropout. Returns the ``(B, L, d)`` token sequence the transformer consumes.
    """

    def __init__(
        self,
        *,
        conv_layers: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        kernel: int = DEFAULT_CONV_KERNEL,
        se_reduction: int = DEFAULT_SE_REDUCTION,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Build ``conv_layers`` stride-2 convs (2->d then d->d), the SE block and the dropout."""
        super().__init__()
        # padding = kernel//2 - 1 gives the exact length-halving L -> L/2 for kernel 4 / stride 2
        # (out = floor((L + 2p - k)/s) + 1 = L/2 when p=1, k=4, s=2), reproducing the paper's
        # L = floor(N / 2^K) token count (128 -> 32 at K=2, 1024 -> 64 at K=4).
        padding = kernel // 2 - 1
        convs: list[nn.Module] = []
        in_ch = 2
        for _ in range(conv_layers):
            convs.append(nn.Conv1d(in_ch, embed_dim, kernel_size=kernel, stride=2, padding=padding))
            convs.append(nn.ReLU(inplace=True))
            in_ch = embed_dim
        self.convs = nn.Sequential(*convs)
        self.se = SEBlock(embed_dim, se_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, 2, N)`` A/P input -> ``(B, L, d)`` token sequence."""
        feats = self.convs(x)  # (B, d, L)
        feats = self.se(feats)  # (B, d, L)
        feats = self.dropout(feats)
        return cast("Tensor", feats.transpose(1, 2))  # (B, L, d)


class TalkingHeadsAttention(nn.Module):
    """Multi-head self-attention with talking-heads projections (Sec. III-B, eqs. 10-12).

    Standard scaled dot-product multi-head attention, augmented with the *talking-heads* variant
    (Shazeer et al.): two learned ``h x h`` linear maps ``Pl`` (applied to the attention logits,
    before softmax) and ``Pw`` (applied to the attention weights, after softmax) mix information
    across the h heads. ``Pl`` and ``Pw`` are initialized to the identity, so at init this is
    exactly ordinary multi-head attention and the talking-heads mixing is *learned* on top.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Build Q/K/V/output projections and the two learnable talking-heads matrices."""
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Talking-heads mixing across the head axis, identity-initialized (== plain MHA at init).
        self.pl = nn.Parameter(torch.eye(num_heads))
        self.pw = nn.Parameter(torch.eye(num_heads))

    def forward(self, x: Tensor) -> Tensor:
        """Self-attention over ``x`` ``(B, L, d)`` -> ``(B, L, d)``."""
        b, length, _ = x.shape
        # Project and split into heads -> (B, h, L, head_dim).
        q = self.q_proj(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, L, L)
        # Talking-heads: mix logits across heads (Pl) before softmax, weights (Pw) after.
        logits = torch.einsum("bhij,hg->bgij", logits, self.pl)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.einsum("bhij,hg->bgij", weights, self.pw)
        context = torch.matmul(weights, v)  # (B, h, L, head_dim)
        context = context.transpose(1, 2).reshape(b, length, -1)  # (B, L, d)
        return cast("Tensor", self.out_proj(context))


class ReGLU(nn.Module):
    """Rectified-GLU position-wise feed-forward network (Sec. III-B, eq. 13).

    ``ReGLU(X) = (relu(X W1) (x) (X W2)) W3`` with element-wise product ``(x)``: two parallel
    gated branches of hidden width ``dffn`` fused before the output projection back to ``d``.
    Substitutes the transformer's usual two-linear+ReLU FFN (following Shazeer's GLU-variants).
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        """Build the two gate branches (d->dffn) and the output projection (dffn->d)."""
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(embed_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, d)`` -> ``(B, L, d)`` through the gated FFN."""
        return cast("Tensor", self.w3(torch.relu(self.w1(x)) * self.w2(x)))


class TransformerEncoderLayer(nn.Module):
    """One transformer encoder layer: talking-heads MHA + ReGLU FFN, each residual + LayerNorm.

    Post-norm residual wrapping (the original transformer's arrangement): the attention output is
    added to its input and layer-normalized, then the ReGLU FFN output is added and normalized.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        """Build the attention sub-block, the ReGLU FFN sub-block and their norms."""
        super().__init__()
        self.attn = TalkingHeadsAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = ReGLU(embed_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, d)`` -> ``(B, L, d)`` through attention then FFN, both residual + normed."""
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TLDNNNet(nn.Module):
    """The TLDNN hybrid transformer+LSTM network (Qu et al. 2024).

    Pipeline: A/P feature embedding (conv stem + SE) -> learnable positional encoding -> ``Mt``
    talking-heads transformer encoder layers -> ``Ml``-layer LSTM -> 3-FC ReLU classifier. See the
    module docstring for the per-stage paper references.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the ``(B, dl)``
    penultimate embedding the probing regimes fit a head on.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        conv_layers: int | None = None,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS,
        lstm_layers: int = DEFAULT_LSTM_LAYERS,
        se_reduction: int = DEFAULT_SE_REDUCTION,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Build the embedding stem, positional encoding, transformer, LSTM and classifier.

        ``conv_layers`` (``K``) defaults to :func:`_conv_layers_for_window` (2 for 128-sample
        windows, 4 for 1024-sample), reproducing the paper's per-dataset choice; pass it
        explicitly to override. The LSTM hidden size ``dl`` equals ``embed_dim`` (``d``), and the
        feed-forward width ``dffn`` equals ``2 * embed_dim`` (both per Sec. III).
        """
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if conv_layers is None:
            conv_layers = _conv_layers_for_window(window)
        if conv_layers < 1:
            raise ValueError(f"conv_layers must be >= 1, got {conv_layers}")
        self.num_classes = num_classes
        self.window = window
        self.conv_layers = conv_layers
        self.embed_dim = embed_dim
        self.seq_len = window // (2**conv_layers)  # L = floor(N / 2^K)

        self.embedding = FeatureEmbedding(
            conv_layers=conv_layers,
            embed_dim=embed_dim,
            se_reduction=se_reduction,
            dropout=dropout,
        )
        # Learnable positional encodings, added to the token sequence (Sec. III-B, "learnable
        # variables with the same dimensions as the input sequence").
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self.encoder = nn.ModuleList(
            TransformerEncoderLayer(embed_dim, num_heads, 2 * embed_dim, dropout)
            for _ in range(transformer_layers)
        )
        # dl = d; the last LSTM time step feeds the classifier.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Classifier: three FC layers with ReLU (Sec. III-C). The penultimate feature (after the
        # second ReLU) is what features() / embed() return; the final Linear maps it to logits.
        self.fc_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, embed_dim)`` penultimate embedding for a ``(B, 2, N)`` raw-IQ batch.

        The A/P transform (:func:`_iq_to_ap`) is applied HERE, inside the net, not in the wrapper:
        the rfbench from-scratch loop optimises ``model.net`` directly (calls ``net(raw_iq)``),
        so the amplitude/phase conditioning must be part of the module's own ``forward`` to be
        seen at TRAINING time -- exactly as MCLDNN/CLDNN/ResNet fold their input normalization
        into the net. ``x`` is therefore raw I/Q ``(B, 2, N)`` (row 0 = I, row 1 = Q).
        """
        x = _iq_to_ap(x)  # (B, 2, N) raw I/Q -> normalized amplitude/phase
        tokens = self.embedding(x)  # (B, L, d)
        tokens = tokens + self.pos_encoding
        for layer in self.encoder:
            tokens = layer(tokens)  # (B, L, d)
        out, _state = self.lstm(tokens)  # (B, L, dl)
        last = out[:, -1, :]  # (B, dl) -- final time step of the LSTM output sequence
        return cast("Tensor", self.fc_embed(last))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` logits for a raw-IQ ``(B, 2, N)`` batch (A/P inside)."""
        return cast("Tensor", self.classifier(self.features(x)))


def _iq_to_tensor(iq_batch: object, device: torch.device, window: int) -> Tensor:
    """Stack the collated ``x["iq"]`` list into a ``(B, 2, window)`` float tensor on ``device``.

    ``iq_batch`` is the per-sample I/Q list :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.amc.dataset.AmcDataset`: each element is a ``(2, window)`` array-like
    (numpy on the cluster, nested lists in a synthetic fixture). ``torch.as_tensor`` handles both;
    the result is coerced to ``float32`` and validated to ``(B, 2, window)`` so a mis-shaped batch
    fails loudly rather than silently mis-classifying. The A/P transform is applied downstream by
    :func:`_iq_to_ap`, so this returns raw I/Q.
    """
    tensor = torch.as_tensor(iq_batch, dtype=torch.float32, device=device)
    if tensor.ndim == 2:  # a single unbatched (2, window) sample -> add the batch axis
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[1] != 2:
        raise ValueError(f"expected IQ batch of shape (B, 2, {window}); got {tuple(tensor.shape)}")
    return tensor


@register_model("tldnn")
class TLDNN(Model):
    """The TLDNN AMC baseline as a :class:`~rfbench.core.model.Model` (registered ``"tldnn"``).

    Wraps :class:`TLDNNNet` to satisfy the frozen ``Model`` contract:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, window)`` I/Q
      windows) to ``(B, num_classes)`` raw logits -- iterated per-sample by the AMC metrics.
    * :meth:`embed` returns the ``(B, embed_dim)`` penultimate feature for ``linear_probe`` /
      ``few_shot``.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    The AMC ``rfbench train`` path instantiates baselines with no arguments
    (``MODELS.get("tldnn")()``), which builds the 2016.10a (11-class, window 128, K=2) model. For
    2018.01a (24-class, window 1024) pass ``num_classes=24, window=1024`` explicitly -- the
    per-dataset conv depth (K=4) is then derived from ``window`` (see the SLURM driver).

    **Training note**: this model contains attention; train it with ``--lr 2e-4`` under the
    rfbench from-scratch loop (no warmup), NOT the default ``lr=1e-3`` that collapses transformers
    (documented lesson). See the module docstring and ``slurm/train_tldnn_arm.sh``.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "tldnn",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        conv_layers: int | None = None,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("TLDNN needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = TLDNNNet(num_classes, window=window, conv_layers=conv_layers).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` class logits for the collated AMC batch ``x``."""
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
    "TLDNN",
    "TLDNNNet",
    "FeatureEmbedding",
    "SEBlock",
    "TalkingHeadsAttention",
    "ReGLU",
    "TransformerEncoderLayer",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_CONV_KERNEL",
    "DEFAULT_SE_REDUCTION",
    "DEFAULT_TRANSFORMER_LAYERS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_LSTM_LAYERS",
    "DEFAULT_DROPOUT",
]
