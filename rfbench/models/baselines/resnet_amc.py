"""ResNet AMC baseline (WP-30) -- the O'Shea et al. 2018 residual-stack modulation classifier.

The residual-stack architecture from O'Shea, Roy & Clancy ("Over-the-Air Deep Learning Based
Radio Signal Classification", IEEE J. Sel. Topics Signal Process. 2018) is the canonical deep
AMC baseline on the RadioML corpora. It processes one IQ window through a stack of identical
**residual stacks**, each of which is a 1x1 channel-mixing conv, two **residual units** (two
convolutions + an identity skip connection each), and a max-pool that halves the time axis.
Repeating the stack ``num_stacks`` times downsamples the ``(2, 128)`` window to a compact feature
map, which is flattened and pushed through two fully-connected layers (the penultimate one is the
embedding) to the 11 RML2016.10a modulation classes. The residual skips let the network go deep
while staying a from-scratch baseline (~a low-million params), so it seeds the AMC board next to
MCLDNN rather than acting as a heavy backbone.

Contract bridge (read ``rfbench/core/model.py``). ``forward`` / ``embed`` receive the COLLATED
batch dict that :func:`rfbench.core.evaluate.evaluate` builds -- ``x["iq"]`` is a *list* of
per-sample IQ payloads, one per sample. :class:`~rfbench.tasks.amc.dataset.AmcDataset` yields
RML2016.10a windows of shape ``(2, 128)`` (I on row 0, Q on row 1; see
``rfbench/data/prepare/amc.py`` ``[N, 2, 128]``), so the collated ``x["iq"]`` is
``list[ (2, 128) ]`` and :func:`_iq_to_tensor` stacks it into a ``(B, 2, 128)`` float tensor.
``forward`` returns ``(B, 11)`` class logits; iterating that tensor yields one per-class score
vector per sample, exactly what the AMC metrics' ``argmax`` decoder consumes. ``embed`` returns
the penultimate feature vector for the ``linear_probe`` / ``few_shot`` regimes.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free. This module is
a torch baseline and is therefore NOT imported by ``rfbench`` or by
``rfbench.models.baselines.__init__``; ``torch`` is imported at THIS module's top. The
``@register_model("resnet_amc")`` entry in :data:`rfbench.core.registry.MODELS` is created only on
an explicit ``import rfbench.models.baselines.resnet_amc``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch

#: RadioML 2016.10a modulation classes (the 11-way closed set the ResNet classifies).
DEFAULT_NUM_CLASSES = 11
#: Canonical RML2016.10a IQ window length (samples per channel).
DEFAULT_WINDOW = 128
#: Convolution feature width shared by every residual stack (O'Shea et al. use 32 filters).
DEFAULT_CONV_FILTERS = 32
#: Number of stacked residual stacks; each halves the time axis via its trailing max-pool.
DEFAULT_NUM_STACKS = 4
#: Width of the two fully-connected head layers; the first is the penultimate embedding.
DEFAULT_FC_DIM = 128


def _same_pad_1d(kernel: int) -> int:
    """Return the symmetric ``padding`` that keeps a 1-D conv's length unchanged (odd kernel)."""
    return (kernel - 1) // 2


class ResidualUnit(nn.Module):
    """One residual unit: two same-length 1-D convs with a ReLU'd identity skip (O'Shea et al.).

    Both convolutions preserve the ``channels`` width and the time length (same-padding), so the
    input can be added straight back onto the second conv's output. The sum is passed through a
    final ReLU, matching the residual-unit block of the 2018 residual-stack classifier.
    """

    def __init__(self, channels: int, *, kernel: int = 3) -> None:
        """Build the two same-width convolutions and their activations."""
        super().__init__()
        pad = _same_pad_1d(kernel)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``ReLU(x + conv2(ReLU(conv1(x))))`` -- shape-preserving over ``(B, C, L)``."""
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return cast("Tensor", self.act(out + residual))


class ResidualStack(nn.Module):
    """One residual stack: a 1x1 channel-mixing conv, two residual units, then a x2 max-pool.

    The leading kernel-1 convolution mixes the incoming channels up to ``out_channels`` (a plain
    projection on the first stack, where the input has only the 2 IQ channels); the two
    :class:`ResidualUnit` blocks refine the feature map at constant width; the trailing max-pool
    halves the time axis. Stacking several of these is the downsampling backbone of the classifier.
    """

    def __init__(self, in_channels: int, out_channels: int, *, kernel: int = 3) -> None:
        """Build the channel-mixing conv, the two residual units and the halving max-pool."""
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.unit1 = ResidualUnit(out_channels, kernel=kernel)
        self.unit2 = ResidualUnit(out_channels, kernel=kernel)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in_channels, L)`` to ``(B, out_channels, L // 2)``."""
        out = self.conv_in(x)
        out = self.unit1(out)
        out = self.unit2(out)
        return cast("Tensor", self.pool(out))


class ResNetAMCNet(nn.Module):
    """The O'Shea et al. 2018 residual-stack network for AMC.

    A ``(B, 2, L)`` IQ window flows through ``num_stacks`` :class:`ResidualStack` blocks (the
    first widens the 2 IQ channels to ``conv_filters``; each block halves the time axis), is
    flattened, and pushed through two dense layers: the first (the penultimate embedding) and the
    classifier to ``num_classes`` logits.

    :meth:`forward` returns ``(B, num_classes)`` logits; :meth:`features` returns the
    ``(B, fc_dim)`` penultimate representation the probing regimes fit a head on.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        window: int = DEFAULT_WINDOW,
        conv_filters: int = DEFAULT_CONV_FILTERS,
        num_stacks: int = DEFAULT_NUM_STACKS,
        fc_dim: int = DEFAULT_FC_DIM,
    ) -> None:
        """Build the residual-stack backbone, the flatten and the two-layer dense head."""
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if num_stacks < 1:
            raise ValueError(f"num_stacks must be >= 1, got {num_stacks}")
        if window >> num_stacks < 1:
            raise ValueError(
                f"window {window} too small for {num_stacks} halving stacks "
                f"(would collapse the time axis below length 1)"
            )
        self.num_classes = num_classes
        self.window = window
        self.conv_filters = conv_filters
        self.num_stacks = num_stacks
        self.fc_dim = fc_dim

        stacks: list[nn.Module] = []
        in_channels = 2  # the two IQ rows
        for _ in range(num_stacks):
            stacks.append(ResidualStack(in_channels, conv_filters))
            in_channels = conv_filters
        self.stacks = nn.Sequential(*stacks)

        # Each stack halves the time axis (floor division via the x2 max-pool).
        pooled_len = window
        for _ in range(num_stacks):
            pooled_len //= 2
        self.flat_dim = conv_filters * pooled_len

        # Penultimate dense layer (the embedding) + the classifier head.
        self.fc_embed = nn.Sequential(
            nn.Linear(self.flat_dim, fc_dim),
            nn.SELU(inplace=True),
        )
        self.classifier = nn.Linear(fc_dim, num_classes)

    def features(self, x: Tensor) -> Tensor:
        """Return the ``(B, fc_dim)`` penultimate embedding for a ``(B, 2, L)`` batch."""
        feat = self.stacks(x)  # (B, conv_filters, L // 2**num_stacks)
        flat = feat.flatten(start_dim=1)  # (B, flat_dim)
        embedded = self.fc_embed(flat)  # (B, fc_dim)
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


@register_model("resnet_amc")
class ResNetAMC(Model):
    """The ResNet AMC baseline as a :class:`~rfbench.core.model.Model` (registered ``resnet_amc``).

    Wraps :class:`ResNetAMCNet` to satisfy the frozen ``Model`` contract exactly:

    * :meth:`forward` maps the COLLATED batch dict (``x["iq"]`` a list of ``(2, window)`` IQ
      windows from :class:`~rfbench.tasks.amc.dataset.AmcDataset`) to ``(B, num_classes)`` logits
      -- iterated per-sample by the AMC metrics.
    * :meth:`embed` returns the ``(B, fc_dim)`` penultimate feature vector for the
      ``linear_probe`` / ``few_shot`` regimes.
    * :attr:`n_params` reports the trainable parameter count; :attr:`family` is ``"baseline"``.

    Instantiated with no arguments by ``MODELS.get("resnet_amc")()`` on the registry path. Eval
    runs in :meth:`eval` mode with gradients disabled; a from-scratch training loop (M3) loads
    weights into :attr:`net` before evaluation.
    """

    family: Literal["baseline"] = "baseline"

    def __init__(
        self,
        *,
        name: str = "resnet_amc",
        num_classes: int = DEFAULT_NUM_CLASSES,
        window: int = DEFAULT_WINDOW,
        device: str | None = None,
    ) -> None:
        """Build the network and move it to ``device`` (auto: CUDA when available, else CPU)."""
        if not name:
            raise ValueError("ResNetAMC needs a non-empty name")
        self.name = name
        self.window = window
        if device is not None:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)
        self.net = ResNetAMCNet(num_classes, window=window).to(self.device)

    def forward(self, x: Batch) -> Tensor:
        """Return ``(B, num_classes)`` class logits for the collated AMC batch ``x``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(iq)

    def embed(self, x: Batch) -> Tensor:
        """Return the ``(B, fc_dim)`` embedding for ``linear_probe`` / ``few_shot``."""
        iq = _iq_to_tensor(x["iq"], self.device, self.window)
        self.net.eval()
        with torch.no_grad():
            return self.net.features(iq)

    @property
    def n_params(self) -> int:
        """Total trainable parameter count (written to ``result.json.model.n_params``)."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


__all__ = [
    "ResNetAMC",
    "ResNetAMCNet",
    "ResidualStack",
    "ResidualUnit",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_WINDOW",
    "DEFAULT_CONV_FILTERS",
    "DEFAULT_NUM_STACKS",
    "DEFAULT_FC_DIM",
]
