"""1-D ShuffleNetV2 backbone for raw-IQ RF signals (reusable FM feature extractor).

This is the shared backbone behind the IQFM wrapper (:mod:`rfbench.models.foundation.iqfm`)
and — by design — the forthcoming WirelessJEPA wrapper: both papers use the **same**
ShuffleNetV2-x0.5 encoder over raw IQ (IQFM: Mashaal & Abou-Zeid, arXiv:2506.06718v2;
WirelessJEPA: arXiv:2601.20190, "ShuffleNetV2-x0.5 matched to IQFM"). Keeping it in one place
means the two FMs share an identical, verified architecture rather than two drifting copies.

It is a faithful 1-D transcription of the ImageNet ShuffleNetV2 (Ma et al., ECCV 2018) — the
same stage layout, inverted-residual unit, and channel-shuffle — with every ``Conv2d``/pool
replaced by its ``Conv1d`` counterpart so it consumes a raw complex window laid out as two real
channels ``(2, L)`` (I and Q). At the ``x0.5`` width the backbone (conv1 → stages → conv5 → the
1024-wide head, mean-pooled over time, **no classifier**) is **335,096 parameters** (measured) —
the small delta from the ~341k IQFM reports is the expected 1-D-vs-2-D transcription difference —
and yields a 1024-D per-sample embedding.

HARD CONSTRAINT: importing this module pulls in **no** third-party dependency. ``torch`` is
imported lazily via :func:`~rfbench.models.foundation.base.require_torch` inside
:func:`build_shufflenet1d`; the ``nn.Module`` subclasses are defined *inside* a build function
that receives the lazily-imported ``torch.nn`` (the ``lwm_spectro`` idiom), so
``import rfbench.models.foundation`` stays torch-free and mypy stays strict without torch present.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, cast

from rfbench.core.types import Tensor
from rfbench.models.foundation.base import require_torch

#: ShuffleNetV2 **x0.5** per-stage output channels ``(conv1, stage2, stage3, stage4, conv5)``.
#: The canonical width from the paper's Table 5; the 1024-wide conv5 head is what makes the
#: mean-pooled embedding 1024-D and the backbone ~341k params (matching IQFM's reported encoder).
STAGES_OUT_CHANNELS_X0_5: tuple[int, int, int, int, int] = (24, 48, 96, 192, 1024)
#: Number of inverted-residual units in stage2 / stage3 / stage4 (shared across all widths).
STAGES_REPEATS: tuple[int, int, int] = (4, 8, 4)
#: Default input channels: a complex IQ window as two real channels (I, Q).
IN_CHANNELS: int = 2
#: Embedding width produced by :func:`build_shufflenet1d` at the ``x0.5`` width (conv5 output).
EMBED_DIM_X0_5: int = STAGES_OUT_CHANNELS_X0_5[-1]


def _channel_shuffle(x: Tensor, groups: int, torch_mod: ModuleType) -> Tensor:
    """Channel-shuffle a ``(B, C, L)`` tensor across ``groups`` (ShuffleNet's mixing op).

    Reshapes channels into ``(groups, C // groups)``, transposes the two, and flattens back —
    the standard permutation that lets information cross the two branches of successive units.
    ``torch_mod`` is unused (kept for signature symmetry); the reshape uses tensor methods only.
    """
    del torch_mod  # tensor methods below need no framework handle
    b, c, length = x.shape
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, length)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, length)


def build_shufflenet1d(
    *,
    in_channels: int = IN_CHANNELS,
    stages_out_channels: Sequence[int] = STAGES_OUT_CHANNELS_X0_5,
    stages_repeats: Sequence[int] = STAGES_REPEATS,
) -> Tensor:
    """Build a 1-D ShuffleNetV2 backbone returning one pooled embedding per sample.

    The returned ``nn.Module`` maps a raw-IQ batch ``(B, in_channels, L)`` to a pooled
    representation ``(B, stages_out_channels[-1])`` (global average pool over time; **no
    classifier head**). Defaults reproduce the ShuffleNetV2-**x0.5** encoder IQFM/WirelessJEPA
    use (~341k params, 1024-D embedding). Torch is imported lazily here, so this is the only
    place the heavy dependency is pulled; the module is typed :data:`Tensor` (``Any``) to keep
    the file torch-free at import.

    Args:
        in_channels: input channels; ``2`` for a complex IQ window laid out as ``(I, Q)``.
        stages_out_channels: the five ``(conv1, stage2, stage3, stage4, conv5)`` widths.
        stages_repeats: number of inverted-residual units per stage ``(stage2, stage3, stage4)``.

    Returns:
        An ``nn.Module`` backbone (``forward(x) -> (B, C)`` pooled embedding).
    """
    if len(stages_out_channels) != 5:
        raise ValueError(f"stages_out_channels must have 5 entries, got {len(stages_out_channels)}")
    if len(stages_repeats) != 3:
        raise ValueError(f"stages_repeats must have 3 entries, got {len(stages_repeats)}")

    torch_mod = require_torch()
    torch = cast("Any", torch_mod)
    nn = cast("Any", torch_mod.nn)

    class _InvertedResidual(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """One ShuffleNetV2 unit (1-D). ``stride==1`` splits+shuffles; ``stride==2`` downsamples.

        For ``stride == 1`` the input is split in half along channels: one half passes through
        unchanged (identity) and the other through a pointwise → depthwise → pointwise branch;
        the two are concatenated then channel-shuffled. For ``stride == 2`` both branches
        downsample (no split) and their concatenation doubles the spatial reduction — the
        standard spatial-downsampling unit at the head of each stage.
        """

        def __init__(self, inp: int, oup: int, stride: int) -> None:
            super().__init__()
            if stride not in (1, 2):
                raise ValueError(f"stride must be 1 or 2, got {stride}")
            self.stride = stride
            branch_features = oup // 2

            if stride == 1:
                self.branch1 = nn.Sequential()
            else:
                self.branch1 = nn.Sequential(
                    self._depthwise(inp, inp, stride),
                    nn.BatchNorm1d(inp),
                    nn.Conv1d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(branch_features),
                    nn.ReLU(inplace=True),
                )

            branch2_inp = inp if stride > 1 else branch_features
            self.branch2 = nn.Sequential(
                nn.Conv1d(branch2_inp, branch_features, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm1d(branch_features),
                nn.ReLU(inplace=True),
                self._depthwise(branch_features, branch_features, stride),
                nn.BatchNorm1d(branch_features),
                nn.Conv1d(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm1d(branch_features),
                nn.ReLU(inplace=True),
            )

        @staticmethod
        def _depthwise(inp: int, oup: int, stride: int) -> Tensor:
            """A depthwise 3-tap 1-D conv (``groups=inp``), the unit's only spatial mixing."""
            return nn.Conv1d(
                inp, oup, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False
            )

        def forward(self, x: Tensor) -> Tensor:
            if self.stride == 1:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)
            else:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            return _channel_shuffle(out, 2, torch)

    class _ShuffleNetV2_1D(nn.Module):  # type: ignore[misc,name-defined]  # nn is Any (lazy torch)
        """1-D ShuffleNetV2 feature extractor: conv1 → maxpool → 3 stages → conv5 → mean-pool.

        ``forward`` returns the ``(B, C)`` global-average-pooled embedding (over time); no
        classifier is attached, so the output is the frozen representation an FM wrapper probes.
        """

        def __init__(self) -> None:
            super().__init__()
            out = list(stages_out_channels)
            input_channels = in_channels

            conv1_out = out[0]
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    input_channels, conv1_out, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm1d(conv1_out),
                nn.ReLU(inplace=True),
            )
            input_channels = conv1_out
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            stage_names = ["stage2", "stage3", "stage4"]
            for name, repeats, output_channels in zip(
                stage_names, stages_repeats, out[1:4], strict=True
            ):
                seq = [_InvertedResidual(input_channels, output_channels, 2)]
                for _ in range(repeats - 1):
                    seq.append(_InvertedResidual(output_channels, output_channels, 1))
                setattr(self, name, nn.Sequential(*seq))
                input_channels = output_channels

            conv5_out = out[4]
            self.conv5 = nn.Sequential(
                nn.Conv1d(
                    input_channels, conv5_out, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm1d(conv5_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            return x.mean(dim=2)  # global average pool over time -> (B, C)

    return _ShuffleNetV2_1D()


def embed_dim(stages_out_channels: Sequence[int] = STAGES_OUT_CHANNELS_X0_5) -> int:
    """Return the embedding width :func:`build_shufflenet1d` produces (the conv5 output width)."""
    return int(stages_out_channels[-1])


__all__ = [
    "build_shufflenet1d",
    "embed_dim",
    "STAGES_OUT_CHANNELS_X0_5",
    "STAGES_REPEATS",
    "IN_CHANNELS",
    "EMBED_DIM_X0_5",
]
