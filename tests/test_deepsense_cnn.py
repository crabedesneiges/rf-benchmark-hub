"""Acceptance tests for the DeepSense CNN spectrum-sensing baseline (`deepsense_cnn`, INFOCOM 2021).

Torch-gated. Exercises the 4-conv + 1-FC multi-label net on synthetic ``(B, 2, 32)`` channel-first
IQ, asserting: the ``(B, 16)`` per-subband head, that the ``Model`` wrapper's ``forward`` returns
per-subband PROBABILITIES in ``[0, 1]`` (sigmoid) while the backing ``nn.Module`` returns raw
LOGITS (for ``BCEWithLogitsLoss``), the ``(B, 64)`` embedding, and the two /2 time pools.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.deepsense_cnn import (  # noqa: E402
    DEFAULT_NUM_SUBBANDS,
    DEFAULT_WINDOW,
    DeepSenseCNN,
    DeepSenseNet,
)

_BATCH = 4


def _iq(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    """A collated batch of channel-first ``(2, window)`` IQ windows (the DeepSense X layout)."""
    gen = torch.Generator().manual_seed(7)
    return {"iq": [w.tolist() for w in torch.randn(batch, 2, window, generator=gen)]}


def test_registered_and_contract() -> None:
    """Registered under 'deepsense_cnn'; a baseline-family Model with a 16-subband head."""
    assert MODELS.get("deepsense_cnn") is DeepSenseCNN
    model = DeepSenseCNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "deepsense_cnn"
    assert model.family == "baseline"
    assert model.forward(_iq()).shape == (_BATCH, DEFAULT_NUM_SUBBANDS)


def test_forward_returns_probabilities() -> None:
    """The Model wrapper applies the per-subband sigmoid -> outputs live in [0, 1]."""
    out = DeepSenseCNN(device="cpu").forward(_iq())
    assert bool((out >= 0.0).all()) and bool((out <= 1.0).all())


def test_net_returns_raw_logits() -> None:
    """The backing nn.Module returns LOGITS (not squashed) so the trainer can use BCEWithLogits."""
    net = DeepSenseNet(DEFAULT_NUM_SUBBANDS, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    logits = net.forward(x)
    assert logits.shape == (_BATCH, DEFAULT_NUM_SUBBANDS)
    # Raw logits routinely fall outside [0, 1]; a random-init net almost surely produces one.
    assert bool((logits < 0.0).any()) or bool((logits > 1.0).any())


def test_embed_and_custom_subbands() -> None:
    """embed -> (B, 64); a custom sub-band count flows to the sigmoid head."""
    assert DeepSenseCNN(device="cpu").embed(_iq()).shape == (_BATCH, 64)
    assert DeepSenseCNN(device="cpu", num_subbands=8).forward(_iq()).shape == (_BATCH, 8)


def test_conv_stack_pools_time_twice() -> None:
    """After both conv stages the 32-sample time axis is pooled /2 twice -> 8, with 32 filters."""
    net = DeepSenseNet(DEFAULT_NUM_SUBBANDS, window=DEFAULT_WINDOW)
    net.eval()
    feats = net._conv_features(torch.randn(_BATCH, 2, DEFAULT_WINDOW))  # flattened (B, 32 * 8)
    assert feats.shape == (_BATCH, 32 * (DEFAULT_WINDOW // 4))


def test_unbatched_and_wrong_channel() -> None:
    """Single-sample promotion + loud failure on a non-2 IQ channel axis."""
    model = DeepSenseCNN(device="cpu")
    assert model.forward({"iq": torch.randn(2, DEFAULT_WINDOW).tolist()}).shape == (
        1,
        DEFAULT_NUM_SUBBANDS,
    )
    with pytest.raises(ValueError, match=r"shape \(B, 2, 32\)"):
        model.forward({"iq": torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()})
