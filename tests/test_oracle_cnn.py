"""Acceptance tests for the ORACLE CNN SEI baseline (`oracle_cnn`, Sankhe et al. 2019).

Torch-gated. Exercises the 2-conv + 2-FC net on synthetic ``(B, 128, 2)`` IQ, asserting the
paper structure: Conv(1x7) keeps the I/Q height, Conv(2x7) collapses it, the ``(256, 80, N)``
dense head, and L2 exposed over the conv + dense kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.oracle_cnn import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    OracleCNN,
    OracleCNNNet,
)

_BATCH = 4


def _iq(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    gen = torch.Generator().manual_seed(7)
    return {"iq": [w.tolist() for w in torch.randn(batch, window, 2, generator=gen)]}


def test_registered_and_contract() -> None:
    """Registered under 'oracle_cnn'; a baseline-family Model with 16-tx default head."""
    assert MODELS.get("oracle_cnn") is OracleCNN
    model = OracleCNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "oracle_cnn"
    assert model.family == "baseline"
    assert model.forward(_iq()).shape == (_BATCH, DEFAULT_NUM_CLASSES)


def test_embed_and_custom_classes() -> None:
    """embed -> (B, 80); a custom class count flows to the head."""
    assert OracleCNN(device="cpu").embed(_iq()).shape == (_BATCH, 80)
    assert OracleCNN(device="cpu", num_classes=5).forward(_iq()).shape == (_BATCH, 5)


def test_conv_collapses_iq_height() -> None:
    """After the two convs the height-2 (I/Q) axis is collapsed to 1, time preserved."""
    net = OracleCNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, DEFAULT_WINDOW, 2)
    feats = net._conv_features(x)  # flattened (B, 50 * 1 * window)
    assert feats.shape == (_BATCH, 50 * DEFAULT_WINDOW)


def test_l2_penalty_covers_conv_and_dense() -> None:
    """l2_penalty sums the conv + dense kernels (ORACLE regularises broadly)."""
    net = OracleCNNNet(4, window=DEFAULT_WINDOW)
    expected = (
        net.conv1.weight.pow(2).sum()
        + net.conv2.weight.pow(2).sum()
        + net.fc1.weight.pow(2).sum()
        + net.fc2.weight.pow(2).sum()
        + net.classifier.weight.pow(2).sum()
    )
    assert torch.allclose(net.l2_penalty(), expected)


def test_input_norm_flag_scale_invariance() -> None:
    """With input_norm=True logits are scale-invariant; with False they are not."""
    x = torch.randn(_BATCH, DEFAULT_WINDOW, 2, generator=torch.Generator().manual_seed(1))
    b1 = {"iq": [r.tolist() for r in x]}
    b2 = {"iq": [(4.0 * r).tolist() for r in x]}
    normed = OracleCNN(device="cpu", num_classes=6)
    normed.net.eval()
    assert torch.allclose(normed.forward(b1), normed.forward(b2), atol=1e-4)
    raw = OracleCNN(device="cpu", num_classes=6, input_norm=False)
    raw.net.eval()
    assert not torch.allclose(raw.forward(b1), raw.forward(b2), atol=1e-3)


def test_unbatched_and_wrong_channel() -> None:
    """Single-sample promotion + loud failure on a non-2 IQ axis."""
    model = OracleCNN(device="cpu")
    assert model.forward({"iq": torch.randn(DEFAULT_WINDOW, 2).tolist()}).shape == (
        1,
        DEFAULT_NUM_CLASSES,
    )
    with pytest.raises(ValueError, match=r"shape \(B, 128, 2\)"):
        model.forward({"iq": torch.randn(_BATCH, DEFAULT_WINDOW, 3).tolist()})
