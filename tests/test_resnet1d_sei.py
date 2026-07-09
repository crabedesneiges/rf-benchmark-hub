"""Acceptance tests for the ResNet-1D SEI baseline (`resnet1d_sei`).

Torch-gated. Exercises the ResNet-18-1D over synthetic ``(B, 256, 2)`` IQ: forward/embed shapes,
the 512-wide pooled embedding, window-agnosticism (adaptive pool), and the unit-average-power
scale invariance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.resnet1d_sei import (  # noqa: E402
    DEFAULT_WINDOW,
    ResNet1dNet,
    ResNet1dSEI,
)

_BATCH = 4


def _iq(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    gen = torch.Generator().manual_seed(9)
    return {"iq": [w.tolist() for w in torch.randn(batch, window, 2, generator=gen)]}


def test_registered_and_contract() -> None:
    """Registered under 'resnet1d_sei'; a baseline Model; forward/embed shapes."""
    assert MODELS.get("resnet1d_sei") is ResNet1dSEI
    model = ResNet1dSEI(device="cpu", num_classes=20)
    assert isinstance(model, Model)
    assert model.family == "baseline"
    assert model.forward(_iq()).shape == (_BATCH, 20)
    assert model.embed(_iq()).shape == (_BATCH, 512)  # ResNet-18 final stage width


def test_window_agnostic_embedding() -> None:
    """The adaptive pool yields a fixed 512-wide embedding for a different window length."""
    model = ResNet1dSEI(device="cpu", window=128)
    assert model.embed(_iq(window=128)).shape == (_BATCH, 512)


def test_net_forward_directly() -> None:
    """The bare net maps (B, window, 2) -> (B, num_classes)."""
    net = ResNet1dNet(11, window=DEFAULT_WINDOW)
    net.eval()
    with torch.no_grad():
        assert net(torch.randn(_BATCH, DEFAULT_WINDOW, 2)).shape == (_BATCH, 11)


def test_scale_invariance() -> None:
    """Unit-average-power normalisation makes the logits scale-invariant."""
    model = ResNet1dSEI(device="cpu", num_classes=6)
    model.net.eval()
    x = torch.randn(_BATCH, DEFAULT_WINDOW, 2, generator=torch.Generator().manual_seed(2))
    b1 = {"iq": [r.tolist() for r in x]}
    b2 = {"iq": [(6.0 * r).tolist() for r in x]}
    assert torch.allclose(model.forward(b1), model.forward(b2), atol=1e-4)


def test_wrong_channel_raises() -> None:
    """A non-2 IQ axis fails loudly."""
    with pytest.raises(ValueError, match=r"shape \(B, 256, 2\)"):
        ResNet1dSEI(device="cpu").forward({"iq": torch.randn(_BATCH, DEFAULT_WINDOW, 3).tolist()})
