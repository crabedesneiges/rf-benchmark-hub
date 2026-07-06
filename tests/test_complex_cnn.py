"""Acceptance tests for the complex-valued CNN SEI baseline (`complex_cnn`).

Torch-gated. Exercises the ``network_20_modrelu_short`` reconstruction on synthetic ``(B, 256, 2)``
IQ, plus focused correctness tests for the two custom complex primitives (:class:`ComplexConv1d`
complex-multiply, :class:`ModReLU` magnitude threshold + phase preservation).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.complex_cnn import (  # noqa: E402
    DEFAULT_WINDOW,
    ComplexCNN,
    ComplexCNNNet,
    ComplexConv1d,
    ModReLU,
)

_BATCH = 4


def _iq(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    gen = torch.Generator().manual_seed(3)
    return {"iq": [w.tolist() for w in torch.randn(batch, window, 2, generator=gen)]}


def test_registered_and_contract() -> None:
    """Registered under 'complex_cnn'; a baseline-family Model; forward/embed shapes."""
    assert MODELS.get("complex_cnn") is ComplexCNN
    model = ComplexCNN(device="cpu", num_classes=12)
    assert isinstance(model, Model)
    assert model.family == "baseline"
    assert model.forward(_iq()).shape == (_BATCH, 12)
    assert model.embed(_iq()).shape == (_BATCH, 100)


def test_short_window_raises() -> None:
    """A window too short for the (kernel-20/stride-10, kernel-10) convs raises clearly."""
    with pytest.raises(ValueError, match="window must be >="):
        ComplexCNNNet(4, window=64)


def test_complex_conv_is_complex_multiply() -> None:
    """ComplexConv1d((xr,xi)) == (conv_r xr - conv_i xi, conv_r xi + conv_i xr)."""
    conv = ComplexConv1d(1, 3, 4, stride=1)
    xr = torch.randn(2, 1, 10)
    xi = torch.randn(2, 1, 10)
    yr, yi = conv(xr, xi)
    assert torch.allclose(yr, conv.conv_r(xr) - conv.conv_i(xi))
    assert torch.allclose(yi, conv.conv_r(xi) + conv.conv_i(xr))


def test_modrelu_preserves_phase_and_thresholds_magnitude() -> None:
    """modReLU keeps the phase z/|z| and scales magnitude by relu(|z|+b)/|z|."""
    mr = ModReLU(1)
    with torch.no_grad():
        mr.b.fill_(0.0)  # b=0 -> relu(|z|)/|z| = 1 for |z|>0, so (xr, xi) pass through unchanged
    xr = torch.tensor([[[3.0, -1.0]]])
    xi = torch.tensor([[[4.0, 0.0]]])
    yr, yi = mr(xr, xi)
    assert torch.allclose(yr, xr, atol=1e-4)
    assert torch.allclose(yi, xi, atol=1e-4)
    # A large negative bias zeroes activations whose magnitude falls below the threshold.
    with torch.no_grad():
        mr.b.fill_(-100.0)
    yr2, yi2 = mr(xr, xi)
    assert torch.allclose(yr2, torch.zeros_like(yr2), atol=1e-3)
    assert torch.allclose(yi2, torch.zeros_like(yi2), atol=1e-3)


def test_l2_penalty_covers_complex_and_dense_kernels() -> None:
    """l2_penalty sums the (real+imag) conv kernels and the dense kernels."""
    net = ComplexCNNNet(4, window=DEFAULT_WINDOW)
    expected = (
        net.cconv1.kernel_l2()
        + net.cconv2.kernel_l2()
        + net.dense.weight.pow(2).sum()
        + net.classifier.weight.pow(2).sum()
    )
    assert torch.allclose(net.l2_penalty(), expected)


def test_wrong_channel_raises() -> None:
    """A non-2 IQ axis fails loudly."""
    with pytest.raises(ValueError, match=r"shape \(B, 256, 2\)"):
        ComplexCNN(device="cpu").forward({"iq": torch.randn(_BATCH, DEFAULT_WINDOW, 3).tolist()})
