"""Acceptance tests for the paper-faithful WiSig ManyTx 2-D CNN (`wisig_cnn_paper`).

Torch-gated (SKIPs in the dep-free venv). Exercises the model on synthetic ``(B, 256, 2)`` IQ
batches shaped exactly like the collated ``x["iq"]`` list :class:`SeiDataset` yields, and asserts
the paper-exact structure: the 5-conv / 4-pool stack flattening to 256, the ``(100, 80, N)`` dense
head with dropout before the classifier, L2 exposed on the three Dense kernels ONLY, and the
per-signal unit-average-power input normalisation (scale-invariant logits).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.wisig_cnn_paper import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    WiSigCNNPaper,
    WiSigCNNPaperNet,
)

_BATCH = 4


def _iq(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    gen = torch.Generator().manual_seed(42)
    return {"iq": [w.tolist() for w in torch.randn(batch, window, 2, generator=gen)]}


def test_registered_under_wisig_cnn_paper() -> None:
    """Importing the module registers it under 'wisig_cnn_paper'."""
    assert "wisig_cnn_paper" in MODELS
    assert MODELS.get("wisig_cnn_paper") is WiSigCNNPaper


def test_model_contract() -> None:
    """WiSigCNNPaper is a baseline-family Model with the right name."""
    model = WiSigCNNPaper(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "wisig_cnn_paper"
    assert model.family == "baseline"


def test_forward_and_embed_shapes() -> None:
    """forward -> (B, n_tx) logits; embed -> (B, 80) penultimate feature (Dense(80) width)."""
    model = WiSigCNNPaper(num_classes=DEFAULT_NUM_CLASSES, device="cpu")
    logits = model.forward(_iq())
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()
    feats = model.embed(_iq())
    assert feats.shape == (_BATCH, 80)


def test_custom_num_classes() -> None:
    """A per-split transmitter count flows to the head width."""
    assert WiSigCNNPaper(device="cpu", num_classes=7).forward(_iq()).shape == (_BATCH, 7)


def test_net_flatten_dim_is_256() -> None:
    """The 5-conv / 4-pool stack over (1, 256, 2) flattens to exactly 16*16*1 = 256."""
    net = WiSigCNNPaperNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    probe = torch.zeros(1, 1, DEFAULT_WINDOW, 2)
    assert net.conv(probe).flatten(1).shape[1] == 256


def test_l2_penalty_covers_only_dense_kernels() -> None:
    """l2_penalty == sum of squared Dense(100/80/N) kernels ONLY -- never the conv kernels."""
    net = WiSigCNNPaperNet(5, window=DEFAULT_WINDOW)
    expected = (
        net.dense_100.weight.pow(2).sum()
        + net.dense_80.weight.pow(2).sum()
        + net.classifier.weight.pow(2).sum()
    )
    assert torch.allclose(net.l2_penalty(), expected)
    # A conv weight change must NOT move the penalty (convs are unregularised in the paper).
    before = float(net.l2_penalty())
    with torch.no_grad():
        for module in net.conv.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.add_(1.0)
    assert float(net.l2_penalty()) == pytest.approx(before)


def test_unit_average_power_makes_logits_scale_invariant() -> None:
    """Scaling the input by a constant leaves the logits unchanged (unit-average-power norm)."""
    model = WiSigCNNPaper(device="cpu", num_classes=6)
    model.net.eval()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, DEFAULT_WINDOW, 2, generator=gen)
    b1 = {"iq": [r.tolist() for r in x]}
    b2 = {"iq": [(5.0 * r).tolist() for r in x]}
    assert torch.allclose(model.forward(b1), model.forward(b2), atol=1e-4)


def test_unbatched_sample_and_wrong_channel() -> None:
    """A single (256, 2) window is promoted to a batch of one; a non-2 IQ axis raises."""
    model = WiSigCNNPaper(device="cpu")
    single = torch.randn(DEFAULT_WINDOW, 2).tolist()
    assert model.forward({"iq": single}).shape == (1, DEFAULT_NUM_CLASSES)
    with pytest.raises(ValueError, match=r"shape \(B, 256, 2\)"):
        model.forward({"iq": torch.randn(_BATCH, DEFAULT_WINDOW, 3).tolist()})


def test_n_params_is_compact() -> None:
    """The paper CNN is compact (~50k params for 150 tx), not a heavy backbone."""
    n = WiSigCNNPaper(device="cpu").n_params
    assert 0 < n < 1_000_000
