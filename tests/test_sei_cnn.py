"""Acceptance tests for the WiSig-CNN SEI baseline (Track A).

The WiSig-CNN model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and
RUNS on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data --
the model is exercised on a synthetic ``(B, 256, 2)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.sei.dataset.SeiDataset` yields (collated to ``x["iq"]`` a list of
per-sample WiSig windows in the ``(256, 2)`` time-major layout).

The registry-resolution check (``"wisig_cnn" in MODELS`` after importing the model module) is
the one assertion that does not need torch to reason about, but it does need the module to
import, which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.sei_cnn import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    WiSigCNN,
    WiSigCNNNet,
)

_BATCH = 4


def _synthetic_iq_batch(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    """Build a collated SEI batch: ``x["iq"]`` a list of ``(window, 2)`` nested-list windows.

    Mirrors the layout :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.sei.dataset.SeiDataset` (per-sample WiSig IQ of shape ``(256, 2)``,
    time-major), without importing numpy: nested Python lists are accepted by
    ``torch.as_tensor``.
    """
    gen = torch.Generator().manual_seed(42)
    windows = torch.randn(batch, window, 2, generator=gen)
    return {"iq": [w.tolist() for w in windows]}


def test_wisig_cnn_is_registered() -> None:
    """Importing the model module registers it under 'wisig_cnn' -> the class (registry path)."""
    assert "wisig_cnn" in MODELS
    assert MODELS.get("wisig_cnn") is WiSigCNN


def test_wisig_cnn_implements_model_contract() -> None:
    """WiSigCNN is a Model in the baseline family with a non-empty name."""
    model = WiSigCNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "wisig_cnn"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 256, 2) batch returns (B, n_tx) transmitter logits."""
    model = WiSigCNN(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_respects_custom_num_classes() -> None:
    """A per-split transmitter count flows through to the head width (Track B labels index tx)."""
    model = WiSigCNN(device="cpu", num_classes=7)
    logits = model.forward(_synthetic_iq_batch())
    assert logits.shape == (_BATCH, 7)


def test_embed_returns_2d_feature() -> None:
    """embed returns a 2-D (B, D) penultimate feature vector, one row per sample."""
    model = WiSigCNN(device="cpu")
    features = model.embed(_synthetic_iq_batch())
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == _BATCH
    assert features.shape[1] > 0
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in the compact board-seeding envelope (not a heavy backbone)."""
    model = WiSigCNN(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 10_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, n_tx) logits yields per-sample rows Rank1Accuracy argmaxes.

    The closed-set SEI metric decodes each prediction row via a lazy argmax, so a per-sample
    logits row must be a length-``num_classes`` sequence -- asserted here so the model plugs
    into the metric path without a shape surprise.
    """
    model = WiSigCNN(device="cpu")
    rows = list(model.forward(_synthetic_iq_batch()))
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (256, 2) window is promoted to a batch of one."""
    model = WiSigCNN(device="cpu")
    single = torch.randn(DEFAULT_WINDOW, 2).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = WiSigCNN(device="cpu")
    bad = torch.randn(_BATCH, DEFAULT_WINDOW, 3).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 256, 2\)"):
        model.forward({"iq": bad})


def test_net_forward_shape_directly() -> None:
    """The bare WiSigCNNNet maps (B, 2, L) -> (B, num_classes) and features -> (B, D)."""
    net = WiSigCNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        assert net.features(x).ndim == 2


def test_variable_window_still_embeds() -> None:
    """The global pool makes the embedding width window-agnostic (adaptive pool over time)."""
    model = WiSigCNN(device="cpu", window=128)
    features = model.embed(_synthetic_iq_batch(window=128))
    assert features.shape[0] == _BATCH
    assert features.ndim == 2
