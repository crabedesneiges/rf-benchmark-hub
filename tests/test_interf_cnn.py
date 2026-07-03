"""Acceptance tests for the interference-CNN baseline (interference-ID task).

The interference-CNN model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and RUNS
on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data -- the model
is exercised on a synthetic ``(B, 2, L)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.interference_id.dataset.InterferenceDataset` yields (collated to
``x["iq"]`` a list of per-sample channel-first IQ windows).

The registry-resolution check (``"interf_cnn" in MODELS`` after importing the model module) is
the one assertion that does not need torch to reason about, but it does need the module to
import, which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.interf_cnn import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    InterferenceCNN,
    InterferenceCNNNet,
)

_BATCH = 4


def _synthetic_iq_batch(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    """Build a collated interference batch: ``x["iq"]`` a list of ``(2, window)`` windows.

    Mirrors the layout :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.interference_id.dataset.InterferenceDataset` (per-sample IQ of shape
    ``(2, L)``, channel-first), without importing numpy: nested Python lists are accepted by
    ``torch.as_tensor``.
    """
    gen = torch.Generator().manual_seed(42)
    windows = torch.randn(batch, 2, window, generator=gen)
    return {"iq": [w.tolist() for w in windows]}


def test_interf_cnn_is_registered() -> None:
    """Importing the model module registers it under 'interf_cnn' -> the class (registry path)."""
    assert "interf_cnn" in MODELS
    assert MODELS.get("interf_cnn") is InterferenceCNN


def test_interf_cnn_implements_model_contract() -> None:
    """InterferenceCNN is a Model in the baseline family with a non-empty name."""
    model = InterferenceCNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "interf_cnn"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, L) batch returns (B, num_classes) interference logits."""
    model = InterferenceCNN(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_respects_custom_num_classes() -> None:
    """A custom class count flows through to the head width."""
    model = InterferenceCNN(device="cpu", num_classes=3)
    logits = model.forward(_synthetic_iq_batch())
    assert logits.shape == (_BATCH, 3)


def test_embed_returns_2d_feature() -> None:
    """embed returns a 2-D (B, D) penultimate feature vector, one row per sample."""
    model = InterferenceCNN(device="cpu")
    features = model.embed(_synthetic_iq_batch())
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == _BATCH
    assert features.shape[1] > 0
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in the compact board-seeding envelope (not a heavy backbone)."""
    model = InterferenceCNN(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 10_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, num_classes) logits yields per-sample rows the metric argmaxes."""
    model = InterferenceCNN(device="cpu")
    rows = list(model.forward(_synthetic_iq_batch()))
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, L) window is promoted to a batch of one."""
    model = InterferenceCNN(device="cpu")
    single = torch.randn(2, DEFAULT_WINDOW).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = InterferenceCNN(device="cpu")
    bad = torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        model.forward({"iq": bad})


def test_net_forward_shape_directly() -> None:
    """The bare InterferenceCNNNet maps (B, 2, L) -> (B, num_classes) and features -> (B, D)."""
    net = InterferenceCNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        assert net.features(x).ndim == 2


def test_variable_window_still_embeds() -> None:
    """The global pool makes the embedding width window-agnostic (adaptive pool over time)."""
    model = InterferenceCNN(device="cpu", window=256)
    features = model.embed(_synthetic_iq_batch(window=256))
    assert features.shape[0] == _BATCH
    assert features.ndim == 2
