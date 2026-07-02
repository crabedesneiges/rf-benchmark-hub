"""WP-30 acceptance tests for the MCLDNN AMC baseline.

The MCLDNN model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and
RUNS on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data --
the model is exercised on a synthetic ``(B, 2, 128)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.amc.dataset.AmcDataset` yields (collated to ``x["iq"]`` a list of
per-sample windows).

The registry-resolution check (``"mcldnn" in MODELS`` after importing the model module) is
the one assertion that does not need torch to reason about, but it does need the module to
import, which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.mcldnn import (  # noqa: E402
    DEFAULT_CONV_FILTERS,
    DEFAULT_FUSE_FILTERS,
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_LSTM_HIDDEN,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    MCLDNN,
    MCLDNNNet,
)

_BATCH = 4


def _synthetic_iq_batch(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    """Build a collated AMC batch: ``x["iq"]`` a list of ``(2, window)`` nested-list windows.

    Mirrors the layout :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.amc.dataset.AmcDataset` (per-sample IQ of shape ``(2, window)``),
    without importing numpy: nested Python lists are accepted by ``torch.as_tensor``.
    """
    gen = torch.Generator().manual_seed(42)
    windows = torch.randn(batch, 2, window, generator=gen)
    return {"iq": [w.tolist() for w in windows]}


def test_mcldnn_is_registered() -> None:
    """Importing the model module registers it under 'mcldnn' -> the class (registry path)."""
    assert "mcldnn" in MODELS
    assert MODELS.get("mcldnn") is MCLDNN


def test_mcldnn_implements_model_contract() -> None:
    """MCLDNN is a Model in the baseline family with a non-empty name."""
    model = MCLDNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "mcldnn"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, 128) batch returns (B, 11) logits."""
    model = MCLDNN(device="cpu")
    batch = _synthetic_iq_batch()
    logits = model.forward(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_embed_returns_2d_feature() -> None:
    """embed returns the (B, lstm_hidden) penultimate feature vector, one row per sample.

    The paper dense head keeps the penultimate width at ``lstm_hidden`` (128): the two SELU
    ``Dense(128)`` layers preserve the 128-d embedding the probing regimes fit a head on.
    """
    model = MCLDNN(device="cpu")
    batch = _synthetic_iq_batch()
    features = model.embed(batch)
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape == (_BATCH, DEFAULT_LSTM_HIDDEN)
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in MCLDNN's single-digit-millions envelope (not a heavy backbone)."""
    model = MCLDNN(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 10_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, 11) logits yields per-sample score vectors the AMC metrics argmax.

    The AMC metrics decode each prediction row via a lazy argmax, so a per-sample logits row
    must be a length-``num_classes`` sequence -- asserted here so the model plugs into the
    metric path without a shape surprise.
    """
    model = MCLDNN(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    rows = list(logits)
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, 128) window is promoted to a batch of one."""
    model = MCLDNN(device="cpu")
    single = torch.randn(2, DEFAULT_WINDOW).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = MCLDNN(device="cpu")
    bad = torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        model.forward({"iq": bad})


def test_net_forward_shape_directly() -> None:
    """The bare MCLDNNNet maps (B, 2, L) -> (B, num_classes) and features -> (B, D)."""
    net = MCLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        assert net.features(x).ndim == 2


def test_fusion_conv_uses_100_filters() -> None:
    """Paper-exact fusion conv outputs 100 filters (Xu et al.), not the 50 per-branch width.

    BIBLIOGRAPHY.md B.1 flags the 50-filter fusion as a MISMATCH; the paper's ``Conv2D(100,(2,5))``
    then drives the LSTM ``input_size``, so both must be 100.
    """
    net = MCLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    fuse_conv = net.conv_fuse[0]
    assert isinstance(fuse_conv, torch.nn.Conv2d)
    assert fuse_conv.out_channels == DEFAULT_FUSE_FILTERS == 100
    assert fuse_conv.in_channels == DEFAULT_CONV_FILTERS == 50
    # The widened fusion feeds the LSTM: input_size must track fuse_filters.
    assert net.lstm.input_size == DEFAULT_FUSE_FILTERS


def test_dense_head_has_two_dense_and_two_dropout() -> None:
    """The restored head is Dense -> Dropout(0.5) -> Dense -> Dropout(0.5) (Xu et al. 2020).

    B.1 flagged the head as a gap driver: it had lost its two ``Dropout(0.5)`` layers and its
    second ``Dense``. Assert exactly two ``Linear`` and two ``Dropout(p=0.5)`` in ``fc_embed``.
    """
    net = MCLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    linears = [m for m in net.fc_embed if isinstance(m, torch.nn.Linear)]
    dropouts = [m for m in net.fc_embed if isinstance(m, torch.nn.Dropout)]
    assert len(linears) == 2
    assert all(lin.in_features == lin.out_features == DEFAULT_LSTM_HIDDEN for lin in linears)
    assert len(dropouts) == 2
    assert all(d.p == DEFAULT_HEAD_DROPOUT == 0.5 for d in dropouts)


def test_dropout_is_inactive_at_eval() -> None:
    """Head dropout is a no-op under eval: forward is deterministic across repeated calls.

    The Model contract evaluates in ``.eval()`` with grads disabled, so the restored dropouts
    must not perturb inference; they only regularize the M3 from-scratch training loop.
    """
    net = MCLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        first = net(x)
        second = net(x)
    assert torch.allclose(first, second)
