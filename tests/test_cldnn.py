"""WP-30 acceptance tests for the CLDNN AMC baseline.

The CLDNN model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and
RUNS on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data --
the model is exercised on a synthetic ``(B, 2, 128)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.amc.dataset.AmcDataset` yields (collated to ``x["iq"]`` a list of
per-sample windows).

The registry-resolution check (``"cldnn" in MODELS`` after importing the model module) is
the one assertion that does not need torch to reason about, but it does need the module to
import, which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.cldnn import (  # noqa: E402
    CLDNN,
    DEFAULT_CONV_FILTERS,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    CLDNNNet,
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


def test_cldnn_is_registered() -> None:
    """Importing the model module registers it under 'cldnn' -> the class (registry path)."""
    assert "cldnn" in MODELS
    assert MODELS.get("cldnn") is CLDNN


def test_cldnn_implements_model_contract() -> None:
    """CLDNN is a Model in the baseline family with a non-empty name."""
    model = CLDNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "cldnn"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, 128) batch returns (B, 11) logits."""
    model = CLDNN(device="cpu")
    batch = _synthetic_iq_batch()
    logits = model.forward(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_embed_returns_2d_feature() -> None:
    """embed returns a 2-D (B, D) penultimate feature vector, one row per sample."""
    model = CLDNN(device="cpu")
    batch = _synthetic_iq_batch()
    features = model.embed(batch)
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == _BATCH
    assert features.shape[1] > 0
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in CLDNN's single-digit-millions envelope (not a heavy backbone)."""
    model = CLDNN(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 10_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, 11) logits yields per-sample score vectors the AMC metrics argmax.

    The AMC metrics decode each prediction row via a lazy argmax, so a per-sample logits row
    must be a length-``num_classes`` sequence -- asserted here so the model plugs into the
    metric path without a shape surprise.
    """
    model = CLDNN(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    rows = list(logits)
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, 128) window is promoted to a batch of one."""
    model = CLDNN(device="cpu")
    single = torch.randn(2, DEFAULT_WINDOW).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = CLDNN(device="cpu")
    bad = torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        model.forward({"iq": bad})


def test_net_forward_shape_directly() -> None:
    """The bare CLDNNNet maps (B, 2, L) -> (B, num_classes) and features -> (B, D)."""
    net = CLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        assert net.features(x).ndim == 2


def test_lstm_has_three_stacked_layers() -> None:
    """Paper-exact (West & O'Shea 2017 §B.2): the recurrent stack is THREE LSTM layers, not two."""
    net = CLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    assert DEFAULT_LSTM_LAYERS == 3
    assert net.lstm.num_layers == 3


def test_raw_waveform_skip_widens_lstm_input() -> None:
    """Paper-exact (§B.2): the raw-IQ bypass concatenates the 2 IQ channels onto conv features.

    The recurrent stack therefore consumes ``conv_filters + 2`` features per timestep -- proof
    the raw waveform is fused in, not dropped -- and the fused conv sequence carries that width.
    """
    net = CLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    expected = DEFAULT_CONV_FILTERS + 2
    assert net.lstm_input_size == expected
    assert net.lstm.input_size == expected

    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        seq = net._conv_sequence(x)
    # (B, L, conv_filters + 2): time axis preserved, raw IQ concatenated onto the feature axis.
    assert seq.shape == (_BATCH, DEFAULT_WINDOW, expected)


def test_raw_waveform_skip_preserves_raw_iq_tail() -> None:
    """The skip concatenates the UNTOUCHED IQ: the last 2 feature columns equal the raw waveform.

    ``_conv_sequence`` returns ``(B, L, conv_filters + 2)`` with the raw IQ appended after the
    conv channels, so transposing the raw ``(B, 2, L)`` input must match the trailing 2 columns
    exactly -- guarding against a future refactor that silently reorders or drops the bypass.
    """
    net = CLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        seq = net._conv_sequence(x)
    raw_tail = seq[:, :, DEFAULT_CONV_FILTERS:]  # (B, L, 2)
    assert torch.equal(raw_tail, x.transpose(1, 2))
