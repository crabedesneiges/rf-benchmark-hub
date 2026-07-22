"""Acceptance tests for the TLDNN AMC baseline (arXiv:2401.01056).

TLDNN is a torch baseline, so the whole module is guarded with ``pytest.importorskip("torch")``:
it SKIPS in the dependency-free ``.venv`` (no torch) and RUNS on the GPU ARM venv where
``rfbench[torch]`` is installed. No network, no real data -- the model is exercised on a synthetic
``(B, 2, N)`` I/Q batch shaped exactly like the one :class:`~rfbench.tasks.amc.dataset.AmcDataset`
yields (collated to ``x["iq"]`` a list of per-sample windows).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.tldnn import (  # noqa: E402
    DEFAULT_EMBED_DIM,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_HEADS,
    DEFAULT_TRANSFORMER_LAYERS,
    DEFAULT_WINDOW,
    TLDNN,
    ReGLU,
    TalkingHeadsAttention,
    TLDNNNet,
    _conv_layers_for_window,
    _iq_to_ap,
)

_BATCH = 4
_CLASSES_2018 = 24
_WINDOW_2018 = 1024


def _synthetic_iq_batch(batch: int = _BATCH, window: int = DEFAULT_WINDOW) -> dict[str, list]:
    """Build a collated AMC batch: ``x["iq"]`` a list of ``(2, window)`` nested-list windows.

    Mirrors the layout :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.amc.dataset.AmcDataset`, without importing numpy.
    """
    gen = torch.Generator().manual_seed(42)
    windows = torch.randn(batch, 2, window, generator=gen)
    return {"iq": [w.tolist() for w in windows]}


def test_tldnn_is_registered() -> None:
    """Importing the model module registers it under 'tldnn' -> the class (registry path)."""
    assert "tldnn" in MODELS
    assert MODELS.get("tldnn") is TLDNN


def test_tldnn_implements_model_contract() -> None:
    """TLDNN is a Model in the baseline family with a non-empty name."""
    model = TLDNN(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "tldnn"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, 128) batch returns (B, 11) finite logits."""
    model = TLDNN(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_embed_returns_2d_feature() -> None:
    """embed returns the (B, embed_dim) penultimate feature vector, one row per sample."""
    model = TLDNN(device="cpu")
    features = model.embed(_synthetic_iq_batch())
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape == (_BATCH, DEFAULT_EMBED_DIM)
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_small() -> None:
    """n_params > 0 and stays in TLDNN's lightweight envelope (a few hundred k params)."""
    n = TLDNN(device="cpu").n_params
    assert isinstance(n, int)
    assert 0 < n < 2_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, 11) logits yields per-sample length-num_classes score vectors."""
    logits = TLDNN(device="cpu").forward(_synthetic_iq_batch())
    rows = list(logits)
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, 128) window is promoted to a batch of one."""
    single = torch.randn(2, DEFAULT_WINDOW).tolist()
    logits = TLDNN(device="cpu").forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    bad = torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        TLDNN(device="cpu").forward({"iq": bad})


def test_conv_layers_rule_matches_paper() -> None:
    """K = 2 for the 128-sample dataset, K = 4 for the 1024-sample one (Sec. III-A / Table I)."""
    assert _conv_layers_for_window(128) == 2
    assert _conv_layers_for_window(1024) == 4


def test_net_2016_token_length_is_32() -> None:
    """RadioML 2016.10a (N=128, K=2) yields L = 32 tokens (paper's Xe in R^{32x64})."""
    net = TLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    assert net.seq_len == 32
    assert net.pos_encoding.shape == (1, 32, DEFAULT_EMBED_DIM)


def test_net_2018_token_length_is_64() -> None:
    """RadioML 2018.01a (N=1024, K=4) yields L = 64 tokens (paper's Xe in R^{64x64})."""
    net = TLDNNNet(_CLASSES_2018, window=_WINDOW_2018)
    assert net.conv_layers == 4
    assert net.seq_len == 64


def test_net_forward_shape_directly_for_2018() -> None:
    """The bare net maps (B, 2, 1024) -> (B, 24) for the 2018.01a configuration."""
    net = TLDNNNet(_CLASSES_2018, window=_WINDOW_2018)
    net.eval()
    x = torch.randn(_BATCH, 2, _WINDOW_2018)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, _CLASSES_2018)
        assert net.features(x).shape == (_BATCH, DEFAULT_EMBED_DIM)


def test_iq_to_ap_ranges() -> None:
    """The A/P transform yields amplitude in [0, 1] and phase in [-1, 1] (Sec. III, eqs. 5-6)."""
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    ap = _iq_to_ap(x)
    assert ap.shape == (_BATCH, 2, DEFAULT_WINDOW)
    amplitude, phase = ap[:, 0, :], ap[:, 1, :]
    assert amplitude.min() >= 0.0 - 1e-6
    assert amplitude.max() <= 1.0 + 1e-6
    assert phase.min() >= -1.0 - 1e-6
    assert phase.max() <= 1.0 + 1e-6


def test_forward_is_ap_scale_invariant() -> None:
    """The A/P input (min-max amplitude) removes absolute capture scale: x and 100x agree.

    TLDNN consumes normalized amplitude/phase, so scaling the raw I/Q by a constant must leave
    the logits unchanged -- confirming the A/P transform (which conditions the input, in lieu of
    the raw-IQ normalization the CNN baselines need) lives inside the net and is scale-invariant.
    """
    net = TLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x * 100.0), atol=1e-4)


def test_ap_transform_runs_inside_net_at_train_time() -> None:
    """The net's own forward applies A/P, so the from-scratch loop (net(raw_iq)) sees it.

    Guards the training-path trap: rfbench optimises ``model.net`` directly by calling
    ``net(raw_iq)``. If the A/P transform lived only in the wrapper it would be bypassed at train
    time. Feeding raw I/Q straight to the net must equal feeding the manually A/P-transformed
    input to a net that skips the transform -- here we assert the net's forward differs from
    running the embedding on the RAW (un-transformed) signal, i.e. the transform is active.
    """
    net = TLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        via_net = net.features(x)  # net applies A/P internally
        # Same pipeline but skipping the A/P transform (raw IQ straight into the conv stem).
        raw_tokens = net.embedding(x) + net.pos_encoding
        for layer in net.encoder:
            raw_tokens = layer(raw_tokens)
        out, _ = net.lstm(raw_tokens)
        via_raw = net.fc_embed(out[:, -1, :])
    assert not torch.allclose(via_net, via_raw, atol=1e-4)


def test_talking_heads_identity_init_equals_plain_mha() -> None:
    """At init Pl = Pw = I, so talking-heads attention is exactly ordinary multi-head attention."""
    attn = TalkingHeadsAttention(DEFAULT_EMBED_DIM, DEFAULT_NUM_HEADS)
    assert torch.allclose(attn.pl, torch.eye(DEFAULT_NUM_HEADS))
    assert torch.allclose(attn.pw, torch.eye(DEFAULT_NUM_HEADS))


def test_reglu_preserves_shape_and_width() -> None:
    """ReGLU maps (B, L, d) -> (B, L, d) with feed-forward width dffn = 2d (Sec. III-B)."""
    ffn = ReGLU(DEFAULT_EMBED_DIM, 2 * DEFAULT_EMBED_DIM)
    assert ffn.w1.out_features == 2 * DEFAULT_EMBED_DIM
    x = torch.randn(_BATCH, 8, DEFAULT_EMBED_DIM)
    assert ffn(x).shape == x.shape


def test_architecture_hyperparams_match_paper() -> None:
    """Mt=2 transformer layers, h=8 heads, Ml=4 LSTM layers, dl=d=64 (Sec. III-B/C, Table)."""
    net = TLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    assert len(net.encoder) == DEFAULT_TRANSFORMER_LAYERS == 2
    assert net.encoder[0].attn.num_heads == DEFAULT_NUM_HEADS == 8
    assert net.lstm.num_layers == DEFAULT_LSTM_LAYERS == 4
    assert net.lstm.hidden_size == DEFAULT_EMBED_DIM
    assert net.encoder[0].attn.head_dim == DEFAULT_EMBED_DIM // DEFAULT_NUM_HEADS


def test_dropout_inactive_at_eval() -> None:
    """Under eval the model is deterministic across repeated calls (dropout is a no-op)."""
    net = TLDNNNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x))


def test_phase_normalization_constant() -> None:
    """Phase is atan2/pi so a pure +j signal (phase pi/2) normalizes to 0.5."""
    x = torch.zeros(1, 2, 4)
    x[:, 1, :] = 1.0  # Q = 1, I = 0 -> phase = atan2(1, 0) = pi/2
    ap = _iq_to_ap(x)
    assert torch.allclose(ap[:, 1, :], torch.full((1, 4), 0.5), atol=1e-6)
    assert math.isclose(float(ap[0, 1, 0]), 0.5, abs_tol=1e-6)
