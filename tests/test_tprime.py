"""Acceptance tests for the T-PRIME baseline (protocol-tech-ID task).

The T-PRIME model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and RUNS
on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data -- the model
is exercised on a synthetic ``(B, 2, N)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset` yields (collated to ``x["iq"]``
a list of per-sample channel-first IQ windows).

The registry-resolution check (``"tprime" in MODELS`` after importing the model module) is the
one assertion that does not need torch to reason about, but it does need the module to import,
which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.tprime import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    LG_SEQUENCE_LEN,
    SM_SEQUENCE_LEN,
    TPrime,
    TPrimeNet,
)

_BATCH = 4


def _synthetic_iq_batch(batch: int = _BATCH, seq_len: int = SM_SEQUENCE_LEN) -> dict[str, list]:
    """Build a collated protocol batch: ``x["iq"]`` a list of ``(2, seq_len)`` windows.

    Mirrors the layout :func:`rfbench.core.evaluate.evaluate` collates from
    :class:`~rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset` (per-sample IQ of shape
    ``(2, L)``, channel-first), without importing numpy: nested Python lists are accepted by
    ``torch.as_tensor``.
    """
    gen = torch.Generator().manual_seed(42)
    windows = torch.randn(batch, 2, seq_len, generator=gen)
    return {"iq": [w.tolist() for w in windows]}


def test_tprime_is_registered() -> None:
    """Importing the model module registers it under 'tprime' -> the class (registry path)."""
    assert "tprime" in MODELS
    assert MODELS.get("tprime") is TPrime


def test_tprime_implements_model_contract() -> None:
    """TPrime is a Model in the baseline family with a non-empty name."""
    model = TPrime(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "tprime"
    assert model.family == "baseline"
    assert model.variant == "SM"  # SM is the default variant


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, N) batch returns (B, num_classes) protocol logits."""
    model = TPrime(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_respects_custom_num_classes() -> None:
    """A custom class count flows through to the head width."""
    model = TPrime(device="cpu", num_classes=6)
    logits = model.forward(_synthetic_iq_batch())
    assert logits.shape == (_BATCH, 6)


def test_embed_returns_2d_feature() -> None:
    """embed returns a 2-D (B, D) pooled encoder feature, one row per sample."""
    model = TPrime(device="cpu")
    features = model.embed(_synthetic_iq_batch())
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == _BATCH
    assert features.shape[1] > 0
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in the published SM envelope (~1.6M params)."""
    model = TPrime(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 5_000_000  # SM is ~1.6M per the T-PRIME paper


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, num_classes) logits yields per-sample rows the metric argmaxes."""
    model = TPrime(device="cpu")
    rows = list(model.forward(_synthetic_iq_batch()))
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, N) window is promoted to a batch of one."""
    model = TPrime(device="cpu")
    single = torch.randn(2, SM_SEQUENCE_LEN).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = TPrime(device="cpu")
    bad = torch.randn(_BATCH, 3, SM_SEQUENCE_LEN).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        model.forward({"iq": bad})


def test_variable_window_is_cropped_or_padded() -> None:
    """A window longer/shorter than N is centre-cropped / zero-padded to the variant's N."""
    model = TPrime(device="cpu")
    longer = model.forward(_synthetic_iq_batch(seq_len=SM_SEQUENCE_LEN + 200))
    shorter = model.forward(_synthetic_iq_batch(seq_len=SM_SEQUENCE_LEN - 200))
    assert longer.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert shorter.shape == (_BATCH, DEFAULT_NUM_CLASSES)


def test_lg_variant_forward_and_size() -> None:
    """The LG variant tokenises the longer window and has more params than SM."""
    sm = TPrime(device="cpu", variant="SM")
    lg = TPrime(device="cpu", variant="LG")
    assert lg.sequence_len == LG_SEQUENCE_LEN
    logits = lg.forward(_synthetic_iq_batch(seq_len=LG_SEQUENCE_LEN))
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert lg.n_params > sm.n_params


def test_net_forward_shape_directly() -> None:
    """The bare TPrimeNet maps (B, 2, N) -> (B, num_classes) and features -> (B, d_model)."""
    net = TPrimeNet(DEFAULT_NUM_CLASSES, variant="SM")
    net.eval()
    x = torch.randn(_BATCH, 2, SM_SEQUENCE_LEN)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        feats = net.features(x)
        assert feats.ndim == 2
        assert feats.shape[1] == net.d_model


def test_unknown_variant_rejected() -> None:
    """An unknown variant id fails loudly at construction."""
    with pytest.raises(ValueError, match="unknown T-PRIME variant"):
        TPrimeNet(DEFAULT_NUM_CLASSES, variant="XL")  # type: ignore[arg-type]
