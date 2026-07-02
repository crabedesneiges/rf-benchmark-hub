"""WP-30 acceptance tests for the ResNet AMC baseline.

The ResNet AMC model is a torch baseline, so the whole module is guarded with
``pytest.importorskip("torch")``: it SKIPS in the dependency-free ``.venv`` (no torch) and RUNS
on the GPU ARM venv where ``rfbench[torch]`` is installed. No network, no real data -- the model
is exercised on a synthetic ``(B, 2, 128)`` IQ batch shaped exactly like the one
:class:`~rfbench.tasks.amc.dataset.AmcDataset` yields (collated to ``x["iq"]`` a list of
per-sample windows).

The registry-resolution check (``"resnet_amc" in MODELS`` after importing the model module) is
the one assertion that does not need torch to reason about, but it does need the module to
import, which pulls torch in -- so it lives behind the same guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rfbench.core.model import Model  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.resnet_amc import (  # noqa: E402
    DEFAULT_ALPHA_DROPOUT,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_STACKS,
    DEFAULT_WINDOW,
    ResNetAMC,
    ResNetAMCNet,
    _unit_variance_normalize,
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


def test_resnet_amc_is_registered() -> None:
    """Importing the model module registers it under 'resnet_amc' -> the class (registry path)."""
    assert "resnet_amc" in MODELS
    assert MODELS.get("resnet_amc") is ResNetAMC


def test_resnet_amc_implements_model_contract() -> None:
    """ResNetAMC is a Model in the baseline family with a non-empty name."""
    model = ResNetAMC(device="cpu")
    assert isinstance(model, Model)
    assert model.name == "resnet_amc"
    assert model.family == "baseline"


def test_forward_returns_class_logits() -> None:
    """forward on a synthetic (B, 2, 128) batch returns (B, 11) logits."""
    model = ResNetAMC(device="cpu")
    batch = _synthetic_iq_batch()
    logits = model.forward(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (_BATCH, DEFAULT_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_embed_returns_2d_feature() -> None:
    """embed returns a 2-D (B, D) penultimate feature vector, one row per sample."""
    model = ResNetAMC(device="cpu")
    batch = _synthetic_iq_batch()
    features = model.embed(batch)
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == _BATCH
    assert features.shape[1] > 0
    assert torch.isfinite(features).all()


def test_n_params_is_positive_and_reasonable() -> None:
    """n_params > 0 and stays in a from-scratch baseline envelope (not a heavy backbone)."""
    model = ResNetAMC(device="cpu")
    n = model.n_params
    assert isinstance(n, int)
    assert 0 < n < 10_000_000


def test_forward_argmax_decodes_per_sample() -> None:
    """Iterating the (B, 11) logits yields per-sample score vectors the AMC metrics argmax.

    The AMC metrics decode each prediction row via a lazy argmax, so a per-sample logits row
    must be a length-``num_classes`` sequence -- asserted here so the model plugs into the
    metric path without a shape surprise.
    """
    model = ResNetAMC(device="cpu")
    logits = model.forward(_synthetic_iq_batch())
    rows = list(logits)
    assert len(rows) == _BATCH
    assert all(len(row) == DEFAULT_NUM_CLASSES for row in rows)


def test_unbatched_sample_is_accepted() -> None:
    """A single unbatched (2, 128) window is promoted to a batch of one."""
    model = ResNetAMC(device="cpu")
    single = torch.randn(2, DEFAULT_WINDOW).tolist()
    logits = model.forward({"iq": single})
    assert logits.shape == (1, DEFAULT_NUM_CLASSES)


def test_wrong_channel_count_raises() -> None:
    """A batch whose channel axis is not 2 fails loudly rather than mis-classifying."""
    model = ResNetAMC(device="cpu")
    bad = torch.randn(_BATCH, 3, DEFAULT_WINDOW).tolist()
    with pytest.raises(ValueError, match=r"shape \(B, 2,"):
        model.forward({"iq": bad})


def test_net_forward_shape_directly() -> None:
    """The bare ResNetAMCNet maps (B, 2, L) -> (B, num_classes) and features -> (B, D)."""
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW)
    with torch.no_grad():
        assert net(x).shape == (_BATCH, DEFAULT_NUM_CLASSES)
        assert net.features(x).ndim == 2


def test_logits_depend_on_input() -> None:
    """Distinct inputs must yield distinct logits -- the regression guard for exact-chance.

    A network that collapses to an input-independent (dead / constant) feature map produces the
    SAME logits for every sample, so training can only fit the uniform class prior and eval pins
    at exactly 1/num_classes (chance). This is precisely how the un-normalised ReLU-only variant
    of this ResNet failed. Asserting that two different batches produce different logits (and that
    rows within one random batch are not all identical) catches that collapse -- the plain
    shape/finiteness checks above pass even on a constant network.
    """
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    gen = torch.Generator().manual_seed(7)
    x_a = torch.randn(_BATCH, 2, DEFAULT_WINDOW, generator=gen)
    x_b = torch.randn(_BATCH, 2, DEFAULT_WINDOW, generator=gen)
    with torch.no_grad():
        logits_a = net(x_a)
        logits_b = net(x_b)
    # Different inputs -> different outputs (not a constant map).
    assert not torch.allclose(logits_a, logits_b), "logits are input-independent (collapsed net)"
    # And within one batch the per-sample rows are not all identical either.
    assert not torch.allclose(
        logits_a, logits_a[0:1].expand_as(logits_a)
    ), "all rows identical -> feature map collapsed to a constant"


# --- Paper-exact fidelity (BIBLIOGRAPHY.md §B.3) -------------------------------------------------


def test_uses_six_residual_stacks() -> None:
    """O'Shea et al.'s L = 6 residual stacks (was 4); six halving pools take len-128 -> 2."""
    assert DEFAULT_NUM_STACKS == 6
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    assert len(net.stacks) == 6
    # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 pooled samples, x conv_filters channels.
    assert net.flat_dim == net.conv_filters * 2


def test_unit_variance_normalize_standardises_each_window() -> None:
    """Each (2, L) window is standardised to ~zero mean / unit variance over both channels."""
    gen = torch.Generator().manual_seed(3)
    # Per-sample offset + scale: normalization must remove both, per window independently.
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW, generator=gen) * 7.0 + 3.0
    normed = _unit_variance_normalize(x)
    mean = normed.mean(dim=(1, 2))
    std = normed.std(dim=(1, 2), unbiased=False)
    assert torch.allclose(mean, torch.zeros(_BATCH), atol=1e-5)
    assert torch.allclose(std, torch.ones(_BATCH), atol=1e-4)


def test_unit_variance_normalize_survives_constant_window() -> None:
    """A degenerate all-constant (zero-variance) window is finite, not NaN (the eps guard)."""
    x = torch.full((1, 2, DEFAULT_WINDOW), 5.0)
    normed = _unit_variance_normalize(x)
    assert torch.isfinite(normed).all()


def test_input_normalization_makes_forward_scale_invariant() -> None:
    """Per-window unit-variance norm -> scaling an input window barely moves the logits.

    O'Shea et al.'s preprocessing removes the absolute capture scale (which carries no
    modulation information). With eval-mode BatchNorm + the input norm, multiplying a window by a
    positive constant is (up to the small BatchNorm eps) a no-op on the class scores.
    """
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW, generator=gen)
    with torch.no_grad():
        logits = net(x)
        logits_scaled = net(x * 5.0)
    assert torch.allclose(logits, logits_scaled, atol=1e-4)


def test_head_has_alpha_dropout_and_two_denses() -> None:
    """The head is SELU Dense -> SELU Dense with AlphaDropout (not a single dense)."""
    assert 0.0 <= DEFAULT_ALPHA_DROPOUT < 1.0
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    denses = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    # fc_embed dense + fc_head dense + classifier = 3 Linear layers total.
    assert len(denses) == 3
    alpha_drops = [m for m in net.modules() if isinstance(m, torch.nn.AlphaDropout)]
    assert len(alpha_drops) == 2
    selus = [m for m in net.modules() if isinstance(m, torch.nn.SELU)]
    assert len(selus) == 2


def test_alpha_dropout_is_noop_in_eval() -> None:
    """AlphaDropout is disabled in eval() -> forward is deterministic for the metrics."""
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    net.eval()
    gen = torch.Generator().manual_seed(19)
    x = torch.randn(_BATCH, 2, DEFAULT_WINDOW, generator=gen)
    with torch.no_grad():
        first = net(x)
        second = net(x)
    assert torch.allclose(first, second)


def test_batchnorm_retained() -> None:
    """The load-bearing per-layer BatchNorm (fixed the chance-level collapse) is kept."""
    net = ResNetAMCNet(DEFAULT_NUM_CLASSES, window=DEFAULT_WINDOW)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    # Per stack: 1 channel-mixing BN + 2 per residual unit x 2 units = 5 BN; x 6 stacks = 30.
    assert len(bns) == 6 * 5
