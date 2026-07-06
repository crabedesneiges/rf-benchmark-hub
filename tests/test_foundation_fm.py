"""WP-62 acceptance tests for the LWM-Spectro public-weights FM wrapper.

Two layers, so the suite is meaningful in BOTH the dep-free ``.venv`` and the cluster ARM GPU
venv:

* **dep-free assertions** (always run): importing
  :mod:`rfbench.models.foundation.lwm_spectro` stays dependency-free and REGISTERS
  ``"lwm-spectro"`` in :data:`rfbench.core.registry.MODELS`; the download helper resolves cache
  paths without importing ``huggingface_hub`` or hitting the network. These need only ``pytest``.
* **torch assertions** (``pytest.importorskip("torch")``): the wrapper satisfies the
  :class:`~rfbench.core.model.Model` contract on a SYNTHETIC ``(2, 128)`` IQ batch --
  ``embed`` -> ``(B, 128)`` features, ``forward`` -> ``(B, 11)`` logits, ``n_params`` a positive
  int -- running on the RANDOMLY-INITIALISED encoder (no weights download in unit tests). This
  proves the adapter + architecture end-to-end; the real pretrained run happens on the cluster.

NEVER downloads weights or touches the network; the real ``checkpoint.pth`` fetch lives in the
guarded :func:`rfbench.models.foundation._download_lwm_spectro.download_lwm_spectro`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from rfbench.core.model import Model
from rfbench.core.registry import MODELS

# Importing the wrapper module registers "lwm-spectro"; it must stay dependency-free.
lwm_spectro = importlib.import_module("rfbench.models.foundation.lwm_spectro")
_download = importlib.import_module("rfbench.models.foundation._download_lwm_spectro")

_NUM_CLASSES = 11
_EMBED_DIM = 128


def _synthetic_iq_batch(batch_size: int = 3) -> dict[str, list]:
    """A collated AMC-shaped batch: ``x["iq"]`` a list of ``(2, 128)`` nested-list windows.

    Distinct per-sample content so embeddings differ; pure Python lists (no numpy), matching the
    dependency-free synthetic path :class:`~rfbench.tasks.amc.dataset.AmcDataset` produces.
    """
    iq: list[list[list[float]]] = []
    for b in range(batch_size):
        i_row = [0.01 * (b + 1) * (t % 7) for t in range(128)]
        q_row = [0.02 * (b + 1) * ((t + 3) % 5) for t in range(128)]
        iq.append([i_row, q_row])
    labels = [b % _NUM_CLASSES for b in range(batch_size)]
    return {"iq": iq, "label": labels, "snr_db": [0] * batch_size}


# --------------------------------------------------------------------------------------------------
# Dependency-free: registration + import hygiene + download-helper path resolution
# --------------------------------------------------------------------------------------------------
def test_lwm_spectro_is_registered() -> None:
    """The wrapper registers under 'lwm-spectro' and resolves to its class."""
    assert "lwm-spectro" in MODELS
    assert MODELS.get("lwm-spectro") is lwm_spectro.LwmSpectroModel


def test_import_stays_dependency_free() -> None:
    """Importing the wrapper pulls in neither torch nor huggingface_hub."""
    import sys

    # The module is import-time dependency-free (heavy deps load lazily inside methods).
    assert "rfbench.models.foundation.lwm_spectro" in sys.modules
    # Constructing it must not import torch either (weights load on first embed/forward).
    model = lwm_spectro.LwmSpectroModel()
    assert model.name == "lwm-spectro"
    assert model.family == "foundation"


def test_construction_is_cheap_and_typed() -> None:
    """A no-arg instance is a foundation Model with a not-yet-loaded (0) param count."""
    model = lwm_spectro.LwmSpectroModel()
    assert isinstance(model, Model)
    assert model.num_classes == _NUM_CLASSES
    assert model.n_params == 0  # loads lazily; still 0 before first embed/forward


def test_download_helper_resolves_cache_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download helper computes ``$RFBENCH_CACHE/lwm-spectro`` paths without heavy imports."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    root = _download.lwm_spectro_dir()
    assert root == tmp_path / "lwm-spectro"
    ckpt = _download.backbone_checkpoint_path()
    assert ckpt == tmp_path / "lwm-spectro" / _download.BACKBONE_CHECKPOINT
    assert _download.HF_REPO_ID == "wi-lab/lwm-spectro"


# --------------------------------------------------------------------------------------------------
# torch-gated: the Model contract on a synthetic batch (random-init encoder, no weights)
# --------------------------------------------------------------------------------------------------
def test_embed_shape_on_synthetic_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() returns one 128-d CLS vector per sample (B, 128) on the random-init encoder."""
    torch = pytest.importorskip("torch")
    # Point the cache at an empty dir so no checkpoint is found -> random init, no download.
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    batch = _synthetic_iq_batch(batch_size=3)
    out = model.embed(batch)
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (3, _EMBED_DIM)


def test_forward_returns_num_classes_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """forward() returns (B, 11) AMC logits from the head over the frozen encoder."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    batch = _synthetic_iq_batch(batch_size=4)
    logits = model.forward(batch)
    assert torch.is_tensor(logits)
    assert tuple(logits.shape) == (4, _NUM_CLASSES)


def test_n_params_positive_after_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """n_params is a positive int once the encoder + head are built."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=2))  # triggers the lazy load
    assert isinstance(model.n_params, int)
    assert model.n_params > 0


def test_missing_checkpoint_flips_pretrained_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no cached checkpoint the wrapper runs but honestly reports pretrained=False."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=1))
    assert model.pretrained is False


def test_embed_is_finite_and_one_row_per_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embed() yields exactly one finite 128-d row per sample (the contract the probe needs).

    We assert the shape/finiteness contract rather than per-sample distinctness: with a
    RANDOMLY-INITIALISED encoder, the mean-pooled representation can collapse across samples.
    Distinctness is a property of the *pretrained* weights (loaded on the cluster), not of this
    dependency-light smoke test.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    out = model.embed(_synthetic_iq_batch(batch_size=3))
    assert tuple(out.shape) == (3, _EMBED_DIM)
    assert bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------------------------------
# torch-gated: architecture + adapter FIDELITY regression guards (WP-62 verification, 2026-07)
# --------------------------------------------------------------------------------------------------
def test_encoder_uses_custom_layernorm_alpha_bias_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconstructed encoder exposes upstream custom-LayerNorm keys (alpha/bias), NOT weight/bias.

    Regression guard for the fatal bug where ``nn.LayerNorm`` (``.weight``/``.bias``) was used
    instead of the repo's custom ``LayerNormalization`` (``.alpha``/``.bias``): the real
    ``checkpoint.pth`` stores ``...norm.alpha``/``...norm.bias`` for all 25 norms, so an
    ``nn.LayerNorm`` reconstruction would silently leave every norm scale at random init.
    """
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = lwm_spectro.LwmSpectroModel(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=1))  # triggers the lazy encoder build
    keys = set(model._encoder.state_dict().keys())
    for present in (
        "embedding.norm.alpha",
        "embedding.norm.bias",
        "layers.0.norm1.alpha",
        "layers.0.norm2.alpha",
        "layers.11.norm2.bias",
        "norm.alpha",
        "norm.bias",
    ):
        assert present in keys, f"missing upstream custom-LayerNorm key {present}"
    for forbidden in ("embedding.norm.weight", "layers.0.norm1.weight", "norm.weight"):
        assert forbidden not in keys, f"nn.LayerNorm-style key {forbidden} must NOT exist"
    # attention / FFN / projection keys present under the upstream names (so real weights load).
    for present in (
        "embedding.proj.weight",
        "embedding.pos_embed.weight",
        "layers.0.enc_self_attn.W_Q.weight",
        "layers.0.enc_self_attn.linear.weight",
        "layers.0.pos_ffn.fc1.weight",
        "linear.weight",
    ):
        assert present in keys, f"missing upstream key {present}"


def test_adapter_produces_1025x32_tokens_with_constant_cls() -> None:
    """The IQ->token adapter yields (B, 1025, 32); row 0 is the constant-0.2 CLS token (upstream).

    Regression guard for the CLS mismatch (upstream uses ``np.full(patch_size, 0.2)``, not zeros)
    and for the ``(1024 patches + 1 CLS)`` sequence-length contract the encoder's positional
    embedding (``Embedding(1025, 128)``) requires.
    """
    torch = pytest.importorskip("torch")
    tokens = lwm_spectro._iq_to_lwm_tokens(_synthetic_iq_batch(batch_size=3)["iq"], torch)
    assert tuple(tokens.shape) == (3, lwm_spectro.MAX_LEN, lwm_spectro.ELEMENT_LENGTH)
    cls_expected = torch.full((3, lwm_spectro.ELEMENT_LENGTH), lwm_spectro.CLS_VALUE)
    assert torch.allclose(tokens[:, 0, :], cls_expected)
    assert bool(torch.isfinite(tokens).all())


def test_missing_encoder_keys_raise_not_silent_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint that does not populate the encoder RAISES (no partly-random 'pretrained' run).

    This is the guard that would have caught the original silent-random-init bug: if the real
    state_dict keys stop matching the reconstruction, loading must fail loudly instead of scoring a
    half-random encoder as if it were the pretrained LWM-Spectro.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    bogus = tmp_path / "bogus_checkpoint.pth"
    torch.save({"totally.unrelated.key": torch.zeros(1)}, bogus)
    model = lwm_spectro.LwmSpectroModel(device="cpu", checkpoint=bogus)
    with pytest.raises(RuntimeError, match="MISSING"):
        model.embed(_synthetic_iq_batch(batch_size=1))
