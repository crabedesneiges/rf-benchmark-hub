"""Acceptance tests for the IQFM raw-IQ SSL foundation-model wrapper.

Two layers, meaningful in BOTH the dep-free ``.venv`` and the cluster ARM GPU venv:

* **dep-free assertions** (always run): importing ``rfbench.models.foundation`` REGISTERS
  ``"iqfm-base"`` in :data:`rfbench.core.registry.MODELS` **without pulling torch**, the wrapper
  constructs cheaply (0 params before first use), and the checkpoint-path helper resolves the
  ``$RFBENCH_CACHE`` cache without heavy imports. These need only ``pytest``.
* **torch assertions** (``pytest.importorskip("torch")``): the ShuffleNetV2-x0.5 backbone builds
  to ~341k params, ``embed`` returns one 1024-D vector per sample on a synthetic ``(2, 128)`` IQ
  batch, unit-max normalisation is applied, a missing checkpoint honestly flips ``pretrained``,
  and a non-matching checkpoint RAISES. No weights are ever downloaded.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from rfbench.core.model import Model
from rfbench.core.registry import MODELS

# Importing the PACKAGE must register "iqfm-base" (re-exported in the package __init__) and stay
# dependency-free — this import failing without torch would break the whole acceptance contract.
_foundation = importlib.import_module("rfbench.models.foundation")
_iqfm = importlib.import_module("rfbench.models.foundation.iqfm")

_EMBED_DIM = 1024
_APPROX_PARAMS = 341_000


def _synthetic_iq_batch(batch_size: int = 3, length: int = 128) -> dict[str, list]:
    """A collated AMC-shaped batch: ``x["iq"]`` a list of ``(2, length)`` nested-list windows.

    Distinct per-sample content (pure Python lists, no numpy) matching the dependency-free
    synthetic path :class:`~rfbench.tasks.amc.dataset.AmcDataset` produces.
    """
    iq: list[list[list[float]]] = []
    for b in range(batch_size):
        i_row = [0.01 * (b + 1) * (t % 7) for t in range(length)]
        q_row = [0.02 * (b + 1) * ((t + 3) % 5) for t in range(length)]
        iq.append([i_row, q_row])
    return {"iq": iq, "label": [b % 11 for b in range(batch_size)], "snr_db": [0] * batch_size}


# --------------------------------------------------------------------------------------------------
# Dependency-free: registration + import hygiene + cache-path resolution
# --------------------------------------------------------------------------------------------------
def test_iqfm_registered_by_package_import() -> None:
    """Importing the package registers 'iqfm-base' -> IqfmBase (so @register_model fired)."""
    assert "iqfm-base" in MODELS
    assert MODELS.get("iqfm-base") is _iqfm.IqfmBase


def test_package_import_is_dependency_free() -> None:
    """Importing the foundation package + constructing IQFM pulls in neither torch nor numpy."""
    # torch may be installed in this env; the point is that construction must not REQUIRE it.
    model = MODELS.get("iqfm-base")()
    assert model.name == "iqfm-base"
    assert model.family == "foundation"


def test_construction_is_cheap_and_typed() -> None:
    """A no-arg instance is a foundation Model with a not-yet-loaded (0) param count."""
    model = _iqfm.IqfmBase()
    assert isinstance(model, Model)
    assert model.n_params == 0  # loads lazily; still 0 before first embed
    assert model.pretrained is True  # honest default; flips to False if no checkpoint is found


def test_checkpoint_path_helpers_resolve_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache-path helpers compute ``$RFBENCH_CACHE/iqfm`` paths without heavy imports."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    assert _iqfm.iqfm_cache_dir() == tmp_path / "iqfm"
    assert _iqfm.backbone_checkpoint_path() == tmp_path / "iqfm" / _iqfm.BACKBONE_CHECKPOINT


# --------------------------------------------------------------------------------------------------
# torch-gated: the reusable ShuffleNetV2-x0.5 1-D backbone
# --------------------------------------------------------------------------------------------------
def test_backbone_param_count_matches_iqfm_encoder() -> None:
    """ShuffleNetV2-x0.5 1-D backbone is ~341k params (the figure IQFM reports for its encoder)."""
    pytest.importorskip("torch")
    from rfbench.models.foundation.shufflenet1d import build_shufflenet1d

    backbone = build_shufflenet1d()
    n_params = sum(p.numel() for p in backbone.parameters())
    assert 300_000 <= n_params <= 400_000, f"expected ~{_APPROX_PARAMS} params, got {n_params}"


def test_backbone_pools_to_one_vector_per_sample() -> None:
    """The backbone maps (B, 2, L) -> (B, 1024): one mean-pooled embedding per sample."""
    torch = pytest.importorskip("torch")
    from rfbench.models.foundation.shufflenet1d import build_shufflenet1d, embed_dim

    backbone = build_shufflenet1d().eval()
    x = torch.randn(4, 2, 128)
    with torch.no_grad():
        out = backbone(x)
    assert tuple(out.shape) == (4, embed_dim())
    assert bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------------------------------
# torch-gated: the IQFM Model contract on a synthetic batch (random-init backbone, no weights)
# --------------------------------------------------------------------------------------------------
def test_embed_shape_on_synthetic_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() returns one 1024-D vector per sample (B, 1024) on the random-init backbone."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))  # empty cache -> random init, no download
    model = _iqfm.IqfmBase(device="cpu")
    out = model.embed(_synthetic_iq_batch(batch_size=3))
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (3, _EMBED_DIM)
    assert bool(torch.isfinite(out).all())


def test_n_params_positive_after_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """n_params is a positive ~341k int once the backbone is built."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = _iqfm.IqfmBase(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=2))  # triggers the lazy load
    assert isinstance(model.n_params, int)
    assert 300_000 <= model.n_params <= 400_000


def test_unit_max_normalisation_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each window is scaled by max complex magnitude -> post-norm max |iq| == 1 (unit-max)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    iq = _synthetic_iq_batch(batch_size=2)["iq"]
    batch = _iqfm._iq_batch_to_tensor(iq, torch)  # (2, 2, 128), unit-max normalised
    assert tuple(batch.shape) == (2, 2, 128)
    mag = torch.sqrt(batch[:, 0, :] ** 2 + batch[:, 1, :] ** 2)  # (B, L)
    peak = mag.amax(dim=1)  # per-sample peak magnitude
    assert torch.allclose(peak, torch.ones_like(peak), atol=1e-5)


def test_missing_checkpoint_flips_pretrained_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no cached checkpoint the wrapper runs but honestly reports pretrained=False."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = _iqfm.IqfmBase(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=1))
    assert model.pretrained is False


def test_non_matching_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint that does not populate the backbone RAISES (no partly-random 'pretrained')."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    bogus = tmp_path / "bogus.pth"
    torch.save({"totally.unrelated.key": torch.zeros(1)}, bogus)
    model = _iqfm.IqfmBase(device="cpu", checkpoint=bogus)
    with pytest.raises(RuntimeError, match="MISSING"):
        model.embed(_synthetic_iq_batch(batch_size=1))


def test_roundtrip_checkpoint_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A state_dict from a fresh backbone loads back with no missing keys (pretrained stays)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    from rfbench.models.foundation.shufflenet1d import build_shufflenet1d

    ckpt = tmp_path / "good.pth"
    torch.save(build_shufflenet1d().state_dict(), ckpt)
    model = _iqfm.IqfmBase(device="cpu", checkpoint=ckpt)
    out = model.embed(_synthetic_iq_batch(batch_size=2))
    assert tuple(out.shape) == (2, _EMBED_DIM)
    assert model.pretrained is True  # matching checkpoint -> stays pretrained
