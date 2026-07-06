"""Acceptance tests for the WirelessJEPA raw-IQ JEPA foundation-model wrapper.

Two layers, meaningful in BOTH the dep-free ``.venv`` and the cluster ARM GPU venv:

* **dep-free assertions** (always run): importing ``rfbench.models.foundation`` REGISTERS
  ``"wireless-jepa"`` in :data:`rfbench.core.registry.MODELS` **without pulling torch**, the wrapper
  constructs cheaply (0 params before first use), and the checkpoint-path helper resolves the
  ``$RFBENCH_CACHE`` cache. These need only ``pytest``.
* **torch assertions** (``pytest.importorskip("torch")``): the shared ShuffleNetV2-x0.5 backbone
  yields one 1024-D vector per sample on a synthetic ``(2, 128)`` IQ batch, unit-max normalisation
  is applied, a missing checkpoint honestly flips ``pretrained``, a non-matching checkpoint RAISES,
  and — the defining property — WirelessJEPA and IQFM share the SAME backbone architecture (a
  round-trip of one FM's backbone state_dict loads into the other). No weights are downloaded.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from rfbench.core.model import Model
from rfbench.core.registry import MODELS

# Importing the PACKAGE must register "wireless-jepa" (re-exported in the package __init__) and stay
# dependency-free — importing without torch must not fail.
_foundation = importlib.import_module("rfbench.models.foundation")
_wjepa = importlib.import_module("rfbench.models.foundation.wireless_jepa")

_EMBED_DIM = 1024


def _synthetic_iq_batch(batch_size: int = 3, length: int = 128) -> dict[str, list]:
    """A collated AMC-shaped batch: ``x["iq"]`` a list of ``(2, length)`` nested-list windows."""
    iq: list[list[list[float]]] = []
    for b in range(batch_size):
        i_row = [0.01 * (b + 1) * (t % 7) for t in range(length)]
        q_row = [0.02 * (b + 1) * ((t + 3) % 5) for t in range(length)]
        iq.append([i_row, q_row])
    return {"iq": iq, "label": [b % 11 for b in range(batch_size)], "snr_db": [0] * batch_size}


# --------------------------------------------------------------------------------------------------
# Dependency-free: registration + import hygiene + cache-path resolution
# --------------------------------------------------------------------------------------------------
def test_wireless_jepa_registered_by_package_import() -> None:
    """Importing the package registers 'wireless-jepa' -> WirelessJepa (@register_model fired)."""
    assert "wireless-jepa" in MODELS
    assert MODELS.get("wireless-jepa") is _wjepa.WirelessJepa


def test_package_import_is_dependency_free() -> None:
    """Constructing WirelessJEPA via the registry pulls in neither torch nor numpy."""
    model = MODELS.get("wireless-jepa")()
    assert model.name == "wireless-jepa"
    assert model.family == "foundation"


def test_construction_is_cheap_and_typed() -> None:
    """A no-arg instance is a foundation Model with a not-yet-loaded (0) param count."""
    model = _wjepa.WirelessJepa()
    assert isinstance(model, Model)
    assert model.n_params == 0  # loads lazily; still 0 before first embed
    assert model.pretrained is True  # honest default; flips to False if no checkpoint is found


def test_checkpoint_path_helpers_resolve_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache-path helpers compute ``$RFBENCH_CACHE/wireless-jepa`` paths (no heavy imports)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    assert _wjepa.wireless_jepa_cache_dir() == tmp_path / "wireless-jepa"
    expected = tmp_path / "wireless-jepa" / _wjepa.BACKBONE_CHECKPOINT
    assert _wjepa.backbone_checkpoint_path() == expected


# --------------------------------------------------------------------------------------------------
# torch-gated: the Model contract on a synthetic batch (random-init backbone, no weights)
# --------------------------------------------------------------------------------------------------
def test_embed_shape_on_synthetic_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() returns one 1024-D vector per sample (B, 1024) on the random-init backbone."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))  # empty cache -> random init, no download
    model = _wjepa.WirelessJepa(device="cpu")
    out = model.embed(_synthetic_iq_batch(batch_size=3))
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (3, _EMBED_DIM)
    assert bool(torch.isfinite(out).all())


def test_n_params_matches_shared_backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """n_params equals the shared ShuffleNetV2-x0.5 backbone param count (~335k)."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = _wjepa.WirelessJepa(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=2))  # triggers the lazy load
    assert 300_000 <= model.n_params <= 400_000


def test_unit_max_normalisation_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each window is scaled by max complex magnitude -> post-norm max |iq| == 1 (unit-max)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    batch = _wjepa._iq_batch_to_tensor(_synthetic_iq_batch(batch_size=2)["iq"], torch)
    assert tuple(batch.shape) == (2, 2, 128)
    peak = torch.sqrt(batch[:, 0, :] ** 2 + batch[:, 1, :] ** 2).amax(dim=1)
    assert torch.allclose(peak, torch.ones_like(peak), atol=1e-5)


def test_missing_checkpoint_flips_pretrained_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no cached checkpoint the wrapper runs but honestly reports pretrained=False."""
    pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    model = _wjepa.WirelessJepa(device="cpu")
    model.embed(_synthetic_iq_batch(batch_size=1))
    assert model.pretrained is False


def test_non_matching_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint that does not populate the backbone RAISES (no partly-random 'pretrained')."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    bogus = tmp_path / "bogus.pth"
    torch.save({"totally.unrelated.key": torch.zeros(1)}, bogus)
    model = _wjepa.WirelessJepa(device="cpu", checkpoint=bogus)
    with pytest.raises(RuntimeError, match="MISSING"):
        model.embed(_synthetic_iq_batch(batch_size=1))


def test_shares_iqfm_backbone_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WirelessJEPA and IQFM share the SAME backbone: an IQFM backbone state_dict loads into WJEPA.

    This is the "matched to IQFM" contract — the two FMs must be architecturally identical (same
    :func:`build_shufflenet1d`), so a checkpoint saved from one loads cleanly into the other.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    from rfbench.models.foundation.shufflenet1d import build_shufflenet1d

    ckpt = tmp_path / "shared_backbone.pth"
    torch.save({"target_encoder_state_dict": build_shufflenet1d().state_dict()}, ckpt)
    model = _wjepa.WirelessJepa(device="cpu", checkpoint=ckpt)
    out = model.embed(_synthetic_iq_batch(batch_size=2))
    assert tuple(out.shape) == (2, _EMBED_DIM)
    assert model.pretrained is True  # matching checkpoint -> stays pretrained
