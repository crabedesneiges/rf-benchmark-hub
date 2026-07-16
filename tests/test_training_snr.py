"""End-to-end tests for the SNR-estimation regression training loop (`rfbench.training_snr`).

Torch-gated. Trains the ``snr_cnn`` baseline end-to-end on a tiny in-memory synthetic SNR task
(a fixed tone plus Gaussian noise whose level encodes the known SNR -- learnable by a small CNN)
on the CPU and asserts the loop: fits, then emits a schema-valid ``result.json`` whose primary
metric is ``rmse_db`` with a finite value. This is the coverage the repo demands for a new
task/loop (eval bugs are invisible to unit tests). Also unit-tests the model contract + the
non-trainable-regime guard.

The AMC/SEI classification loops (``rfbench.training`` / ``rfbench.training_sei``) are NOT touched:
this exercises the separate regression module.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from rfbench.core.evaluate import _resolve_schema_path  # noqa: E402
from rfbench.core.model import Model, Regime, RegimeSpec  # noqa: E402
from rfbench.core.registry import MODELS  # noqa: E402
from rfbench.models.baselines.snr_cnn import SnrCNN, SnrCNNNet  # noqa: E402
from rfbench.tasks.snr_estimation import SnrDataset, SnrEstimationTask  # noqa: E402
from rfbench.training_snr import train_snr_regressor  # noqa: E402

_WINDOW = 64
_CHECKSUM = "sha256:" + "0" * 64


def _signal(snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """A fixed unit-power tone + Gaussian noise whose level encodes ``snr_db`` -> (2, window)."""
    t = np.arange(_WINDOW)
    z = np.exp(2j * math.pi * 0.05 * t)  # unit-power complex tone
    noise_std = 10.0 ** (-snr_db / 20.0)  # higher SNR -> less noise
    noise = (rng.standard_normal(_WINDOW) + 1j * rng.standard_normal(_WINDOW)) * (
        noise_std / math.sqrt(2)
    )
    z = z + noise
    return np.stack([z.real, z.imag]).astype(np.float32)  # (2, window) channel-first


def _samples(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for _ in range(n):
        snr_db = float(rng.uniform(-20.0, 18.0))
        out.append({"iq": _signal(snr_db, rng), "snr_db": snr_db})
    return out


def _dataset() -> SnrDataset:
    return SnrDataset(
        "radioml_2016_10a",
        samples=_samples(48, 0),
        checksum=_CHECKSUM,
    )


class _InMemorySnrTask(SnrEstimationTask):
    """A SnrEstimationTask bound to a fixed in-memory dataset (so evaluate resolves it)."""

    def __init__(self, dataset: SnrDataset) -> None:
        super().__init__(datasets=[dataset])


def test_registered_and_contract() -> None:
    """Registered under 'snr_cnn'; a baseline Model; forward returns one scalar per sample."""
    assert MODELS.get("snr_cnn") is SnrCNN
    model = SnrCNN(device="cpu", window=_WINDOW)
    assert isinstance(model, Model)
    assert model.family == "baseline"
    iq = {"iq": [s["iq"] for s in _samples(4, 1)]}
    out = model.forward(iq)
    assert out.shape == (4,)  # (B,) predicted SNR (dB) -- one scalar per window
    assert model.embed(iq).shape == (4, model.net.embed_dim)


def test_short_window_raises() -> None:
    """A window below the deep stack's minimum raises clearly."""
    with pytest.raises(ValueError, match="window must be >="):
        SnrCNNNet(window=4)


def test_trains_and_emits_valid_rmse_result() -> None:
    """The loop fits and emits a schema-valid result row whose primary is a finite rmse_db."""
    ds = _dataset()
    task = _InMemorySnrTask(ds)
    model = SnrCNN(device="cpu", window=_WINDOW)

    _model, result = train_snr_regressor(
        task,
        model,
        ds,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=2,
        batch_size=16,
        lr=1e-3,
        patience=5,
        seed=42,
        device="cpu",
        compute_bootstrap_ci=False,
    )

    assert result["task"]["name"] == "snr_estimation"
    assert result["model"]["name"] == "snr_cnn"
    assert result["regime"]["name"] == "from_scratch"  # declared verbatim
    assert result["verification"]["status"] == "self_reported"
    assert result["split"]["track"] == "all_snr"  # single full-SNR-range track

    assert result["metrics"]["primary"] == "rmse_db"
    values = result["metrics"]["values"]
    rmse = values["rmse_db"]
    assert isinstance(rmse, float) and math.isfinite(rmse) and rmse >= 0.0
    assert "mae_db" in values  # SECONDARY metric present

    # The emitted row MUST validate against the committed schema (the eval-bug coverage lesson).
    schema = json.loads(_resolve_schema_path("result.schema.json").read_text(encoding="utf-8"))
    from jsonschema import Draft202012Validator

    Draft202012Validator(schema).validate(result)


def test_non_trainable_regime_rejected() -> None:
    """A probing regime is rejected (the SNR loop fits only from_scratch / full_finetune)."""
    ds = _dataset()
    task = _InMemorySnrTask(ds)
    model = SnrCNN(device="cpu", window=_WINDOW)
    with pytest.raises(ValueError, match="fits only"):
        train_snr_regressor(
            task,
            model,
            ds,
            regime=RegimeSpec(Regime.LINEAR_PROBE),
            device="cpu",
        )
