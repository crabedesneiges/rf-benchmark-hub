"""Acceptance tests for the SNR-estimation regression baselines (J4).

The ``mean_snr`` floor is pure stdlib, so its whole fit -> forward -> ``evaluate`` -> schema path
runs with only ``pytest`` + ``jsonschema`` installed (the coverage that was missing for the SEI
open-set baselines). ``snr_moment_ridge`` needs numpy + scikit-learn, so its feature/fit tests are
``importorskip``-guarded and only run where those are installed.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Regime, RegimeSpec
from rfbench.core.registry import MODELS
from rfbench.models.baselines.snr_regressors import (
    N_SNR_FEATURES,
    MeanSnr,
    SnrMomentRidge,
)
from rfbench.tasks.snr_estimation import SnrDataset, SnrEstimationTask

_CHECKSUM = "sha256:" + "0" * 64


def _samples(snrs: list[float]) -> list[dict[str, Any]]:
    """Per-sample dicts ``{"iq", "snr_db"}``; iq is an opaque (2, L) list (unused by mean_snr)."""
    return [{"iq": [[0.1, -0.2], [0.3, 0.0]], "snr_db": s} for s in snrs]


# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------
def test_snr_baselines_registered() -> None:
    """Both regression baselines resolve from the registry under their board names."""
    assert MODELS.get("mean_snr") is MeanSnr
    assert MODELS.get("snr_moment_ridge") is SnrMomentRidge


# --------------------------------------------------------------------------------------------------
# mean_snr floor (pure stdlib)
# --------------------------------------------------------------------------------------------------
def test_mean_snr_predicts_train_mean() -> None:
    """``mean_snr`` learns the train-split mean and broadcasts it to every sample."""
    model = MeanSnr()
    assert model.n_params == 0  # before fit
    model.fit(_samples([-4.0, 0.0, 4.0, 8.0]))  # mean = 2.0
    assert model.n_params == 1
    preds = model.forward({"iq": [object(), object(), object()]})
    assert preds == [2.0, 2.0, 2.0]


def test_mean_snr_forward_before_fit_raises() -> None:
    """Calling ``forward`` before ``fit`` is a loud error, not a silent 0.0."""
    with pytest.raises(RuntimeError, match="before fit"):
        MeanSnr().forward({"iq": [object()]})


def test_mean_snr_evaluate_end_to_end_is_the_std_floor() -> None:
    """End-to-end: ``mean_snr`` scored via ``evaluate`` yields RMSE == std / MAE == mean|dev|.

    The in-memory ``SnrDataset`` returns the SAME sample list for train and test, so the fitted
    mean equals the test mean and the floor's RMSE is exactly the SNR standard deviation. Runs the
    real evaluate + Rmse/Mae + result.json emission and validates it against the committed schema
    -- no numpy, no torch (the coverage lesson from the SEI open-set baselines).
    """
    snrs = [-4.0, 0.0, 4.0, 8.0]  # mean 2.0
    dataset = SnrDataset("radioml_2016_10a", samples=_samples(snrs), checksum=_CHECKSUM)
    task = SnrEstimationTask(datasets=[dataset])

    model = MeanSnr()
    model.fit(dataset.load("train"))
    result = evaluate(
        model, task, "test", RegimeSpec(Regime.FROM_SCRATCH), compute_bootstrap_ci=False
    )

    mean = sum(snrs) / len(snrs)
    expected_rmse = math.sqrt(sum((s - mean) ** 2 for s in snrs) / len(snrs))
    expected_mae = sum(abs(s - mean) for s in snrs) / len(snrs)
    values = result["metrics"]["values"]
    assert result["metrics"]["primary"] == "rmse_db"
    assert values["rmse_db"] == pytest.approx(expected_rmse)
    assert values["mae_db"] == pytest.approx(expected_mae)
    assert result["model"]["name"] == "mean_snr"
    assert result["regime"]["name"] == "from_scratch"

    schema = json.loads(_resolve_schema_path("result.schema.json").read_text(encoding="utf-8"))
    from jsonschema import Draft202012Validator

    Draft202012Validator(schema).validate(result)  # raises if the emitted SNR row is invalid


# --------------------------------------------------------------------------------------------------
# snr_moment_ridge (numpy + scikit-learn)
# --------------------------------------------------------------------------------------------------
def test_snr_features_shape_and_scale_invariance() -> None:
    """``_snr_features`` returns N features and is (approximately) scale-invariant."""
    np = pytest.importorskip("numpy")
    from rfbench.models.baselines.snr_regressors import _snr_features

    rng = np.random.default_rng(0)
    window = rng.standard_normal((2, 128))
    feats = _snr_features(window)
    assert len(feats) == N_SNR_FEATURES
    # Scaling the window by a constant leaves the scale-invariant ratio features unchanged.
    scaled = _snr_features(window * 7.0)
    assert scaled == pytest.approx(feats, rel=1e-9, abs=1e-9)


def test_snr_moment_ridge_fits_and_beats_the_mean_floor() -> None:
    """``snr_moment_ridge`` fits on windows whose noise level encodes SNR and beats the floor.

    Synthetic: a fixed tone plus Gaussian noise whose amplitude decreases with the (known) SNR,
    so the envelope kurtosis/moment features carry the signal. The ridge's test RMSE must be
    strictly below the constant-mean floor's.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(42)
    length = 128
    t = np.arange(length)
    tone = np.exp(2j * np.pi * 0.05 * t)  # unit-power complex tone

    samples = []
    snrs = []
    for _ in range(240):
        snr_db = float(rng.uniform(-20.0, 18.0))
        noise_std = 10.0 ** (-snr_db / 20.0)  # higher SNR -> less noise
        noise = (rng.standard_normal(length) + 1j * rng.standard_normal(length)) * (
            noise_std / math.sqrt(2)
        )
        z = tone + noise
        iq = np.stack([z.real, z.imag]).astype(np.float64)  # (2, L)
        samples.append({"iq": iq, "snr_db": snr_db})
        snrs.append(snr_db)

    split = int(0.75 * len(samples))
    train, test = samples[:split], samples[split:]

    ridge = SnrMomentRidge().fit(train)
    assert ridge.n_params == N_SNR_FEATURES + 1  # coefficients + intercept
    preds = ridge.forward({"iq": np.stack([s["iq"] for s in test])})
    assert len(preds) == len(test)

    test_snrs = [s["snr_db"] for s in test]
    ridge_rmse = math.sqrt(
        sum((p - t) ** 2 for p, t in zip(preds, test_snrs, strict=True)) / len(test)
    )
    floor = MeanSnr().fit(train)
    floor_pred = floor.forward({"iq": np.stack([s["iq"] for s in test])})
    floor_rmse = math.sqrt(
        sum((p - t) ** 2 for p, t in zip(floor_pred, test_snrs, strict=True)) / len(test)
    )
    assert ridge_rmse < floor_rmse  # the DSP regressor beats the predict-the-mean floor
