"""Integration tests for the dedicated SEI training loop (`rfbench.training_sei`).

Torch-gated. Trains an SEI baseline end-to-end on an in-memory, scale-invariant synthetic
fingerprinting task (per-class complex-exponential frequency -- separable by every SEI
architecture AND surviving the per-signal unit-average-power normalisation) and asserts the loop:
computes the head width, learns, and emits a schema-valid ``result.json`` carrying the correct
``split.track`` and both closed-set metrics (rank1 primary + balanced secondary). Also unit-tests
the WiSig class-weight computation and the non-trainable-regime guard.

The shared AMC loop (``rfbench.training``) is NOT touched: this exercises the separate module.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from rfbench.core.model import Regime, RegimeSpec  # noqa: E402
from rfbench.models.baselines.wisig_cnn_paper import WiSigCNNPaper  # noqa: E402
from rfbench.tasks.sei.dataset import SeiDataset  # noqa: E402
from rfbench.tasks.sei.task import SeiTask  # noqa: E402
from rfbench.training_sei import (  # noqa: E402
    _class_weights,
    count_classes,
    train_sei_baseline,
)

_WINDOW = 128
_N_CLASSES = 3


def _signal(cls: int, rng: np.random.Generator) -> np.ndarray:
    """A per-class complex exponential (scale-invariant, unit-power-norm-safe) + light noise."""
    t = np.arange(_WINDOW)
    freq = (cls + 1) / _WINDOW * 6.0
    i = np.cos(2 * math.pi * freq * t)
    q = np.sin(2 * math.pi * freq * t)
    sig = np.stack([i, q], axis=1).astype(np.float32)
    return (sig + 0.15 * rng.standard_normal((_WINDOW, 2))).astype(np.float32)


def _samples(n_per: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for cls in range(_N_CLASSES):
        for _ in range(n_per):
            out.append({"iq": _signal(cls, rng), "label": cls, "meta": {"rx": 0, "day": 0}})
    return out


def _dataset(track: str = "closed_set") -> SeiDataset:
    return SeiDataset(
        "wisig",
        track=track,  # type: ignore[arg-type]
        samples={"train": _samples(50, 0), "val": _samples(12, 1), "test": _samples(12, 2)},
    )


class _InMemorySeiTask(SeiTask):
    """A SeiTask whose datasets() returns a fixed in-memory dataset (so evaluate resolves it)."""

    def __init__(self, track: str, dataset: SeiDataset) -> None:
        super().__init__(track)  # type: ignore[arg-type]
        self._ds = dataset

    def datasets(self) -> list:  # type: ignore[override]
        return [self._ds]


def _train(track: str, model_classes: int = _N_CLASSES) -> dict:
    ds = _dataset(track)
    task = _InMemorySeiTask(track, ds)
    model = WiSigCNNPaper(num_classes=model_classes, window=_WINDOW, device="cpu")
    _model, result = train_sei_baseline(
        task,
        model,
        ds,
        track=track,  # type: ignore[arg-type]
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        num_classes=model_classes,
        epochs=25,
        batch_size=16,
        lr=1e-3,
        l2_lambda=1e-4,
        patience=15,
        seed=42,
        device="cpu",
    )
    return result


def test_count_classes_infers_head_width() -> None:
    """count_classes recovers the number of transmitter classes from the train split."""
    assert count_classes(_dataset()) == _N_CLASSES


def test_trains_and_emits_valid_tracked_result() -> None:
    """The loop learns the separable task and emits a schema-valid closed_set result row."""
    result = _train("closed_set")
    assert result["split"]["track"] == "closed_set"
    assert result["metrics"]["primary"] == "rank1_accuracy"
    values = result["metrics"]["values"]
    assert values["rank1_accuracy"] > 0.6  # well above 1/3 chance -> gradients flow, it learns
    assert "balanced_accuracy" in values  # SECONDARY metric present
    assert result["regime"]["name"] == "from_scratch"  # declared verbatim
    assert result["verification"]["status"] == "self_reported"
    assert result["task"]["name"] == "sei"


@pytest.mark.parametrize("track", ["cross_receiver", "cross_day"])
def test_track_is_threaded_into_the_result(track: str) -> None:
    """The requested track is written verbatim into split.track (separate rows per condition)."""
    assert _train(track)["split"]["track"] == track


def test_class_weights_are_majority_over_count() -> None:
    """_class_weights == max(count)/count per class (WiSig prepare_txid_and_weights semantics)."""
    samples = [
        {"iq": [[0.0, 0.0]], "label": 0, "meta": {}},
        {"iq": [[0.0, 0.0]], "label": 0, "meta": {}},
        {"iq": [[0.0, 0.0]], "label": 0, "meta": {}},
        {"iq": [[0.0, 0.0]], "label": 1, "meta": {}},
    ]
    ds = SeiDataset("wisig", track="closed_set", samples={"train": samples})
    weights = _class_weights(ds.load("train", None), num_classes=2, device="cpu")
    # class 0 count 3, class 1 count 1, max = 3 -> weights [1.0, 3.0].
    assert weights.tolist() == pytest.approx([1.0, 3.0])


def test_non_trainable_regime_rejected() -> None:
    """A probing regime is rejected (the SEI loop fits only from_scratch / full_finetune)."""
    ds = _dataset()
    task = _InMemorySeiTask("closed_set", ds)
    model = WiSigCNNPaper(num_classes=_N_CLASSES, window=_WINDOW, device="cpu")
    with pytest.raises(ValueError, match="fits only"):
        train_sei_baseline(
            task,
            model,
            ds,
            track="closed_set",  # type: ignore[arg-type]
            regime=RegimeSpec(Regime.LINEAR_PROBE),
            num_classes=_N_CLASSES,
            device="cpu",
        )
