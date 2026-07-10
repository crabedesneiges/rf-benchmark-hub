"""Acceptance tests for the SNR-estimation (raw-IQ -> SNR in dB regression) task adapter (J4).

Everything here runs on PURE-PYTHON synthetic predictions/targets -- no numpy, no torch, no
network -- so the suite passes with only ``pytest`` + ``jsonschema`` installed:

* the two SNR regression metrics (:class:`Rmse`, :class:`Mae`) computed on fixtures with
  hand-checked expected values (RMSE, MAE);
* the task registers under ``"snr_estimation"`` and exposes the protocol datasets/metrics/split;
* the canonical SNR split id mirrors the AMC split's indices (byte-identical checksum) while
  keeping its own id;
* an end-to-end :func:`rfbench.core.evaluate.evaluate` over a synthetic in-memory
  :class:`SnrDataset` assembles a ``result.json`` whose regression metrics are correct.

SCHEMA ENUM. ``schemas/result.schema.json`` now includes ``"snr_estimation"`` in the (frozen,
additively extended) ``task.name`` enum, so the produced row validates end-to-end against the
committed schema with no patching.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import TASKS, get_task
from rfbench.core.types import Batch, Tensor
from rfbench.tasks.snr_estimation import Mae, Rmse, SnrDataset, SnrEstimationTask

# --------------------------------------------------------------------------------------------------
# Hand-checked fixtures (6 samples; errors chosen so RMSE != MAE)
# --------------------------------------------------------------------------------------------------
# true SNRs (dB):  [-20, -10, 0, 4, 10, 18]
# predicted SNRs:  [-18, -12, 0, 6,  9, 18]  -> errors: [+2, -2, 0, +2, -1, 0]
_TARGETS = [-20.0, -10.0, 0.0, 4.0, 10.0, 18.0]
_PREDS = [-18.0, -12.0, 0.0, 6.0, 9.0, 18.0]

# |errors| = [2, 2, 0, 2, 1, 0]  -> MAE = 7/6
_EXPECTED_MAE = 7 / 6
# errors^2 = [4, 4, 0, 4, 1, 0] -> mean = 13/6 -> RMSE = sqrt(13/6)
_EXPECTED_RMSE = math.sqrt(13 / 6)


# --------------------------------------------------------------------------------------------------
# Metric compute on synthetic data (known expected values)
# --------------------------------------------------------------------------------------------------
def test_rmse_known_value() -> None:
    """``rmse_db`` matches the hand-computed sqrt(13/6) over the whole split."""
    metric = Rmse()
    metric.update(_PREDS, _TARGETS)
    computed = metric.compute()
    assert metric.primary_key == "rmse_db"
    assert computed["rmse_db"] == pytest.approx(_EXPECTED_RMSE)


def test_mae_known_value() -> None:
    """``mae_db`` matches the hand-computed 7/6 over the whole split."""
    metric = Mae()
    metric.update(_PREDS, _TARGETS)
    computed = metric.compute()
    assert metric.primary_key == "mae_db"
    assert computed["mae_db"] == pytest.approx(_EXPECTED_MAE)


def test_rmse_ge_mae() -> None:
    """RMSE >= MAE always (it penalises the large errors more) -- true on this fixture."""
    assert _EXPECTED_RMSE > _EXPECTED_MAE


def test_perfect_prediction_is_zero() -> None:
    """A perfect estimator scores 0 dB on both RMSE and MAE (lower is better)."""
    assert Rmse().compute()["rmse_db"] == 0.0  # empty stream -> 0.0
    rmse = Rmse()
    rmse.update(_TARGETS, _TARGETS)
    mae = Mae()
    mae.update(_TARGETS, _TARGETS)
    assert rmse.compute()["rmse_db"] == pytest.approx(0.0)
    assert mae.compute()["mae_db"] == pytest.approx(0.0)


def test_metrics_are_streaming_across_batches() -> None:
    """Splitting the batch in two and updating twice yields the same result as one update."""
    whole = Rmse()
    whole.update(_PREDS, _TARGETS)

    streamed = Rmse()
    streamed.update(_PREDS[:3], _TARGETS[:3])
    streamed.update(_PREDS[3:], _TARGETS[3:])

    assert streamed.compute()["rmse_db"] == pytest.approx(whole.compute()["rmse_db"])


def test_reset_clears_state() -> None:
    """After :meth:`reset` the metric behaves as freshly constructed."""
    metric = Mae()
    metric.update(_PREDS, _TARGETS)
    metric.reset()
    metric.update([1.0, 3.0], [1.0, 1.0])  # abs errors [0, 2] -> MAE 1.0
    assert metric.compute()["mae_db"] == pytest.approx(1.0)


def test_scalar_and_vector_and_tensor_predictions_decode() -> None:
    """A per-sample prediction may be a scalar, a length-1 list, or a 0-d tensor-like."""

    class _ScalarTensor:
        """Duck-types a 0-d tensor: ``.item()`` returns the wrapped float."""

        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    metric = Mae()
    # preds: scalar 1.0, length-1 vector [3.0], 0-d tensor 5.0 ; targets 1,1,1 -> |err| 0,2,4
    metric.update([1.0, [3.0], _ScalarTensor(5.0)], [1.0, 1.0, 1.0])
    assert metric.compute()["mae_db"] == pytest.approx(6 / 3)


def test_multi_element_prediction_rejected() -> None:
    """A multi-element per-sample prediction is a caller bug and raises (not silently reduced)."""
    metric = Rmse()
    with pytest.raises(ValueError, match="one scalar per sample"):
        metric.update([[1.0, 2.0]], [0.0])


def test_rmse_declares_full_snr_range_conditions() -> None:
    """The primary metric records the full SNR range so the row attests no cherry-picking."""
    conditions = Rmse().eval_conditions()
    assert conditions["full_snr_range"] is True
    assert conditions["snr_db_min"] == -20
    assert conditions["snr_db_max"] == 18


def test_mae_has_no_eval_conditions() -> None:
    """Only the primary metric carries eval_conditions; MAE stays condition-free."""
    assert not hasattr(Mae(), "eval_conditions")


# --------------------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------------------
def test_task_registered_under_snr_estimation() -> None:
    """``SnrEstimationTask`` resolves by name through the registry."""
    assert "snr_estimation" in TASKS
    assert TASKS.get("snr_estimation") is SnrEstimationTask
    assert isinstance(get_task("snr_estimation"), SnrEstimationTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metrics/split/tracks match EVALUATION_PROTOCOL.md (regression metric)."""
    task = SnrEstimationTask()
    assert task.name == "snr_estimation"
    assert task.version == "v1"
    dataset_names = [ds.name for ds in task.datasets()]
    assert dataset_names == ["radioml_2016_10a"]
    assert task.default_split() == "test"
    assert task.tracks() == ["all_snr"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "rmse_db"  # primary is first
    assert metric_keys == ["rmse_db", "mae_db"]


def test_canonical_split_id_version_matches_task_version() -> None:
    """The dataset's canonical_split_id -v<N> suffix must equal the task version."""
    task = SnrEstimationTask()
    dataset = task.datasets()[0]
    assert dataset.canonical_split_id == "snr-radioml2016-strat-snr-8010-seed42-v1"
    assert dataset.canonical_split_id.endswith(f"-{task.version}")


def test_build_targets_extracts_snr_db() -> None:
    """``build_targets`` returns the per-sample SNR (dB), NOT a class label."""
    batch: Batch = {"iq": [[0.0], [0.0]], "snr_db": [-4.0, 12.0]}
    assert SnrEstimationTask().build_targets(batch) == [-4.0, 12.0]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing an :class:`SnrDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown SNR-estimation dataset"):
        SnrDataset("not_a_dataset")


# --------------------------------------------------------------------------------------------------
# Canonical split: mirrors the AMC indices (byte-identical) under a distinct id
# --------------------------------------------------------------------------------------------------
def test_snr_split_mirrors_amc_indices_and_checksum() -> None:
    """The committed SNR split has AMC's exact indices + checksum but its own id."""
    splits_dir = Path(__file__).resolve().parents[1] / "leaderboard" / "splits" / "radioml_2016_10a"
    amc = json.loads(
        (splits_dir / "amc-radioml2016-strat-snr-8010-seed42-v1.idx.json").read_text("utf-8")
    )
    snr = json.loads(
        (splits_dir / "snr-radioml2016-strat-snr-8010-seed42-v1.idx.json").read_text("utf-8")
    )
    assert snr["canonical_split_id"] == "snr-radioml2016-strat-snr-8010-seed42-v1"
    assert snr["canonical_split_id"] != amc["canonical_split_id"]
    # Indices identical by construction -> the two boards score the SAME held-out signals.
    assert snr["indices"] == amc["indices"]
    assert snr["checksum"] == amc["checksum"]


def test_snr_dataset_reads_committed_checksum() -> None:
    """A disk-backed :class:`SnrDataset` reports the committed split checksum (PR-ready row)."""
    dataset = SnrDataset("radioml_2016_10a")
    # Matches the committed manifest's split_checksum (== the AMC one).
    assert dataset.checksum.startswith("sha256:")
    assert dataset.checksum != "sha256:" + "0" * 64


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory SnrDataset
# --------------------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "cd" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in SNR prediction (no torch)."""

    name = "snr-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["pred"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_samples(preds: list[float]) -> list[Batch]:
    return [
        {"iq": [0.1, 0.2], "snr_db": t, "pred": p} for t, p in zip(_TARGETS, preds, strict=True)
    ]


def _task_with_samples(preds: list[float]) -> SnrEstimationTask:
    dataset = SnrDataset("radioml_2016_10a", samples=_synthetic_samples(preds), checksum=_CHECKSUM)
    return SnrEstimationTask(datasets=[dataset])


def _committed_schema() -> dict[str, Any]:
    """Load the committed result schema; assert ``snr_estimation`` is in the ``task.name`` enum."""
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    enum = schema["properties"]["task"]["properties"]["name"]["enum"]
    assert "snr_estimation" in enum, "the additive task.name enum bump must have landed"
    return schema


def test_evaluate_emits_valid_snr_row() -> None:
    """``evaluate`` validates the ``snr_estimation`` row against the COMMITTED schema.

    The frozen ``task.name`` enum was extended with ``snr_estimation`` (a reviewed additive
    edit), so the row previously blocked at the enum now passes end-to-end unchanged.
    """
    result = evaluate(
        model=_ReplayModel(),
        task=_task_with_samples(_PREDS),
        split="test",
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,
        compute_bootstrap_ci=False,
    )
    assert result["task"]["name"] == "snr_estimation"
    assert result["metrics"]["primary"] == "rmse_db"


def _evaluate_snr(task: SnrEstimationTask, regime: RegimeSpec, batch_size: int) -> dict[str, Any]:
    """Run ``evaluate`` over a synthetic SNR task; the emitted row is schema-validated internally.

    ``evaluate`` validates its output against the committed ``result.schema.json``, whose
    ``task.name`` enum now includes ``snr_estimation``. Bootstrap CIs are disabled so the
    pure-Python path needs no numpy.
    """
    return evaluate(
        model=_ReplayModel(),
        task=task,
        split="test",
        regime=regime,
        batch_size=batch_size,
        compute_bootstrap_ci=False,
    )


def test_end_to_end_evaluate_produces_correct_regression_metrics() -> None:
    """``evaluate`` over a synthetic SNR dataset yields correct RMSE/MAE (primary = rmse_db)."""
    result = _evaluate_snr(
        _task_with_samples(_PREDS),
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 samples
    )
    assert result["task"]["name"] == "snr_estimation"
    assert result["metrics"]["primary"] == "rmse_db"
    values = result["metrics"]["values"]
    assert values["rmse_db"] == pytest.approx(_EXPECTED_RMSE)
    assert values["mae_db"] == pytest.approx(_EXPECTED_MAE)
    # The full SNR range is recorded so the row attests no cherry-picking.
    assert result["eval"]["conditions"]["full_snr_range"] is True


def test_end_to_end_perfect_model_scores_zero() -> None:
    """A perfect model yields rmse_db == mae_db == 0 through the full harness."""
    result = _evaluate_snr(
        _task_with_samples(_TARGETS),
        regime=RegimeSpec(Regime.FULL_FINETUNE),
        batch_size=2,
    )
    assert result["metrics"]["values"]["rmse_db"] == pytest.approx(0.0)
    assert result["metrics"]["values"]["mae_db"] == pytest.approx(0.0)


def test_end_to_end_row_is_schema_valid() -> None:
    """The assembled row validates against the committed schema (task.name enum now extended)."""
    from jsonschema import Draft202012Validator

    result = _evaluate_snr(
        _task_with_samples(_PREDS),
        regime=RegimeSpec(Regime.LINEAR_PROBE),
        batch_size=3,
    )
    Draft202012Validator(_committed_schema()).validate(result)
    assert result["split"]["canonical_split_id"] == "snr-radioml2016-strat-snr-8010-seed42-v1"
