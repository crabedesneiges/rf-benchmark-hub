"""WP-20 acceptance tests for the AMC task adapter.

Everything here runs on PURE-PYTHON synthetic predictions/targets/SNR -- no numpy, no
torch, no network -- so the suite passes with only ``pytest`` + ``jsonschema`` installed:

* the three AMC metrics (:class:`AccuracyOverall`, :class:`AccuracyVsSnr`, :class:`MacroF1`)
  computed on fixtures with hand-checked expected values (overall accuracy, a 2-point
  accuracy-vs-SNR curve, macro-F1);
* the task registers under ``"amc"`` and exposes the protocol datasets/metrics/split;
* an end-to-end :func:`rfbench.core.evaluate.evaluate` over a synthetic in-memory
  :class:`AmcDataset` yields a ``result.json`` that validates against
  ``schemas/result.schema.json`` with ``metrics.primary == "accuracy_overall"`` present in
  ``metrics.values``.
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
from rfbench.tasks.amc import AccuracyOverall, AccuracyVsSnr, AmcDataset, AmcTask, MacroF1

# --------------------------------------------------------------------------------------------------
# Hand-checked fixtures (3 classes, 6 samples)
# --------------------------------------------------------------------------------------------------
# targets:  [0, 1, 2, 0, 1, 2]
# preds:    [0, 1, 2, 0, 2, 2]  -> a single error at index 4 (true=1, pred=2)
_TARGETS = [0, 1, 2, 0, 1, 2]
_PREDS = [0, 1, 2, 0, 2, 2]
# snr assignment splitting into two bins of 3 samples each.
_SNR = [-10, -10, -10, 10, 10, 10]

# accuracy_overall = 5/6
_EXPECTED_ACC = 5 / 6
# macro_f1 = mean(f1(0)=1.0, f1(1)=2/3, f1(2)=0.8)
_EXPECTED_MACRO_F1 = (1.0 + (2 / 3) + 0.8) / 3
# accuracy_vs_snr: bin -10 -> 3/3 = 1.0 ; bin 10 -> 2/3
_EXPECTED_CURVE = [
    {"x": -10.0, "y": 1.0},
    {"x": 10.0, "y": 2 / 3},
]


def _meta() -> dict[str, Any]:
    return {"snr_db": list(_SNR)}


# --------------------------------------------------------------------------------------------------
# Metric compute on synthetic data (known expected values)
# --------------------------------------------------------------------------------------------------
def test_accuracy_overall_known_value() -> None:
    """``accuracy_overall`` matches the hand-computed 5/6 over the full SNR range."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS, _meta())
    computed = metric.compute()
    assert metric.primary_key == "accuracy_overall"
    assert computed["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)


def test_accuracy_overall_also_emits_macro_f1() -> None:
    """The primary metric object emits ``macro_f1`` as a second scalar with the known value."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS, _meta())
    computed = metric.compute()
    assert computed["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_standalone_macro_f1_matches() -> None:
    """The standalone :class:`MacroF1` agrees with the value from :class:`AccuracyOverall`."""
    metric = MacroF1()
    metric.update(_PREDS, _TARGETS, _meta())
    assert metric.compute()["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_accuracy_vs_snr_two_point_curve() -> None:
    """``accuracy_vs_snr`` is the expected SNR-sorted 2-point curve."""
    metric = AccuracyVsSnr()
    metric.update(_PREDS, _TARGETS, _meta())
    curve = metric.compute()["accuracy_vs_snr"]
    assert isinstance(curve, list)
    assert [pt["x"] for pt in curve] == [-10.0, 10.0]
    assert curve[0]["y"] == pytest.approx(_EXPECTED_CURVE[0]["y"])
    assert curve[1]["y"] == pytest.approx(_EXPECTED_CURVE[1]["y"])


def test_metrics_are_streaming_across_batches() -> None:
    """Splitting the batch in two and updating twice yields the same result as one update."""
    whole = AccuracyOverall()
    whole.update(_PREDS, _TARGETS, _meta())

    streamed = AccuracyOverall()
    streamed.update(_PREDS[:3], _TARGETS[:3], {"snr_db": _SNR[:3]})
    streamed.update(_PREDS[3:], _TARGETS[3:], {"snr_db": _SNR[3:]})

    assert streamed.compute() == whole.compute()


def test_reset_clears_state() -> None:
    """After :meth:`reset` the metric behaves as freshly constructed."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS, _meta())
    metric.reset()
    metric.update([0, 0], [0, 0], {"snr_db": [0, 0]})
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_perfect_predictions_are_one() -> None:
    """A perfect classifier scores 1.0 on both accuracy and macro-F1."""
    metric = AccuracyOverall()
    metric.update(_TARGETS, _TARGETS, _meta())
    computed = metric.compute()
    assert computed["accuracy_overall"] == pytest.approx(1.0)
    assert computed["macro_f1"] == pytest.approx(1.0)


def test_argmax_logits_path() -> None:
    """A batch of per-class score vectors is argmax-decoded (no numpy) before scoring."""
    # logits whose argmax reproduces _PREDS for the first three samples.
    logits = [
        [9.0, 0.0, 0.0],  # -> 0
        [0.0, 9.0, 0.0],  # -> 1
        [0.0, 0.0, 9.0],  # -> 2
    ]
    metric = AccuracyOverall()
    metric.update(logits, [0, 1, 2], {"snr_db": [0, 0, 0]})
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_eval_conditions_reports_full_snr_range() -> None:
    """The primary metric attests the full SNR range in ``eval_conditions``."""
    conditions = AccuracyOverall().eval_conditions()
    assert conditions["full_snr_range"] is True
    assert conditions["snr_db_min"] < conditions["snr_db_max"]


# --------------------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------------------
def test_task_registered_under_amc() -> None:
    """``AmcTask`` resolves by name through the registry."""
    assert "amc" in TASKS
    assert TASKS.get("amc") is AmcTask
    assert isinstance(get_task("amc"), AmcTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metrics/split/tracks match EVALUATION_PROTOCOL.md §AMC."""
    task = AmcTask()
    assert task.name == "amc"
    assert task.version == "v1"
    dataset_names = [ds.name for ds in task.datasets()]
    assert dataset_names == ["radioml_2016_10a", "radioml_2018_01a", "sig53"]
    assert task.default_split() == "test"
    assert task.tracks() == ["closed_set"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "accuracy_overall"  # primary is first
    assert set(metric_keys) == {"accuracy_overall", "accuracy_vs_snr", "macro_f1"}


def test_build_targets_extracts_labels() -> None:
    """``build_targets`` returns the per-sample modulation-class labels."""
    batch: Batch = {"iq": [[0.0], [0.0]], "label": [3, 7], "snr_db": [0, 0]}
    assert AmcTask().build_targets(batch) == [3, 7]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing an :class:`AmcDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown AMC dataset"):
        AmcDataset("not_a_dataset")


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory AmcDataset
# --------------------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "ab" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in prediction (no torch).

    Each synthetic sample carries a ``pred`` field, so ``forward`` echoes predictions
    per-row and stays correct under any batch size (order preserved by the collate).
    """

    name = "amc-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["pred"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_samples(preds: list[int]) -> list[Batch]:
    return [
        {"iq": [0.1, 0.2], "label": t, "snr_db": s, "pred": p}
        for t, s, p in zip(_TARGETS, _SNR, preds, strict=True)
    ]


def _amc_task_with_samples(preds: list[int]) -> AmcTask:
    dataset = AmcDataset("radioml_2016_10a", samples=_synthetic_samples(preds), checksum=_CHECKSUM)
    return AmcTask(datasets=[dataset])


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return schema


def test_end_to_end_evaluate_validates_against_schema() -> None:
    """``evaluate`` over a synthetic AMC dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _ReplayModel(),
        _amc_task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 samples
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "amc"
    assert result["metrics"]["primary"] == "accuracy_overall"
    values = result["metrics"]["values"]
    assert "accuracy_overall" in values
    assert values["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)
    assert values["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_end_to_end_curve_and_conditions() -> None:
    """The accuracy-vs-SNR curve and the full-SNR-range condition reach ``result.json``."""
    result = evaluate(
        _ReplayModel(),
        _amc_task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.LINEAR_PROBE),
        batch_size=6,
    )
    curve = result["metrics"]["curves"]["accuracy_vs_snr"]
    xs = [pt["x"] for pt in curve]
    assert xs == [-10.0, 10.0]
    assert curve[1]["y"] == pytest.approx(2 / 3)
    assert result["eval"]["conditions"]["full_snr_range"] is True


def test_end_to_end_perfect_model_scores_one() -> None:
    """A perfect model yields accuracy_overall == 1.0 through the full harness."""
    result = evaluate(
        _ReplayModel(),
        _amc_task_with_samples(_TARGETS),
        "test",
        RegimeSpec(Regime.FULL_FINETUNE),
        batch_size=2,
    )
    assert result["metrics"]["values"]["accuracy_overall"] == pytest.approx(1.0)
    assert math.isclose(result["metrics"]["values"]["macro_f1"], 1.0)


def test_end_to_end_writes_valid_json(tmp_path: Path) -> None:
    """With ``out_path`` set the on-disk result re-validates against the schema."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "amc" / "result.json"
    result = evaluate(
        _ReplayModel(),
        _amc_task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=3,
        out_path=out_path,
    )
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


# --- on-disk loader: index alignment (regression guard, WP-30) ------------------------


def test_radioml2016_arrays_align_with_prepare_labels(tmp_path: Path) -> None:
    """The on-disk IQ flatten order MUST equal prepare's label order (else indices corrupt).

    numpy-guarded: skips in the dep-free venv, runs on the cluster [tasks]/[data] venv.
    """
    np = pytest.importorskip("numpy")
    import pickle

    from rfbench.data.prepare.amc import _expand_radioml2016_table
    from rfbench.tasks.amc.dataset import _load_radioml2016_arrays

    table = {
        ("BPSK", 0): np.zeros((2, 2, 128), dtype=np.float32),
        ("QPSK", 10): np.ones((3, 2, 128), dtype=np.float32),
        ("GFSK", -4): np.full((1, 2, 128), 0.5, dtype=np.float32),
    }
    ds_dir = tmp_path / "radioml_2016_10a"
    ds_dir.mkdir(parents=True)
    (ds_dir / "RML2016.10a_dict.pkl").write_bytes(pickle.dumps(table))

    iq, mods, snrs = _load_radioml2016_arrays(tmp_path)
    labels = _expand_radioml2016_table(table)

    assert len(iq) == len(labels) == 6
    assert list(zip(mods, snrs, strict=True)) == labels  # identical order == index alignment
    assert iq[0].shape == (2, 128)


def test_as_class_index_handles_numpy_arrays() -> None:
    """_as_class_index decodes numpy/torch-style score vectors via argmax (WP-30 train fix)."""
    np = pytest.importorskip("numpy")
    from rfbench.tasks.amc.metrics import _as_class_index

    assert _as_class_index(np.array([0.1, 0.9, 0.2, 0.0])) == 1  # 1-D scores -> argmax
    assert _as_class_index(np.array(3)) == 3  # 0-D array -> already a class id
    assert _as_class_index([0.5, 0.1, 0.9]) == 2  # python list path unchanged
    assert _as_class_index(7) == 7  # scalar class id unchanged


def test_dataset_resolves_committed_split_checksum() -> None:
    """AmcDataset self-resolves the REAL split_checksum from the committed manifest.

    The frozen-embedding FM eval path (``eval_fm_arm.sh``) builds ``AmcDataset(name)`` and never
    calls ``prepare()`` (the split is already committed), so the dataset must read the committed
    manifest's ``split_checksum`` itself -- otherwise ``evaluate`` emits the all-zero placeholder
    and ``submit --check`` rejects the row. Falls back to the placeholder when no manifest exists.
    """
    from rfbench.tasks.amc.dataset import _PLACEHOLDER_CHECKSUM

    ds = AmcDataset("radioml_2016_10a")  # a committed manifest exists for this split
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads(
        (
            root
            / "leaderboard"
            / "splits"
            / "radioml_2016_10a"
            / f"{ds.canonical_split_id}.manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert ds.checksum == manifest["split_checksum"]  # real, not the placeholder
    assert ds.checksum != _PLACEHOLDER_CHECKSUM

    # radioml_2018_01a now ALSO has a committed manifest -> it resolves the real checksum too.
    ds_2018 = AmcDataset("radioml_2018_01a")
    manifest_2018 = json.loads(
        (
            root
            / "leaderboard"
            / "splits"
            / "radioml_2018_01a"
            / f"{ds_2018.canonical_split_id}.manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert ds_2018.checksum == manifest_2018["split_checksum"]
    assert ds_2018.checksum != _PLACEHOLDER_CHECKSUM

    # sig53 has no committed manifest -> honest placeholder fallback, never a fabricated checksum.
    assert AmcDataset("sig53").checksum == _PLACEHOLDER_CHECKSUM

    # An explicit checksum= still wins over the manifest (the synthetic/test path).
    assert AmcDataset("radioml_2016_10a", checksum="sha256:beef").checksum == "sha256:beef"
