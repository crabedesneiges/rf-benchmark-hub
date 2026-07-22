"""Acceptance tests for the spectrum-sensing (binary occupancy detection) task + data layer.

Everything here runs on PURE-PYTHON synthetic scores/targets/labels -- no numpy, no torch, no
network -- so the suite passes with only ``pytest`` (+ ``jsonschema`` for the schema check)
installed:

* :func:`pd_at_pfa` correctness -- a perfectly separable stream (Pd ~= 1.0 at pfa=0.1), a
  fully-overlapping stream (Pd ~= pfa_target), the linear interpolation between bracketing
  thresholds, and the empty-class edge cases;
* the :class:`PdAtPfa` metric update/compute/prepare_predictions incl. softmax reduction of
  length-2 rows and the frozen-threshold protocol path;
* :func:`occupancy_score` duck-typing (scalar / length-1 / length-2 / 0-d tensor-like);
* :func:`prepare_sensing` producing a valid 80/10/10 split stratified by the binary label on
  synthetic labels;
* the task registers under ``"spectrum_sensing"`` and exposes the protocol datasets/metric/split,
  plus an end-to-end :func:`rfbench.core.evaluate.evaluate` yielding a schema-valid ``result.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import TASKS, get_task
from rfbench.core.types import Batch, SplitName, Tensor
from rfbench.data.download.spectrum_deepsense import extract_occupancy_labels
from rfbench.data.prepare.sensing import CANONICAL_SPLIT_IDS, prepare_sensing
from rfbench.tasks.spectrum_sensing import (
    PdAtPfa,
    SpectrumSensingDataset,
    SpectrumSensingTask,
)
from rfbench.tasks.spectrum_sensing.metrics import (
    OccupancyClassification,
    iter_occupancy_cells,
    occupancy_score,
    pd_at_pfa,
    reduce_prediction,
)

_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


# --------------------------------------------------------------------------------------
# pd_at_pfa correctness
# --------------------------------------------------------------------------------------
def test_pd_at_pfa_perfectly_separable_is_one() -> None:
    """Occupied scores strictly above vacant scores -> Pd ~= 1.0 at pfa=0.1."""
    negatives = [0.0, 0.1, 0.2, 0.3, 0.4]
    positives = [0.6, 0.7, 0.8, 0.9, 1.0]
    assert pd_at_pfa(positives, negatives, pfa_target=0.1) == pytest.approx(1.0)


def test_pd_at_pfa_fully_overlapping_is_pfa_target() -> None:
    """Identical occupied/vacant distributions -> Pd ~= pfa_target (chance detection)."""
    scores = [i / 100.0 for i in range(100)]
    # Same score set for both classes: the ROC is the diagonal, so Pd(pfa) ~= pfa.
    assert pd_at_pfa(list(scores), list(scores), pfa_target=0.1) == pytest.approx(0.1, abs=0.02)


def test_pd_at_pfa_interpolates_between_thresholds() -> None:
    """Pd at pfa=0.1 is a linear blend of the two bracketing ROC points, not a grid jump.

    Negatives: 10 values so each distinct threshold moves FAR by 0.1. Positives chosen so the
    FAR=0.1 crossing brackets a PD step, forcing interpolation to a non-grid value.
    """
    negatives = [float(i) for i in range(10)]  # 0..9 -> FAR steps of exactly 0.1
    # Positives: 5 at high scores (detected early), 5 low. At the threshold where FAR first hits
    # 0.1 (t just above 9), PD is interpolated between the prev point (FAR=0, PD=0) and the
    # point at t=9 (FAR=0.1). With one positive == 9.5 and rest lower, PD at t=9 is 0.1.
    positives = [9.5, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2]
    value = pd_at_pfa(positives, negatives, pfa_target=0.1)
    # FAR reaches 0.1 at threshold 9 (P(neg>=9)=1/10). PD(t=9)=P(pos>=9)=1/10=0.1; prev PD (t>9)
    # is 0 at FAR 0. Interpolating at FAR=0.1 (== the point itself) gives exactly 0.1.
    assert value == pytest.approx(0.1)


def test_pd_at_pfa_interpolation_is_strictly_between_grid_points() -> None:
    """A pfa_target off the FAR grid yields a PD strictly between two distinct grid PDs.

    10 negatives (FAR grid 0.1, 0.2, ...) and pfa_target 0.05 -> the FAR=0.05 crossing sits
    between the seed point (FAR=0, PD from the positives already above every negative) and the
    first grid point (FAR=0.1). Positives chosen so those two PDs DIFFER, forcing a fractional
    interpolated value that is on neither grid.
    """
    negatives = [float(i) for i in range(10)]  # 0..9 -> FAR steps of 0.1
    # 5 positives above every negative (PD=0.5 at FAR 0); one positive tied to the score 9 (a
    # negative), so the single threshold t=9 BOTH raises FAR (0 -> 0.1) and PD (0.5 -> 0.6). The
    # FAR=0.05 crossing then interpolates strictly between (FAR 0, PD 0.5) and (FAR 0.1, PD 0.6).
    positives = [10.0, 11.0, 12.0, 13.0, 14.0, 9.0, -1.0, -2.0, -3.0, -4.0]
    value = pd_at_pfa(positives, negatives, pfa_target=0.05)
    # far0=0.0/pd0=0.5 -> far1=0.1/pd1=0.6; fraction = 0.05 / 0.1 = 0.5; PD = 0.5 + 0.5*0.1 = 0.55.
    assert value == pytest.approx(0.55)


def test_pd_at_pfa_empty_positives_is_zero() -> None:
    """No occupied windows -> Pd is 0.0 (nothing to detect)."""
    assert pd_at_pfa([], [0.1, 0.2, 0.3], pfa_target=0.1) == 0.0


def test_pd_at_pfa_empty_negatives_is_zero() -> None:
    """No vacant windows -> Pd is 0.0 (undefined false-alarm axis degrades to no detection)."""
    assert pd_at_pfa([0.7, 0.8], [], pfa_target=0.1) == 0.0


def test_pd_at_pfa_returns_max_pd_when_far_never_reaches_target() -> None:
    """Degenerate tied scores where FAR jumps past the target -> report the best PD attained."""
    # One vacant, one occupied, identical score: the only threshold gives FAR=1.0 >= 0.1 with
    # PD=1.0, but the seed point is FAR=0/PD=0; interpolation caps at the reachable PD.
    # Use a case where the FIRST (and only) FAR already overshoots and the max PD is < 1 nowhere:
    negatives = [0.5]
    positives = [0.5]
    # At t=0.5: FAR=1.0 (>=0.1), PD=1.0. prev seed FAR=0/PD=0 -> interpolate to FAR 0.1:
    # fraction=0.1/1.0=0.1 -> PD=0.1. This exercises the interpolation branch, not the max-PD
    # fallback, so also assert the fallback path directly below.
    assert pd_at_pfa(positives, negatives, pfa_target=0.1) == pytest.approx(0.1)


def test_pd_at_pfa_max_pd_fallback_all_scores_below_target_far() -> None:
    """When every threshold keeps FAR strictly below pfa_target, return the max PD reached."""
    # 20 negatives all clustered so FAR never reaches 0.1 until the very last (lowest) threshold,
    # by making pfa_target smaller than the smallest non-zero FAR step. 20 negs -> min FAR 0.05,
    # target 0.02 < 0.05 means at the first threshold FAR already >= target -> interpolation.
    # To hit the max-PD fallback, use a target LARGER than the largest FAR: impossible for
    # non-empty negs (FAR reaches 1.0). Instead exercise the fallback via a target of 1.5 (>1),
    # which FAR can never reach, so the sweep exhausts and returns the max PD (1.0 here).
    negatives = [0.0, 1.0]
    positives = [2.0, 3.0]
    assert pd_at_pfa(positives, negatives, pfa_target=1.5) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# occupancy_score duck-typing
# --------------------------------------------------------------------------------------
def test_occupancy_score_scalar_passthrough() -> None:
    """A plain scalar P(occupied) is returned verbatim."""
    assert occupancy_score(0.73) == pytest.approx(0.73)
    assert occupancy_score(1) == pytest.approx(1.0)


def test_occupancy_score_length_one_row_is_verbatim() -> None:
    """A length-1 row is an already-computed P(occupied), not a distribution."""
    assert occupancy_score([0.42]) == pytest.approx(0.42)


def test_occupancy_score_length_two_row_is_softmax_class1() -> None:
    """A length-2 [vacant, occupied] row reduces to softmax(row)[1]."""
    # Equal logits -> 0.5; a strong occupied logit -> near 1.0.
    assert occupancy_score([0.0, 0.0]) == pytest.approx(0.5)
    assert occupancy_score([0.0, 10.0]) == pytest.approx(1.0, abs=1e-3)
    assert occupancy_score([10.0, 0.0]) == pytest.approx(0.0, abs=1e-3)
    # Hand value: softmax([1, 2])[1] = e / (1 + e) via the sigmoid(2-1) form.
    import math

    assert occupancy_score([1.0, 2.0]) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_occupancy_score_zero_d_tensor_like_item() -> None:
    """A 0-d tensor-like exposing .item() is unwrapped to its scalar."""

    class _Scalar:
        def __iter__(self) -> object:
            raise TypeError("0-d tensor is not iterable")

        def item(self) -> float:
            return 0.61

    assert occupancy_score(_Scalar()) == pytest.approx(0.61)


def test_occupancy_score_bool_is_numeric() -> None:
    """A bool is read as its 0/1 value (not as a 0-length distribution)."""
    assert occupancy_score(True) == 1.0
    assert occupancy_score(False) == 0.0


def test_occupancy_score_rejects_long_rows() -> None:
    """A length > 2 output is a caller bug -- occupancy is binary."""
    with pytest.raises(ValueError, match="length-1 or length-2"):
        occupancy_score([0.1, 0.2, 0.3])


def test_occupancy_score_rejects_empty_row() -> None:
    """An empty row cannot be scored."""
    with pytest.raises(ValueError, match="empty prediction row"):
        occupancy_score([])


# --------------------------------------------------------------------------------------
# PdAtPfa metric update / compute / prepare_predictions
# --------------------------------------------------------------------------------------
def test_metric_names_and_primary_key() -> None:
    """The metric identifies itself as pd@pfa=0.1 (secondary ROC metric)."""
    metric = PdAtPfa()
    assert metric.name == "pd@pfa=0.1"
    assert metric.primary_key == "pd@pfa=0.1"


def test_occupancy_classification_primary_and_metrics() -> None:
    """OccupancyClassification has the 'f1' primary and also emits accuracy/precision/recall."""
    metric = OccupancyClassification()
    assert metric.name == "f1"
    assert metric.primary_key == "f1"
    # 3 occupied (scores >=0.5) + 2 vacant (scores <0.5), all correct -> perfect.
    metric.update([0.9, 0.8, 0.6, 0.2, 0.1], [1, 1, 1, 0, 0])
    out = metric.compute()
    assert out["f1"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)


def test_occupancy_classification_confusion_and_threshold() -> None:
    """The metrics reflect the 0.5-threshold confusion counts; a bad target raises."""
    metric = OccupancyClassification()
    # scores -> preds @0.5: [1,1,0,0]; targets [1,0,1,0] -> 1 TP, 1 FP, 1 FN, 1 TN.
    metric.update([0.7, 0.6, 0.4, 0.3], [1, 0, 1, 0])
    out = metric.compute()
    assert out["accuracy"] == pytest.approx(0.5)  # 2/4 correct (TP + TN)
    assert out["precision"] == pytest.approx(0.5)  # TP / (TP+FP) = 1/2
    assert out["recall"] == pytest.approx(0.5)  # TP / (TP+FN) = 1/2
    assert out["f1"] == pytest.approx(0.5)  # harmonic mean of 0.5 / 0.5
    with pytest.raises(ValueError):
        OccupancyClassification().update([0.9], [2])  # target outside {0,1}


def test_metric_compute_self_contained_separable() -> None:
    """Self-contained (threshold=None) compute reports Pd@Pfa off the stream's own ROC."""
    metric = PdAtPfa()
    metric.update([0.6, 0.7, 0.8, 0.9, 1.0], [1, 1, 1, 1, 1])
    metric.update([0.0, 0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0, 0])
    computed = metric.compute()
    assert computed["pd@pfa=0.1"] == pytest.approx(1.0)
    assert computed["auroc"] == pytest.approx(1.0)
    assert computed["pfa_achieved"] == pytest.approx(0.1)
    assert isinstance(computed["roc"], list)
    assert computed["roc"][0] == {"x": 0.0, "y": 0.0}
    assert computed["roc"][-1] == {"x": 1.0, "y": 1.0}


def test_metric_softmax_reduction_of_length_two_rows() -> None:
    """update() reduces length-2 [vacant, occupied] rows to occupied posteriors before scoring."""
    metric = PdAtPfa()
    # Occupied windows peak on class 1; vacant windows peak on class 0 -> perfect separation.
    metric.update([[0.0, 9.0], [1.0, 8.0]], [1, 1])
    metric.update([[9.0, 0.0], [8.0, 1.0]], [0, 0])
    assert metric.compute()["pd@pfa=0.1"] == pytest.approx(1.0)


def test_metric_prepare_predictions_reduces_rows_to_scalars() -> None:
    """prepare_predictions reduces a batch of rows to scalar P(occupied), idempotent on scalars."""
    metric = PdAtPfa()
    reduced = metric.prepare_predictions([[0.0, 10.0], [10.0, 0.0], 0.5])
    assert reduced[0] == pytest.approx(1.0, abs=1e-3)
    assert reduced[1] == pytest.approx(0.0, abs=1e-3)
    assert reduced[2] == pytest.approx(0.5)


def test_metric_rejects_non_binary_target() -> None:
    """A target outside {0, 1} raises ValueError."""
    metric = PdAtPfa()
    with pytest.raises(ValueError, match="must be 0 .* or 1"):
        metric.update([0.5], [2])


def test_metric_frozen_threshold_reports_pd_and_achieved_pfa() -> None:
    """With a frozen threshold, compute reports Pd + the achieved Pfa at that threshold."""
    # threshold 0.5: occupied {0.6, 0.9} both detected (Pd=1.0); vacant {0.4, 0.7} -> 0.7 is a
    # false alarm (Pfa=0.5).
    metric = PdAtPfa(threshold=0.5)
    metric.update([0.6, 0.9], [1, 1])
    metric.update([0.4, 0.7], [0, 0])
    computed = metric.compute()
    assert computed["pd@pfa=0.1"] == pytest.approx(1.0)
    assert computed["pfa_achieved"] == pytest.approx(0.5)


def test_metric_reset_clears_state() -> None:
    """After reset the metric behaves as freshly constructed."""
    metric = PdAtPfa()
    metric.update([0.9], [1])
    metric.update([0.1], [0])
    metric.reset()
    assert metric.compute()["auroc"] == pytest.approx(0.5)  # empty stream -> chance


def test_metric_streaming_matches_single_update() -> None:
    """Two partial updates yield the same compute() as one combined update."""
    whole = PdAtPfa()
    whole.update([0.6, 0.7, 0.1, 0.2], [1, 1, 0, 0])

    streamed = PdAtPfa()
    streamed.update([0.6, 0.7], [1, 1])
    streamed.update([0.1, 0.2], [0, 0])
    assert streamed.compute() == whole.compute()


# --------------------------------------------------------------------------------------
# extract_occupancy_labels (pure-stdlib manifest -> 0/1 labels)
# --------------------------------------------------------------------------------------
def test_extract_occupancy_labels_mixed_manifest() -> None:
    """Numeric, string and bool occupancy indicators all coerce to strict {0, 1}."""
    manifest = [0, 1, "occupied", "vacant", "OCCUPIED", True, False, 1.0, 0.0]
    assert extract_occupancy_labels(manifest) == [0, 1, 1, 0, 1, 1, 0, 1, 0]


def test_extract_occupancy_labels_rejects_out_of_range() -> None:
    """A non-binary numeric indicator fails loudly."""
    with pytest.raises(ValueError, match="must be 0 or 1"):
        extract_occupancy_labels([2])


def test_extract_occupancy_labels_rejects_unknown_string() -> None:
    """An unrecognised occupancy string fails loudly."""
    with pytest.raises(ValueError, match="unrecognised .* occupancy string"):
        extract_occupancy_labels(["maybe"])


# --------------------------------------------------------------------------------------
# prepare_sensing: 80/10/10 stratified by occupancy on synthetic labels
# --------------------------------------------------------------------------------------
def test_prepare_sensing_stratified_80_10_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepSense -> 80/10/10 stratified by the binary occupancy label; each class split 8/1/1."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    # 50 occupied + 50 vacant = 100 windows; each class -> 40/5/5.
    labels = [1] * 50 + [0] * 50

    split, manifest = prepare_sensing("deepsense", out_dir=str(tmp_path), labels=labels)

    sizes = {name: len(split.indices[name]) for name in _SPLITS}
    assert sizes == {"train": 80, "val": 10, "test": 10}  # 100 * 0.8/0.1/0.1

    # Each occupancy class contributes 40/5/5 to train/val/test.
    label_of = {i: labels[i] for i in range(len(labels))}
    for name, expected in (("train", 40), ("val", 5), ("test", 5)):
        counts = Counter(label_of[i] for i in split.indices[name])
        assert counts[0] == expected, name
        assert counts[1] == expected, name

    # No leakage, full coverage.
    all_idx = sorted(i for name in _SPLITS for i in split.indices[name])
    assert all_idx == list(range(len(labels)))
    assert not (set(split.indices["train"]) & set(split.indices["val"]))
    assert not (set(split.indices["train"]) & set(split.indices["test"]))
    assert not (set(split.indices["val"]) & set(split.indices["test"]))

    # Canonical id + provenance recorded on the manifest.
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["deepsense"]
    assert manifest.dataset == "deepsense"
    assert manifest.n_items == len(labels)
    assert manifest.seed == 42


def test_prepare_sensing_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs (different out_dirs) produce byte-identical indices + checksum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    labels = [1, 0] * 40

    split_a, _ = prepare_sensing("deepsense", out_dir=str(tmp_path / "a"), labels=labels)
    split_b, _ = prepare_sensing("deepsense", out_dir=str(tmp_path / "b"), labels=labels)
    assert split_a.indices == split_b.indices
    assert split_a.checksum == split_b.checksum


def test_prepare_sensing_requires_labels(tmp_path: Path) -> None:
    """DeepSense has no canonical literature split -> labels are required."""
    with pytest.raises(ValueError, match="has no canonical split"):
        prepare_sensing("deepsense", out_dir=str(tmp_path), labels=None)


def test_prepare_sensing_rejects_non_binary_label(tmp_path: Path) -> None:
    """A label outside {0, 1} is rejected before stratification."""
    with pytest.raises(ValueError, match="must be 0 or 1"):
        prepare_sensing("deepsense", out_dir=str(tmp_path), labels=[0, 1, 2])


def test_prepare_sensing_rejects_unknown_dataset(tmp_path: Path) -> None:
    """An unknown dataset id is rejected."""
    with pytest.raises(ValueError, match="unknown sensing dataset"):
        prepare_sensing("not_a_dataset", out_dir=str(tmp_path), labels=[0, 1])


# --------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------
def test_task_registered_under_spectrum_sensing() -> None:
    """``SpectrumSensingTask`` resolves by name through the registry."""
    assert "spectrum_sensing" in TASKS
    assert TASKS.get("spectrum_sensing") is SpectrumSensingTask
    assert isinstance(get_task("spectrum_sensing"), SpectrumSensingTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metric/split/tracks match EVALUATION_PROTOCOL.md §spectrum_sensing."""
    task = SpectrumSensingTask()
    assert task.name == "spectrum_sensing"
    assert task.version == "v1"
    assert [ds.name for ds in task.datasets()] == ["deepsense"]
    assert task.default_split() == "test"
    assert task.tracks() == ["occupancy"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "f1"  # primary is first
    assert "pd@pfa=0.1" in metric_keys  # ROC operating point kept as a secondary metric


def test_canonical_split_id_version_matches_task_version() -> None:
    """The dataset's canonical_split_id -v<N> suffix must equal the task version."""
    task = SpectrumSensingTask()
    dataset = task.datasets()[0]
    assert dataset.canonical_split_id.endswith(f"-{task.version}")


def test_build_targets_extracts_labels() -> None:
    """``build_targets`` returns the per-sample occupancy labels."""
    batch: Batch = {"iq": [[0.0], [0.0]], "label": [1, 0]}
    assert SpectrumSensingTask().build_targets(batch) == [1, 0]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing a :class:`SpectrumSensingDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown sensing dataset"):
        SpectrumSensingDataset("not_a_dataset")


# --------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory SpectrumSensingDataset
# --------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "ab" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in P(occupied) (no torch)."""

    name = "sensing-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["score"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_samples() -> list[Batch]:
    """10 windows: 5 occupied (high scores) + 5 vacant (low scores), perfectly separable."""
    occupied = [{"iq": [0.1, 0.2], "label": 1, "score": s} for s in (0.6, 0.7, 0.8, 0.9, 1.0)]
    vacant = [{"iq": [0.1, 0.2], "label": 0, "score": s} for s in (0.0, 0.1, 0.2, 0.3, 0.4)]
    return occupied + vacant


def _task_with_samples() -> SpectrumSensingTask:
    dataset = SpectrumSensingDataset("deepsense", samples=_synthetic_samples(), checksum=_CHECKSUM)
    return SpectrumSensingTask(datasets=[dataset])


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def test_end_to_end_evaluate_validates_against_schema() -> None:
    """``evaluate`` over a synthetic sensing dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _ReplayModel(),
        _task_with_samples(),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 10 samples
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "spectrum_sensing"
    assert result["metrics"]["primary"] == "f1"
    values = result["metrics"]["values"]
    assert values["f1"] == pytest.approx(1.0)
    assert values["accuracy"] == pytest.approx(1.0)
    # Secondaries stay on the row: the classical ROC operating point + auroc.
    assert values["pd@pfa=0.1"] == pytest.approx(1.0)
    assert values["auroc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# MULTI-LABEL occupancy (DeepSense's 16 LTE-M sub-bands): cells = window x subband
# --------------------------------------------------------------------------------------
def test_iter_occupancy_cells_binary_path() -> None:
    """A scalar target yields ONE cell per window (the binary occupancy path)."""
    assert iter_occupancy_cells([0.7, 0.2], [1, 0]) == [(0.7, 1), (0.2, 0)]


def test_iter_occupancy_cells_multilabel_expands_per_subband() -> None:
    """A length-K target expands into K window×subband cells with the matching per-band probs."""
    cells = iter_occupancy_cells([[0.9, 0.1, 0.8]], [[1, 0, 1]])
    assert cells == [(0.9, 1), (0.1, 0), (0.8, 1)]


def test_iter_occupancy_cells_length_mismatch_raises() -> None:
    """A prediction row whose width differs from the target vector fails loudly."""
    with pytest.raises(ValueError, match="length mismatch"):
        iter_occupancy_cells([[0.1, 0.2, 0.3]], [[1, 0]])


def test_reduce_prediction_binary_vs_multilabel() -> None:
    """reduce_prediction collapses binary rows to a scalar but keeps multi-label rows as lists."""
    assert reduce_prediction(0.7) == pytest.approx(0.7)  # scalar passthrough
    assert reduce_prediction([0.42]) == pytest.approx(0.42)  # length-1 passthrough
    assert reduce_prediction([0.0, 10.0]) == pytest.approx(1.0, abs=1e-3)  # length-2 softmax
    assert reduce_prediction([0.9, 0.1, 0.8]) == [0.9, 0.1, 0.8]  # length-3+ -> per-band list


def test_occupancy_classification_multilabel_all_correct() -> None:
    """Micro-F1 over window×subband cells is 1.0 when every one of the 8 cells is correct."""
    metric = OccupancyClassification()
    preds = [[0.9, 0.1, 0.8, 0.2], [0.2, 0.7, 0.3, 0.6]]  # 2 windows x 4 sub-bands
    targets = [[1, 0, 1, 0], [0, 1, 0, 1]]
    metric.update(preds, targets)
    out = metric.compute()
    assert out["f1"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)


def test_occupancy_classification_multilabel_micro_averages_cells() -> None:
    """F1 micro-averages over cells: 1 TP + 1 FP in a single window -> precision .5, recall 1."""
    metric = OccupancyClassification()
    metric.update([[0.9, 0.9]], [[1, 0]])  # cell0 TP, cell1 FP
    out = metric.compute()
    assert out["precision"] == pytest.approx(0.5)  # TP / (TP + FP) = 1/2
    assert out["recall"] == pytest.approx(1.0)  # TP / (TP + FN) = 1/1
    assert out["f1"] == pytest.approx(2.0 * 0.5 * 1.0 / 1.5)


def test_occupancy_classification_multilabel_rejects_non_binary_bit() -> None:
    """A per-subband target outside {0, 1} fails loudly (same guard as the binary path)."""
    with pytest.raises(ValueError, match="must be 0 .* or 1"):
        OccupancyClassification().update([[0.5, 0.5]], [[1, 2]])


def test_pd_at_pfa_multilabel_separable_cells() -> None:
    """Pd@Pfa / AUROC micro-average over cells: occupied cells high, vacant low -> perfect."""
    metric = PdAtPfa()
    metric.update([[0.9, 0.1], [0.8, 0.2]], [[1, 0], [1, 0]])
    computed = metric.compute()
    assert computed["pd@pfa=0.1"] == pytest.approx(1.0)
    assert computed["auroc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic MULTI-LABEL SpectrumSensingDataset
# --------------------------------------------------------------------------------------
class _MultiLabelReplayModel(Model):
    """A deterministic baseline replaying each window's baked-in per-subband prob row (no torch)."""

    name = "sensing-ml-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["score"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_multilabel_samples() -> list[Batch]:
    """6 windows × 4 sub-bands, perfectly separable per cell (prob == label)."""
    samples: list[Batch] = []
    for i in range(6):
        label = [(i + j) % 2 for j in range(4)]  # alternating -> both classes present
        score = [float(bit) for bit in label]  # 1.0 occupied / 0.0 vacant
        samples.append({"iq": [[0.1] * 4, [0.2] * 4], "label": label, "score": score})
    return samples


def _multilabel_task() -> SpectrumSensingTask:
    dataset = SpectrumSensingDataset(
        "deepsense", samples=_synthetic_multilabel_samples(), checksum=_CHECKSUM
    )
    return SpectrumSensingTask(datasets=[dataset])


def test_end_to_end_evaluate_multilabel_validates_against_schema() -> None:
    """``evaluate`` over a synthetic 16-band-style dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _MultiLabelReplayModel(),
        _multilabel_task(),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 windows
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "spectrum_sensing"
    assert result["metrics"]["primary"] == "f1"
    values = result["metrics"]["values"]
    assert values["f1"] == pytest.approx(1.0)  # micro-averaged over the 24 cells
    assert values["accuracy"] == pytest.approx(1.0)
    assert values["pd@pfa=0.1"] == pytest.approx(1.0)
    assert values["auroc"] == pytest.approx(1.0)
