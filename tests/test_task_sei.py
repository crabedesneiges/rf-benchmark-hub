"""WP-21 acceptance tests for the SEI task.

Pure stdlib: no numpy/torch, no network. Exercises the SEI metrics on synthetic
predictions/score distributions with known expected values, the task/registry wiring
(closed-set vs open-set tracks kept SEPARATE), and an end-to-end :func:`evaluate` that
emits a schema-valid ``result.json`` for the ``closed_set`` track with a declared regime.
Must pass with only ``pytest`` + ``jsonschema`` installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.dataset import Dataset
from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import TASKS, get_task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.tasks.sei import SEI_TRACKS, OpenSetMetric, Rank1Accuracy, SeiDataset, SeiTask
from rfbench.tasks.sei.metrics import auroc, eer


# --------------------------------------------------------------------------------------------------
# Registry + task wiring
# --------------------------------------------------------------------------------------------------
def test_task_registered_as_sei() -> None:
    """The task resolves by name from the registry and is a ``SeiTask``."""
    assert "sei" in TASKS
    assert TASKS.get("sei") is SeiTask
    assert isinstance(get_task("sei"), SeiTask)


def test_task_identity_and_default_split() -> None:
    """Registered name/version and the default split match the protocol."""
    task = SeiTask()
    assert task.name == "sei"
    assert task.version == "v1"
    assert task.default_split() == "test"
    assert task.track == "closed_set"


def test_tracks_expose_closed_and_open_set_conditions() -> None:
    """``tracks()`` exposes the three closed-set conditions AND the open-set track."""
    tracks = SeiTask().tracks()
    assert tracks == list(SEI_TRACKS)
    assert set(tracks) == {"closed_set", "cross_receiver", "cross_day", "open_set"}


def test_default_track_canonical_split_id_is_closed_set() -> None:
    """The default track's dataset carries the closed-set canonical split id."""
    (dataset,) = SeiTask().datasets()
    assert dataset.canonical_split_id == "sei-wisig-closedset-strat-tx-8010-seed42-v1"


@pytest.mark.parametrize(
    ("track", "expected_split_id"),
    [
        ("closed_set", "sei-wisig-closedset-strat-tx-8010-seed42-v1"),
        ("cross_receiver", "sei-wisig-crossrx-grouped-8010-seed42-v1"),
        ("cross_day", "sei-wisig-crossday-grouped-8010-seed42-v1"),
        ("open_set", "sei-wisig-closedset-strat-tx-8010-seed42-v1"),
    ],
)
def test_each_track_binds_its_canonical_split_id(track: Track, expected_split_id: str) -> None:
    """Each track binds the dataset variant to the right canonical split id."""
    (dataset,) = SeiTask(track).datasets()
    assert dataset.canonical_split_id == expected_split_id


def test_closed_and_open_set_metrics_are_separate() -> None:
    """Closed-set tracks emit rank-1; the open-set track emits AUROC+EER -- never mixed."""
    closed_metrics = SeiTask("closed_set").metrics()
    assert [m.primary_key for m in closed_metrics] == ["rank1_accuracy"]
    assert isinstance(closed_metrics[0], Rank1Accuracy)

    open_metrics = SeiTask("open_set").metrics()
    assert [m.primary_key for m in open_metrics] == ["auroc"]
    assert isinstance(open_metrics[0], OpenSetMetric)
    open_keys = set(open_metrics[0].compute())
    assert open_keys == {"auroc", "eer"}
    # A closed-set row never carries open-set scalars and vice-versa.
    assert "rank1_accuracy" not in open_keys
    assert "auroc" not in SeiTask("cross_receiver").metrics()[0].compute()


def test_unknown_track_raises() -> None:
    """Constructing the task with an unknown track raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown SEI track"):
        SeiTask("bogus")


# --------------------------------------------------------------------------------------------------
# Rank-1 accuracy on synthetic predictions
# --------------------------------------------------------------------------------------------------
def test_rank1_accuracy_on_synthetic_preds() -> None:
    """Rank-1 on pre-argmaxed integer predictions: 4/5 correct -> 0.8."""
    metric = Rank1Accuracy()
    metric.reset()
    metric.update([3, 1, 2, 0, 4], [3, 1, 2, 9, 4])  # one mismatch (idx 3)
    assert metric.compute() == {"rank1_accuracy": pytest.approx(0.8)}


def test_rank1_accuracy_argmaxes_score_rows() -> None:
    """Rank-1 argmaxes per-class score rows before comparing to the target id."""
    metric = Rank1Accuracy()
    metric.reset()
    # rows argmax to 1, 0, 2; targets 1, 0, 0 -> 2/3 correct.
    metric.update([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.1, 0.2, 0.8]], [1, 0, 0])
    assert metric.compute() == {"rank1_accuracy": pytest.approx(2 / 3)}


def test_rank1_accuracy_empty_stream_is_zero() -> None:
    """An empty stream yields 0.0 (no division by zero)."""
    metric = Rank1Accuracy()
    metric.reset()
    assert metric.compute() == {"rank1_accuracy": 0.0}


def test_rank1_accuracy_streams_across_batches() -> None:
    """Streaming several batches accumulates the same total as one big batch."""
    metric = Rank1Accuracy()
    metric.reset()
    metric.update([1, 2], [1, 9])  # 1/2
    metric.update([3, 4], [3, 4])  # 2/2
    assert metric.compute() == {"rank1_accuracy": pytest.approx(0.75)}


# --------------------------------------------------------------------------------------------------
# AUROC + EER on synthetic score distributions with known expected values
# --------------------------------------------------------------------------------------------------
def test_auroc_eer_perfectly_separable() -> None:
    """Perfectly separable scores -> AUROC == 1.0 and EER == 0.0."""
    positives = [0.9, 0.8, 0.95, 0.7]
    negatives = [0.1, 0.2, 0.05, 0.3]
    assert auroc(positives, negatives) == pytest.approx(1.0)
    assert eer(positives, negatives) == pytest.approx(0.0)


def test_auroc_eer_identical_distributions() -> None:
    """Identical positive/negative distributions -> AUROC ~ 0.5 and EER ~ 0.5."""
    positives = [0.5, 0.5, 0.5, 0.5]
    negatives = [0.5, 0.5, 0.5, 0.5]
    assert auroc(positives, negatives) == pytest.approx(0.5)
    assert eer(positives, negatives) == pytest.approx(0.5)


def test_auroc_matches_hand_computed_rank_statistic() -> None:
    """AUROC equals the Mann-Whitney probability on a small hand-checkable set.

    pos={2, 3}, neg={1, 4}: pairs (2,1),(2,4),(3,1),(3,4) -> pos wins 2/4 -> 0.5.
    """
    assert auroc([2.0, 3.0], [1.0, 4.0]) == pytest.approx(0.5)


def test_auroc_counts_ties_at_half_weight() -> None:
    """A tie between a single positive and negative contributes 0.5 to AUROC."""
    assert auroc([1.0], [1.0]) == pytest.approx(0.5)
    # One clean win plus one tie: (win + 0.5*tie)/2 pairs.
    assert auroc([2.0, 1.0], [1.0, 0.0]) == pytest.approx(0.875)


def test_auroc_symmetry_flips_around_half() -> None:
    """Swapping positives and negatives reflects AUROC about 0.5."""
    positives = [0.9, 0.6, 0.8]
    negatives = [0.1, 0.4, 0.2]
    assert auroc(positives, negatives) + auroc(negatives, positives) == pytest.approx(1.0)


def test_eer_partial_overlap_is_between_zero_and_half() -> None:
    """Overlapping-but-separable distributions give an EER strictly in (0, 0.5)."""
    positives = [0.6, 0.7, 0.8, 0.9]
    negatives = [0.1, 0.2, 0.3, 0.65]  # one impostor intrudes into the genuine range
    value = eer(positives, negatives)
    assert 0.0 < value < 0.5


def test_open_set_metric_streams_scores() -> None:
    """The OpenSetMetric accumulates (score, label) batches and reports AUROC+EER."""
    metric = OpenSetMetric()
    metric.reset()
    metric.update([0.9, 0.8], [1, 1])  # genuine
    metric.update([0.1, 0.2], [0, 0])  # impostor
    computed = metric.compute()
    assert computed["auroc"] == pytest.approx(1.0)
    assert computed["eer"] == pytest.approx(0.0)


def test_open_set_metric_rejects_non_binary_label() -> None:
    """A label outside {0, 1} raises so a mislabelled open-set batch fails loudly."""
    metric = OpenSetMetric()
    metric.reset()
    with pytest.raises(ValueError, match="0 .impostor. or 1"):
        metric.update([0.5], [2])


def test_auroc_empty_class_degrades_to_chance() -> None:
    """An empty positive or negative class degrades AUROC to 0.5 and EER to 0.0."""
    assert auroc([], [0.1, 0.2]) == pytest.approx(0.5)
    assert eer([0.9], []) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------------
# Dataset adapter (in-memory synthetic set; no numpy/network)
# --------------------------------------------------------------------------------------------------
def test_dataset_load_yields_iq_label_meta() -> None:
    """The in-memory adapter yields (iq, tx_label, meta{rx, day}) per sample."""
    samples: dict[SplitName, list[Batch]] = {
        "test": [
            {"iq": [0.1, 0.2], "label": 3, "meta": {"rx": 100, "day": 0}},
            {"iq": [0.3, 0.4], "label": 1, "meta": {"rx": 101, "day": 1}},
        ]
    }
    dataset = SeiDataset("wisig", track="closed_set", samples=samples)
    loaded = list(dataset.load("test"))
    assert [s["label"] for s in loaded] == [3, 1]
    assert loaded[0]["meta"] == {"rx": 100, "day": 0}
    assert list(dataset.load("val")) == []  # absent split -> empty


def test_dataset_rejects_mismatched_track() -> None:
    """Loading with a track other than the one it was built for raises."""
    dataset = SeiDataset("wisig", track="closed_set", samples={"test": []})
    with pytest.raises(ValueError, match="serves track 'closed_set'"):
        dataset.load("test", track="cross_day")


def test_dataset_unknown_dataset_and_track_raise() -> None:
    """Unknown dataset / track names raise at construction."""
    with pytest.raises(ValueError, match="unknown SEI dataset"):
        SeiDataset("nonexistent_dataset")
    with pytest.raises(ValueError, match="unknown SEI track"):
        SeiDataset("wisig", track="bogus")
    with pytest.raises(ValueError, match="does not support track"):
        SeiDataset("oracle", track="cross_receiver")


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() -> schema-valid result.json (closed_set track, regime declared)
# --------------------------------------------------------------------------------------------------
class _InMemorySplit:
    """A list-backed map-style dataset of per-sample batches."""

    def __init__(self, samples: Sequence[Batch]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._samples)


class _SeiTaskWithSamples(SeiTask):
    """A ``SeiTask`` whose dataset is backed by an injected in-memory split (no numpy)."""

    def __init__(self, track: Track, samples: dict[SplitName, list[Batch]]) -> None:
        super().__init__(track)
        self._samples = samples

    def datasets(self) -> list[Dataset]:
        return [SeiDataset("wisig", track=self._track, samples=self._samples)]


class _PerfectFingerprinter(Model):
    """A model that reads the batch and echoes the true transmitter id (perfect rank-1)."""

    name = "sei-oracle-classifier"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["label"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 42_000


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    return json.loads(schema_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _closed_set_samples() -> dict[SplitName, list[Batch]]:
    return {
        "test": [
            {"iq": [0.1, 0.2], "label": 0, "meta": {"rx": 100, "day": 0}},
            {"iq": [0.3, 0.4], "label": 1, "meta": {"rx": 101, "day": 0}},
            {"iq": [0.5, 0.6], "label": 2, "meta": {"rx": 100, "day": 1}},
            {"iq": [0.7, 0.8], "label": 1, "meta": {"rx": 102, "day": 1}},
        ]
    }


def test_evaluate_end_to_end_closed_set_is_schema_valid() -> None:
    """evaluate() on the closed_set track emits a schema-valid result with the regime declared."""
    from jsonschema import Draft202012Validator

    task = _SeiTaskWithSamples("closed_set", _closed_set_samples())
    result = evaluate(
        _PerfectFingerprinter(),
        task,
        "test",
        RegimeSpec(Regime.LINEAR_PROBE),
        track="closed_set",
        batch_size=2,
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"] == {"name": "sei", "version": "v1"}
    assert result["regime"] == {"name": "linear_probe"}  # declared, never inferred
    assert result["split"]["track"] == "closed_set"
    assert result["split"]["canonical_split_id"] == "sei-wisig-closedset-strat-tx-8010-seed42-v1"
    assert result["metrics"]["primary"] == "rank1_accuracy"
    assert result["metrics"]["values"]["rank1_accuracy"] == pytest.approx(1.0)
    # Closed-set row carries no open-set scalars.
    assert "auroc" not in result["metrics"]["values"]
    assert result["verification"] == {"status": "self_reported"}
    assert result["eval"]["n_samples"] == 4


def test_evaluate_writes_schema_valid_result_to_disk(tmp_path: Path) -> None:
    """evaluate(out_path=...) writes a file that re-validates against the schema."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "sei" / "result.json"
    task = _SeiTaskWithSamples("closed_set", _closed_set_samples())
    result = evaluate(
        _PerfectFingerprinter(),
        task,
        "test",
        RegimeSpec(Regime.FULL_FINETUNE),
        track="closed_set",
        batch_size=4,
        out_path=out_path,
    )
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)
