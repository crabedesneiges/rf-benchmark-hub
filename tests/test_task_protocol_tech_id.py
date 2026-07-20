"""Acceptance tests for the protocol-tech-ID (WiFi 802.11 standard recognition) task adapter.

Everything here runs on PURE-PYTHON synthetic predictions/targets -- no numpy, no torch, no
network -- so the suite passes with only ``pytest`` + ``jsonschema`` installed:

* the two protocol-tech-ID metrics (:class:`AccuracyOverall`, :class:`MacroF1`) computed on
  fixtures with hand-checked expected values (overall accuracy, macro-F1);
* the task registers under ``"protocol_tech_id"`` and exposes the protocol datasets/metrics/split;
* an end-to-end :func:`rfbench.core.evaluate.evaluate` over a synthetic in-memory
  :class:`ProtocolDataset` yields a ``result.json`` that validates against
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
from rfbench.tasks.protocol_tech_id import (
    AccuracyOverall,
    MacroF1,
    ProtocolDataset,
    ProtocolTechIdTask,
)

# --------------------------------------------------------------------------------------------------
# Hand-checked fixtures (3 classes, 6 samples)
# --------------------------------------------------------------------------------------------------
# targets:  [0, 1, 2, 0, 1, 2]
# preds:    [0, 1, 2, 0, 2, 2]  -> a single error at index 4 (true=1, pred=2)
_TARGETS = [0, 1, 2, 0, 1, 2]
_PREDS = [0, 1, 2, 0, 2, 2]

# accuracy_overall = 5/6
_EXPECTED_ACC = 5 / 6
# macro_f1 = mean(f1(0)=1.0, f1(1)=2/3, f1(2)=0.8)
_EXPECTED_MACRO_F1 = (1.0 + (2 / 3) + 0.8) / 3


# --------------------------------------------------------------------------------------------------
# Metric compute on synthetic data (known expected values)
# --------------------------------------------------------------------------------------------------
def test_accuracy_overall_known_value() -> None:
    """``accuracy_overall`` matches the hand-computed 5/6 over the whole split."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    computed = metric.compute()
    assert metric.primary_key == "accuracy_overall"
    assert computed["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)


def test_accuracy_overall_also_emits_macro_f1() -> None:
    """The primary metric object emits ``macro_f1`` as a second scalar with the known value."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    assert metric.compute()["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_standalone_macro_f1_matches() -> None:
    """The standalone :class:`MacroF1` agrees with the value from :class:`AccuracyOverall`."""
    metric = MacroF1()
    metric.update(_PREDS, _TARGETS)
    assert metric.compute()["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_metrics_are_streaming_across_batches() -> None:
    """Splitting the batch in two and updating twice yields the same result as one update."""
    whole = AccuracyOverall()
    whole.update(_PREDS, _TARGETS)

    streamed = AccuracyOverall()
    streamed.update(_PREDS[:3], _TARGETS[:3])
    streamed.update(_PREDS[3:], _TARGETS[3:])

    assert streamed.compute() == whole.compute()


def test_reset_clears_state() -> None:
    """After :meth:`reset` the metric behaves as freshly constructed."""
    metric = AccuracyOverall()
    metric.update(_PREDS, _TARGETS)
    metric.reset()
    metric.update([0, 0], [0, 0])
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_perfect_predictions_are_one() -> None:
    """A perfect classifier scores 1.0 on both accuracy and macro-F1."""
    metric = AccuracyOverall()
    metric.update(_TARGETS, _TARGETS)
    computed = metric.compute()
    assert computed["accuracy_overall"] == pytest.approx(1.0)
    assert computed["macro_f1"] == pytest.approx(1.0)


def test_argmax_logits_path() -> None:
    """A batch of per-class score vectors is argmax-decoded (no numpy) before scoring."""
    logits = [
        [9.0, 0.0, 0.0, 0.0],  # -> 0
        [0.0, 9.0, 0.0, 0.0],  # -> 1
        [0.0, 0.0, 9.0, 0.0],  # -> 2
    ]
    metric = AccuracyOverall()
    metric.update(logits, [0, 1, 2])
    assert metric.compute()["accuracy_overall"] == pytest.approx(1.0)


def test_accuracy_overall_has_no_snr_conditions() -> None:
    """Protocol-tech-ID's primary metric declares NO extra eval conditions (no SNR grid)."""
    # The AMC AccuracyOverall exposes eval_conditions(); the protocol one must not, so
    # result.json.eval.conditions stays clean for this task.
    assert not hasattr(AccuracyOverall(), "eval_conditions")


# --------------------------------------------------------------------------------------------------
# Task wiring + registry
# --------------------------------------------------------------------------------------------------
def test_task_registered_under_protocol_tech_id() -> None:
    """``ProtocolTechIdTask`` resolves by name through the registry."""
    assert "protocol_tech_id" in TASKS
    assert TASKS.get("protocol_tech_id") is ProtocolTechIdTask
    assert isinstance(get_task("protocol_tech_id"), ProtocolTechIdTask)


def test_task_declares_protocol_surface() -> None:
    """datasets/metrics/split/tracks match EVALUATION_PROTOCOL.md §protocol_tech_id."""
    task = ProtocolTechIdTask()
    assert task.name == "protocol_tech_id"
    assert task.version == "v1"
    dataset_names = [ds.name for ds in task.datasets()]
    assert dataset_names == ["tprime_wifi4"]
    assert task.default_split() == "test"
    assert task.tracks() == ["closed_set", "cross_room"]
    metric_keys = [m.primary_key for m in task.metrics()]
    assert metric_keys[0] == "accuracy_overall"  # primary is first
    assert set(metric_keys) == {"accuracy_overall", "macro_f1"}


def test_canonical_split_id_version_matches_task_version() -> None:
    """The dataset's canonical_split_id -v<N> suffix must equal the task version."""
    task = ProtocolTechIdTask()
    dataset = task.datasets()[0]
    assert dataset.canonical_split_id.endswith(f"-{task.version}")


def test_build_targets_extracts_labels() -> None:
    """``build_targets`` returns the per-sample class labels."""
    batch: Batch = {"iq": [[0.0], [0.0]], "label": [3, 1]}
    assert ProtocolTechIdTask().build_targets(batch) == [3, 1]


def test_unknown_dataset_name_rejected() -> None:
    """Constructing a :class:`ProtocolDataset` with an unknown id raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown protocol dataset"):
        ProtocolDataset("not_a_dataset")


# --------------------------------------------------------------------------------------------------
# End-to-end evaluate() on a synthetic in-memory ProtocolDataset
# --------------------------------------------------------------------------------------------------
_CHECKSUM = "sha256:" + "ab" * 32


class _ReplayModel(Model):
    """A deterministic baseline that replays each row's baked-in prediction (no torch).

    Each synthetic sample carries a ``pred`` field, so ``forward`` echoes predictions per-row
    and stays correct under any batch size (order preserved by the collate).
    """

    name = "protocol-dummy"
    family = "baseline"

    def forward(self, x: Tensor) -> Tensor:
        return list(x["pred"])

    def embed(self, x: Tensor) -> Tensor:  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return 0


def _synthetic_samples(preds: list[int]) -> list[Batch]:
    return [{"iq": [0.1, 0.2], "label": t, "pred": p} for t, p in zip(_TARGETS, preds, strict=True)]


def _task_with_samples(preds: list[int]) -> ProtocolTechIdTask:
    dataset = ProtocolDataset("tprime_wifi4", samples=_synthetic_samples(preds), checksum=_CHECKSUM)
    return ProtocolTechIdTask(datasets=[dataset])


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return schema


def test_end_to_end_evaluate_validates_against_schema() -> None:
    """``evaluate`` over a synthetic protocol dataset yields a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.FROM_SCRATCH),
        batch_size=4,  # forces multi-batch streaming over the 6 samples
    )
    Draft202012Validator(_load_schema()).validate(result)

    assert result["task"]["name"] == "protocol_tech_id"
    assert result["metrics"]["primary"] == "accuracy_overall"
    values = result["metrics"]["values"]
    assert "accuracy_overall" in values
    assert values["accuracy_overall"] == pytest.approx(_EXPECTED_ACC)
    assert values["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)


def test_end_to_end_perfect_model_scores_one() -> None:
    """A perfect model yields accuracy_overall == 1.0 through the full harness."""
    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_TARGETS),
        "test",
        RegimeSpec(Regime.FULL_FINETUNE),
        batch_size=2,
    )
    assert result["metrics"]["values"]["accuracy_overall"] == pytest.approx(1.0)
    assert math.isclose(result["metrics"]["values"]["macro_f1"], 1.0)


def test_end_to_end_writes_valid_json(tmp_path: Path) -> None:
    """With ``out_path`` set the on-disk result re-validates against the schema."""
    from jsonschema import Draft202012Validator

    out_path = tmp_path / "protocol_tech_id" / "result.json"
    result = evaluate(
        _ReplayModel(),
        _task_with_samples(_PREDS),
        "test",
        RegimeSpec(Regime.LINEAR_PROBE),
        batch_size=3,
        out_path=out_path,
    )
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


# --- on-disk loader: recording-level split, windowing + index alignment (regression guards) ---
# The real DS 3.0 captures are LONG recordings nested one level under per-room dirs
# (<root>/RM_*/802.11x/*.bin). The split partitions RECORDINGS (one label per capture); the array
# loader tiles each recording into fixed-length windows AFTER the split, so windows never leak
# across train/test. These guard the enumeration order, the windowing geometry, the leak-free
# split (all pure-stdlib, run in CI) and -- numpy-guarded -- the real tiling of a .bin capture.


def _touch(path: Path) -> None:
    """Create an empty capture file (and its parents) for the pure-stdlib enumeration tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_iter_recording_files_walks_per_room_nesting(tmp_path: Path) -> None:
    """The recording enumeration descends into RM_*/802.11x/ (and a flattened class dir alike)."""
    from rfbench.data.prepare.protocol import _iter_recording_files

    root = tmp_path / "tprime_wifi4"
    _touch(root / "RM_a" / "802.11b" / "c2.bin")
    _touch(root / "RM_a" / "802.11b" / "c1.bin")
    _touch(root / "RM_b" / "802.11b" / "c3.bin")
    _touch(root / "RM_a" / "802.11g" / "g1.bin")
    _touch(root / "802.11n" / "n1.bin")  # flattened-layout fallback still found
    _touch(root / "RM_b" / "802.11ax" / "ax1.bin")

    recordings = _iter_recording_files(root)
    classes = [cls for _path, cls in recordings]
    # Class-major order (b, g, n, ax); every capture found across rooms AND the flattened dir.
    assert classes == ["802.11b", "802.11b", "802.11b", "802.11g", "802.11n", "802.11ax"]
    # Within a class the flatten is deterministic sorted-path order.
    b_paths = [path for path, cls in recordings if cls == "802.11b"]
    assert b_paths == sorted(b_paths)
    assert len(recordings) == 6


def test_load_labels_is_one_per_recording_and_aligned(tmp_path: Path) -> None:
    """``load`` labels are one-per-recording, in the SAME order the array loader enumerates."""
    from rfbench.data.prepare.protocol import _iter_recording_files, _load_tprime_wifi4_labels

    root = tmp_path / "tprime_wifi4"
    _touch(root / "RM_a" / "802.11b" / "b1.bin")
    _touch(root / "RM_a" / "802.11g" / "g1.bin")
    _touch(root / "RM_b" / "802.11g" / "g2.bin")

    labels = _load_tprime_wifi4_labels(tmp_path)
    assert labels == [cls for _path, cls in _iter_recording_files(root)]
    assert labels == ["802.11b", "802.11g", "802.11g"]  # one label per capture, class order


def test_window_offsets_are_deterministic_and_in_bounds() -> None:
    """The tiling offsets are deterministic, evenly spread, capped, and in-bounds."""
    from rfbench.data.prepare.protocol import _window_offsets

    offs = _window_offsets(198080, 1536, 32)  # a real-sized recording
    assert len(offs) == 32
    assert offs == sorted(offs)
    assert offs[0] == 0 and offs[-1] == 198080 - 1536
    assert all(0 <= o <= 198080 - 1536 for o in offs)
    assert _window_offsets(198080, 1536, 32) == offs  # deterministic across calls
    # Capped at the non-overlapping capacity when fewer than max_windows fit.
    assert len(_window_offsets(5 * 1536, 1536, 32)) == 5
    # A capture shorter than (or equal to) one window collapses to a single (padded) window.
    assert _window_offsets(1000, 1536, 32) == [0]
    assert _window_offsets(1536, 1536, 32) == [0]


def test_prepare_split_is_recording_level_and_leak_free(tmp_path: Path) -> None:
    """The split partitions RECORDINGS into disjoint train/val/test -> no window can leak."""
    from rfbench.data.prepare.protocol import (
        CANONICAL_SPLIT_IDS,
        PROTOCOL_CLASSES,
        prepare_protocol,
    )

    labels = [cls for cls in PROTOCOL_CLASSES for _ in range(25)]  # 100 recordings, 25 per class
    split, _manifest = prepare_protocol("tprime_wifi4", out_dir=tmp_path, labels=labels, seed=42)
    idx_path = (
        tmp_path / "splits" / "tprime_wifi4" / f"{CANONICAL_SPLIT_IDS['tprime_wifi4']}.idx.json"
    )
    assert idx_path.is_file()  # versioned split index written under out_dir

    train = set(split.indices["train"])
    val = set(split.indices["val"])
    test = set(split.indices["test"])
    # Recording indices, disjoint across splits: since items ARE recordings, no window straddles.
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(range(len(labels)))
    assert len(train) + len(val) + len(test) == 100
    assert len(train) >= 70 and len(val) >= 1 and len(test) >= 1  # ~80/10/10, every split covered


def test_read_windows_tiles_a_real_capture(tmp_path: Path) -> None:
    """A ``.bin`` capture is read as complex128 and tiled into (2, win) float32 windows.

    numpy-guarded: skips in the dep-free CI venv, runs on the cluster [data] venv (and on the
    ARM validation). Exercises the real byte-reading path the synthetic ``samples=`` tests skip.
    """
    np = pytest.importorskip("numpy")

    from rfbench.data.prepare.protocol import _window_offsets
    from rfbench.tasks.protocol_tech_id.dataset import _read_windows

    n, win, k = 100, 8, 3
    data = (np.arange(n) + 1j * np.arange(n, 2 * n)).astype(np.complex128)
    cap = tmp_path / "cap.bin"
    data.tofile(cap)

    windows = _read_windows(cap, win, k)
    offsets = _window_offsets(n, win, k)
    assert len(windows) == len(offsets) == 3
    for window, offset in zip(windows, offsets, strict=True):
        assert window.shape == (2, win)
        assert window.dtype == np.float32
        assert np.allclose(window[0], data.real[offset : offset + win])  # I on row 0
        assert np.allclose(window[1], data.imag[offset : offset + win])  # Q on row 1

    # A capture shorter than one window yields a single zero-padded window.
    short = tmp_path / "short.bin"
    (np.arange(3) + 1j * np.arange(3)).astype(np.complex128).tofile(short)
    short_windows = _read_windows(short, win, k)
    assert len(short_windows) == 1
    assert short_windows[0].shape == (2, win)
    assert np.allclose(short_windows[0][:, 3:], 0.0)  # zero-padded tail beyond the 3 real samples


# --- cross_room track: leave-one-location-out grouped split (pure-stdlib guards) ---------------


def test_recording_location_groups_per_day_subcollections() -> None:
    """A capture's location is the RM_<id> prefix of its per-day sub-collection dir."""
    from rfbench.data.prepare.protocol import _recording_location

    assert _recording_location(Path("/x/tprime_wifi4/RM_573C_2/802.11b/f.bin")) == "RM_573C"
    assert _recording_location(Path("/x/tprime_wifi4/RM_142_1/802.11ax/g.bin")) == "RM_142"
    assert _recording_location(Path("/x/tprime_wifi4/RM_572C_2/802.11n/h.bin")) == "RM_572C"


def test_crossroom_split_is_leave_one_location_out_and_leak_free(tmp_path: Path) -> None:
    """The cross_room split holds a whole LOCATION out as test -> no per-location leakage."""
    from rfbench.data.prepare.protocol import (
        CROSSROOM_LOCATIONS,
        CROSSROOM_SPLIT_IDS,
        PROTOCOL_CLASSES,
        prepare_crossroom,
    )

    labels: list[str] = []
    locations: list[str] = []
    for loc in CROSSROOM_LOCATIONS:
        for cls in PROTOCOL_CLASSES:
            for _ in range(10):  # 10 recordings per (location, class)
                labels.append(cls)
                locations.append(loc)

    held = "RM_573C"
    split, _manifest = prepare_crossroom(
        held, out_dir=tmp_path, labels=labels, locations=locations, seed=42
    )
    train = set(split.indices["train"])
    val = set(split.indices["val"])
    test = set(split.indices["test"])

    # test == exactly the held-out location's recordings (the whole location, unseen in training).
    assert test == {i for i, loc in enumerate(locations) if loc == held}
    # disjoint + full coverage.
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(range(len(labels)))
    # NO train/val recording comes from the held-out location -> leak-free by location.
    assert all(locations[i] != held for i in train | val)
    # every class is represented in the held-out test location.
    assert {labels[i] for i in test} == set(PROTOCOL_CLASSES)
    # the versioned split index was written under its per-held-out-location canonical id.
    idx_path = tmp_path / "splits" / "tprime_wifi4" / f"{CROSSROOM_SPLIT_IDS[held]}.idx.json"
    assert idx_path.is_file()

    # deterministic: same seed -> identical partition.
    split2, _m2 = prepare_crossroom(
        held, out_dir=tmp_path / "b", labels=labels, locations=locations, seed=42
    )
    assert split2.indices == split.indices
