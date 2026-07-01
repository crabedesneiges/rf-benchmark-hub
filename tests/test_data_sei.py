"""WP-12 acceptance tests for the SEI data layer.

Pure stdlib: no numpy/h5py, no network. The download/loader functions are DEFINED but never
called here; only the split-GENERATION path is exercised, fed synthetic
``(tx_id, rx_id, day_id)`` record tuples so it runs without any heavy dependency.

Covers, per WP-12 acceptance -- the three WiSig conditions generated *separately* with the
correct grouping semantics:
  * ``closed_set`` -> stratified 80/10/10 by transmitter, same tx in every partition;
  * ``cross_receiver`` -> grouped by receiver, **test receivers disjoint from train
    receivers** (no receiver leakage), every item covered exactly once;
  * ``cross_day`` -> grouped by day, **test days disjoint from train days**;
  * each condition writes its own ``<id>.idx.json`` + ``<id>.manifest.json`` under
    ``$RFBENCH_CACHE`` (== tmp_path) with a *distinct* canonical split id;
  * deterministic across runs; grouped conditions require the group field (raise on None);
  * ORACLE closed-set path; argument-contract guards.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rfbench.core.splits import split_checksum
from rfbench.core.types import SplitName
from rfbench.data.prepare.sei import (
    CANONICAL_SPLIT_IDS,
    SeiRecord,
    prepare_sei,
)

_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


# --- synthetic fixtures (pure stdlib; no numpy) -------------------------------------


def _wisig_records(
    txs: tuple[int, ...],
    rxs: tuple[int, ...],
    days: tuple[int, ...],
    per_cell: int,
) -> list[SeiRecord]:
    """Synthetic WiSig records: ``per_cell`` signals for each (tx, rx, day) combination."""
    return [(tx, rx, day) for tx in txs for rx in rxs for day in days for _ in range(per_cell)]


def _mixed_records(per_cell: int = 1) -> list[SeiRecord]:
    """A shared multi-condition fixture: 3 tx x 10 rx x 10 days (grouped 8/1/1, 16/2/2)."""
    return _wisig_records(
        (1, 2, 3), rxs=tuple(range(100, 110)), days=tuple(range(10)), per_cell=per_cell
    )


def _all_indices(indices: dict[SplitName, list[int]]) -> list[int]:
    return sorted(i for name in _SPLITS for i in indices[name])


def _assert_partition(indices: dict[SplitName, list[int]], n_items: int) -> None:
    """Full coverage + no leakage across the three partitions."""
    assert _all_indices(indices) == list(range(n_items))
    assert not (set(indices["train"]) & set(indices["val"]))
    assert not (set(indices["train"]) & set(indices["test"]))
    assert not (set(indices["val"]) & set(indices["test"]))


# --- closed_set: stratified by transmitter ------------------------------------------


def test_prepare_sei_closed_set_stratified_by_tx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """closed_set -> 80/10/10 stratified by tx; same transmitters in every partition."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    txs = (1, 2, 3, 4)
    records = _wisig_records(txs, rxs=(10, 11), days=(0,), per_cell=5)
    # 4 tx x 2 rx x 1 day x 5 = 40 items; per tx = 10 -> 8/1/1.

    split, manifest = prepare_sei("wisig", "closed_set", out_dir=str(tmp_path), records=records)

    sizes = {name: len(split.indices[name]) for name in _SPLITS}
    assert sizes == {"train": 32, "val": 4, "test": 4}

    tx_of = {i: records[i][0] for i in range(len(records))}
    # Every transmitter appears in all three partitions (closed set).
    for name in _SPLITS:
        seen_tx = {tx_of[i] for i in split.indices[name]}
        assert seen_tx == set(txs), name
    # Each tx contributes exactly 8/1/1.
    expected_per_split: tuple[tuple[SplitName, int], ...] = (("train", 8), ("val", 1), ("test", 1))
    for name, expected in expected_per_split:
        counts = Counter(tx_of[i] for i in split.indices[name])
        for tx in txs:
            assert counts[tx] == expected, (name, tx)

    _assert_partition(split.indices, len(records))
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["wisig"]["closed_set"]
    assert manifest.dataset == "wisig"
    assert manifest.n_items == len(records)
    assert manifest.seed == 42


# --- cross_receiver: grouped by receiver --------------------------------------------


def test_prepare_sei_cross_receiver_groups_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cross_receiver -> whole receivers assigned per partition; test rx disjoint train rx."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    txs = (1, 2, 3)
    rxs = tuple(range(100, 110))  # 10 receivers -> 8/1/1 at the receiver level
    records = _wisig_records(txs, rxs, days=(0, 1), per_cell=2)

    split, _ = prepare_sei("wisig", "cross_receiver", out_dir=str(tmp_path), records=records)

    rx_of = {i: records[i][1] for i in range(len(records))}
    rx_by_split = {name: {rx_of[i] for i in split.indices[name]} for name in _SPLITS}

    # Receiver groups partitioned 8/1/1, disjoint across splits (no receiver leakage).
    assert len(rx_by_split["train"]) == 8
    assert len(rx_by_split["val"]) == 1
    assert len(rx_by_split["test"]) == 1
    assert not (rx_by_split["train"] & rx_by_split["test"])
    assert not (rx_by_split["train"] & rx_by_split["val"])
    assert not (rx_by_split["val"] & rx_by_split["test"])
    assert rx_by_split["train"] | rx_by_split["val"] | rx_by_split["test"] == set(rxs)

    # Transmitters are still shared -- the task is "same emitter, unseen receiver".
    tx_of = {i: records[i][0] for i in range(len(records))}
    for name in _SPLITS:
        assert {tx_of[i] for i in split.indices[name]} == set(txs), name

    _assert_partition(split.indices, len(records))
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["wisig"]["cross_receiver"]


# --- cross_day: grouped by day ------------------------------------------------------


def test_prepare_sei_cross_day_groups_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cross_day -> whole days assigned per partition; test days disjoint from train days."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    txs = (1, 2)
    rxs = (100, 101)
    days = tuple(range(20))  # 20 days -> 16/2/2 at the day level
    records = _wisig_records(txs, rxs, days, per_cell=1)

    split, _ = prepare_sei("wisig", "cross_day", out_dir=str(tmp_path), records=records)

    day_of = {i: records[i][2] for i in range(len(records))}
    day_by_split = {name: {day_of[i] for i in split.indices[name]} for name in _SPLITS}

    assert len(day_by_split["train"]) == 16
    assert len(day_by_split["val"]) == 2
    assert len(day_by_split["test"]) == 2
    assert not (day_by_split["train"] & day_by_split["test"])
    assert not (day_by_split["train"] & day_by_split["val"])
    assert not (day_by_split["val"] & day_by_split["test"])
    assert day_by_split["train"] | day_by_split["val"] | day_by_split["test"] == set(days)

    _assert_partition(split.indices, len(records))
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["wisig"]["cross_day"]


# --- three conditions are distinct + all written -------------------------------------


def test_prepare_sei_three_conditions_distinct_and_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three WiSig conditions each write their own idx+manifest with a distinct id."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records = _mixed_records()

    ids: dict[str, str] = {}
    for condition in ("closed_set", "cross_receiver", "cross_day"):
        split, manifest = prepare_sei("wisig", condition, out_dir=str(tmp_path), records=records)
        ids[condition] = split.canonical_split_id
        split_id = CANONICAL_SPLIT_IDS["wisig"][condition]
        idx_path = tmp_path / "splits" / "wisig" / f"{split_id}.idx.json"
        man_path = tmp_path / "splits" / "wisig" / f"{split_id}.manifest.json"
        assert idx_path.exists(), condition
        assert man_path.exists(), condition
        assert manifest.split_checksum == split.checksum

    # All three ids are distinct.
    assert len(set(ids.values())) == 3


def test_prepare_sei_idx_provenance_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """closed_set is a generated split; grouped conditions are written as adopted partitions."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records = _wisig_records((1, 2), rxs=(100, 101, 102, 103, 104), days=(0,), per_cell=1)

    prepare_sei("wisig", "closed_set", out_dir=str(tmp_path), records=records)
    prepare_sei("wisig", "cross_receiver", out_dir=str(tmp_path), records=records)

    closed_id = CANONICAL_SPLIT_IDS["wisig"]["closed_set"]
    grouped_id = CANONICAL_SPLIT_IDS["wisig"]["cross_receiver"]
    closed_doc = json.loads(
        (tmp_path / "splits" / "wisig" / f"{closed_id}.idx.json").read_text(encoding="utf-8")
    )
    grouped_doc = json.loads(
        (tmp_path / "splits" / "wisig" / f"{grouped_id}.idx.json").read_text(encoding="utf-8")
    )
    assert closed_doc["provenance"] == "generated"
    assert closed_doc["stratify_key"] == "stratum"
    # Grouped partitions are pre-computed then adopted verbatim.
    assert grouped_doc["provenance"] == "official"


# --- determinism --------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["closed_set", "cross_receiver", "cross_day"])
def test_prepare_sei_is_deterministic(
    condition: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs (different out_dirs) produce byte-identical indices + checksum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records = _mixed_records()

    a, _ = prepare_sei("wisig", condition, out_dir=str(tmp_path / "a"), records=records)
    b, _ = prepare_sei("wisig", condition, out_dir=str(tmp_path / "b"), records=records)
    assert a.indices == b.indices
    assert a.checksum == b.checksum


def test_prepare_sei_checksum_matches_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-disk index checksum equals the returned split checksum for every condition."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records = _mixed_records()

    for condition in ("closed_set", "cross_receiver", "cross_day"):
        split, _ = prepare_sei("wisig", condition, out_dir=str(tmp_path), records=records)
        split_id = CANONICAL_SPLIT_IDS["wisig"][condition]
        idx_path = tmp_path / "splits" / "wisig" / f"{split_id}.idx.json"
        assert split_checksum(str(idx_path)) == split.checksum, condition


# --- ORACLE closed-set ---------------------------------------------------------------


def test_prepare_sei_oracle_closed_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ORACLE -> closed_set stratified by tx; rx/day absent (None) is fine here."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records: list[SeiRecord] = [(tx, None, None) for tx in range(16) for _ in range(10)]

    split, manifest = prepare_sei("oracle", "closed_set", out_dir=str(tmp_path), records=records)
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["oracle"]["closed_set"]
    assert manifest.n_items == 160
    tx_of = {i: records[i][0] for i in range(len(records))}
    expected_per_split: tuple[tuple[SplitName, int], ...] = (("train", 8), ("val", 1), ("test", 1))
    for name, expected in expected_per_split:
        counts = Counter(tx_of[i] for i in split.indices[name])
        assert all(counts[tx] == expected for tx in range(16)), name
    _assert_partition(split.indices, len(records))


# --- source checksums + string group ids ---------------------------------------------


def test_prepare_sei_passes_source_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-file integrity hashes are recorded in the manifest (never the data itself)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records = _wisig_records((1, 2), rxs=(100, 101, 102, 103, 104), days=(0,), per_cell=2)
    src = {"ManyTx.pkl": "sha256:" + "cd" * 32}
    _, manifest = prepare_sei(
        "wisig", "cross_receiver", out_dir=str(tmp_path), records=records, source_checksums=src
    )
    assert dict(manifest.source_checksums) == src


def test_prepare_sei_cross_receiver_string_group_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouped conditions accept string receiver ids and still partition disjointly."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    rxs = tuple(f"usrp-{k}" for k in range(10))
    records: list[SeiRecord] = [(tx, rx, 0) for tx in (1, 2) for rx in rxs for _ in range(2)]

    split, _ = prepare_sei("wisig", "cross_receiver", out_dir=str(tmp_path), records=records)
    rx_of = {i: records[i][1] for i in range(len(records))}
    rx_by_split = {name: {rx_of[i] for i in split.indices[name]} for name in _SPLITS}
    assert not (rx_by_split["train"] & rx_by_split["test"])
    assert rx_by_split["train"] | rx_by_split["val"] | rx_by_split["test"] == set(rxs)


# --- argument-contract guards --------------------------------------------------------


def test_prepare_sei_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown SEI dataset"):
        prepare_sei("lora_rffi", "closed_set", out_dir=str(tmp_path), records=[(1, 2, 3)])


def test_prepare_sei_unsupported_condition_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support condition"):
        prepare_sei("oracle", "cross_receiver", out_dir=str(tmp_path), records=[(1, 2, 3)])


def test_prepare_sei_cross_receiver_requires_rx(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partitions by receiver"):
        prepare_sei(
            "wisig", "cross_receiver", out_dir=str(tmp_path), records=[(1, None, 0), (2, None, 0)]
        )


def test_prepare_sei_cross_day_requires_day(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partitions by day"):
        prepare_sei(
            "wisig", "cross_day", out_dir=str(tmp_path), records=[(1, 100, None), (2, 101, None)]
        )
