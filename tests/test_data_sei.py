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
    _powder_ids,
    extract_lora_records,
    extract_powder_records,
    extract_wisig_records,
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


def test_prepare_sei_cross_day_few_days_has_nonempty_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Few groups (real WiSig has ~4 days) must still yield a non-empty held-out test.

    Regression guard: plain 80/10/10 largest-remainder on 4 groups gives [3, 1, 0] -> an
    EMPTY test day (useless for a cross-day protocol). The grouped splitter now guarantees
    >= 1 group in val AND test whenever there are >= 3 groups.
    """
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    days = (0, 1, 2, 3)  # only 4 capture days, like real WiSig ManyTx
    records = _wisig_records((1, 2), rxs=(100, 101), days=days, per_cell=3)

    split, _ = prepare_sei("wisig", "cross_day", out_dir=str(tmp_path), records=records)

    day_of = {i: records[i][2] for i in range(len(records))}
    day_by_split = {name: {day_of[i] for i in split.indices[name]} for name in _SPLITS}
    assert len(day_by_split["test"]) >= 1  # <-- the fix: never empty
    assert len(day_by_split["val"]) >= 1
    assert len(day_by_split["train"]) >= 1
    assert not (day_by_split["train"] & day_by_split["test"])
    assert not (day_by_split["train"] & day_by_split["val"])
    assert not (day_by_split["val"] & day_by_split["test"])
    assert split.indices["test"], "held-out test partition must contain items"
    _assert_partition(split.indices, len(records))


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
        prepare_sei("nonexistent_dataset", "closed_set", out_dir=str(tmp_path), records=[(1, 2, 3)])


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


# --- LoRa closed-set (device labels) -------------------------------------------------


def test_prepare_sei_lora_closed_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LoRa -> closed_set stratified by device id; rx/day absent (None) is fine here."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    records: list[SeiRecord] = [(dev, None, None) for dev in range(1, 31) for _ in range(10)]

    split, manifest = prepare_sei("lora", "closed_set", out_dir=str(tmp_path), records=records)
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["lora"]["closed_set"]
    assert manifest.dataset == "lora"
    assert manifest.n_items == 300
    dev_of = {i: records[i][0] for i in range(len(records))}
    expected_per_split: tuple[tuple[SplitName, int], ...] = (("train", 8), ("val", 1), ("test", 1))
    for name, expected in expected_per_split:
        counts = Counter(dev_of[i] for i in split.indices[name])
        assert all(counts[dev] == expected for dev in range(1, 31)), name
    _assert_partition(split.indices, len(records))


def test_prepare_sei_lora_no_cross_conditions(tmp_path: Path) -> None:
    """LoRa is single-condition closed-set only (no receiver/day metadata)."""
    with pytest.raises(ValueError, match="does not support condition"):
        prepare_sei("lora", "cross_receiver", out_dir=str(tmp_path), records=[(1, None, None)])


# --- pure-Python label extraction on synthetic record layouts ------------------------


def _fake_wisig_dataset(
    txs: tuple[str, ...],
    rxs: tuple[str, ...],
    days: tuple[str, ...],
    per_cell: int,
    *,
    equalized_list: tuple[int, ...] = (0,),
) -> dict[str, object]:
    """Pure-stdlib mimic of the real WiSig ManyTx compact pickle dict.

    Mirrors the ``wisig-examples`` layout: axis label lists + a 5-level nested ``data``
    list indexed ``data[tx_i][rx_i][day_i][eq_i]`` whose leaf is a per-signal block. The
    real leaf is a ``(n, 256, 2)`` ndarray; here it is a plain list of ``per_cell`` rows so
    the extraction runs with no numpy (``extract_wisig_records`` only needs ``len(block)``).
    """
    n_tx, n_rx, n_day, n_eq = len(txs), len(rxs), len(days), len(equalized_list)

    def _cell() -> list[list[float]]:
        return [[0.0, 0.0] for _ in range(per_cell)]

    data = [
        [[[_cell() for _ in range(n_eq)] for _ in range(n_day)] for _ in range(n_rx)]
        for _ in range(n_tx)
    ]
    return {
        "tx_list": list(txs),
        "rx_list": list(rxs),
        "capture_date_list": list(days),
        "equalized_list": list(equalized_list),
        "data": data,
    }


def test_extract_wisig_records_flattens_tx_rx_day() -> None:
    """extract_wisig_records emits one (tx, rx, day) per signal, in tx/rx/day order."""
    txs = ("A", "B")
    rxs = ("r1", "r2", "r3")
    days = ("d0", "d1")
    per_cell = 4
    dataset = _fake_wisig_dataset(txs, rxs, days, per_cell)

    records = extract_wisig_records(dataset)

    assert len(records) == len(txs) * len(rxs) * len(days) * per_cell
    # First cell is (tx=A, rx=r1, day=d0) repeated per_cell times.
    assert records[:per_cell] == [("A", "r1", "d0")] * per_cell
    # Group ids round-trip: distinct tx/rx/day sets match the axis labels.
    assert {r[0] for r in records} == set(txs)
    assert {r[1] for r in records} == set(rxs)
    assert {r[2] for r in records} == set(days)


def test_extract_wisig_records_feeds_prepare_sei(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records extracted from the fake ManyTx dict drive all three WiSig conditions."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    dataset = _fake_wisig_dataset(
        ("A", "B", "C"),
        tuple(f"r{k}" for k in range(10)),
        tuple(f"d{k}" for k in range(10)),
        per_cell=1,
    )
    records = extract_wisig_records(dataset)

    for condition in ("closed_set", "cross_receiver", "cross_day"):
        split, _ = prepare_sei("wisig", condition, out_dir=str(tmp_path), records=records)
        _assert_partition(split.indices, len(records))


def test_extract_wisig_records_selects_equalized_slot() -> None:
    """The equalized= slot selects the matching eq_i axis; unknown values raise."""
    dataset = _fake_wisig_dataset(("A",), ("r1",), ("d0",), per_cell=3, equalized_list=(0, 1))
    # eq slot 1 is empty in this fixture (only slot 0 populated) -> different length.
    # Populate slot 1 explicitly to make the selection observable.
    data = dataset["data"]
    assert isinstance(data, list)
    data[0][0][0][1] = [[0.0, 0.0]] * 5  # eq=1 has 5 signals

    recs_eq0 = extract_wisig_records(dataset, equalized=0)
    recs_eq1 = extract_wisig_records(dataset, equalized=1)
    assert len(recs_eq0) == 3
    assert len(recs_eq1) == 5

    with pytest.raises(ValueError, match="equalized"):
        extract_wisig_records(dataset, equalized=9)


def test_extract_wisig_records_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="tx_list"):
        extract_wisig_records({"rx_list": [], "capture_date_list": [], "equalized_list": [0]})


def test_extract_lora_records_maps_device_labels() -> None:
    """extract_lora_records turns a flat device-id label vector into closed-set records."""
    labels = [1, 1, 2, 2, 2, 3]
    records = extract_lora_records(labels)
    assert records == [
        (1, None, None),
        (1, None, None),
        (2, None, None),
        (2, None, None),
        (2, None, None),
        (3, None, None),
    ]
    assert {r[0] for r in records} == {1, 2, 3}


# --- heavy-format parser tests (SKIP without numpy/h5py, RUN on the [data] venv) ------


def test_load_wisig_records_reads_real_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_wisig_records round-trips a REAL pickle in the ManyTx layout (needs numpy).

    Skipped in the dependency-free venv (the loader imports numpy lazily); runs on the
    cluster [data] venv. The fixture is a genuine pickle whose leaves are numpy arrays of
    the real ``(n, 256, 2)`` shape, so this exercises the actual on-disk parse path.
    """
    np = pytest.importorskip("numpy")
    import pickle

    from rfbench.data.prepare.sei import load_wisig_records

    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    txs, rxs, days, per_cell = ("A", "B"), ("r1", "r2"), ("d0",), 3
    dataset = _fake_wisig_dataset(txs, rxs, days, per_cell)
    # Replace stdlib leaves with real (n, 256, 2) float32 arrays.
    data = dataset["data"]
    assert isinstance(data, list)
    for tx_i in range(len(txs)):
        for rx_i in range(len(rxs)):
            for day_i in range(len(days)):
                data[tx_i][rx_i][day_i][0] = np.zeros((per_cell, 256, 2), dtype=np.float32)

    wisig_dir = tmp_path / "wisig"
    wisig_dir.mkdir(parents=True)
    with (wisig_dir / "ManyTx.pkl").open("wb") as fh:
        pickle.dump(dataset, fh)

    records = load_wisig_records(cache=str(tmp_path))
    assert len(records) == len(txs) * len(rxs) * len(days) * per_cell
    assert records[0] == ("A", "r1", "d0")


def test_load_lora_records_reads_real_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_lora_records round-trips a REAL HDF5 in the LoRa layout (needs numpy + h5py).

    Skipped in the dependency-free venv; runs on the cluster [data] venv. Writes a genuine
    HDF5 with a ``(1, N)`` 1-indexed ``label`` row and a matching ``data`` matrix, then
    checks the loader recovers one record per packet with the right device ids.
    """
    np = pytest.importorskip("numpy")
    h5py = pytest.importorskip("h5py")

    from rfbench.data.prepare.sei import load_lora_records

    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir(parents=True)
    # 1-indexed device labels for 6 packets across 3 devices; data is (N, 2*L).
    labels = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.int64)  # shape (1, N)
    data = np.zeros((6, 8), dtype=np.float32)
    with h5py.File(lora_dir / "dataset_training_aug.h5", "w") as fh:
        fh.create_dataset("label", data=labels)
        fh.create_dataset("data", data=data)

    records = load_lora_records(cache=str(tmp_path))
    assert len(records) == 6
    assert [r[0] for r in records] == [1, 1, 2, 2, 3, 3]
    assert all(r[1] is None and r[2] is None for r in records)


# --- POWDER (4-BS WiFi) closed-set prepare + record extraction (pure stdlib) ----------


def test_powder_ids_parses_device_and_day() -> None:
    """_powder_ids extracts (device, day) from [Waveform]_[Day]_[TransmitterBS]_[Set] names."""
    assert _powder_ids("WiFi_Day1_MEB_1.sigmf-data") == ("MEB", "Day1")
    assert _powder_ids("WiFi_Day2_Browning_3.sigmf-data") == ("Browning", "Day2")
    # A deviating name falls back to (stem, unknown_day) rather than raising.
    assert _powder_ids("weird.sigmf-data") == ("weird", "unknown_day")


def test_extract_powder_records_one_per_frame() -> None:
    """extract_powder_records emits one (device, None, day) record per frame, in file order."""
    frame_counts = [("MEB", "Day1", 2), ("Honors", "Day1", 3), ("MEB", "Day2", 1)]
    records = extract_powder_records(frame_counts)
    assert len(records) == 6
    assert [r[0] for r in records] == ["MEB", "MEB", "Honors", "Honors", "Honors", "MEB"]
    assert all(r[1] is None for r in records)  # single fixed receiver
    assert [r[2] for r in records] == ["Day1", "Day1", "Day1", "Day1", "Day1", "Day2"]


def test_powder_closed_set_prepare(tmp_path: Path) -> None:
    """POWDER closed_set: 80/10/10 stratified by device, its own canonical split id + sidecars."""
    # 4 devices x 30 frames each (day-pooled closed set, like the FM evaluators).
    records: list[SeiRecord] = [
        (dev, None, day)
        for dev in ("MEB", "Browning", "Behavioral", "Honors")
        for day in ("Day1", "Day2")
        for _ in range(15)
    ]
    split, manifest = prepare_sei(
        "powder", "closed_set", out_dir=tmp_path, records=records, seed=42
    )
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["powder"]["closed_set"]
    assert manifest.dataset == "powder"
    total = sum(len(split.indices[s]) for s in _SPLITS)
    assert total == len(records)  # every frame covered exactly once
    # Each device appears in every partition (closed-set identity, stratified by device).
    idx_file = tmp_path / "splits" / "powder" / f"{split.canonical_split_id}.idx.json"
    assert idx_file.is_file()
    for partition in _SPLITS:
        devices = {records[i][0] for i in split.indices[partition]}
        assert devices == {"MEB", "Browning", "Behavioral", "Honors"}


def test_powder_supports_only_closed_set(tmp_path: Path) -> None:
    """POWDER (single receiver) rejects the cross_receiver / cross_day conditions."""
    records: list[SeiRecord] = [("MEB", None, "Day1"), ("Honors", None, "Day1")]
    with pytest.raises(ValueError, match="does not support condition"):
        prepare_sei("powder", "cross_receiver", out_dir=tmp_path, records=records)
