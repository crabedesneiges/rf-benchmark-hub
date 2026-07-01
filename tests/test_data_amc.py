"""WP-11 acceptance tests for the AMC data layer.

Pure stdlib: no numpy/h5py/torchsig, no network. The download/generation functions are
DEFINED but never called here; only the split-GENERATION path is exercised, fed synthetic
``(modulation, snr_db)`` label tuples so it runs without any heavy dependency.

Covers, per WP-11 acceptance:
  * ``prepare_amc`` on synthetic RadioML labels -> stratified 80/10/10 indices, deterministic,
    no overlap, full coverage, per-stratum (mod x snr) proportions respected;
  * writes ``<id>.idx.json`` + ``<id>.manifest.json`` under ``$RFBENCH_CACHE`` (== tmp_path);
  * checksum stable across runs and matches the on-disk index;
  * Sig53 adopts the official split verbatim (no shuffling);
  * ``resolve_cache_dir`` honours ``$RFBENCH_CACHE`` with a documented fallback.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rfbench.core.splits import split_checksum
from rfbench.core.types import SplitName
from rfbench.data.prepare._common import (
    encode_strata,
    manifest_checksum,
    prepare_from_labels,
    resolve_cache_dir,
)
from rfbench.data.prepare.amc import (
    _RADIOML2018_MODS,
    CANONICAL_SPLIT_IDS,
    _expand_radioml2016_table,
    _load_radioml2016_labels,
    _radioml2018_labels_from_arrays,
    load_radioml_labels,
    load_sig53_official_split,
    prepare_amc,
)

_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


# --- synthetic fixtures (pure stdlib; no numpy) -------------------------------------


def _radioml_labels(
    mods: tuple[str, ...],
    snrs: tuple[int, ...],
    per_cell: int,
) -> list[tuple[str, int]]:
    """A synthetic RadioML label list: ``per_cell`` items for each (modulation, snr) cell."""
    return [(mod, snr) for mod in mods for snr in snrs for _ in range(per_cell)]


def _all_indices(indices: dict[SplitName, list[int]]) -> list[int]:
    return sorted(i for name in _SPLITS for i in indices[name])


# --- resolve_cache_dir --------------------------------------------------------------


def test_resolve_cache_dir_honours_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``$RFBENCH_CACHE`` wins over the fallback."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path / "cache"))
    assert resolve_cache_dir() == tmp_path / "cache"


def test_resolve_cache_dir_explicit_arg_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit argument overrides the environment variable."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path / "env"))
    assert resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_resolve_cache_dir_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env and no arg, the documented ``./.rfbench_cache`` fallback is used."""
    monkeypatch.delenv("RFBENCH_CACHE", raising=False)
    assert resolve_cache_dir() == Path.cwd() / ".rfbench_cache"


# --- encode_strata ------------------------------------------------------------------


def test_encode_strata_is_deterministic_and_order_independent() -> None:
    """Composite (mod, snr) keys map to dense ids independent of first-appearance order."""
    a = encode_strata([("QPSK", 0), ("BPSK", -4), ("QPSK", 0), ("BPSK", 4)])
    b = encode_strata([("BPSK", 4), ("QPSK", 0), ("BPSK", -4), ("QPSK", 0)])
    # Same key -> same code within a call; codes are assigned in sorted-key order.
    assert a[0] == a[2]  # both ("QPSK", 0)
    assert len(set(a)) == 3
    # Order-independent mapping: the code assigned to a given key is stable across calls.
    code_a = dict(zip([("QPSK", 0), ("BPSK", -4), ("QPSK", 0), ("BPSK", 4)], a, strict=True))
    code_b = dict(zip([("BPSK", 4), ("QPSK", 0), ("BPSK", -4), ("QPSK", 0)], b, strict=True))
    assert code_a == code_b


# --- prepare_amc: stratified RadioML path -------------------------------------------


def test_prepare_amc_radioml_stratified_80_10_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RadioML -> 80/10/10 stratified by (mod x snr); each cell split 8/1/1."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    mods = ("BPSK", "QPSK", "8PSK")
    snrs = (-4, 0, 4, 8)
    per_cell = 10  # 3 mods x 4 snrs x 10 = 120 items; each cell -> 8/1/1
    labels = _radioml_labels(mods, snrs, per_cell)

    split, manifest = prepare_amc("radioml_2016_10a", out_dir=str(tmp_path), labels=labels)

    sizes = {name: len(split.indices[name]) for name in _SPLITS}
    assert sizes == {"train": 96, "val": 12, "test": 12}  # 120 * 0.8/0.1/0.1

    # Every (mod x snr) cell contributes exactly 8/1/1 to train/val/test.
    label_of = {i: labels[i] for i in range(len(labels))}
    expected_per_split: tuple[tuple[SplitName, int], ...] = (
        ("train", 8),
        ("val", 1),
        ("test", 1),
    )
    for name, expected in expected_per_split:
        per_cell_counts = Counter(label_of[i] for i in split.indices[name])
        for mod in mods:
            for snr in snrs:
                assert per_cell_counts[(mod, snr)] == expected, (name, mod, snr)

    # No leakage, full coverage.
    assert _all_indices(split.indices) == list(range(len(labels)))
    assert not (set(split.indices["train"]) & set(split.indices["val"]))
    assert not (set(split.indices["train"]) & set(split.indices["test"]))
    assert not (set(split.indices["val"]) & set(split.indices["test"]))

    # Canonical id + provenance recorded on the manifest.
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["radioml_2016_10a"]
    assert manifest.dataset == "radioml_2016_10a"
    assert manifest.n_items == len(labels)
    assert manifest.seed == 42


def test_prepare_amc_radioml_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs (different out_dirs) produce byte-identical indices + checksum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    labels = _radioml_labels(("BPSK", "QPSK"), (-2, 0, 2), per_cell=15)

    a, _ = prepare_amc("radioml_2018_01a", out_dir=str(tmp_path / "a"), labels=labels)
    b, _ = prepare_amc("radioml_2018_01a", out_dir=str(tmp_path / "b"), labels=labels)
    assert a.indices == b.indices
    assert a.checksum == b.checksum


def test_prepare_amc_writes_idx_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare writes both an idx.json and a manifest.json under $RFBENCH_CACHE == tmp."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    cache = resolve_cache_dir()
    labels = _radioml_labels(("BPSK", "QPSK"), (0, 4), per_cell=20)

    split, manifest = prepare_amc("radioml_2016_10a", out_dir=str(cache), labels=labels)
    split_id = CANONICAL_SPLIT_IDS["radioml_2016_10a"]

    idx_path = cache / "splits" / "radioml_2016_10a" / f"{split_id}.idx.json"
    man_path = cache / "splits" / "radioml_2016_10a" / f"{split_id}.manifest.json"
    assert idx_path.exists()
    assert man_path.exists()

    # idx.json carries the generated provenance + stratify key.
    idx_doc = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx_doc["provenance"] == "generated"
    assert idx_doc["stratify_key"] == "stratum"
    assert idx_doc["checksum"] == split.checksum

    # manifest.json is consistent with the returned DatasetManifest.
    man_doc = json.loads(man_path.read_text(encoding="utf-8"))
    assert man_doc["dataset"] == "radioml_2016_10a"
    assert man_doc["canonical_split_id"] == split_id
    assert man_doc["split_checksum"] == split.checksum
    assert man_doc["n_items"] == manifest.n_items == len(labels)
    assert man_doc["source_checksums"] == {}


def test_prepare_amc_checksum_stable_and_matches_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The on-disk index checksum equals the manifest checksum, stable across recompute."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    labels = _radioml_labels(("BPSK", "QPSK", "8PSK"), (0, 8), per_cell=12)
    split, _ = prepare_amc("radioml_2016_10a", out_dir=str(tmp_path), labels=labels)

    split_id = CANONICAL_SPLIT_IDS["radioml_2016_10a"]
    idx_path = tmp_path / "splits" / "radioml_2016_10a" / f"{split_id}.idx.json"
    assert split_checksum(str(idx_path)) == split.checksum
    assert split_checksum(str(idx_path)) == split.checksum  # recompute -> identical


def test_prepare_amc_passes_source_checksums_into_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-file integrity hashes are recorded in the manifest (never the data itself)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    labels = _radioml_labels(("BPSK", "QPSK"), (0, 4), per_cell=10)
    src = {"RML2016.10a_dict.pkl": "sha256:" + "ab" * 32}
    _, manifest = prepare_amc(
        "radioml_2016_10a", out_dir=str(tmp_path), labels=labels, source_checksums=src
    )
    assert dict(manifest.source_checksums) == src


def test_manifest_checksum_is_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """manifest_checksum is a stable sha256 over the manifest content."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    labels = _radioml_labels(("BPSK", "QPSK"), (0, 4), per_cell=10)
    _, manifest = prepare_amc("radioml_2016_10a", out_dir=str(tmp_path), labels=labels)
    first = manifest_checksum(manifest)
    assert first.startswith("sha256:")
    assert manifest_checksum(manifest) == first


# --- prepare_amc: Sig53 official-split path ------------------------------------------


def test_prepare_amc_sig53_adopts_official_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sig53 adopts the official TorchSig partition verbatim (only sorted)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    official = {"train": [5, 0, 3, 6], "val": [1, 7], "test": [2, 4]}
    split, manifest = prepare_amc("sig53", out_dir=str(tmp_path), official_split=official)
    assert split.canonical_split_id == CANONICAL_SPLIT_IDS["sig53"]
    assert split.indices == {"train": [0, 3, 5, 6], "val": [1, 7], "test": [2, 4]}
    assert manifest.n_items == 8

    idx_path = tmp_path / "splits" / "sig53" / f"{CANONICAL_SPLIT_IDS['sig53']}.idx.json"
    doc = json.loads(idx_path.read_text(encoding="utf-8"))
    assert doc["provenance"] == "official"


# --- argument-contract guards -------------------------------------------------------


def test_prepare_amc_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown AMC dataset"):
        prepare_amc("radioml_9999", out_dir=str(tmp_path), labels=[("BPSK", 0)])


def test_prepare_amc_radioml_requires_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass `labels="):
        prepare_amc("radioml_2016_10a", out_dir=str(tmp_path))


def test_prepare_amc_sig53_requires_official_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass `official_split="):
        prepare_amc("sig53", out_dir=str(tmp_path))


def test_prepare_from_labels_length_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strata length"):
        prepare_from_labels(
            dataset="d",
            split_id="d-v1",
            n_items=2,
            strata=[("BPSK", 0)],
            source_url="http://example",
            out_dir=str(tmp_path),
        )


# --- real RadioML 2016.10a loader on a TINY REAL pickle fixture ----------------------


def _write_radioml2016_pickle(path: Path, table: dict[tuple[object, object], list[object]]) -> None:
    """Write a tiny REAL ``RML2016.10a_dict.pkl`` fixture with stdlib pickle (no numpy).

    Mirrors the published layout: a ``dict`` keyed by ``(modulation, snr)`` whose values are
    per-cell blocks. The real file stores ``ndarray[N, 2, 128]``; we use nested Python lists of
    the same first dimension ``N`` so ``len(block)`` yields the item count the loader expands on.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(table, fh)


def test_expand_radioml2016_table_counts_items_per_cell() -> None:
    """``_expand_radioml2016_table`` emits one (mod, snr) label per item in each cell block."""
    table: dict[tuple[object, object], list[object]] = {
        ("BPSK", -4): [[[0.0] * 128, [0.0] * 128] for _ in range(3)],  # 3 items
        ("QPSK", 8): [[[0.0] * 128, [0.0] * 128] for _ in range(2)],  # 2 items
    }
    labels = _expand_radioml2016_table(table)
    assert Counter(labels) == Counter({("BPSK", -4): 3, ("QPSK", 8): 2})
    assert len(labels) == 5
    # Values coerced to (str, int) exactly as the real loader would.
    assert all(isinstance(m, str) and isinstance(s, int) for m, s in labels)


def test_load_radioml2016_labels_reads_real_pickle_and_prepare_stratifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on a REAL stdlib pickle: load 2016 labels -> prepare_amc strata by (mod x snr).

    Builds a tiny genuine ``RML2016.10a_dict.pkl`` (stdlib pickle, nested lists mimicking the
    ``[N, 2, 128]`` blocks), reads it through the real loader, and asserts prepare_amc produces
    the 80/10/10 stratified split with each (mod, snr) cell split 8/1/1.
    """
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    mods = ("BPSK", "QPSK", "8PSK")
    snrs = (-4, 0, 4, 8)
    per_cell = 10  # 3 x 4 x 10 = 120 items -> each cell 8/1/1
    table: dict[tuple[object, object], list[object]] = {
        (mod, snr): [[[0.0] * 128, [0.0] * 128] for _ in range(per_cell)]
        for mod in mods
        for snr in snrs
    }
    pkl = resolve_cache_dir() / "radioml_2016_10a" / "RML2016.10a_dict.pkl"
    _write_radioml2016_pickle(pkl, table)

    # Real loader (stdlib pickle path) reads the fixture back into per-item labels.
    labels = _load_radioml2016_labels(resolve_cache_dir())
    assert len(labels) == 120
    assert Counter(labels)[("BPSK", -4)] == per_cell

    # Public entry point resolves the same file via $RFBENCH_CACHE.
    assert load_radioml_labels("radioml_2016_10a") == labels

    split, manifest = prepare_amc("radioml_2016_10a", out_dir=str(tmp_path), labels=labels)
    sizes = {name: len(split.indices[name]) for name in _SPLITS}
    assert sizes == {"train": 96, "val": 12, "test": 12}

    label_of = {i: labels[i] for i in range(len(labels))}
    expected_per_split: tuple[tuple[SplitName, int], ...] = (("train", 8), ("val", 1), ("test", 1))
    for name, expected in expected_per_split:
        counts = Counter(label_of[i] for i in split.indices[name])
        for mod in mods:
            for snr in snrs:
                assert counts[(mod, snr)] == expected, (name, mod, snr)
    assert manifest.n_items == 120


def test_load_radioml2016_labels_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clear FileNotFoundError points at the download step when the pickle is absent."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="run the download step first"):
        _load_radioml2016_labels(resolve_cache_dir())


# --- RadioML 2018.01a label logic (pure Python) + guarded HDF5 parser ---------------


def test_radioml2018_labels_from_arrays_maps_onehot_argmax_to_class_names() -> None:
    """One-hot argmax indices + SNRs map to the canonical 24-class names, in order."""
    # Column indices 0,3,23 -> OOK, BPSK, OQPSK per _RADIOML2018_MODS.
    labels = _radioml2018_labels_from_arrays([0, 3, 23], [-20, 0, 30])
    assert labels == [("OOK", -20), ("BPSK", 0), ("OQPSK", 30)]
    assert len(_RADIOML2018_MODS) == 24


def test_radioml2018_labels_from_arrays_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mod/snr length mismatch"):
        _radioml2018_labels_from_arrays([0, 1], [0])


def test_load_radioml2018_labels_parses_real_hdf5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REAL HDF5 parser: skipped in the dep-free venv, runs on the cluster [data] venv.

    Guarded with importorskip so it SKIPS where numpy/h5py are absent and RUNS on the cluster.
    Writes a tiny genuine ``GOLD_XYZ_OSC.0001_1024.hdf5`` with the published ``X/Y/Z`` layout,
    then asserts the loader returns the expected (mod, snr) labels.
    """
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    path = resolve_cache_dir() / "radioml_2018_01a" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3 items: modulation columns 0 (OOK), 3 (BPSK), 23 (OQPSK); SNRs -20/0/30.
    y = np.zeros((3, 24), dtype="float32")
    y[0, 0] = 1.0
    y[1, 3] = 1.0
    y[2, 23] = 1.0
    z = np.array([[-20], [0], [30]], dtype="int64")  # (N, 1) as in the real file
    x = np.zeros((3, 1024, 2), dtype="float32")
    with h5py.File(path, "w") as fh:
        fh.create_dataset("X", data=x)
        fh.create_dataset("Y", data=y)
        fh.create_dataset("Z", data=z)

    labels = load_radioml_labels("radioml_2018_01a")
    assert labels == [("OOK", -20), ("BPSK", 0), ("OQPSK", 30)]


# --- Sig53: no static release -> BLOCKED (must not synthesise) -----------------------


def test_load_sig53_official_split_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sig53 has no static published artifact -> loader raises a clear, actionable blocker."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="NO static published download"):
        load_sig53_official_split()


def test_download_sig53_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Sig53 download entry point refuses to synthesise and reports the blocker + root."""
    from rfbench.data.download.amc_sig53 import download_sig53

    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="generation-only via TorchSig"):
        download_sig53()
