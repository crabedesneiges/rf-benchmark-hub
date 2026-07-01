"""WP-10 acceptance tests for :mod:`rfbench.core.splits`.

Pure stdlib: no numpy/torch/sklearn. Covers determinism (two runs identical), ratio
correctness (~80/10/10, no leakage, union == all indices), stratification (each
stratum's proportion respected), checksum stability + sensitivity, and official-split
pass-through. All writes go to pytest ``tmp_path`` only -- no ``.idx.json`` for real
datasets is ever committed (D3).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rfbench.core.splits import (
    DEFAULT_RATIOS,
    SplitManifest,
    adopt_official_split,
    make_split,
    split_checksum,
    write_split_index,
)
from rfbench.core.types import SplitName

_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


# --- fixtures (synthetic label lists, pure stdlib) ----------------------------------


def _balanced_labels(n_per_class: int, n_classes: int) -> list[int]:
    """A stratify vector with ``n_per_class`` items in each of ``n_classes`` strata."""
    return [c for c in range(n_classes) for _ in range(n_per_class)]


def _all_indices(manifest: SplitManifest) -> list[int]:
    """Every index across the three partitions, sorted."""
    return sorted(i for name in _SPLITS for i in manifest.indices[name])


# --- determinism --------------------------------------------------------------------


def test_two_runs_are_identical() -> None:
    """Same inputs + seed -> byte-identical indices and checksum."""
    labels = _balanced_labels(100, 5)
    a = make_split(len(labels), stratify=labels, split_id="d-v1", dataset="d")
    b = make_split(len(labels), stratify=labels, split_id="d-v1", dataset="d")
    assert a.indices == b.indices
    assert a.checksum == b.checksum


def test_different_seed_changes_assignment_not_membership() -> None:
    """A different seed reshuffles which items land where, but still covers all items."""
    labels = _balanced_labels(50, 4)
    a = make_split(len(labels), seed=42, stratify=labels, split_id="d-v1", dataset="d")
    b = make_split(len(labels), seed=7, stratify=labels, split_id="d-v1", dataset="d")
    assert a.indices != b.indices
    assert _all_indices(a) == _all_indices(b) == list(range(len(labels)))


def test_default_seed_is_42() -> None:
    """Omitting ``seed`` matches an explicit ``seed=42``."""
    labels = _balanced_labels(30, 3)
    default = make_split(len(labels), stratify=labels, split_id="d-v1", dataset="d")
    explicit = make_split(len(labels), seed=42, stratify=labels, split_id="d-v1", dataset="d")
    assert default.seed == 42
    assert default.indices == explicit.indices


# --- ratio correctness + no leakage -------------------------------------------------


def test_default_ratios_are_80_10_10() -> None:
    """The module default ratio is the 80/10/10 SPLIT POLICY."""
    assert DEFAULT_RATIOS == (0.8, 0.1, 0.1)


def test_ratio_counts_and_full_coverage() -> None:
    """Counts track 80/10/10 and the union of partitions == all indices, no overlap."""
    n = 1000
    m = make_split(n, split_id="d-v1", dataset="d")  # no stratify -> single stratum
    sizes = {name: len(m.indices[name]) for name in _SPLITS}
    assert sizes["train"] == 800
    assert sizes["val"] == 100
    assert sizes["test"] == 100

    # union == all, and partitions are pairwise disjoint (no leakage).
    assert _all_indices(m) == list(range(n))
    assert sum(sizes.values()) == n
    assert not (set(m.indices["train"]) & set(m.indices["val"]))
    assert not (set(m.indices["train"]) & set(m.indices["test"]))
    assert not (set(m.indices["val"]) & set(m.indices["test"]))


def test_partitions_are_sorted() -> None:
    """Each partition's indices are emitted in ascending order (canonical layout)."""
    m = make_split(257, split_id="d-v1", dataset="d")
    for name in _SPLITS:
        assert m.indices[name] == sorted(m.indices[name])


def test_ratios_sum_and_bounds_validated() -> None:
    """Bad ratios are rejected early."""
    with pytest.raises(ValueError):
        make_split(10, ratios=(0.5, 0.3, 0.3), split_id="d-v1", dataset="d")
    with pytest.raises(ValueError):
        make_split(10, ratios=(-0.1, 0.6, 0.5), split_id="d-v1", dataset="d")


def test_stratify_length_mismatch_raises() -> None:
    """A stratify vector of the wrong length is rejected."""
    with pytest.raises(ValueError):
        make_split(10, stratify=[0, 1, 2], split_id="d-v1", dataset="d")


# --- stratification -----------------------------------------------------------------


def test_stratification_preserves_per_stratum_proportions() -> None:
    """Every stratum is itself split ~80/10/10, so proportions are preserved."""
    labels = _balanced_labels(200, 5)  # 5 strata x 200 = 1000 items
    m = make_split(len(labels), stratify=labels, split_id="d-v1", dataset="d")

    expected_counts: tuple[tuple[SplitName, int], ...] = (
        ("train", 160),
        ("val", 20),
        ("test", 20),
    )
    for name, expected in expected_counts:
        per_stratum = Counter(labels[i] for i in m.indices[name])
        for stratum in range(5):
            assert per_stratum[stratum] == expected, (name, stratum)


def test_stratification_no_leakage_full_coverage() -> None:
    """Stratified split still partitions the full index set with no overlap."""
    labels = _balanced_labels(37, 6)
    m = make_split(len(labels), stratify=labels, split_id="d-v1", dataset="d")
    assert _all_indices(m) == list(range(len(labels)))
    seen: set[int] = set()
    for name in _SPLITS:
        part = set(m.indices[name])
        assert not (seen & part)
        seen |= part


def test_stratification_independent_of_label_order() -> None:
    """Interleaved vs blocked labels of the same composition give the same split.

    Indices are keyed by position, so a permutation of *positions* changes indices; here
    we assert the per-stratum counts (the stratification guarantee) are identical.
    """
    blocked = _balanced_labels(40, 3)  # 000..111..222..
    interleaved = [c for _ in range(40) for c in range(3)]  # 012012...
    mb = make_split(len(blocked), stratify=blocked, split_id="d-v1", dataset="d")
    mi = make_split(len(interleaved), stratify=interleaved, split_id="d-v1", dataset="d")
    for name in _SPLITS:
        cb = Counter(blocked[i] for i in mb.indices[name])
        ci = Counter(interleaved[i] for i in mi.indices[name])
        assert cb == ci


# --- checksum -----------------------------------------------------------------------


def test_checksum_format() -> None:
    """Checksum is the schema-mandated ``sha256:<64hex>``."""
    m = make_split(100, split_id="d-v1", dataset="d")
    assert m.checksum.startswith("sha256:")
    hexpart = m.checksum.split(":", 1)[1]
    assert len(hexpart) == 64
    assert all(c in "0123456789abcdef" for c in hexpart)


def test_checksum_stable_across_runs(tmp_path: Path) -> None:
    """The on-disk file's recomputed checksum matches the manifest, twice."""
    m = make_split(500, stratify=_balanced_labels(100, 5), split_id="d-v1", dataset="d")
    returned = write_split_index(m, str(tmp_path))
    assert returned == m.checksum

    idx_path = tmp_path / "splits" / "d" / "d-v1.idx.json"
    assert split_checksum(str(idx_path)) == m.checksum
    # Recompute again -> identical (stable).
    assert split_checksum(str(idx_path)) == m.checksum


def test_checksum_sensitive_to_index_change() -> None:
    """Moving a single index between partitions changes the checksum."""
    m = make_split(100, split_id="d-v1", dataset="d")
    from rfbench.core.splits import _checksum_of_indices  # private helper under test

    base = _checksum_of_indices(m.indices)
    mutated: dict[SplitName, list[int]] = {name: list(m.indices[name]) for name in _SPLITS}
    moved = mutated["train"].pop()
    mutated["test"].append(moved)
    assert _checksum_of_indices(mutated) != base


def test_checksum_order_independent() -> None:
    """Shuffling the order of indices within a partition does not change the checksum."""
    from rfbench.core.splits import _checksum_of_indices

    a: dict[SplitName, list[int]] = {"train": [0, 1, 2], "val": [3], "test": [4]}
    b: dict[SplitName, list[int]] = {"train": [2, 0, 1], "val": [3], "test": [4]}
    assert _checksum_of_indices(a) == _checksum_of_indices(b)


# --- serialisation ------------------------------------------------------------------


def test_write_split_index_is_byte_reproducible(tmp_path: Path) -> None:
    """Two writes of the same manifest produce byte-identical files."""
    m = make_split(300, stratify=_balanced_labels(100, 3), split_id="d-v1", dataset="ds")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    write_split_index(m, str(out_a))
    write_split_index(m, str(out_b))
    fa = (out_a / "splits" / "ds" / "d-v1.idx.json").read_bytes()
    fb = (out_b / "splits" / "ds" / "d-v1.idx.json").read_bytes()
    assert fa == fb


def test_written_doc_has_required_fields(tmp_path: Path) -> None:
    """The doc records indices, seed, ratios, stratify key, provenance and checksum."""
    m = make_split(100, split_id="d-v1", dataset="ds")
    write_split_index(m, str(tmp_path))
    doc = json.loads((tmp_path / "splits" / "ds" / "d-v1.idx.json").read_text(encoding="utf-8"))
    assert doc["seed"] == 42
    assert doc["provenance"] == "generated"
    assert doc["stratify_key"] == "stratum"
    assert doc["checksum"] == m.checksum
    assert set(doc["indices"]) == set(_SPLITS)
    # ~80/10/10 realised ratios recorded.
    assert doc["ratios"][0] == pytest.approx(0.8, abs=1e-9)


# --- official-split pass-through ----------------------------------------------------


def test_official_split_passthrough() -> None:
    """Explicit per-split index lists are adopted verbatim, tagged 'official'."""
    official = {"train": [5, 0, 3], "val": [1, 4], "test": [2, 6]}
    m = adopt_official_split(official, split_id="off-v1", dataset="sig53")
    # Sorted for canonical layout, but the *membership* is verbatim.
    assert m.indices == {"train": [0, 3, 5], "val": [1, 4], "test": [2, 6]}
    assert m.seed == 42  # recorded, does not affect indices


def test_official_split_provenance_written(tmp_path: Path) -> None:
    """An adopted split serialises with provenance 'official' and no stratify key."""
    official = {"train": [0, 1], "val": [2], "test": [3]}
    m = adopt_official_split(official, split_id="off-v1", dataset="sig53")
    write_split_index(m, str(tmp_path))
    doc = json.loads(
        (tmp_path / "splits" / "sig53" / "off-v1.idx.json").read_text(encoding="utf-8")
    )
    assert doc["provenance"] == "official"
    assert doc["stratify_key"] is None
    assert doc["ratios"] is None
    assert split_checksum(str(tmp_path / "splits" / "sig53" / "off-v1.idx.json")) == m.checksum


def test_official_split_rejects_overlap() -> None:
    """An official split with leakage between partitions is rejected."""
    official = {"train": [0, 1, 2], "val": [2], "test": [3]}
    with pytest.raises(ValueError):
        adopt_official_split(official, split_id="off-v1", dataset="sig53")


def test_official_split_rejects_unknown_partition() -> None:
    """An unexpected partition name is rejected."""
    official = {"train": [0], "holdout": [1]}
    with pytest.raises(ValueError):
        adopt_official_split(official, split_id="off-v1", dataset="sig53")
