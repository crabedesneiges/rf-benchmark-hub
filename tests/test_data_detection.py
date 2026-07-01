"""WP-13 acceptance tests for the wideband-detection (WBSig53) data layer.

Pure stdlib: no numpy/h5py/torchsig, no network. The generation/loader functions
(``generate_wbsig53``, ``load_wbsig53_annotations``) are DEFINED but never called here;
only the split-GENERATION + annotation path is exercised, fed synthetic wideband samples
(each a handful of time-frequency boxes) so it runs without any heavy dependency.

Covers, per WP-13 acceptance:
  * ``prepare_detection`` on synthetic samples -> deterministic 80/10/10 indices into
    ``$RFBENCH_CACHE`` (== tmp_path), no overlap, full coverage;
  * per-sample T-F box annotations round-trip through the ``.annotations.json`` sidecar,
    with a stable checksum;
  * detection vs recognition tracks stay distinct (separate ids + sidecars, same samples);
  * official WBSig53/TorchSig split adopted verbatim when provided;
  * malformed boxes / unknown track raise clear errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfbench.core.splits import split_checksum
from rfbench.core.types import SplitName
from rfbench.data.download.detection_wbsig53 import (
    RADDET_CLASSES,
    _parse_yolo_label_text,
    _yolo_to_tf_box,
    generate_wbsig53,
    load_raddet_annotations,
    load_wbsig53_annotations,
)
from rfbench.data.prepare.detection import (
    SOURCE_URLS,
    TRACKS,
    annotations_checksum,
    canonical_split_id,
    prepare_detection,
)

_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")


# --- synthetic fixtures (pure stdlib; no numpy/torchsig) ----------------------------


def _box(cls: str, t0: float, t1: float, f0: float, f1: float) -> dict[str, object]:
    return {"class": cls, "t_start": t0, "t_stop": t1, "f_low": f0, "f_high": f1}


def _synthetic_samples(n: int) -> list[dict[str, object]]:
    """``n`` wideband captures, each with a couple of deterministic T-F boxes.

    Box count/positions vary with the index so the round-trip is non-trivial, but every
    value stays inside the normalised ``[0, 1]`` box contract.
    """
    samples: list[dict[str, object]] = []
    classes = ("bpsk", "qpsk", "fm", "ofdm")
    for i in range(n):
        cls = classes[i % len(classes)]
        boxes = [_box(cls, 0.0, 0.5, 0.1, 0.2)]
        if i % 2 == 0:
            boxes.append(_box(classes[(i + 1) % len(classes)], 0.5, 1.0, 0.6, 0.9))
        samples.append({"boxes": boxes})
    return samples


def _all_indices(indices: dict[SplitName, list[int]]) -> list[int]:
    return sorted(i for name in _SPLITS for i in indices[name])


# --- generated 80/10/10 split path --------------------------------------------------


def test_prepare_detection_generated_80_10_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No official split -> deterministic 80/10/10 over samples, seed 42."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(100)

    split, manifest, ann_path = prepare_detection("wbsig53", out_dir=str(tmp_path), samples=samples)

    sizes = {name: len(split.indices[name]) for name in _SPLITS}
    assert sizes == {"train": 80, "val": 10, "test": 10}
    assert _all_indices(split.indices) == list(range(100))
    assert not (set(split.indices["train"]) & set(split.indices["val"]))
    assert not (set(split.indices["train"]) & set(split.indices["test"]))
    assert not (set(split.indices["val"]) & set(split.indices["test"]))

    assert split.canonical_split_id == canonical_split_id("wbsig53", "detection", official=False)
    assert manifest.dataset == "wbsig53"
    assert manifest.n_items == 100
    assert manifest.seed == 42
    assert ann_path.exists()


def test_prepare_detection_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs (different out_dirs) produce byte-identical indices + checksum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(60)

    a, _, _ = prepare_detection("wbsig53", out_dir=str(tmp_path / "a"), samples=samples)
    b, _, _ = prepare_detection("wbsig53", out_dir=str(tmp_path / "b"), samples=samples)
    assert a.indices == b.indices
    assert a.checksum == b.checksum


def test_prepare_detection_writes_idx_manifest_and_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare writes idx.json, manifest.json AND the annotations sidecar under the cache."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(30)

    split, manifest, ann_path = prepare_detection("wbsig53", out_dir=str(tmp_path), samples=samples)
    split_id = canonical_split_id("wbsig53", "detection", official=False)
    base = tmp_path / "splits" / "wbsig53"

    idx_path = base / f"{split_id}.idx.json"
    man_path = base / f"{split_id}.manifest.json"
    assert idx_path.exists()
    assert man_path.exists()
    assert ann_path == base / f"{split_id}.annotations.json"

    idx_doc = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx_doc["provenance"] == "generated"
    assert idx_doc["checksum"] == split.checksum

    man_doc = json.loads(man_path.read_text(encoding="utf-8"))
    assert man_doc["dataset"] == "wbsig53"
    assert man_doc["canonical_split_id"] == split_id
    assert man_doc["split_checksum"] == split.checksum
    assert man_doc["n_items"] == manifest.n_items == 30


# --- annotations round-trip ---------------------------------------------------------


def test_annotations_round_trip_in_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every input sample's T-F boxes survive the sidecar unchanged, with a stable sum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(12)

    _, _, ann_path = prepare_detection("wbsig53", out_dir=str(tmp_path), samples=samples)
    doc = json.loads(ann_path.read_text(encoding="utf-8"))

    assert doc["dataset"] == "wbsig53"
    assert doc["track"] == "detection"
    assert doc["n_samples"] == len(samples)
    assert len(doc["samples"]) == len(samples)

    for i, sample in enumerate(samples):
        recorded = doc["samples"][i]
        assert recorded["sample_id"] == i
        expected_boxes = sample["boxes"]
        assert isinstance(expected_boxes, list)
        assert len(recorded["boxes"]) == len(expected_boxes)
        for got, want in zip(recorded["boxes"], expected_boxes, strict=True):
            assert got["class"] == want["class"]
            assert got["t_start"] == want["t_start"]
            assert got["t_stop"] == want["t_stop"]
            assert got["f_low"] == want["f_low"]
            assert got["f_high"] == want["f_high"]

    # Embedded checksum matches a recompute over the recorded per-sample payload.
    assert doc["checksum"] == annotations_checksum(doc["samples"])
    assert doc["checksum"].startswith("sha256:")


def test_annotations_sidecar_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The annotations file is byte-identical across two independent prepares."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(20)

    _, _, a = prepare_detection("wbsig53", out_dir=str(tmp_path / "a"), samples=samples)
    _, _, b = prepare_detection("wbsig53", out_dir=str(tmp_path / "b"), samples=samples)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


# --- detection vs recognition tracks kept distinct ----------------------------------


def test_detection_and_recognition_tracks_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both tracks share samples but get distinct ids + sidecars, both recording classes."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(40)

    det_split, _, det_ann = prepare_detection(
        "wbsig53", out_dir=str(tmp_path), samples=samples, track="detection"
    )
    rec_split, _, rec_ann = prepare_detection(
        "wbsig53", out_dir=str(tmp_path), samples=samples, track="recognition"
    )

    # Distinct ids and distinct sidecar files.
    assert det_split.canonical_split_id != rec_split.canonical_split_id
    assert "detection" in det_split.canonical_split_id
    assert "recognition" in rec_split.canonical_split_id
    assert det_ann != rec_ann
    assert det_ann.exists() and rec_ann.exists()

    det_doc = json.loads(det_ann.read_text(encoding="utf-8"))
    rec_doc = json.loads(rec_ann.read_text(encoding="utf-8"))
    assert det_doc["track"] == "detection"
    assert rec_doc["track"] == "recognition"

    # Same underlying samples -> identical index partition + identical recorded boxes.
    assert det_split.indices == rec_split.indices
    assert det_doc["samples"] == rec_doc["samples"]


def test_all_declared_tracks_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every track in ``TRACKS`` produces a coherent sidecar (guards the Literal set)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(10)
    for track in TRACKS:
        _, _, ann = prepare_detection(
            "wbsig53", out_dir=str(tmp_path / track), samples=samples, track=track
        )
        assert json.loads(ann.read_text(encoding="utf-8"))["track"] == track


# --- official-split adoption --------------------------------------------------------


def test_prepare_detection_adopts_official_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An official WBSig53/TorchSig partition is adopted verbatim (only sorted)."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(8)
    official = {"train": [5, 0, 3, 6], "val": [1, 7], "test": [2, 4]}

    split, manifest, ann_path = prepare_detection(
        "wbsig53", out_dir=str(tmp_path), samples=samples, official_split=official
    )
    assert split.canonical_split_id == canonical_split_id("wbsig53", "detection", official=True)
    assert split.indices == {"train": [0, 3, 5, 6], "val": [1, 7], "test": [2, 4]}
    assert manifest.n_items == 8

    idx_path = tmp_path / "splits" / "wbsig53" / f"{split.canonical_split_id}.idx.json"
    assert json.loads(idx_path.read_text(encoding="utf-8"))["provenance"] == "official"
    assert ann_path.exists()


def test_on_disk_index_checksum_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-disk index checksum recomputes to the returned split checksum."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    samples = _synthetic_samples(50)
    split, _, _ = prepare_detection("wbsig53", out_dir=str(tmp_path), samples=samples)
    idx_path = tmp_path / "splits" / "wbsig53" / f"{split.canonical_split_id}.idx.json"
    assert split_checksum(str(idx_path)) == split.checksum


# --- argument-contract guards -------------------------------------------------------


def test_prepare_detection_unknown_track_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown detection track"):
        prepare_detection(
            "wbsig53", out_dir=str(tmp_path), samples=_synthetic_samples(4), track="bogus"
        )


def test_prepare_detection_rejects_inverted_box(tmp_path: Path) -> None:
    bad = [{"boxes": [_box("bpsk", 0.8, 0.2, 0.1, 0.5)]}]  # t_start > t_stop
    with pytest.raises(ValueError, match="inverted"):
        prepare_detection("wbsig53", out_dir=str(tmp_path), samples=bad)


def test_prepare_detection_rejects_out_of_range_box(tmp_path: Path) -> None:
    bad = [{"boxes": [_box("bpsk", 0.0, 1.5, 0.1, 0.5)]}]  # t_stop > 1
    with pytest.raises(ValueError, match=r"normalised \[0, 1\]"):
        prepare_detection("wbsig53", out_dir=str(tmp_path), samples=bad)


def test_prepare_detection_rejects_missing_box_field(tmp_path: Path) -> None:
    bad = [{"boxes": [{"class": "bpsk", "t_start": 0.0, "t_stop": 0.5, "f_low": 0.1}]}]
    with pytest.raises(ValueError, match="each T-F box needs"):
        prepare_detection("wbsig53", out_dir=str(tmp_path), samples=bad)


def test_prepare_detection_rejects_non_list_boxes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a list of box mappings"):
        prepare_detection("wbsig53", out_dir=str(tmp_path), samples=[{"boxes": "nope"}])


# --- RadDet (REAL published dataset) YOLO box extraction ----------------------------
#
# RadDet (ICASSP 2025, arXiv:2501.10407) is the real, static, downloadable wideband
# detection dataset: spectrogram PNGs with sibling YOLO ``.txt`` box labels. The parse +
# T-F box conversion + directory walk are all pure stdlib, so the REAL loader is exercised
# here on a synthetic tree that mimics the published layout -- no numpy/torchsig/network.


def _write_raddet_tree(root: Path, tree: dict[str, dict[str, str]]) -> None:
    """Materialise a synthetic RadDet tree: ``images/<split>/<stem>.png`` + ``.txt``.

    ``tree`` maps ``split -> {stem: yolo_label_text}``. The PNGs are empty placeholders
    (the loader only reads their names); the ``.txt`` files carry real YOLO rows.
    """
    for split, items in tree.items():
        split_dir = root / "raddet" / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem, label_text in items.items():
            (split_dir / f"{stem}.png").write_bytes(b"")  # placeholder image
            (split_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def _boxes_of(sample: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a loader sample's ``boxes`` field to a typed list (keeps mypy --strict happy)."""
    boxes = sample["boxes"]
    assert isinstance(boxes, list)
    return boxes


def test_yolo_label_text_parses_rows() -> None:
    """A YOLO label file parses to ``(class_id, xc, yc, w, h)`` rows; blanks are skipped."""
    text = "0 0.5 0.5 0.2 0.4\n\n  9 0.611153 0.540000 0.250000 0.039563  \n"
    rows = _parse_yolo_label_text(text)
    assert rows == [
        (0, 0.5, 0.5, 0.2, 0.4),
        (9, 0.611153, 0.540000, 0.250000, 0.039563),
    ]


def test_yolo_label_text_empty_file_is_background_frame() -> None:
    """An empty label file is a valid background frame (no boxes), not an error."""
    assert _parse_yolo_label_text("") == []
    assert _parse_yolo_label_text("\n  \n") == []


def test_yolo_label_text_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="needs 5 fields"):
        _parse_yolo_label_text("0 0.5 0.5 0.2")


def test_yolo_label_text_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        _parse_yolo_label_text("0 x 0.5 0.2 0.4")


def test_yolo_to_tf_box_center_size_to_extents() -> None:
    """YOLO center/size maps to T-F extents (x=time, y=freq); class id -> RadDet name."""
    box = _yolo_to_tf_box(0, 0.5, 0.5, 0.2, 0.4)
    assert box["class"] == RADDET_CLASSES[0] == "Rect"
    assert box["t_start"] == pytest.approx(0.4)
    assert box["t_stop"] == pytest.approx(0.6)
    assert box["f_low"] == pytest.approx(0.3)
    assert box["f_high"] == pytest.approx(0.7)


def test_yolo_to_tf_box_clamps_subpixel_overflow() -> None:
    """A box straddling the edge is clamped into normalised [0, 1] (published rounding)."""
    box = _yolo_to_tf_box(10, 0.02, 0.99, 0.1, 0.1)  # class 10 == FMCW
    assert box["class"] == "FMCW"
    assert box["t_start"] == 0.0  # 0.02 - 0.05 -> clamped
    assert 0.0 <= float(box["t_stop"]) <= 1.0  # type: ignore[arg-type]
    assert box["f_high"] == 1.0  # 0.99 + 0.05 -> clamped


def test_yolo_to_tf_box_unknown_class_falls_back() -> None:
    """An out-of-range class id degrades to a raw ``class_<id>`` name (no crash)."""
    box = _yolo_to_tf_box(99, 0.5, 0.5, 0.1, 0.1)
    assert box["class"] == "class_99"


def test_load_raddet_annotations_reads_real_yolo_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL RadDet loader walks images/<split>/*.png + sibling .txt into T-F boxes."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    _write_raddet_tree(
        tmp_path,
        {
            "train": {
                "000000000000": "0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\n",
                "000000000001": "",  # background frame -> no boxes
            },
            "val": {"000000000002": "9 0.5 0.5 0.5 0.5\n"},
        },
    )

    samples = load_raddet_annotations()
    by_id = {s["sample_id"]: s for s in samples}
    # Sample ids are ``<split>/<stem>``; train sorted before val by the walk order.
    assert set(by_id) == {"train/000000000000", "train/000000000001", "val/000000000002"}

    first_boxes = _boxes_of(by_id["train/000000000000"])
    assert len(first_boxes) == 2
    assert first_boxes[0]["class"] == "Rect"
    assert first_boxes[0]["t_start"] == pytest.approx(0.4)

    # Background frame keeps an empty box list (valid detection negative).
    assert _boxes_of(by_id["train/000000000001"]) == []
    assert _boxes_of(by_id["val/000000000002"])[0]["class"] == "LFM"


def test_load_raddet_annotations_feeds_prepare_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: real loader output is a valid ``samples=`` input for prepare_detection."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    tree = {
        "train": {f"{i:012d}": "0 0.5 0.5 0.2 0.2\n" for i in range(8)},
        "val": {f"{i:012d}": "1 0.3 0.3 0.1 0.1\n" for i in range(8, 10)},
    }
    _write_raddet_tree(tmp_path, tree)

    samples = load_raddet_annotations()
    split, manifest, ann_path = prepare_detection(
        "raddet", out_dir=str(tmp_path / "out"), samples=samples
    )

    assert manifest.dataset == "raddet"
    assert manifest.n_items == len(samples) == 10
    assert split.canonical_split_id == canonical_split_id("raddet", "detection", official=False)
    assert manifest.source_url == SOURCE_URLS["raddet"]
    assert ann_path.exists()


def test_load_raddet_annotations_missing_tree_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent RadDet tree raises FileNotFoundError with manual-download guidance."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="RadDet not found"):
        load_raddet_annotations()


def test_prepare_detection_raddet_records_real_source_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raddet dataset id records the real Kaggle provenance URL in its manifest."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    _, manifest, _ = prepare_detection(
        "raddet", out_dir=str(tmp_path), samples=_synthetic_samples(20)
    )
    assert manifest.source_url == SOURCE_URLS["raddet"]
    assert "kaggle.com" in manifest.source_url


# --- WBSig53 blocker stubs (generation-only -> no synthetic generation) --------------


def test_wbsig53_generate_is_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """WBSig53 generation is disabled: real datasets only, no torchsig generation."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="no static published artifact"):
        generate_wbsig53()


def test_wbsig53_load_is_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """WBSig53 annotation loading is a documented blocker stub, not faked."""
    monkeypatch.setenv("RFBENCH_CACHE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="RadDet"):
        load_wbsig53_annotations()
