"""Protocol-tech-ID (WiFi 802.11 standard recognition) canonical splits.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§protocol_tech_id):

* **tprime-wifi4** (T-PRIME, Genesys Lab / Northeastern, DRS item ``neu:h989s8519``
  = dataset **DS 3.0** in the T-PRIME data table, paper arXiv:2401.04837, code
  github.com/genesys-neu/t-prime) is the **real over-the-air raw-IQ 4-class**
  WiFi-standard set (``802.11b``, ``802.11g``, ``802.11n``, ``802.11ax``), single
  protocol per capture, multi-room, 7279 transmissions. CONFIRMED (2026-07) against
  ``t-prime/data/README.md``'s dataset table -- DS 3.0 is the single-protocol capture;
  DS 3.1-3.4 are multi-protocol *overlapping-mixture* captures for a different task
  (overlap detection) and must NOT be used here. The T-PRIME repo/README ships **no**
  train/test index lists (only capture folders per protocol), so -- per the split
  policy's "official-if-exists, else deterministic" rule -- there is no official
  split to adopt here: we generate a deterministic **80/10/10** split **stratified
  by protocol class**, seed 42. Canonical id ``proto-tprime-wifi4-8010-seed42-v1``.

LICENSE / redistribution: the DRS collection is **openly downloadable** (no login), but
the dataset's redistribution LICENSE is **unstated** on the landing page. We only ship
split indices + checksums (never raw IQ, D3), so redistribution is not at issue here; the
dataset card in ``docs/EVALUATION_PROTOCOL.md`` flags the unconfirmed terms.

Split GENERATION is decoupled from data loading: :func:`prepare_protocol` accepts
already-extracted class labels, so the whole path runs on pure-stdlib synthetic fixtures
with no numpy. The heavy label EXTRACTION from the extracted ``.bin`` IQ captures lives in
the lazy loader below (:func:`load_protocol_labels`), which is never called in unit tests.

Module top imports are stdlib + the frozen core contracts only; numpy is imported lazily
inside the loader with a clear ``pip install rfbench[data]`` error.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import (
    prepare_from_labels,
    prepare_from_official,
    resolve_cache_dir,
)

#: The protocol-tech-ID datasets this module prepares.
ProtocolDataset = Literal["tprime_wifi4"]

#: Canonical split id per dataset (baked ratios+seed; changing either bumps task version). This is
#: the WITHIN-DISTRIBUTION track (``closed_set``): recordings from every room are mixed 80/10/10.
CANONICAL_SPLIT_IDS: dict[str, str] = {
    "tprime_wifi4": "proto-tprime-wifi4-8010-seed42-v1",
}

#: The physical LOCATIONS (rooms) of the T-PRIME OTA single-protocol collection, used for the
#: ``cross_room`` track (the paper's scenario-split: leave-one-location-out). A location groups its
#: per-day sub-collections (e.g. ``RM_573C_1``/``RM_573C_2`` -> ``RM_573C``). Power-scaling and
#: ``_z``/upsampled/noise variants are not plain ``802.11{b,g,n,ax}`` dirs, so they are excluded by
#: the recording enumeration and never enter any split.
CROSSROOM_LOCATIONS: tuple[str, ...] = ("RM_142", "RM_572C", "RM_573C")

#: Per-held-out-location canonical split id for the ``cross_room`` track. Each is a leave-one-
#: location-out grouped split: the held-out location is the TEST set in full (never seen in train),
#: the other locations are train + a class-stratified val carve. The board's ``cross_room`` number
#: is the cross-validation mean over these folds (paper's scenario-split), never mixed with the
#: within-distribution ``closed_set`` track.
CROSSROOM_SPLIT_IDS: dict[str, str] = {
    loc: f"proto-tprime-wifi4-crossroom-heldout-{loc.split('_')[1].lower()}-v1"
    for loc in CROSSROOM_LOCATIONS
}

#: Official source URL recorded in the dataset's manifest (provenance, never redistributed).
#: The Northeastern DRS item page for DS 3.0 (the single-protocol capture), not the code repo.
SOURCE_URLS: dict[str, str] = {
    "tprime_wifi4": "https://repository.library.northeastern.edu/files/neu:h989s8519",
}

#: The 4 WiFi-standard classes, in canonical order. This is the class index order the
#: task/model use, and the loader below must emit labels consistent with it. Ordered by the
#: 802.11 amendment letter as reported by T-PRIME (b, g, n, ax).
PROTOCOL_CLASSES: tuple[str, ...] = (
    "802.11b",
    "802.11g",
    "802.11n",
    "802.11ax",
)

#: Canonical raw-IQ window length (samples) drawn from each recording for the T-PRIME **SM**
#: baseline (paper Table II: N = M*S = 24*64 = 1536). Kept in lockstep with
#: ``rfbench.models.baselines.tprime.SM_SEQUENCE_LEN``. The canonical split is over RECORDINGS,
#: so this is a pure LOAD-time reshaping knob: changing it does NOT invalidate the committed
#: split (only how many samples each window carries).
TPRIME_WINDOW_LEN: int = 1536

#: How many fixed-length windows are tiled from each recording at load time, spread evenly
#: across the capture (DS 3.0 ``.bin`` captures are long recordings ~198k samples, not single
#: windows). A per-recording cap keeps the materialised dataset tractable and weights every
#: recording equally; it does NOT affect the split (which partitions recordings, never windows,
#: so windows from one recording never leak across train/test).
WINDOWS_PER_RECORDING: int = 32


def prepare_protocol(
    dataset: ProtocolDataset | str,
    *,
    out_dir: str | Path,
    labels: Sequence[str] | None = None,
    source_checksums: dict[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical protocol-tech-ID split + manifest for ``dataset``.

    The split-GENERATION path takes pre-extracted class labels so it runs without numpy on
    synthetic fixtures: pass ``labels`` as a sequence of per-item class names (one per item)
    -> **80/10/10 stratified by class**, seed 42.

    On the cluster the caller first extracts these via :func:`load_protocol_labels`
    (lazy numpy), then calls this.

    Writes ``<out_dir>/splits/<dataset>/<id>.idx.json`` and ``...manifest.json`` only; never
    raw data (D3). Returns the ``(SplitManifest, DatasetManifest)`` pair.
    """
    if dataset not in CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown protocol dataset {dataset!r}; expected one of "
            f"{sorted(CANONICAL_SPLIT_IDS)}"
        )
    split_id = CANONICAL_SPLIT_IDS[dataset]
    source_url = SOURCE_URLS[dataset]

    if labels is None:
        raise ValueError(
            f"{dataset!r} has no official split; pass `labels=` as per-item class names "
            "(extracted via load_protocol_labels) to stratify by class"
        )
    strata: list[tuple[object, ...]] = [(label,) for label in labels]
    return prepare_from_labels(
        dataset=dataset,
        split_id=split_id,
        n_items=len(strata),
        strata=strata,
        source_url=source_url,
        out_dir=out_dir,
        source_checksums=source_checksums,
        seed=seed,
    )


# --- lazy loaders (cluster-only; heavy deps; NEVER called in unit tests) -------------


def load_protocol_labels(
    dataset: Literal["tprime_wifi4"] = "tprime_wifi4",
    cache: str | Path | None = None,
) -> list[str]:
    """Extract per-item class-name labels from the extracted T-PRIME ``.bin`` captures on disk.

    The T-PRIME OTA WiFi corpus extracts to one raw-IQ capture tree per 802.11 protocol. We
    enumerate the four class subtrees in :data:`PROTOCOL_CLASSES` order, and for each emit one
    label per contained capture window, so the label order matches the flat sample order
    materialised by
    :meth:`rfbench.tasks.protocol_tech_id.dataset.ProtocolDataset.load`.

    Returns one class-name string per item, ready to hand to :func:`prepare_protocol` as
    ``labels=``. Never called in unit tests (needs the real data + heavy deps).

    TODO (cluster): the T-PRIME README documents per-protocol capture folders but NOT an
    on-disk window granularity; confirm on Lustre whether each ``.bin`` file is already one
    fixed-length window or a long recording to be tiled into ``N``-sample windows
    (:data:`rfbench.models.baselines.tprime.SM_SEQUENCE_LEN`). If it is a long recording, the
    label loader and the array loader must tile with the SAME window length + stride so the
    committed split indices stay aligned.
    """
    cache_dir = resolve_cache_dir(cache)
    return _load_tprime_wifi4_labels(cache_dir)


def _class_dir_names() -> dict[str, tuple[str, ...]]:
    """Map each canonical class to the on-disk sub-directory name(s) used per protocol.

    The T-PRIME captures are grouped per 802.11 amendment. The exact folder spelling is not
    pinned by the README, so each class carries a few plausible spellings (``802_11b`` /
    ``80211b`` / ``11b`` / ``b``) as a robustness fallback; confirm the real names on the
    cluster and prune this map to the one that matches.
    """
    return {
        "802.11b": ("802_11b", "80211b", "11b", "b", "802.11b"),
        "802.11g": ("802_11g", "80211g", "11g", "g", "802.11g"),
        "802.11n": ("802_11n", "80211n", "11n", "n", "802.11n"),
        "802.11ax": ("802_11ax", "80211ax", "11ax", "ax", "802.11ax"),
    }


def _load_tprime_wifi4_labels(cache_dir: Path) -> list[str]:
    """Enumerate the extracted per-class ``.bin`` captures and emit one class label PER RECORDING.

    Reads only directory listings (via :mod:`pathlib`), so no numpy is needed to build the label
    list -- the heavy numpy load of the IQ arrays themselves happens later in the dataset loader.
    One label per recording (not per window) is what makes the split RECORDING-LEVEL: the array
    loader tiles each recording into windows AFTER the split, so windows from one capture never
    straddle train/test (no leakage). Shares :func:`_iter_recording_files` with the array loader
    so the flat recording order is identical on both sides and the committed indices stay aligned.
    """
    ds_dir = _resolve_dataset_root(cache_dir)
    if ds_dir is None:
        raise FileNotFoundError(
            f"tprime_wifi4 not found under {cache_dir / 'tprime_wifi4'}; run the download step "
            "first (rfbench.data.download.protocol_tprime.download_tprime_wifi4)."
        )
    recordings = _iter_recording_files(ds_dir)
    if not recordings:
        raise FileNotFoundError(
            f"no per-class IQ captures found under {ds_dir}; the extracted layout of the "
            "T-PRIME collection may differ from the expected <room>/<class>/*.bin tree (confirm "
            "on the cluster)."
        )
    return [class_name for _path, class_name in recordings]


def _iter_recording_files(ds_dir: Path) -> list[tuple[Path, str]]:
    """Canonical ordered ``(capture_path, class_name)`` list over the extracted T-PRIME tree.

    Walks each class in :data:`PROTOCOL_CLASSES` order and, within a class, every capture file
    in sorted-path order. This is the ONE ordering the label loader (one label per recording)
    and the array loader (which tiles each recording into windows) SHARE, so recording index
    ``i`` denotes the same capture on both sides and the committed split indices stay aligned.
    Handles the real DS 3.0 layout, where the per-protocol folders are nested one level under
    per-room directories (``<root>/RM_*/802.11x/*.bin``), and a flattened
    ``<root>/802.11x/*.bin`` layout alike.
    """
    dir_names = _class_dir_names()
    recordings: list[tuple[Path, str]] = []
    for class_name in PROTOCOL_CLASSES:
        for path in _iter_class_files(ds_dir, dir_names[class_name]):
            recordings.append((path, class_name))
    return recordings


def _window_offsets(n_samples: int, window_len: int, max_windows: int) -> list[int]:
    """Deterministic start offsets tiling one recording into fixed-length windows.

    Returns up to ``max_windows`` start indices for length-``window_len`` windows, spread evenly
    across ``[0, n_samples - window_len]`` so the whole capture is represented (not just its
    head). Falls back to a single ``[0]`` window when the recording is shorter than one window
    (the array loader zero-pads it). Purely arithmetic -> deterministic + numpy-free, so the
    tiling geometry is unit-testable without the heavy IO, and identical every run (idempotent).
    """
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    if max_windows <= 0:
        raise ValueError(f"max_windows must be positive, got {max_windows}")
    if n_samples <= window_len:
        return [0]
    n_full = n_samples // window_len  # non-overlapping capacity
    m = min(max_windows, n_full)
    if m <= 1:
        return [0]
    last = n_samples - window_len
    return [(j * last) // (m - 1) for j in range(m)]


def _recording_location(path: Path) -> str:
    """Return the LOCATION (room) of a capture from its path.

    A capture lives at ``<root>/RM_<id>_<day>/<class>/<file>`` so its per-day sub-collection dir is
    ``path.parents[1]`` (e.g. ``RM_573C_2``); the location groups the per-day sub-collections to the
    ``RM_<id>`` prefix (``RM_573C``). Falls back to the sub-collection name if it lacks that shape.
    """
    room = path.parents[1].name  # e.g. RM_573C_2
    parts = room.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else room


def _iter_recording_locations(ds_dir: Path) -> list[str]:
    """One LOCATION per recording, in the canonical recording order (aligned with the labels)."""
    return [_recording_location(path) for path, _cls in _iter_recording_files(ds_dir)]


def load_protocol_locations(
    dataset: Literal["tprime_wifi4"] = "tprime_wifi4",
    cache: str | Path | None = None,
) -> list[str]:
    """Extract one LOCATION per recording, aligned 1:1 with :func:`load_protocol_labels`.

    Same directory-listing-only enumeration (no numpy) and same canonical order, so a recording's
    label and its location line up by index -- what :func:`prepare_crossroom` needs to hold whole
    locations out. Never called in unit tests (needs the real cache).
    """
    ds_dir = _resolve_dataset_root(resolve_cache_dir(cache))
    if ds_dir is None:
        raise FileNotFoundError(
            f"tprime_wifi4 not found under {resolve_cache_dir(cache) / 'tprime_wifi4'}; run the "
            "download step first."
        )
    return _iter_recording_locations(ds_dir)


def prepare_crossroom(
    held_out_location: str,
    *,
    out_dir: str | Path,
    labels: Sequence[str],
    locations: Sequence[str],
    source_checksums: dict[str, str] | None = None,
    seed: int = 42,
    val_fraction: float = 0.1,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the leave-one-location-out ``cross_room`` split holding out ``held_out_location``.

    Reproduces the T-PRIME paper's scenario-split: the held-out location is the TEST set in full
    (never seen in training), the OTHER locations are the training pool, from which a
    class-stratified ``val_fraction`` val set is carved (deterministically, seeded) to drive the
    early-stopping / LR schedule. Whole locations go to one partition, so no recording -- and hence
    no window -- leaks across the train/test location boundary. ``labels`` and ``locations`` are the
    per-recording class-name / location lists (aligned; from :func:`load_protocol_labels` /
    :func:`load_protocol_locations`), so this stays numpy-free and unit-testable on fixtures.
    """
    if held_out_location not in CROSSROOM_SPLIT_IDS:
        raise ValueError(
            f"unknown held-out location {held_out_location!r}; expected one of "
            f"{sorted(CROSSROOM_SPLIT_IDS)}"
        )
    if len(labels) != len(locations):
        raise ValueError(
            f"labels ({len(labels)}) and locations ({len(locations)}) must be the same length"
        )
    test = [i for i, loc in enumerate(locations) if loc == held_out_location]
    rest = [i for i, loc in enumerate(locations) if loc != held_out_location]
    if not test:
        raise ValueError(f"no recordings in held-out location {held_out_location!r}")
    if not rest:
        raise ValueError(f"held-out location {held_out_location!r} leaves no training data")

    # Deterministic class-stratified val carve from the TRAINING locations (never from the held-out
    # test location), so val monitors in-distribution generalisation while test stays fully unseen.
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {}
    for i in rest:
        by_class.setdefault(labels[i], []).append(i)
    val: list[int] = []
    for cls in sorted(by_class):
        idxs = list(by_class[cls])
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * val_fraction))
        val.extend(idxs[:k])
    val_set = set(val)
    train = [i for i in rest if i not in val_set]
    official = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    return prepare_from_official(
        dataset="tprime_wifi4",
        split_id=CROSSROOM_SPLIT_IDS[held_out_location],
        official=official,
        source_url=SOURCE_URLS["tprime_wifi4"],
        out_dir=out_dir,
        source_checksums=source_checksums,
        seed=seed,
    )


def _resolve_dataset_root(cache_dir: Path) -> Path | None:
    """Return the directory that holds the per-protocol capture folders (or ``None``).

    The archive extracts under ``<cache>/tprime_wifi4/``; accept either a nested
    ``T-PRIME``/``dataset`` root (if the archive kept one) or the ``tprime_wifi4`` dir
    directly (if it was flattened).
    """
    base = cache_dir / "tprime_wifi4"
    for candidate in (base / "T-PRIME", base / "dataset", base):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return base if base.exists() else None


def _iter_class_files(ds_dir: Path, dir_candidates: tuple[str, ...]) -> list[Path]:
    """Return the sorted raw-IQ capture files for one class, across any per-room nesting.

    Collects captures from a class sub-directory located EITHER directly under ``ds_dir``
    (flattened layout ``ds_dir/802.11x/``) OR one level down under a per-room directory (the real
    DS 3.0 layout ``ds_dir/RM_*/802.11x/``). Raw-IQ files are ``.bin`` interleaved-IQ first, plus
    a few fallbacks. The order is deterministic (sorted, de-duplicated full paths) so the label
    flatten and the array flatten stay in lockstep across every room.
    """
    files: list[Path] = []
    for candidate in dir_candidates:
        class_dirs = [ds_dir / candidate]  # flattened layout
        class_dirs += sorted(ds_dir.glob(f"*/{candidate}"))  # per-room layout (RM_*/<class>)
        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
            for pattern in ("*.bin", "*.npy", "*.iq", "*.dat", "*.sigmf-data"):
                files.extend(class_dir.glob(pattern))
    return sorted(set(files))


__all__ = [
    "ProtocolDataset",
    "CANONICAL_SPLIT_IDS",
    "CROSSROOM_LOCATIONS",
    "CROSSROOM_SPLIT_IDS",
    "SOURCE_URLS",
    "PROTOCOL_CLASSES",
    "TPRIME_WINDOW_LEN",
    "WINDOWS_PER_RECORDING",
    "prepare_protocol",
    "prepare_crossroom",
    "load_protocol_labels",
    "load_protocol_locations",
]
