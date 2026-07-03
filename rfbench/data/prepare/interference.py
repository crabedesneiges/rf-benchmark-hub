"""Interference-ID (GNSS jamming classification) canonical splits.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§interference_id):

* **interf-gnss6** (Swinney & Woods 2021, Zenodo record 4629685, DOI
  10.5281/zenodo.4629685, CC-BY-4.0) is the raw-IQ 6-class GNSS-jamming set
  (``DME``, ``narrowband``, ``single_am``, ``single_chirp``, ``single_fm``,
  ``no_jamming``). It has no canonical literature split adopted here -> a deterministic
  **80/10/10** split **stratified by class**, seed 42. Canonical id
  ``interf-gnss6-8010-seed42-v1``.

HONESTY: the six jamming signals are MATLAB-synthesised (a ``signal_generation.m`` script
ships alongside the archive), BUT the corpus is distributed as a ready-to-download raw-IQ
archive (``Raw_IQ_Dataset.zip``, ~1.9 GB, no login), so this is treated as a public
DOWNLOAD dataset (not a generation-only blocker like Sig53 / WBSig53): we fetch the
published archive rather than regenerate it.

Split GENERATION is decoupled from data loading: :func:`prepare_interference` accepts
already-extracted class labels, so the whole path runs on pure-stdlib synthetic fixtures
with no numpy. The heavy label EXTRACTION from the extracted IQ files lives in the lazy
loader below (:func:`load_interference_labels`), which is never called in unit tests.

Module top imports are stdlib + the frozen core contracts only; numpy is imported lazily
inside the loader with a clear ``pip install rfbench[data]`` error.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import (
    prepare_from_labels,
    resolve_cache_dir,
)

#: The interference-ID datasets this module prepares.
InterferenceDataset = Literal["interf_gnss6"]

#: Canonical split id per dataset (baked ratios+seed; changing either bumps task version).
CANONICAL_SPLIT_IDS: dict[str, str] = {
    "interf_gnss6": "interf-gnss6-8010-seed42-v1",
}

#: Official source URL recorded in the dataset's manifest (provenance, never redistributed).
SOURCE_URLS: dict[str, str] = {
    "interf_gnss6": "https://zenodo.org/records/4629685",
}

#: The 6 GNSS-jamming classes, in canonical (sorted) order. This is the class index order the
#: task/model use, and the loader below must emit labels consistent with it.
INTERFERENCE_CLASSES: tuple[str, ...] = (
    "DME",
    "narrowband",
    "no_jamming",
    "single_am",
    "single_chirp",
    "single_fm",
)


def prepare_interference(
    dataset: InterferenceDataset | str,
    *,
    out_dir: str | Path,
    labels: Sequence[str] | None = None,
    source_checksums: dict[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical interference-ID split + manifest for ``dataset``.

    The split-GENERATION path takes pre-extracted class labels so it runs without numpy on
    synthetic fixtures: pass ``labels`` as a sequence of per-item class names (one per item)
    -> **80/10/10 stratified by class**, seed 42.

    On the cluster the caller first extracts these via :func:`load_interference_labels`
    (lazy numpy), then calls this.

    Writes ``<out_dir>/splits/<dataset>/<id>.idx.json`` and ``...manifest.json`` only; never
    raw data (D3). Returns the ``(SplitManifest, DatasetManifest)`` pair.
    """
    if dataset not in CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown interference dataset {dataset!r}; expected one of "
            f"{sorted(CANONICAL_SPLIT_IDS)}"
        )
    split_id = CANONICAL_SPLIT_IDS[dataset]
    source_url = SOURCE_URLS[dataset]

    if labels is None:
        raise ValueError(
            f"{dataset!r} has no canonical split; pass `labels=` as per-item class names "
            "(extracted via load_interference_labels) to stratify by class"
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


def load_interference_labels(
    dataset: Literal["interf_gnss6"] = "interf_gnss6",
    cache: str | Path | None = None,
) -> list[str]:
    """Extract per-item class-name labels from the extracted GNSS-jamming IQ files on disk.

    The Zenodo archive (``Raw_IQ_Dataset.zip``) extracts to per-class IQ files split across a
    training and a testing folder (Swinney & Woods 2021: ~1000 train + 250 test samples per
    class). We enumerate the six class subtrees in :data:`INTERFERENCE_CLASSES` order, and for
    each emit one label per contained sample, so the label order matches the flat sample order
    materialised by :meth:`rfbench.tasks.interference_id.dataset.InterferenceDataset.load`.

    Returns one class-name string per item, ready to hand to :func:`prepare_interference` as
    ``labels=``. Never called in unit tests (needs the real data + heavy deps).

    Confirmed extracted layout (Zenodo 4629685, ``Raw_IQ_Dataset.zip``): the archive extracts
    to a ``Raw_IQ_Dataset/`` root holding ``Training/`` and ``Testing/`` folders, each with the
    six per-class sub-directories ``DME``, ``NB``, ``NoJam``, ``SingleAM``, ``SingleChirp``,
    ``SingleFM`` (1000 train + 250 test ``.mat`` files per class, 7500 total).
    """
    cache_dir = resolve_cache_dir(cache)
    return _load_interference_gnss6_labels(cache_dir)


def _class_dir_names() -> dict[str, tuple[str, ...]]:
    """Map each canonical class to the on-disk sub-directory name(s) used inside a split folder.

    The published archive labels its per-class folders as ``DME`` / ``NB`` / ``NoJam`` /
    ``SingleAM`` / ``SingleChirp`` / ``SingleFM`` (confirmed against the extracted
    ``Raw_IQ_Dataset.zip``); a couple of spelling variants are kept as a robustness fallback.
    """
    return {
        "DME": ("DME",),
        "narrowband": ("NB", "narrowband"),
        "no_jamming": ("NoJam", "no_jamming"),
        "single_am": ("SingleAM", "single_am"),
        "single_chirp": ("SingleChirp", "single_chirp"),
        "single_fm": ("SingleFM", "single_fm"),
    }


def _load_interference_gnss6_labels(cache_dir: Path) -> list[str]:
    """Enumerate the extracted per-class IQ files and emit one class label per sample.

    Reads only directory listings (via :mod:`pathlib`), so no numpy is needed to build the
    label list -- the heavy numpy load of the IQ arrays themselves happens later in the
    dataset loader. Walks each class's Training + Testing subtree in canonical class order.
    """
    ds_dir = _resolve_dataset_root(cache_dir)
    if ds_dir is None:
        raise FileNotFoundError(
            f"interf_gnss6 not found under {cache_dir / 'interf_gnss6'}; run the download step "
            "first (rfbench.data.download.interference_gnss.download_interference_gnss6)."
        )
    dir_names = _class_dir_names()
    labels: list[str] = []
    for class_name in INTERFERENCE_CLASSES:
        files = _iter_class_files(ds_dir, dir_names[class_name])
        labels.extend(class_name for _ in files)
    if not labels:
        raise FileNotFoundError(
            f"no per-class IQ files found under {ds_dir}; the extracted layout of "
            "Raw_IQ_Dataset.zip may differ from the expected Training/Testing/<Class>/ tree."
        )
    return labels


def _resolve_dataset_root(cache_dir: Path) -> Path | None:
    """Return the directory that holds the ``Training``/``Testing`` split folders (or ``None``).

    The archive extracts a ``Raw_IQ_Dataset/`` subfolder, so the class dirs live under
    ``<cache>/interf_gnss6/Raw_IQ_Dataset/{Training,Testing}/<Class>/``. Accept either the
    ``Raw_IQ_Dataset`` root or the ``interf_gnss6`` dir directly (if the zip was flattened).
    """
    base = cache_dir / "interf_gnss6"
    for candidate in (base / "Raw_IQ_Dataset", base):
        if (candidate / "Training").is_dir() or (candidate / "Testing").is_dir():
            return candidate
    return base if base.exists() else None


def _iter_class_files(ds_dir: Path, dir_candidates: tuple[str, ...]) -> list[Path]:
    """Return the sorted IQ files for one class across the Training/Testing subtrees.

    Looks for each candidate class sub-directory name under ``ds_dir/Training`` and
    ``ds_dir/Testing`` (falling back to a direct child of ``ds_dir`` if the split folders are
    absent) and collects the ``.mat`` IQ files. The order is deterministic (sorted paths) so
    the label flatten and the array flatten stay in lockstep.
    """
    files: list[Path] = []
    split_roots = [ds_dir / "Training", ds_dir / "Testing", ds_dir]
    for split_root in split_roots:
        for candidate in dir_candidates:
            class_dir = split_root / candidate
            if not class_dir.is_dir():
                continue
            for pattern in ("*.mat", "*.npy", "*.bin", "*.iq", "*.dat"):
                files.extend(sorted(class_dir.glob(pattern)))
    return sorted(set(files))


__all__ = [
    "InterferenceDataset",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "INTERFERENCE_CLASSES",
    "prepare_interference",
    "load_interference_labels",
]
