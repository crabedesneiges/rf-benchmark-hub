"""Protocol-tech-ID (WiFi 802.11 standard recognition) canonical splits.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§protocol_tech_id):

* **tprime-wifi4** (T-PRIME, Genesys Lab / Northeastern, DRS collection
  ``neu:h989s847q``, paper arXiv:2401.04837, code github.com/genesys-neu/t-prime)
  is the **real over-the-air raw-IQ 4-class** WiFi-standard set (``802.11b``,
  ``802.11g``, ``802.11n``, ``802.11ax``). The T-PRIME repo/README ships **no**
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

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import (
    prepare_from_labels,
    resolve_cache_dir,
)

#: The protocol-tech-ID datasets this module prepares.
ProtocolDataset = Literal["tprime_wifi4"]

#: Canonical split id per dataset (baked ratios+seed; changing either bumps task version).
CANONICAL_SPLIT_IDS: dict[str, str] = {
    "tprime_wifi4": "proto-tprime-wifi4-8010-seed42-v1",
}

#: Official source URL recorded in the dataset's manifest (provenance, never redistributed).
#: The Northeastern DRS collection landing page (dataset), not the code repo.
SOURCE_URLS: dict[str, str] = {
    "tprime_wifi4": "https://repository.library.northeastern.edu/collections/neu:h989s847q",
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
    """Enumerate the extracted per-class ``.bin`` captures and emit one class label per sample.

    Reads only directory listings (via :mod:`pathlib`), so no numpy is needed to build the
    label list -- the heavy numpy load of the IQ arrays themselves happens later in the
    dataset loader. Walks each class subtree in canonical class order.
    """
    ds_dir = _resolve_dataset_root(cache_dir)
    if ds_dir is None:
        raise FileNotFoundError(
            f"tprime_wifi4 not found under {cache_dir / 'tprime_wifi4'}; run the download step "
            "first (rfbench.data.download.protocol_tprime.download_tprime_wifi4)."
        )
    dir_names = _class_dir_names()
    labels: list[str] = []
    for class_name in PROTOCOL_CLASSES:
        files = _iter_class_files(ds_dir, dir_names[class_name])
        labels.extend(class_name for _ in files)
    if not labels:
        raise FileNotFoundError(
            f"no per-class IQ captures found under {ds_dir}; the extracted layout of the "
            "T-PRIME collection may differ from the expected <Class>/*.bin tree (confirm on "
            "the cluster)."
        )
    return labels


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
    """Return the sorted raw-IQ capture files for one class under ``ds_dir``.

    Looks for each candidate class sub-directory name directly under ``ds_dir`` and collects
    the raw-IQ files (``.bin`` interleaved-IQ first, plus a few fallbacks). The order is
    deterministic (sorted paths) so the label flatten and the array flatten stay in lockstep.
    """
    files: list[Path] = []
    for candidate in dir_candidates:
        class_dir = ds_dir / candidate
        if not class_dir.is_dir():
            continue
        for pattern in ("*.bin", "*.npy", "*.iq", "*.dat", "*.sigmf-data"):
            files.extend(sorted(class_dir.glob(pattern)))
    return sorted(set(files))


__all__ = [
    "ProtocolDataset",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "PROTOCOL_CLASSES",
    "prepare_protocol",
    "load_protocol_labels",
]
