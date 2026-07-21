"""Spectrum-sensing (DeepSense occupancy) canonical splits.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§"Spectrum sensing"):

* **deepsense** (Uvaydov et al., INFOCOM 2021, DOI 10.1109/INFOCOM42981.2021.9488764;
  wineslab repo https://github.com/wineslab/deepsense-spectrum-sensing-datasets) is the OTA
  wideband-IQ set whose fixed-length raw-IQ windows each carry a binary spectrum-occupancy label
  (``0`` vacant / ``1`` occupied). It has no canonical literature split adopted here -> a
  deterministic **80/10/10** split **stratified by the binary occupancy label**, seed 42.
  Canonical id ``sensing-deepsense-8010-seed42-v1``.

LICENSE: the DeepSense dataset license is UNSTATED on the wineslab repo -- the corpus is fetched
manually (gated / external host) and never redistributed (D3).

Split GENERATION is decoupled from data loading: :func:`prepare_sensing` accepts already-extracted
binary occupancy labels, so the whole path runs on pure-stdlib synthetic fixtures with no numpy.
The heavy label EXTRACTION from the extracted windows lives in the lazy loader
:func:`rfbench.data.download.spectrum_deepsense.load_deepsense_occupancy`, which is never called
in unit tests.

Module top imports are stdlib + the frozen core contracts only; numpy stays in the lazy loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import prepare_from_labels

#: The spectrum-sensing datasets this module prepares.
SensingDataset = Literal["deepsense"]

#: Canonical split id per dataset (baked ratios+seed; changing either bumps task version).
CANONICAL_SPLIT_IDS: dict[str, str] = {
    "deepsense": "sensing-deepsense-8010-seed42-v1",
}

#: Official source URL recorded in the dataset's manifest (provenance, never redistributed).
SOURCE_URLS: dict[str, str] = {
    "deepsense": "https://github.com/wineslab/deepsense-spectrum-sensing-datasets",
}

#: The two binary occupancy classes, index == label: 0 vacant, 1 occupied.
OCCUPANCY_CLASSES: tuple[str, str] = ("vacant", "occupied")


def prepare_sensing(
    dataset: SensingDataset | str,
    *,
    out_dir: str | Path,
    labels: Sequence[int] | None = None,
    source_checksums: dict[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical spectrum-sensing split + manifest for ``dataset``.

    The split-GENERATION path takes pre-extracted per-window binary occupancy labels so it runs
    without numpy on synthetic fixtures: pass ``labels`` as a sequence of ``0`` (vacant) / ``1``
    (occupied) ints (one per window) -> **80/10/10 stratified by the occupancy label**, seed 42.

    On the cluster the caller first extracts these via
    :func:`rfbench.data.download.spectrum_deepsense.load_deepsense_occupancy` (lazy numpy), then
    calls this. Labels outside ``{0, 1}`` raise :class:`ValueError` so a mislabelled stream fails
    loudly before stratification.

    Writes ``<out_dir>/splits/<dataset>/<id>.idx.json`` and ``...manifest.json`` only; never raw
    data (D3). Returns the ``(SplitManifest, DatasetManifest)`` pair.
    """
    if dataset not in CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown sensing dataset {dataset!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
        )
    split_id = CANONICAL_SPLIT_IDS[dataset]
    source_url = SOURCE_URLS[dataset]

    if labels is None:
        raise ValueError(
            f"{dataset!r} has no canonical split; pass `labels=` as per-window binary occupancy "
            "labels (0 vacant / 1 occupied, extracted via load_deepsense_occupancy) to stratify"
        )
    strata: list[tuple[object, ...]] = [(_check_binary(label),) for label in labels]
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


def _check_binary(label: int) -> int:
    """Return ``label`` if it is ``0`` or ``1``; raise :class:`ValueError` otherwise."""
    if label in (0, 1):
        return int(label)
    raise ValueError(f"spectrum-sensing occupancy label must be 0 or 1, got {label!r}")


__all__ = [
    "SensingDataset",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "OCCUPANCY_CLASSES",
    "prepare_sensing",
]
