"""Dataset provenance and integrity: manifests beside the split indices.

A :class:`DatasetManifest` records the provenance and integrity of a *prepared*
dataset and is serialised next to its split indices under
``leaderboard/splits/<dataset>/``. ``source_checksums`` attest the integrity of the
RAW source files -- these are *verified*, never stored or committed (D3).
:class:`DataProvenance` is the trimmed subset (ids + checksums only) carried into a
submission's ``data_provenance`` block.

The generation/verification helpers are delivered by the data layer (M1); their bodies
here raise :class:`NotImplementedError` so the frozen contract imports and type-checks
in M0. Any ``hashlib`` / ``json`` use stays lazy inside the functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rfbench.core.splits import SplitManifest


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Provenance and integrity of a prepared dataset.

    Serialised next to the split indices. ``source_checksums`` are the integrity
    hashes of the RAW files, which are verified but never stored/committed (D3).
    """

    #: Dataset id, e.g. ``"radioml_2016_10a"``.
    dataset: str
    #: Deterministic split id, e.g. ``"amc-strat-snr-seed42-v1"``.
    canonical_split_id: str
    #: Official source URL the raw data was fetched from.
    source_url: str
    #: Seed used to generate the split.
    seed: int
    #: Number of items in the prepared dataset.
    n_items: int
    #: ``"sha256:<hex>"`` of the split-index file.
    split_checksum: str
    #: Raw-file name -> ``"sha256:<hex>"`` (integrity check only, never committed).
    source_checksums: Mapping[str, str]
    #: ISO-8601 UTC timestamp of creation.
    created_at: str


@dataclass(frozen=True, slots=True)
class DataProvenance:
    """The subset carried into ``submission.schema.json``'s ``data_provenance``.

    Only ids and checksums -- never raw data (D3).
    """

    #: Deterministic split id.
    canonical_split_id: str
    #: ``"sha256:<hex>"`` of the split-index file.
    split_checksum: str
    #: ``"sha256:<hex>"`` of the serialised :class:`DatasetManifest`.
    manifest_checksum: str


def write_manifest(
    split: SplitManifest,
    source_url: str,
    source_checksums: Mapping[str, str],
    out_dir: Path,
) -> DatasetManifest:
    """Write the dataset manifest JSON beside ``leaderboard/splits/<dataset>/``.

    Returns the constructed :class:`DatasetManifest`. Implemented by the data layer (M1).
    """
    raise NotImplementedError("write_manifest is implemented in the data layer (M1)")


def verify_manifest(manifest: DatasetManifest, idx_path: Path) -> bool:
    """Return ``True`` iff the on-disk index checksum matches ``manifest.split_checksum``.

    Used by the CI split lint. Implemented by the data layer (M1).
    """
    raise NotImplementedError("verify_manifest is implemented in the data layer (M1)")


def provenance_of(manifest: DatasetManifest, out_dir: Path) -> DataProvenance:
    """Compute the :class:`DataProvenance` (incl. ``manifest_checksum``) for a submission.

    Implemented by the data layer (M1).
    """
    raise NotImplementedError("provenance_of is implemented in the data layer (M1)")


__all__ = [
    "DatasetManifest",
    "DataProvenance",
    "write_manifest",
    "verify_manifest",
    "provenance_of",
]
