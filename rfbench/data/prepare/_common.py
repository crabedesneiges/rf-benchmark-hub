"""Shared ``prepare`` helpers reused by every data WP (AMC, SEI, detection).

This module is the SPLIT-GENERATION TEMPLATE for the data layer (M1). It turns
already-extracted per-item *stratum labels* into a canonical, deterministic split and
its provenance sidecar, wiring together the frozen core contracts:

* :func:`rfbench.core.splits.make_split` / :func:`~rfbench.core.splits.adopt_official_split`
  build the :class:`~rfbench.core.splits.SplitManifest`;
* :func:`rfbench.core.splits.write_split_index` serialises the ``.idx.json``;
* :class:`rfbench.core.manifest.DatasetManifest` records provenance + integrity, written
  here beside the split index (``core.manifest.write_manifest`` is only a frozen stub).

The split-generation path is deliberately fed *label lists* (never raw arrays), so it is
exercisable on pure-stdlib synthetic fixtures with no numpy/h5py: a per-WP loader (lazy
numpy/h5py) extracts the labels from the real files on the cluster, then hands them here.

Pure stdlib only -- all ``json`` / ``hashlib`` / ``datetime`` use is at module top from
the standard library, so importing this module pulls in no third-party dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import (
    DEFAULT_RATIOS,
    SplitManifest,
    adopt_official_split,
    make_split,
    write_split_index,
)
from rfbench.core.types import SplitName

#: Environment variable holding the dataset cache dir (raw downloads + extracted files).
CACHE_ENV_VAR = "RFBENCH_CACHE"

#: Canonical partition order (kept in lockstep with ``core.splits._SPLIT_ORDER``).
_SPLITS: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")

#: Fallback cache dir (relative to CWD) used when ``$RFBENCH_CACHE`` is unset. Kept in
#: lockstep with the CLI's ``_default_cache`` so both resolve to the same place.
_FALLBACK_CACHE_DIRNAME = ".rfbench_cache"


def resolve_cache_dir(cache: str | Path | None = None) -> Path:
    """Resolve the dataset cache directory, honouring ``$RFBENCH_CACHE``.

    Resolution order:

    1. an explicit ``cache`` argument (a caller/CLI override), if given;
    2. the ``$RFBENCH_CACHE`` environment variable, if set and non-empty;
    3. a local ``./.rfbench_cache`` fallback under the current working directory.

    No path is ever hard-coded to an absolute location: on the cluster ``$RFBENCH_CACHE``
    points at Lustre storage; tests set it to a ``tmp_path``. The directory is *not*
    created here -- callers create the specific subtree they need.
    """
    if cache is not None:
        return Path(cache).expanduser()
    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.cwd() / _FALLBACK_CACHE_DIRNAME


def encode_strata(labels: Sequence[tuple[object, ...]]) -> list[int]:
    """Encode per-item composite stratum keys (tuples) into dense integer group ids.

    ``labels[i]`` is a tuple describing item ``i``'s stratum -- for AMC the natural key is
    ``(modulation, snr_db)``; SEI/detection reuse this with their own key. Distinct keys
    map to ``0..k-1`` in sorted-key order, so the encoding is deterministic and independent
    of the order in which items appear. The result is the ``stratify`` vector consumed by
    :func:`rfbench.core.splits.make_split`.
    """
    unique = sorted({tuple(key) for key in labels}, key=_stratum_sort_key)
    code_of: dict[tuple[object, ...], int] = {key: i for i, key in enumerate(unique)}
    return [code_of[tuple(key)] for key in labels]


def prepare_from_labels(
    *,
    dataset: str,
    split_id: str,
    n_items: int,
    strata: Sequence[tuple[object, ...]] | None,
    source_url: str,
    out_dir: str | Path,
    source_checksums: Mapping[str, str] | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build a deterministic **generated** split from per-item stratum labels.

    This is the generic prepare-from-labels helper the whole data layer reuses: given a
    dataset name and one composite stratum key per item (e.g. ``(modulation, snr_db)`` for
    AMC, ``(transmitter, receiver)`` for SEI), it

    1. encodes the strata into dense group ids (:func:`encode_strata`);
    2. calls :func:`rfbench.core.splits.make_split` (80/10/10 by default, ``seed`` 42),
       stratified by those ids;
    3. writes the ``.idx.json`` via :func:`rfbench.core.splits.write_split_index`;
    4. writes a :class:`~rfbench.core.manifest.DatasetManifest` sidecar next to it.

    ``strata`` may be ``None`` for an unstratified split (a single stratum). ``n_items``
    must equal ``len(strata)`` when ``strata`` is given. Nothing here touches numpy: the
    caller supplies plain Python label tuples, so the path runs on synthetic fixtures.

    Returns the ``(SplitManifest, DatasetManifest)`` pair. Writes only under ``out_dir``
    (``leaderboard/splits/<dataset>/``) -- never raw data (D3).
    """
    if strata is not None and len(strata) != n_items:
        raise ValueError(f"strata length {len(strata)} does not match n_items {n_items}")

    stratify = encode_strata(strata) if strata is not None else None
    split = make_split(
        n_items,
        seed=seed,
        ratios=ratios,
        stratify=stratify,
        split_id=split_id,
        dataset=dataset,
    )
    dataset_manifest = _finalise(
        split=split,
        source_url=source_url,
        out_dir=out_dir,
        n_items=n_items,
        source_checksums=source_checksums,
    )
    return split, dataset_manifest


def prepare_from_official(
    *,
    dataset: str,
    split_id: str,
    official: Mapping[str, Sequence[int]],
    source_url: str,
    out_dir: str | Path,
    source_checksums: Mapping[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Adopt an **official** literature/vendor split verbatim and write its sidecars.

    Thin wrapper over :func:`rfbench.core.splits.adopt_official_split` that mirrors
    :func:`prepare_from_labels`' write side (idx.json + :class:`DatasetManifest`). Used
    when a dataset ships a canonical split (e.g. Sig53's TorchSig split); ``n_items`` in
    the manifest is the total number of adopted indices.
    """
    split = adopt_official_split(official, split_id=split_id, dataset=dataset, seed=seed)
    n_items = sum(len(split.indices[name]) for name in _SPLITS)
    dataset_manifest = _finalise(
        split=split,
        source_url=source_url,
        out_dir=out_dir,
        n_items=n_items,
        source_checksums=source_checksums,
    )
    return split, dataset_manifest


def write_dataset_manifest(manifest: DatasetManifest, out_dir: str | Path) -> Path:
    """Write ``leaderboard/splits/<dataset>/<split_id>.manifest.json`` (deterministic).

    Serialised beside the split index with ``sort_keys`` so the file is reproducible
    byte-for-byte. ``source_checksums`` attest raw-file integrity but the raw data itself
    is never written or committed (D3). Returns the written path.
    """
    dest_dir = Path(out_dir) / "splits" / manifest.dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{manifest.canonical_split_id}.manifest.json"
    dest.write_text(
        json.dumps(_manifest_to_doc(manifest), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def manifest_checksum(manifest: DatasetManifest) -> str:
    """``"sha256:<hex>"`` over the canonical serialisation of a :class:`DatasetManifest`.

    Order-independent and format-stable (``sort_keys`` + compact separators), so it can be
    carried into a submission's ``data_provenance`` block and re-verified later.
    """
    payload = json.dumps(_manifest_to_doc(manifest), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --- private helpers ----------------------------------------------------------------


def _finalise(
    *,
    split: SplitManifest,
    source_url: str,
    out_dir: str | Path,
    n_items: int,
    source_checksums: Mapping[str, str] | None,
) -> DatasetManifest:
    """Write the split index + manifest for a built :class:`SplitManifest`."""
    split_checksum = write_split_index(split, str(out_dir))
    manifest = DatasetManifest(
        dataset=split.dataset,
        canonical_split_id=split.canonical_split_id,
        source_url=source_url,
        seed=split.seed,
        n_items=n_items,
        split_checksum=split_checksum,
        source_checksums=dict(source_checksums or {}),
        created_at=_utc_now_iso(),
    )
    write_dataset_manifest(manifest, out_dir)
    return manifest


def _manifest_to_doc(manifest: DatasetManifest) -> dict[str, object]:
    """Plain-dict view of a :class:`DatasetManifest` for stable JSON serialisation."""
    return {
        "dataset": manifest.dataset,
        "canonical_split_id": manifest.canonical_split_id,
        "source_url": manifest.source_url,
        "seed": manifest.seed,
        "n_items": manifest.n_items,
        "split_checksum": manifest.split_checksum,
        "source_checksums": dict(manifest.source_checksums),
        "created_at": manifest.created_at,
    }


def _stratum_sort_key(key: tuple[object, ...]) -> tuple[tuple[int, str], ...]:
    """Type-agnostic total order over composite stratum keys.

    Keys may mix ints and strs (e.g. ``("QPSK", 8)``), which are not mutually comparable.
    Each element is mapped to ``(type_rank, str_value)`` so sorting is deterministic and
    never raises ``TypeError`` regardless of the element types used by a given WP.
    """
    return tuple((0, repr(part)) if isinstance(part, str) else (1, f"{part!r}") for part in key)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` timestamp (matches the CLI's format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "CACHE_ENV_VAR",
    "resolve_cache_dir",
    "encode_strata",
    "prepare_from_labels",
    "prepare_from_official",
    "write_dataset_manifest",
    "manifest_checksum",
]
