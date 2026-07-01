"""Deterministic canonical splits: identity, indices and checksum.

Only a split's *identity* is ever versioned in ``leaderboard/splits/`` -- the
:class:`SplitManifest` (id + checksum + indices) -- never raw data (D3).

This module defines the :class:`SplitManifest` value object and the deterministic
split API (WP-10). Reproducibility is the acceptance criterion: two calls with the
same ``(n_items, seed, ratios, stratify)`` return byte-identical indices, and the
written ``.idx.json`` is reproducible byte-for-byte (``sort_keys``).

Per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``): if a dataset ships a split
used by the literature, adopt it verbatim (``provenance='official'``); otherwise
generate a deterministic 80/10/10 train/val/test split, stratified by the task's
label structure, seed 42 (``provenance='generated'``).

No third-party imports at module top; all ``hashlib`` / ``json`` use stays lazy inside
the functions (``import rfbench.core`` stays dependency-free -- stdlib only).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rfbench.core.types import SplitName

#: How a split's indices were obtained: ``"generated"`` (deterministic, seeded) or
#: ``"official"`` (a literature/vendor split adopted verbatim).
Provenance = Literal["generated", "official"]

#: Canonical partition order used everywhere indices are laid out or serialised.
_SPLIT_ORDER: tuple[SplitName, SplitName, SplitName] = ("train", "val", "test")

#: Default ratios per SPLIT POLICY: 80/10/10 train/val/test.
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Deterministic split identity plus its indices.

    Only this object (``canonical_split_id`` + ``checksum`` + ``indices``) is versioned
    under ``leaderboard/splits/<dataset>/``; the raw data it indexes never is (D3).
    """

    #: Deterministic split id, e.g. ``"amc-strat-snr-seed42-v1"``.
    canonical_split_id: str
    #: Dataset id the split was drawn from, e.g. ``"radioml_2016_10a"``.
    dataset: str
    #: Seed used to generate the split (42 by protocol convention).
    seed: int
    #: ``"sha256:<64hex>"`` over the sorted indices / the ``.idx.json`` file.
    checksum: str
    #: Partition name -> item indices, e.g. ``{"train": [...], "val": [...],
    #: "test": [...]}``.
    indices: dict[SplitName, list[int]]


#: Provenance carried alongside a manifest until it is serialised. ``SplitManifest`` is
#: a frozen contract we may not extend (and its ``dict`` field makes it unhashable), so
#: provenance -- a serialisation concern, not part of the split *identity* -- is tracked
#: here keyed by ``canonical_split_id`` and consulted lazily by
#: :func:`write_split_index`, defaulting to ``"generated"``.
_PROVENANCE: dict[str, Provenance] = {}


def make_split(
    n_items: int,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    stratify: list[int] | None = None,
    split_id: str,
    dataset: str,
) -> SplitManifest:
    """Build a deterministic :class:`SplitManifest`.

    Pure and deterministic: the same ``(n_items, seed, ratios, stratify)`` produce
    byte-identical indices (WP-10 acceptance). ``stratify`` holds integer group ids
    (e.g. encoded ``modulation x snr`` for AMC): each stratum is split by ``ratios`` and
    the rounding remainder is distributed deterministically so overall proportions match
    while every stratum's proportion is respected. Without ``stratify`` a single stratum
    (all items) is used. The returned manifest carries the computed checksum and is
    tagged ``provenance='generated'``.
    """
    if n_items < 0:
        raise ValueError(f"n_items must be non-negative, got {n_items}")
    if stratify is not None and len(stratify) != n_items:
        raise ValueError(f"stratify length {len(stratify)} does not match n_items {n_items}")
    _validate_ratios(ratios)

    rng = random.Random(seed)
    labels: Sequence[int] = stratify if stratify is not None else [0] * n_items

    indices: dict[SplitName, list[int]] = {name: [] for name in _SPLIT_ORDER}
    # Iterate strata in a stable (sorted) order so the result is independent of the
    # order in which labels first appear.
    for stratum in sorted(_group_by_stratum(labels)):
        members = _group_by_stratum(labels)[stratum]
        shuffled = list(members)
        rng.shuffle(shuffled)
        for name, chunk in zip(_SPLIT_ORDER, _partition(shuffled, ratios), strict=True):
            indices[name].extend(chunk)

    for name in _SPLIT_ORDER:
        indices[name].sort()

    manifest = SplitManifest(
        canonical_split_id=split_id,
        dataset=dataset,
        seed=seed,
        checksum=_checksum_of_indices(indices),
        indices=indices,
    )
    _PROVENANCE[manifest.canonical_split_id] = "generated"
    return manifest


def adopt_official_split(
    official: Mapping[str, Sequence[int]],
    *,
    split_id: str,
    dataset: str,
    seed: int = 42,
) -> SplitManifest:
    """Adopt an official/literature split, passing its indices through verbatim.

    ``official`` maps partition name (``"train" | "val" | "test"``) to an explicit,
    author-provided list of item indices. The lists are copied as given (only sorted
    for a canonical, reproducible on-disk layout) -- no shuffling, no re-stratification.
    The returned manifest is tagged ``provenance='official'``. ``seed`` is recorded for
    provenance only; it does not influence the indices.

    Overlapping indices between partitions raise :class:`ValueError` -- an official split
    must still be a clean partition (no leakage).
    """
    indices: dict[SplitName, list[int]] = {name: [] for name in _SPLIT_ORDER}
    for name, items in official.items():
        if name not in indices:
            raise ValueError(f"unknown split partition {name!r}; expected one of {_SPLIT_ORDER}")
        as_list = [int(i) for i in items]
        if any(i < 0 for i in as_list):
            raise ValueError(f"split {name!r} has a negative index")
        indices[name] = sorted(as_list)

    _assert_disjoint(indices)

    manifest = SplitManifest(
        canonical_split_id=split_id,
        dataset=dataset,
        seed=seed,
        checksum=_checksum_of_indices(indices),
        indices=indices,
    )
    _PROVENANCE[manifest.canonical_split_id] = "official"
    return manifest


def write_split_index(manifest: SplitManifest, out_dir: str) -> str:
    """Write ``leaderboard/splits/<dataset>/<canonical_split_id>.idx.json``.

    The JSON doc records the indices, seed, ratios, stratify key, provenance
    (``'generated' | 'official'``) and a content checksum, with deterministic key
    ordering (``sort_keys``) so the file is reproducible byte-for-byte.

    Returns the ``"sha256:<hex>"`` checksum of the indices, which equals
    ``manifest.checksum``.
    """
    import json
    from pathlib import Path

    provenance: Provenance = _PROVENANCE.get(manifest.canonical_split_id, "generated")
    ratios = _infer_ratios(manifest.indices) if provenance == "generated" else None

    doc = {
        "canonical_split_id": manifest.canonical_split_id,
        "dataset": manifest.dataset,
        "seed": manifest.seed,
        "provenance": provenance,
        "ratios": list(ratios) if ratios is not None else None,
        "stratify_key": "stratum" if provenance == "generated" else None,
        "checksum": manifest.checksum,
        "indices": {name: list(manifest.indices[name]) for name in _SPLIT_ORDER},
    }

    dest_dir = Path(out_dir) / "splits" / manifest.dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{manifest.canonical_split_id}.idx.json"
    dest.write_text(
        json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest.checksum


def split_checksum(idx_path: str) -> str:
    """Recompute the ``"sha256:<hex>"`` of an on-disk ``.idx.json`` file.

    The checksum is taken over the canonical serialisation of the *indices* (not the
    surrounding metadata), so it is stable across formatting and matches
    :attr:`SplitManifest.checksum`. Used by ``rfbench data verify`` and the CI split
    lint.
    """
    import json
    from pathlib import Path

    doc = json.loads(Path(idx_path).read_text(encoding="utf-8"))
    raw_indices = doc["indices"]
    indices: dict[SplitName, list[int]] = {
        name: [int(i) for i in raw_indices.get(name, [])] for name in _SPLIT_ORDER
    }
    return _checksum_of_indices(indices)


# --- private helpers ----------------------------------------------------------------


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    """Reject non-positive or non-normalised ratios."""
    if len(ratios) != 3:
        raise ValueError(f"ratios must be a 3-tuple, got {ratios!r}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios!r}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios!r} (sum={total})")


def _group_by_stratum(labels: Sequence[int]) -> dict[int, list[int]]:
    """Map each stratum label to the sorted list of item indices carrying it."""
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)
    return groups


def _partition(
    members: list[int], ratios: tuple[float, float, float]
) -> tuple[list[int], list[int], list[int]]:
    """Split one (already shuffled) stratum into train/val/test by ``ratios``.

    Uses the largest-remainder method so counts sum exactly to ``len(members)`` and the
    per-stratum proportions track ``ratios`` as closely as integer counts allow. Ties in
    the remainder are broken by partition order (train, then val, then test), keeping the
    result deterministic.
    """
    n = len(members)
    raw = [r * n for r in ratios]
    floors = [int(x) for x in raw]
    remainder = n - sum(floors)
    # Distribute the remaining items to the partitions with the largest fractional part;
    # break ties by canonical partition order (stable via the index tag).
    order = sorted(range(3), key=lambda i: (-(raw[i] - floors[i]), i))
    counts = list(floors)
    for i in order[:remainder]:
        counts[i] += 1

    train_n, val_n, _ = counts
    train = members[:train_n]
    val = members[train_n : train_n + val_n]
    test = members[train_n + val_n :]
    return train, val, test


def _infer_ratios(indices: Mapping[SplitName, list[int]]) -> tuple[float, float, float]:
    """Recover the realised (train, val, test) ratios from partition sizes."""
    sizes = [len(indices[name]) for name in _SPLIT_ORDER]
    total = sum(sizes)
    if total == 0:
        return DEFAULT_RATIOS
    return (sizes[0] / total, sizes[1] / total, sizes[2] / total)


def _assert_disjoint(indices: Mapping[SplitName, list[int]]) -> None:
    """Raise if any index appears in more than one partition (no leakage)."""
    seen: set[int] = set()
    for name in _SPLIT_ORDER:
        part = set(indices[name])
        overlap = seen & part
        if overlap:
            raise ValueError(
                f"split leakage: indices {sorted(overlap)} appear in multiple partitions"
            )
        seen |= part


def _checksum_of_indices(indices: Mapping[SplitName, list[int]]) -> str:
    """Stable ``"sha256:<hex>"`` over a canonical JSON view of the indices.

    Order-independent-safe: each partition's indices are sorted and the partitions are
    emitted in canonical order via ``sort_keys``, so the digest depends only on the *set*
    of indices per partition, not on input ordering or formatting.
    """
    import hashlib
    import json

    canonical = {name: sorted(indices.get(name, [])) for name in _SPLIT_ORDER}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "SplitManifest",
    "Provenance",
    "DEFAULT_RATIOS",
    "make_split",
    "adopt_official_split",
    "write_split_index",
    "split_checksum",
]
