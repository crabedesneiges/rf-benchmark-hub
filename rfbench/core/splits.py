"""Deterministic canonical splits: identity, indices and checksum.

Only a split's *identity* is ever versioned in ``leaderboard/splits/`` -- the
:class:`SplitManifest` (id + checksum + indices) -- never raw data (D3).

This module defines the :class:`SplitManifest` value object and the *signatures* of
the deterministic split API. The reproducible generation logic itself is delivered by
WP-10 (``core/splits`` acceptance: two runs produce byte-identical indices); the
function bodies here raise :class:`NotImplementedError` so the frozen contract can be
imported and type-checked in M0 without pulling in any implementation detail.

No third-party imports at module top; any ``hashlib`` / ``json`` use stays lazy inside
the functions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfbench.core.types import SplitName


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


def make_split(
    n_items: int,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    stratify: list[int] | None = None,
    split_id: str,
    dataset: str,
) -> SplitManifest:
    """Build a deterministic :class:`SplitManifest`.

    Pure and deterministic: the same ``(n_items, seed, ratios, stratify)`` produce
    byte-identical indices (WP-10 acceptance). ``stratify`` holds integer group ids
    (e.g. encoded ``modulation x snr`` for AMC). The returned manifest carries the
    computed checksum.

    Implemented by WP-10.
    """
    raise NotImplementedError("make_split is implemented in WP-10 (deterministic splits)")


def write_split_index(manifest: SplitManifest, out_dir: str) -> str:
    """Write ``leaderboard/splits/<dataset>/<canonical_split_id>.idx.json``.

    Returns the ``"sha256:<hex>"`` checksum of the written file, which MUST equal
    ``manifest.checksum``.

    Implemented by WP-10.
    """
    raise NotImplementedError("write_split_index is implemented in WP-10")


def split_checksum(idx_path: str) -> str:
    """Recompute the ``"sha256:<hex>"`` of an on-disk ``.idx.json`` file.

    Used by ``rfbench data verify`` and the CI split lint.

    Implemented by WP-10.
    """
    raise NotImplementedError("split_checksum is implemented in WP-10")


__all__ = ["SplitManifest", "make_split", "write_split_index", "split_checksum"]
