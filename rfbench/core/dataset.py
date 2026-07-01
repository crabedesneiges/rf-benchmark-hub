"""The ``Dataset`` contract: one dataset variant for one task.

A :class:`Dataset` never redistributes raw data (D3): :meth:`Dataset.download` fetches
into ``$RFBENCH_CACHE`` and :meth:`Dataset.prepare` writes *only* deterministic split
indices plus a checksum. :meth:`Dataset.load` returns a ``torch.utils.data.Dataset``
for a given ``(split, track)``.

``torch`` is imported only under ``TYPE_CHECKING`` so ``import rfbench.core`` stays
dependency-free while type checkers still see the real ``torch.utils.data.Dataset``
return type. At runtime the annotation is a string forward reference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.splits import SplitManifest
from rfbench.core.types import SplitName, Track

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data


class Dataset(ABC):
    """A dataset variant for one task.

    Never redistributes raw data (D3): :meth:`download` fetches into
    ``$RFBENCH_CACHE`` and :meth:`prepare` writes only indices plus a checksum.
    """

    #: Dataset id, e.g. ``"radioml_2016_10a"``.
    name: str
    #: Deterministic split id of the active split, e.g. ``"amc-strat-snr-seed42-v1"``.
    canonical_split_id: str
    #: ``"sha256:<hex>"`` of the active split-index file.
    checksum: str

    @abstractmethod
    def download(self, cache: Path | None = None) -> None:
        """Fetch raw data from the official source into ``cache``.

        ``cache`` defaults to ``$RFBENCH_CACHE``. Writes NO git-tracked files.
        """

    @abstractmethod
    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build deterministic splits (via ``core/splits``) and a manifest.

        Writes indices to ``leaderboard/splits/<name>/``. Idempotent.
        """

    @abstractmethod
    def load(
        self,
        split: SplitName,
        track: Track | None = None,
    ) -> torch.utils.data.Dataset[Any]:
        """Return a torch ``Dataset`` for ``(split, track)``.

        SEI uses ``track`` for ``closed_set`` / ``cross_receiver`` / ``cross_day`` /
        ``open_set``.
        """


__all__ = ["Dataset"]
