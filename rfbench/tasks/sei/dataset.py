"""SEI dataset adapter (WP-21): binds a canonical SEI split to the eval loop.

:class:`SeiDataset` implements the frozen :class:`rfbench.core.dataset.Dataset` contract
for the WiSig / ORACLE fingerprinting datasets. Its :meth:`load` yields per-sample
canonical batches ``{"iq": ..., "label": tx_label, "meta": {"rx": ..., "day": ...}}`` for a
given ``(split, track)``, where ``track`` selects one of the SEI conditions
(``closed_set`` / ``cross_receiver`` / ``cross_day``) or the ``open_set`` protocol.

The REAL loader (reading IQ from the WiSig/ORACLE files on the cluster) is lazy and never
exercised in unit tests: it lives in :meth:`_load_from_cache`, which imports numpy behind a
clear ``rfbench[tasks]`` hint and reads the canonical split indices produced by
:mod:`rfbench.data.prepare.sei`. Unit tests inject an in-memory sample list via
``samples=`` so the whole adapter runs dependency-free.

Module-top imports are stdlib + the frozen core contracts + the SEI split-id table only;
numpy is imported lazily inside the real loader.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.dataset import Dataset
from rfbench.core.splits import SplitManifest
from rfbench.core.types import Batch, SplitName, Track
from rfbench.data.prepare.sei import CANONICAL_SPLIT_IDS

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data

#: The SEI conditions/tracks this adapter serves. ``open_set`` reuses the ``closed_set``
#: canonical split id (same emitters/captures) but is scored with the open-set metrics.
_TRACK_TO_CONDITION: dict[Track, str] = {
    "closed_set": "closed_set",
    "cross_receiver": "cross_receiver",
    "cross_day": "cross_day",
    "open_set": "closed_set",
}

#: Placeholder checksum for the in-memory synthetic adapter (schema pattern
#: ``^sha256:[0-9a-f]{64}$``). The real cluster loader overrides ``checksum`` from the
#: on-disk ``.idx.json`` it reads.
_SYNTHETIC_CHECKSUM = "sha256:" + "00" * 32


class _InMemorySplit:
    """A tiny map-style dataset: a list of per-sample canonical ``Batch`` dicts."""

    def __init__(self, samples: Sequence[Batch]) -> None:
        """Wrap an already-materialised list of per-sample batches."""
        self._samples = list(samples)

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self._samples)

    def __iter__(self) -> Iterator[Batch]:
        """Iterate the per-sample batches in order (deterministic)."""
        return iter(self._samples)


class SeiDataset(Dataset):
    """A SEI dataset variant (WiSig / ORACLE) for one canonical condition.

    ``track`` selects the active SEI condition and hence the ``canonical_split_id``:
    ``closed_set`` / ``cross_receiver`` / ``cross_day`` map to their WiSig split ids;
    ``open_set`` reuses the closed-set split with the open-set metrics. Pass ``samples=`` a
    list of ``{"iq", "label", "meta"}`` dicts to build an in-memory dataset for tests; leave
    it ``None`` and the real numpy loader reads the prepared indices on the cluster.
    """

    def __init__(
        self,
        name: str = "wisig",
        *,
        track: Track = "closed_set",
        samples: Mapping[SplitName, Sequence[Batch]] | None = None,
    ) -> None:
        """Bind the adapter to a ``(dataset, track)`` and optionally in-memory samples."""
        if name not in CANONICAL_SPLIT_IDS:
            raise ValueError(
                f"unknown SEI dataset {name!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
            )
        condition = _TRACK_TO_CONDITION.get(track)
        if condition is None:
            raise ValueError(
                f"unknown SEI track {track!r}; expected one of {sorted(_TRACK_TO_CONDITION)}"
            )
        ids_for_dataset = CANONICAL_SPLIT_IDS[name]
        if condition not in ids_for_dataset:
            raise ValueError(
                f"dataset {name!r} does not support track {track!r}; "
                f"supported: {sorted(ids_for_dataset)}"
            )
        self.name = name
        self._track = track
        self.canonical_split_id = ids_for_dataset[condition]
        self.checksum = _SYNTHETIC_CHECKSUM
        self._samples = samples

    def download(self, cache: Path | None = None) -> None:  # pragma: no cover - cluster-only
        """Fetch raw WiSig/ORACLE data (cluster-only; delegated to the download layer)."""
        raise NotImplementedError(
            "SEI raw download runs on the cluster; use rfbench.data.download.sei_* "
            "then rfbench.data.prepare.sei.prepare_sei."
        )

    def prepare(self, seed: int = 42) -> SplitManifest:  # pragma: no cover - cluster-only
        """Build the canonical split (cluster-only; delegated to the prepare layer)."""
        raise NotImplementedError(
            "SEI split preparation runs on the cluster via rfbench.data.prepare.sei.prepare_sei "
            "against records extracted from the real WiSig/ORACLE files."
        )

    def load(
        self,
        split: SplitName,
        track: Track | None = None,
    ) -> torch.utils.data.Dataset[Any]:
        """Return a map-style dataset for ``(split, track)``.

        With in-memory ``samples`` (tests) the requested ``split`` slice is returned
        directly; otherwise the real numpy loader reads the prepared indices + IQ from the
        cluster cache. A ``track`` argument, if given, must match the track this adapter was
        constructed for (the adapter is per-condition, so its ``canonical_split_id`` is
        fixed at construction).
        """
        if track is not None and track != self._track:
            raise ValueError(
                f"this SeiDataset serves track {self._track!r}; got load(track={track!r}). "
                "Construct a SeiDataset per track."
            )
        if self._samples is not None:
            return _InMemorySplit(self._samples.get(split, ()))
        return self._load_from_cache(split)

    def _load_from_cache(
        self, split: SplitName
    ) -> torch.utils.data.Dataset[Any]:  # pragma: no cover - cluster-only
        """Real loader: read canonical indices + IQ from the cluster cache (lazy numpy).

        Never called in unit tests. Imports numpy behind a clear ``rfbench[tasks]`` hint,
        loads ``leaderboard/splits/<name>/<canonical_split_id>.idx.json`` for the requested
        ``split`` partition, and materialises ``{"iq", "label", "meta"}`` per index. The
        concrete IQ layout is wired on the cluster against the real files.
        """
        try:
            import numpy as np  # noqa: F401 - surfaces the install hint early
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading SEI IQ needs numpy; install it with `pip install rfbench[tasks]`."
            ) from exc
        raise NotImplementedError(
            f"SEI IQ loading for split {split!r} runs on the cluster against the real "
            f"{self.name!r} files and the prepared {self.canonical_split_id!r} indices."
        )


__all__ = ["SeiDataset"]
