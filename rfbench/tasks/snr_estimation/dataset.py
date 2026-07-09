"""The SNR-estimation :class:`~rfbench.core.dataset.Dataset` adapter (J4).

One :class:`SnrDataset` instance is one SNR-estimation dataset variant (currently only
RadioML 2016.10a). It ties the SNR canonical split id + checksum (from
:mod:`rfbench.data.prepare.snr_estimation`) to a ``load(split, track)`` that yields
``(iq, snr_db, meta)`` samples -- the target here is the per-window SNR (dB), a float, rather
than a class label.

Two loading paths share one adapter (mirrors :class:`rfbench.tasks.amc.dataset.AmcDataset`):

* **cluster path** -- :meth:`load` reads the versioned SNR split indices, then the real IQ
  arrays + per-item SNRs from the cached RadioML pickle via the AMC array loader (LAZY
  numpy). Never touched by unit tests.
* **synthetic path** -- an in-memory list of per-sample dicts injected at construction
  (``samples=``). :meth:`load` returns it verbatim, so the whole adapter/metric/evaluate
  path runs on pure-Python fixtures with only ``pytest`` installed.

The underlying IQ + SNR arrays are IDENTICAL to AMC's -- SNR estimation reuses the same
signals and the same 80/10/10 partition (see :mod:`rfbench.data.prepare.snr_estimation`) --
so this adapter delegates array loading to :func:`rfbench.tasks.amc.dataset._load_amc_arrays`
and only re-labels the target as the SNR instead of the modulation class.

Module-top imports are stdlib + the frozen core contracts only; numpy stays lazy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.dataset import Dataset
from rfbench.core.splits import SplitManifest
from rfbench.core.types import Batch, SplitName, Track
from rfbench.data.prepare.snr_estimation import SNR_CANONICAL_SPLIT_IDS

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data


#: Placeholder checksum used before a split index has been prepared/loaded on disk. Matches
#: the schema pattern ``^sha256:[0-9a-f]{64}$`` so a synthetic-fixture ``result.json`` still
#: validates; the cluster path overwrites it from the on-disk ``.idx.json``.
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


class _InMemorySnrSplit:
    """A tiny map-style dataset over a list of per-sample SNR ``Batch`` dicts.

    Duck-types the ``torch.utils.data.Dataset`` surface ``evaluate`` actually uses
    (``__len__`` + iteration), so the synthetic path needs no torch. Each sample is a dict
    with ``iq`` and ``snr_db`` (the regression target), plus any extra meta the fixture carries.
    """

    def __init__(self, samples: Sequence[Batch]) -> None:
        """Wrap an already-materialised list of per-sample dicts."""
        self._samples = list(samples)

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self._samples)

    def __getitem__(self, index: int) -> Batch:
        """Return the sample at ``index`` (map-style access)."""
        return self._samples[index]

    def __iter__(self) -> Iterator[Batch]:
        """Iterate samples in a deterministic, insertion order."""
        return iter(self._samples)


class SnrDataset(Dataset):
    """One SNR-estimation dataset variant, loadable as ``(iq, snr_db, meta)`` samples.

    ``name`` is the dataset id (currently only ``"radioml_2016_10a"``); ``canonical_split_id``
    is the deterministic SNR split id from
    :data:`rfbench.data.prepare.snr_estimation.SNR_CANONICAL_SPLIT_IDS` (indices identical to
    the AMC split, distinct id). Pass ``samples=`` to drive the synthetic in-memory path
    (tests); leave it ``None`` for the lazy cluster path that reads the prepared split + real
    arrays.
    """

    def __init__(
        self,
        name: str,
        *,
        samples: Sequence[Batch] | None = None,
        checksum: str | None = None,
    ) -> None:
        """Bind the dataset id to its canonical SNR split id and (optional) synthetic samples."""
        if name not in SNR_CANONICAL_SPLIT_IDS:
            raise ValueError(
                f"unknown SNR-estimation dataset {name!r}; expected one of "
                f"{sorted(SNR_CANONICAL_SPLIT_IDS)}"
            )
        self.name = name
        self.canonical_split_id = SNR_CANONICAL_SPLIT_IDS[name]
        if checksum is not None:
            self.checksum = checksum
        else:
            resolved = _resolve_committed_checksum(name, self.canonical_split_id)
            self.checksum = resolved if resolved is not None else _PLACEHOLDER_CHECKSUM
        self._samples = None if samples is None else list(samples)

    def download(self, cache: Path | None = None) -> None:
        """Fetch raw RadioML data (delegated to the AMC download layer; identical source).

        Never called in unit tests: the synthetic path injects ``samples`` instead. The
        concrete fetch lives in ``rfbench.data.download.amc_radioml`` and writes NO git-tracked
        files -- SNR estimation reuses the exact same download as AMC.
        """
        raise NotImplementedError(
            "SNR-estimation download reuses AMC's: run rfbench.data.download.amc_radioml on the "
            "cluster; unit tests use the in-memory `samples=` path instead."
        )

    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build the canonical SNR split (delegates to ``rfbench.data.prepare.snr_estimation``).

        Extracts ``(modulation, snr_db)`` labels from the cached RadioML pickle (lazy numpy via
        the AMC loader) and stratifies 80/10/10 by ``(modulation x snr)`` at ``seed`` -- the
        SAME construction as AMC, written under the SNR id. Never called in unit tests; the
        real label extraction requires the dataset + heavy deps.
        """
        from rfbench.data.prepare.amc import load_radioml_labels
        from rfbench.data.prepare.snr_estimation import prepare_snr_estimation

        labels = load_radioml_labels(self.name)  # type: ignore[arg-type]
        split, _manifest = prepare_snr_estimation(
            self.name, out_dir="leaderboard", labels=labels, seed=seed
        )
        self.canonical_split_id = split.canonical_split_id
        self.checksum = split.checksum
        return split

    def load(
        self,
        split: SplitName,
        track: Track | None = None,
    ) -> torch.utils.data.Dataset[Any]:
        """Return the ``(split, track)`` dataset of ``(iq, snr_db, meta)`` samples.

        Synthetic path: returns the injected in-memory split verbatim (ignoring ``split`` /
        ``track`` -- SNR estimation has a single track). Cluster path: lazily loads the
        prepared indices + real arrays and materialises the per-sample dicts. The returned
        object duck-types ``torch.utils.data.Dataset`` (``__len__`` + iteration).
        """
        if self._samples is not None:
            return _InMemorySnrSplit(self._samples)
        return self._load_from_disk(split)

    def _load_from_disk(self, split: SplitName) -> _InMemorySnrSplit:
        """Materialise ``(iq, snr_db)`` samples from the prepared SNR split (cluster-only).

        Reads the versioned SNR ``.idx.json`` for ``split`` then slices the cached RadioML IQ
        arrays with a LAZY numpy import (delegated to the AMC array loader, which the SNR split
        indexes by construction). The flat sample order MUST match
        ``rfbench.data.prepare.amc``'s label flattening so the split indices align. Never
        exercised in the dep-free unit venv (needs the real dataset + heavy deps).
        """
        from rfbench.tasks.amc.dataset import _load_amc_arrays

        indices = self._read_split_indices(split)
        iq_all, _mods, snrs = _load_amc_arrays(self.name)
        samples: list[Batch] = [{"iq": iq_all[i], "snr_db": float(snrs[i])} for i in indices]
        return _InMemorySnrSplit(samples)

    def _read_split_indices(self, split: SplitName) -> list[int]:
        """Return the item indices for ``split`` from the versioned SNR ``.idx.json``."""
        import json

        idx_path = _find_split_index(self.name, self.canonical_split_id)
        if idx_path is None:
            raise FileNotFoundError(
                f"no SNR split index for {self.name!r} ({self.canonical_split_id}); run "
                "`rfbench data prepare` first so leaderboard/splits/<dataset>/*.idx.json exists."
            )
        doc = json.loads(idx_path.read_text(encoding="utf-8"))
        indices = doc.get("indices", {}).get(split)
        if indices is None:
            raise KeyError(f"split {split!r} absent from {idx_path.name}")
        return [int(i) for i in indices]


def _find_split_index(name: str, split_id: str) -> Path | None:
    """Locate ``leaderboard/splits/<name>/<split_id>.idx.json`` by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "leaderboard" / "splits" / name / f"{split_id}.idx.json"
        if candidate.is_file():
            return candidate
    return None


def _resolve_committed_checksum(name: str, split_id: str) -> str | None:
    """Return the committed SNR split's ``split_checksum`` from its manifest, or ``None``.

    Lets a disk-backed :class:`SnrDataset` report the real split checksum (so ``evaluate``
    writes a PR-ready ``result.json``) without re-running ``prepare``. Returns ``None`` when no
    manifest is committed for ``split_id`` (e.g. the synthetic dep-free path). Never raises: a
    missing/malformed manifest degrades to ``None``.
    """
    import json

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "leaderboard" / "splits" / name / f"{split_id}.manifest.json"
        if candidate.is_file():
            try:
                doc = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            checksum = doc.get("split_checksum")
            return checksum if isinstance(checksum, str) and checksum else None
    return None


__all__ = ["SnrDataset", "_InMemorySnrSplit"]
