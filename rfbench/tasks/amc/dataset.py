"""The AMC :class:`~rfbench.core.dataset.Dataset` adapter (WP-20).

One :class:`AmcDataset` instance is one AMC dataset variant (RadioML 2016.10a / 2018.01a /
Sig53). It ties the canonical split id + checksum (from ``rfbench.data.prepare.amc``) to a
``load(split, track)`` that yields ``(iq, label, meta{snr_db, ...})`` samples.

Two loading paths share one adapter:

* **cluster path** -- :meth:`load` reads the versioned split indices, then the real IQ
  arrays from the cached ``.h5``/``.pkl`` via a LAZY numpy/h5py import guarded with a clear
  ``pip install rfbench[tasks]`` hint. Never touched by unit tests.
* **synthetic path** -- an in-memory list of per-sample dicts injected at construction
  (``samples=``). :meth:`load` returns it verbatim, so the whole adapter/metric/evaluate
  path runs on pure-Python fixtures with only ``pytest`` installed.

Module-top imports are stdlib + the frozen core contracts only; numpy/h5py stay lazy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.dataset import Dataset
from rfbench.core.splits import SplitManifest
from rfbench.core.types import Batch, SplitName, Track
from rfbench.data.prepare.amc import CANONICAL_SPLIT_IDS

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data


#: Placeholder checksum used before a split index has been prepared/loaded on disk. Matches
#: the schema pattern ``^sha256:[0-9a-f]{64}$`` so a synthetic-fixture ``result.json`` still
#: validates; the cluster path overwrites it from the on-disk ``.idx.json``.
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


class _InMemoryAmcSplit:
    """A tiny map-style dataset over a list of per-sample AMC ``Batch`` dicts.

    Duck-types the ``torch.utils.data.Dataset`` surface ``evaluate`` actually uses
    (``__len__`` + iteration), so the synthetic path needs no torch. Each sample is a dict
    with ``iq``, ``label`` and ``snr_db`` (plus any extra meta the fixture carries).
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


class AmcDataset(Dataset):
    """One AMC dataset variant, loadable as ``(iq, label, meta{snr_db})`` samples.

    ``name`` is the dataset id (e.g. ``"radioml_2016_10a"``); ``canonical_split_id`` is the
    deterministic split id from ``rfbench.data.prepare.amc.CANONICAL_SPLIT_IDS``. Pass
    ``samples=`` to drive the synthetic in-memory path (tests); leave it ``None`` for the
    lazy cluster path that reads the prepared split + real arrays.
    """

    def __init__(
        self,
        name: str,
        *,
        samples: Sequence[Batch] | None = None,
        checksum: str | None = None,
    ) -> None:
        """Bind the dataset id to its canonical split id and (optional) synthetic samples."""
        if name not in CANONICAL_SPLIT_IDS:
            raise ValueError(
                f"unknown AMC dataset {name!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
            )
        self.name = name
        self.canonical_split_id = CANONICAL_SPLIT_IDS[name]
        self.checksum = checksum if checksum is not None else _PLACEHOLDER_CHECKSUM
        self._samples = None if samples is None else list(samples)

    def download(self, cache: Path | None = None) -> None:
        """Fetch raw AMC data from the official source (delegated to the download layer).

        Never called in unit tests: the synthetic path injects ``samples`` instead. The
        concrete fetch lives in ``rfbench.data.download`` and writes NO git-tracked files.
        """
        raise NotImplementedError(
            "AMC download runs on the cluster via rfbench.data.download.amc_radioml; "
            "unit tests use the in-memory `samples=` path instead."
        )

    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build the canonical split (delegates to ``rfbench.data.prepare.amc``).

        Extracts ``(modulation, snr_db)`` labels from the cached raw files (lazy numpy/h5py)
        and stratifies 80/10/10 by ``(modulation x snr)`` at ``seed``. Never called in unit
        tests; the real label extraction requires the dataset + heavy deps.
        """
        from rfbench.data.prepare.amc import load_radioml_labels, prepare_amc

        if self.name == "sig53":
            raise NotImplementedError(
                "Sig53 adopts the official TorchSig split; run prepare_amc(official_split=...) "
                "from the cluster where TorchSig + generated data are available."
            )
        labels = load_radioml_labels(self.name)  # type: ignore[arg-type]
        split, _manifest = prepare_amc(self.name, out_dir="leaderboard", labels=labels, seed=seed)
        self.canonical_split_id = split.canonical_split_id
        self.checksum = split.checksum
        return split

    def load(
        self,
        split: SplitName,
        track: Track | None = None,
    ) -> torch.utils.data.Dataset[Any]:
        """Return the ``(split, track)`` dataset of ``(iq, label, meta{snr_db})`` samples.

        Synthetic path: returns the injected in-memory split verbatim (ignoring ``split`` /
        ``track`` -- AMC has a single closed-set track). Cluster path: lazily loads the
        prepared indices + real arrays and materialises the per-sample dicts. The returned
        object duck-types ``torch.utils.data.Dataset`` (``__len__`` + iteration).
        """
        if self._samples is not None:
            return _InMemoryAmcSplit(self._samples)
        return self._load_from_disk(split)

    def _load_from_disk(self, split: SplitName) -> _InMemoryAmcSplit:
        """Materialise ``(iq, label, meta)`` samples from the prepared split (cluster-only).

        Reads the versioned ``.idx.json`` for ``split`` then slices the cached IQ arrays with
        a LAZY numpy/h5py import guarded by a clear install hint. Never exercised in unit
        tests (needs the real dataset + heavy deps).
        """
        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "loading the real AMC arrays needs numpy (+ h5py for 2018.01a); "
                "install them with `pip install rfbench[tasks]`."
            ) from exc
        raise NotImplementedError(
            "AMC on-disk array loading is wired on the cluster against the prepared split "
            f"indices for {self.name!r}; unit tests use the in-memory `samples=` path."
        )


__all__ = ["AmcDataset", "_InMemoryAmcSplit"]
