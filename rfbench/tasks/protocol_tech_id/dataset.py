"""The protocol-tech-ID :class:`~rfbench.core.dataset.Dataset` adapter.

One :class:`ProtocolDataset` instance is one WiFi-standard dataset variant
(``tprime_wifi4``). It ties the canonical split id + checksum (from
``rfbench.data.prepare.protocol``) to a ``load(split, track)`` that yields
``(iq, label)`` samples, where ``label`` is the integer index of one of the four
802.11 standards (``802.11b``, ``802.11g``, ``802.11n``, ``802.11ax``).

Two loading paths share one adapter:

* **cluster path** -- :meth:`load` reads the versioned split indices, then the real IQ
  windows from the cached extracted ``.bin`` captures via a LAZY numpy import guarded with a
  clear ``pip install rfbench[tasks]`` hint. Never touched by unit tests.
* **synthetic path** -- an in-memory list of per-sample dicts injected at construction
  (``samples=``). :meth:`load` returns it verbatim, so the whole adapter/metric/evaluate
  path runs on pure-Python fixtures with only ``pytest`` installed.

Module-top imports are stdlib + the frozen core contracts only; numpy stays lazy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.dataset import Dataset
from rfbench.core.splits import SplitManifest
from rfbench.core.types import Batch, SplitName, Track
from rfbench.data.prepare.protocol import CANONICAL_SPLIT_IDS, PROTOCOL_CLASSES

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data


#: Placeholder checksum used before a split index has been prepared/loaded on disk. Matches
#: the schema pattern ``^sha256:[0-9a-f]{64}$`` so a synthetic-fixture ``result.json`` still
#: validates; the cluster path overwrites it from the on-disk ``.idx.json``.
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


class _InMemoryProtocolSplit:
    """A tiny map-style dataset over a list of per-sample protocol ``Batch`` dicts.

    Duck-types the ``torch.utils.data.Dataset`` surface ``evaluate`` actually uses
    (``__len__`` + iteration), so the synthetic path needs no torch. Each sample is a dict
    with ``iq`` and ``label`` (plus any extra meta the fixture carries).
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


class ProtocolDataset(Dataset):
    """One WiFi-standard dataset variant, loadable as ``(iq, label)`` samples.

    ``name`` is the dataset id (e.g. ``"tprime_wifi4"``); ``canonical_split_id`` is the
    deterministic split id from ``rfbench.data.prepare.protocol.CANONICAL_SPLIT_IDS``.
    Pass ``samples=`` to drive the synthetic in-memory path (tests); leave it ``None`` for
    the lazy cluster path that reads the prepared split + real arrays.
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
                f"unknown protocol dataset {name!r}; expected one of "
                f"{sorted(CANONICAL_SPLIT_IDS)}"
            )
        self.name = name
        self.canonical_split_id = CANONICAL_SPLIT_IDS[name]
        self.checksum = checksum if checksum is not None else _PLACEHOLDER_CHECKSUM
        self._samples = None if samples is None else list(samples)

    def download(self, cache: Path | None = None) -> None:
        """Fetch the raw T-PRIME data from the DRS (delegated to the download layer).

        Never called in unit tests: the synthetic path injects ``samples`` instead. The
        concrete fetch lives in ``rfbench.data.download`` and writes NO git-tracked files.
        """
        raise NotImplementedError(
            "protocol_tech_id download runs on the cluster via "
            "rfbench.data.download.protocol_tprime.download_tprime_wifi4; "
            "unit tests use the in-memory `samples=` path instead."
        )

    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build the canonical split (delegates to ``rfbench.data.prepare.protocol``).

        Extracts the per-item class labels from the cached extracted ``.bin`` captures (lazy
        numpy) and stratifies 80/10/10 by class at ``seed``. Never called in unit tests; the
        real label extraction requires the dataset + heavy deps.
        """
        from rfbench.data.prepare.protocol import (
            load_protocol_labels,
            prepare_protocol,
        )

        labels = load_protocol_labels(self.name)  # type: ignore[arg-type]
        split, _manifest = prepare_protocol(
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
        """Return the ``(split, track)`` dataset of ``(iq, label)`` samples.

        Synthetic path: returns the injected in-memory split verbatim (ignoring ``split`` /
        ``track`` -- protocol-tech-ID has a single closed-set track). Cluster path: lazily
        loads the prepared indices + real arrays and materialises the per-sample dicts. The
        returned object duck-types ``torch.utils.data.Dataset`` (``__len__`` + iteration).
        """
        if self._samples is not None:
            return _InMemoryProtocolSplit(self._samples)
        return self._load_from_disk(split)

    def _load_from_disk(self, split: SplitName) -> _InMemoryProtocolSplit:
        """Materialise ``(iq, label)`` WINDOW samples from the prepared split (cluster-only).

        Reads the versioned ``.idx.json`` RECORDING indices for ``split`` then tiles each
        selected recording into fixed-length windows with a LAZY numpy import (guarded by a clear
        install hint). The recording enumeration MUST equal
        ``rfbench.data.prepare.protocol._iter_recording_files`` -- the same order the label loader
        used to build the split -- so recording index ``i`` denotes the same capture on both
        sides. Because the split partitions RECORDINGS (never windows), every window of a
        recording lands in one split, so no window leaks across train/test. Never exercised in the
        dep-free unit venv (needs the real dataset + heavy deps).
        """
        from rfbench.data.prepare._common import resolve_cache_dir
        from rfbench.data.prepare.protocol import (
            TPRIME_WINDOW_LEN,
            WINDOWS_PER_RECORDING,
            _iter_recording_files,
            _resolve_dataset_root,
        )

        indices = self._read_split_indices(split)
        ds_dir = _resolve_dataset_root(resolve_cache_dir(None))
        if ds_dir is None:
            raise FileNotFoundError(
                "tprime_wifi4 not found in the cache; run the download step first "
                "(rfbench.data.download.protocol_tprime.download_tprime_wifi4)."
            )
        recordings = _iter_recording_files(ds_dir)
        class_to_idx = {c: i for i, c in enumerate(PROTOCOL_CLASSES)}
        samples: list[Batch] = []
        for i in indices:
            path, class_name = recordings[i]
            label = class_to_idx[class_name]
            for window in _read_windows(path, TPRIME_WINDOW_LEN, WINDOWS_PER_RECORDING):
                samples.append({"iq": window, "label": label})
        return _InMemoryProtocolSplit(samples)

    def _read_split_indices(self, split: SplitName) -> list[int]:
        """Return the item indices for ``split`` from the versioned ``.idx.json`` (repo tree)."""
        import json

        idx_path = _find_split_index(self.name, self.canonical_split_id)
        if idx_path is None:
            raise FileNotFoundError(
                f"no split index for {self.name!r} ({self.canonical_split_id}); run "
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


def _read_windows(path: Path, window_len: int, max_windows: int) -> list[Any]:
    """Tile one raw-IQ capture into up to ``max_windows`` ``(2, window_len)`` float32 windows.

    Reads the capture with a LAZY numpy import (guarded by a clear install hint). ``.bin`` /
    ``.iq`` / ``.dat`` / ``.sigmf-data`` are read as **native complex128** (CONFIRMED 2026-07
    against the official loader ``t-prime/preprocessing/TPrime_dataset.py``:
    ``np.fromfile(path, dtype=np.complex128)`` -- NOT interleaved float32 pairs) and split into I
    (row 0) / Q (row 1); ``.npy`` is read as whatever ``(2, L)`` float32 layout was saved. DS 3.0
    captures are LONG recordings (~198k samples), so each is tiled into windows at the offsets
    from :func:`rfbench.data.prepare.protocol._window_offsets` (deterministic, spread evenly
    across the recording); a capture shorter than one window is zero-padded. All windows of a
    capture carry that recording's single class label -- and, since the split is over recordings,
    they never straddle train/test. Cluster-only (needs the real data + numpy).
    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "loading the real T-PRIME arrays needs numpy; install it with "
            "`pip install rfbench[data]`."
        ) from exc

    from rfbench.data.prepare.protocol import _window_offsets

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
        iq2 = arr if arr.ndim == 2 and arr.shape[0] == 2 else arr.reshape(2, -1)
    else:  # native complex128 -> (2, L) float32 (I on row 0, Q on row 1)
        complex_iq = np.fromfile(path, dtype=np.complex128)
        iq2 = np.stack([complex_iq.real, complex_iq.imag]).astype(np.float32)
    n_samples = int(iq2.shape[1])

    windows: list[Any] = []
    for offset in _window_offsets(n_samples, window_len, max_windows):
        chunk = iq2[:, offset : offset + window_len]
        if chunk.shape[1] < window_len:
            pad = np.zeros((2, window_len - chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)
        windows.append(np.ascontiguousarray(chunk, dtype=np.float32))
    return windows


__all__ = ["ProtocolDataset", "_InMemoryProtocolSplit"]
