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
        """Materialise ``(iq, label)`` samples from the prepared split (cluster-only).

        Reads the versioned ``.idx.json`` for ``split`` then slices the cached IQ windows with
        a LAZY numpy import guarded by a clear install hint. The flat sample order MUST match
        ``rfbench.data.prepare.protocol``'s label flattening so the split indices align.
        Never exercised in the dep-free unit venv (needs the real dataset + heavy deps).
        """
        indices = self._read_split_indices(split)
        iq_all, class_names = _load_protocol_arrays(self.name)
        class_to_idx = {c: i for i, c in enumerate(PROTOCOL_CLASSES)}
        samples: list[Batch] = [
            {"iq": iq_all[i], "label": class_to_idx[class_names[i]]} for i in indices
        ]
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


def _load_protocol_arrays(name: str) -> tuple[list[Any], list[str]]:
    """Load flat ``(iq_rows, class_names)`` from the cached T-PRIME captures (lazy numpy).

    The iteration order mirrors ``rfbench.data.prepare.protocol`` exactly so the versioned
    split indices line up: the class subtrees are walked in :data:`PROTOCOL_CLASSES` order,
    and within each class the ``.bin`` captures are read in sorted-path order (the same order
    the label loader enumerates). Cluster-only.

    Each raw-IQ ``.bin`` capture is read as flat interleaved float32 ``[I0, Q0, I1, Q1, ...]``
    into a ``(2, L)`` window, I on row 0 / Q on row 1, matching the T-PRIME channel-first
    layout the ``tprime`` baseline consumes.

    TODO (cluster): confirm the on-disk dtype/interleaving of the T-PRIME ``.bin`` files
    (float32 interleaved is assumed here; some captures store complex64). If a capture is a
    long recording rather than one window, tile it with the SAME window length + stride the
    label loader uses so the committed split indices remain aligned.
    """
    if name != "tprime_wifi4":
        raise NotImplementedError(f"on-disk array loading for {name!r} is not wired.")

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "loading the real T-PRIME arrays needs numpy; install it with "
            "`pip install rfbench[tasks]`."
        ) from exc

    from rfbench.data.prepare._common import resolve_cache_dir
    from rfbench.data.prepare.protocol import (
        _class_dir_names,
        _iter_class_files,
        _resolve_dataset_root,
    )

    ds_dir = _resolve_dataset_root(resolve_cache_dir(None))
    if ds_dir is None:
        raise FileNotFoundError(
            "tprime_wifi4 not found in the cache; run the download step first "
            "(rfbench.data.download.protocol_tprime.download_tprime_wifi4)."
        )
    dir_names = _class_dir_names()
    iq: list[Any] = []
    class_names: list[str] = []
    for class_name in PROTOCOL_CLASSES:
        for path in _iter_class_files(ds_dir, dir_names[class_name]):
            # Read one raw-IQ file into a (2, L) float32 array (I on row 0, Q on row 1).
            # Handles numpy .npy and a flat interleaved binary [I0, Q0, I1, Q1, ...]
            # (.bin/.iq/.dat/.sigmf-data). Cluster-only; confirm the actual format there.
            suffix = path.suffix.lower()
            if suffix == ".npy":
                arr = np.asarray(np.load(path), dtype=np.float32)
                window = arr if arr.shape[0] == 2 else arr.reshape(2, -1)
            else:  # flat interleaved [I, Q, ...] -> (2, L)
                window = np.fromfile(path, dtype=np.float32).reshape(-1, 2).T
            iq.append(window)
            class_names.append(class_name)
    return iq, class_names


__all__ = ["ProtocolDataset", "_InMemoryProtocolSplit"]
