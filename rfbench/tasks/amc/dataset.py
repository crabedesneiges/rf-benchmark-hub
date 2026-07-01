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
        a LAZY numpy/h5py import guarded by a clear install hint. The flat sample order MUST
        match ``rfbench.data.prepare.amc``'s label flattening so the split indices align.
        Never exercised in the dep-free unit venv (needs the real dataset + heavy deps).
        """
        indices = self._read_split_indices(split)
        iq_all, mods, snrs = _load_amc_arrays(self.name)
        classes = sorted(set(mods))  # deterministic, consistent for train & test loads
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples: list[Batch] = [
            {"iq": iq_all[i], "label": class_to_idx[mods[i]], "snr_db": snrs[i]} for i in indices
        ]
        return _InMemoryAmcSplit(samples)

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


def _load_amc_arrays(name: str) -> tuple[list[Any], list[str], list[int]]:
    """Load flat ``(iq_rows, mod_names, snrs)`` from the cached raw AMC data (lazy numpy/h5py).

    The iteration order mirrors ``rfbench.data.prepare.amc`` exactly so the versioned split
    indices line up: RadioML 2016 iterates the pickle ``dict.items()`` then each cell row;
    RadioML 2018 iterates the HDF5 rows in stored order. Cluster-only.
    """
    from rfbench.data.prepare._common import resolve_cache_dir

    cache_dir = resolve_cache_dir(None)
    if name == "radioml_2016_10a":
        return _load_radioml2016_arrays(cache_dir)
    if name == "radioml_2018_01a":
        return _load_radioml2018_arrays(cache_dir)
    raise NotImplementedError(
        f"on-disk array loading for {name!r} is not wired (Sig53 is generation-only/blocked)."
    )


def _load_radioml2016_arrays(cache_dir: Path) -> tuple[list[Any], list[str], list[int]]:
    """Flatten the RadioML 2016.10a pickle to per-item ``(iq, mod, snr)`` in prepare order."""
    import pickle

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "loading the real AMC arrays needs numpy; install it with `pip install rfbench[tasks]`."
        ) from exc

    ds_dir = cache_dir / "radioml_2016_10a"
    candidates = ("RML2016.10a_dict.pkl", "RML2016.10a_dict_optimized.pkl")
    path = next((ds_dir / n for n in candidates if (ds_dir / n).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"RadioML 2016.10a pickle not found in {ds_dir} ({list(candidates)})."
        )
    with path.open("rb") as fh:
        table = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset file

    iq: list[Any] = []
    mods: list[str] = []
    snrs: list[int] = []
    for (mod, snr), block in table.items():  # SAME order as _expand_radioml2016_table
        arr = np.asarray(block, dtype=np.float32)  # (N, 2, 128)
        for row in arr:
            iq.append(row)
            mods.append(str(mod))
            snrs.append(int(str(snr)))
    return iq, mods, snrs


def _load_radioml2018_arrays(cache_dir: Path) -> tuple[list[Any], list[str], list[int]]:
    """Flatten the RadioML 2018.01a HDF5 to per-item ``(iq(2,1024), mod, snr)`` in stored order."""
    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "loading RadioML 2018.01a needs numpy + h5py; install with `pip install rfbench[data]`."
        ) from exc

    from rfbench.data.prepare.amc import _RADIOML2018_MODS

    path = cache_dir / "radioml_2018_01a" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"RadioML 2018.01a HDF5 not found at {path}.")
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["X"][:], dtype=np.float32)  # (N, 1024, 2)
        onehot = np.asarray(handle["Y"][:])
        snr = np.asarray(handle["Z"][:]).reshape(-1)
    iq = [row.T for row in x]  # (1024, 2) -> (2, 1024) to match the (channels, time) layout
    mods = [_RADIOML2018_MODS[int(i)] for i in onehot.argmax(axis=1)]
    snrs = [int(s) for s in snr]
    return iq, mods, snrs


__all__ = ["AmcDataset", "_InMemoryAmcSplit"]
