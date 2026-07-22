"""The spectrum-sensing :class:`~rfbench.core.dataset.Dataset` adapter.

One :class:`SpectrumSensingDataset` instance is one occupancy-detection dataset variant
(``deepsense``). It ties the canonical split id + checksum (from
``rfbench.data.prepare.sensing``) to a ``load(split, track)`` that yields ``(iq, label)``
samples. DeepSense is MULTI-LABEL: ``label`` is the length-16 per-subband occupancy vector
(one ``0`` vacant / ``1`` occupied bit per LTE-M sub-band), so the metric scores window×subband
cells. (The synthetic-fixture path may still inject a scalar ``0/1`` label to exercise the
binary code path end-to-end.)

Two loading paths share one adapter (mirrors
:class:`rfbench.tasks.interference_id.dataset.InterferenceDataset`):

* **cluster path** -- :meth:`load` reads the committed ``sensing-deepsense-official-v1`` split
  indices, then the real ``(2, 32)`` raw-IQ windows + 16-band labels from the cached extracted
  DeepSense ``lte_m/*.h5`` tree via a LAZY numpy/h5py loader
  (:func:`rfbench.data.download.spectrum_deepsense.load_deepsense_arrays`), guarded with a clear
  ``pip install rfbench[tasks]`` hint. Never touched by unit tests.
* **synthetic path** -- an in-memory list of per-sample dicts injected at construction
  (``samples=``). :meth:`load` returns it verbatim, so the whole adapter/metric/evaluate path
  runs on pure-Python fixtures with only ``pytest`` installed.

Each per-sample dict is ``{"iq": window, "label": [0/1]*16}`` (or a scalar ``0/1`` in the binary
fixture path) -- consistent with how
:meth:`rfbench.tasks.spectrum_sensing.task.SpectrumSensingTask.build_targets` reads
``batch["label"]``. Module-top imports are stdlib + the frozen core contracts only; numpy stays
lazy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfbench.core.dataset import Dataset
from rfbench.core.splits import SplitManifest
from rfbench.core.types import Batch, SplitName, Track
from rfbench.data.prepare.sensing import CANONICAL_SPLIT_IDS, OFFICIAL_SPLIT_IDS

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data


#: Placeholder checksum used before a split index has been prepared/loaded on disk. Matches the
#: schema pattern ``^sha256:[0-9a-f]{64}$`` so a synthetic-fixture ``result.json`` still
#: validates; the cluster path overwrites it from the on-disk ``.idx.json``.
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


class _InMemorySensingSplit:
    """A tiny map-style dataset over a list of per-sample sensing ``Batch`` dicts.

    Duck-types the ``torch.utils.data.Dataset`` surface ``evaluate`` actually uses (``__len__`` +
    iteration), so the synthetic path needs no torch. Each sample is a dict with ``iq`` and a
    binary ``label`` (plus any extra meta the fixture carries).
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


class SpectrumSensingDataset(Dataset):
    """One occupancy-detection dataset variant, loadable as ``(iq, label)`` samples.

    ``name`` is the dataset id (e.g. ``"deepsense"``); ``canonical_split_id`` is the deterministic
    split id from ``rfbench.data.prepare.sensing.CANONICAL_SPLIT_IDS``. Pass ``samples=`` to drive
    the synthetic in-memory path (tests); leave it ``None`` for the lazy cluster path that reads
    the prepared split + real windows/labels.
    """

    def __init__(
        self,
        name: str = "deepsense",
        *,
        samples: Sequence[Batch] | None = None,
        checksum: str | None = None,
    ) -> None:
        """Bind the dataset id to its canonical split id and (optional) synthetic samples."""
        if name not in CANONICAL_SPLIT_IDS:
            raise ValueError(
                f"unknown sensing dataset {name!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
            )
        self.name = name
        # The COMMITTED split adopts DeepSense's own train/test partition verbatim (official id);
        # the cluster loader reads leaderboard/splits/<name>/<official_id>.idx.json.
        self.canonical_split_id = OFFICIAL_SPLIT_IDS[name]
        self.checksum = checksum if checksum is not None else _PLACEHOLDER_CHECKSUM
        self._samples = None if samples is None else list(samples)

    def download(self, cache: Path | None = None) -> None:
        """Locate the extracted DeepSense tree (delegated to the download layer).

        Never called in unit tests: the synthetic path injects ``samples`` instead. The concrete
        (gated, manual) fetch lives in ``rfbench.data.download.spectrum_deepsense`` and writes NO
        git-tracked files.
        """
        raise NotImplementedError(
            "DeepSense is gated/manual: locate it on the cluster via "
            "rfbench.data.download.spectrum_deepsense.download_deepsense; "
            "unit tests use the in-memory `samples=` path instead."
        )

    def prepare(self, seed: int = 42) -> SplitManifest:
        """Build the canonical split (delegates to ``rfbench.data.prepare.sensing``).

        Adopts DeepSense's OWN published train/test window partition verbatim (official split):
        :func:`rfbench.data.download.spectrum_deepsense.load_deepsense_records` reads the
        ``lte_m/*.h5`` file shapes (lazy h5py) to build the ``{train, val, test}`` index sets, and
        ``prepare_sensing`` writes the committed ``sensing-deepsense-official-v1`` index. Never
        called in unit tests; reading the real shapes requires the dataset + heavy deps.
        """
        from rfbench.data.download.spectrum_deepsense import load_deepsense_records
        from rfbench.data.prepare.sensing import prepare_sensing

        _n_items, official_split = load_deepsense_records()
        split, _manifest = prepare_sensing(
            self.name, out_dir="leaderboard", official_split=official_split, seed=seed
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
        ``track`` -- spectrum sensing has a single ``occupancy`` track). Cluster path: lazily
        loads the prepared indices + real windows/labels and materialises the per-sample dicts.
        The returned object duck-types ``torch.utils.data.Dataset`` (``__len__`` + iteration).
        """
        if self._samples is not None:
            return _InMemorySensingSplit(self._samples)
        return self._load_from_disk(split)

    def _load_from_disk(self, split: SplitName) -> _InMemorySensingSplit:
        """Materialise ``(iq, label)`` samples from the prepared split (cluster-only).

        Reads the committed ``.idx.json`` for ``split`` then slices the cached ``(2, 32)`` windows +
        16-band labels via the lazy loader (numpy/h5py inside). The flat sample order MUST match
        ``rfbench.data.download.spectrum_deepsense.load_deepsense_arrays``'s order (same sorted-file
        + first-``kept`` cap as ``load_deepsense_records``) so the split indices align. Never
        exercised in the dep-free unit venv (needs the real dataset + heavy deps).
        """
        from rfbench.data.download.spectrum_deepsense import load_deepsense_arrays

        indices = self._read_split_indices(split)
        pairs = load_deepsense_arrays()
        samples: list[Batch] = [{"iq": pairs[i][0], "label": list(pairs[i][1])} for i in indices]
        return _InMemorySensingSplit(samples)

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


__all__ = ["SpectrumSensingDataset", "_InMemorySensingSplit"]
