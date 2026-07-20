"""SEI dataset adapter (WP-21): binds a canonical SEI split to the eval loop.

:class:`SeiDataset` implements the frozen :class:`rfbench.core.dataset.Dataset` contract
for the WiSig / ORACLE fingerprinting datasets. Its :meth:`load` yields per-sample
canonical batches ``{"iq": ..., "label": tx_label, "meta": {"rx": ..., "day": ...}}`` for a
given ``(split, track)``, where ``track`` selects one of the SEI conditions
(``closed_set`` / ``cross_receiver`` / ``cross_day``) or the ``open_set`` protocol.

The REAL loader (reading IQ from the WiSig files on the cluster) is lazy and never
exercised in the dep-free unit venv: it lives in :meth:`_load_from_cache`, which reads the
versioned canonical split indices produced by :mod:`rfbench.data.prepare.sei` and then the
real IQ from ``$RFBENCH_CACHE/wisig/ManyTx.pkl``, importing numpy behind a clear
``rfbench[tasks]`` hint. Crucially, the on-disk IQ is flattened in the SAME
``data[tx_i][rx_i][day_i][eq_i]`` record order that
:func:`rfbench.data.prepare.sei.extract_wisig_records` used, so the committed split indices
line up with the materialised samples (mirrors the AMC on-disk adapter). Unit tests inject
an in-memory sample list via ``samples=`` so the whole adapter runs dependency-free; a
numpy-guarded regression test asserts the array/record order alignment.

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
from rfbench.data.prepare.sei import CANONICAL_SPLIT_IDS, SeiRecord

if TYPE_CHECKING:  # pragma: no cover - typing-only import, never executed at runtime
    import torch.utils.data

#: The SEI conditions/tracks this adapter serves. ``open_set`` has its OWN canonical split
#: (whole transmitters held out as novel/impostor identities) and is scored with the open-set
#: metrics; the closed-set conditions map to their own stratified/grouped split ids.
_TRACK_TO_CONDITION: dict[Track, str] = {
    "closed_set": "closed_set",
    "cross_receiver": "cross_receiver",
    "cross_day": "cross_day",
    "open_set": "open_set",
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

    def __getitem__(self, index: int) -> Batch:
        """Return the ``index``-th per-sample batch (map-style access for a ``DataLoader``).

        Needed so the SEI training loop (:mod:`rfbench.training_sei`) can wrap this split in a
        ``torch.utils.data.DataLoader`` (which requires ``__getitem__``); the eval loop only
        iterates, but map-style access is the more general contract.
        """
        return self._samples[index]

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
    ) -> _InMemorySplit:  # pragma: no cover - cluster-only
        """Real loader: read canonical indices + IQ from the cluster cache (lazy numpy).

        Never called in the dep-free unit venv. Reads the versioned
        ``leaderboard/splits/<name>/<canonical_split_id>.idx.json`` for the requested
        ``split`` partition, loads the real IQ arrays + ``(tx, rx, day)`` records from
        ``$RFBENCH_CACHE/<name>/ManyTx.pkl`` in the SAME record order
        :func:`rfbench.data.prepare.sei.extract_wisig_records` used (so the committed split
        indices align), then materialises one ``{"iq", "label", "meta": {"rx", "day"}}``
        canonical batch per index. ``label`` is the transmitter id mapped to a dense class
        index via a deterministic (sorted-id) map, consistent across train/val/test loads
        exactly like the AMC on-disk adapter.
        """
        iq_all, records = _load_sei_arrays(self.name)
        if self._track == "open_set":
            # Open-set: derive the gallery (KNOWN tx = those in the train partition) and tag
            # each probe genuine (in-gallery) vs impostor (novel/held-out) -- see
            # rfbench.data.prepare.sei._prepare_open_set. The class map spans known tx only.
            train_indices = self._read_split_indices("train")
            split_indices = self._read_split_indices(split)
            return _InMemorySplit(open_set_samples(iq_all, records, split_indices, train_indices))
        indices = self._read_split_indices(split)
        tx_ids = sorted({_tx_key(rec[0]) for rec in records})
        class_of = {tx: i for i, tx in enumerate(tx_ids)}
        samples: list[Batch] = [
            {
                "iq": iq_all[i],
                "label": class_of[_tx_key(records[i][0])],
                "meta": {"rx": records[i][1], "day": records[i][2]},
            }
            for i in indices
        ]
        return _InMemorySplit(samples)

    def _read_split_indices(self, split: SplitName) -> list[int]:  # pragma: no cover - cluster-only
        """Return the item indices for ``split`` from the versioned ``.idx.json`` (repo tree)."""
        import json

        idx_path = _find_split_index(self.name, self.canonical_split_id)
        if idx_path is None:
            raise FileNotFoundError(
                f"no split index for {self.name!r} ({self.canonical_split_id}); run "
                "`rfbench data prepare` first so leaderboard/splits/<dataset>/*.idx.json exists."
            )
        doc = json.loads(idx_path.read_text(encoding="utf-8"))
        self.checksum = doc.get("checksum", self.checksum)
        indices = doc.get("indices", {}).get(split)
        if indices is None:
            raise KeyError(f"split {split!r} absent from {idx_path.name}")
        return [int(i) for i in indices]


def _find_split_index(name: str, split_id: str) -> Path | None:  # pragma: no cover - cluster-only
    """Locate ``leaderboard/splits/<name>/<split_id>.idx.json`` by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "leaderboard" / "splits" / name / f"{split_id}.idx.json"
        if candidate.is_file():
            return candidate
    return None


def _tx_key(tx_id: object) -> tuple[int, str]:
    """Type-agnostic canonical key for a transmitter id (mixes int/str safely).

    Maps each id to ``(type_rank, str_value)`` so a set of mixed int/str transmitter ids
    has a deterministic total order and ``sorted(...)`` never raises ``TypeError`` -- the
    same trick :func:`rfbench.data.prepare.sei._norm_group` uses for grouped conditions,
    keeping the class-index assignment stable across train/val/test loads.
    """
    if isinstance(tx_id, str):
        return (0, tx_id)
    return (1, f"{tx_id!r}")


def open_set_samples(
    iq_all: Sequence[Any],
    records: Sequence[SeiRecord],
    split_indices: Sequence[int],
    train_indices: Sequence[int],
) -> list[Batch]:
    """Materialise open-set probes with a genuine/impostor flag (pure Python, unit-tested).

    The gallery is the set of KNOWN transmitters -- those present in the open-set split's
    ``train`` partition. Each probe in ``split_indices`` becomes a canonical batch carrying
    ``genuine == 1`` if its transmitter is in the gallery (in-gallery probe) or ``0`` if it
    is a held-out novel identity (impostor). ``label`` is the dense gallery-class index for a
    known tx (over the sorted known set, consistent with what the model was trained on) and
    ``-1`` for an impostor (unused by :class:`~rfbench.tasks.sei.metrics.OpenSetMetric`, which
    reads only the score and the genuine flag). No numpy: ``iq_all`` rows are passed through
    verbatim, so this runs on stdlib fixtures.
    """
    known = {_tx_key(records[i][0]) for i in train_indices}
    class_of = {tx: index for index, tx in enumerate(sorted(known))}
    samples: list[Batch] = []
    for i in split_indices:
        tx = _tx_key(records[i][0])
        samples.append(
            {
                "iq": iq_all[i],
                "label": class_of.get(tx, -1),
                "genuine": 1 if tx in known else 0,
                "meta": {"rx": records[i][1], "day": records[i][2]},
            }
        )
    return samples


def _load_sei_arrays(
    name: str,
) -> tuple[list[Any], list[SeiRecord]]:  # pragma: no cover - cluster-only
    """Dispatch to the per-dataset flat ``(iq_rows, records)`` loader (WiSig / ORACLE / POWDER).

    Each loader returns per-signal ``(window, 2)`` IQ rows + ``(tx, rx, day)`` records in the
    EXACT order the matching :mod:`rfbench.data.prepare.sei` extractor used, so the committed
    split indices line up element-for-element. Cluster-only (needs the real data + numpy).
    """
    if name == "wisig":
        return _load_wisig_arrays(name)
    if name == "powder":
        return _load_powder_arrays(name)
    if name == "oracle":
        return _load_oracle_arrays(name)
    raise NotImplementedError(
        f"on-disk IQ loading is wired for WiSig / ORACLE / POWDER only; {name!r} has no loader."
    )


def _load_oracle_arrays(
    name: str,
) -> tuple[list[Any], list[SeiRecord]]:  # pragma: no cover - cluster-only
    """Load flat ``(iq_rows, records)`` from the cached ORACLE SigMF captures (lazy numpy).

    Mirrors :func:`rfbench.data.prepare.sei.load_oracle_records` EXACTLY (same
    ``sorted(rglob('*.sigmf-data'))`` file order, same ``_ORACLE_WINDOW`` framing, same
    ``_oracle_tx_id`` identity), emitting one ``(window, 2)`` row + one ``(tx_id, None, None)``
    record per capture window -- so ``iq[k]`` corresponds to ``records[k]`` and both align with
    the committed ORACLE split indices. ORACLE has a single fixed receiver, so ``rx``/``day`` are
    ``None`` (closed-set identity only).
    """
    import json  # stdlib

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading SEI IQ needs numpy; install it with `pip install rfbench[tasks]`."
        ) from exc

    from rfbench.data.prepare._common import resolve_cache_dir
    from rfbench.data.prepare.sei import (
        _ORACLE_WINDOW,
        _ORACLE_WINDOWS_PER_CAPTURE,
        _oracle_tx_id,
        _sigmf_np_dtype,
    )

    cache_dir = resolve_cache_dir(None)
    root = cache_dir / name
    if not root.is_dir():
        raise FileNotFoundError(
            f"ORACLE not found at {root}; run the download step first "
            "(rfbench.data.download.sei_oracle)."
        )
    window = _ORACLE_WINDOW
    iq: list[Any] = []
    records: list[SeiRecord] = []
    for data_path in sorted(root.rglob("*.sigmf-data")):
        tx_id = _oracle_tx_id(data_path.name)
        meta_path = data_path.with_suffix(".sigmf-meta")
        dtype = _sigmf_np_dtype(np, meta_path, json) if meta_path.exists() else np.float32
        raw = np.fromfile(data_path, dtype=dtype).astype(np.float32)
        n_complex = int(raw.size // 2)
        # Same per-capture cap as load_oracle_records (first k contiguous windows) -> aligned indices.
        n_windows = min(_ORACLE_WINDOWS_PER_CAPTURE, n_complex // window)
        if n_windows == 0:
            continue
        frames = raw[: n_windows * window * 2].reshape(n_windows, window, 2)
        for row in frames:  # (window, 2) per capture window
            iq.append(row)
            records.append((tx_id, None, None))
    if not records:
        raise FileNotFoundError(f"no ORACLE .sigmf-data captures found under {root}.")
    return iq, records


def _load_powder_arrays(
    name: str,
) -> tuple[list[Any], list[SeiRecord]]:  # pragma: no cover - cluster-only
    """Load flat ``(iq_rows, records)`` from the cached POWDER SigMF captures (lazy numpy).

    Walks ``$RFBENCH_CACHE/powder/*.sigmf-data`` in sorted file order (the SAME order
    :func:`rfbench.data.prepare.sei.load_powder_records` uses), reading each recording's
    interleaved I/Q, slicing it into non-overlapping ``_POWDER_WINDOW``-sample frames and emitting
    one ``(window, 2)`` row + one ``(device_id, None, day_id)`` record per frame -- so ``iq[k]``
    corresponds to ``records[k]`` and both align with the committed split indices.
    """
    import json  # stdlib

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading SEI IQ needs numpy; install it with `pip install rfbench[tasks]`."
        ) from exc

    from rfbench.data.prepare._common import resolve_cache_dir
    from rfbench.data.prepare.sei import _powder_ids, _sigmf_np_dtype

    cache_dir = resolve_cache_dir(None)
    root = cache_dir / name
    if not root.is_dir():
        raise FileNotFoundError(
            f"POWDER not found at {root}; place the SigMF captures there first "
            "(see rfbench.data.download.sei_powder for the manual-download procedure)."
        )
    window = _POWDER_WINDOW
    iq: list[Any] = []
    records: list[SeiRecord] = []
    for data_path in sorted(root.rglob("*.sigmf-data")):
        device_id, day_id = _powder_ids(data_path.name)
        meta_path = data_path.with_suffix(".sigmf-meta")
        dtype = _sigmf_np_dtype(np, meta_path, json) if meta_path.exists() else np.float32
        raw = np.fromfile(data_path, dtype=dtype).astype(np.float32)
        n_complex = int(raw.size // 2)
        n_frames = n_complex // window
        if n_frames == 0:
            continue
        frames = raw[: n_frames * window * 2].reshape(n_frames, window, 2)
        for row in frames:  # (window, 2) per frame
            iq.append(row)
            records.append((device_id, None, day_id))
    return iq, records


#: POWDER slice length (samples per frame); the FM convention (256), fed to the (window, 2) models.
_POWDER_WINDOW = 256


def _load_wisig_arrays(
    name: str,
) -> tuple[list[Any], list[SeiRecord]]:  # pragma: no cover - cluster-only
    """Load flat ``(iq_rows, records)`` from the cached WiSig ManyTx pickle (lazy numpy).

    Reads ``$RFBENCH_CACHE/<name>/ManyTx.pkl`` and flattens the 5-level nested ``data``
    tensor ``data[tx_i][rx_i][day_i][eq_i]`` -> ``ndarray(n_signals, 256, 2)`` into one
    ``iq`` row + one ``(tx_id, rx_id, day_id)`` record per signal, walking ``(tx, rx, day)``
    in the EXACT nested order :func:`rfbench.data.prepare.sei.extract_wisig_records` uses so
    the committed split indices align element-for-element. ``eq_i`` is pinned to the
    non-equalised captures (``equalized=0``) to match the default prepare extraction.
    Cluster-only: needs the real dataset + numpy.
    """
    import pickle  # stdlib

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading SEI IQ needs numpy; install it with `pip install rfbench[tasks]`."
        ) from exc

    from rfbench.data.prepare._common import resolve_cache_dir

    if name != "wisig":
        raise NotImplementedError(
            f"on-disk IQ loading is wired for WiSig only; {name!r} has no ManyTx.pkl layout."
        )

    cache_dir = resolve_cache_dir(None)
    path = cache_dir / name / "ManyTx.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"WiSig not found at {path}; run the download step first "
            "(rfbench.data.download.sei_wisig)."
        )
    with path.open("rb") as fh:
        dataset = pickle.load(fh)  # noqa: S301 - trusted local dataset file
    return _flatten_wisig(np, dataset, equalized=0)


def _flatten_wisig(
    np_mod: object, dataset: Mapping[str, Any], *, equalized: int
) -> tuple[list[Any], list[SeiRecord]]:  # pragma: no cover - cluster-only
    """Flatten a loaded WiSig ManyTx ``dataset`` dict to ``(iq_rows, records)`` in prepare order.

    Mirrors :func:`rfbench.data.prepare.sei.extract_wisig_records` exactly: it walks
    ``data[tx_i][rx_i][day_i][eq_i]`` for the requested ``equalized`` slot using the real
    ``tx_list`` / ``rx_list`` / ``capture_date_list`` axis labels, emitting one IQ row and
    one ``(tx_id, rx_id, day_id)`` record per signal in the same ``(tx, rx, day)`` nesting.
    Each block is coerced to a ``float32`` ``ndarray(n_signals, 256, 2)`` and iterated
    row-wise so ``iq[k]`` corresponds to ``records[k]``.
    """
    tx_list = list(_require_seq(dataset, "tx_list"))
    rx_list = list(_require_seq(dataset, "rx_list"))
    day_list = list(_require_seq(dataset, "capture_date_list"))
    eq_list = list(_require_seq(dataset, "equalized_list"))
    if equalized not in eq_list:
        raise ValueError(f"equalized={equalized!r} not in dataset equalized_list {eq_list!r}")
    eq_i = eq_list.index(equalized)

    data = dataset.get("data")
    if not isinstance(data, Sequence):
        raise ValueError("WiSig dataset 'data' must be a nested sequence")

    iq: list[Any] = []
    records: list[SeiRecord] = []
    for tx_i, tx_id in enumerate(tx_list):
        for rx_i, rx_id in enumerate(rx_list):
            for day_i, day_id in enumerate(day_list):
                block = np_mod.asarray(  # type: ignore[attr-defined]
                    data[tx_i][rx_i][day_i][eq_i], dtype=np_mod.float32  # type: ignore[attr-defined]
                )
                for row in block:  # (256, 2) per signal
                    iq.append(row)
                    records.append((tx_id, rx_id, day_id))
    return iq, records


def _require_seq(dataset: Mapping[str, Any], key: str) -> Sequence[Any]:  # pragma: no cover
    """Return ``dataset[key]`` as a sequence or raise a clear ``ValueError``."""
    value = dataset.get(key)
    if not isinstance(value, Sequence):
        raise ValueError(f"WiSig dataset is missing sequence field {key!r}")
    return value


__all__ = ["SeiDataset", "open_set_samples"]
