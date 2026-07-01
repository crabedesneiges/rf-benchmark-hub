"""SEI (RF fingerprinting) canonical splits -- WP-12, on the WP-11 AMC template.

Builds the SEI split CONDITIONS per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§SEI): the three WiSig conditions are reported *separately*, each with its own canonical
split id, ``.idx.json`` and manifest sidecar:

* ``closed_set`` -- a standard deterministic **80/10/10** split **stratified by
  transmitter** (:func:`rfbench.core.splits.make_split`); the same transmitters appear in
  train/val/test (identity classification with i.i.d. captures).
* ``cross_receiver`` -- a **grouped** split partitioned by *receiver id*: whole receivers
  go to train / val / test, so the test receivers are disjoint from the train receivers
  (no receiver leakage). Transmitters are shared across partitions -- the model must
  recognise the same emitters seen through *unseen receivers*.
* ``cross_day`` -- a **grouped** split partitioned by *capture day*: whole days go to
  train / val / test, so the test days are disjoint from the train days (no day leakage),
  measuring robustness to channel/temporal drift.

ORACLE (16-tx) is a single-condition closed-set dataset here: 80/10/10 stratified by
transmitter.

Split GENERATION is decoupled from data loading exactly as in AMC: :func:`prepare_sei`
accepts already-extracted ``(tx_id, rx_id, day_id)`` record tuples, so the whole
grouping/stratification path runs on pure-stdlib synthetic fixtures with no numpy. The
heavy metadata EXTRACTION from the real WiSig/ORACLE files lives in the lazy loaders below
(:func:`load_wisig_records`, :func:`load_oracle_records`), which are never called in unit
tests.

Module-top imports are stdlib + the frozen core contracts + the ``_common`` template
helpers only; numpy/h5py are imported lazily inside the loaders with a clear
``pip install rfbench[data]`` error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import (
    prepare_from_labels,
    prepare_from_official,
    resolve_cache_dir,
)

#: The SEI datasets this WP prepares.
SeiDataset = Literal["wisig", "oracle"]

#: The SEI split conditions, reported separately (``docs/EVALUATION_PROTOCOL.md`` §SEI).
SeiCondition = Literal["closed_set", "cross_receiver", "cross_day"]

#: A per-item SEI record: ``(transmitter_id, receiver_id, day_id)``. ``rx_id`` / ``day_id``
#: may be omitted (``None``) for datasets/conditions that do not carry them, but the
#: grouped conditions require the relevant field to be present.
SeiRecord = tuple[object, object | None, object | None]

#: Canonical split id per (dataset, condition); baked ratios+seed. Changing either the
#: ratios or the seed is a breaking change -> bump the task version.
CANONICAL_SPLIT_IDS: dict[str, dict[str, str]] = {
    "wisig": {
        "closed_set": "sei-wisig-closedset-strat-tx-8010-seed42-v1",
        "cross_receiver": "sei-wisig-crossrx-grouped-8010-seed42-v1",
        "cross_day": "sei-wisig-crossday-grouped-8010-seed42-v1",
    },
    "oracle": {
        "closed_set": "sei-oracle-closedset-strat-tx-8010-seed42-v1",
    },
}

#: Official source URL recorded in each dataset's manifest (provenance, never redistributed).
SOURCE_URLS: dict[str, str] = {
    "wisig": "https://cores.ee.ucla.edu/downloads/datasets/wisig/",
    "oracle": "https://www.genesys-lab.org/oracle",
}

#: Conditions each dataset supports (WiSig carries receiver + day metadata; ORACLE does not).
_CONDITIONS: dict[str, tuple[str, ...]] = {
    "wisig": ("closed_set", "cross_receiver", "cross_day"),
    "oracle": ("closed_set",),
}

#: Which record field a grouped condition partitions on (index into :data:`SeiRecord`).
_GROUP_FIELD: dict[str, int] = {
    "cross_receiver": 1,  # receiver_id
    "cross_day": 2,  # day_id
}


def prepare_sei(
    dataset: SeiDataset | str,
    condition: SeiCondition | str,
    *,
    out_dir: str | Path,
    records: Sequence[SeiRecord],
    source_checksums: Mapping[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical SEI split + manifest for one ``(dataset, condition)``.

    The split-GENERATION path takes pre-extracted ``(tx_id, rx_id, day_id)`` records so it
    runs without numpy on synthetic fixtures. Each ``condition`` yields its own canonical
    split id + ``.idx.json`` + manifest:

    * ``closed_set`` -> **80/10/10 stratified by transmitter** (same transmitters in every
      partition). ``rx_id`` / ``day_id`` may be ``None``.
    * ``cross_receiver`` -> a **grouped** split by ``rx_id``: whole receivers are assigned
      to train / val / test, so **test receivers are disjoint from train receivers**
      (no receiver leakage). Requires every record to carry a non-``None`` ``rx_id``.
    * ``cross_day`` -> a **grouped** split by ``day_id`` with the analogous day-disjoint
      guarantee. Requires every record to carry a non-``None`` ``day_id``.

    On the cluster the caller first extracts ``records`` via :func:`load_wisig_records` /
    :func:`load_oracle_records` (lazy numpy/h5py), then calls this. Writes
    ``<out_dir>/splits/<dataset>/<id>.idx.json`` and ``...manifest.json`` only; never raw
    data (D3). Returns the ``(SplitManifest, DatasetManifest)`` pair.
    """
    if dataset not in CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown SEI dataset {dataset!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
        )
    supported = _CONDITIONS[dataset]
    if condition not in supported:
        raise ValueError(
            f"dataset {dataset!r} does not support condition {condition!r}; "
            f"supported: {sorted(supported)}"
        )

    split_id = CANONICAL_SPLIT_IDS[dataset][condition]
    source_url = SOURCE_URLS[dataset]

    if condition == "closed_set":
        return _prepare_closed_set(
            dataset=dataset,
            split_id=split_id,
            source_url=source_url,
            out_dir=out_dir,
            records=records,
            source_checksums=source_checksums,
            seed=seed,
        )
    return _prepare_grouped(
        dataset=dataset,
        condition=condition,
        split_id=split_id,
        source_url=source_url,
        out_dir=out_dir,
        records=records,
        source_checksums=source_checksums,
        seed=seed,
    )


def _prepare_closed_set(
    *,
    dataset: str,
    split_id: str,
    source_url: str,
    out_dir: str | Path,
    records: Sequence[SeiRecord],
    source_checksums: Mapping[str, str] | None,
    seed: int,
) -> tuple[SplitManifest, DatasetManifest]:
    """Standard 80/10/10 split stratified by transmitter (identity is the label)."""
    strata: list[tuple[object, ...]] = [(rec[0],) for rec in records]
    return prepare_from_labels(
        dataset=dataset,
        split_id=split_id,
        n_items=len(strata),
        strata=strata,
        source_url=source_url,
        out_dir=out_dir,
        source_checksums=source_checksums,
        seed=seed,
    )


def _prepare_grouped(
    *,
    dataset: str,
    condition: str,
    split_id: str,
    source_url: str,
    out_dir: str | Path,
    records: Sequence[SeiRecord],
    source_checksums: Mapping[str, str] | None,
    seed: int,
) -> tuple[SplitManifest, DatasetManifest]:
    """Grouped split: whole groups (receiver / day) go to one partition -> no leakage."""
    field = _GROUP_FIELD[condition]
    groups = [rec[field] for rec in records]
    if any(g is None for g in groups):
        raise ValueError(
            f"condition {condition!r} for {dataset!r} partitions by "
            f"{'receiver' if field == 1 else 'day'} id, but some records carry no such id "
            "(None); every record must have it."
        )

    official = _partition_by_group(groups, seed=seed)
    return prepare_from_official(
        dataset=dataset,
        split_id=split_id,
        official=official,
        source_url=source_url,
        out_dir=out_dir,
        source_checksums=source_checksums,
        seed=seed,
    )


def _partition_by_group(groups: Sequence[object], *, seed: int) -> dict[str, list[int]]:
    """Assign every *group* to exactly one of train/val/test, then map items by group.

    The distinct group ids are ordered deterministically (:func:`_group_sort_key`),
    shuffled with a fixed ``seed`` and partitioned 80/10/10 *at the group level* via the
    same largest-remainder logic as ``core.splits``. Every item inherits its group's
    partition, so the resulting train/val/test partitions never share a group -- e.g. the
    test receivers/days are disjoint from the train ones (no leakage across the boundary
    the condition guards).

    Returns ``{"train": [...], "val": [...], "test": [...]}`` index lists suitable for
    :func:`rfbench.core.splits.adopt_official_split` (via ``prepare_from_official``).
    """
    import random

    unique = sorted({_norm_group(g) for g in groups})
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)

    train_g, val_g, test_g = _partition_groups(shuffled)
    partition_of: dict[tuple[int, str], str] = {}
    for g in train_g:
        partition_of[g] = "train"
    for g in val_g:
        partition_of[g] = "val"
    for g in test_g:
        partition_of[g] = "test"

    indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, raw in enumerate(groups):
        indices[partition_of[_norm_group(raw)]].append(idx)
    for name in indices:
        indices[name].sort()
    return indices


def _partition_groups(
    groups: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
    """Split an ordered group list 80/10/10 (largest-remainder, canonical tie-break).

    Mirrors ``core.splits._partition`` so grouped conditions share the exact rounding
    behaviour of the stratified path: counts sum to ``len(groups)`` and remainders go to
    the partitions with the largest fractional part, ties broken by train, val, test.
    """
    ratios = (0.8, 0.1, 0.1)
    n = len(groups)
    raw = [r * n for r in ratios]
    floors = [int(x) for x in raw]
    remainder = n - sum(floors)
    order = sorted(range(3), key=lambda i: (-(raw[i] - floors[i]), i))
    counts = list(floors)
    for i in order[:remainder]:
        counts[i] += 1
    train_n, val_n, _ = counts
    return (
        groups[:train_n],
        groups[train_n : train_n + val_n],
        groups[train_n + val_n :],
    )


def _norm_group(group: object) -> tuple[int, str]:
    """Type-agnostic canonical key for a group id (mixes int/str safely, like strata).

    Maps each group id to ``(type_rank, str_value)`` so a set of mixed int/str ids has a
    deterministic total order and sorting never raises ``TypeError`` -- the same trick
    ``_common._stratum_sort_key`` uses for composite strata.
    """
    if isinstance(group, str):
        return (0, group)
    return (1, f"{group!r}")


# --- lazy loaders (cluster-only; heavy deps; NEVER called in unit tests) -------------


def load_wisig_records(
    cache: str | Path | None = None,
) -> list[SeiRecord]:
    """Extract per-item ``(tx_id, rx_id, day_id)`` records from the WiSig files on disk.

    WiSig (ManyTx) ships as a compressed pickle of per-(transmitter, receiver, day)
    capture blocks; this flattens them into one ``(tx_id, rx_id, day_id)`` tuple per
    signal, in file order, ready to hand to :func:`prepare_sei` as ``records=`` for any of
    the three conditions. numpy/pickle are imported lazily so ``import
    rfbench.data.prepare.sei`` stays dependency-free. Never called in unit tests (needs
    real data + heavy deps).
    """
    try:
        import pickle  # noqa: F401 - stdlib; kept explicit for the FileNotFound path below

        import numpy as np  # noqa: F401 - surfaces the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading WiSig needs numpy; install it with `pip install rfbench[data]`."
        ) from exc

    cache_dir = resolve_cache_dir(cache)
    path = cache_dir / "wisig" / "ManyTx.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"WiSig not found at {path}; run the download step first "
            "(rfbench.data.download.sei_wisig)."
        )
    raise NotImplementedError(
        "WiSig record extraction runs on the cluster against the real ManyTx pickle; wire "
        "it to the concrete (tx, rx, day) capture layout there."
    )


def load_oracle_records(
    cache: str | Path | None = None,
) -> list[SeiRecord]:
    """Extract per-item ``(tx_id, None, None)`` records from the ORACLE files on disk.

    ORACLE is a 16-transmitter closed-set dataset captured on a single receiver; only the
    transmitter id is a meaningful group here, so ``rx_id`` / ``day_id`` are ``None``.
    numpy is imported lazily. Never called in unit tests (needs real data + heavy deps).
    """
    try:
        import numpy as np  # noqa: F401 - surfaces the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading ORACLE needs numpy; install it with `pip install rfbench[data]`."
        ) from exc

    cache_dir = resolve_cache_dir(cache)
    root = cache_dir / "oracle"
    if not root.is_dir():
        raise FileNotFoundError(
            f"ORACLE not found at {root}; run the download step first "
            "(rfbench.data.download.sei_oracle)."
        )
    raise NotImplementedError(
        "ORACLE record extraction runs on the cluster against the real capture tree; wire "
        "it to the concrete per-transmitter SigMF/binary layout there."
    )


__all__ = [
    "SeiDataset",
    "SeiCondition",
    "SeiRecord",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "prepare_sei",
    "load_wisig_records",
    "load_oracle_records",
]
