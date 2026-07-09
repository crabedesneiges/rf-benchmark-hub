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

ORACLE (16-tx) and LoRa RFFI are single-condition closed-set datasets here: 80/10/10
stratified by transmitter (ORACLE) / device (LoRa).

Split GENERATION is decoupled from data loading exactly as in AMC: :func:`prepare_sei`
accepts already-extracted ``(tx_id, rx_id, day_id)`` record tuples, so the whole
grouping/stratification path runs on pure-stdlib synthetic fixtures with no numpy. The
heavy metadata EXTRACTION from the real WiSig/ORACLE/LoRa files lives in the lazy loaders
below (:func:`load_wisig_records`, :func:`load_oracle_records`, :func:`load_lora_records`),
which are never called in unit tests.

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
SeiDataset = Literal["wisig", "oracle", "lora", "powder"]

#: The SEI split conditions, reported separately (``docs/EVALUATION_PROTOCOL.md`` §SEI).
SeiCondition = Literal["closed_set", "cross_receiver", "cross_day", "open_set"]

#: Fraction of distinct transmitters kept as the KNOWN gallery in the open-set split; the
#: rest are held out as novel/impostor identities (never seen in train/val). Changing it is
#: a breaking change -> bump the task version.
_OPEN_SET_KNOWN_FRACTION = 0.8

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
        "open_set": "sei-wisig-openset-heldouttx-8010-seed42-v1",
    },
    "oracle": {
        "closed_set": "sei-oracle-closedset-strat-tx-8010-seed42-v1",
    },
    "lora": {
        "closed_set": "sei-lora-closedset-strat-dev-8010-seed42-v1",
    },
    "powder": {
        "closed_set": "sei-powder-wifi4-closedset-strat-dev-8010-seed42-v1",
    },
}

#: Official source URL recorded in each dataset's manifest (provenance, never redistributed).
SOURCE_URLS: dict[str, str] = {
    "wisig": "https://cores.ee.ucla.edu/downloads/datasets/wisig/",
    "oracle": "https://www.genesys-lab.org/oracle",
    "lora": "https://ieee-dataport.org/open-access/lorarffidataset",
    "powder": "https://genesys-lab.org/powder",
}

#: Conditions each dataset supports (WiSig carries receiver + day metadata; the others do not).
#: POWDER (4 BS, one fixed receiver) is closed-set only -- like the two FM SEI evaluators
#: (WirelessJEPA, IQFM), which pool the two capture days into a single closed-set task.
_CONDITIONS: dict[str, tuple[str, ...]] = {
    "wisig": ("closed_set", "cross_receiver", "cross_day", "open_set"),
    "oracle": ("closed_set",),
    "lora": ("closed_set",),
    "powder": ("closed_set",),
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
    if condition == "open_set":
        return _prepare_open_set(
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


def partition_known_unknown_tx(
    tx_ids: Sequence[object],
    *,
    seed: int,
    known_fraction: float = _OPEN_SET_KNOWN_FRACTION,
) -> tuple[set[tuple[int, str]], set[tuple[int, str]]]:
    """Split the distinct transmitters into a KNOWN gallery and UNKNOWN (impostor) set.

    The distinct transmitter ids (normalised via :func:`_norm_group` for a type-agnostic
    deterministic order) are shuffled with a fixed ``seed`` and partitioned so the first
    ``round(known_fraction * n)`` are the known gallery and the rest are held-out novel
    identities. Guarantees at least one known AND one unknown transmitter (needs >= 2
    distinct tx). Returns ``(known_keys, unknown_keys)`` as sets of ``_norm_group`` keys,
    so callers classify a record's tx via ``_norm_group(tx) in known``. Pure stdlib.
    """
    import random

    unique = sorted({_norm_group(t) for t in tx_ids})
    n = len(unique)
    if n < 2:
        raise ValueError(f"open-set needs >= 2 distinct transmitters to hold one out; got {n}")
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n_known = max(1, min(n - 1, round(known_fraction * n)))
    return set(shuffled[:n_known]), set(shuffled[n_known:])


def _prepare_open_set(
    *,
    dataset: str,
    split_id: str,
    source_url: str,
    out_dir: str | Path,
    records: Sequence[SeiRecord],
    source_checksums: Mapping[str, str] | None,
    seed: int,
) -> tuple[SplitManifest, DatasetManifest]:
    """Open-set verification split: hold out whole transmitters as novel/impostor identities.

    Partitions the transmitters into a KNOWN gallery (~80%) and UNKNOWN impostors (~20%,
    :func:`partition_known_unknown_tx`), then:

    * ``train`` / ``val`` = the known transmitters' train/val samples (an 80/10/10 split
      **stratified by transmitter** over the known records only), so the model is fit as a
      ``|known|``-class identifier that never sees an impostor;
    * ``test`` = the known transmitters' test samples (**genuine** probes) PLUS **every**
      impostor sample (**novel** probes). Genuine/impostor is not stored in the split file:
      the dataset derives it as ``tx in {transmitters present in train}`` (the gallery), so
      the split stays a plain ``{train, val, test}`` index partition.

    The open-set score is the model's max-softmax probability and AUROC/EER separate genuine
    from impostor (``docs/EVALUATION_PROTOCOL.md`` §SEI open-set). Runs on pure-stdlib record
    tuples (no numpy), like the sibling conditions.
    """
    from rfbench.core.splits import make_split

    tx_ids = [rec[0] for rec in records]
    known, _unknown = partition_known_unknown_tx(tx_ids, seed=seed)

    known_global = [i for i, tx in enumerate(tx_ids) if _norm_group(tx) in known]
    unknown_global = [i for i, tx in enumerate(tx_ids) if _norm_group(tx) not in known]

    # Stratify the known records by transmitter into train/val/test (dense int codes so
    # make_split's canonical stratified partition applies); map subset indices back to global.
    known_keys = sorted({_norm_group(tx_ids[i]) for i in known_global})
    code_of = {key: code for code, key in enumerate(known_keys)}
    stratify = [code_of[_norm_group(tx_ids[i])] for i in known_global]
    sub = make_split(
        len(known_global),
        seed=seed,
        stratify=stratify,
        split_id=f"{split_id}-knownsub",
        dataset=dataset,
    )
    official = {
        "train": [known_global[j] for j in sub.indices["train"]],
        "val": [known_global[j] for j in sub.indices["val"]],
        "test": [known_global[j] for j in sub.indices["test"]] + unknown_global,
    }
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
    train_n, val_n, test_n = counts
    # Grouped splits can have very few groups (e.g. WiSig has ~4 capture days), where plain
    # largest-remainder starves val/test: 4 groups -> [3, 1, 0], leaving an EMPTY held-out
    # test day (useless for a cross-day protocol). Guarantee a non-empty val AND test
    # whenever there are >= 3 groups, borrowing from train (train stays >= 1).
    if n >= 3:
        test_n = max(test_n, 1)
        val_n = max(val_n, 1)
        train_n = n - val_n - test_n
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
    *,
    equalized: int = 0,
) -> list[SeiRecord]:
    """Extract per-item ``(tx_id, rx_id, day_id)`` records from the WiSig ManyTx pickle.

    WiSig (Hanna et al., IEEE Access 2022) ships each compact subset as a single pickle
    (``ManyTx.pkl``) holding a ``dict`` with the axis label lists ``tx_list`` / ``rx_list``
    / ``capture_date_list`` / ``equalized_list`` and a 5-level nested ``data`` list indexed
    ``data[tx_i][rx_i][day_i][eq_i]`` -> an ``ndarray`` of shape ``(n_signals, 256, 2)``
    (the WiSig ``wisig-examples`` layout). This flattens the whole tensor into one
    ``(tx_id, rx_id, day_id)`` tuple per signal -- one per row of every ``(tx, rx, day)``
    capture block -- in ``(tx, rx, day)`` file order, ready to hand to :func:`prepare_sei`
    as ``records=`` for any of the three conditions (``eq_i`` is fixed to the
    non-equalised captures by default, ``equalized=0``).

    ``pickle`` (stdlib) reads the archive; ``numpy`` is imported lazily only to read each
    block's row count, so ``import rfbench.data.prepare.sei`` stays dependency-free. Never
    called in unit tests (needs real data + numpy). See :func:`extract_wisig_records` for
    the pure-Python extraction used by the tests on a stdlib fixture.
    """
    import pickle  # stdlib

    try:
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
    with path.open("rb") as fh:
        dataset = pickle.load(fh)  # noqa: S301 - trusted local dataset file
    return extract_wisig_records(dataset, equalized=equalized)


def extract_wisig_records(
    dataset: Mapping[str, object],
    *,
    equalized: int = 0,
) -> list[SeiRecord]:
    """Flatten a loaded WiSig compact ``dataset`` dict into ``(tx, rx, day)`` records.

    Pure-Python view of the WiSig ManyTx layout (see :func:`load_wisig_records`): given the
    already-unpickled ``dict`` it walks ``data[tx_i][rx_i][day_i][eq_i]`` for the requested
    ``equalized`` slot and emits one ``(tx_id, rx_id, day_id)`` record per signal row, using
    the real ``tx_list`` / ``rx_list`` / ``capture_date_list`` axis labels as the ids.
    ``block`` may be any object exposing ``len(...)`` == its row count (an ``ndarray`` on
    the cluster; a plain ``list`` in the stdlib test fixture), so this function pulls in no
    third-party dependency and is exercised directly in unit tests.
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

    records: list[SeiRecord] = []
    for tx_i, tx_id in enumerate(tx_list):
        for rx_i, rx_id in enumerate(rx_list):
            for day_i, day_id in enumerate(day_list):
                block = data[tx_i][rx_i][day_i][eq_i]
                n_signals = _block_len(block)
                records.extend((tx_id, rx_id, day_id) for _ in range(n_signals))
    return records


def load_oracle_records(
    cache: str | Path | None = None,
) -> list[SeiRecord]:
    """Extract per-item ``(tx_id, None, None)`` records from the ORACLE capture tree.

    ORACLE (Sankhe et al., Genesys/Northeastern) is a 16-transmitter closed-set dataset
    captured on a single receiver and distributed in the SigMF format: the raw-IQ release
    lays capture files out under distance folders (``<dist>ft/``) named
    ``WiFi_air_X310_<serial>_<dist>ft_run<n>.sigmf-data`` with a sibling ``.sigmf-meta``
    JSON header. The transmitter identity is the USRP X310 ``<serial>`` in the file name;
    only that id is a meaningful group here, so ``rx_id`` / ``day_id`` are ``None``.

    One record is emitted per IQ *capture window* of ``window`` complex samples in each
    ``.sigmf-data`` file (``float32`` interleaved I/Q, so ``2 * window`` floats per window),
    in sorted file order. numpy is imported lazily; never called in unit tests (needs real
    data + numpy).
    """
    import json  # stdlib

    try:
        import numpy as np
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

    window = _ORACLE_WINDOW
    records: list[SeiRecord] = []
    for data_path in sorted(root.rglob("*.sigmf-data")):
        tx_id = _oracle_tx_id(data_path.name)
        # SigMF raw IQ: interleaved I/Q float32 (confirm datatype in the .sigmf-meta).
        meta_path = data_path.with_suffix(".sigmf-meta")
        dtype = _sigmf_np_dtype(np, meta_path, json) if meta_path.exists() else np.float32
        iq = np.fromfile(data_path, dtype=dtype)
        n_complex = iq.size // 2
        n_windows = n_complex // window
        records.extend((tx_id, None, None) for _ in range(n_windows))
    if not records:
        raise FileNotFoundError(
            f"no ORACLE .sigmf-data captures found under {root}; check the extraction."
        )
    return records


def load_lora_records(
    cache: str | Path | None = None,
    *,
    filename: str = "dataset_training_aug.h5",
) -> list[SeiRecord]:
    """Extract per-item ``(device_id, None, None)`` records from the LoRa RFFI HDF5.

    LoRa RFFI (Shen et al., IEEE JSAC 2021) is distributed as a single HDF5 archive
    (``LoRa_RFFI.zip`` on IEEE DataPort, DOI 10.21227/qqt4-kz19) whose training file
    ``Train/dataset_training_aug.h5`` holds two datasets: ``data`` of shape
    ``(n_packets, 2 * n_samples)`` (the first half is the real part, the second half the
    imaginary part of the preamble IQ) and ``label`` of shape ``(1, n_packets)`` carrying
    the **1-indexed** device id of each packet (the ``gxhen/LoRa_RFFI`` layout). This reads
    only the ``label`` row and emits one ``(device_id, None, None)`` record per packet, in
    file order, ready for :func:`prepare_sei` as ``records=`` for the closed-set condition.

    h5py + numpy are imported lazily; never called in unit tests (needs real data + heavy
    deps). See :func:`extract_lora_records` for the pure-Python label -> record mapping.
    """
    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading LoRa RFFI needs numpy + h5py to read the HDF5 file; "
            "install them with `pip install rfbench[data]`."
        ) from exc

    cache_dir = resolve_cache_dir(cache)
    path = cache_dir / "lora" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"LoRa RFFI not found at {path}; run the download step first "
            "(rfbench.data.download.sei_lora)."
        )
    with h5py.File(path, "r") as handle:
        label = np.asarray(handle["label"][:]).astype(int).reshape(-1)
    return extract_lora_records(label.tolist())


def extract_lora_records(labels: Sequence[int]) -> list[SeiRecord]:
    """Map a flat sequence of LoRa **1-indexed** device labels to closed-set records.

    Pure-Python view of the LoRa label layout (see :func:`load_lora_records`): each label
    is the transmitter/device id of one packet; this emits one ``(device_id, None, None)``
    record per packet, preserving file order. Pulls in no third-party dependency, so it is
    exercised directly in unit tests.
    """
    return [(int(dev), None, None) for dev in labels]


def load_powder_records(
    cache: str | Path | None = None,
    *,
    window: int = 256,
) -> list[SeiRecord]:
    """Extract per-item ``(device_id, None, day_id)`` records from the POWDER SigMF captures.

    POWDER RF Fingerprinting (Reus-Muns, Jaisinghani, Sankhe, Chowdhury, "Trust in 5G Open RANs
    through Machine Learning: RF Fingerprinting on the POWDER PAWR Platform", IEEE GLOBECOM 2020)
    is the 4-base-station WiFi hardware-fingerprinting set used by the two FM SEI evaluators
    (WirelessJEPA arXiv:2601.20190, IQFM arXiv:2506.06718). It is distributed as SigMF captures
    (a ``.sigmf-data`` raw-IQ file + a ``.sigmf-meta`` JSON header per recording) named
    ``[Waveform]_[Day]_[TransmitterBS]_[RecordingSet]`` -- the transmitter base station (field 3)
    is the fingerprint identity, and the day (field 2) is carried so a future cross-day track is
    possible (the closed-set track pools both days, matching the FM evals).

    Each recording is sliced into non-overlapping ``window``-sample frames (the FM papers use 256;
    the origin paper used 512), emitting one ``(device_id, None, day_id)`` record per frame in
    sorted file order. numpy is imported lazily; never called in unit tests (needs real data +
    numpy). See :func:`extract_powder_records` for the pure-Python frame-count -> record mapping.

    NOTE (download): the POWDER DRS host anti-scrapes programmatic fetches, so the raw captures
    must be placed under ``$RFBENCH_CACHE/powder/`` manually -- see
    :mod:`rfbench.data.download.sei_powder` for the exact procedure. This loader only reads what
    is already there.
    """
    import json  # stdlib

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading POWDER needs numpy; install it with `pip install rfbench[data]`."
        ) from exc

    cache_dir = resolve_cache_dir(cache)
    root = cache_dir / "powder"
    if not root.is_dir():
        raise FileNotFoundError(
            f"POWDER not found at {root}; place the SigMF captures there first "
            "(see rfbench.data.download.sei_powder for the manual-download procedure)."
        )

    frame_counts: list[tuple[str, str, int]] = []
    for data_path in sorted(root.rglob("*.sigmf-data")):
        device_id, day_id = _powder_ids(data_path.name)
        meta_path = data_path.with_suffix(".sigmf-meta")
        dtype = _sigmf_np_dtype(np, meta_path, json) if meta_path.exists() else np.float32
        iq = np.fromfile(data_path, dtype=dtype)
        n_complex = int(iq.size // 2)
        n_frames = n_complex // window
        frame_counts.append((device_id, day_id, n_frames))
    if not frame_counts:
        raise FileNotFoundError(
            f"no POWDER .sigmf-data captures found under {root}; check the manual extraction."
        )
    return extract_powder_records(frame_counts)


def extract_powder_records(frame_counts: Sequence[tuple[object, object, int]]) -> list[SeiRecord]:
    """Map ``(device_id, day_id, n_frames)`` triples to per-frame closed-set POWDER records.

    Pure-Python view of the POWDER layout (see :func:`load_powder_records`): each entry is one
    recording's ``(device_id, day_id, n_frames)`` and this emits one ``(device_id, None,
    day_id)`` record per frame, preserving file order. ``rx_id`` is ``None`` (POWDER has a single
    fixed receiver). Pulls in no third-party dependency, so it is exercised directly in tests.
    """
    records: list[SeiRecord] = []
    for device_id, day_id, n_frames in frame_counts:
        records.extend((device_id, None, day_id) for _ in range(int(n_frames)))
    return records


#: ORACLE capture-window length (complex samples per record); matches the reference
#: IEEE802.11a burst length used to slice each raw-IQ ``.sigmf-data`` file.
_ORACLE_WINDOW = 128


def _require_seq(dataset: Mapping[str, object], key: str) -> Sequence[object]:
    """Return ``dataset[key]`` as a sequence or raise a clear ``ValueError``."""
    value = dataset.get(key)
    if not isinstance(value, Sequence):
        raise ValueError(f"WiSig dataset is missing sequence field {key!r}")
    return value


def _block_len(block: object) -> int:
    """Row count of one WiSig capture block (``ndarray`` on cluster, ``list`` in tests)."""
    shape = getattr(block, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) else 0
    return len(block)  # type: ignore[arg-type]


def _oracle_tx_id(filename: str) -> str:
    """Parse the USRP X310 serial (transmitter id) out of an ORACLE capture file name.

    Expects the reference naming ``WiFi_air_X310_<serial>_<dist>ft_run<n>.sigmf-data``;
    returns ``<serial>``. Falls back to the file stem if the name deviates.
    """
    parts = filename.split("_")
    if len(parts) >= 4 and parts[0] == "WiFi" and parts[2] == "X310":
        return parts[3]
    return filename.rsplit(".", 1)[0]


def _powder_ids(filename: str) -> tuple[str, str]:
    """Parse ``(device_id, day_id)`` from a POWDER SigMF capture file name.

    Expects the reference naming ``[Waveform]_[Day]_[TransmitterBS]_[RecordingSet]`` (e.g.
    ``WiFi_Day1_MEB_1.sigmf-data``): the transmitter base station (field 3, index 2) is the
    fingerprint identity and the day (field 2, index 1) is carried for provenance / a future
    cross-day track. Falls back to ``(stem, "unknown_day")`` if the name deviates.
    """
    stem = filename.rsplit(".", 1)[0].split(".sigmf")[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[2], parts[1]
    return stem, "unknown_day"


def _sigmf_np_dtype(np_mod: object, meta_path: Path, json_mod: object) -> object:
    """Map a SigMF ``core:datatype`` to a numpy dtype (defaults to interleaved f32)."""
    meta = json_mod.loads(meta_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    datatype = str(meta.get("global", {}).get("core:datatype", "cf32_le"))
    table = {
        "cf32_le": np_mod.float32,  # type: ignore[attr-defined]
        " cf32": np_mod.float32,  # type: ignore[attr-defined]
        "ci16_le": np_mod.int16,  # type: ignore[attr-defined]
        "cf64_le": np_mod.float64,  # type: ignore[attr-defined]
    }
    return table.get(datatype, np_mod.float32)  # type: ignore[attr-defined]


__all__ = [
    "SeiDataset",
    "SeiCondition",
    "SeiRecord",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "prepare_sei",
    "partition_known_unknown_tx",
    "load_wisig_records",
    "extract_wisig_records",
    "load_oracle_records",
    "load_lora_records",
    "extract_lora_records",
    "load_powder_records",
    "extract_powder_records",
]
