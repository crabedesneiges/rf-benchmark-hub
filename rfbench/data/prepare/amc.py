"""AMC (automatic modulation classification) canonical splits -- WP-11 template.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md`` §AMC):

* **RadioML 2016.10a / 2018.01a** have no canonical literature split -> a deterministic
  **80/10/10** split **stratified by (modulation x snr)**, seed 42. Canonical ids
  ``amc-radioml2016-strat-snr-8010-seed42-v1`` / ``amc-radioml2018-...``.
* **Sig53** ships the official TorchSig split -> adopt it verbatim
  (``adopt_official_split``), id ``amc-sig53-official-v1``.

Split GENERATION is decoupled from data loading: :func:`prepare_amc` accepts
already-extracted ``(modulation, snr_db)`` label tuples (RadioML) or an explicit official
partition (Sig53), so the whole path runs on pure-stdlib synthetic fixtures with no numpy.
The heavy label EXTRACTION from the real ``.h5`` / TorchSig files lives in the lazy
loaders below (:func:`load_radioml_labels`, :func:`load_sig53_official_split`), which are
never called in unit tests.

Module top imports are stdlib + the frozen core contracts only; numpy/h5py/torchsig are
imported lazily inside the loaders with a clear ``pip install rfbench[data]`` error.
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

#: The AMC datasets this WP prepares.
AmcDataset = Literal["radioml_2016_10a", "radioml_2018_01a", "sig53"]

#: Canonical split id per dataset (baked ratios+seed; changing either bumps task version).
CANONICAL_SPLIT_IDS: dict[str, str] = {
    "radioml_2016_10a": "amc-radioml2016-strat-snr-8010-seed42-v1",
    "radioml_2018_01a": "amc-radioml2018-strat-snr-8010-seed42-v1",
    "sig53": "amc-sig53-official-v1",
}

#: Official source URL recorded in each dataset's manifest (provenance, never redistributed).
SOURCE_URLS: dict[str, str] = {
    "radioml_2016_10a": "https://www.deepsig.ai/datasets/",
    "radioml_2018_01a": "https://www.deepsig.ai/datasets/",
    "sig53": "https://github.com/TorchDSP/torchsig",
}

#: Datasets whose split is generated (stratified) vs adopted from an official source.
_STRATIFIED = ("radioml_2016_10a", "radioml_2018_01a")
_OFFICIAL = ("sig53",)


def prepare_amc(
    dataset: AmcDataset | str,
    *,
    out_dir: str | Path,
    labels: Sequence[tuple[object, int]] | None = None,
    official_split: Mapping[str, Sequence[int]] | None = None,
    source_checksums: Mapping[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical AMC split + manifest for ``dataset``.

    The split-GENERATION path takes pre-extracted labels/partitions so it runs without
    numpy on synthetic fixtures:

    * RadioML (``radioml_2016_10a`` / ``radioml_2018_01a``): pass ``labels`` as a sequence
      of ``(modulation, snr_db)`` tuples, one per item -> **80/10/10 stratified by
      (modulation x snr)**, seed 42.
    * Sig53: pass ``official_split`` as ``{"train": [...], "val": [...], "test": [...]}``
      (the TorchSig split) -> adopted verbatim.

    On the cluster the caller first extracts these via :func:`load_radioml_labels` /
    :func:`load_sig53_official_split` (lazy numpy/h5py/torchsig), then calls this.

    Writes ``<out_dir>/splits/<dataset>/<id>.idx.json`` and ``...manifest.json`` only;
    never raw data (D3). Returns the ``(SplitManifest, DatasetManifest)`` pair.
    """
    if dataset not in CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown AMC dataset {dataset!r}; expected one of {sorted(CANONICAL_SPLIT_IDS)}"
        )
    split_id = CANONICAL_SPLIT_IDS[dataset]
    source_url = SOURCE_URLS[dataset]

    if dataset in _OFFICIAL:
        if official_split is None:
            raise ValueError(
                f"{dataset!r} adopts the official TorchSig split; pass `official_split=` "
                "(extracted via load_sig53_official_split)"
            )
        return prepare_from_official(
            dataset=dataset,
            split_id=split_id,
            official=official_split,
            source_url=source_url,
            out_dir=out_dir,
            source_checksums=source_checksums,
            seed=seed,
        )

    # Stratified RadioML path.
    if labels is None:
        raise ValueError(
            f"{dataset!r} has no canonical split; pass `labels=` as (modulation, snr_db) "
            "tuples (extracted via load_radioml_labels) to stratify by modulation x snr"
        )
    strata: list[tuple[object, ...]] = [(mod, snr) for mod, snr in labels]
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


# --- lazy loaders (cluster-only; heavy deps; NEVER called in unit tests) -------------


def load_radioml_labels(
    dataset: Literal["radioml_2016_10a", "radioml_2018_01a"],
    cache: str | Path | None = None,
) -> list[tuple[str, int]]:
    """Extract per-item ``(modulation, snr_db)`` labels from a RadioML file on disk.

    2016.10a is a pickled ``dict[(mod, snr) -> ndarray]``; 2018.01a is an HDF5 file with
    ``X`` / ``Y`` (one-hot modulation) / ``Z`` (SNR) datasets. Both are read lazily
    (numpy/h5py/pickle) so ``import rfbench.data.prepare.amc`` stays dependency-free.

    Returns one ``(modulation, snr_db)`` tuple per item, in file order, ready to hand to
    :func:`prepare_amc` as ``labels=``. Never called in unit tests (needs real data +
    heavy deps).
    """
    cache_dir = resolve_cache_dir(cache)
    if dataset == "radioml_2016_10a":
        return _load_radioml2016_labels(cache_dir)
    return _load_radioml2018_labels(cache_dir)


def _load_radioml2016_labels(cache_dir: Path) -> list[tuple[str, int]]:
    """Read ``(mod, snr)`` per item from the RadioML 2016.10a pickle (lazy numpy/pickle)."""
    try:
        import pickle
    except ModuleNotFoundError as exc:  # pragma: no cover - pickle is stdlib
        raise RuntimeError("pickle is required to read RadioML 2016.10a") from exc

    path = cache_dir / "radioml_2016_10a" / "RML2016.10a_dict.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"RadioML 2016.10a not found at {path}; run the download step first "
            "(rfbench.data.download.amc_radioml)."
        )
    with path.open("rb") as fh:
        table = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset file

    labels: list[tuple[str, int]] = []
    for (mod, snr), array in table.items():
        n = len(array)
        labels.extend((str(mod), int(snr)) for _ in range(n))
    return labels


def _load_radioml2018_labels(cache_dir: Path) -> list[tuple[str, int]]:
    """Read ``(mod, snr)`` per item from the RadioML 2018.01a HDF5 (lazy numpy/h5py)."""
    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RadioML 2018.01a needs numpy + h5py to read the HDF5 file; "
            "install them with `pip install rfbench[data]`."
        ) from exc

    path = cache_dir / "radioml_2018_01a" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    if not path.exists():
        raise FileNotFoundError(
            f"RadioML 2018.01a not found at {path}; run the download step first "
            "(rfbench.data.download.amc_radioml)."
        )
    # 24 canonical modulation classes in 2018.01a, in the dataset's one-hot column order.
    mod_classes = _RADIOML2018_MODS
    with h5py.File(path, "r") as handle:
        onehot = handle["Y"][:]  # (N, 24) one-hot modulation
        snr = handle["Z"][:]  # (N, 1) SNR in dB
    mod_idx = np.asarray(onehot).argmax(axis=1)
    snr_flat = np.asarray(snr).reshape(-1).astype(int)
    return [(mod_classes[int(m)], int(s)) for m, s in zip(mod_idx, snr_flat, strict=True)]


#: Canonical 24-class modulation order of RadioML 2018.01a (Y one-hot column order).
_RADIOML2018_MODS: tuple[str, ...] = (
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
)


def load_sig53_official_split(
    cache: str | Path | None = None,
) -> dict[str, list[int]]:
    """Extract the official TorchSig Sig53 train/val(/test) partition as index lists.

    TorchSig ships Sig53 already partitioned (``impaired`` train/val roots); this maps the
    on-disk partition to explicit ``{"train": [...], "val": [...], "test": [...]}`` index
    lists suitable for :func:`prepare_amc`'s ``official_split=``. ``torchsig`` is imported
    lazily. Never called in unit tests (needs TorchSig + generated data).
    """
    resolve_cache_dir(cache)  # validate cache resolution eagerly; TorchSig reads real files
    try:
        import torchsig  # noqa: F401 - imported to surface the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Sig53 uses the official TorchSig split; install TorchSig with "
            "`pip install rfbench[detection]` (provides torchsig) to read it."
        ) from exc
    raise NotImplementedError(
        "Sig53 official-split extraction runs on the cluster against a generated TorchSig "
        "dataset; wire it to the concrete torchsig.datasets.Sig53 layout there."
    )


__all__ = [
    "AmcDataset",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "prepare_amc",
    "load_radioml_labels",
    "load_sig53_official_split",
]
