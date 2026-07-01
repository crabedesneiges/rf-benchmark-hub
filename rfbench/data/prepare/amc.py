"""AMC (automatic modulation classification) canonical splits -- WP-11 template.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md`` §AMC):

* **RadioML 2016.10a / 2018.01a** have no canonical literature split -> a deterministic
  **80/10/10** split **stratified by (modulation x snr)**, seed 42. Canonical ids
  ``amc-radioml2016-strat-snr-8010-seed42-v1`` / ``amc-radioml2018-...``. Both are fetched as
  the REAL published DeepSig artifacts (pickle / HDF5) -- see
  :mod:`rfbench.data.download.amc_radioml`.
* **Sig53** would adopt the official TorchSig split verbatim (``adopt_official_split``, id
  ``amc-sig53-official-v1``) -- BUT Sig53 has no static published release (generation-only via
  TorchSig), so per policy we do NOT synthesise it and the track is BLOCKED. Both
  :func:`load_sig53_official_split` and :func:`rfbench.data.download.amc_sig53.download_sig53`
  raise a clear blocker; the ``prepare_amc`` official-split path still works if a real split is
  ever supplied.

Split GENERATION is decoupled from data loading: :func:`prepare_amc` accepts
already-extracted ``(modulation, snr_db)`` label tuples (RadioML) or an explicit official
partition (Sig53), so the whole path runs on pure-stdlib synthetic fixtures with no numpy.
The heavy label EXTRACTION from the real pickle / ``.h5`` files lives in the lazy loaders
below (:func:`load_radioml_labels`), which are never called in unit tests.

Module top imports are stdlib + the frozen core contracts only; numpy/h5py/torchsig are
imported lazily inside the loaders with a clear ``pip install rfbench[data]`` error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
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
    "radioml_2016_10a": "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
    "radioml_2018_01a": "https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz",
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
    """Read ``(mod, snr)`` per item from the RadioML 2016.10a pickle (stdlib pickle).

    The published ``RML2016.10a_dict.pkl`` is a ``dict`` keyed by ``(modulation, snr)`` whose
    values are per-cell signal blocks of shape ``[N, 2, 128]`` (numpy on the real file, but any
    length-supporting sequence in a fixture). We expand each key to ``N`` per-item labels using
    only ``len(block)`` (the number of items in the cell), so this parser is exercisable on a
    pure-stdlib pickle fixture -- no numpy import is needed to extract the labels.
    """
    import pickle  # stdlib

    ds_dir = cache_dir / "radioml_2016_10a"
    # DeepSig's original artifact is ``RML2016.10a_dict.pkl``; the Zenodo mirror
    # (record 18397070, valid cert) ships the same dict re-pickled as
    # ``RML2016.10a_dict_optimized.pkl``. Accept either.
    candidates = ("RML2016.10a_dict.pkl", "RML2016.10a_dict_optimized.pkl")
    path = next((ds_dir / name for name in candidates if (ds_dir / name).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"RadioML 2016.10a not found in {ds_dir} (looked for {list(candidates)}); run the "
            "download step first (rfbench.data.download.amc_radioml.download_radioml)."
        )
    with path.open("rb") as fh:
        table = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset file

    return _expand_radioml2016_table(table)


def _expand_radioml2016_table(
    table: Mapping[tuple[object, object], Sized],
) -> list[tuple[str, int]]:
    """Expand a ``{(mod, snr): block}`` table into one ``(mod, snr)`` label per item.

    Split out from :func:`_load_radioml2016_labels` so the label-extraction logic (the only
    part with real parsing semantics) is unit-testable on a pure-stdlib fixture that mimics the
    published ``dict[(mod, snr) -> [N, 2, 128]]`` layout, with no numpy and no pickle file. The
    block value need only be :class:`~typing.Sized` (real: ``ndarray[N, 2, 128]``; fixture:
    nested list of length ``N``) since we take just ``len(block)`` -- the item count in the cell.
    """
    labels: list[tuple[str, int]] = []
    for (mod, snr), block in table.items():
        n = len(block)
        labels.extend((str(mod), int(str(snr))) for _ in range(n))
    return labels


def _load_radioml2018_labels(cache_dir: Path) -> list[tuple[str, int]]:
    """Read ``(mod, snr)`` per item from the RadioML 2018.01a HDF5 (lazy numpy/h5py).

    The published ``GOLD_XYZ_OSC.0001_1024.hdf5`` holds three datasets: ``X`` ``(N, 1024, 2)``
    signals, ``Y`` ``(N, 24)`` one-hot modulation, ``Z`` ``(N, 1)`` (or ``(N,)``) SNR in dB.
    We only need ``Y``/``Z`` for stratification. The pure-index conversion is factored into
    :func:`_radioml2018_labels_from_arrays`, so the label logic is testable without HDF5.
    """
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
            "(rfbench.data.download.amc_radioml.download_radioml)."
        )
    with h5py.File(path, "r") as handle:
        onehot = np.asarray(handle["Y"][:])  # (N, 24) one-hot modulation
        snr = np.asarray(handle["Z"][:])  # (N, 1) or (N,) SNR in dB
    mod_idx = [int(i) for i in onehot.argmax(axis=1)]
    snr_flat = [int(s) for s in snr.reshape(-1)]
    return _radioml2018_labels_from_arrays(mod_idx, snr_flat)


def _radioml2018_labels_from_arrays(
    mod_idx: Sequence[int],
    snr_db: Sequence[int],
) -> list[tuple[str, int]]:
    """Map decoded (one-hot argmax) modulation indices + SNRs to ``(mod_name, snr)`` labels.

    Pure Python: given the per-item modulation column index and SNR (already extracted from the
    HDF5 ``Y``/``Z`` datasets), resolve each index through the canonical 24-class order. Kept
    numpy-free so the class-mapping logic is unit-testable on plain lists.
    """
    if len(mod_idx) != len(snr_db):
        raise ValueError(f"mod/snr length mismatch: {len(mod_idx)} vs {len(snr_db)}")
    mod_classes = _RADIOML2018_MODS
    return [(mod_classes[m], int(s)) for m, s in zip(mod_idx, snr_db, strict=True)]


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
    """Report the Sig53 blocker: no static release, so no official split to extract.

    Sig53 has NO statically-downloadable published artifact -- it is generation-only via
    TorchSig -- and RF-Benchmark does not synthesise datasets in lieu of a real published
    release (see :mod:`rfbench.data.download.amc_sig53`). There is therefore nothing on disk
    to extract an official split from, and this always raises :class:`NotImplementedError`
    with the same actionable blocker message as :func:`download_sig53`. If the team opts in to
    on-cluster TorchSig generation as a separate reviewed step, wire this to the concrete
    ``torchsig.datasets`` layout there and feed the result to :func:`prepare_amc` via
    ``official_split=``. Never called in unit tests.
    """
    resolve_cache_dir(cache)  # keep the cache-resolution contract identical to the real loader
    from rfbench.data.download.amc_sig53 import download_sig53

    # Delegates to the single source of truth for the blocker message + expected root path.
    return download_sig53(cache=cache)  # type: ignore[return-value]  # always raises


__all__ = [
    "AmcDataset",
    "CANONICAL_SPLIT_IDS",
    "SOURCE_URLS",
    "prepare_amc",
    "load_radioml_labels",
    "load_sig53_official_split",
]
