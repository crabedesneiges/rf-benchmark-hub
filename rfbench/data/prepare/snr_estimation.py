"""SNR-estimation canonical split (J4).

SNR estimation is scored on the SAME underlying dataset as AMC -- RadioML 2016.10a -- because
every sample already carries its ground-truth SNR (dB) label. The two tasks therefore share
one partition **by construction**: the split is generated with the identical stratification
key (``modulation x snr``), ratios (80/10/10) and seed (42) as AMC, so the resulting train /
val / test *indices are byte-identical* to
``amc-radioml2016-strat-snr-8010-seed42-v1`` and the two boards are directly comparable on
the exact same held-out signals.

It is nonetheless given its OWN canonical id --
``snr-radioml2016-strat-snr-8010-seed42-v1`` -- so that a ``result.json``'s
``split.canonical_split_id`` is unambiguous about which task's protocol the score attests.
Sharing AMC's id would conflate two distinct (task, split) pairs under one checksum. The
manifest records the intentional link to the AMC split in its ``source_url`` provenance.

Because the indices are a pure function of the ``(modulation, snr)`` label list (never of the
raw IQ), the split is reproducible on the frontend with no numpy/data: either
:func:`prepare_snr_estimation` from the extracted labels (cluster path, mirrors AMC), or
:func:`derive_from_amc_split` which re-uses the already-committed AMC index verbatim under the
SNR id (frontend path -- no dataset needed, indices copied 1:1).

Module top imports are stdlib + the frozen core contracts only; numpy/h5py are never needed
here (label extraction lives in :mod:`rfbench.data.prepare.amc`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest, adopt_official_split, write_split_index
from rfbench.data.prepare._common import prepare_from_labels, write_dataset_manifest
from rfbench.data.prepare.amc import SOURCE_URLS as AMC_SOURCE_URLS

#: Canonical SNR-estimation split id per dataset (distinct id, identical indices to AMC).
SNR_CANONICAL_SPLIT_IDS: dict[str, str] = {
    "radioml_2016_10a": "snr-radioml2016-strat-snr-8010-seed42-v1",
}

#: The AMC split id whose indices each SNR split intentionally mirrors 1:1 (comparability).
MIRRORED_AMC_SPLIT_IDS: dict[str, str] = {
    "radioml_2016_10a": "amc-radioml2016-strat-snr-8010-seed42-v1",
}

#: Provenance note recorded in the SNR manifest's ``source_url`` to make the AMC link explicit.
_LINK_NOTE = "shares the AMC (modulation x snr) 80/10/10 seed-42 split by construction"


def _snr_source_url(dataset: str) -> str:
    """Dataset source URL + the explicit note that the split mirrors AMC's indices."""
    base = AMC_SOURCE_URLS.get(dataset, "")
    return f"{base} ({_LINK_NOTE}: {MIRRORED_AMC_SPLIT_IDS.get(dataset, '?')})"


def prepare_snr_estimation(
    dataset: str,
    *,
    out_dir: str | Path,
    labels: Sequence[tuple[object, int]],
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Build the canonical SNR-estimation split from pre-extracted ``(mod, snr)`` labels.

    Mirrors :func:`rfbench.data.prepare.amc.prepare_amc` exactly (same ``(modulation, snr)``
    stratification, 80/10/10, seed 42) but writes under the SNR canonical id, so the produced
    indices are byte-identical to the AMC split while the ``result.json`` split id stays
    unambiguous. Runs without numpy on synthetic fixtures: the caller extracts ``labels`` via
    :func:`rfbench.data.prepare.amc.load_radioml_labels` (lazy numpy/pickle) on the cluster,
    then calls this. Writes only ``<out_dir>/splits/<dataset>/<snr-id>.{idx,manifest}.json``
    -- never raw data (D3).
    """
    if dataset not in SNR_CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown SNR-estimation dataset {dataset!r}; expected one of "
            f"{sorted(SNR_CANONICAL_SPLIT_IDS)}"
        )
    split_id = SNR_CANONICAL_SPLIT_IDS[dataset]
    strata: list[tuple[object, ...]] = [(mod, snr) for mod, snr in labels]
    return prepare_from_labels(
        dataset=dataset,
        split_id=split_id,
        n_items=len(strata),
        strata=strata,
        source_url=_snr_source_url(dataset),
        out_dir=out_dir,
        seed=seed,
    )


def derive_from_amc_split(
    dataset: str,
    *,
    out_dir: str | Path,
    amc_index_path: str | Path,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest]:
    """Write the SNR split by re-using an already-committed AMC ``.idx.json`` verbatim.

    Frontend-reproducible path (NO numpy, NO dataset): reads the committed AMC split index,
    adopts its exact train / val / test indices under the SNR canonical id via
    :func:`rfbench.core.splits.adopt_official_split`, and writes the SNR ``.idx.json`` +
    manifest. The resulting index checksum is IDENTICAL to the AMC one (the checksum is a
    function of the indices only), which is exactly the property that makes the two boards
    comparable. ``seed`` is recorded for provenance only (the indices are copied, not
    regenerated).

    Use this when the AMC split is already committed and you only need the SNR alias; use
    :func:`prepare_snr_estimation` when generating both from scratch on the cluster.
    """
    import json

    if dataset not in SNR_CANONICAL_SPLIT_IDS:
        raise ValueError(
            f"unknown SNR-estimation dataset {dataset!r}; expected one of "
            f"{sorted(SNR_CANONICAL_SPLIT_IDS)}"
        )
    split_id = SNR_CANONICAL_SPLIT_IDS[dataset]
    doc = json.loads(Path(amc_index_path).read_text(encoding="utf-8"))
    raw_indices = doc.get("indices", {})
    official = {
        name: [int(i) for i in raw_indices.get(name, [])] for name in ("train", "val", "test")
    }
    n_items = sum(len(v) for v in official.values())

    split = adopt_official_split(official, split_id=split_id, dataset=dataset, seed=seed)
    split_checksum = write_split_index(split, str(out_dir))
    manifest = DatasetManifest(
        dataset=dataset,
        canonical_split_id=split_id,
        source_url=_snr_source_url(dataset),
        seed=seed,
        n_items=n_items,
        split_checksum=split_checksum,
        source_checksums={},
        created_at=_utc_now_iso(),
    )
    write_dataset_manifest(manifest, out_dir)
    return split, manifest


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` timestamp (matches the CLI/prepare format)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "SNR_CANONICAL_SPLIT_IDS",
    "MIRRORED_AMC_SPLIT_IDS",
    "prepare_snr_estimation",
    "derive_from_amc_split",
]
