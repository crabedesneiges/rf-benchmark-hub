"""Wideband detection canonical splits + T-F box annotations -- WP-13.

Builds the canonical split per the SPLIT POLICY (``docs/EVALUATION_PROTOCOL.md``
§Wideband detection):

* adopt the **official split** verbatim when one is provided;
* otherwise a deterministic **80/10/10** split over samples, seed 42.

Canonical ids ``detect-<dataset>-<track>-8010-seed42-v1`` (generated) /
``detect-<dataset>-<track>-official-v1`` (adopted), where ``<track>`` keeps the
**detection** and **recognition** tracks distinct: the *detection* track scores signal
presence/localisation (boxes only), the *recognition* track additionally scores the
per-box signal class. Both tracks share the same samples but are prepared and reported
separately (never mixed in one column).

Dataset choice (see :data:`SOURCE_URLS`): the protocol names **WBSig53**, but WBSig53 has
no static published artifact -- it is generated via TorchSig, which the data-layer rule
forbids (real published datasets only). WBSig53 is therefore a blocker (kept as a
documented stub in the download module). The real, static, published stand-in used by a
reference wideband-detection paper is **RadDet** (ICASSP 2025, arXiv:2501.10407): published
spectrograms with time-frequency bounding boxes in YOLO format. Both ids are supported here
so the split/annotation machinery is dataset-agnostic; ``raddet`` is the one that resolves
to real data on the cluster.

Unlike AMC (a single scalar label per item), each detection sample carries a *list of
time-frequency boxes*. The frozen :class:`~rfbench.core.manifest.DatasetManifest` has no
place for those, so this module writes an **annotations sidecar**
(``<id>.annotations.json``) next to the split index + dataset manifest, round-tripping the
boxes with a stable checksum.

Split GENERATION + annotation recording are decoupled from data loading:
:func:`prepare_detection` accepts already-extracted per-sample boxes (plain Python dicts),
so the whole path runs on pure-stdlib synthetic fixtures with no numpy/torchsig. The heavy
box EXTRACTION from the real dataset root lives in the lazy
:func:`rfbench.data.download.detection_wbsig53.load_raddet_annotations` (RadDet;
``load_wbsig53_annotations`` remains a blocker stub), never called in unit tests.

Module-top imports are stdlib + the frozen core contracts only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from rfbench.core.manifest import DatasetManifest
from rfbench.core.splits import SplitManifest
from rfbench.data.prepare._common import (
    prepare_from_labels,
    prepare_from_official,
)

#: The wideband-detection datasets this WP prepares. ``raddet`` is the real, published
#: dataset (ICASSP 2025); ``wbsig53`` is the protocol-named dataset kept for compatibility
#: but blocked (generation-only, see the download module).
DetectionDataset = Literal["raddet", "wbsig53"]

#: The two reporting tracks kept distinct (detection = boxes only, recognition = +class).
DetectionTrack = Literal["detection", "recognition"]

#: Recognised tracks (kept in lockstep with :data:`DetectionTrack`).
TRACKS: tuple[DetectionTrack, DetectionTrack] = ("detection", "recognition")

#: Official source URL per dataset, recorded in the manifest (provenance; never
#: redistributed, D3). RadDet is the real published artifact; WBSig53 points at TorchSig
#: (its generation-only origin).
SOURCE_URLS: dict[str, str] = {
    "raddet": "https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025",
    "wbsig53": "https://github.com/TorchDSP/torchsig",
}

#: Fallback provenance URL for a dataset id not in :data:`SOURCE_URLS`.
_DEFAULT_SOURCE_URL = "https://github.com/abcxyzi/RadDet"

#: Backward-compatible alias (WBSig53 provenance); prefer :data:`SOURCE_URLS`.
SOURCE_URL = SOURCE_URLS["wbsig53"]


def canonical_split_id(
    dataset: DetectionDataset | str, track: DetectionTrack | str, *, official: bool
) -> str:
    """Canonical split id for a ``(dataset, track)`` pair.

    ``official`` selects the adopted-split id (``...-official-v1``) vs the generated
    deterministic-split id (``...-8010-seed42-v1``); the ratios+seed are baked in so
    changing either is a breaking change that bumps the task version. The ``track``
    segment keeps detection and recognition ids -- and therefore their on-disk sidecars --
    distinct.
    """
    stem = f"detect-{dataset}-{track}"
    return f"{stem}-official-v1" if official else f"{stem}-8010-seed42-v1"


def prepare_detection(
    dataset: DetectionDataset | str,
    *,
    out_dir: str | Path,
    samples: Sequence[Mapping[str, object]],
    track: DetectionTrack | str = "detection",
    official_split: Mapping[str, Sequence[int]] | None = None,
    source_checksums: Mapping[str, str] | None = None,
    seed: int = 42,
) -> tuple[SplitManifest, DatasetManifest, Path]:
    """Build the canonical detection split + T-F box annotations for ``dataset``.

    ``samples`` is one mapping per wideband capture, in canonical (index) order, each with:

    * ``"sample_id"`` (optional): a stable id; defaults to the positional index;
    * ``"boxes"``: a list of time-frequency boxes, each a mapping with a signal
      ``"class"`` and a normalised extent ``"t_start" <= "t_stop"`` in ``[0, 1]`` and
      ``"f_low" <= "f_high"`` in ``[0, 1]``.

    Split GENERATION runs on these plain dicts (no numpy/torchsig), so it is exercisable on
    synthetic fixtures. If ``official_split`` is given the WBSig53/TorchSig partition is
    adopted verbatim; otherwise a deterministic **80/10/10** split over the samples (seed
    42) is generated. The split is unstratified: detection samples have no single scalar
    label to stratify on (each carries a *set* of boxes).

    ``track`` (``"detection"`` | ``"recognition"``) is baked into the canonical id so the
    two tracks produce distinct sidecars and are never conflated. For the *detection* track
    box classes are still recorded (round-tripped) but are not part of the scored target;
    the *recognition* track scores them too -- the distinction is a reporting concern, kept
    explicit in the manifest via ``track``.

    Writes, under ``<out_dir>/splits/<dataset>/`` only (never raw data, D3):

    * ``<id>.idx.json`` -- the split indices (via the core split writer);
    * ``<id>.manifest.json`` -- the :class:`DatasetManifest` sidecar;
    * ``<id>.annotations.json`` -- the per-sample T-F boxes + a stable checksum.

    Returns ``(SplitManifest, DatasetManifest, annotations_path)``.
    """
    if track not in TRACKS:
        raise ValueError(f"unknown detection track {track!r}; expected one of {list(TRACKS)}")

    n_items = len(samples)
    boxes_by_sample = _normalise_samples(samples)
    is_official = official_split is not None
    split_id = canonical_split_id(dataset, track, official=is_official)
    source_url = SOURCE_URLS.get(str(dataset), _DEFAULT_SOURCE_URL)

    if is_official:
        assert official_split is not None  # narrowed for mypy; guarded by is_official
        split, manifest = prepare_from_official(
            dataset=dataset,
            split_id=split_id,
            official=official_split,
            source_url=source_url,
            out_dir=out_dir,
            source_checksums=source_checksums,
            seed=seed,
        )
    else:
        split, manifest = prepare_from_labels(
            dataset=dataset,
            split_id=split_id,
            n_items=n_items,
            strata=None,  # detection: one box-set per sample, no scalar stratum label
            source_url=source_url,
            out_dir=out_dir,
            source_checksums=source_checksums,
            seed=seed,
        )

    # The idx + manifest sidecars are already written by prepare_from_{labels,official} (via
    # _finalise); we only add the annotations sidecar (boxes live separately because
    # DatasetManifest is a frozen contract).
    annotations_path = write_detection_annotations(
        dataset=dataset,
        split_id=split_id,
        track=track,
        boxes_by_sample=boxes_by_sample,
        out_dir=out_dir,
    )
    return split, manifest, annotations_path


def write_detection_annotations(
    *,
    dataset: DetectionDataset | str,
    split_id: str,
    track: DetectionTrack | str,
    boxes_by_sample: Sequence[Sequence[Mapping[str, object]]],
    out_dir: str | Path,
) -> Path:
    """Write ``<out_dir>/splits/<dataset>/<split_id>.annotations.json`` (deterministic).

    The document records the ``dataset`` / ``canonical_split_id`` / ``track`` plus one
    entry per sample: its ``sample_id`` and its list of normalised T-F ``boxes``. A stable
    ``sha256`` over the canonical box payload is embedded so the annotations can be
    integrity-checked alongside the split index. Serialised with ``sort_keys`` so the file
    is reproducible byte-for-byte. Returns the written path.
    """
    per_sample = [
        {"sample_id": i, "boxes": [_canonical_box(box) for box in boxes]}
        for i, boxes in enumerate(boxes_by_sample)
    ]
    doc = {
        "dataset": dataset,
        "canonical_split_id": split_id,
        "track": track,
        "n_samples": len(per_sample),
        "checksum": annotations_checksum(per_sample),
        "samples": per_sample,
    }
    dest_dir = Path(out_dir) / "splits" / str(dataset)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{split_id}.annotations.json"
    dest.write_text(
        json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def annotations_checksum(per_sample: Sequence[Mapping[str, object]]) -> str:
    """``"sha256:<hex>"`` over the canonical per-sample box payload.

    Order-stable and format-stable (``sort_keys`` + compact separators), so it can be
    recomputed to verify the annotations sidecar was not tampered with.
    """
    payload = json.dumps(list(per_sample), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --- private helpers ----------------------------------------------------------------


def _normalise_samples(
    samples: Sequence[Mapping[str, object]],
) -> list[list[dict[str, object]]]:
    """Validate + extract the ordered per-sample box lists from raw sample mappings."""
    boxes_by_sample: list[list[dict[str, object]]] = []
    for pos, sample in enumerate(samples):
        raw_boxes = sample.get("boxes", [])
        if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, (str, bytes)):
            raise ValueError(f"sample {pos} 'boxes' must be a list of box mappings")
        boxes_by_sample.append([_canonical_box(box) for box in raw_boxes])
    return boxes_by_sample


def _canonical_box(box: Mapping[str, object]) -> dict[str, object]:
    """Validate one T-F box and return it as a plain dict in canonical field order.

    A box localises a signal in time (``t_start`` <= ``t_stop``) and frequency
    (``f_low`` <= ``f_high``), both normalised to ``[0, 1]``, and carries a signal
    ``class`` label. Bounds are checked so a malformed annotation fails loudly at prepare
    time rather than during evaluation.
    """
    try:
        cls = box["class"]
        t_start = float(box["t_start"])  # type: ignore[arg-type]
        t_stop = float(box["t_stop"])  # type: ignore[arg-type]
        f_low = float(box["f_low"])  # type: ignore[arg-type]
        f_high = float(box["f_high"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "each T-F box needs 'class' + numeric 't_start'/'t_stop'/'f_low'/'f_high'; "
            f"got {box!r}"
        ) from exc

    for name, lo, hi in (("time", t_start, t_stop), ("freq", f_low, f_high)):
        if lo > hi:
            raise ValueError(f"box {name} extent is inverted ({lo} > {hi}) in {box!r}")
    for name, value in (
        ("t_start", t_start),
        ("t_stop", t_stop),
        ("f_low", f_low),
        ("f_high", f_high),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"box {name}={value} outside normalised [0, 1] in {box!r}")

    return {
        "class": cls,
        "t_start": t_start,
        "t_stop": t_stop,
        "f_low": f_low,
        "f_high": f_high,
    }


__all__ = [
    "DetectionDataset",
    "DetectionTrack",
    "TRACKS",
    "SOURCE_URL",
    "SOURCE_URLS",
    "canonical_split_id",
    "prepare_detection",
    "write_detection_annotations",
    "annotations_checksum",
]
