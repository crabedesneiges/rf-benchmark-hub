"""Download the REAL wideband RF detection dataset into ``$RFBENCH_CACHE`` -- WP-13.

The wideband-detection track in ``docs/EVALUATION_PROTOCOL.md`` names **WBSig53**. Recon
established that WBSig53 has **no static published artifact**: like Sig53 it is *generated*
from TorchSig's wideband generator, and per the data-layer rule (real published datasets
only; NO synthetic generation) TorchSig generation is OUT. WBSig53 is therefore reported as
a blocker and left as a clear, documented stub (:func:`load_wbsig53_annotations`) that
raises with guidance rather than faking data.

The real, static, downloadable stand-in used by a reference wideband-detection paper is
**RadDet** (Huang et al., "RadDet: A Wideband Dataset for Real-Time Radar Spectrum
Detection", ICASSP 2025, arXiv:2501.10407). RadDet is a published wideband radar-spectrum
detection corpus of **spectrogram images with time-frequency bounding boxes in YOLO
format** -- exactly the per-sample T-F boxes
:func:`rfbench.data.prepare.detection.prepare_detection` consumes. It is hosted on Kaggle
(``abcxyzi/raddet-icassp-2025``, ~58 GB, CC BY-NC 4.0) as multi-part tar archives; the
extracted layout is ``images/{train,val,test}/*.png`` with a sibling ``*.txt`` YOLO label
file per image and a ``data.yaml`` class map.

We never redistribute either dataset (D3): the download helpers only *fetch* the artifact
the entitled user is authorised to obtain into the local cache and extract it; nothing is
ever committed. Kaggle gates downloads behind account credentials / an accepted dataset
licence, so we do NOT scrape -- :func:`download_raddet` raises a clear error with
manual-download instructions and the expected on-disk path/filename.

Dependency discipline: module-top imports are stdlib only. ``requests`` (transfer) and
``numpy`` (optional post-parse sanity checks) are imported LAZILY inside functions with a
``pip install rfbench[data]`` hint, so ``import rfbench.data.download.detection_wbsig53``
stays dependency-free and is NEVER exercised for its heavy paths in CI. The YOLO
label-parsing helper is pure stdlib so the box-extraction path is unit-testable on
synthetic fixtures with no numpy/network.

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

# --- WBSig53 (named dataset; generation-only -> blocker, kept as a stub) --------------

#: Official TorchSig repository (WBSig53 is generated here; no static release exists).
TORCHSIG_REPO = "https://github.com/TorchDSP/torchsig"

_WBSIG53_INSTALL_HINT = (
    "WBSig53 has no static published artifact -- it is generated via TorchSig (torch + "
    "numpy). The RF-Benchmark-Hub data layer uses REAL published datasets only and does "
    "NOT generate synthetic data, so WBSig53 is a blocker; use RadDet "
    "(download_raddet / load_raddet_annotations) for the wideband-detection track instead."
)


def generate_wbsig53(
    *,
    cache: str | Path | None = None,
    impaired: bool = True,
    force: bool = False,
) -> Path:
    """Blocker stub: WBSig53 generation is disabled (real published datasets only).

    WBSig53 has no static downloadable artifact -- it exists only as the output of
    TorchSig's wideband generator. The RF-Benchmark-Hub data layer uses REAL published
    datasets and does NOT generate synthetic data, so this generation entry point is
    intentionally disabled and raises with guidance rather than driving TorchSig. Use
    :func:`download_raddet` + :func:`load_raddet_annotations` (the real, published RadDet
    dataset) for the wideband-detection track.

    The signature (``cache`` / ``impaired`` / ``force``) is preserved so the CLI/task import
    surface is unchanged. ``cache`` is resolved eagerly only to keep parity with the real
    downloaders. NEVER called in unit tests.
    """
    _ = (impaired, force)
    resolve_cache_dir(cache)  # keep signature parallel; nothing is generated/read.
    raise NotImplementedError(_WBSIG53_INSTALL_HINT)


def load_wbsig53_annotations(
    cache: str | Path | None = None,
) -> list[dict[str, object]]:
    """Blocker stub: WBSig53 is generation-only, so it is not loaded here.

    WBSig53 ships no static downloadable artifact; obtaining it requires running TorchSig's
    wideband generator, which the data-layer rule forbids (real published datasets only, no
    synthetic generation). This function is intentionally left as a documented stub that
    raises with guidance -- it does NOT fabricate boxes. Use :func:`load_raddet_annotations`
    (the real, published RadDet dataset) for the wideband-detection track.

    ``cache`` is resolved eagerly only to keep the call signature parallel with the real
    loaders. NEVER called in unit tests.
    """
    resolve_cache_dir(cache)  # keep signature parallel; nothing is read.
    raise NotImplementedError(_WBSIG53_INSTALL_HINT)


# --- RadDet (REAL, static, published wideband-detection dataset) ---------------------

#: Official RadDet landing page (paper + Kaggle download links live here).
RADDET_REPO = "https://github.com/abcxyzi/RadDet"

#: Kaggle dataset the RadDet archives are published under (credential/licence gated).
RADDET_KAGGLE_SLUG = "abcxyzi/raddet-icassp-2025"

#: Kaggle dataset page recorded in the manifest for provenance (never redistributed, D3).
RADDET_KAGGLE_URL = "https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025"

#: Cache subdirectory the extracted RadDet tree is written under.
_RADDET_SUBDIR = "raddet"

#: RadDet YOLO class order (from the published ``data.yaml``); index == class id.
RADDET_CLASSES: tuple[str, ...] = (
    "Rect",
    "Barker",
    "Frank",
    "P1",
    "P2",
    "P3",
    "P4",
    "Px",
    "ZadoffChu",
    "LFM",
    "FMCW",
)

#: RadDet YOLO splits, in canonical order (subdirs of ``images/``).
RADDET_YOLO_SPLITS: tuple[str, str, str] = ("train", "val", "test")

_RADDET_INSTALL_HINT = (
    "Downloading/verifying RadDet needs requests (+ numpy for optional checks); "
    "install them with `pip install rfbench[data]`."
)

_RADDET_MANUAL_HINT = (
    "RadDet is published on Kaggle behind account credentials + an accepted dataset "
    f"licence (CC BY-NC 4.0): {RADDET_KAGGLE_URL}\n"
    "It is not scraped. Download it manually, e.g.:\n"
    f"    kaggle datasets download -d {RADDET_KAGGLE_SLUG} -p <RFBENCH_CACHE>/raddet\n"
    "then extract + recombine the multi-part tars so the tree looks like:\n"
    "    <RFBENCH_CACHE>/raddet/images/{train,val,test}/*.png\n"
    "    <RFBENCH_CACHE>/raddet/images/{train,val,test}/*.txt   (YOLO boxes)\n"
    "    <RFBENCH_CACHE>/raddet/data.yaml\n"
    "(the .txt sits next to its .png with the same stem). Pass the resulting root as "
    "`cache=` / via $RFBENCH_CACHE."
)


def download_raddet(
    *,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Locate (or refuse to auto-fetch) the RadDet dataset under ``$RFBENCH_CACHE/raddet/``.

    RadDet is distributed on Kaggle behind account credentials and an accepted dataset
    licence, so it CANNOT be fetched non-interactively without the user's Kaggle token. We
    do not scrape credential-gated sources: if the extracted tree is already present (and
    ``force`` is ``False``) its root is returned; otherwise a :class:`RuntimeError` is
    raised with manual-download instructions and the exact expected on-disk layout.

    Returns the RadDet root directory (``$RFBENCH_CACHE/raddet``). ``requests`` is imported
    lazily only if a non-gated ``source`` mechanism is ever added; today the gate always
    routes to the manual path. NEVER called in unit tests.
    """
    root = resolve_cache_dir(cache) / _RADDET_SUBDIR
    if _looks_populated(root) and not force:
        return root

    # Surface the lazy-dep hint before the manual-download hint so a user missing the
    # optional extra gets the actionable install message first.
    try:
        import requests  # noqa: F401 - imported to surface the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(_RADDET_INSTALL_HINT) from exc

    raise RuntimeError(_RADDET_MANUAL_HINT)


def load_raddet_annotations(
    cache: str | Path | None = None,
    *,
    splits: tuple[str, ...] = RADDET_YOLO_SPLITS,
) -> list[dict[str, object]]:
    """Extract per-sample time-frequency box annotations from a RadDet YOLO tree.

    Reads the extracted RadDet layout under ``$RFBENCH_CACHE/raddet/images/<split>/`` --
    one spectrogram ``*.png`` per wideband capture with a sibling YOLO ``*.txt`` label file
    (same stem) holding ``class_id x_center y_center width height`` rows, all normalised to
    ``[0, 1]``. Each returned element is a mapping with a stable ``sample_id`` (``"<split>/
    <stem>"``) and a ``boxes`` list; every box carries a signal ``class`` (mapped through
    :data:`RADDET_CLASSES`) plus its time/frequency extent
    ``(t_start, t_stop, f_low, f_high)`` in normalised ``[0, 1]`` coordinates -- exactly the
    shape :func:`rfbench.data.prepare.detection.prepare_detection` consumes as ``samples=``.

    Axis convention (RadDet spectrograms, arXiv:2501.10407): the horizontal image axis is
    **time**, the vertical is **frequency**; YOLO ``y`` is measured from the image top, so a
    higher ``y`` is a lower frequency. The center/width YOLO box is converted to
    ``[lo, hi]`` extents by :func:`_yolo_to_tf_box`, which clamps to ``[0, 1]`` to absorb
    sub-pixel rounding in the published labels.

    Both the directory walk and the YOLO text parsing
    (:func:`_parse_yolo_label_file`, :func:`_yolo_to_tf_box`) are pure stdlib -- no
    numpy/requests/network -- so this loader is unit-tested end-to-end on a synthetic RadDet
    tree (dummy ``*.png`` + real ``*.txt`` YOLO labels) that mimics the published layout.
    Raises :class:`FileNotFoundError` with the download guidance when the tree is absent.
    """
    root = resolve_cache_dir(cache) / _RADDET_SUBDIR
    images_root = root / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"RadDet not found at {images_root}; run the download step first.\n"
            + _RADDET_MANUAL_HINT
        )

    samples: list[dict[str, object]] = []
    for split in splits:
        split_dir = images_root / split
        if not split_dir.is_dir():
            continue
        for png in sorted(split_dir.glob("*.png")):
            label_file = png.with_suffix(".txt")
            rows = _parse_yolo_label_file(label_file) if label_file.exists() else []
            boxes = [_yolo_row_to_box(row) for row in rows]
            samples.append({"sample_id": f"{split}/{png.stem}", "boxes": boxes})
    if not samples:
        raise FileNotFoundError(
            f"RadDet tree at {images_root} contained no <split>/*.png captures.\n"
            + _RADDET_MANUAL_HINT
        )
    return samples


# --- pure-stdlib YOLO parsing (unit-testable; no numpy/network) ----------------------


def _parse_yolo_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO ``*.txt`` label file into ``(class_id, xc, yc, w, h)`` rows (stdlib).

    Each non-blank line is ``class_id x_center y_center width height`` with the four box
    fields normalised to ``[0, 1]``. Blank lines are skipped (YOLO's convention for a
    background/empty frame is simply an empty file). Malformed lines raise
    :class:`ValueError` so a corrupt label fails loudly at prepare time.
    """
    return _parse_yolo_label_text(path.read_text(encoding="utf-8"), source=str(path))


def _parse_yolo_label_text(
    text: str, *, source: str = "<yolo>"
) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label *text* into ``(class_id, xc, yc, w, h)`` rows (stdlib; testable)."""
    rows: list[tuple[int, float, float, float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{source}:{lineno}: YOLO row needs 5 fields "
                f"(class_id xc yc w h); got {len(parts)}: {line!r}"
            )
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: non-numeric YOLO field in {line!r}") from exc
        rows.append((class_id, xc, yc, w, h))
    return rows


def _yolo_row_to_box(row: tuple[int, float, float, float, float]) -> dict[str, object]:
    """Map a parsed YOLO row to a prepare_detection T-F box dict (class name + extents)."""
    class_id, xc, yc, w, h = row
    return _yolo_to_tf_box(class_id, xc, yc, w, h)


def _yolo_to_tf_box(class_id: int, xc: float, yc: float, w: float, h: float) -> dict[str, object]:
    """Convert one normalised YOLO box to a T-F box (x=time, y=frequency), clamped to [0,1].

    ``(xc, yc, w, h)`` are normalised center/size. Time uses the horizontal axis, frequency
    the vertical axis; both extents are ``center +/- size/2``, clamped to ``[0, 1]`` to
    absorb the sub-pixel rounding present in the published RadDet labels. The class id is
    resolved through :data:`RADDET_CLASSES`; an out-of-range id falls back to its raw
    ``"class_<id>"`` string rather than raising, so an unseen class does not abort a load.
    """
    t_start, t_stop = _extent(xc, w)
    f_low, f_high = _extent(yc, h)
    if 0 <= class_id < len(RADDET_CLASSES):
        cls: object = RADDET_CLASSES[class_id]
    else:
        cls = f"class_{class_id}"
    return {
        "class": cls,
        "t_start": t_start,
        "t_stop": t_stop,
        "f_low": f_low,
        "f_high": f_high,
    }


def _extent(center: float, size: float) -> tuple[float, float]:
    """``center +/- size/2`` clamped to ``[0, 1]`` (guards published sub-pixel rounding)."""
    half = size / 2.0
    lo = _clamp01(center - half)
    hi = _clamp01(center + half)
    if lo > hi:  # only possible from a degenerate/negative size; normalise defensively.
        lo, hi = hi, lo
    return lo, hi


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the normalised ``[0, 1]`` range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _looks_populated(root: Path) -> bool:
    """Cheap idempotency check: a RadDet root with an ``images/`` subtree is populated."""
    return (root / "images").is_dir() and any((root / "images").iterdir())


__all__ = [
    "TORCHSIG_REPO",
    "generate_wbsig53",
    "load_wbsig53_annotations",
    "RADDET_REPO",
    "RADDET_KAGGLE_SLUG",
    "RADDET_KAGGLE_URL",
    "RADDET_CLASSES",
    "RADDET_YOLO_SPLITS",
    "download_raddet",
    "load_raddet_annotations",
]
