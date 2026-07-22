"""Locate (or refuse to auto-fetch) the DeepSense spectrum-sensing dataset -- spectrum_sensing.

The spectrum-sensing track in ``docs/EVALUATION_PROTOCOL.md`` §"Spectrum sensing" names
**DeepSense** (Uvaydov, D'Oro, Restuccia, Melodia, "DeepSense: Fast Wideband Spectrum Sensing
Through Real-Time In-the-Loop Deep Learning", IEEE INFOCOM 2021, DOI
10.1109/INFOCOM42981.2021.9488764). DeepSense is over-the-air (OTA) wideband IQ captured with
802.11 a/g + LTE-M transmitters, sliced into fixed-length windows each carrying a binary
spectrum-occupancy label (occupied vs vacant). Code + data live on the wineslab GitHub repo:
https://github.com/wineslab/deepsense-spectrum-sensing-datasets .

The dataset is LARGE and distributed as gated / manual archives (on the wineslab repo or an
external host it points to), so -- exactly like RadDet (``detection_wbsig53.py``) -- we do NOT
scrape it: if the extracted tree is already present its root is returned, otherwise
:func:`download_deepsense` raises a clear error with the manual-download procedure and the exact
expected on-disk layout under ``$RFBENCH_CACHE/deepsense/``.

LICENSE: the DeepSense dataset license is **UNSTATED** on the wineslab repo at the time of
writing; treat redistribution as forbidden (D3) and consult the authors before any re-hosting.

We never redistribute the data (D3): these helpers only *locate/parse* an artifact the entitled
user has already obtained into their local cache; nothing is ever committed. Module-top imports
are stdlib only; ``requests`` (transfer) and ``numpy`` (window/label loading) are imported
LAZILY inside functions with a ``pip install rfbench[data]`` hint, so
``import rfbench.data.download.spectrum_deepsense`` stays dependency-free and is NEVER exercised
for its heavy paths in CI.

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official wineslab landing page (paper + dataset download links live here).
DEEPSENSE_REPO = "https://github.com/wineslab/deepsense-spectrum-sensing-datasets"

#: Cache subdirectory the extracted DeepSense tree is written under.
_DEEPSENSE_SUBDIR = "deepsense"

#: The two binary occupancy classes, in canonical (index) order: 0 == vacant, 1 == occupied.
OCCUPANCY_CLASSES: tuple[str, str] = ("vacant", "occupied")

_DEEPSENSE_INSTALL_HINT = (
    "Loading/verifying DeepSense needs numpy (+ requests for any transfer); "
    "install them with `pip install rfbench[data]`."
)

_DEEPSENSE_MANUAL_HINT = (
    "DeepSense is a large OTA wideband-IQ spectrum-sensing dataset distributed manually "
    f"(gated / external host) from the wineslab repo: {DEEPSENSE_REPO}\n"
    "Its license is UNSTATED -- do not re-host it. It is not scraped. Download it manually "
    "following the repo's instructions, then extract it so the tree looks like:\n"
    "    <RFBENCH_CACHE>/deepsense/windows.npy    (N x 2 x L raw-IQ windows, or N x L complex)\n"
    "    <RFBENCH_CACHE>/deepsense/labels.npy     (N binary labels: 0 vacant / 1 occupied)\n"
    "  (or the per-split equivalents documented in load_deepsense_occupancy)\n"
    "Pass the resulting root as `cache=` / via $RFBENCH_CACHE.\n"
    "NOTE: the exact wineslab on-disk binary layout is NOT confirmed in-repo; a cluster run "
    "must verify it and, if it differs, adjust load_deepsense_occupancy's loader (one function)."
)


def download_deepsense(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Locate (or refuse to auto-fetch) DeepSense under ``$RFBENCH_CACHE/deepsense/``.

    DeepSense is distributed manually (gated / external host) from the wineslab repo, so it
    CANNOT be fetched non-interactively. We do not scrape gated sources: if the extracted tree
    is already present (and ``force`` is ``False``) its root is returned; otherwise a
    :class:`RuntimeError` is raised with manual-download instructions and the exact expected
    on-disk layout.

    ``source_url`` is accepted for signature parity with the other downloaders (and for a
    future mirror), but today the gate always routes to the manual path. Returns the DeepSense
    root directory (``$RFBENCH_CACHE/deepsense``). ``requests`` is imported lazily only to
    surface a clear install hint before the manual hint. NEVER called in unit tests.
    """
    _ = source_url  # accepted for parity; the manual gate ignores it today.
    root = resolve_cache_dir(cache) / _DEEPSENSE_SUBDIR
    if _looks_populated(root) and not force:
        return root

    # Surface the lazy-dep hint before the manual-download hint so a user missing the optional
    # extra gets the actionable install message first.
    try:
        import requests  # noqa: F401 - imported to surface the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(_DEEPSENSE_INSTALL_HINT) from exc

    raise RuntimeError(_DEEPSENSE_MANUAL_HINT)


def load_deepsense_occupancy(cache: str | Path | None = None) -> list[tuple[object, int]]:
    """Load per-window ``(iq, label)`` pairs from the extracted DeepSense tree (lazy numpy).

    Reads the extracted DeepSense layout under ``$RFBENCH_CACHE/deepsense/`` and returns one
    ``(window, label)`` pair per raw-IQ window, where ``label`` is ``1`` for an occupied window
    and ``0`` for a vacant one -- exactly the per-window supervision
    :class:`~rfbench.tasks.spectrum_sensing.dataset.SpectrumSensingDataset` materialises and
    :func:`rfbench.data.prepare.sensing.prepare_sensing` stratifies by. This is parallel to
    :func:`rfbench.data.download.detection_wbsig53.load_raddet_annotations`.

    ON-DISK LAYOUT (see :data:`_DEEPSENSE_MANUAL_HINT`): the exact wineslab binary format is NOT
    confirmed in-repo. The single, centralised assumption a cluster run may adjust is captured in
    :func:`extract_occupancy_labels`, which maps a described manifest -> a ``0/1`` label list.
    This loader reads sibling ``windows.npy`` (``N x 2 x L`` real, or ``N x L`` complex converted
    to ``2 x L`` I/Q) and ``labels.npy`` (``N`` binary occupancy) arrays; if the real archive
    instead ships per-split files or a different container, change only this reader + the label
    extractor. Never called in unit tests (needs the real data + numpy).

    Raises :class:`FileNotFoundError` with the manual-download guidance when the tree is absent.
    """
    root = resolve_cache_dir(cache) / _DEEPSENSE_SUBDIR
    windows_path = root / "windows.npy"
    labels_path = root / "labels.npy"
    if not windows_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"DeepSense not found at {root} (expected windows.npy + labels.npy); "
            "run the download step first.\n" + _DEEPSENSE_MANUAL_HINT
        )

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(_DEEPSENSE_INSTALL_HINT) from exc

    raw = np.load(windows_path)
    labels = extract_occupancy_labels(list(np.asarray(np.load(labels_path))))
    if len(raw) != len(labels):
        raise ValueError(
            f"DeepSense windows/labels length mismatch: {len(raw)} windows vs {len(labels)} labels"
        )
    pairs: list[tuple[object, int]] = []
    for window, label in zip(raw, labels, strict=True):
        arr = np.asarray(window)
        if np.iscomplexobj(arr):  # N x L complex -> (2, L) real I/Q, channel-first.
            iq = np.stack([arr.real, arr.imag]).astype(np.float32)
        else:  # already real; keep (2, L) if channel-first, else reshape.
            iq = arr.astype(np.float32)
            iq = iq if iq.ndim == 2 and iq.shape[0] == 2 else iq.reshape(2, -1)
        pairs.append((iq, label))
    return pairs


#: Max windows kept PER DeepSense ``.h5`` file when building the committed split index. Each lte_m
#: file holds ~470k (train) / ~52k (test) windows; an uncapped split over all SNRs would be a ~30 MB
#: index. Capping the first-k windows/file keeps it committable (like ORACLE/POWDER) while keeping
#: DeepSense's official train/test partition; the array loader must apply the SAME cap + sorted-file
#: order so indices align.
_DEEPSENSE_WINDOWS_PER_FILE = 8192


def load_deepsense_records(
    cache: str | Path | None = None,
    *,
    windows_per_file: int = _DEEPSENSE_WINDOWS_PER_FILE,
) -> tuple[int, dict[str, list[int]]]:
    """Build DeepSense's OFFICIAL train/test window partition from the published ``lte_m`` files.

    The published DeepSense LTE-M release ships per-SNR HDF5 files
    ``deepsense/lte_m/lte_<snr>_32_{train,test}.h5`` -- ``X`` is ``(2, 32, N)`` raw I/Q (2 channels,
    32-sample window, N windows) and ``y`` is ``(16, N)`` binary per-subband occupancy (multi-label,
    16 LTE-M bands). This reader adopts that official split verbatim (split-policy:
    official-if-provided): every ``*_train.h5`` window joins the train pool, every ``*_test.h5``
    window the test pool (so our test IS DeepSense's test), then a deterministic val slice is carved
    from the train pool. Only the first ``windows_per_file`` windows of each file are kept (see
    :data:`_DEEPSENSE_WINDOWS_PER_FILE`) so the committed index stays small. Returns
    ``(n_items, {"train": [...], "val": [...], "test": [...]})``.

    Reads only the ``y`` dataset SHAPE (fast); h5py is imported lazily. Never called in unit tests.
    """
    root = resolve_cache_dir(cache) / _DEEPSENSE_SUBDIR / "lte_m"
    files = sorted(root.glob("*.h5")) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"DeepSense not found at {root} (expected lte_m/lte_<snr>_32_{{train,test}}.h5); "
            "run the download step first.\n" + _DEEPSENSE_MANUAL_HINT
        )
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(_DEEPSENSE_INSTALL_HINT) from exc

    train: list[int] = []
    test: list[int] = []
    index = 0
    for path in files:
        is_test = path.name.endswith("_test.h5")
        with h5py.File(path, "r") as handle:
            n_windows = int(handle["y"].shape[-1])
        # DeepSense's official split is train-heavy (~9:1); cap test files ~8x smaller than train
        # files so the capped subset keeps a sensible train >> test ratio.
        cap = max(1, windows_per_file // 8) if is_test else windows_per_file
        kept = min(cap, n_windows) if windows_per_file else n_windows
        bucket = test if is_test else train
        for _ in range(kept):
            bucket.append(index)
            index += 1
    # Deterministic val carve: every 10th train window (~10% of train), disjoint from train_only.
    val = train[::10]
    val_set = set(val)
    train_only = [i for i in train if i not in val_set]
    return index, {"train": train_only, "val": val, "test": test}


def extract_occupancy_labels(manifest: list[object]) -> list[int]:
    """Map a described per-window occupancy manifest to a list of ``0/1`` labels (pure stdlib).

    This is the ONE small, centralised, numpy-free place that encodes the DeepSense binary
    occupancy convention, so the prepare/split path is unit-testable with no numpy/data (mirrors
    :func:`rfbench.data.download.sei_powder`'s record extraction). Each ``manifest`` element is a
    per-window occupancy indicator obtained from the wineslab archive; it is coerced to a strict
    ``{0, 1}`` label:

    * a numeric ``0`` / ``1`` (int, float, or a 0-d array exposing ``.item()``) passes through;
    * a truthy/falsey occupancy string (``"occupied"`` / ``"vacant"``, case-insensitive) maps to
      ``1`` / ``0`` respectively (:data:`OCCUPANCY_CLASSES` is the canonical order);
    * anything else raises :class:`ValueError` so a mislabelled manifest fails loudly at prepare
      time rather than silently corrupting the stratification.

    Returns one ``0/1`` int per window, in input order -- ready to hand to
    :func:`rfbench.data.prepare.sensing.prepare_sensing` as ``labels=``. The exact archive
    representation is UNCONFIRMED in-repo; if the real DeepSense manifest encodes occupancy
    differently (e.g. a per-window energy above a threshold), adjust only this function.
    """
    labels: list[int] = []
    for raw in manifest:
        labels.append(_coerce_occupancy(raw))
    return labels


def _coerce_occupancy(raw: object) -> int:
    """Coerce one manifest occupancy indicator to a strict ``{0, 1}`` label (stdlib only)."""
    if isinstance(raw, bool):  # bool is an int subclass; treat it as its 0/1 value.
        return int(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("occupied", "1"):
            return 1
        if token in ("vacant", "0"):
            return 0
        raise ValueError(f"unrecognised DeepSense occupancy string {raw!r}")
    item = getattr(raw, "item", None)
    value = item() if callable(item) else raw
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        as_int = int(value)
        if as_int in (0, 1) and float(value) == as_int:
            return as_int
    raise ValueError(f"DeepSense occupancy label must be 0 or 1, got {raw!r}")


def _looks_populated(root: Path) -> bool:
    """Cheap idempotency check: a DeepSense root holding any extracted file is populated."""
    return root.is_dir() and any(p.is_file() for p in root.rglob("*"))


__all__ = [
    "DEEPSENSE_REPO",
    "OCCUPANCY_CLASSES",
    "download_deepsense",
    "load_deepsense_occupancy",
    "load_deepsense_records",
    "extract_occupancy_labels",
]
