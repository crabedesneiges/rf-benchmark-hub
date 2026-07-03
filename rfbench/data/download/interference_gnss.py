"""Download the Swinney & Woods GNSS-jamming raw-IQ dataset into ``$RFBENCH_CACHE``.

Swinney & Woods (2021), "GNSS Jamming Classification via CNN, Transfer Learning & the Loss
Curvature Metric", is distributed on Zenodo (record 4629685, DOI 10.5281/zenodo.4629685)
under CC-BY-4.0 as a single ``Raw_IQ_Dataset.zip`` (~1.9 GB, no login required) plus the
``signal_generation.m`` MATLAB script that synthesised the six jamming classes. The archive
extracts to per-class raw-IQ files split into a training and a testing folder
(~1000 train + 250 test samples per class).

We never redistribute the archive (D3): this only *fetches* the Zenodo file into the local
cache and extracts it; nothing is ever committed. The transfer (``requests``) is imported
LAZILY with a clear ``pip install rfbench[data]`` error, so importing this module stays
dependency-free and it is NEVER exercised in CI (no network, no heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official Zenodo landing page (dataset description + download links live here).
ZENODO_RECORD_PAGE = "https://zenodo.org/records/4629685"

#: Direct download URL for the raw-IQ archive on Zenodo (no login; CC-BY-4.0).
_ARCHIVE_URL = "https://zenodo.org/records/4629685/files/Raw_IQ_Dataset.zip?download=1"

#: Archive filename fetched from Zenodo.
_ARCHIVE_NAME = "Raw_IQ_Dataset.zip"

#: Cache subdirectory the extracted GNSS-jamming files are written under.
_INTERF_SUBDIR = "interf_gnss6"

_INSTALL_HINT = (
    "Downloading the GNSS-jamming set needs requests; install it with `pip install rfbench[data]`."
)


def download_interference_gnss6(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract ``Raw_IQ_Dataset.zip`` into ``$RFBENCH_CACHE/interf_gnss6/``.

    By default the archive is fetched from its canonical Zenodo URL (:data:`_ARCHIVE_URL`,
    obtained from :data:`ZENODO_RECORD_PAGE`); pass a mirror ``source_url`` (a direct archive
    URL) to override. If the destination directory already holds extracted files and ``force``
    is ``False`` the download is skipped (idempotent). Returns the extraction directory.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    dest_dir = resolve_cache_dir(cache) / _INTERF_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not force and _has_extracted_files(dest_dir):
        return dest_dir

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    url = source_url or _ARCHIVE_URL
    archive = dest_dir / (Path(source_url).name if source_url else _ARCHIVE_NAME)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    except requests.exceptions.RequestException as exc:  # network/HTTP failure
        raise RuntimeError(
            f"could not fetch the GNSS-jamming archive from {url}; obtain "
            f"{_ARCHIVE_NAME!r} manually from {ZENODO_RECORD_PAGE}, place it in {dest_dir}, "
            "and re-run (or pass source_url= for a mirror)."
        ) from exc

    _extract_archive(archive, dest_dir)
    if not _has_extracted_files(dest_dir):
        raise FileNotFoundError(
            f"fetched {archive.name} but no extracted files were produced under {dest_dir}; "
            "the archive layout may have changed (confirm on the cluster)."
        )
    return dest_dir


def _has_extracted_files(dest_dir: Path) -> bool:
    """Return whether ``dest_dir`` already holds any extracted (non-archive) file."""
    return any(p.is_file() and p.suffix.lower() != ".zip" for p in dest_dir.rglob("*"))


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.zip`` / ``.tar[.gz]`` archive into ``dest_dir`` (stdlib only)."""
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    # else: the download was already the raw file; nothing to do.


__all__ = [
    "ZENODO_RECORD_PAGE",
    "download_interference_gnss6",
]
