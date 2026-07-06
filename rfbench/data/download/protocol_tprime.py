"""Download the T-PRIME OTA WiFi raw-IQ dataset into ``$RFBENCH_CACHE``.

T-PRIME (Belgiovine et al., "T-PRIME: Transformer-based Protocol Identification for
Machine-learning at the Edge", arXiv:2401.04837, extended IEEE INFOCOM 2024; Genesys Lab /
Northeastern, code at github.com/genesys-neu/t-prime) is a **real over-the-air** raw-IQ
dataset of four 802.11 standards (``802.11b``, ``802.11g``, ``802.11n``, ``802.11ax``),
captured at 20 MHz. The corpus is hosted on Northeastern's Digital Repository Service (DRS)
and is openly downloadable (no login).

CONFIRMED (2026-07, cross-checked against ``t-prime/data/README.md``'s dataset table): the
single-protocol OTA capture we need is **DS 3.0** ("OTA, single protocol, signals collected in
multiple rooms, 7279 transmissions") -- NOT the ``neu:h989s847q`` collection previously
referenced here, which is the *multi-protocol overlapping-mixture* DS 3.3 (item
``neu:h989s8544``, wrong task: overlap detection, not single-label 4-class classification).
DS 3.0's handle link (``http://hdl.handle.net/2047/D20621423``) resolves (302) to DRS item
``neu:h989s8519``; the direct artifact follows the same ``downloads/<item>?datastream_id=content``
pattern used by the sibling ORACLE dataset from the same lab.

We never redistribute the archive (D3): this only *fetches* the DRS artifact into the local
cache and extracts it; nothing is ever committed. The transfer (``requests``) is imported
LAZILY with a clear ``pip install rfbench[data]`` error, so importing this module stays
dependency-free and it is NEVER exercised in CI (no network, no heavy deps).

LICENSE: the DRS landing page states no explicit redistribution license -- the data is
openly downloadable but its terms are unconfirmed. We fetch it for local use only.

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre. If the
``downloads/`` URL below 404s (DRS item links can be item-specific), fall back to resolving
``DS3_0_HANDLE_URL`` manually and pass the resulting artifact URL as ``source_url=``.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: T-PRIME's own dataset table (ground truth for which DS id maps to which capture mode).
DATA_README = "https://github.com/genesys-neu/t-prime/blob/main/data/README.md"

#: DS 3.0 (single-protocol OTA, multi-room, 7279 transmissions) -- the dataset this module
#: fetches. Resolves (302) to the DRS item below.
DS3_0_HANDLE_URL = "http://hdl.handle.net/2047/D20621423"

#: Official Northeastern DRS item page for DS 3.0 (confirmed via the handle-link redirect).
DRS_COLLECTION_PAGE = "https://repository.library.northeastern.edu/files/neu:h989s8519"

#: T-PRIME code + split scaffolding (raw data is on the DRS, not here).
CODE_REPO = "https://github.com/genesys-neu/t-prime"

#: Direct download URL for the DS 3.0 raw-IQ archive (``DATASET3_0.zip``), following the DRS
#: ``downloads/<item>?datastream_id=content`` pattern confirmed on the lab's ORACLE dataset.
#: Pass ``source_url=`` to override if this 404s (DRS item-download links can rot/change).
_ARCHIVE_URL: str | None = (
    "https://repository.library.northeastern.edu/downloads/neu:h989s8519?datastream_id=content"
)

#: Archive filename fetched from the DRS (nominal; overridden by the URL's basename).
_ARCHIVE_NAME = "tprime_wifi4.tar.gz"

#: Cache subdirectory the extracted T-PRIME captures are written under.
_TPRIME_SUBDIR = "tprime_wifi4"

_INSTALL_HINT = (
    "Downloading the T-PRIME set needs requests; install it with `pip install rfbench[data]`."
)


def download_tprime_wifi4(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
    manual_archive: str | Path | None = None,
) -> Path:
    """Fetch + extract the T-PRIME WiFi archive into ``$RFBENCH_CACHE/tprime_wifi4/``.

    The archive is fetched from ``source_url`` if given, else from :data:`_ARCHIVE_URL` (the
    DS 3.0 DRS item, confirmed above). If the destination already holds extracted files and
    ``force`` is ``False`` the download is skipped (idempotent). Returns the extraction
    directory.

    The DRS host has served an incomplete TLS certificate chain (missing intermediate) in the
    past, which a strict client correctly refuses; downgrading verification here would be a
    silent, code-level TLS weakening applied to every future run. Instead, pass
    ``manual_archive=`` with the path to an archive fetched out-of-band (browser, or a
    deliberate one-off ``curl``/pinned-CA command run by a human) -- it is extracted with the
    same PEP 706 path-traversal guard as the network path, with no code-level trust downgrade.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    dest_dir = resolve_cache_dir(cache) / _TPRIME_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not force and _has_extracted_files(dest_dir):
        return dest_dir

    if manual_archive is not None:
        archive = Path(manual_archive)
        if not archive.is_file():
            raise FileNotFoundError(f"manual_archive {archive} does not exist")
        _extract_archive(archive, dest_dir)
        if not _has_extracted_files(dest_dir):
            raise FileNotFoundError(
                f"extracted {archive} but no files landed under {dest_dir}; check the archive."
            )
        return dest_dir

    url = source_url or _ARCHIVE_URL
    if not url:
        raise ValueError(
            f"no T-PRIME download URL; find the DS 3.0 artifact on {DRS_COLLECTION_PAGE} and "
            "pass its direct URL as source_url=, or fetch it out-of-band and pass "
            "manual_archive=<path> (see this function's docstring)."
        )

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    archive = dest_dir / (Path(url).name or _ARCHIVE_NAME)
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    except requests.exceptions.RequestException as exc:  # network/HTTP failure
        raise RuntimeError(
            f"could not fetch the T-PRIME archive from {url}; obtain it manually from "
            f"{DRS_COLLECTION_PAGE}, place the extracted captures in {dest_dir}, and re-run "
            "(or pass source_url= for a mirror)."
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
    archive_suffixes = {".zip", ".gz", ".tar", ".tgz"}
    return any(
        p.is_file() and p.suffix.lower() not in archive_suffixes for p in dest_dir.rglob("*")
    )


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.zip`` / ``.tar[.gz]`` archive into ``dest_dir`` (stdlib only)."""
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            # PEP 706: reject path-traversal/symlink members even for a "trusted"
            # archive (defense-in-depth if the transfer channel is ever downgraded).
            tf.extractall(dest_dir, filter=getattr(tarfile, "data_filter", None))
    # else: the download was already the raw file; nothing to do.


__all__ = [
    "DATA_README",
    "DS3_0_HANDLE_URL",
    "DRS_COLLECTION_PAGE",
    "CODE_REPO",
    "download_tprime_wifi4",
]
