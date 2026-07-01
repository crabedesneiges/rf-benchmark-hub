"""Download the WiSig (ManyTx) SEI dataset into ``$RFBENCH_CACHE``.

WiSig is distributed by the UCLA CoRes lab as a set of compressed pickles (per-transmitter
capture blocks across many receivers and days). We never redistribute it (D3) -- this only
*fetches* the archive the user is entitled to into the local cache and extracts it; nothing
is ever committed. The transfer (``requests``) and any post-extract sanity check
(``numpy``) are imported LAZILY with a clear ``pip install rfbench[data]`` error, so
importing this module stays dependency-free and it is NEVER exercised in CI (no network, no
heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official WiSig landing page (dataset description + download links live here).
WISIG_DATASET_PAGE = "https://cores.ee.ucla.edu/downloads/datasets/wisig/"

#: Cache subdirectory the extracted WiSig files are written under.
_WISIG_SUBDIR = "wisig"

#: Expected extracted file (the flattened ManyTx capture pickle read by the loader).
_EXPECTED_FILE = "ManyTx.pkl"

_INSTALL_HINT = (
    "Downloading/verifying WiSig needs requests + numpy; "
    "install them with `pip install rfbench[data]`."
)


def download_wisig(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract WiSig (ManyTx) into ``$RFBENCH_CACHE/wisig/``.

    ``source_url`` is the archive URL obtained from :data:`WISIG_DATASET_PAGE`; we never
    embed or redistribute it. If the expected extracted file is already present and
    ``force`` is ``False`` the download is skipped (idempotent). Returns the path to the
    extracted ``ManyTx.pkl``.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    dest_dir = resolve_cache_dir(cache) / _WISIG_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / _EXPECTED_FILE
    if extracted.exists() and not force:
        return extracted

    if source_url is None:
        raise ValueError(
            f"WiSig must be downloaded from {WISIG_DATASET_PAGE}; obtain the archive URL "
            "there and pass it as `source_url=`."
        )

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    archive = dest_dir / Path(source_url).name
    with requests.get(source_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with archive.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    _extract_archive(archive, dest_dir)
    if not extracted.exists():
        raise FileNotFoundError(
            f"extracted {archive.name} but expected {_EXPECTED_FILE} was not produced "
            f"under {dest_dir}"
        )
    return extracted


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.tar[.gz]`` / ``.zip`` archive into ``dest_dir`` (stdlib only)."""
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    # else: the download was already the raw pickle; nothing to do.


__all__ = [
    "WISIG_DATASET_PAGE",
    "download_wisig",
]
