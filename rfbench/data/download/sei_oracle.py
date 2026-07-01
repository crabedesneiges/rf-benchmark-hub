"""Download the ORACLE (16-tx) SEI dataset into ``$RFBENCH_CACHE``.

ORACLE (Genesys Lab, Northeastern) is a 16-transmitter RF-fingerprinting dataset captured
on a fixed receiver, distributed as per-device capture archives. We never redistribute it
(D3) -- this only *fetches* the archive the user is entitled to into the local cache and
extracts it; nothing is ever committed. The transfer (``requests``) and any post-extract
sanity check (``numpy``) are imported LAZILY with a clear ``pip install rfbench[data]``
error, so importing this module stays dependency-free and it is NEVER exercised in CI (no
network, no heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official ORACLE landing page (dataset description + download links live here).
ORACLE_DATASET_PAGE = "https://www.genesys-lab.org/oracle"

#: Cache subdirectory the extracted ORACLE capture tree is written under.
_ORACLE_SUBDIR = "oracle"

_INSTALL_HINT = (
    "Downloading/verifying ORACLE needs requests + numpy; "
    "install them with `pip install rfbench[data]`."
)


def download_oracle(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract ORACLE (16-tx) into ``$RFBENCH_CACHE/oracle/``.

    ``source_url`` is the archive URL obtained from :data:`ORACLE_DATASET_PAGE`; we never
    embed or redistribute it. If the capture tree is already present and ``force`` is
    ``False`` the download is skipped (idempotent). Returns the ORACLE root directory.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    root = resolve_cache_dir(cache) / _ORACLE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    if _looks_populated(root) and not force:
        return root

    if source_url is None:
        raise ValueError(
            f"ORACLE must be downloaded from {ORACLE_DATASET_PAGE}; obtain the archive URL "
            "there and pass it as `source_url=`."
        )

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    archive = root / Path(source_url).name
    with requests.get(source_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with archive.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    _extract_archive(archive, root)
    if not _looks_populated(root):
        raise FileNotFoundError(
            f"extracted {archive.name} but ORACLE root {root} still looks empty"
        )
    return root


def _looks_populated(root: Path) -> bool:
    """Cheap idempotency check: a non-empty ORACLE root is treated as already fetched."""
    return root.is_dir() and any(p for p in root.iterdir() if p.suffix != ".zip")


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
    # else: the download was already the raw capture file; nothing to do.


__all__ = [
    "ORACLE_DATASET_PAGE",
    "download_oracle",
]
