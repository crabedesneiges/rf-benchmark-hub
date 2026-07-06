"""Download the WiSig (ManyTx) SEI dataset into ``$RFBENCH_CACHE``.

WiSig (Hanna, Karunaratne & Cabric, IEEE Access 2022, doi 10.1109/ACCESS.2022.3154790) is
distributed by the UCLA CoRes lab as compact, pre-packaged pickles hosted on Google Drive;
the ``ManyTx`` subset is a single ~2.5 GB zip that extracts to a ``ManyTx.pkl`` compact
capture tensor. We never redistribute it (D3, CC BY-NC-SA 4.0) -- this only *fetches* the
archive the user is entitled to into the local cache and extracts it; nothing is ever
committed. The transfer (``requests``) is imported LAZILY with a clear
``pip install rfbench[data]`` error, so importing this module stays dependency-free and it
is NEVER exercised in CI (no network, no heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official WiSig landing page (dataset description + Google Drive download links live here).
WISIG_DATASET_PAGE = "https://cores.ee.ucla.edu/downloads/datasets/wisig/"

#: Google Drive file id of the ManyTx compact subset (~2.5 GB zip), taken from the WiSig
#: download page. Kept as the id (not a full URL) so the confirm-token flow can be applied.
MANYTX_GDRIVE_ID = "17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l"

#: Cache subdirectory the extracted WiSig files are written under.
_WISIG_SUBDIR = "wisig"

#: Expected extracted file (the flattened ManyTx capture pickle read by the loader).
_EXPECTED_FILE = "ManyTx.pkl"

#: Google Drive direct-download endpoint (large files need a confirm token, handled below).
_GDRIVE_URL = "https://drive.google.com/uc?export=download"

_INSTALL_HINT = "Downloading WiSig needs requests; install it with `pip install rfbench[data]`."


def download_wisig(
    *,
    source_url: str | None = None,
    gdrive_id: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract WiSig (ManyTx) into ``$RFBENCH_CACHE/wisig/``.

    By default the ManyTx subset is fetched from its canonical Google Drive id
    (:data:`MANYTX_GDRIVE_ID`, obtained from :data:`WISIG_DATASET_PAGE`); pass a mirror
    ``source_url`` (a direct archive/pickle URL) or a different ``gdrive_id`` to override.
    If the expected extracted file is already present and ``force`` is ``False`` the
    download is skipped (idempotent). Returns the path to the extracted ``ManyTx.pkl``.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    dest_dir = resolve_cache_dir(cache) / _WISIG_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / _EXPECTED_FILE
    if extracted.exists() and not force:
        return extracted

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    if source_url is not None:
        archive = dest_dir / Path(source_url).name
        _download_direct(requests, source_url, archive)
    else:
        file_id = gdrive_id or MANYTX_GDRIVE_ID
        archive = dest_dir / "ManyTx.zip"
        _download_gdrive(requests, file_id, archive)

    _extract_archive(archive, dest_dir)
    if not extracted.exists():
        raise FileNotFoundError(
            f"fetched {archive.name} but expected {_EXPECTED_FILE} was not produced under "
            f"{dest_dir}; if the compact pickle uses a different name, pass source_url= to "
            f"a direct link or rename it to {_EXPECTED_FILE}."
        )
    return extracted


def _download_direct(requests_mod: object, url: str, dest: Path) -> None:
    """Stream a plain HTTP(S) URL to ``dest`` (chunked, no confirm token)."""
    with requests_mod.get(url, stream=True, timeout=60) as resp:  # type: ignore[attr-defined]
        resp.raise_for_status()
        _write_stream(resp, dest)


def _download_gdrive(requests_mod: object, file_id: str, dest: Path) -> None:
    """Stream a large Google Drive file to ``dest``, handling its confirm-token gate.

    Google Drive interrupts large downloads with a virus-scan warning; the confirm token
    is carried in a ``download_warning`` cookie (older flow) or as a ``confirm=t`` query
    param (newer flow). We honour both so the transfer completes non-interactively.
    """
    session = requests_mod.Session()  # type: ignore[attr-defined]
    params = {"id": file_id, "export": "download"}
    resp = session.get(_GDRIVE_URL, params=params, stream=True, timeout=60)
    resp.raise_for_status()
    token = _gdrive_confirm_token(resp)
    if token is not None:
        params = {"id": file_id, "export": "download", "confirm": token}
        resp = session.get(_GDRIVE_URL, params=params, stream=True, timeout=60)
        resp.raise_for_status()
    _write_stream(resp, dest)


def _gdrive_confirm_token(resp: object) -> str | None:
    """Extract Google Drive's large-file confirm token from cookies (or ``None``)."""
    cookies = getattr(resp, "cookies", {})
    for key, value in cookies.items():
        if key.startswith("download_warning"):
            return str(value)
    return "t"  # newer flow: a static confirm value is accepted for public files


def _write_stream(resp: object, dest: Path) -> None:
    """Write a streaming response body to ``dest`` in 1 MiB chunks."""
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # type: ignore[attr-defined]
            if chunk:
                fh.write(chunk)


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.tar[.gz]`` / ``.zip`` archive into ``dest_dir`` (stdlib only)."""
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
    # else: the download was already the raw pickle; nothing to do.


__all__ = [
    "WISIG_DATASET_PAGE",
    "MANYTX_GDRIVE_ID",
    "download_wisig",
]
