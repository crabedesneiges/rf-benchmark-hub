"""Download the ORACLE (16-tx) SEI dataset into ``$RFBENCH_CACHE``.

ORACLE (Sankhe et al., Genesys Lab, Northeastern) is a 16-transmitter RF-fingerprinting
dataset captured on a fixed receiver, distributed as SigMF captures via the Northeastern
digital-repository handle service (persistent ``hdl.handle.net/2047/...`` identifiers on
:data:`ORACLE_DATASET_PAGE`). We never redistribute it (D3) -- this only *fetches* the
archive the user is entitled to into the local cache and extracts it into the
per-distance capture tree (``<dist>ft/WiFi_air_X310_<serial>_<dist>ft_run<n>.sigmf-data``
+ ``.sigmf-meta``); nothing is ever committed. The transfer (``requests``) is imported
LAZILY with a clear ``pip install rfbench[data]`` error, so importing this module stays
dependency-free and it is NEVER exercised in CI (no network, no heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official ORACLE landing page (dataset description + handle download links live here).
ORACLE_DATASET_PAGE = "https://www.genesys-lab.org/oracle"

#: Northeastern repository handle for the raw-IQ release (resolves to the archive).
ORACLE_RAW_IQ_HANDLE = "http://hdl.handle.net/2047/D20324547"

#: Northeastern repository handle for the demodulated-IQ release.
ORACLE_DEMOD_IQ_HANDLE = "http://hdl.handle.net/2047/D20324548"

#: Cache subdirectory the extracted ORACLE capture tree is written under.
_ORACLE_SUBDIR = "oracle"

_INSTALL_HINT = "Downloading ORACLE needs requests; install it with `pip install rfbench[data]`."


def download_oracle(
    *,
    source_url: str | None = None,
    handle: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract ORACLE (16-tx) into ``$RFBENCH_CACHE/oracle/``.

    By default the raw-IQ release is resolved from its Northeastern handle
    (:data:`ORACLE_RAW_IQ_HANDLE`, obtained from :data:`ORACLE_DATASET_PAGE`); pass a
    direct ``source_url`` (a mirror/archive link) or a different ``handle`` (e.g.
    :data:`ORACLE_DEMOD_IQ_HANDLE`) to override. If the capture tree is already present and
    ``force`` is ``False`` the download is skipped (idempotent). Returns the ORACLE root
    directory.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    root = resolve_cache_dir(cache) / _ORACLE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    if _looks_populated(root) and not force:
        return root

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    url = source_url if source_url is not None else (handle or ORACLE_RAW_IQ_HANDLE)
    # The handle service redirects to the repository's actual download; follow it. If the
    # repository fronts the file behind an interactive accept/login page we cannot scrape
    # it -- surface a clear manual-download instruction with the expected on-disk layout.
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        content_type = str(resp.headers.get("Content-Type", ""))
        if "text/html" in content_type:
            raise RuntimeError(
                "ORACLE handle resolved to an HTML page, not a downloadable archive -- the "
                "Northeastern repository likely fronts the file behind an interactive "
                f"accept/login step. Download it manually from {ORACLE_DATASET_PAGE} "
                f"(handle {url}) and extract the SigMF capture tree under {root} so that "
                "files land as <dist>ft/WiFi_air_X310_<serial>_<dist>ft_run<n>.sigmf-data."
            )
        archive = root / (Path(source_url).name if source_url else "oracle_raw_iq.zip")
        with archive.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)

    _extract_archive(archive, root)
    if not _looks_populated(root):
        raise FileNotFoundError(
            f"extracted {archive.name} but ORACLE root {root} still looks empty; expected "
            "a SigMF capture tree of .sigmf-data / .sigmf-meta files."
        )
    return root


def _looks_populated(root: Path) -> bool:
    """Idempotency check: a root holding any ``.sigmf-data`` capture is already fetched."""
    return root.is_dir() and next(root.rglob("*.sigmf-data"), None) is not None


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
    # else: the download was already the raw capture file; nothing to do.


__all__ = [
    "ORACLE_DATASET_PAGE",
    "ORACLE_RAW_IQ_HANDLE",
    "ORACLE_DEMOD_IQ_HANDLE",
    "download_oracle",
]
