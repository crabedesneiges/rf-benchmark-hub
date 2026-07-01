"""Download the LoRa RFFI SEI dataset into ``$RFBENCH_CACHE``.

LoRa RFFI (Shen, Zhang & Marshall, IEEE JSAC 2021; dataset ``LoRa_RFFI_dataset``, DOI
10.21227/qqt4-kz19) is distributed on IEEE DataPort as a single ``LoRa_RFFI.zip``
(~15.5 GB) that extracts to HDF5 files, notably ``Train/dataset_training_aug.h5`` (the
``gxhen/LoRa_RFFI`` closed-set layout: 30 training LoRa devices). IEEE DataPort gates
"open-access" files behind a **free IEEE account login**, so we do NOT
scrape it: this only *fetches* an archive the entitled user has already obtained a direct
link for into the local cache and extracts it; nothing is ever committed (D3). The transfer
(``requests``) is imported LAZILY with a clear ``pip install rfbench[data]`` error, so
importing this module stays dependency-free and it is NEVER exercised in CI (no network, no
heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official LoRa RFFI landing page on IEEE DataPort (login-gated download).
LORA_DATASET_PAGE = "https://ieee-dataport.org/open-access/lorarffidataset"

#: IEEE DataPort DOI for the LoRa RFFI dataset (provenance). ``qqt4-kz19`` is the canonical
#: ``LoRa_RFFI_dataset`` used by the closed-set reference code; the newer multiple-receiver
#: variant is ``10.21227/d6vx-r538``.
LORA_DOI = "10.21227/qqt4-kz19"

#: Reference GitHub repo describing the HDF5 layout (data/label datasets).
LORA_CODE_REPO = "https://github.com/gxhen/LoRa_RFFI"

#: Cache subdirectory the extracted LoRa RFFI files are written under.
_LORA_SUBDIR = "lora"

#: Expected extracted training file (the HDF5 read by the loader).
_EXPECTED_FILE = "dataset_training_aug.h5"

_INSTALL_HINT = "Downloading LoRa RFFI needs requests; install it with `pip install rfbench[data]`."


def download_lora(
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract LoRa RFFI into ``$RFBENCH_CACHE/lora/``.

    IEEE DataPort gates the file behind a free IEEE-account login; we never scrape it or
    embed credentials. ``source_url`` is the direct archive URL the entitled user obtained
    from :data:`LORA_DATASET_PAGE` while logged in. If the expected extracted file is
    already present and ``force`` is ``False`` the download is skipped (idempotent).
    Returns the path to the extracted ``dataset_training_aug.h5``.

    Raises a clear, actionable error (with the expected on-disk path/filename) when no
    ``source_url`` is given, since the login wall makes non-interactive fetching
    impossible. Heavy deps are imported lazily; NEVER called in unit tests.
    """
    dest_dir = resolve_cache_dir(cache) / _LORA_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / _EXPECTED_FILE
    if extracted.exists() and not force:
        return extracted

    if source_url is None:
        raise RuntimeError(
            f"LoRa RFFI is gated behind a free IEEE-account login on IEEE DataPort "
            f"({LORA_DATASET_PAGE}, DOI {LORA_DOI}); it cannot be fetched "
            "non-interactively. Log in, download LoRa_RFFI.zip manually, and either pass "
            f"its direct link as source_url= or extract it so that {extracted} exists "
            f"(the HDF5 'data'/'label' layout is documented at {LORA_CODE_REPO})."
        )

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    archive = dest_dir / Path(source_url).name
    with requests.get(source_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with archive.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)

    _extract_archive(archive, dest_dir)
    if not extracted.exists():
        # The zip nests the HDF5 under Train/; surface the first .h5 we can find so the
        # loader's fixed path works, otherwise fail loudly with the expected name.
        found = next(dest_dir.rglob(_EXPECTED_FILE), None)
        if found is not None and found != extracted:
            found.replace(extracted)
        if not extracted.exists():
            raise FileNotFoundError(
                f"extracted {archive.name} but expected {_EXPECTED_FILE} was not produced "
                f"under {dest_dir}; check the archive layout ({LORA_CODE_REPO})."
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
    # else: the download was already the raw HDF5 file; nothing to do.


__all__ = [
    "LORA_DATASET_PAGE",
    "LORA_DOI",
    "LORA_CODE_REPO",
    "download_lora",
]
