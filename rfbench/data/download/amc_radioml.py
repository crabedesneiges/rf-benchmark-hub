"""Download RadioML 2016.10a / 2018.01a from DeepSig into ``$RFBENCH_CACHE``.

DeepSig distributes the RadioML datasets under a CC BY-NC-SA 4.0 license and does NOT
permit redistribution (D3), so these functions only *fetch* the published archive into the
local cache and extract it; nothing is ever committed. The heavy work (``requests`` for the
transfer) is imported LAZILY with a clear ``pip install rfbench[data]`` error, so importing
this module stays dependency-free and it is NEVER exercised in CI (no network, no heavy
deps).

Verified official source (July 2026, https://www.deepsig.ai/datasets/):

* RadioML 2016.10a -> ``RML2016.10a.tar.bz2`` (a bzip2 tarball wrapping the pickle
  ``RML2016.10a_dict.pkl``), from ``https://opendata.deepsig.io/datasets/2016.10/``.
* RadioML 2018.01a -> ``2018.01.OSC.0001_1024x2M.h5.tar.gz`` (a gzip tarball wrapping the
  HDF5 ``GOLD_XYZ_OSC.0001_1024.hdf5``, ~21.5 GB), from
  ``https://opendata.deepsig.io/datasets/2018.01/``.

The ``opendata.deepsig.io`` mirror currently serves the archives directly; DeepSig may move
them back behind a registration/EULA wall at any time. If a fetch fails with an auth error
we do NOT scrape -- we raise a clear error telling the user to obtain the archive manually
and drop it at the expected cache path.

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rfbench.data.prepare._common import resolve_cache_dir

RadioMLDataset = Literal["radioml_2016_10a", "radioml_2018_01a"]

#: Official DeepSig landing page (dataset provenance; links to the open-data mirror).
DEEPSIG_DATASETS_PAGE = "https://www.deepsig.ai/datasets/"

#: Verified per-dataset published archive name on the DeepSig open-data mirror.
_ARCHIVE_NAME: dict[str, str] = {
    "radioml_2016_10a": "RML2016.10a.tar.bz2",
    "radioml_2018_01a": "2018.01.OSC.0001_1024x2M.h5.tar.gz",
}

#: Verified default download URL per dataset (DeepSig open-data mirror, July 2026).
_DEFAULT_URL: dict[str, str] = {
    "radioml_2016_10a": ("https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"),
    "radioml_2018_01a": (
        "https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz"
    ),
}

#: Expected extracted file per dataset (consumed by the loaders in ``prepare/amc.py``).
_EXPECTED_FILE: dict[str, str] = {
    "radioml_2016_10a": "RML2016.10a_dict.pkl",
    "radioml_2018_01a": "GOLD_XYZ_OSC.0001_1024.hdf5",
}

_INSTALL_HINT = "Downloading RadioML needs requests; install it with `pip install rfbench[data]`."


def download_radioml(
    dataset: RadioMLDataset,
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract a RadioML dataset into ``$RFBENCH_CACHE/<dataset>/``.

    ``source_url`` overrides the verified default download URL (see :data:`_DEFAULT_URL`);
    pass it when DeepSig has moved the archive or gated it behind a per-EULA link the
    entitled user obtained from :data:`DEEPSIG_DATASETS_PAGE`. If the expected extracted
    file is already present and ``force`` is ``False`` the download is skipped (idempotent).
    Returns the path to the extracted file.

    If the download fails with an authentication/authorisation error (the archive was moved
    behind a registration wall) we do NOT scrape: a clear error tells the user to download
    the archive manually and place it at the returned path's directory. ``requests`` is
    imported lazily; NEVER called in unit tests.
    """
    if dataset not in _EXPECTED_FILE:
        raise ValueError(f"unknown RadioML dataset {dataset!r}; expected {sorted(_EXPECTED_FILE)}")

    dest_dir = resolve_cache_dir(cache) / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / _EXPECTED_FILE[dataset]
    if extracted.exists() and not force:
        return extracted

    url = source_url or _DEFAULT_URL[dataset]

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    archive = dest_dir / _ARCHIVE_NAME[dataset]
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code in (401, 403):
                raise RuntimeError(_gated_message(dataset, url, dest_dir))
            resp.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.exceptions.RequestException as exc:  # network/HTTP failure
        raise RuntimeError(_gated_message(dataset, url, dest_dir)) from exc

    _extract_archive(archive, dest_dir)
    if not extracted.exists():
        raise FileNotFoundError(
            f"extracted {archive.name} but expected {extracted.name} was not produced "
            f"under {dest_dir}; the archive layout may have changed."
        )
    return extracted


def _gated_message(dataset: str, url: str, dest_dir: Path) -> str:
    """Manual-download instructions when the automated fetch is blocked (no scraping)."""
    return (
        f"could not fetch RadioML {dataset!r} from {url} (source unreachable or gated behind "
        f"DeepSig registration). Do NOT scrape: obtain {_ARCHIVE_NAME[dataset]!r} manually "
        f"from {DEEPSIG_DATASETS_PAGE}, place it in {dest_dir}, and re-run "
        f"(or pass source_url= for a per-EULA link). Expected extracted file: "
        f"{dest_dir / _EXPECTED_FILE[dataset]}"
    )


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.tar[.bz2/.gz]`` / ``.zip`` archive into ``dest_dir`` (stdlib only)."""
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)  # noqa: S202 - trusted per-EULA dataset archive
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)  # noqa: S202 - trusted per-EULA dataset archive
    # else: the download was already the raw file (e.g. a bare .pkl/.hdf5); nothing to do.


__all__ = [
    "RadioMLDataset",
    "DEEPSIG_DATASETS_PAGE",
    "download_radioml",
]
