"""Download RadioML 2016.10a / 2018.01a from DeepSig into ``$RFBENCH_CACHE``.

DeepSig distributes RadioML behind a (free) registration/EULA wall and does NOT permit
redistribution (D3), so these functions only *fetch* an archive the user is entitled to
into the local cache and extract it; nothing is ever committed. The heavy work
(``requests`` for the transfer, ``numpy``/``h5py`` for a post-extract sanity check) is
imported LAZILY with a clear ``pip install rfbench[data]`` error, so importing this module
stays dependency-free and it is NEVER exercised in CI (no network, no heavy deps).

On the cluster: run inside the ARM venv, with ``$RFBENCH_CACHE`` pointing at Lustre.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rfbench.data.prepare._common import resolve_cache_dir

RadioMLDataset = Literal["radioml_2016_10a", "radioml_2018_01a"]

#: Official DeepSig landing page (registration required; direct URLs are gated per-EULA).
DEEPSIG_DATASETS_PAGE = "https://www.deepsig.ai/datasets/"

#: Expected extracted file per dataset (used by the loaders in ``prepare/amc.py``).
_EXPECTED_FILE: dict[str, str] = {
    "radioml_2016_10a": "RML2016.10a_dict.pkl",
    "radioml_2018_01a": "GOLD_XYZ_OSC.0001_1024.hdf5",
}

_INSTALL_HINT = (
    "Downloading/verifying RadioML needs requests + numpy/h5py; "
    "install them with `pip install rfbench[data]`."
)


def download_radioml(
    dataset: RadioMLDataset,
    *,
    source_url: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract a RadioML dataset into ``$RFBENCH_CACHE/<dataset>/``.

    ``source_url`` is the per-EULA archive URL the entitled user obtained from
    :data:`DEEPSIG_DATASETS_PAGE` (DeepSig gates the direct link; we never embed or
    redistribute it). If the expected extracted file is already present and ``force`` is
    ``False`` the download is skipped (idempotent). Returns the path to the extracted file.

    Heavy deps are imported lazily; NEVER called in unit tests.
    """
    if dataset not in _EXPECTED_FILE:
        raise ValueError(f"unknown RadioML dataset {dataset!r}; expected {sorted(_EXPECTED_FILE)}")

    dest_dir = resolve_cache_dir(cache) / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = dest_dir / _EXPECTED_FILE[dataset]
    if extracted.exists() and not force:
        return extracted

    if source_url is None:
        raise ValueError(
            f"{dataset!r} is gated behind DeepSig registration ({DEEPSIG_DATASETS_PAGE}); "
            "obtain the archive URL under its EULA and pass it as `source_url=`."
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
            f"extracted {archive.name} but expected {extracted.name} was not produced "
            f"under {dest_dir}"
        )
    return extracted


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.tar[.gz]`` / ``.zip`` archive into ``dest_dir`` (stdlib only)."""
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
