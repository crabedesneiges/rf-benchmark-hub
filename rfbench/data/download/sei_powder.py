"""Download the POWDER RF-fingerprinting (4-BS WiFi) SEI dataset into ``$RFBENCH_CACHE``.

POWDER RF Fingerprinting (Reus-Muns, Jaisinghani, Sankhe, Chowdhury, "Trust in 5G Open RANs
through Machine Learning: RF Fingerprinting on the POWDER PAWR Platform", IEEE GLOBECOM 2020,
pp. 1-6; GENESYS Lab, Northeastern) is the 4-base-station WiFi hardware-fingerprinting set used
as the SEI downstream by BOTH public foundation-model evaluators -- WirelessJEPA (arXiv:2601.20190,
500-shot linear probe 90.5%) and IQFM (arXiv:2506.06718, LoRA 96.05% @ 500/class). It is the
dataset our POWDER board track targets for a like-for-like FM comparison.

The data is **publicly available WITHOUT POWDER/Emulab credentials** -- it lives in the
Northeastern University Digital Repository Service (DRS), reachable via the stable Handle
:data:`POWDER_HANDLE` -> DRS record ``neu:gm80mp276`` ("POWDER-4BS-IQsample"), distributed as
SigMF captures (``.sigmf-data`` + ``.sigmf-meta`` per recording). We never redistribute it (D3);
this only fetches what the user is entitled to into the local cache.

**IMPORTANT -- anti-scraping gate.** The DRS host returns HTTP 403 to programmatic clients
(``curl``/``requests``) and is NOT unblocked merely by spoofing a browser User-Agent (verified:
403 persists with a Chrome UA -- the challenge is JS/cookie/TLS-based). So this is functionally a
**manual-download** step, not a credential wall. :func:`download_powder` attempts the fetch and,
on the expected 403 / HTML response, raises a clear instruction to download it in a real browser
and drop the SigMF tree under ``$RFBENCH_CACHE/powder/`` -- exactly the layout
:func:`rfbench.data.prepare.sei.load_powder_records` reads. Heavy deps (``requests``) are imported
lazily; NEVER exercised in CI.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official GENESYS landing page (download link + dataset description live here).
POWDER_DATASET_PAGE = "https://genesys-lab.org/powder"

#: Stable Northeastern DRS Handle for the POWDER-4BS-IQsample record (resolves 302 -> the DRS file
#: listing ``repository.library.northeastern.edu/files/neu:gm80mp276``).
POWDER_HANDLE = "http://hdl.handle.net/2047/D20385049"

#: Cache subdirectory the extracted POWDER SigMF capture tree is written under.
_POWDER_SUBDIR = "powder"

_INSTALL_HINT = "Downloading POWDER needs requests; install it with `pip install rfbench[data]`."

#: The manual-download instruction raised when the DRS anti-bot gate blocks a programmatic fetch.
_MANUAL_STEPS = (
    "POWDER RF-fingerprinting could not be fetched programmatically: the Northeastern DRS host "
    "anti-scrapes non-browser clients (HTTP 403, not a login wall -- a browser User-Agent does "
    "NOT defeat it). Download it manually:\n"
    f"  1. Open {POWDER_DATASET_PAGE} in a real browser and follow the download link, or go "
    f"straight to the DRS record via the Handle {POWDER_HANDLE}\n"
    "     (resolves to https://repository.library.northeastern.edu/files/neu:gm80mp276, "
    "'POWDER-4BS-IQsample').\n"
    "  2. Download the SigMF captures (.sigmf-data + .sigmf-meta pairs, named "
    "[Waveform]_[Day]_[TransmitterBS]_[RecordingSet], e.g. WiFi_Day1_MEB_1.sigmf-data).\n"
    "  3. Extract them under {dest} so files land as "
    "{dest}/<...>/[Waveform]_[Day]_[TransmitterBS]_[RecordingSet].sigmf-data.\n"
    "  4. Re-run `rfbench data prepare --task sei --dataset powder` to build the split indices.\n"
    "License: unspecified on the GENESYS/DRS pages (cite the GLOBECOM 2020 paper; do not assert "
    "a CC license). We ship only split indices + checksums, never raw IQ (D3)."
)


def download_powder(
    *,
    source_url: str | None = None,
    handle: str | None = None,
    cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch + extract POWDER RF-fingerprinting into ``$RFBENCH_CACHE/powder/`` (best-effort).

    Attempts the DRS Handle (:data:`POWDER_HANDLE`, or a direct ``source_url`` mirror). Because
    the DRS anti-bot gate typically blocks this, the common outcome is a clear
    :class:`RuntimeError` with the exact MANUAL-download procedure (:data:`_MANUAL_STEPS`) and
    the expected on-disk layout. If the capture tree is already present and ``force`` is
    ``False`` the fetch is skipped (idempotent). Returns the POWDER root directory. Heavy deps
    are imported lazily; NEVER called in unit tests.
    """
    root = resolve_cache_dir(cache) / _POWDER_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    if _looks_populated(root) and not force:
        return root

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    url = source_url if source_url is not None else (handle or POWDER_HANDLE)
    # Send a browser-class UA as a courtesy, but do NOT rely on it defeating the gate.
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) rfbench-data-prepare"}
    try:
        with requests.get(
            url, stream=True, timeout=120, allow_redirects=True, headers=headers
        ) as resp:
            if resp.status_code == 403 or "text/html" in str(resp.headers.get("Content-Type", "")):
                raise RuntimeError(_MANUAL_STEPS.format(dest=root))
            resp.raise_for_status()
            archive = root / (Path(source_url).name if source_url else "powder_4bs_iqsample.zip")
            with archive.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    except requests.exceptions.RequestException as exc:  # network/cert/anti-bot failure
        raise RuntimeError(_MANUAL_STEPS.format(dest=root)) from exc

    _extract_archive(archive, root)
    if not _looks_populated(root):
        raise RuntimeError(_MANUAL_STEPS.format(dest=root))
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
            tf.extractall(dest_dir)  # noqa: S202 - trusted dataset archive
    # else: the download was already the raw capture file; nothing to do.


__all__ = [
    "POWDER_DATASET_PAGE",
    "POWDER_HANDLE",
    "download_powder",
]
