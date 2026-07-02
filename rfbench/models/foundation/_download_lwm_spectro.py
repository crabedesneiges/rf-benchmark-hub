"""Download the REAL LWM-Spectro public weights into ``$RFBENCH_CACHE`` (WP-62).

LWM-Spectro (``wi-lab/lwm-spectro`` on the Hugging Face Hub) is a **public, non-gated**
RF foundation model: a 12-layer, ``d_model=128`` Transformer pretrained on RF spectrograms,
plus a Mixture-of-Experts head over WiFi / LTE / 5G experts. The pretraining backbone
checkpoint (``checkpoints/checkpoint.pth``) is the weight file this benchmark loads into
:class:`~rfbench.models.foundation.lwm_spectro.LwmSpectroModel`; the MoE / expert files are
fetched too so the full model is available on the cluster if needed.

This module is the **guarded weights fetcher**: heavy deps (``huggingface_hub``) are imported
lazily inside the function, and NOTHING here is called from the unit tests (they run in the
dep-free ``.venv`` and never touch the network). On the cluster ARM GPU venv, run
:func:`download_lwm_spectro` once (or ``python -m rfbench.models.foundation._download_lwm_spectro``)
with ``$RFBENCH_CACHE`` pointing at Lustre; the wrapper then resolves the cached files.

HARD CONSTRAINT: ``import rfbench.models.foundation`` stays dependency-free -- this module is
NOT imported by that package's ``__init__``; ``huggingface_hub`` loads only when
:func:`download_lwm_spectro` actually runs.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Hugging Face repo id of the public LWM-Spectro model (no login / no gating).
HF_REPO_ID = "wi-lab/lwm-spectro"

#: Sub-directory of ``$RFBENCH_CACHE`` the weights land in (kept off the dataset trees).
CACHE_SUBDIR = "lwm-spectro"

#: The pretraining backbone checkpoint the wrapper's ``embed`` loads (12-layer LWM encoder).
BACKBONE_CHECKPOINT = "checkpoints/checkpoint.pth"

#: The Mixture-of-Experts head checkpoint (protocol router + task classifier), fetched for
#: completeness so the full MoE model can be reconstructed on the cluster.
MOE_CHECKPOINT = "moe_checkpoint.pth"

#: The per-protocol expert weights and the hub config -- the remaining artefacts of the model.
EXTRA_FILES: tuple[str, ...] = (
    "config.json",
    "experts/WiFi_expert.pth",
    "experts/LTE_expert.pth",
    "experts/5G_expert.pth",
)

#: Actionable message if ``huggingface_hub`` is missing (the extra that ships it).
_INSTALL_HINT = (
    "downloading LWM-Spectro weights needs `huggingface_hub`; install the optional extra "
    "with `pip install rfbench[foundation]` (or `pip install huggingface_hub` in the "
    "cluster ARM GPU venv)."
)

#: Env var holding the cache root (shared with the data layer's ``resolve_cache_dir``).
CACHE_ENV_VAR = "RFBENCH_CACHE"


def _cache_root(cache: str | Path | None) -> Path:
    """Resolve the cache root: explicit arg, then ``$RFBENCH_CACHE``, then ``./.rfbench_cache``.

    Mirrors :func:`rfbench.data.prepare._common.resolve_cache_dir` (kept local so this module
    imports nothing heavy). No absolute path is hard-coded: on Dalia ``$RFBENCH_CACHE`` points
    at Lustre.
    """
    if cache is not None:
        return Path(cache).expanduser()
    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.cwd() / ".rfbench_cache"


def lwm_spectro_dir(cache: str | Path | None = None) -> Path:
    """Return the ``$RFBENCH_CACHE/lwm-spectro`` directory the weights are cached under."""
    return _cache_root(cache) / CACHE_SUBDIR


def backbone_checkpoint_path(cache: str | Path | None = None) -> Path:
    """Return the expected local path of the pretraining backbone checkpoint (may not exist)."""
    return lwm_spectro_dir(cache) / BACKBONE_CHECKPOINT


def download_lwm_spectro(
    *,
    cache: str | Path | None = None,
    include_experts: bool = True,
    force: bool = False,
) -> Path:
    """Fetch the real LWM-Spectro weights into ``$RFBENCH_CACHE/lwm-spectro`` and return its dir.

    Downloads the pretraining backbone (:data:`BACKBONE_CHECKPOINT`) that the wrapper's
    ``embed`` loads, plus the MoE checkpoint and (when ``include_experts``) the per-protocol
    expert weights + config. Idempotent: ``huggingface_hub`` caches by content hash, so a
    second call is a no-op unless ``force`` re-downloads. Returns the local directory holding
    the files.

    Heavy deps are imported lazily; NEVER called in unit tests (network + weights).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_HINT) from exc

    dest = lwm_spectro_dir(cache)
    dest.mkdir(parents=True, exist_ok=True)

    wanted = [BACKBONE_CHECKPOINT, MOE_CHECKPOINT]
    if include_experts:
        wanted.extend(EXTRA_FILES)

    for filename in wanted:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(dest),
            force_download=force,
        )

    resolved = dest / BACKBONE_CHECKPOINT
    if not resolved.exists():  # pragma: no cover - hub layout regression guard
        raise FileNotFoundError(
            f"fetched {HF_REPO_ID} but the backbone checkpoint {BACKBONE_CHECKPOINT!r} is "
            f"missing under {dest}; check the repo layout at "
            f"https://huggingface.co/{HF_REPO_ID}/tree/main."
        )
    return dest


def main() -> None:  # pragma: no cover - CLI convenience, not unit-tested
    """CLI entry: ``python -m rfbench.models.foundation._download_lwm_spectro``."""
    dest = download_lwm_spectro()
    print(f"LWM-Spectro weights ready under: {dest}")  # noqa: T201 - user-facing CLI output


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "HF_REPO_ID",
    "CACHE_SUBDIR",
    "BACKBONE_CHECKPOINT",
    "MOE_CHECKPOINT",
    "EXTRA_FILES",
    "lwm_spectro_dir",
    "backbone_checkpoint_path",
    "download_lwm_spectro",
]
