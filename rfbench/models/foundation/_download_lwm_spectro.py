"""Download the REAL LWM-Spectro public weights into ``$RFBENCH_CACHE`` (WP-62).

LWM-Spectro (``wi-lab/lwm-spectro`` on the Hugging Face Hub) is a **public, non-gated**
RF foundation model: a 12-layer, ``d_model=128`` Transformer pretrained on RF spectrograms,
plus a Mixture-of-Experts head over WiFi / LTE / 5G experts. The real 12-layer LWM encoders are the
per-protocol **expert** files (``experts/<proto>_expert.pth``); the wrapper loads one as its frozen
backbone (:class:`~rfbench.models.foundation.lwm_spectro.LwmSpectroModel`). NOTE
``checkpoints/checkpoint.pth`` is the ``snr_mobility`` MoE bundle (router + classifier), NOT the
encoder -- it is fetched for completeness only.

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

#: The per-protocol **expert** files ARE the real 12-layer LWM encoders (verified: each is a
#: 203-tensor state_dict with ``module.embedding.proj``/``layers.i...``/``norm.alpha`` keys). The
#: wrapper's ``embed`` loads ONE of these as the frozen backbone.
EXPERT_CHECKPOINTS: tuple[str, ...] = (
    "experts/WiFi_expert.pth",
    "experts/LTE_expert.pth",
    "experts/5G_expert.pth",
)

#: Default protocol expert whose encoder is used as the frozen backbone. The three experts are
#: fine-tuned from a shared pretraining base (epoch-0 checkpoints), so any is a reasonable generic
#: RF encoder; WiFi is the default. Override via ``LwmSpectroModel(expert=...)``.
DEFAULT_EXPERT = "WiFi"

#: The encoder-backbone checkpoint the wrapper loads by default (a protocol expert, NOT the MoE).
BACKBONE_CHECKPOINT = f"experts/{DEFAULT_EXPERT}_expert.pth"

#: The MoE bundle (``checkpoints/checkpoint.pth``) is the ``snr_mobility`` downstream router +
#: classifier + expert list -- NOT the encoder. Fetched for completeness only; never loaded as the
#: backbone. (``moe_checkpoint.pth`` is a same-shaped duplicate bundle.)
MOE_CHECKPOINT = "checkpoints/checkpoint.pth"

#: The remaining artefacts (top-level MoE bundle + hub config) fetched for completeness.
EXTRA_FILES: tuple[str, ...] = (
    "config.json",
    "moe_checkpoint.pth",
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


def backbone_checkpoint_path(cache: str | Path | None = None, expert: str = DEFAULT_EXPERT) -> Path:
    """Return the local path of the encoder-backbone (a protocol expert; may not exist).

    ``expert`` selects which protocol expert's LWM encoder to load (``WiFi`` / ``LTE`` / ``5G``);
    the default is :data:`DEFAULT_EXPERT`.
    """
    return lwm_spectro_dir(cache) / f"experts/{expert}_expert.pth"


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

    # Always fetch the default backbone expert; `include_experts` adds the other protocols.
    wanted = [MOE_CHECKPOINT, *EXTRA_FILES, BACKBONE_CHECKPOINT]
    if include_experts:
        wanted.extend(e for e in EXPERT_CHECKPOINTS if e != BACKBONE_CHECKPOINT)

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
    "EXPERT_CHECKPOINTS",
    "DEFAULT_EXPERT",
    "lwm_spectro_dir",
    "backbone_checkpoint_path",
    "download_lwm_spectro",
]
