"""Generate the WBSig53 wideband-detection dataset via TorchSig into ``$RFBENCH_CACHE``.

WBSig53 is not a downloadable archive: like Sig53 it is *generated* from TorchSig's
official wideband generator into a local root (impaired train/val), each sample being a
wideband capture annotated with time-frequency signal boxes. We never redistribute it
(D3) -- this only drives the generator into the cache. ``torchsig`` (and its
``torch``/``numpy`` stack) is imported LAZILY with a clear ``pip install
rfbench[detection]`` error, so importing this module stays dependency-free and it is NEVER
exercised in CI (heavy deps, hours of generation).

On the cluster: run inside the ARM venv; generation is long and disk-heavy -- point
``$RFBENCH_CACHE`` at Lustre. The lazy loader :func:`load_wbsig53_annotations` extracts the
per-sample T-F boxes from the generated root and hands them to
:func:`rfbench.data.prepare.detection.prepare_detection`.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official TorchSig repository (WBSig53 wideband generator + any canonical split live here).
TORCHSIG_REPO = "https://github.com/TorchDSP/torchsig"

#: Cache subdirectory the generated WBSig53 root is written under.
_WBSIG53_SUBDIR = "wbsig53"

_INSTALL_HINT = (
    "Generating/reading WBSig53 needs TorchSig (torch + numpy); "
    "install it with `pip install rfbench[detection]`."
)


def generate_wbsig53(
    *,
    cache: str | Path | None = None,
    impaired: bool = True,
    force: bool = False,
) -> Path:
    """Generate the WBSig53 dataset via TorchSig into ``$RFBENCH_CACHE/wbsig53/``.

    ``impaired`` selects the impaired variant used by the wideband-detection benchmark
    (the version the DETR/YOLO baselines are reported on). If the root already exists and
    ``force`` is ``False`` generation is skipped (idempotent). Returns the WBSig53 root
    path.

    ``torchsig`` is imported lazily; NEVER called in unit tests (heavy deps + long runtime).
    """
    root = resolve_cache_dir(cache) / _WBSIG53_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    if _looks_generated(root) and not force:
        return root

    try:
        from torchsig.datasets.wideband_sig53 import WidebandSig53
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    # Drive TorchSig's wideband generator for both the train and val partitions of the
    # official split; the exact generator kwargs are pinned on the cluster against the
    # installed torchsig version. Kept minimal here to avoid coupling to a specific API
    # revision.
    for train in (True, False):
        WidebandSig53(
            root=str(root),
            train=train,
            impaired=impaired,
            regenerate=force,
        )
    return root


def load_wbsig53_annotations(
    cache: str | Path | None = None,
) -> list[dict[str, object]]:
    """Extract per-sample time-frequency box annotations from a generated WBSig53 root.

    Each returned element describes one wideband capture as a mapping with a stable
    ``sample_id`` and a ``boxes`` list; every box carries a signal ``class`` plus its
    time/frequency extent ``(t_start, t_stop, f_low, f_high)`` in normalised
    ``[0, 1]`` coordinates -- exactly the shape
    :func:`rfbench.data.prepare.detection.prepare_detection` consumes as ``samples=``.

    ``torchsig`` / ``numpy`` are imported lazily. NEVER called in unit tests (needs a
    generated WBSig53 root + heavy deps).
    """
    resolve_cache_dir(cache)  # validate cache resolution eagerly; TorchSig reads real files
    try:
        import torchsig  # noqa: F401 - imported to surface the clear install error early
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    raise NotImplementedError(
        "WBSig53 annotation extraction runs on the cluster against a generated TorchSig "
        "wideband dataset; wire it to the concrete torchsig WidebandSig53 SignalMetadata "
        "layout there (map each SignalDescription to a normalised t/f box)."
    )


def _looks_generated(root: Path) -> bool:
    """Cheap idempotency check: a non-empty WBSig53 root is treated as already generated."""
    return root.is_dir() and any(root.iterdir())


__all__ = [
    "TORCHSIG_REPO",
    "generate_wbsig53",
    "load_wbsig53_annotations",
]
