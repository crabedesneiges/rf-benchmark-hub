"""Generate the Sig53 AMC dataset via TorchSig into ``$RFBENCH_CACHE``.

Sig53 is not a downloadable archive: it is *generated* from TorchSig's official generator
into a local root (impaired train/val). We never redistribute it (D3) -- this only drives
the generator into the cache. ``torchsig`` (and its ``torch``/``numpy`` stack) is imported
LAZILY with a clear ``pip install rfbench[detection]`` error, so importing this module
stays dependency-free and it is NEVER exercised in CI (heavy deps, hours of generation).

On the cluster: run inside the ARM venv, GPU not required for generation but the process
is long and disk-heavy -- point ``$RFBENCH_CACHE`` at Lustre.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official TorchSig repository (Sig53 generator + canonical split live here).
TORCHSIG_REPO = "https://github.com/TorchDSP/torchsig"

#: Cache subdirectory the generated Sig53 root is written under.
_SIG53_SUBDIR = "sig53"

_INSTALL_HINT = (
    "Generating Sig53 needs TorchSig (torch + numpy); "
    "install it with `pip install rfbench[detection]`."
)


def generate_sig53(
    *,
    cache: str | Path | None = None,
    impaired: bool = True,
    force: bool = False,
) -> Path:
    """Generate the Sig53 dataset via TorchSig into ``$RFBENCH_CACHE/sig53/``.

    ``impaired`` selects the impaired variant used by the AMC benchmark (the version the
    official split and the XCiT baseline are reported on). If the root already exists and
    ``force`` is ``False`` generation is skipped (idempotent). Returns the Sig53 root path.

    ``torchsig`` is imported lazily; NEVER called in unit tests (heavy deps + long runtime).
    """
    root = resolve_cache_dir(cache) / _SIG53_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    if _looks_generated(root) and not force:
        return root

    try:
        from torchsig.datasets.sig53 import Sig53
    except ModuleNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    # Drive TorchSig's generator for both the train and val partitions of the official
    # split; the exact generator kwargs are pinned on the cluster against the installed
    # torchsig version. Kept minimal here to avoid coupling to a specific API revision.
    for train in (True, False):
        Sig53(
            root=str(root),
            train=train,
            impaired=impaired,
            regenerate=force,
            use_signal_data=True,
        )
    return root


def _looks_generated(root: Path) -> bool:
    """Cheap idempotency check: a non-empty Sig53 root is treated as already generated."""
    return root.is_dir() and any(root.iterdir())


__all__ = [
    "TORCHSIG_REPO",
    "generate_sig53",
]
