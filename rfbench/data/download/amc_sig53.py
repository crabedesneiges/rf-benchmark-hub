"""Sig53 AMC dataset acquisition -- BLOCKED: no static published release exists.

Sig53 (Boegner et al., "Large Scale Radio Frequency Signal Classification", 2022) is the
53-class AMC dataset from the TorchSig project. Unlike RadioML, DeepSig/TorchSig do NOT
publish a static, statically-downloadable Sig53 artifact: the dataset is *generated* locally
from the TorchSig generator (verified July 2026 against the TorchSig repo, its docs at
``torchsig.com/dist/downloads.html``, and Papers with Code -- all point to local generation,
none to a Zenodo/HuggingFace/S3 download).

Per the project's data policy, we do NOT fabricate a dataset by generation here: RF-Benchmark
uses the REAL published artifacts used by the reference papers, and generation-only datasets
with no static release are treated as a **blocker** rather than silently synthesised. This
module therefore does not download or generate anything -- it raises a clear, actionable
error so the AMC Sig53 track stays disabled until a static release is confirmed (or the team
explicitly opts in to on-cluster TorchSig generation as a separate, reviewed step).

``import`` of this module stays dependency-free; it never touches torch/torchsig/numpy.
"""

from __future__ import annotations

from pathlib import Path

from rfbench.data.prepare._common import resolve_cache_dir

#: Official TorchSig repository (Sig53 generator + canonical split live here).
TORCHSIG_REPO = "https://github.com/TorchDSP/torchsig"

#: TorchSig dataset documentation page consulted for a static release (none offered).
TORCHSIG_DOWNLOADS_PAGE = "https://torchsig.com/dist/downloads.html"

#: Reference paper introducing Sig53 (arXiv:2207.09918).
SIG53_PAPER = "https://arxiv.org/abs/2207.09918"

#: Cache subdirectory a Sig53 root would live under (created for the manual/opt-in path).
_SIG53_SUBDIR = "sig53"

_BLOCKER_MESSAGE = (
    "Sig53 has NO static published download: it is generation-only via TorchSig "
    f"({TORCHSIG_REPO}; see {TORCHSIG_DOWNLOADS_PAGE}). RF-Benchmark does not synthesise "
    "datasets in place of a real published artifact, so the Sig53 AMC track is BLOCKED "
    "pending a static release (Zenodo/HuggingFace/S3) or an explicit, separately-reviewed "
    "decision to run TorchSig generation on the cluster. To opt in manually: generate Sig53 "
    "with TorchSig on an ARM compute node, place the generated root at "
    "{root}, then wire load_sig53_official_split (rfbench.data.prepare.amc) to that layout. "
    f"Paper: {SIG53_PAPER}."
)


def download_sig53(*, cache: str | Path | None = None) -> Path:
    """Report the Sig53 acquisition blocker (no static release; generation-only).

    Sig53 cannot be fetched as a published artifact, and per policy we do NOT generate it
    here in lieu of a real download. This always raises :class:`NotImplementedError` with
    actionable manual-generation instructions and the expected on-disk root path; it never
    imports torch/torchsig or writes any data.
    """
    root = resolve_cache_dir(cache) / _SIG53_SUBDIR
    raise NotImplementedError(_BLOCKER_MESSAGE.format(root=root))


__all__ = [
    "TORCHSIG_REPO",
    "TORCHSIG_DOWNLOADS_PAGE",
    "SIG53_PAPER",
    "download_sig53",
]
