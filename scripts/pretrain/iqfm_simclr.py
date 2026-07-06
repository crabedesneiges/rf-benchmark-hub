"""Contrastive (SimCLR / InfoNCE) pre-training of the IQFM ShuffleNetV2-x0.5 raw-IQ backbone.

Reproduces IQFM's *recipe* (Mashaal & Abou-Zeid, arXiv:2506.06718v2) — a small ShuffleNetV2-x0.5
backbone over raw IQ, pre-trained with contrastive SSL and unit-max input normalisation — on the
RadioML 2016.10a **train** split with its **labels discarded** (delabelised). The learned backbone
weights are saved to ``$RFBENCH_CACHE/iqfm/iqfm_shufflenet1d_simclr.pth`` for the
:class:`~rfbench.models.foundation.iqfm.IqfmBase` wrapper to load and probe on AMC.

HONESTY — this is NOT the paper's OOD setting. IQFM's headline (38.1% on RML2016.10a, linear
probe, 50/cls) pre-trains on the authors' OTA MIMO testbed (which we do NOT have) and probes on
RadioML out-of-distribution. Here we pre-train IN-DISTRIBUTION on RadioML-train and probe on
RadioML-test — a different, easier setting — so the resulting score is **ours**, not the paper's.
The wrapper labels the board row accordingly.

Augmentations (SSL positives — a signal-appropriate, deliberately simple set; each documented):
* **circular time shift** — ``roll`` the window by a random offset (invariance to symbol timing);
* **additive Gaussian noise** — SNR-jitter around the sample (invariance to channel noise);
* **global phase rotation** — multiply the complex signal by ``e^{jθ}`` (invariance to carrier
  phase; the receiver has no absolute phase reference).
Two independent augmentations of each window form the positive pair; all other windows in the
batch are negatives (NT-Xent / InfoNCE).

Run (cluster ARM GPU node, never the Intel frontend):
    uv run python scripts/pretrain/iqfm_simclr.py --epochs 100 --batch-size 512 --seed 42
or via ``sbatch slurm/pretrain_iqfm_arm.sh``. Requires the RadioML 2016.10a split prepared first
(``sbatch slurm/download_prepare_arm.sh radioml_2016_10a``).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rfbench.core.registry import get_task
from rfbench.models.foundation.iqfm import backbone_checkpoint_path
from rfbench.models.foundation.shufflenet1d import EMBED_DIM_X0_5, build_shufflenet1d

_LOG = logging.getLogger("iqfm_simclr")

#: Complex-magnitude floor so unit-max normalisation never divides by zero.
_MAX_EPS = 1e-8


def _unit_max(iq: Tensor) -> Tensor:
    """Per-sample unit-max normalisation ``iq / max(|iq|)`` over a ``(2, L)`` window."""
    scale = torch.sqrt(iq[0] ** 2 + iq[1] ** 2).max().clamp_min(_MAX_EPS)
    return iq / scale


class _AmcUnlabelledWindows(torch.utils.data.Dataset[Tensor]):
    """The AMC train split as **unlabelled** unit-max-normalised ``(2, L)`` IQ windows.

    Wraps the task's ``train`` dataset and exposes only the ``"iq"`` field as a ``float32``
    tensor — labels are intentionally dropped for self-supervised pre-training.
    """

    def __init__(self, dataset_name: str = "radioml_2016_10a") -> None:
        task = get_task("amc")
        ds = next(d for d in task.datasets() if d.name == dataset_name)
        self._samples = list(ds.load("train"))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tensor:
        iq = torch.as_tensor(self._samples[idx]["iq"], dtype=torch.float32)
        return _unit_max(iq)


def _batch_unit_max(batch: Tensor) -> Tensor:
    """Per-sample unit-max normalisation ``iq / max(|iq|)`` over a whole ``(B, 2, L)`` batch.

    Vectorised counterpart of :func:`_unit_max`: the per-sample scale is the max complex magnitude
    over each window, so every sample is divided by its own constant with no Python-level loop.
    """
    magnitude = torch.sqrt(batch[:, 0, :] ** 2 + batch[:, 1, :] ** 2)  # (B, L)
    scale = magnitude.amax(dim=1).clamp_min(_MAX_EPS)  # (B,)
    return batch / scale[:, None, None]


def _augment(batch: Tensor, noise_std: float, generator: torch.Generator) -> Tensor:
    """Apply one random SSL augmentation composition to a ``(B, 2, L)`` batch (fully vectorised).

    Composes a circular time shift, additive Gaussian noise, and a global phase rotation (all
    per-sample, sampled from the CPU ``generator`` for reproducibility then moved to ``batch``'s
    device), then re-applies unit-max norm so the augmented view keeps IQFM's input convention.
    No per-sample Python loop: the shift is a batched ``gather`` (``torch.roll`` cannot take a
    per-sample shift), and noise / phase / unit-max are broadcast over the batch.
    """
    b, _, length = batch.shape
    device = batch.device

    # Per-sample circular time shift via gather: out[..., t] = in[..., (t - shift) % L].
    shifts = torch.randint(0, length, (b,), generator=generator).to(device)  # (B,)
    positions = torch.arange(length, device=device)[None, :]  # (1, L)
    idx = ((positions - shifts[:, None]) % length)[:, None, :].expand(b, 2, length)  # (B, 2, L)
    rolled = torch.gather(batch, 2, idx)

    # Additive Gaussian noise.
    noise = torch.randn(batch.shape, generator=generator).to(device) * noise_std
    noisy = rolled + noise

    # Global phase rotation e^{jθ}: [I';Q'] = [[cosθ,-sinθ],[sinθ,cosθ]] [I;Q].
    theta = torch.rand(b, generator=generator).to(device) * (2.0 * math.pi)
    cos, sin = torch.cos(theta)[:, None], torch.sin(theta)[:, None]  # (B, 1)
    i, q = noisy[:, 0, :], noisy[:, 1, :]  # (B, L)
    rotated = torch.stack([i * cos - q * sin, i * sin + q * cos], dim=1)  # (B, 2, L)

    return _batch_unit_max(rotated)


class _ProjectionHead(nn.Module):
    """SimCLR projection MLP: ``embed_dim -> hidden -> proj_dim`` (discarded after pre-training)."""

    def __init__(self, embed_dim: int, hidden: int = 512, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, proj_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def _nt_xent(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    """NT-Xent / InfoNCE loss for a batch of positive pairs ``(z1[i], z2[i])`` (L2-normalised).

    Concatenates the two views into ``2B`` embeddings; for each anchor its positive is the other
    view of the same sample and every other embedding is a negative. Standard SimCLR formulation.
    """
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B, 2B) cosine sims (z is unit-norm)
    sim.fill_diagonal_(float("-inf"))  # mask self-similarity
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, targets)


def train(args: argparse.Namespace) -> Path:
    """Run the SimCLR pre-training loop and save the backbone state_dict; return its path."""
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    dataset = _AmcUnlabelledWindows(args.dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers,
    )
    _LOG.info("pre-training on %d unlabelled %s windows (seed=%d)", len(dataset), args.dataset,
              args.seed)

    backbone = build_shufflenet1d().to(device)
    head = _ProjectionHead(EMBED_DIM_X0_5).to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    backbone.train()
    head.train()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            batch = batch.to(device)
            view1 = _augment(batch, args.noise_std, generator).to(device)
            view2 = _augment(batch, args.noise_std, generator).to(device)
            z1 = head(backbone(view1))
            z2 = head(backbone(view2))
            loss = _nt_xent(z1, z2, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        mean_loss = running / max(len(loader), 1)
        _LOG.info("epoch %3d/%d  nt_xent=%.4f", epoch + 1, args.epochs, mean_loss)

    out = Path(args.out) if args.out else backbone_checkpoint_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backbone_state_dict": backbone.state_dict(),
        "recipe": "simclr_infonce",
        "provenance": "IQFM recipe (arXiv:2506.06718v2); weights unpublished, retrained in-repo",
        "pretrain_dataset": f"{args.dataset} train split (delabelised, in-distribution)",
        "seed": args.seed,
        "epochs": args.epochs,
        "temperature": args.temperature,
    }
    torch.save(payload, out)
    _LOG.info("saved backbone -> %s", out)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (all defaulted; ``seed=42`` per the reproducibility convention)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="radioml_2016_10a")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="checkpoint path (default: $RFBENCH_CACHE/iqfm/...)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging, parse args, run pre-training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(_parse_args(argv))


if __name__ == "__main__":
    main()
