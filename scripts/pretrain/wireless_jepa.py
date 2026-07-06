"""JEPA (masked-latent prediction) pre-training of the WirelessJEPA ShuffleNetV2-x0.5 backbone.

Reproduces WirelessJEPA's *recipe* (arXiv:2601.20190) — the **same** ShuffleNetV2-x0.5 raw-IQ
backbone as IQFM, pre-trained by predicting the latent of an unmasked view from a **masked** view,
with an **EMA teacher** (momentum 0.996 → 1.0) and **no data augmentation** (the masking is the
whole self-supervised signal) — on the RadioML 2016.10a **train** split with labels discarded. The
learned **EMA target encoder** weights are saved to
``$RFBENCH_CACHE/wireless-jepa/wireless_jepa_shufflenet1d.pth`` for the
:class:`~rfbench.models.foundation.wireless_jepa.WirelessJepa` wrapper to load and probe on AMC.

HONESTY — this is NOT the paper's OOD setting. WirelessJEPA's headline (74.78% on RML2016.10a,
linear probe, 500-shot) pre-trains on the authors' OTA MIMO testbed (which we do NOT have) and
probes on RadioML out-of-distribution. Here we pre-train IN-DISTRIBUTION on RadioML-train and probe
on RadioML-test — a different, easier setting — so the resulting score is **ours**, not the
paper's, and must never be presented as the 74.78% figure.

JEPA design (a DOCUMENTED approximation — the exact masking recipe is unpublished, weights aren't
released):
* **context view** = the unit-max window with a random contiguous time block **zero-masked**
  (mask ratio ``--mask-ratio``); **no augmentation** — masking is the only perturbation.
* **target view** = the full unit-max window, encoded by the **EMA target encoder** under
  ``no_grad``; its pooled latent is LayerNorm-normalised (I-JEPA-style target normalisation).
* a small **predictor** MLP maps the context encoder's pooled latent to the target latent; the loss
  is smooth-L1 between prediction and the (stop-grad) normalised target.
* after each step the **target encoder** is EMA-updated from the context encoder with a cosine
  momentum schedule 0.996 → 1.0.

Run (cluster ARM GPU node, never the Intel frontend):
    uv run python scripts/pretrain/wireless_jepa.py --epochs 100 --batch-size 512 --seed 42
or via ``sbatch slurm/pretrain_wireless_jepa_arm.sh``. Requires the RadioML 2016.10a split prepared
first (``sbatch slurm/download_prepare_arm.sh radioml_2016_10a``).
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import rfbench.tasks.amc  # noqa: F401  -- import side effect registers the 'amc' task in TASKS
from rfbench.core.registry import get_task
from rfbench.models.foundation.shufflenet1d import EMBED_DIM_X0_5, build_shufflenet1d
from rfbench.models.foundation.wireless_jepa import backbone_checkpoint_path

_LOG = logging.getLogger("wireless_jepa")

#: Complex-magnitude floor so unit-max normalisation never divides by zero.
_MAX_EPS = 1e-8


def _unit_max(iq: Tensor) -> Tensor:
    """Per-sample unit-max normalisation ``iq / max(|iq|)`` over a ``(2, L)`` window."""
    scale = torch.sqrt(iq[0] ** 2 + iq[1] ** 2).max().clamp_min(_MAX_EPS)
    return iq / scale


class _AmcUnlabelledWindows(torch.utils.data.Dataset[Tensor]):
    """The AMC train split as **unlabelled** unit-max-normalised ``(2, L)`` IQ windows.

    Wraps the task's ``train`` dataset and exposes only the ``"iq"`` field — labels are dropped for
    self-supervised pre-training. NO augmentation is applied here (JEPA's signal is the masking,
    applied per batch in the loop).
    """

    def __init__(self, dataset_name: str = "radioml_2016_10a") -> None:
        task = get_task("amc")
        ds = next(d for d in task.datasets() if d.name == dataset_name)
        self._samples = list(ds.load("train"))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tensor:
        return _unit_max(torch.as_tensor(self._samples[idx]["iq"], dtype=torch.float32))


def _mask_context(batch: Tensor, mask_ratio: float, generator: torch.Generator) -> Tensor:
    """Zero a random contiguous time block (length ``mask_ratio * L``) per sample (context view).

    Vectorised: each sample gets an independent block start; the masked positions are set to 0 on a
    copy. This is the ONLY perturbation (no augmentation), matching WirelessJEPA's recipe.
    """
    b, _, length = batch.shape
    device = batch.device
    block = max(1, int(round(mask_ratio * length)))
    starts = torch.randint(0, length - block + 1, (b,), generator=generator).to(device)  # (B,)
    positions = torch.arange(length, device=device)[None, :]  # (1, L)
    masked = (positions >= starts[:, None]) & (positions < (starts[:, None] + block))  # (B, L)
    keep = (~masked).to(batch.dtype)[:, None, :]  # (B, 1, L) broadcast over I/Q
    return batch * keep


class _Predictor(nn.Module):
    """JEPA predictor MLP: maps the context pooled latent to the target pooled latent."""

    def __init__(self, dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@torch.no_grad()
def _ema_update(target: nn.Module, context: nn.Module, momentum: float) -> None:
    """EMA-update ``target`` parameters (and buffers) toward ``context`` with ``momentum``."""
    for t_p, c_p in zip(target.parameters(), context.parameters(), strict=True):
        t_p.mul_(momentum).add_(c_p.detach(), alpha=1.0 - momentum)
    for t_b, c_b in zip(target.buffers(), context.buffers(), strict=True):
        t_b.copy_(c_b)


def _momentum_at(step: int, total: int, base: float, end: float) -> float:
    """Cosine EMA momentum schedule from ``base`` (step 0) to ``end`` (step ``total``)."""
    if total <= 0:
        return end
    progress = min(step, total) / total
    return end - (end - base) * (math.cos(math.pi * progress) + 1.0) / 2.0


def train(args: argparse.Namespace) -> Path:
    """Run JEPA pre-training; save the EMA target-encoder state_dict and return its path."""
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    dataset = _AmcUnlabelledWindows(args.dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers,
    )
    total_steps = args.epochs * max(len(loader), 1)
    _LOG.info("JEPA pre-training on %d unlabelled %s windows (seed=%d, %d steps)", len(dataset),
              args.dataset, args.seed, total_steps)

    context_encoder = build_shufflenet1d().to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    predictor = _Predictor(EMBED_DIM_X0_5).to(device)

    trainable = list(context_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    context_encoder.train()
    predictor.train()
    target_encoder.eval()
    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            batch = batch.to(device)
            context_view = _mask_context(batch, args.mask_ratio, generator)

            z_context = predictor(context_encoder(context_view))
            with torch.no_grad():
                z_target = target_encoder(batch)  # full (unmasked) view
                z_target = torch.nn.functional.layer_norm(z_target, (z_target.shape[-1],))
            loss = torch.nn.functional.smooth_l1_loss(z_context, z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _ema_update(target_encoder, context_encoder,
                        _momentum_at(step, total_steps, args.ema_base, args.ema_end))
            running += float(loss.item())
            step += 1
        _LOG.info("epoch %3d/%d  jepa_smoothl1=%.5f", epoch + 1, args.epochs,
                  running / max(len(loader), 1))

    out = Path(args.out) if args.out else backbone_checkpoint_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "target_encoder_state_dict": target_encoder.state_dict(),  # the probed representation
        "recipe": "jepa_masked_latent_ema",
        "provenance": "WirelessJEPA recipe (arXiv:2601.20190); weights unpublished, retrained",
        "pretrain_dataset": f"{args.dataset} train split (delabelised, in-distribution)",
        "seed": args.seed,
        "epochs": args.epochs,
        "mask_ratio": args.mask_ratio,
        "ema": [args.ema_base, args.ema_end],
    }
    torch.save(payload, out)
    _LOG.info("saved EMA target encoder -> %s", out)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (all defaulted; ``seed=42`` per the reproducibility convention)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="radioml_2016_10a")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5, help="contiguous time fraction masked")
    p.add_argument("--ema-base", type=float, default=0.996, help="EMA momentum at step 0")
    p.add_argument("--ema-end", type=float, default=1.0, help="EMA momentum at the final step")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="checkpoint path (default: $RFBENCH_CACHE/...)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging, parse args, run pre-training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(_parse_args(argv))


if __name__ == "__main__":
    main()
