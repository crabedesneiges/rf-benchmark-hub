"""Per-epoch CLDNN collapse diagnostic (fix/cldnn-collapse).

Instruments the AMC from-scratch training loop for the CLDNN baseline and logs, EVERY epoch,
the signals that distinguish the competing root-cause hypotheses for the 0.0909 (chance)
collapse:

* ``val_acc``        -- does the model ever leave chance (1/11 = 0.0909)? A per-epoch trajectory
                        that is flat at 0.0909 is a genuine optimization failure (not a
                        checkpoint-restore artifact, which would show a rising then discarded acc).
* ``lr``             -- what the ReduceLROnPlateau schedule actually does (ratchets down on a
                        flat val loss, or holds high),
* ``grad_norm_pre``  -- the GLOBAL gradient norm BEFORE clipping (``clip_grad_norm_`` returns
                        it): a persistently >clip value means the clip masks an instability;
                        a tiny decaying value means a dead/vanishing gradient,
* ``clip_frac``      -- fraction of steps the 5.0 clip actually bit,
* ``pred_entropy`` / ``top_class_frac`` -- the val prediction distribution: a constant-class
                        (dead) predictor has entropy 0 and ``top_class_frac`` 1.0,
* ``conv_std`` / ``lstm_std`` -- activation spread at the conv output and the LSTM output
                        (dead features collapse toward 0).

It runs several VARIANTS back-to-back in ONE job. ``input_norm`` uses the SHIPPED ``CLDNNNet``
toggle (so ``norm`` IS the model ``MODELS.get('cldnn')()`` builds -- the run validates the real
fix); the LSTM-init half is applied inline (:func:`_stabilize_lstm`) since it was ruled out:

* ``broken``     -- ``CLDNNNet(input_norm=False)``: raw ~1e-2-RMS IQ -> FRAGILE (chance on some
                    init draws; this is what collapsed on the board's unseeded init),
* ``norm``       -- ``input_norm=True`` (per-sample unit-variance normalization, the O'Shea/ResNet
                    transform that cured ResNet's identical 1/11 collapse) -- **the shipped fix**,
* ``init``       -- inline forget-bias-1 + orthogonal LSTM init, NO norm (verified HARMFUL:
                    collapses to chance -- the deep LSTM ignores the tiny input),
* ``norm_init``  -- both (init is inert once normalized: same score as ``norm``).

Empirical verdict (job 86194, seed 42, 20 ep): broken 0.5659, norm 0.5848, init 0.0909, norm_init
0.5848 -> normalization is necessary AND sufficient; the LSTM re-init is not the fix. Re-run across
seeds (``--seed``) to confirm ``norm`` is init-robust before the 150-epoch retrain.

MCLDNN/ResNet are never constructed here, so they cannot be affected.

Run (short, on an ARM GB200 node via the GPU venv)::

    uv run python slurm/diagnose_cldnn.py --variants broken,norm --epochs 20 \
        --out "$WORK/logs/cldnn_diag_${SLURM_JOB_ID}.json"

Needs ``rfbench[torch,tasks,data]`` (torch + numpy) and a prepared RadioML 2016.10a split under
``$RFBENCH_CACHE``. CPU-runnable in principle but intended for the cluster (real data + GPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from torch import Tensor, nn

logger = logging.getLogger("diagnose_cldnn")

# Torch runtime types aliased to Any so this module imports torch-free and stays ANN401-clean
# (a TypeAlias token, not a bare ``Any``, satisfies ruff -- same pattern as rfbench.training).
MapDataset: TypeAlias = Any
DataLoaderT: TypeAlias = Any
OptimizerT: TypeAlias = Any
CriterionT: TypeAlias = Any

#: The recipe defaults mirrored from rfbench.training so the diagnostic reproduces the exact loop
#: the board retrain uses (ReduceLROnPlateau on val loss, grad clip 5.0).
_DEFAULT_LR = 1e-3
_DEFAULT_BATCH = 256
_DEFAULT_GRAD_CLIP = 5.0
_DEFAULT_LR_FACTOR = 0.5
_DEFAULT_LR_PATIENCE = 10
_DEFAULT_MIN_LR = 1e-7

#: The variants -> (input_norm via the shipped toggle, apply_init inline for the ablation).
_VARIANTS: dict[str, tuple[bool, bool]] = {
    "broken": (False, False),
    "norm": (True, False),
    "init": (False, True),
    "norm_init": (True, True),
}


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs (mirrors rfbench.training._seed_everything)."""
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stabilize_lstm(lstm: nn.Module) -> None:
    """Ablation-only: forget-gate bias 1.0 + orthogonal recurrent / Xavier input init on an LSTM.

    Applied HERE (not in ``cldnn.py``) because the diagnostic PROVED this intervention is not the
    fix: it is inert once the input is normalized and actively causes the chance-collapse when
    applied without normalization. Kept solely so the ``init`` / ``norm_init`` ablation variants
    stay reproducible. ``nn.LSTM`` bias layout is ``[i, f, g, o]`` so the forget slice is [H:2H].
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            with torch.no_grad():
                param.zero_()
                hidden = param.shape[0] // 4
                param[hidden : 2 * hidden].fill_(1.0)


def _build_cldnn_net(variant: str, device: str) -> nn.Module:
    """Build a ``CLDNNNet`` for the ``variant``: input_norm via the SHIPPED toggle, init inline.

    ``input_norm`` uses the real ``cldnn.py`` toggle (so ``norm`` IS the shipped model
    ``MODELS.get('cldnn')()`` builds -- the diagnostic validates the real fix). The LSTM-init half
    is applied inline via :func:`_stabilize_lstm` because it was ruled out and no longer lives in
    the model.
    """
    from rfbench.models.baselines.cldnn import CLDNNNet  # noqa: PLC0415

    input_norm, apply_init = _VARIANTS[variant]
    net = CLDNNNet(input_norm=input_norm).to(device)
    if apply_init:
        _stabilize_lstm(net.lstm)
    return net


def _make_loader(source: MapDataset, *, batch_size: int, device: str, shuffle: bool) -> DataLoaderT:
    """A DataLoader over ``(iq, label)`` pairs collated to ``(B, 2, L)`` float / ``(B,)`` long."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    pairs = [(source[i]["iq"], int(source[i]["label"])) for i in range(len(source))]

    def collate(batch: list[tuple[MapDataset, int]]) -> tuple[Tensor, Tensor]:
        iqs = np.asarray([iq for iq, _ in batch], dtype=np.float32)
        labels = [label for _, label in batch]
        x = torch.as_tensor(iqs, device=device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        return x, y

    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _prediction_stats(counts: list[int]) -> tuple[float, float]:
    """Return ``(entropy_bits, top_class_fraction)`` of a predicted-class histogram.

    A healthy classifier spreads predictions (high entropy, low top fraction); a collapsed
    constant-class predictor has entropy 0 and top fraction 1.0 -- the direct chance-collapse tell.
    """
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    probs = [c / total for c in counts if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy, max(counts) / total


def _evaluate(
    net: nn.Module, loader: DataLoaderT, criterion: CriterionT, num_classes: int
) -> dict[str, float]:
    """One no-grad val pass -> loss, accuracy, prediction entropy/top-fraction, activation stds.

    Reconstructs the forward from ``_conv_sequence`` (which applies the model's own input
    normalization when enabled) so ``conv_std`` / ``lstm_std`` can be read off the fused sequence
    and the LSTM output -- the dead-feature tell.
    """
    import torch  # noqa: PLC0415

    net.eval()
    running_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    class_counts = [0] * num_classes
    conv_std_sum = 0.0
    lstm_std_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            seq = net._conv_sequence(x)  # (B, L, F+2) -- (maybe-normalized) conv features + skip
            out, _ = net.lstm(seq)
            logits = net.classifier(net.fc_embed(out[:, -1, :]))
            conv_std_sum += float(seq[:, :, : net.conv_filters].std().item())
            lstm_std_sum += float(out.std().item())
            running_loss += float(criterion(logits, y).item())
            n_batches += 1
            predicted = logits.argmax(dim=1)
            correct += int((predicted == y).sum().item())
            total += int(y.numel())
            for cls in predicted.tolist():
                class_counts[cls] += 1
    entropy, top_frac = _prediction_stats(class_counts)
    return {
        "val_loss": running_loss / n_batches if n_batches else float("nan"),
        "val_acc": correct / total if total else float("nan"),
        "pred_entropy": entropy,
        "top_class_frac": top_frac,
        "conv_std": conv_std_sum / n_batches if n_batches else float("nan"),
        "lstm_std": lstm_std_sum / n_batches if n_batches else float("nan"),
    }


def _train_one_epoch(
    net: nn.Module,
    loader: DataLoaderT,
    optimizer: OptimizerT,
    criterion: CriterionT,
    grad_clip: float,
) -> dict[str, float]:
    """One optimisation pass -> mean train loss, mean PRE-clip grad-norm, clip-bite fraction."""
    import torch  # noqa: PLC0415

    net.train()
    running = 0.0
    grad_pre_sum = 0.0
    clipped = 0
    n_batches = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        pre = float(torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip).item())
        optimizer.step()
        running += float(loss.item())
        grad_pre_sum += pre
        clipped += int(pre > grad_clip)
        n_batches += 1
    return {
        "train_loss": running / n_batches if n_batches else float("nan"),
        "grad_norm_pre": grad_pre_sum / n_batches if n_batches else float("nan"),
        "clip_frac": clipped / n_batches if n_batches else float("nan"),
    }


def _run_variant(
    variant: str,
    train_source: MapDataset,
    val_source: MapDataset,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_clip: float,
    seed: int,
    device: str,
    num_classes: int,
) -> dict[str, Any]:
    """Train one variant for ``epochs`` epochs with full per-epoch instrumentation."""
    import torch  # noqa: PLC0415

    _seed_everything(seed)
    net = _build_cldnn_net(variant, device)
    input_norm, apply_init = _VARIANTS[variant]
    train_loader = _make_loader(train_source, batch_size=batch_size, device=device, shuffle=True)
    val_loader = _make_loader(val_source, batch_size=batch_size, device=device, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=_DEFAULT_LR_FACTOR,
        patience=_DEFAULT_LR_PATIENCE,
        min_lr=_DEFAULT_MIN_LR,
    )

    epochs_log: list[dict[str, float]] = []
    for epoch in range(epochs):
        train_stats = _train_one_epoch(net, train_loader, optimizer, criterion, grad_clip)
        val_stats = _evaluate(net, val_loader, criterion, num_classes)
        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_stats["val_loss"])
        epochs_log.append({"epoch": epoch + 1, "lr": lr_now, **train_stats, **val_stats})
        logger.info(
            "[%s] ep %3d/%d | val_acc=%.4f | lr=%.2e | grad_pre=%.3g clip=%.0f%% | "
            "H(pred)=%.2fb top=%.2f | conv_std=%.3g lstm_std=%.3g | tl=%.3f vl=%.3f",
            variant,
            epoch + 1,
            epochs,
            val_stats["val_acc"],
            lr_now,
            train_stats["grad_norm_pre"],
            100 * train_stats["clip_frac"],
            val_stats["pred_entropy"],
            val_stats["top_class_frac"],
            val_stats["conv_std"],
            val_stats["lstm_std"],
            train_stats["train_loss"],
            val_stats["val_loss"],
        )
    best = max((r["val_acc"] for r in epochs_log), default=float("nan"))
    logger.info("[%s] DONE -- best val_acc over %d epochs = %.4f", variant, epochs, best)
    return {
        "variant": variant,
        "input_norm": input_norm,
        "apply_init": apply_init,
        "best_val_acc": best,
        "epochs": epochs_log,
    }


def _load_dataset(cache: str) -> tuple[MapDataset, int]:
    """Resolve the real RadioML 2016.10a AMC dataset (on-disk path) + its class count."""
    os.environ.setdefault("RFBENCH_CACHE", cache)
    import rfbench.models.baselines.cldnn  # noqa: F401, PLC0415  (registers the model)
    import rfbench.tasks.amc  # noqa: F401, PLC0415  (registers the task)
    from rfbench.core.registry import get_task  # noqa: PLC0415
    from rfbench.models.baselines.cldnn import DEFAULT_NUM_CLASSES  # noqa: PLC0415
    from rfbench.training import resolve_amc_dataset  # noqa: PLC0415

    task = get_task("amc")
    dataset = resolve_amc_dataset(task, "radioml_2016_10a")
    return dataset, DEFAULT_NUM_CLASSES


def main(argv: list[str] | None = None) -> int:
    """Parse args, run each variant, log per-epoch trajectories, dump a JSON report."""
    parser = argparse.ArgumentParser(description="Per-epoch CLDNN collapse diagnostic.")
    parser.add_argument(
        "--variants",
        default="broken,norm_init",
        help=f"comma-separated subset of {tuple(_VARIANTS)} (default: broken,norm_init).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per variant (short run).")
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR, help="Adam LR (default 1e-3).")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH, dest="batch_size")
    parser.add_argument("--grad-clip", type=float, default=_DEFAULT_GRAD_CLIP, dest="grad_clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="cuda|cpu (default cuda).")
    parser.add_argument("--cache", default=os.environ.get("RFBENCH_CACHE", ".rfbench_cache"))
    parser.add_argument("--out", default=None, help="Path to write the JSON report.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in _VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s) {unknown}; choose from {tuple(_VARIANTS)}")

    dataset, num_classes = _load_dataset(args.cache)
    # Materialise each split ONCE (the on-disk loader re-reads the whole RML pickle per call).
    train_source = dataset.load("train")
    val_source = dataset.load("val")
    logger.info(
        "dataset=radioml_2016_10a train=%d val=%d classes=%d | variants=%s epochs=%d lr=%.2e",
        len(train_source),
        len(val_source),
        num_classes,
        variants,
        args.epochs,
        args.lr,
    )

    reports = [
        _run_variant(
            v,
            train_source,
            val_source,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            seed=args.seed,
            device=args.device,
            num_classes=num_classes,
        )
        for v in variants
    ]

    logger.info("=== SUMMARY (best val_acc over %d epochs) ===", args.epochs)
    for rep in reports:
        logger.info("  %-10s best_val_acc=%.4f", rep["variant"], rep["best_val_acc"])

    if args.out:
        payload = {
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
            },
            "reports": reports,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote report -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
