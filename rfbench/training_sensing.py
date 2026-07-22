"""Multi-label training loop for the DeepSense spectrum-sensing baseline (from_scratch).

:func:`train_sensing_baseline` is the sensing analogue of :func:`rfbench.training.train_baseline`,
but for MULTI-LABEL occupancy: each ``(2, 32)`` window carries a length-16 per-subband occupancy
vector, so the loss is ``BCEWithLogitsLoss`` (16 independent sigmoid decisions), NOT the single
softmax cross-entropy the AMC/SEI loops use. It:

1. resolves the underlying ``nn.Module`` (``DeepSenseNet``, whose ``forward`` returns raw
   ``(B, 16)`` LOGITS so BCE stays numerically stable),
2. streams the task's ``train`` split through a ``DataLoader`` with a multi-label collate
   (``x (B, 2, 32)`` float32, ``y (B, 16)`` float32),
3. optimises with Adam + BCE for up to ``epochs``, MONITORING the ``val`` split each epoch to drive
   a ``ReduceLROnPlateau`` on val LOSS, keep the BEST-VAL-MICRO-F1 ``state_dict`` and EARLY-STOP,
4. restores that best state, then hands the model to :func:`rfbench.core.evaluate.evaluate` on the
   ``test`` split so the ONE canonical writer emits a schema-valid ``result.json``.

Checkpoint selection / early stopping key on the val MICRO-F1 over window×subband cells -- the same
quantity :class:`rfbench.tasks.spectrum_sensing.metrics.OccupancyClassification` reports as the
board primary -- so the restored checkpoint is the F1-peak epoch. When no usable ``val`` split
exists the loop degrades to the train-LOSS signal, exactly like the AMC loop.

The shared AMC loop (:mod:`rfbench.training`) is UNTOUCHED; this module reuses its device/seed/
checkpoint helpers and adds only the multi-label view, collate, BCE step and micro-F1 monitor.

HARD CONSTRAINT: ``import rfbench`` stays dependency-free -- ``torch`` is imported LAZILY inside
every function here. This module is NOT imported by ``import rfbench``; a caller (``rfbench
sensing-train`` / a torch test) imports it explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from rfbench.core.dataset import Dataset
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import SplitName
from rfbench.training import (
    DEFAULT_LR_FACTOR,
    DEFAULT_LR_PATIENCE,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_LR,
    DEFAULT_VAL_SPLIT,
    _atomic_save_checkpoint,
    _load_val_source,
    _seed_everything,
    _snapshot_state,
    _sync_model_device,
    resolve_device,
    resolve_module,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only imports, never executed at runtime
    import torch
    from torch import nn

logger = logging.getLogger(__name__)

MapDataset: TypeAlias = Any
DataLoaderT: TypeAlias = Any
OptimizerT: TypeAlias = Any
CriterionT: TypeAlias = Any
#: A ``DataLoader`` ``collate_fn`` mapping ``(iq, label_vector)`` pairs to batched tensors. Aliased
#: (not a bare ``Any``) so we neither import torch at module top nor trip ruff's ANN401 rule.
CollateFn: TypeAlias = Any

#: The regimes this loop fits (the pass-through, real-fit regimes). Probing regimes go elsewhere.
TRAINABLE_REGIMES = (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE)

#: Sensing early-stopping patience (epochs of no val-micro-F1 gain). Higher than AMC's default is
#: unnecessary; DeepSense converges fast on the capped split.
DEFAULT_PATIENCE = 15
#: Global gradient-norm cap before each optimizer step (stabilises the fit; ``None`` disables).
DEFAULT_GRAD_CLIP = 5.0
#: Hard-decision threshold on the per-subband ``P(occupied)`` for the val micro-F1 monitor.
_OCCUPANCY_THRESHOLD = 0.5


class _SensingSupervisedView:
    """A map-style ``torch`` dataset yielding ``(iq, label_vector)`` for the DataLoader.

    Wraps :meth:`Dataset.load` (a map-style dataset of per-sample dicts with ``iq`` (2, 32) and a
    length-16 ``label`` occupancy vector) into pairs a ``DataLoader`` can batch. Kept a plain class
    (not subclassing ``torch.utils.data.Dataset`` at module top) so the import stays torch-free.
    """

    def __init__(self, source: MapDataset) -> None:
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[Any, list[float]]:
        sample = self._source[index]
        return sample["iq"], [float(bit) for bit in sample["label"]]


def _make_collate(device: str) -> CollateFn:
    """Return a ``collate_fn`` stacking ``(iq, label)`` into ``(x, y)`` tensors on ``device``.

    Each ``iq`` payload is a ``(2, 32)`` array-like (numpy on the cluster, nested lists in a
    fixture); ``np.asarray`` handles both. Produces ``x (B, 2, 32)`` float32 and a ``y (B, 16)``
    float32 multi-label target for ``BCEWithLogitsLoss``.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def collate(pairs: list[tuple[Any, list[float]]]) -> tuple[torch.Tensor, torch.Tensor]:
        iqs = [iq for iq, _ in pairs]
        labels = [label for _, label in pairs]
        x = torch.as_tensor(np.asarray(iqs, dtype=np.float32), device=device)
        if x.ndim == 2:  # a single (2, 32) sample slipped through -> add the batch axis
            x = x.unsqueeze(0)
        y = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=device)
        return x, y

    return collate


def _make_loader(
    source: MapDataset,
    *,
    batch_size: int,
    device: str,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoaderT:
    """Build a ``DataLoader`` over ``source`` with the multi-label IQ collate on ``device``."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    return DataLoader(
        _SensingSupervisedView(source),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_make_collate(device),
    )


def _train_one_epoch(
    module: nn.Module,
    loader: DataLoaderT,
    optimizer: OptimizerT,
    criterion: CriterionT,
    grad_clip: float | None,
) -> float:
    """Run one BCE optimisation pass over ``loader`` and return the mean per-batch train loss."""
    import torch  # noqa: PLC0415

    module.train()
    running = 0.0
    n_batches = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(module(x), y)  # module(x) -> (B, 16) logits; y -> (B, 16) float
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
        optimizer.step()
        running += float(loss.detach().item())
        n_batches += 1
    return running / n_batches if n_batches else float("nan")


def _val_loss_and_micro_f1(
    module: nn.Module, loader: DataLoaderT, criterion: CriterionT
) -> tuple[float, float]:
    """Return ``(mean BCE loss, micro-F1)`` over ``loader`` -- no gradient updates.

    The micro-F1 uses the SAME per-subband sigmoid + 0.5 threshold + cell micro-averaging as
    :class:`rfbench.tasks.spectrum_sensing.metrics.OccupancyClassification`, so val selection and
    the final test score agree on what "correct" means. Both scalars are ``nan`` on an empty
    loader so the caller can fall back to the train signal.
    """
    import torch  # noqa: PLC0415

    module.eval()
    running_loss = 0.0
    n_batches = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            logits = module(x)
            running_loss += float(criterion(logits, y).detach().item())
            n_batches += 1
            predicted = (torch.sigmoid(logits) >= _OCCUPANCY_THRESHOLD).to(y.dtype)
            truth = (y >= 0.5).to(y.dtype)
            tp += int((predicted * truth).sum().item())
            fp += int((predicted * (1 - truth)).sum().item())
            fn += int(((1 - predicted) * truth).sum().item())
    if n_batches == 0:
        return float("nan"), float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return running_loss / n_batches, f1


def train_sensing_baseline(
    task: Task,
    model: Model,
    dataset: Dataset,
    *,
    regime: RegimeSpec,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
    device: str | None = None,
    out_path: Path | None = None,
    num_workers: int = 0,
    val_split: SplitName = DEFAULT_VAL_SPLIT,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    lr_factor: float = DEFAULT_LR_FACTOR,
    lr_patience: int = DEFAULT_LR_PATIENCE,
    min_lr: float = DEFAULT_MIN_LR,
    grad_clip: float | None = DEFAULT_GRAD_CLIP,
    checkpoint_out: Path | None = None,
) -> tuple[Model, dict[str, Any]]:
    """Fit ``model`` on ``dataset``'s TRAIN split (multi-label BCE), then ``evaluate`` on TEST.

    Runs the DeepSense recipe for the ``from_scratch`` / ``full_finetune`` regimes (Adam + BCE over
    the 16 sub-bands), MONITORING ``val_split`` each epoch to drive a ``ReduceLROnPlateau`` on val
    LOSS, keep the BEST-VAL-MICRO-F1 ``state_dict`` (restored before the final eval) and EARLY-STOP
    once ``patience`` epochs pass with no micro-F1 gain > ``min_delta``. When ``val_split`` is
    unavailable/empty the loop degrades to the train-LOSS signal.

    After fitting, restores the best state and calls :func:`rfbench.core.evaluate.evaluate` on the
    task's default (``test``) split with the same declared ``regime`` so the single canonical writer
    emits (and, if ``out_path`` is set, writes) the ``result.json``. When ``checkpoint_out`` is set,
    the best-val checkpoint is also persisted atomically. Returns ``(trained_model, result_dict)``.
    Raises ``ValueError`` on a non-trainable regime or non-positive ``epochs``/``batch_size``/
    ``patience``.
    """
    import torch  # noqa: PLC0415

    from rfbench.core.evaluate import evaluate  # noqa: PLC0415

    if regime.name not in TRAINABLE_REGIMES:
        raise ValueError(
            f"train_sensing_baseline fits only {[r.value for r in TRAINABLE_REGIMES]} regimes; "
            f"got {regime.name.value!r}."
        )
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    resolved_device = resolve_device(device)
    _seed_everything(seed)

    module = resolve_module(model).to(resolved_device)
    _sync_model_device(model, resolved_device)

    train_loader = _make_loader(
        dataset.load("train", None),
        batch_size=batch_size,
        device=resolved_device,
        shuffle=True,
        num_workers=num_workers,
    )
    val_source = _load_val_source(dataset, val_split)
    val_loader = (
        None
        if val_source is None
        else _make_loader(
            val_source,
            batch_size=batch_size,
            device=resolved_device,
            shuffle=False,
            num_workers=num_workers,
        )
    )
    if val_loader is None:
        logger.warning(
            "no usable '%s' split for %r; monitoring TRAIN loss for the LR schedule / early "
            "stopping / best-checkpoint restore instead (val-micro-F1 selection unavailable).",
            val_split,
            dataset.name,
        )

    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    best_f1 = -1.0  # any real micro-F1 in [0, 1] beats this -> epoch 0 always snapshots
    best_loss = float("inf")  # only used in the no-val (train-loss) fallback branch
    best_state = _snapshot_state(module)
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        train_loss = _train_one_epoch(module, train_loader, optimizer, criterion, grad_clip)

        if train_loss != train_loss:  # NaN -> diverged; keep the best checkpoint so far
            logger.error(
                "training DIVERGED at epoch %d/%d (train loss is NaN); stopping and restoring the "
                "best checkpoint (epoch %d).",
                epoch + 1,
                epochs,
                best_epoch + 1,
            )
            break

        if val_loader is not None:
            val_loss, val_f1 = _val_loss_and_micro_f1(module, val_loader, criterion)
            step_loss = train_loss if val_loss != val_loss else val_loss  # NaN check w/o math
            scheduler.step(step_loss)
            improved = val_f1 == val_f1 and val_f1 > best_f1 + min_delta
            if improved:
                best_f1 = val_f1
                best_state = _snapshot_state(module)
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            logger.info(
                "epoch %d/%d: train loss = %.4f, val loss = %.4f, val micro-F1 = %.4f, "
                "lr = %.2e%s",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_f1,
                optimizer.param_groups[0]["lr"],
                " (best)" if improved else "",
            )
        else:
            scheduler.step(train_loss)
            improved = train_loss < best_loss - min_delta
            if improved:
                best_loss = train_loss
                best_state = _snapshot_state(module)
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            logger.info(
                "epoch %d/%d: train loss = %.4f, lr = %.2e%s",
                epoch + 1,
                epochs,
                train_loss,
                optimizer.param_groups[0]["lr"],
                " (best)" if improved else "",
            )

        if epochs_since_improve >= patience:
            logger.info(
                "early stopping at epoch %d/%d (no improvement for %d epochs; best epoch %d)",
                epoch + 1,
                epochs,
                patience,
                best_epoch + 1,
            )
            break

    module.load_state_dict(best_state)
    if val_loader is not None:
        logger.info(
            "restored best-val-micro-F1 checkpoint from epoch %d (val micro-F1 %.4f)",
            best_epoch + 1,
            best_f1,
        )
    else:
        logger.info(
            "restored best-train-loss checkpoint from epoch %d (train loss %.4f)",
            best_epoch + 1,
            best_loss,
        )

    if checkpoint_out is not None:
        checkpoint: dict[str, Any] = {"state_dict": best_state, "epoch": best_epoch, "seed": seed}
        if val_loader is not None:
            checkpoint["val_micro_f1"] = best_f1
        else:
            checkpoint["train_loss"] = best_loss
        _atomic_save_checkpoint(checkpoint, checkpoint_out)
        logger.info("saved best checkpoint -> %s", checkpoint_out)

    result = evaluate(
        model,
        task,
        task.default_split(),
        regime,
        dataset=dataset.name,
        seed=seed,
        batch_size=batch_size,
        device=resolved_device,
        out_path=out_path,
    )
    return model, result


__all__ = [
    "train_sensing_baseline",
    "TRAINABLE_REGIMES",
    "DEFAULT_PATIENCE",
    "DEFAULT_GRAD_CLIP",
]
