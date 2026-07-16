"""SNR-estimation from-scratch REGRESSION training loop -- kept OUT of ``rfbench.training``.

The AMC / SEI loops are classification (cross-entropy on a class label). SNR estimation is a
scalar REGRESSION (raw-IQ window -> SNR in dB), so it gets its own module here. Versus the SEI
loop (:mod:`rfbench.training_sei`) this one differs on the load-bearing axes:

1. **MSE loss on the ``snr_db`` target.** The supervision signal is the per-window SNR (dB), a
   continuous float, so the criterion is mean-squared-error, not cross-entropy; there is no
   class head, no class weighting and no ``num_classes``.
2. **Best checkpoint + early stop on validation LOSS** (the MSE), restoring best weights -- the
   same monitor as SEI, but the monitored quantity is the regression loss.
3. **A single ``all_snr`` track** scored over the full SNR range: SNR estimation never blends
   tracks, so there is no per-condition ``track`` sweep -- the final ``evaluate`` runs once and
   the primary metric it writes is ``rmse_db`` (lower is better).

The model's ``nn.Module.forward`` accepts the natural ``(B, 2, window)`` AMC/SNR batch (see
:mod:`rfbench.models.baselines.snr_cnn`), so the collate here and the ``Model`` wrapper's eval
path feed the SAME tensor -- no train/eval layout skew.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported lazily inside every function; this module is imported explicitly by the SNR training
driver (``rfbench snr-train``), never by ``import rfbench``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from rfbench.core.dataset import Dataset
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import SplitName

if TYPE_CHECKING:  # pragma: no cover - typing-only imports, never executed at runtime
    import torch
    from torch import nn

logger = logging.getLogger(__name__)

# Named aliases (not literal ``Any``) so the torch-free import path stays typed without importing
# torch at module top, while ruff's ANN401 "no bare Any" rule is satisfied -- same trick as
# ``rfbench.training`` / ``rfbench.training_sei``.
#: A map-style dataset of per-sample SNR ``Batch`` dicts (what ``Dataset.load`` returns).
MapDataset: TypeAlias = Any
#: A ``torch.utils.data.DataLoader`` yielding ``(iq, snr_db)`` batches.
DataLoaderT: TypeAlias = Any
#: A ``torch.optim.Optimizer`` (Adam here).
OptimizerT: TypeAlias = Any
#: A ``torch.Tensor`` (predictions / targets).
TensorT: TypeAlias = Any
#: A ``DataLoader`` ``collate_fn`` mapping ``(iq, snr)`` pairs to ``((B, 2, window), (B,))``.
CollateFn: TypeAlias = Callable[..., "tuple[torch.Tensor, torch.Tensor]"]

#: SNR regressors train from scratch (a real fit here); ``full_finetune`` runs the same fit.
_TRAINABLE_REGIMES = (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE)

# --- Recipe defaults (the driver overrides per run) ---------------------------------------------
#: Adam learning rate.
DEFAULT_LR = 1e-3
#: Batch size.
DEFAULT_BATCH_SIZE = 256
#: Max epochs (early stopping usually stops sooner).
DEFAULT_EPOCHS = 100
#: Early-stopping / best-checkpoint patience on val LOSS.
DEFAULT_PATIENCE = 10
#: Adam weight decay (L2). SNR regressors have no ``l2_penalty`` hook, so this is the decay path.
DEFAULT_WEIGHT_DECAY = 1e-4


def _seed_everything(seed: int) -> None:
    """Seed Python / numpy / torch RNGs for a reproducible fit."""
    import random  # noqa: PLC0415

    import torch  # noqa: PLC0415

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ModuleNotFoundError:  # pragma: no cover - numpy present on the training venv
        pass


class _SnrSupervisedView:
    """A map-style dataset over per-sample SNR ``Batch`` dicts, yielding ``(iq, snr_db)`` pairs."""

    def __init__(self, source: MapDataset) -> None:
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[Any, float]:
        sample = self._source[index]
        return sample["iq"], float(sample["snr_db"])


def _make_snr_collate(device: str) -> CollateFn:
    """Return a ``collate_fn`` stacking ``(iq, snr)`` -> ``((B, 2, window), (B,))`` on ``device``.

    Each ``iq`` is a ``(2, window)`` channel-first array-like (numpy on the cluster, nested lists
    in a fixture). The result keeps the AMC/SNR ``(2, window)`` layout the model's
    ``nn.Module.forward`` expects -- the SAME layout fed by the ``Model`` wrapper at eval time.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def collate(pairs: list[tuple[Any, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        iqs = [iq for iq, _ in pairs]
        targets = [snr for _, snr in pairs]
        x = torch.as_tensor(np.asarray(iqs, dtype=np.float32), device=device)
        if x.ndim == 2:  # a single (2, window) sample slipped through -> add the batch axis
            x = x.unsqueeze(0)
        y = torch.as_tensor(targets, dtype=torch.float32, device=device)
        return x, y

    return collate


def _make_loader(
    source: MapDataset, *, batch_size: int, device: str, num_workers: int, shuffle: bool
) -> DataLoaderT:
    """Build a ``DataLoader`` over ``source`` with the SNR ``(2, window)`` collate on ``device``."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    return DataLoader(
        _SnrSupervisedView(source),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_make_snr_collate(device),
    )


def _snapshot(module: nn.Module) -> dict[str, Any]:
    """Return a detached CPU copy of ``module``'s ``state_dict`` (the best-val checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _mse(pred: TensorT, target: TensorT) -> TensorT:
    """Mean-squared error between a ``(B,)`` prediction and a ``(B,)`` SNR target."""
    import torch  # noqa: PLC0415

    return torch.nn.functional.mse_loss(pred.reshape(-1), target.reshape(-1))


def _val_loss(module: nn.Module, loader: DataLoaderT) -> float:
    """Mean per-batch validation MSE (the quantity the best-checkpoint / early-stop monitor)."""
    import torch  # noqa: PLC0415

    module.eval()
    running = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            running += float(_mse(module(x), y).item())
            n_batches += 1
    return running / n_batches if n_batches else float("nan")


def train_snr_regressor(
    task: Task,
    model: Model,
    dataset: Dataset,
    *,
    regime: RegimeSpec,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 42,
    device: str | None = None,
    out_path: Path | None = None,
    num_workers: int = 0,
    patience: int = DEFAULT_PATIENCE,
    val_split: SplitName = "val",
    compute_bootstrap_ci: bool = True,
) -> tuple[Model, dict[str, Any]]:
    """Fit an SNR regressor on ``dataset``'s TRAIN split (MSE), then ``evaluate`` on TEST.

    Runs a from-scratch regression recipe (MSE on the ``snr_db`` target, Adam, best-checkpoint +
    early-stop on val LOSS, best-weights restore), then calls
    :func:`rfbench.core.evaluate.evaluate` on ``task.default_split()`` with the declared ``regime``
    so the ONE canonical writer emits a schema-valid ``result.json`` whose primary metric is
    ``rmse_db`` (lower is better). When the ``val`` split is unavailable/empty the loop degrades to
    monitoring the TRAIN loss.

    ``compute_bootstrap_ci`` forwards to :func:`evaluate`; pass ``False`` in a multi-seed sweep
    (where uncertainty is the across-seed std) to skip the per-run bootstrap and speed eval up on
    large test splits. Returns ``(trained_model, result_dict)``.
    """
    import torch  # noqa: PLC0415

    from rfbench.core.evaluate import evaluate  # noqa: PLC0415
    from rfbench.training import _sync_model_device, resolve_device, resolve_module  # noqa: PLC0415

    if regime.name not in _TRAINABLE_REGIMES:
        raise ValueError(
            f"train_snr_regressor fits only {[r.value for r in _TRAINABLE_REGIMES]} regimes; "
            f"got {regime.name.value!r}."
        )
    for value, label in ((epochs, "epochs"), (batch_size, "batch_size"), (patience, "patience")):
        if value < 1:
            raise ValueError(f"{label} must be >= 1, got {value}")

    resolved_device = resolve_device(device)
    _seed_everything(seed)

    module = resolve_module(model).to(resolved_device)
    _sync_model_device(model, resolved_device)

    train_source = dataset.load("train", None)
    train_loader = _make_loader(
        train_source,
        batch_size=batch_size,
        device=resolved_device,
        num_workers=num_workers,
        shuffle=True,
    )

    val_source = _load_val_source(dataset, val_split)
    val_loader = (
        None
        if val_source is None
        else _make_loader(
            val_source,
            batch_size=batch_size,
            device=resolved_device,
            num_workers=num_workers,
            shuffle=False,
        )
    )
    if val_loader is None:
        logger.warning(
            "no usable '%s' split for %r; monitoring TRAIN loss for early stopping / best "
            "checkpoint instead.",
            val_split,
            dataset.name,
        )

    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = _snapshot(module)
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        train_loss = _train_one_epoch(module, train_loader, optimizer)
        if train_loss != train_loss:  # NaN -> diverged; keep best checkpoint
            logger.error(
                "SNR training DIVERGED at epoch %d/%d (NaN loss); restoring best checkpoint "
                "(epoch %d).",
                epoch + 1,
                epochs,
                best_epoch + 1,
            )
            break

        monitored = _val_loss(module, val_loader) if val_loader is not None else train_loss
        improved = monitored == monitored and monitored < best_loss  # NaN-safe strict improvement
        if improved:
            best_loss = monitored
            best_state = _snapshot(module)
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        logger.info(
            "epoch %d/%d: train loss = %.4f, monitored (%s) loss = %.4f%s",
            epoch + 1,
            epochs,
            train_loss,
            "val" if val_loader is not None else "train",
            monitored,
            " (best)" if improved else "",
        )
        if epochs_since_improve >= patience:
            logger.info(
                "early stopping at epoch %d/%d (no improvement for %d epochs; best epoch %d, "
                "best loss %.4f)",
                epoch + 1,
                epochs,
                patience,
                best_epoch + 1,
                best_loss,
            )
            break

    module.load_state_dict(best_state)
    module.to(resolved_device)
    module.eval()
    logger.info("restored best checkpoint from epoch %d (loss %.4f)", best_epoch + 1, best_loss)

    result = evaluate(
        model,
        task,
        task.default_split(),
        regime,
        dataset=dataset.name,
        track=task.tracks()[0],
        seed=seed,
        batch_size=max(batch_size, 256),  # eval throughput; scoring is batch-size invariant
        device=resolved_device,
        out_path=out_path,
        compute_bootstrap_ci=compute_bootstrap_ci,
    )
    return model, result


def _train_one_epoch(module: nn.Module, loader: DataLoaderT, optimizer: OptimizerT) -> float:
    """Run one optimisation pass; return the mean per-batch train MSE."""
    module.train()
    running = 0.0
    n_batches = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = _mse(module(x), y)
        loss.backward()
        optimizer.step()
        running += float(loss.detach().item())
        n_batches += 1
    return running / n_batches if n_batches else float("nan")


def _load_val_source(dataset: Dataset, val_split: SplitName) -> MapDataset | None:
    """Return the ``val_split`` map-dataset, or ``None`` when unavailable/empty (then use train)."""
    try:
        source = dataset.load(val_split, None)
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError):
        return None
    try:
        if len(source) == 0:
            return None
    except TypeError:  # pragma: no cover - a source without __len__; assume usable
        return source
    return source


__all__ = [
    "train_snr_regressor",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_PATIENCE",
    "DEFAULT_WEIGHT_DECAY",
]
