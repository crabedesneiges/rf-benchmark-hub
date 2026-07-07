"""SEI-specific from-scratch training loop -- kept OUT of the shared ``rfbench.training``.

The AMC training loop (:func:`rfbench.training.train_baseline`) is deliberately untouched by the
SEI work: the WiSig / ORACLE recipe differs from it on four load-bearing axes, so SEI gets its own
module here. Versus the AMC loop this one adds:

1. **Class-weighted cross-entropy.** WiSig ManyTx is imbalanced (``p=0.9``); the reference passes
   ``class_weight = max(per-class count) / per-class count`` to Keras ``fit``. We reproduce the
   Keras semantics exactly -- ``sum_i w[y_i] * CE_i / N`` -- via ``cross_entropy(..., weight=w,
   reduction='sum') / batch`` (torch's default ``reduction='mean'`` with a weight normalises by
   ``sum(w[y_i])`` instead, which is NOT what Keras does).
2. **Explicit L2 on the regularised kernels only.** The WiSig / ORACLE / complex nets expose
   ``l2_penalty()`` (the sum of squared *kernels* Keras' ``kernel_regularizer=l2(lambda)`` targets
   -- e.g. WiSig's three Dense layers only, never the convs). We add ``l2_lambda * penalty`` to the
   loss, matching Keras, instead of torch's coupled ``weight_decay`` on every parameter. Models
   without the hook (ResNet-1D) fall back to Adam ``weight_decay``.
3. **Best checkpoint + early stop on validation LOSS** (WiSig: ``ModelCheckpoint(monitor=
   'val_loss', save_best_only=True)`` + ``EarlyStopping(monitor='val_loss', patience=5)`` +
   ``load_weights``), NOT the AMC loop's val-ACCURACY selection.
4. **The SEI ``(window, 2)`` time-major layout** and a **track-aware** final ``evaluate`` (the AMC
   loop is single-track): closed_set / cross_receiver / cross_day are trained and scored as
   SEPARATE rows via the ``track`` argument threaded into :func:`rfbench.core.evaluate.evaluate`.

The models' ``nn.Module.forward`` accepts the natural ``(B, window, 2)`` batch (see the SEI
baseline modules), so the collate here and the Model wrapper's eval path feed the SAME tensor --
no train/eval layout skew.

HARD CONSTRAINT: ``import rfbench`` / ``import rfbench.core`` stay dependency-free -- ``torch`` is
imported lazily inside every function; this module is imported explicitly by the SEI training
driver, never by ``import rfbench``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from rfbench.core.dataset import Dataset
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import SplitName, Track
from rfbench.training import resolve_device, resolve_module

if TYPE_CHECKING:  # pragma: no cover - typing-only imports, never executed at runtime
    import torch
    from torch import nn

logger = logging.getLogger(__name__)

# Named aliases (not literal ``Any``) so the torch-free import path stays typed without importing
# torch at module top, while ruff's ANN401 "no bare Any" rule is satisfied -- same trick as
# ``rfbench.training``.
#: A map-style dataset of per-sample SEI ``Batch`` dicts (what ``Dataset.load`` returns).
MapDataset: TypeAlias = Any
#: A ``torch.utils.data.DataLoader`` yielding ``(x, y)`` batches.
DataLoaderT: TypeAlias = Any
#: A ``torch.optim.Optimizer`` (Adam here).
OptimizerT: TypeAlias = Any
#: A ``torch.Tensor`` (logits / targets / class-weight vector).
TensorT: TypeAlias = Any
#: A ``DataLoader`` ``collate_fn`` mapping ``(iq, label)`` pairs to ``((B, window, 2), (B,))``.
CollateFn: TypeAlias = Callable[..., "tuple[torch.Tensor, torch.Tensor]"]

#: SEI baselines train from scratch (a real fit here); ``full_finetune`` runs the same fit.
_TRAINABLE_REGIMES = (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE)

# --- WiSig d006 recipe defaults (docs/BIBLIOGRAPHY.md B.4; the driver overrides per model) -------
#: Adam learning rate (WiSig ``Adam(5e-4)``).
DEFAULT_LR = 5e-4
#: Batch size (WiSig relies on the Keras ``fit`` default of 32).
DEFAULT_BATCH_SIZE = 32
#: Max epochs (WiSig ``n_epochs = 100``; early stopping usually stops sooner).
DEFAULT_EPOCHS = 100
#: Early-stopping / best-checkpoint patience on val LOSS (WiSig ``patience = 5``).
DEFAULT_PATIENCE = 5
#: L2 strength added via the model's ``l2_penalty()`` hook (WiSig / ORACLE ``l2(1e-4)``).
DEFAULT_L2_LAMBDA = 1e-4


def count_classes(dataset: Dataset, *, split: SplitName = "train") -> int:
    """Return the transmitter-class count = ``max(label) + 1`` over ``split`` (head width).

    The SEI on-disk loader assigns dense class indices from the sorted set of transmitter ids
    across ALL records, so every WiSig condition's ``train`` split spans the full ``0..n_tx-1``
    label space -- ``max(label) + 1`` recovers ``n_tx`` for building the model head. Raises if
    the split is empty.
    """
    source = dataset.load(split, None)
    max_label = -1
    n = 0
    for sample in source:
        max_label = max(max_label, int(sample["label"]))
        n += 1
    if n == 0:
        raise ValueError(f"cannot infer class count: '{split}' split of {dataset.name!r} is empty")
    return max_label + 1


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


class _SeiSupervisedView:
    """A map-style dataset over per-sample SEI ``Batch`` dicts, yielding ``(iq, label)`` pairs."""

    def __init__(self, source: MapDataset) -> None:
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self._source[index]
        return sample["iq"], int(sample["label"])


def _make_sei_collate(device: str) -> CollateFn:
    """Return a ``collate_fn`` stacking ``(iq, label)`` -> ``((B, window, 2), (B,))`` on ``device``.

    Each ``iq`` is a ``(window, 2)`` time-major array-like (numpy on the cluster, nested lists in a
    fixture). The result keeps the SEI ``(window, 2)`` layout the baselines' ``nn.Module.forward``
    expects -- the SAME layout fed by the Model wrapper at eval time.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def collate(pairs: list[tuple[Any, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        iqs = [iq for iq, _ in pairs]
        labels = [label for _, label in pairs]
        x = torch.as_tensor(np.asarray(iqs, dtype=np.float32), device=device)
        if x.ndim == 2:  # a single (window, 2) sample slipped through -> add the batch axis
            x = x.unsqueeze(0)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        return x, y

    return collate


def _make_loader(
    source: MapDataset, *, batch_size: int, device: str, num_workers: int, shuffle: bool
) -> DataLoaderT:
    """Build a ``DataLoader`` over ``source`` with the SEI ``(window, 2)`` collate on ``device``."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    return DataLoader(
        _SeiSupervisedView(source),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_make_sei_collate(device),
    )


def _class_weights(source: MapDataset, num_classes: int, device: str) -> TensorT:
    """Compute WiSig ``class_weight = max(count) / count`` per class over the train ``source``.

    Mirrors ``prepare_txid_and_weights`` in the reference ``data_utilities.py``: count the labels,
    then weight each class by the majority-class count over its own count (so rare emitters are
    up-weighted). A class absent from the train split gets weight ``1.0`` (never divides by zero).
    Returns a ``(num_classes,)`` float tensor on ``device``.
    """
    import torch  # noqa: PLC0415

    counts = [0] * num_classes
    for sample in source:
        counts[int(sample["label"])] += 1
    max_count = max(counts) if counts else 0
    weights = [max_count / c if c > 0 else 1.0 for c in counts]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _weighted_ce(logits: TensorT, targets: TensorT, weight: TensorT) -> TensorT:
    """Keras-``class_weight`` cross-entropy: ``sum_i w[y_i] * CE_i / N`` (not torch's norm)."""
    import torch  # noqa: PLC0415

    n = int(targets.shape[0])
    total = torch.nn.functional.cross_entropy(logits, targets, weight=weight, reduction="sum")
    return total / max(n, 1)


def _snapshot(module: nn.Module) -> dict[str, Any]:
    """Return a detached CPU copy of ``module``'s ``state_dict`` (the best-val checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _val_loss(module: nn.Module, loader: DataLoaderT, l2_lambda: float) -> float:
    """Mean per-batch validation loss: **UNWEIGHTED** CE + the L2 term (Keras ``val_loss``).

    Matches what Keras' ``ModelCheckpoint``/``EarlyStopping`` monitor: ``class_weight`` is applied
    to the TRAINING loss only (sample weighting in ``fit``), NOT to the validation loss, while the
    L2 regularisation loss IS included in both. Monitoring a *weighted* val loss for checkpoint
    selection is a fidelity bug (it selects a different checkpoint than the reference) -- so the CE
    here is unweighted.
    """
    import torch  # noqa: PLC0415

    module.eval()
    running = 0.0
    n_batches = 0
    penalty = _l2_term(module, l2_lambda)
    with torch.no_grad():
        for x, y in loader:
            loss = torch.nn.functional.cross_entropy(
                module(x), y
            )  # unweighted, like Keras val_loss
            running += float(loss.item()) + penalty
            n_batches += 1
    return running / n_batches if n_batches else float("nan")


def _l2_term(module: nn.Module, l2_lambda: float) -> float:
    """Scalar ``l2_lambda * module.l2_penalty()`` (0.0 if the model has no L2 hook)."""
    if l2_lambda <= 0 or not hasattr(module, "l2_penalty"):
        return 0.0
    return float(l2_lambda) * float(module.l2_penalty().item())


def train_sei_baseline(
    task: Task,
    model: Model,
    dataset: Dataset,
    *,
    track: Track,
    regime: RegimeSpec,
    num_classes: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    weight_decay: float = 0.0,
    use_class_weight: bool = True,
    seed: int = 42,
    device: str | None = None,
    out_path: Path | None = None,
    num_workers: int = 0,
    patience: int = DEFAULT_PATIENCE,
    val_split: SplitName = "val",
) -> tuple[Model, dict[str, Any]]:
    """Fit an SEI baseline on ``dataset``'s TRAIN split, then track-aware ``evaluate`` on TEST.

    Runs the WiSig-faithful recipe (class-weighted CE, explicit L2 via the model's ``l2_penalty``,
    Adam, best-checkpoint + early-stop on val LOSS, best-weights restore), then calls
    :func:`rfbench.core.evaluate.evaluate` on ``task.default_split()`` with the declared ``regime``
    and ``track`` so the ONE canonical writer emits a schema-valid ``result.json`` carrying
    ``split.track``. ``num_classes`` MUST match the model head width (use :func:`count_classes`).

    ``l2_lambda`` adds ``l2_lambda * model.l2_penalty()`` to the loss for models exposing the hook
    (WiSig / ORACLE / complex); ``weight_decay`` is the fallback Adam decay for models without it
    (ResNet-1D). ``use_class_weight`` toggles the WiSig ``max(count)/count`` weighting. When
    ``val`` is unavailable/empty the loop degrades to monitoring TRAIN loss. Returns
    ``(trained_model, result_dict)``.
    """
    import torch  # noqa: PLC0415

    from rfbench.core.evaluate import evaluate  # noqa: PLC0415
    from rfbench.training import _sync_model_device  # noqa: PLC0415

    if regime.name not in _TRAINABLE_REGIMES:
        raise ValueError(
            f"train_sei_baseline fits only {[r.value for r in _TRAINABLE_REGIMES]} regimes; "
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
    weight = (
        _class_weights(train_source, num_classes, resolved_device) if use_class_weight else None
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

    use_wd = 0.0 if hasattr(module, "l2_penalty") and l2_lambda > 0 else weight_decay
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=use_wd)

    best_loss = float("inf")
    best_state = _snapshot(module)
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        train_loss = _train_one_epoch(module, train_loader, optimizer, weight, l2_lambda)
        if train_loss != train_loss:  # NaN -> diverged; keep best checkpoint
            logger.error(
                "SEI training DIVERGED at epoch %d/%d (NaN loss); restoring best checkpoint "
                "(epoch %d).",
                epoch + 1,
                epochs,
                best_epoch + 1,
            )
            break

        monitored = (
            _val_loss(module, val_loader, l2_lambda) if val_loader is not None else train_loss
        )
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
        track=track,
        seed=seed,
        batch_size=max(batch_size, 256),  # eval throughput; scoring is batch-size invariant
        device=resolved_device,
        out_path=out_path,
    )
    return model, result


def _train_one_epoch(
    module: nn.Module, loader: DataLoaderT, optimizer: OptimizerT, weight: TensorT, l2_lambda: float
) -> float:
    """Run one optimisation pass; return the mean per-batch train loss (CE + L2 penalty)."""
    import torch  # noqa: PLC0415

    module.train()
    running = 0.0
    n_batches = 0
    add_l2 = l2_lambda > 0 and hasattr(module, "l2_penalty")
    for x, y in loader:
        optimizer.zero_grad()
        logits = module(x)
        loss = (
            _weighted_ce(logits, y, weight)
            if weight is not None
            else torch.nn.functional.cross_entropy(logits, y)
        )
        if add_l2:
            loss = loss + l2_lambda * module.l2_penalty()
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
    "train_sei_baseline",
    "count_classes",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_PATIENCE",
    "DEFAULT_L2_LAMBDA",
]
