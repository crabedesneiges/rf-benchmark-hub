"""Real training loop for the from-scratch / full-finetune regimes (WP-30).

:func:`train_baseline` is the M3 training entry point the pass-through regimes
(``from_scratch`` / ``full_finetune``) delegate their *fit* to. It:

1. resolves the underlying ``nn.Module`` of a :class:`~rfbench.core.model.Model`,
2. streams the task's ``train`` split through a ``torch.utils.data.DataLoader``,
3. optimises it with Adam + cross-entropy for up to ``epochs`` epochs (AMC is single-label
   closed-set classification), MONITORING the ``val`` split each epoch to drive a
   ``ReduceLROnPlateau`` LR schedule + early stopping, and keeping the BEST-VAL model state,
4. RESTORES that best-val state, then hands the model to :func:`rfbench.core.evaluate.evaluate`
   on the ``test`` split so the ONE canonical writer emits a schema-valid ``result.json``.

This is the standard AMC training recipe (every RML2016.10a paper trains to convergence with a
plateau schedule + early stopping + best-weights restore -- see ``docs/BIBLIOGRAPHY.md`` Part B,
audit item 1). The previous fixed-epoch, no-schedule, no-early-stop, no-best-val loop was the
single biggest source of the consistent ~1-2 pt shortfall below every published baseline.

The pass-through adapters' ``fit`` stays a no-op at *eval* time (the model is already trained
by this function); the regime is written VERBATIM by ``evaluate`` (D5), so the two regimes
differ only by the declared name -- both run the same real fit here.

HARD CONSTRAINT: ``import rfbench`` and ``import rfbench.core`` stay dependency-free, so
``torch`` is imported LAZILY inside every function here. This module is *not* imported by
``import rfbench``; a caller (``rfbench train`` / a test with torch) imports it explicitly.
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

#: A map-style dataset of per-sample AMC ``Batch`` dicts (what ``Dataset.load`` returns).
#: Aliased to ``Any`` so we never import ``torch.utils.data`` at module top; the alias token
#: (not a literal ``Any``) also keeps ruff's ANN401 "no bare Any" rule satisfied.
MapDataset: TypeAlias = Any
#: A ``DataLoader`` ``collate_fn`` mapping a list of ``(iq, label)`` pairs to batched tensors.
CollateFn: TypeAlias = Callable[[list[tuple[Any, int]]], "tuple[torch.Tensor, torch.Tensor]"]
#: A ``torch.utils.data.DataLoader`` yielding ``(x, y)`` batches. Aliased (not a bare ``Any``) so
#: we neither import ``torch`` at module top nor trip ruff's ANN401 "no bare Any" rule.
DataLoaderT: TypeAlias = Any
#: A ``torch.optim.Optimizer`` (Adam here); aliased for the same torch-free, ANN401-clean reason.
OptimizerT: TypeAlias = Any
#: A loss criterion (``nn.CrossEntropyLoss``); aliased for the same reason.
CriterionT: TypeAlias = Any

#: The regimes ``train_baseline`` actually fits. ``linear_probe`` / ``few_shot`` are adapter
#: regimes (a head fit on a frozen backbone) and go through ``rfbench.regimes`` instead.
TRAINABLE_REGIMES = (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE)

# --- Standard AMC training-recipe defaults (docs/BIBLIOGRAPHY.md Part B, audit item 1) ----------
#: The split monitored each epoch for the LR schedule / early stopping / best-val checkpoint.
DEFAULT_VAL_SPLIT: SplitName = "val"
#: Early-stopping patience: stop after this many epochs with no val-ACCURACY gain > min_delta.
DEFAULT_PATIENCE = 40
#: Minimum val-ACCURACY increase that counts as an improvement (for early stop). ``0.0`` -> any
#: strictly-higher accuracy resets patience (RadioML best-accuracy restore, no dead-band).
DEFAULT_MIN_DELTA = 0.0
#: ``ReduceLROnPlateau`` multiplicative LR factor applied when val loss plateaus.
DEFAULT_LR_FACTOR = 0.5
#: ``ReduceLROnPlateau`` patience (epochs of no val-loss improvement before the LR is scaled).
DEFAULT_LR_PATIENCE = 10
#: Floor the plateau scheduler will not reduce the LR below.
DEFAULT_MIN_LR = 1e-7


def resolve_device(device: str | None) -> str:
    """Return the concrete compute device string, defaulting to CUDA when available.

    ``device=None`` (or ``"auto"``) picks ``"cuda"`` if ``torch.cuda.is_available()`` else
    ``"cpu"``; an explicit ``"cuda"`` / ``"cpu"`` is honoured verbatim. torch is imported
    lazily so this stays out of the dependency-free import path.
    """
    import torch  # noqa: PLC0415 - lazy by design

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_module(model: Model) -> nn.Module:
    """Return the trainable ``nn.Module`` backing a :class:`Model`.

    Baselines expose their network in one of two shapes: the :class:`Model` *is* an
    ``nn.Module`` (multiple inheritance), or it wraps one under a conventional attribute
    (``net`` / ``module`` / ``model``, as the MCLDNN wrapper does with ``self.net``). This
    returns whichever it finds so the training loop is model-agnostic. Raises ``TypeError``
    when no ``nn.Module`` can be located.
    """
    import torch  # noqa: PLC0415 - lazy by design

    if isinstance(model, torch.nn.Module):
        return model
    for attr in ("net", "module", "model"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise TypeError(
        f"model {model.name!r} exposes no trainable nn.Module (checked the model itself and "
        "the .net/.module/.model attributes); train_baseline needs a torch module to optimise."
    )


def _seed_everything(seed: int) -> None:
    """Seed Python-, numpy- (if present) and torch RNGs for a reproducible fit."""
    import random  # noqa: PLC0415

    import torch  # noqa: PLC0415

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ModuleNotFoundError:  # pragma: no cover - numpy is present on the training venv
        pass


class _SupervisedView:
    """A map-style ``torch`` dataset over per-sample AMC ``Batch`` dicts.

    Wraps the object returned by :meth:`Dataset.load` (a map-style dataset of per-sample dicts
    with ``iq`` / ``label``) into ``(iq, label)`` pairs a ``DataLoader`` can batch. Kept a plain
    class (not subclassing ``torch.utils.data.Dataset`` at module top) so the module import
    stays torch-free; ``DataLoader`` only needs ``__len__`` + ``__getitem__``.
    """

    def __init__(self, source: MapDataset) -> None:
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self._source[index]
        return sample["iq"], int(sample["label"])


def _make_collate(device: str) -> CollateFn:
    """Return a ``collate_fn`` stacking ``(iq, label)`` pairs into batched tensors on ``device``.

    Each ``iq`` payload is a ``(2, L)`` array-like (numpy on the cluster, nested lists in a
    fixture); ``torch.as_tensor`` handles both. Produces ``(x (B, 2, L) float32, y (B,) long)``.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def collate(pairs: list[tuple[Any, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        iqs = [iq for iq, _ in pairs]
        labels = [label for _, label in pairs]
        # Stack into one ndarray first: torch.as_tensor over a LIST of ndarrays is very slow
        # (and warns). np.asarray handles both real numpy (2, L) arrays and nested-list fixtures.
        x = torch.as_tensor(np.asarray(iqs, dtype=np.float32), device=device)
        if x.ndim == 2:  # a single (2, L) sample slipped through -> add the batch axis
            x = x.unsqueeze(0)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        return x, y

    return collate


def _make_loader(
    source: MapDataset,
    *,
    batch_size: int,
    device: str,
    num_workers: int,
    shuffle: bool,
) -> DataLoaderT:
    """Build a ``DataLoader`` over ``source`` with the shared IQ collate on ``device``.

    Factored out so the train and val splits share one construction path; ``shuffle`` is on for
    train (SGD needs it) and off for val (a deterministic pass for a stable monitoring signal).
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415 - lazy by design

    return DataLoader(
        _SupervisedView(source),
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
) -> float:
    """Run one optimisation pass over ``loader`` and return the mean per-batch train loss."""
    module.train()
    running = 0.0
    n_batches = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(module(x), y)
        loss.backward()
        optimizer.step()
        running += float(loss.detach().item())
        n_batches += 1
    return running / n_batches if n_batches else float("nan")


def _mean_accuracy(
    module: nn.Module, loader: DataLoaderT, criterion: CriterionT
) -> tuple[float, float]:
    """Return ``(mean loss, top-1 accuracy)`` of ``module`` over ``loader``, no gradient updates.

    The accuracy uses the SAME argmax/label-decoding convention as
    :func:`rfbench.core.evaluate.evaluate` -> ``rfbench.tasks.amc.metrics``: the predicted class
    is the ``argmax`` over the per-class logits axis (dim 1 of the ``(B, num_classes)`` model
    output) and the true class is the integer label, so val and test agree on what "correct" means.
    Both scalars are ``nan`` when the loader is empty, so the caller can fall back to the train
    signal. Computed in one pass so the LR schedule keeps stepping on val LOSS while checkpoint
    selection / early stopping key on val ACCURACY.
    """
    import torch  # noqa: PLC0415 - lazy by design

    module.eval()
    running_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            outputs = module(x)
            running_loss += float(criterion(outputs, y).detach().item())
            n_batches += 1
            predicted = outputs.argmax(dim=1)
            correct += int((predicted == y).sum().item())
            total += int(y.numel())
    if n_batches == 0 or total == 0:
        return float("nan"), float("nan")
    return running_loss / n_batches, correct / total


def _snapshot_state(module: nn.Module) -> dict[str, Any]:
    """Return a detached CPU copy of ``module``'s ``state_dict`` (the best-val checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_val_source(dataset: Dataset, val_split: SplitName) -> MapDataset | None:
    """Return the ``val_split`` map-dataset, or ``None`` when it is unavailable/empty.

    The AMC on-disk loader raises (``FileNotFoundError`` / ``KeyError``) when no ``val`` index was
    prepared, and a synthetic fixture may expose an empty val split; either way we return ``None``
    so :func:`train_baseline` degrades gracefully to monitoring the train loss instead of crashing.
    """
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


def train_baseline(
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
) -> tuple[Model, dict[str, Any]]:
    """Fit ``model`` on ``dataset``'s TRAIN split with val-monitoring, then ``evaluate`` on TEST.

    Runs the standard AMC training recipe for the ``from_scratch`` / ``full_finetune`` regimes
    (DataLoader over ``dataset.load('train')``, Adam + cross-entropy), MONITORING the ``val_split``
    each epoch to:

    * drive a ``torch.optim.lr_scheduler.ReduceLROnPlateau`` on the val LOSS
      (``mode="min"``, ``factor=lr_factor``, ``patience=lr_patience``, ``min_lr=min_lr``),
    * keep the BEST-VAL-ACCURACY ``state_dict`` and RESTORE it before the final evaluation, and
    * EARLY-STOP once ``patience`` epochs pass with no val-ACCURACY improvement greater than
      ``min_delta``.

    Checkpoint selection and early stopping key on val ACCURACY (higher is better) using the same
    argmax/label decoding as :func:`rfbench.core.evaluate.evaluate`, so the restored checkpoint is
    the accuracy-peak epoch (on RadioML the CE-loss minimum precedes the accuracy peak, so a
    loss-min checkpoint restored a suboptimal model). The LR schedule still steps on val LOSS.

    When the ``val_split`` is unavailable or empty (e.g. an on-disk dataset without a prepared
    ``val`` index, or a single-split fixture) the loop degrades gracefully to monitoring the
    *train* loss instead (loss-min checkpoint selection) -- the schedule, early stopping and
    best-checkpoint logic all still run.

    After fitting, restores the best state and calls :func:`rfbench.core.evaluate.evaluate` on
    ``task``'s default (``test``) split with the same declared ``regime`` so the single canonical
    writer emits (and, if ``out_path`` is set, writes) a schema-valid ``result.json``.

    ``epochs`` is now an *upper bound* (max epochs); early stopping usually stops sooner. The
    optional recipe params (``patience``, ``min_delta``, ``lr_factor``, ``lr_patience``,
    ``min_lr``, ``val_split``) default to the standard AMC values so existing callers are
    unaffected. Note ``min_delta`` now applies to val ACCURACY (an accuracy gain, not a loss
    drop). Returns ``(trained_model, result_dict)``. Raises ``ValueError`` if ``regime`` is not a
    trainable regime, or ``epochs``/``batch_size``/``patience`` are non-positive.
    """
    import torch  # noqa: PLC0415

    from rfbench.core.evaluate import evaluate  # noqa: PLC0415

    if regime.name not in TRAINABLE_REGIMES:
        raise ValueError(
            f"train_baseline fits only {[r.value for r in TRAINABLE_REGIMES]} regimes; "
            f"got {regime.name.value!r} (probing regimes go through rfbench.regimes)."
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
            "no usable '%s' split for %r; monitoring TRAIN loss for the LR schedule / early "
            "stopping / best-checkpoint restore instead (val-accuracy selection unavailable).",
            val_split,
            dataset.name,
        )

    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    # Checkpoint selection / early stopping key on val ACCURACY (higher is better) when a val
    # loader exists; with no val split we degrade to the train-LOSS signal (lower is better). The
    # LR schedule always steps on a LOSS (val loss when available, else train loss).
    best_acc = -1.0  # any real accuracy in [0, 1] beats this -> epoch 0 always snapshots
    best_loss = float("inf")  # only used in the no-val (train-loss) fallback branch
    best_state = _snapshot_state(module)
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        train_loss = _train_one_epoch(module, train_loader, optimizer, criterion)

        if val_loader is not None:
            val_loss, val_acc = _mean_accuracy(module, val_loader, criterion)
            # An empty/degenerate val signal (nan) must not poison the schedule or selection: fall
            # back to the train loss for the plateau step and skip the (nan) accuracy improvement.
            step_loss = train_loss if val_loss != val_loss else val_loss  # NaN check w/o math
            scheduler.step(step_loss)
            improved = val_acc == val_acc and val_acc > best_acc + min_delta
            if improved:
                best_acc = val_acc
                best_state = _snapshot_state(module)
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            logger.info(
                "epoch %d/%d: train loss = %.4f, val loss = %.4f, val acc = %.4f, lr = %.2e%s",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_acc,
                optimizer.param_groups[0]["lr"],
                " (best)" if improved else "",
            )
        else:
            # No usable val split: monitor the TRAIN loss (lower is better) for both the LR
            # plateau and the best-checkpoint / early-stop decision (accuracy analog unavailable).
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
            if val_loader is not None:
                logger.info(
                    "early stopping at epoch %d/%d (no val-accuracy improvement for %d epochs; "
                    "best epoch %d, best val acc %.4f)",
                    epoch + 1,
                    epochs,
                    patience,
                    best_epoch + 1,
                    best_acc,
                )
            else:
                logger.info(
                    "early stopping at epoch %d/%d (no train-loss improvement for %d epochs; "
                    "best epoch %d, best train loss %.4f)",
                    epoch + 1,
                    epochs,
                    patience,
                    best_epoch + 1,
                    best_loss,
                )
            break

    # Restore the best checkpoint before the ONE canonical test evaluation.
    module.load_state_dict(best_state)
    module.to(resolved_device)
    module.eval()
    if val_loader is not None:
        logger.info(
            "restored best-val-accuracy checkpoint from epoch %d (val acc %.4f)",
            best_epoch + 1,
            best_acc,
        )
    else:
        logger.info(
            "restored best-train-loss checkpoint from epoch %d (train loss %.4f)",
            best_epoch + 1,
            best_loss,
        )

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


def _sync_model_device(model: Model, device: str) -> None:
    """Best-effort: update a wrapper model's cached ``device`` so its ``forward`` matches.

    Wrapper baselines (e.g. MCLDNN) cache a ``torch.device`` used inside ``forward`` to move
    the input batch onto the network's device. After :func:`resolve_module` moves the module,
    keep that cache in step so evaluation runs on the same device the module was trained on. A
    model without a ``device`` attribute (e.g. a plain ``nn.Module`` model) is left untouched.
    """
    import torch  # noqa: PLC0415

    if hasattr(model, "device"):
        try:
            model.device = torch.device(device)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass


def resolve_amc_dataset(task: Task, dataset_name: str) -> Dataset:
    """Return the ``task`` dataset variant whose ``name`` matches ``dataset_name``.

    Mirrors ``evaluate``'s dataset resolution so ``rfbench train`` and ``evaluate`` agree on
    which variant is scored. Raises ``ValueError`` naming the available datasets on a miss.
    """
    for candidate in task.datasets():
        if candidate.name == dataset_name:
            return candidate
    available = ", ".join(sorted(ds.name for ds in task.datasets())) or "<none>"
    raise ValueError(
        f"unknown dataset {dataset_name!r} for task {task.name!r}; available: {available}"
    )


__all__ = [
    "train_baseline",
    "resolve_device",
    "resolve_module",
    "resolve_amc_dataset",
    "TRAINABLE_REGIMES",
    "DEFAULT_VAL_SPLIT",
    "DEFAULT_PATIENCE",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_LR_FACTOR",
    "DEFAULT_LR_PATIENCE",
    "DEFAULT_MIN_LR",
]
