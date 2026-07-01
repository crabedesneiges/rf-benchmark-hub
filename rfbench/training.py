"""Real training loop for the from-scratch / full-finetune regimes (WP-30).

:func:`train_baseline` is the M3 training entry point the pass-through regimes
(``from_scratch`` / ``full_finetune``) delegate their *fit* to. It:

1. resolves the underlying ``nn.Module`` of a :class:`~rfbench.core.model.Model`,
2. streams the task's ``train`` split through a ``torch.utils.data.DataLoader``,
3. optimises it with Adam + cross-entropy for ``epochs`` epochs (AMC is single-label
   closed-set classification), moving everything to CUDA when available, then
4. hands the trained model to :func:`rfbench.core.evaluate.evaluate` on the ``test`` split so
   the ONE canonical writer emits a schema-valid ``result.json`` with the regime declared.

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

#: The regimes ``train_baseline`` actually fits. ``linear_probe`` / ``few_shot`` are adapter
#: regimes (a head fit on a frozen backbone) and go through ``rfbench.regimes`` instead.
TRAINABLE_REGIMES = (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE)


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
) -> tuple[Model, dict[str, Any]]:
    """Fit ``model`` on ``dataset``'s TRAIN split, then ``evaluate`` it on TEST.

    Runs a real torch training loop (DataLoader over ``dataset.load('train')``,
    cross-entropy, Adam, ``epochs`` epochs) for the ``from_scratch`` / ``full_finetune``
    regimes, moving the model + batches to CUDA when available. After fitting, calls
    :func:`rfbench.core.evaluate.evaluate` on ``task``'s default (``test``) split with the
    same declared ``regime`` so the single canonical writer emits (and, if ``out_path`` is
    set, writes) a schema-valid ``result.json``.

    Returns ``(trained_model, result_dict)``. Raises ``ValueError`` if ``regime`` is not a
    trainable regime, or ``epochs``/``batch_size`` are non-positive.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

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

    resolved_device = resolve_device(device)
    _seed_everything(seed)

    module = resolve_module(model).to(resolved_device)
    _sync_model_device(model, resolved_device)

    train_source = dataset.load("train", None)
    loader = DataLoader(
        _SupervisedView(train_source),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_make_collate(resolved_device),
    )

    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    module.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = module(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().item())
            n_batches += 1
        mean_loss = running / n_batches if n_batches else float("nan")
        logger.info("epoch %d/%d: mean train loss = %.4f", epoch + 1, epochs, mean_loss)

    module.eval()

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
]
