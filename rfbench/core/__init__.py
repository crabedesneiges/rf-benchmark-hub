"""Frozen core contracts for RF-Benchmark-Hub (WP-02).

This package exposes the abstract base classes and value objects that every task,
model and metric plugs into: :class:`Task`, :class:`Dataset`, :class:`Metric`,
:class:`Model` (+ :class:`Regime` / :class:`RegimeSpec`), the split and manifest value
objects, the registries, and the single canonical :func:`evaluate` writer.

These signatures are frozen at M0 and must not change without an explicit review and a
version bump (see ``CLAUDE.md`` and ``docs/ARCHITECTURE.md``). The package is kept
**dependency-free at import time**: no ``torch`` or ``jsonschema`` is imported at module
top, so ``import rfbench.core`` works in any environment. Concrete implementations
import those libraries lazily inside their function bodies.
"""

from __future__ import annotations

from rfbench.core.dataset import Dataset
from rfbench.core.evaluate import evaluate
from rfbench.core.manifest import (
    DataProvenance,
    DatasetManifest,
    provenance_of,
    verify_manifest,
    write_manifest,
)
from rfbench.core.metric import Metric
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import (
    METRICS,
    MODELS,
    TASKS,
    Registry,
    get_task,
    register_metric,
    register_model,
    register_task,
)
from rfbench.core.splits import (
    SplitManifest,
    make_split,
    split_checksum,
    write_split_index,
)
from rfbench.core.task import Task
from rfbench.core.types import (
    Batch,
    RegimeName,
    RerunMode,
    SplitName,
    TaskName,
    Tensor,
    Track,
    VerificationStatus,
)

__all__ = [
    # Contracts (ABCs)
    "Task",
    "Dataset",
    "Metric",
    "Model",
    # Regime
    "Regime",
    "RegimeSpec",
    # Splits
    "SplitManifest",
    "make_split",
    "write_split_index",
    "split_checksum",
    # Manifest / provenance
    "DatasetManifest",
    "DataProvenance",
    "write_manifest",
    "verify_manifest",
    "provenance_of",
    # Registry
    "Registry",
    "TASKS",
    "MODELS",
    "METRICS",
    "register_task",
    "register_model",
    "register_metric",
    "get_task",
    # The single canonical writer
    "evaluate",
    # Shared types
    "Tensor",
    "Batch",
    "TaskName",
    "RegimeName",
    "VerificationStatus",
    "RerunMode",
    "SplitName",
    "Track",
]
