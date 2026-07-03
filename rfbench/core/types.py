"""Shared literal types for the ``rfbench`` core contracts.

These aliases keep the Python code in lockstep with the JSON schemas
(``schemas/result.schema.json`` and ``schemas/submission.schema.json``). Whenever
an ``enum`` in a schema changes, the corresponding ``Literal`` here changes too, so
that a mismatch surfaces at type-check time rather than at validation time.

This module has **no third-party imports** so that ``import rfbench.core`` stays
dependency-free (no ``torch``/``jsonschema`` at import time).
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

#: Alias for a runtime tensor (``torch.Tensor`` in practice). Declared as ``Any`` so
#: that the core contracts never import ``torch`` at module top and
#: ``import rfbench.core`` stays dependency-free. Concrete task/model code annotates
#: with the real ``torch.Tensor``; the contracts only need the structural shape.
Tensor: TypeAlias = Any

#: A canonical batch: a mapping of field name to tensor (e.g. ``{"iq": ...,
#: "label": ..., "snr_db": ...}``). Kept as ``dict[str, Any]`` for the same reason.
Batch: TypeAlias = dict[str, Any]

#: Registered benchmark task ids (``result.json.task.name`` enum).
TaskName = Literal[
    "amc", "sei", "wideband_detection", "spectrum_sensing", "interference_id", "protocol_tech_id"
]

#: The four locked adaptation regimes (D5, ``result.json.regime.name`` enum).
RegimeName = Literal["from_scratch", "full_finetune", "linear_probe", "few_shot"]

#: Two-tier verification state (``result.json.verification.status`` enum).
VerificationStatus = Literal["self_reported", "verified"]

#: How a maintainer reproduced a run (``verification.method`` / ``rerun_mode``).
RerunMode = Literal["eval_only", "full_retrain"]

#: Partition of a canonical split. ``split.name`` in the schema is ``{test, val}``
#: only; ``train`` exists here because it is needed to *fit* models even though it
#: is never itself scored on the board.
SplitName = Literal["train", "val", "test"]

#: Per-task evaluation track / condition. Intentionally an open ``str`` to match the
#: free-form ``split.track`` field: SEI uses ``closed_set|cross_receiver|cross_day|
#: open_set``, wideband detection uses ``detection|recognition``, and Wave-B tasks
#: may add tracks without any schema bump.
Track = str

__all__ = [
    "Tensor",
    "Batch",
    "TaskName",
    "RegimeName",
    "VerificationStatus",
    "RerunMode",
    "SplitName",
    "Track",
]
