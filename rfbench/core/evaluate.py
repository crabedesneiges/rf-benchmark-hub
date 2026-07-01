"""``evaluate()`` -- the ONLY writer of ``result.json``.

Every leaderboard row is produced here and nowhere else. :func:`evaluate` runs the
eval loop, aggregates ``task.metrics()``, assembles a result dict that VALIDATES
against ``schemas/result.schema.json`` (jsonschema, Draft 2020-12), optionally writes
it, and returns it. An invalid result never leaves the harness: a schema failure
raises ``jsonschema.ValidationError``.

Contract invariants (enforced by the implementation, WP-40):

* ``regime.name`` is written VERBATIM into ``result["regime"]`` -- ALWAYS declared,
  never inferred (D5); ``result["regime"]["k_shot"]`` is present iff the regime is
  ``few_shot``.
* ``result["metrics"]["primary"]`` equals the task's primary ``Metric.primary_key`` and
  appears as a key of ``result["metrics"]["values"]``.
* ``result["verification"]["status"]`` is initialised to ``"self_reported"``; only
  ``rfbench verify`` may flip it to ``"verified"``.
* The split identity (``canonical_split_id``, ``name``, ``seed``, ``checksum`` and the
  optional ``track``) is copied from the :class:`~rfbench.core.dataset.Dataset`.
* ``eval.conditions`` records the full-protocol conditions (AMC: the full SNR range).

``jsonschema`` and ``torch`` are imported lazily inside the function so that
``import rfbench.core`` stays dependency-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rfbench.core.model import Model, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import SplitName, Track


def evaluate(
    model: Model,
    task: Task,
    split: SplitName,
    regime: RegimeSpec,
    *,
    dataset: str | None = None,
    track: Track | None = None,
    seed: int = 42,
    batch_size: int = 256,
    device: str = "cuda",
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run the eval loop and emit a schema-valid ``result.json`` dict.

    Aggregates ``task.metrics()`` over ``split`` (optionally restricted to ``track``),
    assembles the result dict, validates it against ``schemas/result.schema.json`` with
    ``jsonschema`` (Draft 2020-12), writes ``out_path`` if given, and returns the dict.

    Raises ``jsonschema.ValidationError`` on schema failure so an invalid result never
    leaves the harness. See the module docstring for the full list of contract
    invariants this implementation guarantees.

    Implemented by WP-40 (``evaluate()`` + ``result.json``).
    """
    raise NotImplementedError("evaluate is implemented in WP-40 (eval harness)")


__all__ = ["evaluate"]
