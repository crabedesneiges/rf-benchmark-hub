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

import json
import os
import platform
import tempfile
from collections.abc import Iterable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.model import Model, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Track

#: SemVer of the result schema this writer targets (mirrors ``schema_version.const``).
SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------------------------------
# Schema resolution + validation (mirrors ``rfbench.cli._resolve_schema_path``)
# --------------------------------------------------------------------------------------------------
def _resolve_schema_path(schema_name: str) -> Path | None:
    """Locate a JSON schema file.

    Resolution order matches the packaging contract used by the CLI: prefer the
    force-included package data (``rfbench/_schemas`` in an installed wheel), then fall
    back to the repo ``schemas/`` directory when running from a source checkout. Returns
    ``None`` if neither location has the file.
    """
    try:
        packaged = resources.files("rfbench").joinpath("_schemas").joinpath(schema_name)
        if packaged.is_file():
            with resources.as_file(packaged) as concrete:
                return concrete
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schemas" / schema_name
        if candidate.is_file():
            return candidate
    return None


def _validate_or_raise(document: dict[str, Any]) -> None:
    """Validate ``document`` against ``result.schema.json`` or raise.

    ``jsonschema`` is imported lazily so ``import rfbench.core`` stays dependency-free.
    On a schema failure the underlying ``jsonschema.ValidationError`` is raised so an
    invalid result never leaves the harness; a missing schema/library raises
    ``RuntimeError`` with a clear message.
    """
    schema_path = _resolve_schema_path("result.schema.json")
    if schema_path is None:
        raise RuntimeError(
            "could not locate result.schema.json (checked rfbench/_schemas and repo schemas/)"
        )
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError as exc:  # pragma: no cover - jsonschema is a hard dep of the harness
        raise RuntimeError(
            "jsonschema is required to validate result.json; install rfbench with its deps"
        ) from exc

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(document)


# --------------------------------------------------------------------------------------------------
# Eval loop helpers (pure stdlib: no torch/numpy import at module top)
# --------------------------------------------------------------------------------------------------
def _iter_batches(data: Iterable[Batch], batch_size: int) -> Iterator[list[Batch]]:
    """Yield lists of at most ``batch_size`` per-sample ``Batch`` dicts.

    Iterates the loaded dataset in order (deterministic given a deterministic dataset)
    and groups samples into fixed-size chunks without importing any tensor framework.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    chunk: list[Batch] = []
    for sample in data:
        chunk.append(sample)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _collate(samples: list[Batch]) -> Batch:
    """Collate a list of per-sample ``Batch`` dicts into a batch-of-lists ``Batch``.

    Each field maps to the list of that field's per-sample values, so the batch keeps
    the canonical field layout (``{"iq": [...], "label": [...], "snr_db": [...]}``)
    without depending on ``torch``'s default collate.
    """
    if not samples:
        return {}
    keys = samples[0].keys()
    return {key: [sample[key] for sample in samples] for key in keys}


def _resolve_dataset(task: Task, dataset: str | None) -> Dataset:
    """Return the task dataset to evaluate on.

    Picks the dataset whose ``name`` matches ``dataset`` (if given), else the task's
    first declared dataset. Raises ``ValueError`` naming the available datasets on a
    miss or when the task declares none.
    """
    candidates = task.datasets()
    if not candidates:
        raise ValueError(f"task '{task.name}' declares no datasets to evaluate on")
    if dataset is None:
        return candidates[0]
    for candidate in candidates:
        if candidate.name == dataset:
            return candidate
    available = ", ".join(sorted(ds.name for ds in candidates))
    raise ValueError(f"unknown dataset '{dataset}' for task '{task.name}'; available: {available}")


def _primary_metric(task: Task) -> Metric:
    """Return the task's primary metric (the first declared).

    The primary metric's ``primary_key`` names ``result.metrics.primary`` and ranks the
    board. Raises ``ValueError`` if the task declares no metrics.
    """
    metrics = task.metrics()
    if not metrics:
        raise ValueError(f"task '{task.name}' declares no metrics")
    return metrics[0]


def _partition_metric_output(
    computed: dict[str, float | list[dict[str, float]]],
) -> tuple[dict[str, float], dict[str, list[dict[str, float]]]]:
    """Split a ``Metric.compute()`` mapping into scalar ``values`` and ``curves``.

    Scalars (``int``/``float``, excluding ``bool``) land in ``metrics.values``; list
    payloads (``[{"x": .., "y": ..}, ...]``) land in ``metrics.curves``. Any other value
    type raises ``TypeError`` so a malformed metric fails loudly before validation.
    """
    values: dict[str, float] = {}
    curves: dict[str, list[dict[str, float]]] = {}
    for key, payload in computed.items():
        if isinstance(payload, bool):
            raise TypeError(f"metric '{key}' produced a bool; scalars must be numeric")
        if isinstance(payload, (int, float)):
            values[key] = float(payload)
        elif isinstance(payload, list):
            curves[key] = payload
        else:
            raise TypeError(
                f"metric '{key}' produced unsupported type {type(payload).__name__}; "
                "expected a scalar or a list of curve points"
            )
    return values, curves


def _environment_fingerprint(seed: int) -> dict[str, Any]:
    """Assemble the reproducibility fingerprint from ``platform``/``sys`` (stdlib only).

    Deterministic within a machine/interpreter: seed plus Python version and rfbench
    version. Hardware is recorded only from ``$RFBENCH_HARDWARE`` when set, and the
    framework string is added lazily only if ``torch`` happens to be importable, so the
    dependency-free path stays fully deterministic for tests.
    """
    from rfbench import __version__

    env: dict[str, Any] = {
        "seed": seed,
        "rfbench_version": __version__,
        "python_version": platform.python_version(),
    }
    hardware = os.environ.get("RFBENCH_HARDWARE")
    if hardware:
        env["hardware"] = hardware
    try:  # optional enrichment; never a hard dependency
        import torch  # noqa: PLC0415

        env["framework"] = f"torch {torch.__version__}"
    except ModuleNotFoundError:
        pass
    return env


# --------------------------------------------------------------------------------------------------
# The single canonical result.json writer
# --------------------------------------------------------------------------------------------------
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
    resolved = _resolve_dataset(task, dataset)
    metrics = task.metrics()
    primary = _primary_metric(task)

    # --- Eval loop: stream the split through every metric ---------------------------------------
    for metric in metrics:
        metric.reset()

    data = resolved.load(split, track)
    n_samples = 0
    for chunk in _iter_batches(data, batch_size):
        batch = _collate(chunk)
        target = task.build_targets(batch)
        pred = model.forward(batch)
        for metric in metrics:
            metric.update(pred, target, batch)
        n_samples += len(chunk)

    # --- Aggregate: scalars -> values, lists -> curves ------------------------------------------
    values: dict[str, float] = {}
    curves: dict[str, list[dict[str, float]]] = {}
    for metric in metrics:
        metric_values, metric_curves = _partition_metric_output(metric.compute())
        values.update(metric_values)
        curves.update(metric_curves)

    if primary.primary_key not in values:
        raise ValueError(
            f"primary metric key '{primary.primary_key}' is absent from the computed "
            f"metrics.values (got: {', '.join(sorted(values)) or '<none>'})"
        )

    # --- Assemble the result document (regime declared VERBATIM, never inferred) ----------------
    regime_block: dict[str, Any] = {"name": regime.name.value}
    if regime.k_shot is not None:
        regime_block["k_shot"] = regime.k_shot

    split_block: dict[str, Any] = {
        "canonical_split_id": resolved.canonical_split_id,
        "name": split,
        "seed": seed,
        "checksum": resolved.checksum,
    }
    if track is not None:
        split_block["track"] = track

    metrics_block: dict[str, Any] = {"primary": primary.primary_key, "values": values}
    if curves:
        metrics_block["curves"] = curves

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": {"name": task.name, "version": task.version},
        "model": {"name": model.name, "family": model.family, "n_params": model.n_params},
        "regime": regime_block,
        "dataset": {"name": resolved.name},
        "split": split_block,
        "metrics": metrics_block,
        "eval": {
            "conditions": _eval_conditions(metrics),
            "n_samples": n_samples,
            "batch_size": batch_size,
        },
        "environment": _environment_fingerprint(seed),
        "verification": {"status": "self_reported"},
    }

    # --- Validate BEFORE anything leaves the harness --------------------------------------------
    _validate_or_raise(result)

    if out_path is not None:
        _atomic_write_json(result, out_path)

    return result


def _eval_conditions(metrics: Iterable[Metric]) -> dict[str, Any]:
    """Collect full-protocol ``eval.conditions`` reported by the task's metrics.

    A metric MAY expose an ``eval_conditions() -> dict`` hook to record the full-protocol
    guard rails (AMC: the full SNR range, ``full_snr_range: true``). Absent that hook, an
    empty open map is returned; the field itself always exists so the row records that
    the full protocol was run.
    """
    conditions: dict[str, Any] = {}
    for metric in metrics:
        hook = getattr(metric, "eval_conditions", None)
        if callable(hook):
            reported = hook()
            if isinstance(reported, dict):
                conditions.update(reported)
    return conditions


def _atomic_write_json(document: dict[str, Any], out_path: Path) -> None:
    """Write ``document`` to ``out_path`` atomically with sorted keys.

    Serialises to a temp file in the destination directory, then ``os.replace`` swaps it
    into place so a reader never observes a partial file. ``sort_keys=True`` keeps the
    on-disk artifact byte-stable for a given result dict.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=out_path.parent, prefix=out_path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, out_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


__all__ = ["evaluate"]
