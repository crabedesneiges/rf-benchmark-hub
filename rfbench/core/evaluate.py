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
import random
import tempfile
from collections.abc import Iterable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from rfbench.core.dataset import Dataset
from rfbench.core.metric import Metric
from rfbench.core.model import Model, RegimeSpec
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track

#: SemVer of the result schema this writer targets (mirrors ``schema_version.const``).
#: 1.2.0 adds the optional ``metrics.uncertainty`` block (percentile-bootstrap CIs).
SCHEMA_VERSION = "1.2.0"

#: Bootstrap defaults (EVALUATION_PROTOCOL.md "Statistical rigor & uncertainty"): a
#: percentile bootstrap over the accumulated per-sample predictions.
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
#: Below this many accumulated samples a bootstrap CI is statistically meaningless, so we
#: skip it rather than emit a noisy interval.
BOOTSTRAP_MIN_SAMPLES = 2


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
# Bootstrap confidence intervals (percentile bootstrap, pure stdlib random)
# --------------------------------------------------------------------------------------------------
def _concat(prev: Tensor, nxt: Tensor) -> Tensor:
    """Concatenate two dim-0-aligned chunks, preserving container type.

    Used to grow the accumulated ``pred``/``target``/``meta`` across eval chunks: torch
    tensors are ``torch.cat``-ed (lazy import), collated ``{field: [...]}`` dicts merge
    field-wise, and lists/tuples concatenate. ``prev is None`` means "first chunk", so
    ``nxt`` is adopted as-is.
    """
    if prev is None:
        return nxt
    try:
        import torch  # noqa: PLC0415

        if isinstance(prev, torch.Tensor) and isinstance(nxt, torch.Tensor):
            return torch.cat([prev, nxt], dim=0)
    except ModuleNotFoundError:
        pass
    if isinstance(prev, dict) and isinstance(nxt, dict):
        return {key: _concat(prev.get(key), nxt.get(key)) for key in {*prev, *nxt}}
    if isinstance(prev, list):
        return prev + list(nxt)
    if isinstance(prev, tuple):
        return prev + tuple(nxt)
    # Unknown / scalar payload: fall back to a plain list so resampling still works.
    return [prev, nxt]


class _BootstrapAccumulator:
    """Grows whole ``pred``/``target``/``meta`` across eval chunks for later resampling.

    Kept minimal and type-agnostic: :func:`_concat` handles tensors, collated meta dicts
    and lists identically, so the accumulator never needs to know the metric's payload
    shape. Only instantiated when ``compute_bootstrap_ci`` is on.
    """

    def __init__(self) -> None:
        self.pred: Tensor = None
        self.target: Tensor = None
        self.meta: Batch | None = None

    def add(self, pred: Tensor, target: Tensor, meta: Batch) -> None:
        self.pred = _concat(self.pred, pred)
        self.target = _concat(self.target, target)
        self.meta = _concat(self.meta, meta)


def _index_select(obj: Tensor, indices: list[int]) -> Tensor:
    """Return ``obj`` gathered along dim 0 by ``indices``, preserving its container type.

    Type-preserving gather used to build bootstrap resamples of a per-sample-batched
    ``pred``/``target``/``meta`` (see :func:`_collate`) without importing a tensor
    framework at module top:

    * a ``torch.Tensor`` -> ``Tensor.index_select`` along dim 0 (torch is imported lazily
      and only when a tensor is actually seen);
    * a ``dict`` (a collated ``{field: [values...]}`` meta batch) -> each value is gathered
      recursively, so the batch keeps its field layout;
    * any other sequence (``list``/``tuple``) -> a plain comprehension.

    ``indices`` may repeat entries (sampling WITH replacement), which the list and tensor
    paths both honour. Scalar/unindexable payloads are returned untouched.
    """
    # Lazy tensor path: only import torch if we are actually holding a tensor, so the
    # dependency-free import contract of ``rfbench.core`` is preserved.
    try:
        import torch  # noqa: PLC0415

        if isinstance(obj, torch.Tensor):
            index = torch.as_tensor(indices, dtype=torch.long, device=obj.device)
            return obj.index_select(0, index)
    except ModuleNotFoundError:
        pass

    if isinstance(obj, dict):
        return {key: _index_select(value, indices) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        gathered = [obj[i] for i in indices]
        return type(obj)(gathered) if isinstance(obj, tuple) else gathered
    return obj


def _accumulated_length(pred: Tensor, target: Tensor) -> int:
    """Best-effort per-sample count of an accumulated ``pred``/``target`` pair.

    Prefers ``len(pred)``, falling back to ``len(target)``; returns 0 when neither is a
    sized sequence/tensor (in which case bootstrap is silently skipped).
    """
    for candidate in (pred, target):
        try:
            return len(candidate)
        except TypeError:
            continue
    return 0


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Return the ``quantile`` (in [0, 1]) of a pre-sorted list via linear interpolation.

    Matches the common "linear"/type-7 percentile so the interval is stable and does not
    depend on numpy. ``sorted_values`` must be non-empty and ascending.
    """
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _bootstrap_uncertainty(
    metrics: Iterable[Metric],
    pred: Tensor,
    target: Tensor,
    meta: Batch | None,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Percentile-bootstrap CIs for every scalar metric, keyed by metric name.

    Resamples the accumulated per-sample ``(pred, target, meta)`` with replacement
    ``n_resamples`` times, re-running each metric's ``reset()/update()/compute()`` on the
    resample and collecting its scalar outputs, then reports the two-sided percentile
    interval at the given ``confidence``. Randomness is stdlib ``random.Random`` seeded per
    resample from ``seed`` so the interval is reproducible; NO numpy/torch RNG is used.

    Returns a mapping ``{metric_name: {ci_low, ci_high, method, confidence, n_resamples}}``
    ready to drop into ``result["metrics"]["uncertainty"]``. Returns an empty mapping when
    there are too few samples for a meaningful interval.
    """
    n = _accumulated_length(pred, target)
    if n < BOOTSTRAP_MIN_SAMPLES or n_resamples < 1:
        return {}

    metric_list = list(metrics)
    # Pre-reduce each metric's predictions ONCE via the optional ``prepare_predictions`` hook.
    # Bootstrap re-runs ``update`` ``n_resamples`` times, so if a metric derives its per-sample
    # scalar from an expensive representation (e.g. the open-set per-class rows -> max-softmax
    # score) leaving ``pred`` raw would recompute that reduction n*resamples times and stall the
    # eval (the SEI open-set 144k-probe row reduction is ~O(n*classes) per resample). Reducing up
    # front makes each resample a cheap gather + scalar update. Metrics without the hook keep the
    # raw accumulated ``pred`` unchanged, so their CIs are byte-identical to before.
    prepared_pred: list[Tensor] = []
    for metric in metric_list:
        prepare = getattr(metric, "prepare_predictions", None)
        prepared_pred.append(prepare(pred) if callable(prepare) else pred)

    samples: dict[str, list[float]] = {}
    for i in range(n_resamples):
        rng = random.Random(seed + i)
        idx = [rng.randrange(n) for _ in range(n)]
        target_bs = _index_select(target, idx)
        meta_bs = _index_select(meta, idx) if meta is not None else None
        for metric, metric_pred in zip(metric_list, prepared_pred, strict=True):
            pred_bs = _index_select(metric_pred, idx)
            metric.reset()
            metric.update(pred_bs, target_bs, meta_bs)
            computed = metric.compute()
            for key, payload in computed.items():
                if isinstance(payload, bool) or not isinstance(payload, (int, float)):
                    continue  # curves / non-scalars carry no CI
                samples.setdefault(key, []).append(float(payload))

    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    uncertainty: dict[str, dict[str, Any]] = {}
    for key, draws in samples.items():
        if len(draws) < BOOTSTRAP_MIN_SAMPLES:
            continue
        draws.sort()
        uncertainty[key] = {
            "ci_low": _percentile(draws, lo_q),
            "ci_high": _percentile(draws, hi_q),
            "method": "bootstrap_percentile",
            "confidence": confidence,
            "n_resamples": n_resamples,
        }
    return uncertainty


# --------------------------------------------------------------------------------------------------
# Per-bin curve confidence bands (percentile bootstrap, stratified WITHIN each bin)
# --------------------------------------------------------------------------------------------------
def _bootstrap_curve_bands(
    metric: Metric,
    curve_name: str,
    points: list[dict[str, float]],
    pred: Tensor,
    target: Tensor,
    meta: Batch | None,
    bins: list[float],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Attach a per-bin percentile-bootstrap ``y_low``/``y_high`` band to ``points``.

    For every curve bin the samples belonging to that bin are resampled WITH replacement
    ``n_resamples`` times (stratified WITHIN the bin: a fixed draw count equal to the bin's
    observed size, so the interval is the sampling spread of *that bin's* value and does not
    borrow strength from the other bins). Each resample re-runs
    ``metric.reset()/update()/compute()`` on the bin's resampled slice and reads the bin's
    ``y`` back from the re-emitted curve; the two-sided percentile interval at ``confidence``
    becomes ``y_low``/``y_high`` on the matching point. Bounds are clamped to ``[0, 1]``
    (these curves are accuracies/rates). The point's own ``x``/``y``/``label`` are preserved
    verbatim.

    Randomness is stdlib ``random.Random`` seeded from ``seed`` (per bin + per resample) so
    the band is reproducible and stable under bin re-ordering; NO numpy/torch RNG is used.
    Bins with fewer than ``BOOTSTRAP_MIN_SAMPLES`` members, or points the metric does not
    re-emit, are passed through with no band.
    """
    # Group accumulated per-sample indices by their bin key. ``bins[i]`` is the bin of the
    # i-th accumulated sample, aligned with ``pred``/``target``/``meta``.
    members: dict[float, list[int]] = {}
    for index, bin_value in enumerate(bins):
        members.setdefault(float(bin_value), []).append(index)

    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    banded: list[dict[str, Any]] = []
    for point in points:
        out_point: dict[str, Any] = dict(point)  # copy verbatim: x / y / label survive
        bin_value = float(point["x"])
        bin_indices = members.get(bin_value, [])
        if len(bin_indices) < BOOTSTRAP_MIN_SAMPLES:
            banded.append(out_point)
            continue

        # Deterministic per-bin seed so re-ordering the curve never shifts a bin's band.
        bin_seed = seed + hash((curve_name, bin_value)) % 1_000_000
        n = len(bin_indices)
        draws: list[float] = []
        for i in range(n_resamples):
            rng = random.Random(bin_seed + i)
            idx = [bin_indices[rng.randrange(n)] for _ in range(n)]
            pred_bs = _index_select(pred, idx)
            target_bs = _index_select(target, idx)
            meta_bs = _index_select(meta, idx) if meta is not None else None
            metric.reset()
            metric.update(pred_bs, target_bs, meta_bs)
            resampled = metric.compute().get(curve_name)
            if not isinstance(resampled, list):
                continue
            # The resample holds ONLY this bin's samples, so the re-emitted curve carries a
            # single point at ``bin_value``; read its ``y`` back.
            for rp in resampled:
                if isinstance(rp, dict) and float(rp.get("x", float("nan"))) == bin_value:
                    draws.append(float(rp["y"]))
                    break

        if len(draws) < BOOTSTRAP_MIN_SAMPLES:
            banded.append(out_point)
            continue

        draws.sort()
        out_point["y_low"] = max(0.0, min(1.0, _percentile(draws, lo_q)))
        out_point["y_high"] = max(0.0, min(1.0, _percentile(draws, hi_q)))
        banded.append(out_point)
    return banded


def _curve_uncertainty_bands(
    metrics: Iterable[Metric],
    curves: dict[str, list[dict[str, float]]],
    pred: Tensor,
    target: Tensor,
    meta: Batch | None,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Add per-bin bootstrap bands to every curve whose metric exposes its bins.

    A curve metric MAY expose ``sample_bins(pred, target, meta) -> list[float]`` returning
    the per-sample bin key aligned with the accumulated samples (``AccuracyVsSnr`` returns
    the per-sample ``snr_db``). When present, :func:`_bootstrap_curve_bands` conditions a
    percentile bootstrap on each bin's own samples and attaches ``y_low``/``y_high``. Curves
    from metrics WITHOUT the hook are left unchanged (no band), so ROC-style threshold curves
    -- where a point is a threshold, not a stratum of samples -- are never mis-banded.

    Returns ``{curve_name: banded_points}`` only for curves it actually banded; the caller
    merges these over the existing curves.
    """
    if _accumulated_length(pred, target) < BOOTSTRAP_MIN_SAMPLES or n_resamples < 1:
        return {}

    n = _accumulated_length(pred, target)
    out: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        bin_hook = getattr(metric, "sample_bins", None)
        if not callable(bin_hook):
            continue
        try:
            bins = bin_hook(pred, target, meta)
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(bins, list) or len(bins) != n:
            continue
        bins_f = [float(b) for b in bins]
        for curve_name in metric.compute():
            points = curves.get(curve_name)
            if not isinstance(points, list) or not points:
                continue
            out[curve_name] = _bootstrap_curve_bands(
                metric,
                curve_name,
                points,
                pred,
                target,
                meta,
                bins_f,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
    return out


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
    compute_bootstrap_ci: bool = True,
    bootstrap_n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Run the eval loop and emit a schema-valid ``result.json`` dict.

    Aggregates ``task.metrics()`` over ``split`` (optionally restricted to ``track``),
    assembles the result dict, validates it against ``schemas/result.schema.json`` with
    ``jsonschema`` (Draft 2020-12), writes ``out_path`` if given, and returns the dict.

    When ``compute_bootstrap_ci`` is true (default), a percentile bootstrap over the
    accumulated per-sample predictions adds a ``metrics.uncertainty`` block (schema 1.2.0):
    one two-sided interval per scalar metric at ``bootstrap_confidence`` from
    ``bootstrap_n_resamples`` resamples (EVALUATION_PROTOCOL.md "Statistical rigor &
    uncertainty"). Set ``compute_bootstrap_ci=False`` (or lower ``bootstrap_n_resamples``)
    when the resampling cost is prohibitive; ``bootstrap_seed`` defaults to ``seed`` so the
    interval is reproducible.

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
    # Accumulate the whole pred/target/meta ONLY when a bootstrap CI is requested, so the
    # default cost stays a streaming pass and memory is untouched when CIs are disabled.
    acc = _BootstrapAccumulator() if compute_bootstrap_ci else None
    for chunk in _iter_batches(data, batch_size):
        batch = _collate(chunk)
        target = task.build_targets(batch)
        pred = model.forward(batch)
        for metric in metrics:
            metric.update(pred, target, batch)
        if acc is not None:
            acc.add(pred, target, batch)
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

    # --- Bootstrap CIs over the accumulated predictions (schema 1.2.0) --------------------------
    # Runs AFTER aggregation because the resamples reset()/update()/compute() the metrics; the
    # point estimates in ``values`` are already captured, so trashing metric state is harmless.
    uncertainty: dict[str, dict[str, Any]] = {}
    if acc is not None:
        uncertainty = _bootstrap_uncertainty(
            metrics,
            acc.pred,
            acc.target,
            acc.meta,
            n_resamples=bootstrap_n_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed if bootstrap_seed is not None else seed,
        )
        # Only keep CIs for metrics we actually reported as scalar point estimates.
        uncertainty = {k: v for k, v in uncertainty.items() if k in values}

    # --- Per-bin curve bands (percentile bootstrap, stratified WITHIN each SNR bin) --------------
    # Same accumulated predictions, same seeded stdlib RNG as the scalar CI: for a mono-run
    # (single seed) row this is the ONLY source of a per-bin y_low/y_high band on the curves
    # (the multi-seed aggregator writes its own across-seed band and never reaches here).
    if acc is not None and curves:
        banded = _curve_uncertainty_bands(
            metrics,
            curves,
            acc.pred,
            acc.target,
            acc.meta,
            n_resamples=bootstrap_n_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed if bootstrap_seed is not None else seed,
        )
        curves.update(banded)

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
    if uncertainty:
        metrics_block["uncertainty"] = uncertainty

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
    out_path = Path(out_path)  # accept str paths from any caller
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
