"""Maintainer verification pipeline (WP-53): flip ``self_reported`` -> ``verified``.

This module implements the Tier-2 verification step described in ``docs/SUBMISSION.md``: a
maintainer re-runs a submitted evaluation and, if the recomputed metrics match the submitter's
``expected_metrics`` within the manifest's ``tolerance``, the row's ``verification.status`` is
flipped from ``self_reported`` to ``verified`` with full provenance stamped
(``verified_by`` / ``verified_date`` / ``verified_hardware`` / ``method``).

The design keeps the numerics **injectable**: :func:`verify_result` takes the recomputed metric
values (``rerun_metrics``) as an argument rather than running a model, so the whole pipeline is unit
testable on pure stdlib without torch/torchsig or a real re-run. The actual re-run (eval_only /
full_retrain on a GPU station) lives in the CLI/cluster layer and merely feeds its measured metrics
in here.

Invariants (mirroring ``schemas/result.schema.json`` + ``schemas/submission.schema.json``):

* This module is the ONLY code path that may set ``verification.status = "verified"`` (besides a
  hand-authored seed row); it never invents provenance -- ``verified_by`` / ``verified_hardware`` /
  the date come from the caller, ``observed_primary`` / ``tolerance`` / ``method`` from the actual
  re-run + manifest.
* The manifest MUST be complete against ``submission.schema.json`` before any flip; an incomplete
  manifest can only leave the row ``self_reported``.
* A verified output is itself validated against ``result.schema.json`` before being returned, so a
  verified row can never lack the provenance the schema requires.
* On any mismatch or incompleteness the ORIGINAL result is returned unchanged (still
  ``self_reported``) alongside a machine-readable report of what failed.

``jsonschema`` is imported LAZILY inside the validation helpers so ``import rfbench.verify`` (and
``import rfbench``) stay dependency-free; the unit tests run with only ``pytest`` + ``jsonschema``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "VerificationError",
    "MetricCheck",
    "VerificationReport",
    "verify_result",
]

#: Name of the JSON schema a verified result is validated against.
_RESULT_SCHEMA = "result.schema.json"
#: Name of the JSON schema the accompanying manifest must satisfy to be "complete".
_SUBMISSION_SCHEMA = "submission.schema.json"


class VerificationError(Exception):
    """Raised for unrecoverable verification failures (bad schema/tooling, not a metric mismatch).

    A metric mismatch or an incomplete manifest is a *normal* outcome reported via
    :class:`VerificationReport` (``status`` stays ``self_reported``); this exception is reserved for
    situations where verification cannot even be attempted -- e.g. the result/manifest schema cannot
    be located, ``jsonschema`` is unavailable, or the manifest's primary metric is absent from its
    own ``expected_metrics`` block.
    """


# --------------------------------------------------------------------------------------------------
# Schema resolution + validation (mirrors ``rfbench.cli`` / ``rfbench.core.evaluate``)
# --------------------------------------------------------------------------------------------------
def _resolve_schema_path(schema_name: str) -> Path | None:
    """Locate a JSON schema file.

    Resolution order matches the packaging contract used elsewhere in the harness: prefer the
    force-included package data (``rfbench/_schemas`` in an installed wheel), then fall back to the
    repo ``schemas/`` directory when running from a source checkout. Returns ``None`` if neither
    location has the file.
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


def _schema_errors(document: dict[str, Any], schema_name: str) -> list[str]:
    """Validate ``document`` against ``schema_name``; return human-readable error strings.

    Returns an empty list when the document is valid. If the schema cannot be located or
    ``jsonschema`` is missing, a single explanatory error is returned so callers fail loudly rather
    than silently accepting an unvalidated artifact. ``jsonschema`` is imported lazily so
    ``import rfbench.verify`` stays dependency-free.
    """
    schema_path = _resolve_schema_path(schema_name)
    if schema_path is None:
        return [f"could not locate {schema_name} (checked rfbench/_schemas and repo schemas/)"]

    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        return ["jsonschema is not installed; install the 'rfbench' package to validate JSON"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in errors
    ]


# --------------------------------------------------------------------------------------------------
# Report dataclasses
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricCheck:
    """Outcome of comparing one expected metric to its recomputed value.

    ``tolerance`` is the absolute bound the delta was compared against (already resolved from the
    manifest's ``absolute`` / ``relative`` / ``per_metric`` block for this specific metric).
    ``within`` is ``True`` iff ``abs(observed - expected) <= tolerance``.
    """

    name: str
    expected: float
    observed: float
    tolerance: float
    within: bool

    @property
    def delta(self) -> float:
        """Signed difference ``observed - expected`` (absolute value is compared to tolerance)."""
        return self.observed - self.expected


@dataclass(frozen=True)
class VerificationReport:
    """Machine-readable outcome of a verification attempt.

    ``verified`` is the single source of truth for success. ``result`` is the document a caller
    should persist: on success it is a NEW verified result (deep copy, never mutating the input); on
    failure it is the ORIGINAL self_reported result, untouched. ``errors`` explains any hard
    incompleteness (missing manifest fields, primary metric absent, schema failure); ``checks``
    holds the per-metric comparison for the primary (and any additional expected metric).
    """

    verified: bool
    result: dict[str, Any]
    checks: list[MetricCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """The ``verification.status`` of :attr:`result` (``verified`` or ``self_reported``)."""
        verification = self.result.get("verification", {})
        if not isinstance(verification, dict):
            return "self_reported"
        return str(verification.get("status", "self_reported"))

    @property
    def primary_check(self) -> MetricCheck | None:
        """The check for the task's primary metric, if a comparison was made."""
        return self.checks[0] if self.checks else None

    def summary(self) -> str:
        """A one-line human summary suitable for CLI output."""
        if self.verified:
            primary = self.primary_check
            detail = (
                f" primary {primary.name}: expected {primary.expected:.6g}, "
                f"observed {primary.observed:.6g} (|delta|={abs(primary.delta):.3g} "
                f"<= {primary.tolerance:.3g})"
                if primary is not None
                else ""
            )
            return f"verified.{detail}"
        if self.errors:
            return "NOT verified (stays self_reported): " + "; ".join(self.errors)
        offending = [c for c in self.checks if not c.within]
        detail = "; ".join(
            f"{c.name}: expected {c.expected:.6g}, observed {c.observed:.6g} "
            f"(|delta|={abs(c.delta):.3g} > {c.tolerance:.3g})"
            for c in offending
        )
        return f"NOT verified (stays self_reported): out of tolerance -> {detail}"


# --------------------------------------------------------------------------------------------------
# Tolerance resolution
# --------------------------------------------------------------------------------------------------
def _resolve_tolerance(metric: str, expected: float, tolerance: dict[str, Any]) -> float:
    """Resolve the absolute tolerance bound to apply to ``metric``.

    Precedence mirrors ``submission.schema.json``'s ``tolerance`` block: a ``per_metric`` entry for
    this metric overrides the global bound; otherwise ``absolute`` is used directly, and/or
    ``relative`` is turned into an absolute bound via ``relative * abs(expected)``. When both
    ``absolute`` and ``relative`` are present the LARGER (more permissive) bound wins, matching the
    "matches within tolerance" spirit of the protocol. Returns ``0.0`` if nothing applies (an exact
    match is then required).
    """
    per_metric = tolerance.get("per_metric")
    if isinstance(per_metric, dict) and metric in per_metric:
        return float(per_metric[metric])

    bounds: list[float] = []
    if "absolute" in tolerance:
        bounds.append(float(tolerance["absolute"]))
    if "relative" in tolerance:
        bounds.append(float(tolerance["relative"]) * abs(expected))
    return max(bounds) if bounds else 0.0


def _primary_metric_name(result: dict[str, Any]) -> str | None:
    """Extract ``metrics.primary`` from a result document, if present and well-formed."""
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    primary = metrics.get("primary")
    return primary if isinstance(primary, str) else None


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
def verify_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    rerun_metrics: dict[str, float],
    *,
    verified_by: str,
    verified_hardware: str,
    verified_date: date | str | None = None,
    method: str | None = None,
    note: str | None = None,
) -> VerificationReport:
    """Verify a submitted result against a manifest + a re-run's recomputed metrics.

    Steps, in order:

    1. **Manifest completeness** -- the manifest is validated against ``submission.schema.json``.
       Any error leaves the row ``self_reported`` (reported via
       :attr:`VerificationReport.errors`); no flip is attempted.
    2. **Primary metric present** -- the result's ``metrics.primary`` must appear in the manifest's
       ``expected_metrics`` and in ``rerun_metrics``; otherwise the row stays ``self_reported``.
    3. **Tolerance comparison** -- for the primary metric (and every other metric the manifest lists
       in ``expected_metrics`` that the re-run also measured) the absolute delta is compared to the
       tolerance resolved from the manifest's ``tolerance`` block. All compared metrics must be
       within tolerance to verify.
    4. **Stamp + validate** -- on success a NEW result (deep copy) gets
       ``verification.status = "verified"`` plus ``verified_by`` / ``verified_date`` /
       ``verified_hardware`` / ``method`` / ``tolerance`` / ``observed_primary`` filled from the
       arguments and the re-run, then validated against ``result.schema.json`` before being
       returned.

    Args:
        result: the submitted ``result.json`` document (must be ``self_reported``).
        manifest: the accompanying ``submission.schema.json`` manifest.
        rerun_metrics: metric name -> recomputed value from the maintainer's re-run (injectable so
            tests need no real model).
        verified_by: maintainer handle -> ``verification.verified_by`` (must be non-empty).
        verified_hardware: hardware string -> ``verification.verified_hardware`` (must be
            non-empty).
        verified_date: ISO ``date`` (or ``YYYY-MM-DD`` string) of the re-run; defaults to today
            (UTC-agnostic local date) when omitted.
        method: ``eval_only`` | ``full_retrain`` -> ``verification.method``; defaults to the
            manifest's ``rerun_mode`` (or ``eval_only``).
        note: optional free-form note -> ``verification.note``.

    Returns:
        A :class:`VerificationReport`. When :attr:`~VerificationReport.verified` is ``True`` its
        ``result`` is the verified document; otherwise it is the untouched original.

    Raises:
        VerificationError: only when verification cannot be attempted at all -- ``verified_by`` /
            ``verified_hardware`` empty, the result lacks a primary metric, or the produced verified
            document fails ``result.schema.json`` (a programming/stamping error, not a submitter
            fault).
    """
    if not verified_by:
        raise VerificationError("verified_by must be a non-empty maintainer handle")
    if not verified_hardware:
        raise VerificationError("verified_hardware must be a non-empty string")

    # 1. Manifest completeness (self_reported on any incompleteness).
    manifest_errors = [f"manifest: {msg}" for msg in _schema_errors(manifest, _SUBMISSION_SCHEMA)]
    if manifest_errors:
        return VerificationReport(verified=False, result=result, errors=manifest_errors)

    # 2. Primary metric identity.
    primary = _primary_metric_name(result)
    if primary is None:
        raise VerificationError("result has no metrics.primary to verify against")

    expected_metrics = manifest.get("expected_metrics")
    if not isinstance(expected_metrics, dict) or primary not in expected_metrics:
        return VerificationReport(
            verified=False,
            result=result,
            errors=[
                f"manifest expected_metrics is missing the primary metric {primary!r} "
                "(cannot verify)"
            ],
        )
    if primary not in rerun_metrics:
        return VerificationReport(
            verified=False,
            result=result,
            errors=[f"re-run did not measure the primary metric {primary!r} (cannot verify)"],
        )

    tolerance = manifest.get("tolerance")
    if not isinstance(tolerance, dict):  # schema guarantees this, but stay defensive.
        return VerificationReport(
            verified=False,
            result=result,
            errors=["manifest tolerance block is missing or malformed"],
        )

    # 3. Compare the primary metric first, then every other expected metric the re-run measured.
    compare_order = [primary] + [m for m in expected_metrics if m != primary]
    checks: list[MetricCheck] = []
    for metric in compare_order:
        if metric not in rerun_metrics:
            # An expected non-primary metric the re-run did not report: skip it (the primary is the
            # ranking gate; a missing secondary metric does not by itself veto the flip).
            continue
        expected = float(expected_metrics[metric])
        observed = float(rerun_metrics[metric])
        bound = _resolve_tolerance(metric, expected, tolerance)
        checks.append(
            MetricCheck(
                name=metric,
                expected=expected,
                observed=observed,
                tolerance=bound,
                within=abs(observed - expected) <= bound,
            )
        )

    if not all(check.within for check in checks):
        return VerificationReport(verified=False, result=result, errors=[], checks=checks)

    # 4. Stamp a NEW verified result and validate it.
    verified = _stamp_verified(
        result,
        primary=primary,
        observed_primary=float(rerun_metrics[primary]),
        primary_tolerance=checks[0].tolerance,
        verified_by=verified_by,
        verified_hardware=verified_hardware,
        verified_date=verified_date,
        method=method if method is not None else _default_method(manifest),
        note=note,
    )
    schema_errors = _schema_errors(verified, _RESULT_SCHEMA)
    if schema_errors:
        raise VerificationError(
            "stamped verified result failed result.schema.json: " + "; ".join(schema_errors)
        )
    return VerificationReport(verified=True, result=verified, checks=checks, errors=[])


def _default_method(manifest: dict[str, Any]) -> str:
    """Pick the re-run method: the manifest's ``rerun_mode`` if valid, else ``eval_only``."""
    mode = manifest.get("rerun_mode")
    return mode if mode in ("eval_only", "full_retrain") else "eval_only"


def _stamp_verified(
    result: dict[str, Any],
    *,
    primary: str,
    observed_primary: float,
    primary_tolerance: float,
    verified_by: str,
    verified_hardware: str,
    verified_date: date | str | None,
    method: str,
    note: str | None,
) -> dict[str, Any]:
    """Return a deep copy of ``result`` with a fully-stamped ``verification`` block.

    Never mutates the input. ``verified_date`` accepts an ISO ``date`` or a ``YYYY-MM-DD`` string;
    ``None`` stamps today's date. The stamped block satisfies the ``status == "verified"`` branch of
    ``result.schema.json`` (``verified_by`` / ``verified_date`` / ``verified_hardware`` / ``method``
    all present), with ``tolerance`` and ``observed_primary`` recording the exact match criterion.
    """
    stamped = copy.deepcopy(result)
    if isinstance(verified_date, date):
        date_str = verified_date.isoformat()
    elif isinstance(verified_date, str):
        date_str = verified_date
    else:
        date_str = date.today().isoformat()

    verification: dict[str, Any] = {
        "status": "verified",
        "verified_by": verified_by,
        "verified_date": date_str,
        "verified_hardware": verified_hardware,
        "method": method,
        "tolerance": primary_tolerance,
        "observed_primary": observed_primary,
    }
    if note is not None:
        verification["note"] = note
    stamped["verification"] = verification
    return stamped


# --------------------------------------------------------------------------------------------------
# Convenience loaders (used by the CLI; kept here so the file layer lives with the pipeline)
# --------------------------------------------------------------------------------------------------
def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from ``path`` or raise ``VerificationError`` with a clear message."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"could not read JSON from {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise VerificationError(f"{path} must contain a JSON object, got {type(loaded).__name__}")
    return loaded


def rerun_metrics_from_result(rerun: dict[str, Any]) -> dict[str, float]:
    """Extract ``metrics.values`` (name -> float) from a re-run ``result.json`` document.

    A maintainer's re-run is itself a ``result.json`` (produced by ``rfbench eval``); this pulls its
    scalar metrics into the ``rerun_metrics`` mapping :func:`verify_result` expects. Raises
    ``VerificationError`` if the document has no ``metrics.values`` scalar map.
    """
    metrics = rerun.get("metrics")
    values = metrics.get("values") if isinstance(metrics, dict) else None
    if not isinstance(values, dict) or not values:
        raise VerificationError("re-run result has no metrics.values scalar map")
    return {str(name): float(value) for name, value in values.items()}
