"""Backfill Wilson-score confidence intervals onto already-committed leaderboard rows.

Some board rows were committed BEFORE ``evaluate()`` learned to emit a bootstrap CI, and
their raw per-sample predictions no longer exist -- so a percentile bootstrap is
impossible. For those rows the protocol (EVALUATION_PROTOCOL.md "Statistical rigor &
uncertainty") allows an approximate **Wilson (binomial) score interval** reconstructed from
the reported accuracy and ``n`` alone, but ONLY for metrics that are genuine accuracy
proportions (``accuracy_overall`` / ``rank1_accuracy``) and ONLY for harness-produced tiers
(``self_reported`` / ``verified``). It is NEVER applied to ``from_paper`` / literature rows
(we do not invent statistics on numbers we did not run) nor to non-proportion metrics like
``mAP`` / ``pd@pfa=0.1``.

Each eligible row gains::

    metrics.uncertainty.<primary_metric> = {
        "ci_low": ..., "ci_high": ...,
        "method": "wilson_backfill",
        "confidence": 0.95,
        "note": "backfill a posteriori, ..."
    }

The write is atomic (:func:`rfbench.core.evaluate._atomic_write_json`) and the mutated
document is RE-VALIDATED against ``schemas/result.schema.json`` before it lands, so an
invalid row never overwrites a valid one. The script is idempotent: a row that already
carries a ``wilson_backfill`` interval for its primary metric is left untouched.

Run from the repo root::

    python scripts/backfill_uncertainty.py            # backfill in place
    python scripts/backfill_uncertainty.py --dry-run  # report only, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from rfbench.core.evaluate import _atomic_write_json

logger = logging.getLogger("rfbench.backfill_uncertainty")

#: Metrics that are genuine binomial accuracy proportions (Wilson is only valid for these).
_PROPORTION_METRICS = frozenset({"accuracy_overall", "rank1_accuracy"})

#: Harness tiers a backfill may touch. ``from_paper`` / ``from_paper_uncertain`` are
#: literature rows we never re-run, so they are excluded on purpose.
_BACKFILLABLE_STATUSES = frozenset({"self_reported", "verified"})

#: Two-sided z for a 95% interval (stdlib only -- avoids a scipy dependency).
_Z_95 = 1.959963984540054

_BACKFILL_NOTE = (
    "backfill a posteriori, pas de resampling sur predictions brutes "
    "(Wilson score reconstruit depuis accuracy et n)"
)

#: A row carrying ``metrics.uncertainty`` genuinely uses a 1.2.0 feature, so a backfilled
#: row's declared ``schema_version`` is lifted to this (additive, non-breaking) version.
_UNCERTAINTY_SCHEMA_VERSION = "1.2.0"


def wilson_interval(p_hat: float, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Return the Wilson score interval ``(ci_low, ci_high)`` for a binomial proportion.

    ``p_hat`` is the observed accuracy in ``[0, 1]`` and ``n`` the number of trials. Uses
    the closed-form Wilson score interval (better small-sample coverage than the naive
    normal approximation, and it never leaves ``[0, 1]``); ``z`` is the two-sided normal
    quantile (default ``z_0.975`` for a 95% interval). Pure stdlib ``math``.

    Raises ``ValueError`` on ``n <= 0`` or ``p_hat`` outside ``[0, 1]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError(f"p_hat must be in [0, 1], got {p_hat}")
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _repo_schema_path() -> Path:
    """Locate ``schemas/result.schema.json`` in THIS script's repo checkout.

    Deliberately resolves relative to this file rather than to the imported ``rfbench``
    package: an editable install may point ``rfbench`` at a *different* worktree whose
    schema predates 1.2.0, which would spuriously reject the (valid) backfilled row. We
    always validate against the schema living next to the results we are editing.

    Raises ``RuntimeError`` if the repo schema cannot be found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schemas" / "result.schema.json"
        if candidate.is_file():
            return candidate
    raise RuntimeError("could not locate schemas/result.schema.json next to this script")


def _validate_or_raise(document: dict[str, Any]) -> None:
    """Validate ``document`` against this repo's ``result.schema.json`` or raise.

    Mirrors :func:`rfbench.core.evaluate._validate_or_raise` but pins the schema to this
    repo (see :func:`_repo_schema_path`) so the check is independent of where ``rfbench``
    happens to be installed.
    """
    from jsonschema import Draft202012Validator

    schema = json.loads(_repo_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


def _semver_tuple(version: str) -> tuple[int, ...]:
    """Parse a ``X.Y.Z`` SemVer string into a comparable int tuple (best effort)."""
    parts: list[int] = []
    for chunk in str(version).split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _primary_key(result: dict[str, Any]) -> str:
    return str(result["metrics"]["primary"])


def _n_samples(result: dict[str, Any]) -> int | None:
    """Return the reported sample count (``eval.n_samples``) or ``None`` if absent.

    ``from_paper`` rows have no ``eval`` block at all, so this returns ``None`` for them --
    a second, structural reason they are ineligible on top of the status check.
    """
    eval_block = result.get("eval")
    if not isinstance(eval_block, dict):
        return None
    n = eval_block.get("n_samples")
    return int(n) if isinstance(n, int) else None


def _eligibility_reason(result: dict[str, Any]) -> str | None:
    """Return ``None`` if the row is eligible for Wilson backfill, else a skip reason.

    Eligible = harness tier (``self_reported`` / ``verified``) AND a proportion primary
    metric (``accuracy_overall`` / ``rank1_accuracy``) present in ``metrics.values`` AND a
    positive ``eval.n_samples`` AND no existing ``wilson_backfill`` interval on that metric
    (idempotence).
    """
    status = str(result.get("verification", {}).get("status", ""))
    if status not in _BACKFILLABLE_STATUSES:
        return f"status={status!r} is not a backfillable harness tier"

    primary = _primary_key(result)
    if primary not in _PROPORTION_METRICS:
        return f"primary metric {primary!r} is not a binomial proportion"

    values = result["metrics"].get("values", {})
    if primary not in values:
        return f"primary metric {primary!r} absent from metrics.values"

    n = _n_samples(result)
    if n is None or n <= 0:
        return "no positive eval.n_samples to reconstruct a Wilson interval from"

    existing = result["metrics"].get("uncertainty", {}).get(primary)
    if isinstance(existing, dict) and existing.get("method") == "wilson_backfill":
        return "already carries a wilson_backfill interval (idempotent skip)"

    return None


def backfill_document(result: dict[str, Any]) -> bool:
    """Add a Wilson interval for the primary metric to ``result`` in place.

    Returns ``True`` if the document was mutated, ``False`` if it was already up to date /
    ineligible (the caller has usually filtered ineligible rows already; this stays safe if
    not). The mutated ``result`` still validates against the schema.
    """
    if _eligibility_reason(result) is not None:
        return False
    primary = _primary_key(result)
    p_hat = float(result["metrics"]["values"][primary])
    n = _n_samples(result)
    assert n is not None  # guaranteed by _eligibility_reason
    ci_low, ci_high = wilson_interval(p_hat, n)

    metrics = result["metrics"]
    uncertainty = metrics.setdefault("uncertainty", {})
    uncertainty[primary] = {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": "wilson_backfill",
        "confidence": 0.95,
        "note": _BACKFILL_NOTE,
    }
    # The row now uses a 1.2.0 feature; lift its declared schema_version so consumers that
    # gate on version see the uncertainty block. Additive + non-breaking, so never a
    # downgrade: only bump when the row currently declares an older version.
    current = _semver_tuple(str(result.get("schema_version", "0.0.0")))
    if current < _semver_tuple(_UNCERTAINTY_SCHEMA_VERSION):
        result["schema_version"] = _UNCERTAINTY_SCHEMA_VERSION
    return True


def _iter_result_files(results_root: Path) -> list[Path]:
    """Return every ``*.json`` under ``results_root`` in a stable, sorted order."""
    return sorted(results_root.rglob("*.json"))


def backfill_tree(results_root: Path, *, dry_run: bool = False) -> int:
    """Walk ``results_root`` and Wilson-backfill every eligible row; return #rows changed.

    Logs one line per file: skipped rows say why (ineligible/idempotent), changed rows
    report the reconstructed interval. When ``dry_run`` is set nothing is written but the
    would-be changes are still logged and validated.
    """
    changed = 0
    for path in _iter_result_files(results_root):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skip %s: unreadable (%s)", path, exc)
            continue

        reason = _eligibility_reason(result)
        if reason is not None:
            logger.info("skip %s: %s", path, reason)
            continue

        primary = _primary_key(result)
        backfill_document(result)
        interval = result["metrics"]["uncertainty"][primary]
        # Re-validate the mutated row BEFORE it can overwrite a valid file on disk.
        _validate_or_raise(result)
        logger.info(
            "backfill %s: %s in [%.4f, %.4f] (wilson, n=%d)%s",
            path,
            primary,
            interval["ci_low"],
            interval["ci_high"],
            _n_samples(result),
            " [dry-run]" if dry_run else "",
        )
        if not dry_run:
            _atomic_write_json(result, path)
        changed += 1
    return changed


def _default_results_root() -> Path:
    """Locate ``leaderboard/results`` relative to this file's repo checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "leaderboard" / "results"
        if candidate.is_dir():
            return candidate
    return Path("leaderboard/results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=_default_results_root(),
        help="root of the committed result.json tree (default: leaderboard/results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing any file",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.results.is_dir():
        logger.error("results root %s does not exist", args.results)
        return 2

    changed = backfill_tree(args.results, dry_run=args.dry_run)
    logger.info(
        "%s %d row(s)%s",
        "would backfill" if args.dry_run else "backfilled",
        changed,
        " (dry-run)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
