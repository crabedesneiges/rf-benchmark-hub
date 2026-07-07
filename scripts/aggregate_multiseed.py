"""Aggregate N per-seed staging result.json files into one canonical leaderboard row.

After running a model with seeds 42, 43, 44 (or any list of seeds), the harness emits
one result.json per seed under ``$WORK/logs/multiseed/<task>/<model>-seed<S>.json``.
This script reads those staging files, verifies their comparability (same task / model /
regime / split.checksum), computes the mean of every scalar metric, and emits a single
canonical ``leaderboard/results/<task>/<model>.json`` row whose ``metrics.uncertainty``
carries an honest ±1 standard-deviation interval (``multi_seed_std``).

Convention (decided in Phase-0 planning, non-negotiable):
  * ``metrics.values`` = mean of each scalar metric across seeds
  * ``metrics.uncertainty[<metric>]`` = {ci_low: mean - stdev, ci_high: mean + stdev,
    method: "multi_seed_std", n_seeds: N, note: <descriptive>}
  * ``stdev`` uses ``statistics.stdev`` (sample stdev, ddof = N-1)
  * Curves / per_class come from the **seed-42 run** (or the first seed in the list when
    42 is absent) -- the note documents this.
  * ``environment.seed`` stays as it appears in the seed-42 file (base run).
  * The result is validated against ``schemas/result.schema.json`` from THIS repo before
    the atomic write. The script is idempotent.

IMPORTANT: this is NOT a 95% confidence interval. The note in every uncertainty entry
explicitly states it is a ±1 descriptive stdev across seeds 42/43/44. Per-seed bootstrap
CIs remain available in the staging files.

Run from the repo root::

    python scripts/aggregate_multiseed.py \\
        --task amc --model cldnn --seeds 42,43,44

Override staging dir (default: $WORK/logs/multiseed or RFBENCH_MULTISEED_STAGING_DIR)::

    python scripts/aggregate_multiseed.py \\
        --task amc --model cldnn --staging-dir /path/to/staging \\
        --out leaderboard/results/amc/cldnn.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("rfbench.aggregate_multiseed")

# ---------------------------------------------------------------------------
# Schema / repo helpers
# ---------------------------------------------------------------------------

_MULTISEED_NOTE_TEMPLATE = (
    "Moyenne ±1 écart-type DESCRIPTIF sur les seeds {seeds}. "
    "Ce N'EST PAS un intervalle de confiance à 95%. "
    "Les IC bootstrap par seed restent disponibles dans les fichiers de staging. "
    "Les curves/per_class proviennent du seed de référence ({ref_seed})."
)

#: Metric keys whose values are genuine proportions in [0, 1]; their descriptive interval
#: bounds are clamped to [0, 1] so a near-perfect score with a small spread cannot produce a
#: ci_high above 1 (mirrors ``slurm/eval_fm_episodic.py``'s convention).
_PROPORTION_METRICS = frozenset(
    {"accuracy_overall", "macro_f1", "rank1_accuracy", "auroc", "eer", "mAP", "mAR", "IoU"}
)


def _clamp_unit(value: float, key: str) -> float:
    """Clamp ``value`` to [0, 1] iff ``key`` is a bounded-proportion metric, else pass through.

    Clamping never widens the descriptive interval, it only keeps it inside the metric's
    domain (e.g. interf_cnn's mean 0.9996 + stdev 0.0008 would otherwise report ci_high
    1.0003, meaningless for an accuracy).
    """
    if key not in _PROPORTION_METRICS:
        return value
    return max(0.0, min(1.0, value))


def _repo_root() -> Path:
    """Locate the repo root relative to this file (walks up looking for schemas/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "schemas" / "result.schema.json").is_file():
            return parent
    raise RuntimeError("could not locate schemas/result.schema.json next to this script")


def _repo_schema_path() -> Path:
    """Return the path to schemas/result.schema.json in THIS repo checkout.

    Deliberately resolves relative to this file rather than to the installed ``rfbench``
    package: an editable install may point ``rfbench`` at a different worktree whose schema
    predates 1.2.0, which would spuriously reject a valid aggregated row.
    """
    path = _repo_root() / "schemas" / "result.schema.json"
    if not path.is_file():
        raise RuntimeError(f"schema not found at {path}")
    return path


def _validate_or_raise(document: dict[str, Any]) -> None:
    """Validate ``document`` against this repo's result.schema.json or raise."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_repo_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


def _atomic_write_json(document: dict[str, Any], out_path: Path) -> None:
    """Serialize ``document`` to ``out_path`` atomically (temp file + os.replace)."""
    out_path = Path(out_path)
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


# ---------------------------------------------------------------------------
# Staging-dir resolution
# ---------------------------------------------------------------------------

_ENV_STAGING_VAR = "RFBENCH_MULTISEED_STAGING_DIR"
_FALLBACK_ENV_VAR = "WORK"


def _default_staging_dir() -> Path:
    """Return the default staging dir from env vars, without hardcoding paths."""
    env_val = os.environ.get(_ENV_STAGING_VAR)
    if env_val:
        return Path(env_val)
    work = os.environ.get(_FALLBACK_ENV_VAR)
    if work:
        return Path(work) / "logs" / "multiseed"
    raise RuntimeError(
        f"Cannot determine staging dir: set {_ENV_STAGING_VAR} "
        f"or {_FALLBACK_ENV_VAR} environment variable, or pass --staging-dir."
    )


def _default_out_path(task: str, model: str) -> Path:
    """Return leaderboard/results/<task>/<model>.json relative to repo root."""
    return _repo_root() / "leaderboard" / "results" / task / f"{model}.json"


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS: list[tuple[str, ...]] = [
    # (json_path_tuple, label_for_error)
    # We compare scalar string/int leaves; nested dicts are traversed manually below.
]


def _deep_get(doc: dict[str, Any], *keys: str) -> object:
    """Navigate nested dicts; return None if any key is missing."""
    cur: object = doc
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _check_run_compatibility(docs: list[dict[str, Any]], seeds: list[int]) -> None:
    """Raise ValueError with a clear message if the runs are not comparable.

    Comparability conditions:
    1. task.name, task.version agree across all seeds.
    2. model.name agrees across all seeds.
    3. regime.name (and k_shot if few_shot) agrees across all seeds.
    4. split.checksum agrees across all seeds (same canonical split).
    """
    ref = docs[0]
    ref_seed = seeds[0]

    checks: list[tuple[str, Any]] = [
        ("task.name", _deep_get(ref, "task", "name")),
        ("task.version", _deep_get(ref, "task", "version")),
        ("model.name", _deep_get(ref, "model", "name")),
        ("regime.name", _deep_get(ref, "regime", "name")),
        ("split.checksum", _deep_get(ref, "split", "checksum")),
    ]
    # k_shot only when regime is few_shot
    if _deep_get(ref, "regime", "name") == "few_shot":
        checks.append(("regime.k_shot", _deep_get(ref, "regime", "k_shot")))

    for doc, seed in zip(docs[1:], seeds[1:], strict=True):
        for path, ref_val in checks:
            keys = path.split(".")
            val = _deep_get(doc, *keys)
            if val != ref_val:
                raise ValueError(
                    f"Run incompatibility: seed {seed} has {path}={val!r} "
                    f"but seed {ref_seed} has {path}={ref_val!r}. "
                    "Refusing to aggregate incomparable runs."
                )


def _extract_scalar_metrics(doc: dict[str, Any]) -> dict[str, float]:
    """Return the scalar values dict from metrics.values."""
    values = _deep_get(doc, "metrics", "values")
    if not isinstance(values, dict):
        raise ValueError("metrics.values is missing or not a dict")
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}


def _check_metric_sets_compatible(scalar_sets: list[dict[str, float]], seeds: list[int]) -> None:
    """Raise ValueError if the scalar metric key sets differ across seeds."""
    ref_keys = set(scalar_sets[0].keys())
    for scalars, seed in zip(scalar_sets[1:], seeds[1:], strict=True):
        if set(scalars.keys()) != ref_keys:
            raise ValueError(
                f"Metric key mismatch: seed {seed} has keys {sorted(scalars.keys())} "
                f"but seed {seeds[0]} has keys {sorted(ref_keys)}. "
                "Refusing to aggregate runs with different metric sets."
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    docs: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Build the aggregated result dict from per-seed staging documents.

    - Base structure = run with seed 42, or first seed if 42 absent.
    - metrics.values = mean of each scalar.
    - metrics.uncertainty[metric] = multi_seed_std entry.
    - curves/per_class from the reference seed (noted in uncertainty note).
    - environment.seed stays as in the reference doc.
    """
    # Pick reference seed (42 if present, else first)
    ref_idx = seeds.index(42) if 42 in seeds else 0
    ref_doc = docs[ref_idx]
    ref_seed = seeds[ref_idx]

    scalar_sets = [_extract_scalar_metrics(d) for d in docs]
    _check_metric_sets_compatible(scalar_sets, seeds)

    import copy

    result: dict[str, Any] = copy.deepcopy(ref_doc)

    # --- Compute mean and stdev per metric ---
    metric_names = list(scalar_sets[0].keys())
    mean_values: dict[str, float] = {}
    uncertainty: dict[str, dict[str, Any]] = {}

    seeds_str = "/".join(str(s) for s in sorted(seeds))
    note = _MULTISEED_NOTE_TEMPLATE.format(seeds=seeds_str, ref_seed=ref_seed)

    for metric in metric_names:
        vals = [scalars[metric] for scalars in scalar_sets]
        m = mean(vals)
        mean_values[metric] = m
        if len(vals) >= 2:
            sd = stdev(vals)  # sample stdev, ddof=N-1
        else:
            sd = 0.0
        uncertainty[metric] = {
            "ci_low": _clamp_unit(m - sd, metric),
            "ci_high": _clamp_unit(m + sd, metric),
            "method": "multi_seed_std",
            "n_seeds": len(seeds),
            "note": note,
        }

    result["metrics"]["values"] = mean_values
    result["metrics"]["uncertainty"] = uncertainty

    # Ensure schema_version is 1.2.0 (uncertainty block requires it)
    result["schema_version"] = "1.2.0"

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def aggregate(
    staging_dir: Path,
    task: str,
    model: str,
    seeds: list[int],
    out_path: Path,
) -> dict[str, Any]:
    """Read per-seed staging files, aggregate, validate, and write ``out_path``.

    Returns the aggregated document. Raises ``FileNotFoundError`` if a staging file
    is missing, ``ValueError`` for incompatibility or metric mismatch errors.
    """
    # 1. Load staging files
    docs: list[dict[str, Any]] = []
    for seed in seeds:
        staging_file = staging_dir / task / f"{model}-seed{seed}.json"
        if not staging_file.is_file():
            raise FileNotFoundError(
                f"Missing staging file for seed {seed}: {staging_file}\n"
                "Ensure all seeds have been evaluated before aggregating."
            )
        try:
            doc = json.loads(staging_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {staging_file}: {exc}") from exc
        docs.append(doc)
        logger.info("loaded staging file: %s", staging_file)

    # 2. Integrity checks
    _check_run_compatibility(docs, seeds)

    # 3. Aggregate
    result = _aggregate(docs, seeds)

    # 4. Idempotence: if the output already exists and is byte-identical, skip the write
    if out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing == result:
            logger.info("idempotent: %s is already up to date, skipping write", out_path)
            return result

    # 5. Validate before writing
    _validate_or_raise(result)
    logger.info("schema validation passed for %s / %s (%d seeds)", task, model, len(seeds))

    # 6. Atomic write
    _atomic_write_json(result, out_path)
    logger.info(
        "wrote aggregated result to %s (mean across seeds %s)",
        out_path,
        "/".join(str(s) for s in seeds),
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(raw: str) -> list[int]:
    try:
        return [int(s.strip()) for s in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Seeds must be comma-separated integers, got: {raw!r}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-seed staging files "
            f"(default: ${_ENV_STAGING_VAR} or ${_FALLBACK_ENV_VAR}/logs/multiseed)"
        ),
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task name (e.g. amc, interference_id). Used as sub-dir in staging-dir.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. cldnn). Used to build the staging filename.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[42, 43, 44],
        metavar="42,43,44",
        help="Comma-separated seed list (default: 42,43,44).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path for the aggregated result.json "
            "(default: leaderboard/results/<task>/<model>.json relative to repo root)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Resolve staging dir
    staging_dir: Path
    if args.staging_dir is not None:
        staging_dir = args.staging_dir
    else:
        try:
            staging_dir = _default_staging_dir()
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2

    # Resolve output path
    out_path: Path
    if args.out is not None:
        out_path = args.out
    else:
        try:
            out_path = _default_out_path(args.task, args.model)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 2

    try:
        aggregate(
            staging_dir=staging_dir,
            task=args.task,
            model=args.model,
            seeds=args.seeds,
            out_path=out_path,
        )
    except FileNotFoundError as exc:
        logger.error("Missing staging file: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Aggregation error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
