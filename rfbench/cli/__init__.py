"""``rfbench`` command-line interface (Sprint-0 stub surface).

The entry point (``pyproject`` ``[project.scripts]``) is ``rfbench = "rfbench.cli:main"``. The whole
CLI is built on the stdlib :mod:`argparse` (no click/typer -> one fewer runtime dependency). The
full subcommand tree, flags and help strings are wired now; leaf bodies are stubs that print a
one-line intent and exit, EXCEPT ``rfbench eval`` which already emits a schema-valid ``result.json``
skeleton so that downstream tooling/tests have a real artifact to consume.

POSIX exit codes are honoured throughout:

* ``0`` success,
* ``2`` usage error (argparse, or the ``--k-shot`` <-> ``few_shot`` coupling check),
* ``1`` validation / verification failure (e.g. an emitted result fails schema validation).

Paths always default to the repo layout or ``$RFBENCH_CACHE``; nothing is hard-coded to an absolute
location.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from rfbench import __version__

# --- Exit codes (POSIX-ish convention shared across the CLI) ---
EXIT_OK = 0
EXIT_FAILURE = 1  # validation / verification failure
EXIT_USAGE = 2  # usage error

# --- Canonical enumerations (in sync with docs/EVALUATION_PROTOCOL.md + the JSON schema) ---
TASK_NAMES: tuple[str, ...] = ("amc", "sei", "wideband_detection", "spectrum_sensing")
REGIME_NAMES: tuple[str, ...] = ("from_scratch", "full_finetune", "linear_probe", "few_shot")
SPLIT_NAMES: tuple[str, ...] = ("test", "val")
DATASET_NAMES: tuple[str, ...] = (
    "radioml_2016_10a",
    "radioml_2018_01a",
    "sig53",
    "wisig",
    "oracle",
    "lora_rffi",
    "wbsig53",
    "deepsense",
)

# Per-task Sprint-0 defaults used to assemble the eval skeleton. Kept minimal and
# normative-adjacent: the real values become authoritative once the task registry (WP-20..23) lands.
_TASK_DEFAULTS: dict[str, dict[str, str]] = {
    "amc": {
        "dataset": "radioml_2016_10a",
        "primary": "accuracy_overall",
        "canonical_split_id": "amc-strat-snr-seed42-v1",
        "track": "closed_set",
    },
    "sei": {
        "dataset": "wisig",
        "primary": "rank1_accuracy",
        "canonical_split_id": "sei-closed-set-seed42-v1",
        "track": "closed_set",
    },
    "wideband_detection": {
        "dataset": "wbsig53",
        "primary": "mAP",
        "canonical_split_id": "wideband-detection-seed42-v1",
        "track": "detection",
    },
    "spectrum_sensing": {
        "dataset": "deepsense",
        "primary": "pd@pfa=0.1",
        "canonical_split_id": "spectrum-sensing-seed42-v1",
        "track": "occupancy",
    },
}

# 64-zero SHA-256 placeholder: satisfies the schema pattern until real split indices exist (WP-10).
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


# --------------------------------------------------------------------------------------------------
# Schema resolution + validation (used by `rfbench eval` before it writes result.json)
# --------------------------------------------------------------------------------------------------
def _resolve_schema_path(schema_name: str) -> Path | None:
    """Locate a JSON schema file.

    Resolution order matches the packaging contract: prefer the force-included package data
    (``rfbench/_schemas`` in an installed wheel), then fall back to the repo ``schemas/`` directory
    when running from a source checkout. Returns ``None`` if neither location has the file.
    """
    try:
        packaged = resources.files("rfbench").joinpath("_schemas").joinpath(schema_name)
        if packaged.is_file():
            with resources.as_file(packaged) as concrete:
                return concrete
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    # Source checkout: walk up from this file to the repo root's schemas/ directory.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schemas" / schema_name
        if candidate.is_file():
            return candidate
    return None


def _validate_result(document: dict[str, Any]) -> list[str]:
    """Validate a result document against ``result.schema.json``.

    Returns a list of human-readable error messages (empty means valid). If the schema cannot be
    located or ``jsonschema`` is unavailable, a single explanatory error is returned so the caller
    fails loudly rather than silently emitting an unvalidated artifact.
    """
    schema_path = _resolve_schema_path("result.schema.json")
    if schema_path is None:
        return ["could not locate result.schema.json (checked rfbench/_schemas and repo schemas/)"]

    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        return ["jsonschema is not installed; run `pip install rfbench` to validate result.json"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in errors
    ]


# --------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------
def _default_cache() -> str:
    """Return the dataset cache dir: ``$RFBENCH_CACHE`` if set, else local ``.rfbench_cache``."""
    return os.environ.get("RFBENCH_CACHE", str(Path.cwd() / ".rfbench_cache"))


def _print_intent(message: str, *, verbose: bool) -> None:
    """Emit a one-line intent for a stub subcommand (and a hint when verbose)."""
    print(message)
    if verbose:
        print("  (Sprint-0 stub: no side effects were performed.)")


# --------------------------------------------------------------------------------------------------
# Leaf handlers
# --------------------------------------------------------------------------------------------------
def _cmd_data_download(args: argparse.Namespace) -> int:
    _print_intent(
        f"[data download] would fetch raw '{args.dataset}' into cache={args.cache} "
        "(writes no git-tracked files) — not yet implemented (WP-11..14).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    _print_intent(
        f"[data prepare] would build deterministic splits for '{args.target}' "
        f"(dataset={args.dataset}, seed={args.seed}, cache={args.cache}, force={args.force}) -> "
        "leaderboard/splits/<dataset>/<id>.idx.json (+checksum) — not yet implemented (WP-10..14).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_data_list(args: argparse.Namespace) -> int:
    _print_intent(
        "[data list] would list known datasets/tasks and their prepared status "
        f"(known datasets: {', '.join(DATASET_NAMES)}) — not yet implemented (WP-14).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_data_verify(args: argparse.Namespace) -> int:
    target = args.dataset if args.dataset else "all prepared datasets"
    _print_intent(
        f"[data verify] would recompute split-index checksums for {target} and diff vs the "
        "versioned manifest — not yet implemented (WP-14).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _emit_target_path(args: argparse.Namespace) -> Path:
    """Resolve the result.json output path (``--emit`` or the default leaderboard layout)."""
    if args.emit:
        return Path(str(args.emit))
    return Path("leaderboard") / "results" / str(args.task) / f"{args.model}.json"


def _build_result_skeleton(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble a minimal, schema-valid ``result.json`` skeleton for the eval stub."""
    defaults = _TASK_DEFAULTS[args.task]
    dataset_name = args.dataset if args.dataset else defaults["dataset"]
    track = args.track if args.track else defaults["track"]
    primary = defaults["primary"]

    regime: dict[str, Any] = {"name": args.regime}
    if args.regime == "few_shot":
        regime["k_shot"] = args.k_shot

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": {"name": args.task, "version": "v1"},
        "model": {"name": args.model},
        "regime": regime,
        "dataset": {"name": dataset_name},
        "split": {
            "canonical_split_id": defaults["canonical_split_id"],
            "name": args.split,
            "track": track,
            "seed": args.seed,
            "checksum": _PLACEHOLDER_CHECKSUM,
        },
        "metrics": {
            "primary": primary,
            "values": {primary: 0.0},
        },
        "eval": {
            "batch_size": args.batch_size,
        },
        "environment": {
            "seed": args.seed,
            "rfbench_version": __version__,
            "python_version": ".".join(str(n) for n in sys.version_info[:3]),
        },
        "verification": {"status": "self_reported"},
    }
    return result


def _cmd_eval(args: argparse.Namespace) -> int:
    # The --k-shot <-> few_shot coupling is a usage-level contract (exit 2).
    if args.regime == "few_shot" and args.k_shot is None:
        print("error: --k-shot is REQUIRED when --regime few_shot", file=sys.stderr)
        return EXIT_USAGE
    if args.regime != "few_shot" and args.k_shot is not None:
        print(
            f"error: --k-shot is FORBIDDEN unless --regime few_shot (got --regime {args.regime})",
            file=sys.stderr,
        )
        return EXIT_USAGE

    result = _build_result_skeleton(args)

    errors = _validate_result(result)
    if errors:
        print("error: emitted result.json failed schema validation:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return EXIT_FAILURE

    if args.dry_run:
        print("[eval] --dry-run: assembled and schema-validated result.json (not written).")
        if args.verbose:
            print(json.dumps(result, indent=2))
        return EXIT_OK

    out_path = _emit_target_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[eval] wrote schema-valid result.json skeleton -> {out_path} "
        "(Sprint-0 stub: metrics are placeholders until WP-40 lands)."
    )
    return EXIT_OK


def _cmd_submit(args: argparse.Namespace) -> int:
    _print_intent(
        f"[submit --check] would locally mirror validate-submission.yml for '{args.result}' "
        f"(manifest={args.manifest}) — not yet implemented (WP-51).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_leaderboard_build(args: argparse.Namespace) -> int:
    _print_intent(
        f"[leaderboard build] would read {args.results}/**.json and render a static site to "
        f"{args.out} (sorted by metrics.primary, columned by regime, verified/self_reported "
        "badges) — not yet implemented (WP-50).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_leaderboard_validate(args: argparse.Namespace) -> int:
    _print_intent(
        f"[leaderboard validate] would validate every row under {args.results} against "
        "result.schema.json — not yet implemented (WP-50).",
        verbose=args.verbose,
    )
    return EXIT_OK


def _cmd_verify(args: argparse.Namespace) -> int:
    _print_intent(
        f"[verify] would re-run '{args.result}' per manifest={args.manifest} (mode={args.mode}, "
        f"by={args.by}, hardware={args.hardware}, device={args.device}) and flip "
        "verification.status self_reported -> verified within tolerance — "
        "not yet implemented (WP-53).",
        verbose=args.verbose,
    )
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# Parser construction
# --------------------------------------------------------------------------------------------------
def _global_flags_parser() -> argparse.ArgumentParser:
    """Build a reusable parent parser carrying the global flags.

    Attaching it both to the top-level parser and to every leaf subparser lets ``--cache`` and
    ``-v/--verbose`` appear on either side of the subcommand (``rfbench -v eval ...`` and
    ``rfbench eval ... -v`` are both accepted).
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--cache",
        default=_default_cache(),
        metavar="DIR",
        help="Dataset cache directory (default: $RFBENCH_CACHE or ./.rfbench_cache).",
    )
    parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print extra detail.",
    )
    return parent


def _build_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    data = subparsers.add_parser(
        "data",
        help="Download / prepare / list / verify datasets and their canonical splits (M1).",
        description="Data-layer commands. Never versions raw data; only split indices + checksums.",
    )
    data_sub = data.add_subparsers(dest="subcommand", metavar="<verb>", required=True)

    download = data_sub.add_parser(
        "download",
        parents=[parent],
        help="Fetch raw data from the official source into the cache (no git-tracked files).",
    )
    download.add_argument("dataset", help="Dataset id to download.")
    download.set_defaults(func=_cmd_data_download)

    prepare = data_sub.add_parser(
        "prepare",
        parents=[parent],
        help="Build deterministic splits + manifest into leaderboard/splits/ (idempotent).",
    )
    prepare.add_argument("target", help="Task or dataset to prepare.")
    prepare.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        help="Concrete dataset variant to prepare.",
    )
    prepare.add_argument("--seed", type=int, default=42, help="Split seed (default: 42).")
    prepare.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the split index already exists.",
    )
    prepare.set_defaults(func=_cmd_data_prepare)

    list_p = data_sub.add_parser(
        "list",
        parents=[parent],
        help="List known datasets/tasks and their prepared status.",
    )
    list_p.set_defaults(func=_cmd_data_list)

    verify_p = data_sub.add_parser(
        "verify",
        parents=[parent],
        help="Recompute split-index checksums and diff vs the versioned manifest (exit 1 on diff).",
    )
    verify_p.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset id to verify (default: all prepared datasets).",
    )
    verify_p.set_defaults(func=_cmd_data_verify)


def _build_eval_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    ev = subparsers.add_parser(
        "eval",
        parents=[parent],
        help="Evaluate a model on a task and emit a schema-valid result.json (ONLY emitter, M4).",
        description=(
            "Produces a result.json validated against schemas/result.schema.json BEFORE write. The "
            "regime is written verbatim, never inferred; verification.status = 'self_reported'."
        ),
    )
    ev.add_argument("task", choices=TASK_NAMES, help="Task to evaluate.")
    ev.add_argument("--model", required=True, metavar="NAME", help="Model name (required).")
    ev.add_argument(
        "--regime",
        required=True,
        choices=REGIME_NAMES,
        help="Adaptation regime (required, ALWAYS declared).",
    )
    ev.add_argument(
        "--k-shot",
        dest="k_shot",
        type=int,
        default=None,
        metavar="INT",
        help="Examples per class. REQUIRED iff --regime few_shot, else FORBIDDEN.",
    )
    ev.add_argument(
        "--dataset",
        metavar="NAME",
        help="Dataset name (default: the task's first canonical dataset).",
    )
    ev.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default="test",
        help="Partition to score (default: test).",
    )
    ev.add_argument(
        "--track",
        metavar="NAME",
        help="Sub-protocol / evaluation condition (default: the task's primary track).",
    )
    ev.add_argument("--seed", type=int, default=42, help="Global seed (default: 42).")
    ev.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    ev.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Compute device (default: auto).",
    )
    ev.add_argument(
        "--emit",
        metavar="PATH",
        help="Output path (default: leaderboard/results/<task>/<model>.json).",
    )
    ev.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Assemble + validate but do not write.",
    )
    ev.set_defaults(func=_cmd_eval)


def _build_submit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    submit = subparsers.add_parser(
        "submit",
        parents=[parent],
        help="Local dry-run of PR CI for a result.json (mirrors validate-submission.yml, M5).",
        description="Contributor-side pre-flight: exit 0 = PR-ready, else 1.",
    )
    submit.add_argument(
        "--check",
        dest="result",
        required=True,
        metavar="RESULT.JSON",
        help="Path to the result.json to check.",
    )
    submit.add_argument(
        "--manifest",
        metavar="PATH",
        help="Standalone submission manifest if not embedded / referenced by submission_ref.",
    )
    submit.set_defaults(func=_cmd_submit)


def _build_leaderboard_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    lb = subparsers.add_parser(
        "leaderboard",
        help="Build or validate the static leaderboard site from results/**.json (M5).",
        description="Never mixes two regimes (or two tracks) in one column.",
    )
    lb_sub = lb.add_subparsers(dest="subcommand", metavar="<verb>", required=True)

    build = lb_sub.add_parser(
        "build",
        parents=[parent],
        help="Render results/**.json into a static site.",
    )
    build.add_argument(
        "--results",
        default="leaderboard/results",
        metavar="DIR",
        help="Results directory (default: leaderboard/results).",
    )
    build.add_argument(
        "--out",
        default="site_build",
        metavar="DIR",
        help="Output site directory (default: site_build).",
    )
    build.set_defaults(func=_cmd_leaderboard_build)

    validate = lb_sub.add_parser(
        "validate",
        parents=[parent],
        help="Validate every row against result.schema.json (exit 1 on any invalid row).",
    )
    validate.add_argument(
        "--results",
        default="leaderboard/results",
        metavar="DIR",
        help="Results directory (default: leaderboard/results).",
    )
    validate.set_defaults(func=_cmd_leaderboard_validate)


def _build_verify_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    verify = subparsers.add_parser(
        "verify",
        parents=[parent],
        help="Maintainer re-run that flips self_reported -> verified within tolerance (M5).",
        description="The ONLY writer of verification.status = 'verified'.",
    )
    verify.add_argument("result", help="Path to the result.json to verify.")
    verify.add_argument(
        "--manifest",
        required=True,
        metavar="PATH",
        help="The submission.schema.json manifest (required).",
    )
    verify.add_argument(
        "--mode",
        choices=("eval_only", "full_retrain"),
        default="eval_only",
        help="Re-run mode (default: eval_only, the cost guard-rail).",
    )
    verify.add_argument(
        "--by",
        metavar="HANDLE",
        help="Maintainer handle -> verification.verified_by.",
    )
    verify.add_argument(
        "--hardware",
        metavar="STR",
        help="Hardware string -> verification.verified_hardware.",
    )
    verify.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Compute device (default: cuda).",
    )
    verify.set_defaults(func=_cmd_verify)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse surface for the ``rfbench`` CLI."""
    parser = argparse.ArgumentParser(
        prog="rfbench",
        description=(
            "RF-Benchmark-Hub: reproducible benchmarks + leaderboard harness for terrestrial RF ML "
            "tasks."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the package version and exit.",
    )

    # Global flags (`--cache`, `-v/--verbose`) are attached to every leaf subparser via this shared
    # parent, so they are accepted after the subcommand (`rfbench eval ... -v`). They are NOT
    # duplicated on the top-level parser: argparse resets subparser defaults, which would silently
    # clobber a pre-subcommand value.
    parent = _global_flags_parser()

    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    _build_data_parser(subparsers, parent)
    _build_eval_parser(subparsers, parent)
    _build_submit_parser(subparsers, parent)
    _build_leaderboard_parser(subparsers, parent)
    _build_verify_parser(subparsers, parent)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX exit code (0 ok, 2 usage error, 1 validation failure)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - argparse `required=True` guards this.
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
