"""``rfbench`` command-line interface (WP-42: wired to the real harness).

The entry point (``pyproject`` ``[project.scripts]``) is ``rfbench = "rfbench.cli:main"``. The whole
CLI is built on the stdlib :mod:`argparse` (no click/typer -> one fewer runtime dependency). The
full subcommand tree, flags and help strings are stable; the leaf bodies now dispatch to the real
implementations:

* ``data prepare`` -> :mod:`rfbench.data.prepare` (``prepare_amc`` / ``prepare_sei`` /
  ``prepare_detection``): builds deterministic split indices + manifests under
  ``leaderboard/splits/<dataset>/``. The split-GENERATION path is exercisable on pure-stdlib
  synthetic labels via ``--labels-file``; without it the CLI drives the lazy cluster-only loaders
  (``rfbench[data]`` / ``rfbench[detection]``).
* ``data download`` -> :mod:`rfbench.data.download` (requests/torchsig lazy; cluster-only).
* ``data list`` -> enumerate the known tasks/datasets and their prepared status.
* ``data verify`` -> recompute on-disk split-index checksums and diff vs the manifest.
* ``eval`` -> assembles a schema-valid ``result.json`` through a :mod:`rfbench.regimes` adapter
  (``make_adapter(RegimeSpec)``), preserving the ``--k-shot`` <-> ``few_shot`` coupling and the
  frozen result contract.
* ``train`` -> runs the real from-scratch / full-finetune training loop
  (:func:`rfbench.training.train_baseline`) for a registered baseline (e.g. ``mcldnn`` on
  ``radioml_2016_10a``) and then emits the trained model's ``result.json`` via ``evaluate``. The
  torch model module is imported lazily inside the handler (M3, GPU/cluster).
* ``submit --check`` -> validate a ``result.json`` against ``schemas/result.schema.json`` plus a
  manifest-completeness gate: any manifest supplied via ``--manifest`` or referenced by the result's
  ``submission_ref`` is validated for completeness against ``submission.schema.json`` and
  cross-checked for consistency with the result it reproduces. On top of schema shape it enforces
  the full-protocol / comparability gates promised by CONTRIBUTING.md: AMC must record the full SNR
  range (``eval.conditions.full_snr_range == true``, no cherry-picking); no committed sibling may
  already occupy the same ``(task, model, regime, dataset, split, track)`` cell (anti-doublon); and
  the row's ``split.checksum`` must match the committed split index under
  ``leaderboard/splits/<dataset>/`` when one exists.
* ``leaderboard build`` -> :func:`leaderboard.site.generate.build_site`.
* ``verify`` -> :mod:`rfbench.verify` (WP-53): compares a re-run's metrics against the manifest's
  ``expected_metrics`` within ``tolerance`` and, on a match, writes a NEW result.json with
  ``verification.status='verified'`` and provenance stamped.

Every heavy or optional import (``jsonschema``, numpy/h5py/torch/torchsig/requests, and the
``leaderboard/site`` generator) is performed LAZILY inside the handler that needs it, so
``import rfbench`` and ``rfbench --help`` stay dependency-free.

POSIX exit codes are honoured throughout:

* ``0`` success,
* ``2`` usage error (argparse, or the ``--k-shot`` <-> ``few_shot`` coupling check),
* ``1`` validation / verification / runtime failure (e.g. a result fails schema validation).

Paths always default to the repo layout or ``$RFBENCH_CACHE``; nothing is hard-coded to an absolute
location.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from rfbench import __version__

if TYPE_CHECKING:
    from rfbench.core.model import RegimeSpec

#: A ``--labels-file`` payload: pre-extracted labels/records/samples keyed by family field.
LabelsPayload = dict[str, Any]

# --- Exit codes (POSIX-ish convention shared across the CLI) ---
EXIT_OK = 0
EXIT_FAILURE = 1  # validation / verification / runtime failure
EXIT_USAGE = 2  # usage error

# --- Canonical enumerations (in sync with docs/EVALUATION_PROTOCOL.md + the JSON schema) ---
TASK_NAMES: tuple[str, ...] = (
    "amc",
    "sei",
    "wideband_detection",
    "spectrum_sensing",
    "interference_id",
    "protocol_tech_id",
    "snr_estimation",
)
REGIME_NAMES: tuple[str, ...] = ("from_scratch", "full_finetune", "linear_probe", "few_shot")
SPLIT_NAMES: tuple[str, ...] = ("test", "val")
DATASET_NAMES: tuple[str, ...] = (
    "radioml_2016_10a",
    "radioml_2018_01a",
    "sig53",
    "wisig",
    "oracle",
    "lora",
    "powder",
    "raddet",
    "wbsig53",
    "deepsense",
    "interf_gnss6",
    "tprime_wifi4",
)

#: Which prepare task each dataset belongs to, and the family of prepare_* it dispatches to.
#: Kept in lockstep with the ``rfbench.data.prepare`` modules (WP-11/12/13).
_DATASET_FAMILY: dict[str, str] = {
    "radioml_2016_10a": "amc",
    "radioml_2018_01a": "amc",
    "sig53": "amc",
    "wisig": "sei",
    "oracle": "sei",
    "lora": "sei",
    "powder": "sei",
    "raddet": "detection",
    "wbsig53": "detection",
    "interf_gnss6": "interference",
    "tprime_wifi4": "protocol",
}

#: Datasets a given data task can prepare (task name -> concrete datasets).
#: Detection targets RadDet (the real published ICASSP-2025 artifact); wbsig53 stays a
#: reachable name but only yields a blocker (generation-only, no static release).
#: NOTE: ``snr_estimation`` is intentionally absent here. Its canonical split is *derived*
#: byte-identically from the AMC ``radioml_2016_10a`` split (see
#: rfbench.data.prepare.snr_estimation.derive_from_amc_split), so it is not a standalone
#: ``rfbench data prepare`` target: prepare ``radioml_2016_10a`` (family ``amc``) and the SNR
#: split falls out with the same indices/checksum.
_TASK_DATASETS: dict[str, tuple[str, ...]] = {
    "amc": ("radioml_2016_10a", "radioml_2018_01a", "sig53"),
    "sei": ("wisig", "oracle", "lora", "powder"),
    "wideband_detection": ("raddet",),
    "detection": ("raddet",),
    "interference_id": ("interf_gnss6",),
    "protocol_tech_id": ("tprime_wifi4",),
}

# Per-task eval defaults used to assemble the result.json when no task registry row exists yet
# (the tasks/ registry lands with WP-20..23). The regime is still declared via a real
# rfbench.regimes adapter so the k<->few_shot coupling is enforced by the same code paths eval uses.
_TASK_DEFAULTS: dict[str, dict[str, str]] = {
    "amc": {
        "dataset": "radioml_2016_10a",
        "primary": "accuracy_overall",
        "canonical_split_id": "amc-radioml2016-strat-snr-8010-seed42-v1",
        "track": "closed_set",
    },
    "sei": {
        "dataset": "wisig",
        "primary": "rank1_accuracy",
        "canonical_split_id": "sei-wisig-closedset-strat-tx-8010-seed42-v1",
        "track": "closed_set",
    },
    "wideband_detection": {
        "dataset": "wbsig53",
        "primary": "mAP",
        "canonical_split_id": "detect-wbsig53-detection-8010-seed42-v1",
        "track": "detection",
    },
    "spectrum_sensing": {
        "dataset": "deepsense",
        "primary": "pd@pfa=0.1",
        "canonical_split_id": "sensing-deepsense-8010-seed42-v1",
        "track": "occupancy",
    },
    "interference_id": {
        "dataset": "interf_gnss6",
        "primary": "accuracy_overall",
        "canonical_split_id": "interf-gnss6-8010-seed42-v1",
        "track": "closed_set",
    },
    "protocol_tech_id": {
        "dataset": "tprime_wifi4",
        "primary": "accuracy_overall",
        "canonical_split_id": "proto-tprime-wifi4-8010-seed42-v1",
        "track": "closed_set",
    },
    "snr_estimation": {
        # Regression (raw-IQ -> SNR dB): primary is rmse_db (lower-is-better). The split is
        # derived from the AMC radioml_2016_10a partition (byte-identical indices, own id).
        "dataset": "radioml_2016_10a",
        "primary": "rmse_db",
        "canonical_split_id": "snr-radioml2016-strat-snr-8010-seed42-v1",
        "track": "all_snr",
    },
}

# 64-zero SHA-256 placeholder: satisfies the schema pattern until real split indices exist (WP-10).
_PLACEHOLDER_CHECKSUM = "sha256:" + "0" * 64


# --------------------------------------------------------------------------------------------------
# Schema resolution + validation (used by `rfbench eval` and `rfbench submit --check`)
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


def _validate_against_schema(document: dict[str, Any], schema_name: str) -> list[str]:
    """Validate ``document`` against ``schema_name`` and return human-readable error strings.

    Returns an empty list when the document is valid. If the schema cannot be located or
    ``jsonschema`` is unavailable, a single explanatory error is returned so the caller fails
    loudly rather than silently accepting an unvalidated artifact. ``jsonschema`` is imported
    lazily so ``import rfbench`` stays dependency-free.
    """
    schema_path = _resolve_schema_path(schema_name)
    if schema_path is None:
        return [f"could not locate {schema_name} (checked rfbench/_schemas and repo schemas/)"]

    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        return ["jsonschema is not installed; run `pip install rfbench` to validate JSON"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in errors
    ]


def _validate_result(document: dict[str, Any]) -> list[str]:
    """Validate a result document against ``result.schema.json``.

    Thin wrapper over :func:`_validate_against_schema`.
    """
    return _validate_against_schema(document, "result.schema.json")


# --------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------
def _default_cache() -> str:
    """Return the dataset cache dir: ``$RFBENCH_CACHE`` if set, else local ``.rfbench_cache``."""
    return os.environ.get("RFBENCH_CACHE", str(Path.cwd() / ".rfbench_cache"))


def _repo_root() -> Path:
    """Best-effort repo root: the nearest ancestor of this file that holds a ``schemas/`` dir.

    Used to default ``data prepare``'s output tree to ``<repo>/leaderboard`` and to locate the
    non-packaged ``leaderboard/site/generate.py`` module. Falls back to the current working
    directory when running from an installed wheel with no source checkout nearby.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "schemas").is_dir():
            return parent
    return Path.cwd()


def _print_intent(message: str, *, verbose: bool) -> None:
    """Emit a one-line intent for a stub subcommand (and a hint when verbose)."""
    print(message)
    if verbose:
        print("  (stub: no side effects were performed.)")


# --------------------------------------------------------------------------------------------------
# `data` handlers
# --------------------------------------------------------------------------------------------------
def _resolve_prepare_targets(target: str, dataset: str | None) -> list[str]:
    """Map a ``prepare`` target (task or dataset) + optional ``--dataset`` to concrete datasets.

    ``target`` may be a data task (``amc`` / ``sei`` / ``wideband_detection`` / ``detection``) or a
    concrete dataset id. ``--dataset`` narrows a task target to a single dataset. Raises
    ``ValueError`` (surfaced as a usage error) on an unknown/mismatched combination.
    """
    if dataset is not None:
        if dataset not in DATASET_NAMES:
            raise ValueError(f"unknown --dataset {dataset!r}; known: {', '.join(DATASET_NAMES)}")

    if target in _TASK_DATASETS:
        datasets = list(_TASK_DATASETS[target])
        if dataset is not None:
            if dataset not in datasets:
                raise ValueError(
                    f"--dataset {dataset!r} is not part of task {target!r} "
                    f"(datasets: {', '.join(datasets)})"
                )
            return [dataset]
        return datasets

    # target is (presumably) a concrete dataset id.
    if target in _DATASET_FAMILY:
        if dataset is not None and dataset != target:
            raise ValueError(
                f"target {target!r} already names a dataset; drop --dataset {dataset!r}"
            )
        return [target]

    raise ValueError(
        f"unknown prepare target {target!r}; expected a task ({', '.join(_TASK_DATASETS)}) "
        f"or a dataset ({', '.join(_DATASET_FAMILY)})"
    )


def _load_labels_file(path: str) -> LabelsPayload:
    """Load a synthetic labels/records/samples payload for ``data prepare`` (pure stdlib JSON).

    The file lets the split-GENERATION path be driven without the heavy cluster loaders (used by
    tests and by advanced users who pre-extracted labels). Its shape is per-family:

    * ``amc``: ``{"labels": [[mod, snr], ...]}`` (stratify by mod x snr) or
      ``{"official_split": {"train": [...], "val": [...], "test": [...]}}`` for Sig53;
    * ``sei``: ``{"records": [[tx, rx, day], ...]}``;
    * ``detection``: ``{"samples": [{"boxes": [...], "sample_id"?}, ...],
      "official_split"?: {...}, "track"?: "detection"|"recognition"}``.
    * ``interference``: ``{"labels": ["DME", "narrowband", ...]}`` (one class name per item;
      stratify by class).
    * ``protocol``: ``{"labels": ["802.11b", "802.11g", ...]}`` (one class name per item;
      stratify by class).
    """
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(
            f"--labels-file {path} must contain a JSON object, got {type(loaded).__name__}"
        )
    payload: LabelsPayload = loaded
    return payload


def _prepare_one(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    """Prepare a single dataset's split, returning the canonical split id written.

    ``payload`` (from ``--labels-file``) supplies synthetic labels so the split path runs on pure
    stdlib. When ``payload`` is ``None`` the heavy cluster-only loaders are invoked lazily to
    extract labels from the real files under ``cache`` -- that branch NEEDS ``rfbench[data]`` /
    ``rfbench[detection]`` and is never taken in unit tests.
    """
    family = _DATASET_FAMILY[dataset]
    if family == "amc":
        return _prepare_amc(dataset, out_dir, payload, seed, cache)
    if family == "sei":
        return _prepare_sei(dataset, out_dir, payload, seed, cache)
    if family == "detection":
        return _prepare_detection(dataset, out_dir, payload, seed, cache)
    if family == "interference":
        return _prepare_interference(dataset, out_dir, payload, seed, cache)
    if family == "protocol":
        return _prepare_protocol(dataset, out_dir, payload, seed, cache)
    raise ValueError(f"no prepare family registered for dataset {dataset!r}")  # pragma: no cover


def _prepare_amc(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    from rfbench.data.prepare.amc import prepare_amc

    labels: Any = None
    official_split: Any = None
    if payload is not None:
        labels = payload.get("labels")
        official_split = payload.get("official_split")
    elif dataset == "sig53":
        from rfbench.data.prepare.amc import load_sig53_official_split

        official_split = load_sig53_official_split(cache=cache)
    else:
        from rfbench.data.prepare.amc import load_radioml_labels

        labels = load_radioml_labels(dataset, cache=cache)  # type: ignore[arg-type]

    split, _manifest = prepare_amc(
        dataset,
        out_dir=out_dir,
        labels=[tuple(x) for x in labels] if labels is not None else None,
        official_split=official_split,
        seed=seed,
    )
    return split.canonical_split_id


def _prepare_sei(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    from rfbench.data.prepare.sei import CANONICAL_SPLIT_IDS as SEI_IDS
    from rfbench.data.prepare.sei import prepare_sei

    if payload is not None:
        records = [tuple(r) for r in payload["records"]]
        conditions = payload.get("conditions") or list(SEI_IDS[dataset].keys())
    else:
        from rfbench.data.prepare.sei import (
            load_lora_records,
            load_oracle_records,
            load_powder_records,
            load_wisig_records,
        )

        if dataset == "wisig":
            records = load_wisig_records(cache=cache)
        elif dataset == "oracle":
            records = load_oracle_records(cache=cache)
        elif dataset == "powder":
            records = load_powder_records(cache=cache)
        else:  # lora
            records = load_lora_records(cache=cache)
        conditions = list(SEI_IDS[dataset].keys())

    written: list[str] = []
    for condition in conditions:
        split, _manifest = prepare_sei(
            dataset,
            condition,
            out_dir=out_dir,
            records=records,
            seed=seed,
        )
        written.append(split.canonical_split_id)
    return ", ".join(written)


def _prepare_detection(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    from rfbench.data.prepare.detection import prepare_detection

    if payload is not None:
        samples = payload["samples"]
        official_split = payload.get("official_split")
        track = payload.get("track", "detection")
    else:
        from rfbench.data.download.detection_wbsig53 import (
            load_raddet_annotations,
            load_wbsig53_annotations,
        )

        samples = (
            load_raddet_annotations(cache=cache)
            if dataset == "raddet"
            else load_wbsig53_annotations(cache=cache)
        )
        official_split = None
        track = "detection"

    split, _manifest, _annotations = prepare_detection(
        dataset,
        out_dir=out_dir,
        samples=samples,
        track=track,
        official_split=official_split,
        seed=seed,
    )
    return split.canonical_split_id


def _prepare_interference(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    from rfbench.data.prepare.interference import prepare_interference

    if payload is not None:
        labels = [str(label) for label in payload["labels"]]
    else:
        from rfbench.data.prepare.interference import load_interference_labels

        labels = load_interference_labels(dataset, cache=cache)  # type: ignore[arg-type]

    split, _manifest = prepare_interference(
        dataset,
        out_dir=out_dir,
        labels=labels,
        seed=seed,
    )
    return split.canonical_split_id


def _prepare_protocol(
    dataset: str, out_dir: Path, payload: LabelsPayload | None, seed: int, cache: str
) -> str:
    from rfbench.data.prepare.protocol import prepare_protocol

    if payload is not None:
        labels = [str(label) for label in payload["labels"]]
    else:
        from rfbench.data.prepare.protocol import load_protocol_labels

        labels = load_protocol_labels(dataset, cache=cache)  # type: ignore[arg-type]

    split, _manifest = prepare_protocol(
        dataset,
        out_dir=out_dir,
        labels=labels,
        seed=seed,
    )
    return split.canonical_split_id


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    try:
        datasets = _resolve_prepare_targets(args.target, args.dataset)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    out_dir = Path(args.out) if getattr(args, "out", None) else _repo_root() / "leaderboard"
    payload = _load_labels_file(args.labels_file) if getattr(args, "labels_file", None) else None

    for dataset in datasets:
        split_dir = out_dir / "splits" / dataset
        if not args.force and split_dir.is_dir() and any(split_dir.glob("*.idx.json")):
            print(
                f"[data prepare] {dataset}: split index already present under {split_dir} "
                "(skip; --force to rebuild)."
            )
            continue
        try:
            written = _prepare_one(dataset, out_dir, payload, args.seed, args.cache)
        except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
            print(f"error: [data prepare] {dataset}: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        print(
            f"[data prepare] {dataset}: wrote split index(es) {written} "
            f"under {out_dir}/splits/{dataset}/"
        )
    return EXIT_OK


def _cmd_data_download(args: argparse.Namespace) -> int:
    dataset = args.dataset
    if dataset not in _DATASET_FAMILY and dataset not in DATASET_NAMES:
        print(
            f"error: unknown dataset {dataset!r}; known: {', '.join(DATASET_NAMES)}",
            file=sys.stderr,
        )
        return EXIT_USAGE
    print(
        f"[data download] fetching raw '{dataset}' into cache={args.cache} (heavy deps + network "
        "are lazy; real runs need rfbench[data]/rfbench[detection] on the cluster)."
    )
    try:
        path = _download_dispatch(
            dataset,
            cache=args.cache,
            source_url=args.source_url,
            manual_archive=args.manual_archive,
        )
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"error: [data download] {dataset}: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"[data download] {dataset}: available at {path}")
    return EXIT_OK


def _download_dispatch(
    dataset: str, *, cache: str, source_url: str | None, manual_archive: str | None = None
) -> Path:
    """Dispatch a dataset id to its (lazy) download/generation function (cluster-only)."""
    if dataset in ("radioml_2016_10a", "radioml_2018_01a"):
        from rfbench.data.download.amc_radioml import download_radioml

        return download_radioml(dataset, source_url=source_url, cache=cache)  # type: ignore[arg-type]
    if dataset == "sig53":
        from rfbench.data.download.amc_sig53 import download_sig53

        return download_sig53(cache=cache)
    if dataset == "wisig":
        from rfbench.data.download.sei_wisig import download_wisig

        return download_wisig(source_url=source_url, cache=cache)
    if dataset == "oracle":
        from rfbench.data.download.sei_oracle import download_oracle

        return download_oracle(source_url=source_url, cache=cache)
    if dataset == "lora":
        from rfbench.data.download.sei_lora import download_lora

        return download_lora(source_url=source_url, cache=cache)
    if dataset == "raddet":
        from rfbench.data.download.detection_wbsig53 import download_raddet

        return download_raddet(cache=cache)
    if dataset == "wbsig53":
        from rfbench.data.download.detection_wbsig53 import generate_wbsig53

        return generate_wbsig53(cache=cache)
    if dataset == "interf_gnss6":
        from rfbench.data.download.interference_gnss import download_interference_gnss6

        return download_interference_gnss6(source_url=source_url, cache=cache)
    if dataset == "tprime_wifi4":
        from rfbench.data.download.protocol_tprime import download_tprime_wifi4

        return download_tprime_wifi4(
            source_url=source_url, cache=cache, manual_archive=manual_archive
        )
    raise ValueError(
        f"no download function wired for {dataset!r} yet "
        "(known: radioml_2016_10a, radioml_2018_01a, sig53, wisig, oracle, lora, raddet, "
        "wbsig53, interf_gnss6, tprime_wifi4)."
    )


def _cmd_data_list(args: argparse.Namespace) -> int:
    out_dir = Path(args.out) if getattr(args, "out", None) else _repo_root() / "leaderboard"
    print("Known data tasks and datasets (prepared = a split index under leaderboard/splits/):")
    for task, datasets in _TASK_DATASETS.items():
        if task == "detection":  # alias of wideband_detection; list once
            continue
        print(f"  {task}:")
        for dataset in datasets:
            split_dir = out_dir / "splits" / dataset
            ids = sorted(p.name[: -len(".idx.json")] for p in split_dir.glob("*.idx.json"))
            status = f"prepared ({', '.join(ids)})" if ids else "not prepared"
            print(f"    - {dataset}: {status}")
    if args.verbose:
        print(f"  (scanned {out_dir / 'splits'}; datasets download to cache={args.cache})")
    return EXIT_OK


def _cmd_data_verify(args: argparse.Namespace) -> int:
    from rfbench.core.splits import split_checksum

    out_dir = Path(args.out) if getattr(args, "out", None) else _repo_root() / "leaderboard"
    splits_root = out_dir / "splits"
    if args.dataset is not None:
        idx_dirs = [splits_root / args.dataset]
    elif splits_root.is_dir():
        idx_dirs = sorted(p for p in splits_root.glob("*") if p.is_dir())
    else:
        idx_dirs = []

    idx_files = [p for d in idx_dirs for p in sorted(d.glob("*.idx.json"))]
    if not idx_files:
        target = args.dataset if args.dataset else "any dataset"
        print(f"[data verify] no split index found for {target} under {splits_root}.")
        return EXIT_OK

    mismatches = 0
    for idx in idx_files:
        recomputed = split_checksum(str(idx))
        doc = json.loads(idx.read_text(encoding="utf-8"))
        stored = doc.get("checksum")
        ok = recomputed == stored
        mismatches += 0 if ok else 1
        marker = "OK" if ok else "MISMATCH"
        print(f"[data verify] {idx.relative_to(out_dir)}: {marker} ({recomputed})")
    if mismatches:
        print(f"error: {mismatches} split index(es) failed checksum verification", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `eval` handler
# --------------------------------------------------------------------------------------------------
def _emit_target_path(args: argparse.Namespace) -> Path:
    """Resolve the result.json output path (``--emit`` or the default leaderboard layout)."""
    if args.emit:
        return Path(str(args.emit))
    return Path("leaderboard") / "results" / str(args.task) / f"{args.model}.json"


def _regime_spec(regime: str, k_shot: int | None) -> RegimeSpec:
    """Build a :class:`rfbench.core.model.RegimeSpec` for a declared regime (lazy import).

    Constructs the spec through the frozen contract so the ``k_shot`` <-> ``few_shot`` coupling is
    enforced by the same code path :func:`rfbench.core.evaluate.evaluate` uses. A regime that
    violates the coupling raises ``ValueError`` here.
    """
    from rfbench.core.model import Regime, RegimeSpec

    return RegimeSpec(name=Regime(regime), k_shot=k_shot)


def _build_result_skeleton(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble a minimal, schema-valid ``result.json`` for the eval command.

    The regime block is derived from a real :class:`RegimeSpec` routed through
    :func:`rfbench.regimes.make_adapter`, so the declared-not-inferred regime and the
    ``k_shot`` <-> ``few_shot`` coupling are produced by the harness' own regime plumbing rather
    than a hand-rolled dict. Metric values remain placeholders until the task registry (WP-20..23)
    and the trained baselines (M3) land.
    """
    from rfbench.regimes import make_adapter

    defaults = _TASK_DEFAULTS[args.task]
    dataset_name = args.dataset if args.dataset else defaults["dataset"]
    track = args.track if args.track else defaults["track"]
    primary = defaults["primary"]

    spec = _regime_spec(args.regime, args.k_shot)
    adapter = make_adapter(spec)  # validates the regime is adaptable; keeps the plumbing honest
    regime: dict[str, Any] = {"name": adapter.regime.name.value}
    if adapter.regime.k_shot is not None:
        regime["k_shot"] = adapter.regime.k_shot

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

    try:
        result = _build_result_skeleton(args)
    except ValueError as exc:  # RegimeSpec coupling / positivity (belt-and-braces)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

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
        f"[eval] wrote schema-valid result.json -> {out_path} "
        "(regime declared via rfbench.regimes; metrics are placeholders until WP-40/M3 land)."
    )
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `train` handler (real from-scratch / full-finetune training, WP-30)
# --------------------------------------------------------------------------------------------------
_MODEL_MODULES: dict[str, str] = {
    # Baseline + foundation models register on explicit import of their (torch) module.
    "mcldnn": "rfbench.models.baselines.mcldnn",
    "resnet_amc": "rfbench.models.baselines.resnet_amc",
    "cldnn": "rfbench.models.baselines.cldnn",
    "interf_cnn": "rfbench.models.baselines.interf_cnn",
    "tprime": "rfbench.models.baselines.tprime",
    # SEI baselines (trained via `rfbench sei-train`; also eval-reachable here).
    "wisig_cnn": "rfbench.models.baselines.sei_cnn",
    "wisig_cnn_paper": "rfbench.models.baselines.wisig_cnn_paper",
    "oracle_cnn": "rfbench.models.baselines.oracle_cnn",
    "complex_cnn": "rfbench.models.baselines.complex_cnn",
    "resnet1d_sei": "rfbench.models.baselines.resnet1d_sei",
    # SNR-estimation regression baseline (trained via `rfbench snr-train`).
    "snr_cnn": "rfbench.models.baselines.snr_cnn",
    "lwm-spectro": "rfbench.models.foundation.lwm_spectro",
    "dummy-fm": "rfbench.models.foundation.dummy",
}


def _import_model_module(model: str) -> None:
    """Import the module that registers ``model`` in ``MODELS`` (opt-in, pulls torch lazily).

    Baseline model modules import torch at their top and are therefore NOT imported by
    ``import rfbench``; importing them here is the explicit opt-in that both loads torch and
    creates the ``@register_model`` entry. Unknown names are left to the registry, which raises
    a clear ``KeyError`` listing the registered models.
    """
    module = _MODEL_MODULES.get(model)
    if module is not None:
        import importlib

        importlib.import_module(module)


_TASK_MODULES: dict[str, str] = {
    # Tasks register (via @register_task) on import of their module. These are dependency-free
    # (torch stays lazy in their Dataset loaders), so importing them here just populates TASKS.
    "amc": "rfbench.tasks.amc",
    "sei": "rfbench.tasks.sei",
    "wideband_detection": "rfbench.tasks.wideband_detection",
    "interference_id": "rfbench.tasks.interference_id",
    "protocol_tech_id": "rfbench.tasks.protocol_tech_id",
    "snr_estimation": "rfbench.tasks.snr_estimation",
}


def _import_task_module(task: str) -> None:
    """Import the module that registers ``task`` in ``TASKS`` (so ``get_task`` can resolve it)."""
    module = _TASK_MODULES.get(task)
    if module is not None:
        import importlib

        importlib.import_module(module)


def _load_split_checksum(dataset: str, out_dir: Path) -> str | None:
    """Return the versioned split-index checksum for ``dataset`` (or ``None`` if absent).

    Reads ``<out_dir>/splits/<dataset>/<canonical_split_id>.idx.json`` and returns its
    ``checksum`` field so the emitted ``result.json`` references the concrete split (not the
    all-zero placeholder). Missing indices yield ``None`` -- the training run still emits a
    valid (self_reported) row and the checksum is filled once the split is prepared.
    """
    split_dir = out_dir / "splits" / dataset
    if not split_dir.is_dir():
        return None
    for idx in sorted(split_dir.glob("*.idx.json")):
        try:
            doc = json.loads(idx.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checksum = doc.get("checksum")
        if isinstance(checksum, str) and checksum:
            return checksum
    return None


def _cmd_train(args: argparse.Namespace) -> int:
    """Train a baseline on a task's train split, then emit a schema-valid result.json.

    Loads the registered task + model (importing the model's torch module lazily), runs the
    real training loop in :func:`rfbench.training.train_baseline` for the declared regime
    (``from_scratch`` / ``full_finetune``), and writes the result under ``--out``. When
    ``--out-checkpoint`` is set, the best-val model ``state_dict`` is also persisted to disk.
    torch and the training module are imported inside this handler so ``import rfbench`` /
    ``rfbench --help`` stay dependency-free.
    """
    if args.regime not in ("from_scratch", "full_finetune"):
        print(
            f"error: [train] --regime must be from_scratch or full_finetune, got {args.regime!r} "
            "(probing regimes are evaluated via `rfbench eval`).",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        from rfbench.core.registry import MODELS, get_task
        from rfbench.training import resolve_amc_dataset, train_baseline
    except ModuleNotFoundError as exc:
        print(
            f"error: [train] training needs the torch extra (`pip install rfbench[torch]`): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    try:
        _import_model_module(args.model)
    except ModuleNotFoundError as exc:
        print(
            f"error: [train] could not import the '{args.model}' model module "
            f"(needs the torch extra): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    try:
        spec = _regime_spec(args.regime, None)
        _import_task_module(args.task)  # populate TASKS so get_task can resolve it
        task = get_task(args.task)
        dataset = resolve_amc_dataset(task, args.dataset)
        model = MODELS.get(args.model)()
    except (KeyError, ValueError) as exc:
        print(f"error: [train] {exc}", file=sys.stderr)
        return EXIT_USAGE

    # Point the dataset's cache at --cache so the on-disk loader finds the prepared arrays.
    os.environ.setdefault("RFBENCH_CACHE", args.cache)
    checksum = _load_split_checksum(args.dataset, _repo_root() / "leaderboard")
    if checksum is not None and hasattr(dataset, "checksum"):
        dataset.checksum = checksum

    out_path = (
        Path(args.out)
        if args.out
        else Path("leaderboard") / "results" / str(args.task) / f"{args.model}.json"
    )
    print(
        f"[train] training '{args.model}' on {args.task}/{args.dataset} "
        f"(regime={args.regime}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, device={args.device})..."
    )
    try:
        _model, result = train_baseline(
            task,
            model,
            dataset,
            regime=spec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=None if args.device == "auto" else args.device,
            out_path=out_path,
            checkpoint_out=Path(args.out_checkpoint) if args.out_checkpoint else None,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        print(f"error: [train] {exc}", file=sys.stderr)
        return EXIT_FAILURE

    primary = result["metrics"]["primary"]
    score = result["metrics"]["values"].get(primary)
    print(f"[train] wrote result.json -> {out_path} ({primary}={score}).")
    if args.out_checkpoint:
        print(f"[train] wrote checkpoint -> {args.out_checkpoint}.")
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `sei-train` handler (track-aware SEI from-scratch training, dedicated recipe)
# --------------------------------------------------------------------------------------------------
#: SEI tracks reachable from `sei-train`. The three closed-set identification conditions plus
#: `open_set`: the model is still fit as a `|known|`-class identifier (CE on the gallery train
#: split), but the FINAL score is the open-set AUROC/EER (max-softmax genuine-vs-impostor) that
#: `evaluate` computes for the open_set track -- no separate training recipe needed.
_SEI_TRAIN_TRACKS: tuple[str, ...] = ("closed_set", "cross_receiver", "cross_day", "open_set")


def _cmd_sei_train(args: argparse.Namespace) -> int:
    """Train an SEI baseline on one (dataset, track), then emit a track-tagged result.json.

    Uses the SEI-specific loop :func:`rfbench.training_sei.train_sei_baseline` (class-weighted CE,
    explicit L2 via the model's ``l2_penalty``, best checkpoint + early stop on val LOSS) so the
    shared AMC loop stays untouched, and threads ``--track`` into ``evaluate`` so the closed-set
    conditions are scored as SEPARATE rows. torch + the SEI training module are imported inside
    this handler so ``import rfbench`` / ``rfbench --help`` stay dependency-free.
    """
    import logging

    # Surface the per-epoch training trajectory (train/val loss, best-checkpoint, early stop) on
    # stdout so a cluster .out log shows convergence -- otherwise the loop's logger.info lines are
    # swallowed at the default WARNING level.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from rfbench.core.registry import MODELS, TASKS
        from rfbench.training_sei import count_classes, train_sei_baseline
    except ModuleNotFoundError as exc:
        print(
            f"error: [sei-train] needs the torch extra (`pip install rfbench[torch]`): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    try:
        _import_model_module(args.model)
    except ModuleNotFoundError as exc:
        print(
            f"error: [sei-train] could not import the '{args.model}' model module "
            f"(needs the torch extra): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    os.environ.setdefault("RFBENCH_CACHE", args.cache)
    try:
        _import_task_module("sei")
        task = TASKS.get("sei")(args.track, dataset=args.dataset)
        datasets = task.datasets()
        if not datasets:
            raise ValueError("SEI task declares no datasets")
        dataset = datasets[0]
        spec = _regime_spec(args.regime, None)
        num_classes = count_classes(dataset)
        model = MODELS.get(args.model)(
            num_classes=num_classes,
            window=args.window,
            device=None if args.device == "auto" else args.device,
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: [sei-train] {exc}", file=sys.stderr)
        return EXIT_USAGE

    out_path = (
        Path(args.out)
        if args.out
        else Path("leaderboard") / "results" / "sei" / f"{args.model}-{args.track}.json"
    )
    print(
        f"[sei-train] training '{args.model}' on sei/{args.dataset} track={args.track} "
        f"(n_classes={num_classes}, regime={args.regime}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}, l2={args.l2_lambda}, "
        f"class_weight={not args.no_class_weight}, device={args.device})..."
    )
    try:
        _model, result = train_sei_baseline(
            task,
            model,
            dataset,
            track=args.track,
            regime=spec,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l2_lambda=args.l2_lambda,
            weight_decay=args.weight_decay,
            use_class_weight=not args.no_class_weight,
            seed=args.seed,
            device=None if args.device == "auto" else args.device,
            out_path=out_path,
            patience=args.patience,
            compute_bootstrap_ci=not args.no_bootstrap,
        )
    except (ValueError, RuntimeError, TypeError, FileNotFoundError) as exc:
        print(f"error: [sei-train] {exc}", file=sys.stderr)
        return EXIT_FAILURE

    primary = result["metrics"]["primary"]
    values = result["metrics"]["values"]
    extras = ", ".join(f"{k}={v:.4f}" for k, v in values.items() if k != primary)
    print(
        f"[sei-train] wrote result.json -> {out_path} "
        f"({primary}={values.get(primary):.4f}{'; ' + extras if extras else ''})."
    )
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `snr-train` handler (from-scratch SNR REGRESSION training, dedicated recipe)
# --------------------------------------------------------------------------------------------------
def _cmd_snr_train(args: argparse.Namespace) -> int:
    """Train an SNR regressor on radioml_2016_10a, then emit a result.json (primary rmse_db).

    Uses the SNR-specific REGRESSION loop :func:`rfbench.training_snr.train_snr_regressor` (MSE on
    the ``snr_db`` target, Adam, best checkpoint + early stop on val LOSS) so the AMC/SEI
    classification loops stay untouched, and runs the single ``all_snr`` track. torch + the SNR
    training module are imported inside this handler so ``import rfbench`` / ``rfbench --help``
    stay dependency-free.
    """
    import logging

    # Surface the per-epoch training trajectory (train/val loss, best-checkpoint, early stop) on
    # stdout so a cluster .out log shows convergence -- otherwise the loop's logger.info lines are
    # swallowed at the default WARNING level.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from rfbench.core.registry import MODELS, TASKS
        from rfbench.training_snr import train_snr_regressor
    except ModuleNotFoundError as exc:
        print(
            f"error: [snr-train] needs the torch extra (`pip install rfbench[torch]`): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    try:
        _import_model_module(args.model)
    except ModuleNotFoundError as exc:
        print(
            f"error: [snr-train] could not import the '{args.model}' model module "
            f"(needs the torch extra): {exc}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    os.environ.setdefault("RFBENCH_CACHE", args.cache)
    try:
        _import_task_module("snr_estimation")
        task = TASKS.get("snr_estimation")()
        datasets = task.datasets()
        if not datasets:
            raise ValueError("snr_estimation task declares no datasets")
        dataset = next((d for d in datasets if d.name == args.dataset), None)
        if dataset is None:
            available = ", ".join(sorted(d.name for d in datasets))
            raise ValueError(f"unknown --dataset {args.dataset!r}; available: {available}")
        spec = _regime_spec(args.regime, None)
        model = MODELS.get(args.model)(device=None if args.device == "auto" else args.device)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: [snr-train] {exc}", file=sys.stderr)
        return EXIT_USAGE

    checksum = _load_split_checksum(args.dataset, _repo_root() / "leaderboard")
    if checksum is not None and hasattr(dataset, "checksum"):
        dataset.checksum = checksum

    out_path = (
        Path(args.out)
        if args.out
        else Path("leaderboard") / "results" / "snr_estimation" / f"{args.model}.json"
    )
    print(
        f"[snr-train] training '{args.model}' on snr_estimation/{args.dataset} "
        f"(regime={args.regime}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, device={args.device})..."
    )
    try:
        _model, result = train_snr_regressor(
            task,
            model,
            dataset,
            regime=spec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=None if args.device == "auto" else args.device,
            out_path=out_path,
            patience=args.patience,
            compute_bootstrap_ci=not args.no_bootstrap,
        )
    except (ValueError, RuntimeError, TypeError, FileNotFoundError) as exc:
        print(f"error: [snr-train] {exc}", file=sys.stderr)
        return EXIT_FAILURE

    primary = result["metrics"]["primary"]
    values = result["metrics"]["values"]
    extras = ", ".join(f"{k}={v:.4f}" for k, v in values.items() if k != primary)
    print(
        f"[snr-train] wrote result.json -> {out_path} "
        f"({primary}={values.get(primary):.4f}{'; ' + extras if extras else ''})."
    )
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `submit` handler
# --------------------------------------------------------------------------------------------------
def _resolve_manifest_path(
    document: dict[str, Any], manifest_path: str | None, result_path: Path
) -> str | None:
    """Resolve which manifest (if any) backs this result.

    Precedence: an explicit ``--manifest`` wins; otherwise the result's ``submission_ref`` is
    honoured, resolved relative to the repo root (matching its repo-relative form). Returns ``None``
    when no manifest is referenced at all (a bare Tier-1 self_reported dump).
    """
    if manifest_path is not None:
        return manifest_path
    ref = document.get("submission_ref")
    if isinstance(ref, str) and ref:
        candidate = _repo_root() / ref
        if candidate.is_file():
            return str(candidate)
        # Fall back to a path relative to the result itself (fork checkouts).
        sibling = result_path.parent / Path(ref).name
        return str(sibling) if sibling.is_file() else str(candidate)
    return None


def _row_identity(document: dict[str, Any]) -> tuple[str, str, str, str, str, str, int | None]:
    """The comparability key of a leaderboard row (CONTRIBUTING.md 'One result per ...').

    A row is unique per ``(task, model, regime[, k_shot], dataset, split, track)``. Two committed
    results that agree on this tuple are the SAME cell of the board -- an accidental duplicate. The
    ``track`` defaults to the empty string so single-track tasks (AMC/sensing, which omit it)
    compare cleanly against each other.
    """
    task = document.get("task") if isinstance(document.get("task"), dict) else {}
    model = document.get("model") if isinstance(document.get("model"), dict) else {}
    regime = document.get("regime") if isinstance(document.get("regime"), dict) else {}
    dataset = document.get("dataset") if isinstance(document.get("dataset"), dict) else {}
    split = document.get("split") if isinstance(document.get("split"), dict) else {}
    k_shot = regime.get("k_shot")
    return (
        str(task.get("name", "")),
        str(model.get("name", "")),
        str(regime.get("name", "")),
        str(dataset.get("name", "")),
        str(split.get("name", "")),
        str(split.get("track", "")),
        int(k_shot) if isinstance(k_shot, int) else None,
    )


def _amc_full_snr_errors(document: dict[str, Any]) -> list[str]:
    """AMC full-protocol gate: the row MUST record the full SNR range (no cherry-picking).

    Enforces the hard rule ``eval.conditions.full_snr_range == true`` for AMC results
    (CONTRIBUTING.md / docs/EVALUATION_PROTOCOL.md). A high-SNR-only cherry-pick -- a missing or
    falsy flag -- is rejected so the board never blends a partial-SNR score into the AMC column.
    Non-AMC tasks are untouched.
    """
    task = document.get("task") if isinstance(document.get("task"), dict) else {}
    if task.get("name") != "amc":
        return []
    ev = document.get("eval") if isinstance(document.get("eval"), dict) else {}
    conditions = ev.get("conditions") if isinstance(ev.get("conditions"), dict) else {}
    if conditions.get("full_snr_range") is not True:
        return [
            "AMC requires the full SNR range: set eval.conditions.full_snr_range=true "
            "(no high-SNR cherry-picking; see docs/EVALUATION_PROTOCOL.md)"
        ]
    return []


def _duplicate_row_errors(document: dict[str, Any], result_path: Path) -> list[str]:
    """Anti-doublon gate: reject a row that duplicates a COMMITTED sibling's comparability key.

    Scans the sibling ``leaderboard/results/<task>/*.json`` next to the submitted result (skipping
    the file under check itself, matched by resolved path). If any sibling shares the full row
    identity -- same ``(task, model, regime[, k_shot], dataset, split, track)`` -- it is the same
    board cell and the submission is an accidental duplicate. Unreadable/invalid siblings are
    skipped (they fail their own validation elsewhere), so this never false-positives on garbage.
    """
    sibling_dir = result_path.parent
    if not sibling_dir.is_dir():
        return []
    identity = _row_identity(document)
    try:
        this_path = result_path.resolve()
    except OSError:  # pragma: no cover - defensive
        this_path = result_path
    clashes: list[str] = []
    for sibling in sorted(sibling_dir.glob("*.json")):
        try:
            if sibling.resolve() == this_path:
                continue
        except OSError:  # pragma: no cover - defensive
            if sibling == result_path:
                continue
        try:
            other = json.loads(sibling.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(other, dict) and _row_identity(other) == identity:
            clashes.append(sibling.name)
    if clashes:
        return [
            "duplicate leaderboard row: this (task, model, regime, dataset, split, track) already "
            f"exists in {', '.join(clashes)} -- one result per cell (CONTRIBUTING.md)"
        ]
    return []


def _split_checksum_match_errors(document: dict[str, Any]) -> list[str]:
    """Split-checksum lint: the row's ``split.checksum`` must match the COMMITTED split index.

    When ``leaderboard/splits/<dataset>/<canonical_split_id>.idx.json`` exists, its stored
    ``checksum`` is the source of truth. A result whose ``split.checksum`` disagrees was scored on
    a different (or stale) partition than the versioned indices claim -- exactly the drift the tie
    between ``canonical_split_id`` + ``checksum`` and ``leaderboard/splits/`` guards against. When
    no committed index exists yet the check is silent (the split lands in a later PR).
    """
    split = document.get("split") if isinstance(document.get("split"), dict) else {}
    dataset = document.get("dataset") if isinstance(document.get("dataset"), dict) else {}
    dataset_name = dataset.get("name")
    split_id = split.get("canonical_split_id")
    result_checksum = split.get("checksum")
    if not (isinstance(dataset_name, str) and isinstance(split_id, str)):
        return []
    idx = _repo_root() / "leaderboard" / "splits" / dataset_name / f"{split_id}.idx.json"
    if not idx.is_file():
        return []
    try:
        committed_checksum = json.loads(idx.read_text(encoding="utf-8")).get("checksum")
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read committed split index {idx.name}: {exc}"]
    if isinstance(committed_checksum, str) and result_checksum != committed_checksum:
        return [
            f"split.checksum {result_checksum} does not match the committed split index "
            f"{split_id}.idx.json ({committed_checksum}); the row was scored on a different split"
        ]
    return []


def _manifest_consistency_errors(
    document: dict[str, Any], manifest_doc: dict[str, Any]
) -> list[str]:
    """Cross-check a Tier-2 manifest against the result it claims to reproduce.

    The submission schema guarantees each field's SHAPE; this guards that the manifest and result
    describe the SAME evaluation: matching task (name+version) and regime, and an
    ``expected_metrics`` block that names the result's primary metric (so the maintainer's re-run
    has a target for the ranking metric). Mismatches here are exactly what would let a verified row
    drift from the score it displays.
    """
    errors: list[str] = []

    result_task = document.get("task") if isinstance(document.get("task"), dict) else {}
    manifest_task = manifest_doc.get("task") if isinstance(manifest_doc.get("task"), dict) else {}
    if result_task and manifest_task and result_task != manifest_task:
        errors.append(f"manifest.task {manifest_task} does not match result.task {result_task}")

    result_regime = document.get("regime") if isinstance(document.get("regime"), dict) else {}
    manifest_regime = (
        manifest_doc.get("regime") if isinstance(manifest_doc.get("regime"), dict) else {}
    )
    if result_regime and manifest_regime and result_regime != manifest_regime:
        errors.append(
            f"manifest.regime {manifest_regime} does not match result.regime {result_regime}"
        )

    metrics = document.get("metrics") if isinstance(document.get("metrics"), dict) else {}
    primary = metrics.get("primary") if isinstance(metrics, dict) else None
    expected = manifest_doc.get("expected_metrics")
    if isinstance(primary, str) and isinstance(expected, dict) and primary not in expected:
        errors.append(
            f"manifest.expected_metrics is missing the result's primary metric {primary!r}"
        )
    return errors


def _manifest_completeness_errors(
    document: dict[str, Any], manifest_path: str | None, result_path: Path
) -> list[str]:
    """Manifest-completeness gate mirroring the CI pre-flight (WP-51).

    Two layers:

    * **Data-provenance identity** every leaderboard row must carry -- a concrete
      ``canonical_split_id`` and a non-placeholder ``split.checksum``.
    * **Reproducibility manifest** (Tier-2) -- when a manifest is supplied via ``--manifest`` OR
      referenced by the result's ``submission_ref``, it is validated for COMPLETENESS against
      ``submission.schema.json`` (all required repro fields: code_commit, command,
      weights_url/docker_image, hardware, expected_metrics, tolerance) and cross-checked for
      consistency with the result it reproduces (same task/regime, primary metric expected).
    """
    errors: list[str] = []
    split = document.get("split", {})
    if not isinstance(split, dict) or not split.get("canonical_split_id"):
        errors.append("split.canonical_split_id is required for a leaderboard submission")
    checksum = split.get("checksum") if isinstance(split, dict) else None
    if checksum == _PLACEHOLDER_CHECKSUM:
        errors.append(
            "split.checksum is the all-zero placeholder; prepare the real split "
            "(rfbench data prepare) so the row references a concrete split index"
        )

    resolved = _resolve_manifest_path(document, manifest_path, result_path)
    if resolved is not None:
        try:
            manifest_doc = json.loads(Path(resolved).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"could not read manifest {resolved}: {exc}")
        else:
            schema_errors = _validate_against_schema(manifest_doc, "submission.schema.json")
            errors.extend(f"manifest: {msg}" for msg in schema_errors)
            if not schema_errors and isinstance(manifest_doc, dict):
                errors.extend(
                    f"manifest: {msg}"
                    for msg in _manifest_consistency_errors(document, manifest_doc)
                )
    return errors


def _cmd_submit(args: argparse.Namespace) -> int:
    result_path = Path(args.result)
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: [submit --check] could not read {result_path}: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    errors = _validate_result(document)
    errors.extend(_manifest_completeness_errors(document, args.manifest, result_path))
    # Full-protocol / comparability gates promised by CONTRIBUTING.md (beyond schema shape):
    # AMC full-SNR range, one-row-per-cell (anti-doublon), split-checksum tie to committed indices.
    errors.extend(_amc_full_snr_errors(document))
    errors.extend(_duplicate_row_errors(document, result_path))
    errors.extend(_split_checksum_match_errors(document))
    if errors:
        print(f"error: [submit --check] {result_path} is NOT PR-ready:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return EXIT_FAILURE

    print(f"[submit --check] {result_path}: PR-ready (schema-valid + manifest complete).")
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# `leaderboard` handlers
# --------------------------------------------------------------------------------------------------
def _load_site_generator() -> ModuleType:
    """Import the non-packaged ``leaderboard/site/generate.py`` module by path (lazy).

    ``leaderboard/site`` is not part of the installed ``rfbench`` wheel (no ``__init__.py``), so it
    is loaded from the repo source tree via :mod:`importlib.util`. Kept lazy so ``import rfbench``
    stays dependency-free. Raises ``RuntimeError`` if the generator cannot be located.
    """
    import importlib.util

    generate_py = _repo_root() / "leaderboard" / "site" / "generate.py"
    if not generate_py.is_file():
        raise RuntimeError(
            f"leaderboard site generator not found at {generate_py}; "
            "run from a source checkout with leaderboard/site/generate.py present."
        )
    spec = importlib.util.spec_from_file_location("rfbench._leaderboard_site_generate", generate_py)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load a module spec for {generate_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd_leaderboard_build(args: argparse.Namespace) -> int:
    try:
        generate = _load_site_generator()
        index_path = generate.build_site(args.results, args.out)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: [leaderboard build] {exc}", file=sys.stderr)
        return EXIT_FAILURE
    index = Path(index_path)
    print(f"[leaderboard build] wrote static site to {index.parent} (index: {index}).")
    return EXIT_OK


def _cmd_leaderboard_validate(args: argparse.Namespace) -> int:
    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(
            f"error: [leaderboard validate] results dir not found: {results_dir}",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    files = sorted(results_dir.rglob("*.json"))
    invalid = 0
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[leaderboard validate] {path}: UNREADABLE ({exc})", file=sys.stderr)
            invalid += 1
            continue
        errors = _validate_result(document)
        if errors:
            invalid += 1
            print(f"[leaderboard validate] {path}: INVALID", file=sys.stderr)
            for err in errors:
                print(f"    - {err}", file=sys.stderr)
        elif args.verbose:
            print(f"[leaderboard validate] {path}: ok")
    print(f"[leaderboard validate] {len(files) - invalid}/{len(files)} rows valid.")
    return EXIT_FAILURE if invalid else EXIT_OK


# --------------------------------------------------------------------------------------------------
# `verify` handler (maintainer verification pipeline, WP-53)
# --------------------------------------------------------------------------------------------------
def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-run a submission and flip ``verification.status`` self_reported -> verified in tolerance.

    Delegates the numerics to :func:`rfbench.verify.verify_result`. The recomputed metrics come
    from a re-run ``result.json`` (``--rerun``, itself produced by ``rfbench eval`` on the
    maintainer's station); when omitted the submitted result's own metrics are used, which is only
    meaningful for a smoke re-check (it will always verify) and is called out in the output. On a
    successful flip the verified result is written back (``--out`` or in place); on a mismatch /
    incomplete manifest nothing is written and the diff is reported (exit 1).
    """
    from rfbench.verify import (
        VerificationError,
        load_json,
        rerun_metrics_from_result,
        verify_result,
    )

    if not args.by:
        print(
            "error: [verify] --by (maintainer handle) is required to sign a flip",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if not args.hardware:
        print("error: [verify] --hardware is required to stamp a verified row", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = load_json(args.result)
        manifest = load_json(args.manifest)
        rerun_doc = load_json(args.rerun) if args.rerun else result
        rerun_metrics = rerun_metrics_from_result(rerun_doc)
    except VerificationError as exc:
        print(f"error: [verify] {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if not args.rerun:
        print(
            "[verify] no --rerun supplied: comparing the submitted result against itself "
            "(smoke check only; supply --rerun <recomputed result.json> for a real re-run)."
        )

    try:
        report = verify_result(
            result,
            manifest,
            rerun_metrics,
            verified_by=args.by,
            verified_hardware=args.hardware,
            method=args.mode,
        )
    except VerificationError as exc:
        print(f"error: [verify] {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if not report.verified:
        print(f"error: [verify] {args.result}: {report.summary()}", file=sys.stderr)
        return EXIT_FAILURE

    out_path = Path(args.out) if getattr(args, "out", None) else Path(args.result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.result, indent=2) + "\n", encoding="utf-8")
    print(f"[verify] {args.result}: {report.summary()} -> wrote verified result to {out_path}.")
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
    download.add_argument(
        "--source-url",
        dest="source_url",
        default=None,
        metavar="URL",
        help="EULA-gated archive URL for datasets whose link is not embedded (RadioML/WiSig).",
    )
    download.add_argument(
        "--manual-archive",
        dest="manual_archive",
        default=None,
        metavar="PATH",
        help=(
            "Path to an already-fetched archive to extract instead of downloading (tprime_wifi4 "
            "only) -- for hosts whose TLS chain a strict client correctly refuses; fetch the "
            "archive out-of-band and point here rather than weakening verification in code."
        ),
    )
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
        "--out",
        metavar="DIR",
        help="Output tree for splits/<dataset>/ (default: <repo>/leaderboard).",
    )
    prepare.add_argument(
        "--labels-file",
        dest="labels_file",
        metavar="PATH",
        help=(
            "JSON of pre-extracted labels/records/samples to drive the split without the heavy "
            "cluster loaders (else the lazy loaders read the cached dataset; needs rfbench[data])."
        ),
    )
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
    list_p.add_argument(
        "--out",
        metavar="DIR",
        help="Splits tree to scan for prepared status (default: <repo>/leaderboard).",
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
    verify_p.add_argument(
        "--out",
        metavar="DIR",
        help="Splits tree to verify (default: <repo>/leaderboard).",
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
            "regime is written verbatim (via a rfbench.regimes adapter), never inferred; "
            "verification.status = 'self_reported'."
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


def _build_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    tr = subparsers.add_parser(
        "train",
        parents=[parent],
        help="Train a baseline on a task's train split, then emit a result.json (M3, GPU).",
        description=(
            "Runs the real torch training loop (rfbench.training.train_baseline) for the "
            "from_scratch / full_finetune regimes, then evaluates on the test split and writes a "
            "schema-valid result.json. Needs the torch extra; runs on the cluster GPU."
        ),
    )
    tr.add_argument("--task", required=True, choices=TASK_NAMES, help="Task to train on.")
    tr.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_NAMES,
        help="Dataset variant to train + score on.",
    )
    tr.add_argument("--model", required=True, metavar="NAME", help="Registered model name.")
    tr.add_argument(
        "--regime",
        default="from_scratch",
        choices=("from_scratch", "full_finetune"),
        help="Trainable regime, ALWAYS declared (default: from_scratch).",
    )
    tr.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50).")
    tr.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    tr.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).")
    tr.add_argument("--seed", type=int, default=42, help="Global seed (default: 42).")
    tr.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Compute device (default: auto -> cuda when available).",
    )
    tr.add_argument(
        "--out",
        metavar="PATH",
        help="Output path (default: leaderboard/results/<task>/<model>.json).",
    )
    tr.add_argument(
        "--out-checkpoint",
        dest="out_checkpoint",
        metavar="PATH",
        help="Persist the best-val model checkpoint (torch.save) to this path (default: unset, "
        "no checkpoint written to disk).",
    )
    tr.set_defaults(func=_cmd_train)


def _build_sei_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    st = subparsers.add_parser(
        "sei-train",
        parents=[parent],
        help="Train an SEI baseline on one (dataset, track) with the SEI recipe (M3, GPU).",
        description=(
            "Runs the SEI-specific training loop (rfbench.training_sei.train_sei_baseline: "
            "class-weighted CE, explicit L2 via the model's l2_penalty, best checkpoint + early "
            "stop on validation LOSS) and threads --track into evaluate so closed_set / "
            "cross_receiver / cross_day are scored as SEPARATE result.json rows. The shared AMC "
            "loop (rfbench.training) is left untouched. Needs the torch extra; runs on the cluster."
        ),
    )
    st.add_argument(
        "--dataset",
        default="wisig",
        choices=("wisig", "oracle", "powder"),
        help="SEI dataset variant (default: wisig).",
    )
    st.add_argument("--model", required=True, metavar="NAME", help="Registered SEI model name.")
    st.add_argument(
        "--track",
        default="closed_set",
        choices=_SEI_TRAIN_TRACKS,
        help="SEI condition, scored as its own row: closed_set / cross_receiver / cross_day "
        "(rank-1) or open_set (AUROC/EER). Default: closed_set.",
    )
    st.add_argument(
        "--regime",
        default="from_scratch",
        choices=("from_scratch", "full_finetune"),
        help="Trainable regime, ALWAYS declared (default: from_scratch).",
    )
    st.add_argument("--window", type=int, default=256, help="IQ window length (default: 256).")
    st.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100).")
    st.add_argument(
        "--batch-size", dest="batch_size", type=int, default=32, help="Batch size (default: 32)."
    )
    st.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate (default: 5e-4).")
    st.add_argument(
        "--l2-lambda",
        dest="l2_lambda",
        type=float,
        default=1e-4,
        help="L2 strength on the model's regularised kernels (default: 1e-4).",
    )
    st.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.0,
        help="Fallback Adam weight_decay for models with no l2_penalty hook (default: 0.0).",
    )
    st.add_argument(
        "--no-class-weight",
        dest="no_class_weight",
        action="store_true",
        help="Disable the WiSig max(count)/count class weighting.",
    )
    st.add_argument(
        "--no-bootstrap",
        dest="no_bootstrap",
        action="store_true",
        help="Skip the per-run bootstrap CI (much faster eval). Use for multi-seed sweeps where "
        "uncertainty comes from the across-seed std, not per-seed bootstrap.",
    )
    st.add_argument("--patience", type=int, default=5, help="Early-stop patience (default: 5).")
    st.add_argument("--seed", type=int, default=42, help="Global seed (default: 42).")
    st.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Compute device (default: auto -> cuda when available).",
    )
    st.add_argument(
        "--out",
        metavar="PATH",
        help="Output path (default: leaderboard/results/sei/<model>-<track>.json).",
    )
    st.set_defaults(func=_cmd_sei_train)


def _build_snr_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent: argparse.ArgumentParser,
) -> None:
    sn = subparsers.add_parser(
        "snr-train",
        parents=[parent],
        help="Train an SNR regressor (MSE) and emit a result.json with primary rmse_db (M3, GPU).",
        description=(
            "Runs the SNR-specific REGRESSION loop (rfbench.training_snr.train_snr_regressor: MSE "
            "on the snr_db target, Adam, best checkpoint + early stop on validation LOSS) over the "
            "single all_snr track, then evaluates on the test split and writes a schema-valid "
            "result.json (primary rmse_db, lower is better). The AMC/SEI classification loops are "
            "left untouched. Needs the torch extra; runs on the cluster GPU."
        ),
    )
    sn.add_argument(
        "--dataset",
        default="radioml_2016_10a",
        choices=("radioml_2016_10a",),
        help="SNR-estimation dataset variant (default: radioml_2016_10a).",
    )
    sn.add_argument("--model", required=True, metavar="NAME", help="Registered SNR model name.")
    sn.add_argument(
        "--regime",
        default="from_scratch",
        choices=("from_scratch", "full_finetune"),
        help="Trainable regime, ALWAYS declared (default: from_scratch).",
    )
    sn.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100).")
    sn.add_argument(
        "--batch-size", dest="batch_size", type=int, default=256, help="Batch size (default: 256)."
    )
    sn.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).")
    sn.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
        help="Adam weight_decay / L2 (default: 1e-4).",
    )
    sn.add_argument("--patience", type=int, default=10, help="Early-stop patience (default: 10).")
    sn.add_argument("--seed", type=int, default=42, help="Global seed (default: 42).")
    sn.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Compute device (default: auto -> cuda when available).",
    )
    sn.add_argument(
        "--out",
        metavar="PATH",
        help="Output path (default: leaderboard/results/snr_estimation/<model>.json).",
    )
    sn.add_argument(
        "--no-bootstrap",
        dest="no_bootstrap",
        action="store_true",
        help="Skip the per-run bootstrap CI (much faster eval). Use for multi-seed sweeps where "
        "uncertainty comes from the across-seed std, not per-seed bootstrap.",
    )
    sn.set_defaults(func=_cmd_snr_train)


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
        "--rerun",
        metavar="RESULT.JSON",
        help=(
            "Re-run result.json whose metrics.values are the recomputed metrics to compare against "
            "the manifest's expected_metrics (default: the submitted result itself, a smoke check)."
        ),
    )
    verify.add_argument(
        "--out",
        metavar="PATH",
        help="Where to write the verified result.json (default: overwrite the input in place).",
    )
    verify.add_argument(
        "--mode",
        choices=("eval_only", "full_retrain"),
        default="eval_only",
        help="Re-run mode -> verification.method (default: eval_only, the cost guard-rail).",
    )
    verify.add_argument(
        "--by",
        metavar="HANDLE",
        help="Maintainer handle -> verification.verified_by (required).",
    )
    verify.add_argument(
        "--hardware",
        metavar="STR",
        help="Hardware string -> verification.verified_hardware (required).",
    )
    verify.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Compute device for the re-run (default: cuda).",
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
    _build_train_parser(subparsers, parent)
    _build_sei_train_parser(subparsers, parent)
    _build_snr_train_parser(subparsers, parent)
    _build_submit_parser(subparsers, parent)
    _build_leaderboard_parser(subparsers, parent)
    _build_verify_parser(subparsers, parent)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX exit code (0 ok, 2 usage error, 1 validation failure)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - argparse `required=True` guards this.
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
