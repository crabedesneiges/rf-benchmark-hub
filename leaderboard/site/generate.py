"""WP-50 -- static leaderboard site generator (design + genericity overhaul).

Reads every ``leaderboard/results/**/*.json`` row (each MUST validate against
``schemas/result.schema.json``), groups the valid rows by task, and renders a polished,
**fully static** HTML site into an ``--out`` directory. The generator is stdlib-only (no
Jinja, no Chart.js, no build step): pages are assembled with manual string building +
``html.escape``, and every chart is an **inline SVG** whose polylines are computed here in
Python. The one non-stdlib dependency is ``jsonschema`` (a hard dep of the harness),
imported LAZILY so importing this module stays dependency-free. The site's only external
network / runtime additions are a single Google Fonts ``<link>`` (Space Grotesk / IBM Plex,
every selector falls back to system fonts if it's blocked) and a small homepage-only inline
vanilla-JS search/filter script (no dependencies, degrades to "everything visible" if JS is
disabled) -- every other page stays zero-JS.

The renderer is **data-driven, never task-specific**:

* The set of tasks shown is DECLARED in a committed manifest (``leaderboard/tasks.json``):
  every declared task appears on the board. Tasks that have valid result rows render as
  full leaderboards (tables + plots); declared tasks that have NO results (or no baseline
  yet) render a WIP card on the index and a minimal "Work in progress" task page -- never a
  broken empty table. Any task that has results but is missing from the manifest is still
  rendered (the manifest is additive, not a filter). The manifest supplies each task's
  title, ``status`` (implemented | wip | planned), ``priority`` (P1/P2/P3), a short blurb,
  and an optional ``scope`` (``terrestrial_iq`` | ``csi_sensing``, default ``terrestrial_iq``)
  that groups tasks into the homepage/sidebar's two sections.
* A fixed display order is applied to the known tasks (amc, sei, wideband_detection,
  spectrum_sensing, ...) then any others land alphabetically. No task is hardcoded into the
  rendering path.
* Per task, the SCALAR metrics are discovered from every key of ``metrics.values`` and the
  CURVE metrics from every key of ``metrics.curves``. SEI (rank1_accuracy/auroc/eer),
  detection (mAP/mAR/IoU), sensing (pd@pfa=0.1/latency + ROC) etc. therefore render
  automatically the moment their result JSONs appear -- nothing about their metric names
  is baked in.
* EDUCATIONAL header: every task page (full leaderboard OR minimal WIP page) is topped by an
  explanatory header assembled from the manifest's OPTIONAL educational fields --
  ``description`` (what/why), a compact ``dataset`` card (source, #classes, modality,
  real/synthetic, conditions, license, split) and metric definitions (``primary_metric`` +
  ``secondary_metrics``). Every field is optional and rendered generically, so a task
  missing any piece simply omits it; nothing about a specific task is hardcoded.
* A ``guide.html`` page renders the shared educational content (``_GUIDE`` below): what I/Q
  is, the four evaluation regimes, verified-vs-self_reported, the data + split policies and a
  metrics glossary (each entry an up/down arrow for higher/lower-is-better). It is linked from
  the top nav of every page.

Protocol invariants (docs/EVALUATION_PROTOCOL.md / D5), enforced structurally:

* One HTML page per task, plus an ``index.html`` landing page.
* Rows are partitioned into ``(regime, track)`` groups. Each group renders exactly one
  leaderboard TABLE (a column per scalar metric, primary first + emphasised) and one line
  PLOT per curve metric (overlaying every model in that group). A table or plot therefore
  NEVER compares across two regimes -- nor two tracks.
* ``track`` is read from ``eval.conditions.track`` or ``split.track`` (free-form/optional);
  rows without one land in a default ``all`` bucket so single-track tasks still render.
* A badge distinguishes the four ``verification.status`` tiers -- ``verified``, ``self_reported``,
  and the two never-re-run literature tiers ``from_paper`` / ``from_paper_uncertain`` (schema
  1.1.0, see docs/BIBLIOGRAPHY.md) -- a chip distinguishes the model ``family`` ``baseline`` from
  ``foundation``.

Invalid rows are skipped with a warning (stderr) and never reach the board.

CLI::

    python leaderboard/site/generate.py --results leaderboard/results --out <dir>

Importable::

    from leaderboard.site.generate import build_site
    build_site("leaderboard/results", "site_build")
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from jsonschema import Draft202012Validator

# --------------------------------------------------------------------------------------------------
# Board presentation constants (labels only -- never metric/task-specific rendering logic)
# --------------------------------------------------------------------------------------------------

#: Human-readable task titles for the board (falls back to the raw task id).
TASK_TITLES: dict[str, str] = {
    "amc": "Automatic modulation classification",
    "sei": "Specific emitter identification",
    "wideband_detection": "Wideband detection",
    "spectrum_sensing": "Spectrum sensing",
    "interference_id": "Interference identification",
    "protocol_tech_id": "Protocol / technology identification",
}

#: Sensible fixed order for the known tasks; unknown tasks sort alphabetically AFTER these.
TASK_ORDER: tuple[str, ...] = (
    "amc",
    "sei",
    "wideband_detection",
    "spectrum_sensing",
    "interference_id",
    "protocol_tech_id",
    "snr_mobility_recognition",
    "beam_prediction",
    "direction_finding",
    "los_nlos",
    "positioning",
    "har",
    "channel_estimation",
)

#: Committed manifest of DECLARED tasks (canonical id -> title/status/priority/blurb). The
#: generator renders every declared task, so tasks without results still appear (as WIP).
_MANIFEST_NAME: str = "tasks.json"

#: Recognised declared-task build states and their board label + CSS class. ``implemented``
#: tasks render full leaderboards when they have results; ``wip``/``planned`` (and any
#: implemented task still lacking results) render a WIP card + a minimal WIP page.
_STATUS_BADGE: dict[str, tuple[str, str]] = {
    "implemented": ("implemented", "status-implemented"),
    "wip": ("work in progress", "status-wip"),
    "planned": ("planned", "status-planned"),
}

#: Fallback status when a manifest entry omits/typos ``status`` (treated as WIP -> no table).
_DEFAULT_STATUS: str = "wip"

#: The two board sections a declared task can belong to (drives homepage/sidebar grouping).
#: ``terrestrial_iq`` = raw-IQ signal tasks (the board's current focus); ``csi_sensing`` =
#: CSI/channel-domain tasks (a separate out-of-scope track, see docs/DOWNSTREAM_TASKS.md).
_SCOPE_TITLES: dict[str, str] = {
    "terrestrial_iq": "Terrestrial IQ tasks",
    "csi_sensing": "CSI / RF-sensing tasks",
}

#: Fallback scope when a manifest entry omits/typos ``scope``.
_DEFAULT_SCOPE: str = "terrestrial_iq"


def _scope_label(scope: str) -> str:
    """Human label for a task scope bucket (falls back to a de-underscored form)."""
    return _SCOPE_TITLES.get(scope, scope.replace("_", " "))


#: Stable display order of the four locked regimes (D5); unknown regimes sort after.
REGIME_ORDER: tuple[str, ...] = (
    "from_scratch",
    "full_finetune",
    "linear_probe",
    "few_shot",
)

#: Human-readable regime labels (falls back to a de-underscored title-ish form).
REGIME_TITLES: dict[str, str] = {
    "from_scratch": "from scratch",
    "full_finetune": "full finetune",
    "linear_probe": "linear probe",
    "few_shot": "few shot",
}

#: Bucket for rows whose split declares no ``track`` (single closed-set track tasks).
_DEFAULT_TRACK: str = "all"

#: Human-readable track titles for the board (falls back to a de-underscored form).
TRACK_TITLES: dict[str, str] = {
    "all": "all",
    "closed_set": "closed set",
    "cross_receiver": "cross receiver",
    "cross_day": "cross day",
    "open_set": "open set",
    "detection": "detection",
    "recognition": "recognition",
}

#: Verification badge text/CSS-class per status. ``from_paper``/``from_paper_uncertain`` are
#: hand-curated bibliography rows citing the model's own paper -- never re-run by us (schema
#: 1.1.0); ``from_paper_uncertain`` additionally flags that the split/protocol match with our
#: canonical setting is NOT confirmed (only the dataset family is the same).
_BADGE: dict[str, tuple[str, str]] = {
    "verified": ("verified", "badge-verified"),
    "self_reported": ("self reported", "badge-self"),
    "from_paper": ("from paper", "badge-paper"),
    "from_paper_uncertain": ("from paper (unconfirmed split)", "badge-paper-uncertain"),
}

#: Tie-break trust order for same-score rows: harness-verified first, then a harness self-run,
#: then a confirmed-split paper citation, then a paper citation with unconfirmed split/protocol.
_STATUS_TRUST_RANK: dict[str, int] = {
    "verified": 0,
    "self_reported": 1,
    "from_paper": 2,
    "from_paper_uncertain": 3,
}

#: Family chip text/CSS-class per family.
_FAMILY_CHIP: dict[str, tuple[str, str]] = {
    "baseline": ("baseline", "chip-baseline"),
    "foundation": ("foundation", "chip-foundation"),
}

#: Shared educational content for the Guide page (guide.html). Embedded as a module constant
#: (not read from a file) so the generator stays a single self-contained, stdlib-only module.
#: Each metrics-glossary entry carries ``higher_is_better`` -> an up/down arrow on the page.
_GUIDE: dict[str, Any] = {
    "iq_explainer": (
        "RF receivers sample the radio wave as complex baseband: each sample has an "
        "in-phase (I) part and a quadrature (Q) part, together capturing the signal's "
        "amplitude and phase. Across this board a signal window is stored as a real-valued "
        "(2, L) array — row 0 is the I stream, row 1 is the Q stream, for L time samples "
        "(e.g. (2, 128) on RadioML, (2, 256) on WiSig, (2, 1024)). RF ML models consume this "
        "raw I/Q directly because the modulation and hardware-imprint information lives in the "
        "fine time-domain structure, so most tasks (AMC, SEI, interference/protocol ID) feed "
        "the (2, L) waveform straight in. Spectrograms (a 2-channel real/imag time-frequency "
        "image, e.g. (2, 512, 512)) are used instead for wideband time-frequency detection, "
        "and CSI/channel representations are used for the separate channel-domain tasks."
    ),
    "regimes": (
        (
            "from_scratch",
            "A task-specific model is trained from random initialization on the task's "
            "training split (the standard setting for the specialized baselines).",
        ),
        (
            "linear_probe",
            "The pretrained backbone is frozen and only a linear head is fit on its "
            "per-sample embed() features.",
        ),
        (
            "full_finetune",
            "The whole pretrained model, backbone plus head, is updated end-to-end on the "
            "task's training data.",
        ),
        (
            "few_shot",
            "A frozen-backbone probe fit on only k labelled examples per class (a k-shot "
            "support set).",
        ),
    ),
    "verification": (
        "A score marked verified means a maintainer independently re-ran the evaluation on a "
        "multi-GPU station (eval-only when weights and a Docker image are provided, or a full "
        "re-train for seed baselines) and the result matched the submitted numbers within the "
        "declared tolerance, then signed it with verified_by/date/hardware. self_reported means "
        "someone ran rfbench themselves and submitted the number, unverified by a maintainer. "
        "from paper and from paper (unconfirmed split) are a different kind of row entirely: "
        "nobody ran anything through this repo -- the number is copied from the model's own "
        'publication (docs/BIBLIOGRAPHY.md) purely as a literature reference point. "from '
        "paper\" is used only when the paper's own dataset AND the board's exact canonical "
        'split/protocol match (e.g. AMC on RadioML 2016.10a full-SNR); "from paper (unconfirmed '
        'split)" is used when only the dataset family matches and the exact split, '
        "preprocessing, or sample overlap with our canonical split could not be confirmed. "
        "Confidence on the board comes from the verified track first, self-reported second; the "
        "two paper tiers are context, not a ranking claim against harness-run rows."
    ),
    "data_policy": (
        "No raw data is ever redistributed: datasets are downloaded from their official "
        "source and rebuilt locally via rfbench data prepare, and only the split-index files "
        "plus dataset checksums (and provenance) are versioned in the repo "
        "(leaderboard/splits/). Raw formats (.h5/.npy/.bin/.sigmf-data) are git-ignored and "
        "blocked in CI, so redistribution licensing is never an issue."
    ),
    "split_policy": (
        "If a dataset ships a split already used by the literature, the board adopts it "
        "verbatim and records its provenance in the manifest (e.g. Sig53 uses the official "
        "TorchSig split; RadDet/DeepSense adopt their official splits if provided). Otherwise "
        "it generates a deterministic 80/10/10 train/val/test split stratified by the task's "
        "label structure (e.g. modulation x SNR for AMC, class for interference/protocol ID) "
        "with seed 42; the ratios and seed are baked into the canonical_split_id, and changing "
        "either is a breaking change that bumps the task version."
    ),
    "metrics_glossary": (
        (
            "accuracy_overall",
            "Top-1 classification accuracy over the whole test split; for AMC it is computed "
            "over the full SNR range with no cherry-picking of high-SNR points. Primary metric "
            "for AMC, interference_id, and protocol_tech_id.",
            True,
        ),
        (
            "accuracy_vs_snr",
            "The accuracy-versus-SNR curve reported alongside overall accuracy on datasets "
            "that carry an SNR grid (AMC only; sets with no SNR grid omit it).",
            True,
        ),
        (
            "macro_f1",
            "Unweighted mean of the per-class F1 scores, so every class counts equally "
            "regardless of frequency.",
            True,
        ),
        (
            "rank1_accuracy",
            "Closed-set specific-emitter-identification accuracy: fraction of signals whose "
            "top-1 predicted device is correct. Primary SEI metric, reported separately on the "
            "closed_set, cross_receiver, and cross_day tracks (which are never merged).",
            True,
        ),
        (
            "auroc",
            "Area under the ROC curve for open-set SEI (detecting whether an emitter is in the "
            "known set).",
            True,
        ),
        (
            "eer",
            "Equal-error rate for open-set SEI, the operating point where false-accept and "
            "false-reject rates are equal.",
            False,
        ),
        (
            "mAP",
            "Mean average precision (COCO-style) for wideband time-frequency detection, "
            "averaged over classes/IoU thresholds. Primary detection metric on RadDet.",
            True,
        ),
        (
            "mAR",
            "Mean average recall for wideband detection, reported alongside mAP.",
            True,
        ),
        (
            "IoU",
            "Intersection-over-union between predicted and ground-truth time-frequency boxes; "
            "measures localization quality for detection.",
            True,
        ),
        (
            "pd@pfa",
            "Probability of detection at a fixed probability of false alarm for spectrum-"
            "sensing occupancy; the board's primary sensing operating point is pd@pfa=0.1.",
            True,
        ),
        (
            "inference_latency_ms",
            "Per-window inference latency in milliseconds, reported for the spectrum-sensing "
            "(Wave B) track.",
            False,
        ),
    ),
}

#: The Guide's fixed slug (rendered to ``<guide>.html`` + linked in the nav on every page).
_GUIDE_SLUG: str = "guide"

#: The live GitHub repo (matches the deployed GitHub Pages origin, not the README's stale
#: template-org badge links).
_REPO_URL: str = "https://github.com/crabedesneiges/rf-benchmark-hub"

#: The submission guide on GitHub -- target of the "Submit" top-nav tab and the task-page
#: "Submit a result" sidebar card. No standalone submit.html page exists on this site.
_SUBMISSION_GUIDE_URL: str = f"{_REPO_URL}/blob/main/docs/SUBMISSION.md"

#: A generic "code repository" glyph (NOT the GitHub Octocat, to avoid any trademark/asset
#: concern) used as the homepage's repo-link icon.
_REPO_ICON_SVG: str = (
    '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><polyline points="8 6 2 12 8 18"></polyline>'
    '<polyline points="16 6 22 12 16 18"></polyline></svg>'
)

#: Distinct (color, dash-pattern) pairs cycled across model lines in a curve plot. Chosen
#: for contrast in both light and dark themes; the dash pattern makes lines distinguishable
#: without relying on color alone (accessibility).
_PLOT_SERIES_STYLES: tuple[tuple[str, str], ...] = (
    ("#0b5fff", "none"),
    ("#d1495b", "6 3"),
    ("#00897b", "2 3"),
    ("#8e44ad", "8 3 2 3"),
    ("#e08e0b", "4 2"),
    ("#2c7fb8", "1 3"),
    ("#c0392b", "10 4"),
    ("#16a085", "3 3 1 3"),
)


# --------------------------------------------------------------------------------------------------
# Schema resolution + validation (lazy jsonschema; mirrors evaluate._resolve_schema_path)
# --------------------------------------------------------------------------------------------------
def _resolve_schema_path(schema_name: str) -> Path | None:
    """Locate a JSON schema file.

    Prefers the force-included package data (``rfbench/_schemas`` in an installed wheel),
    then falls back to the repo ``schemas/`` directory when running from a source
    checkout. Returns ``None`` if neither location has the file.
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


def _load_validator() -> Draft202012Validator:
    """Build a Draft 2020-12 validator for ``result.schema.json`` (lazy jsonschema).

    ``jsonschema`` is imported inside this function so importing this module stays
    dependency-free. Raises ``RuntimeError`` (pointing at the install extra) if the schema
    or the library is missing.
    """
    schema_path = _resolve_schema_path("result.schema.json")
    if schema_path is None:
        raise RuntimeError(
            "could not locate result.schema.json (checked rfbench/_schemas and repo schemas/)"
        )
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError as exc:  # pragma: no cover - jsonschema is a hard dep of rfbench
        raise RuntimeError(
            "jsonschema is required to validate leaderboard results; install rfbench "
            "(pip install rfbench) which pins jsonschema>=4.21"
        ) from exc

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _warn(message: str) -> None:
    """Emit a warning to stderr (no logging config needed for a one-shot CLI)."""
    print(f"[generate] warning: {message}", file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# Declared-task manifest (data-driven task list: every declared task appears on the board)
# --------------------------------------------------------------------------------------------------
#: Ordered (key, human-label) pairs of the compact dataset card. Only keys PRESENT in the
#: manifest's ``dataset`` object render a row, so a partial dataset object degrades cleanly.
_DATASET_FIELDS: tuple[tuple[str, str], ...] = (
    ("source", "source"),
    ("n_classes", "classes"),
    ("modality", "modality"),
    ("real_or_synthetic", "real / synthetic"),
    ("conditions", "conditions"),
    ("license", "license"),
    ("split", "split"),
)


class MetricDef(NamedTuple):
    """A metric name + its human-readable definition (from the manifest / guide)."""

    name: str
    definition: str


class DeclaredTask(NamedTuple):
    """One declared downstream task from the manifest (``leaderboard/tasks.json``).

    ``status`` is the declared build state: ``implemented`` (has or will have results),
    ``wip`` (task/track defined, baseline pending) or ``planned`` (blocked / out of the
    current scope). A task renders its full leaderboard only when it actually has result
    rows; otherwise -- whatever its declared status -- it renders a WIP page/card so the
    board never shows a broken empty table.

    Educational fields are OPTIONAL and drive the explanatory header on the task page:
    ``description`` (one-paragraph what/why), ``dataset`` (a name -> value map rendered as a
    compact card), ``primary_metric`` (a ``MetricDef``) and ``secondary_metrics`` (a list of
    ``MetricDef``). Any of them may be empty/None, in which case that piece of the header is
    simply omitted.

    A ``NamedTuple`` (not a ``@dataclass``) so this module stays safe to load BY PATH via
    ``importlib`` without pre-registering it in ``sys.modules`` (the CLI + tests load it that
    way); ``@dataclass`` resolves its module by name at class-definition time, which a
    by-path load breaks.
    """

    id: str
    title: str
    status: str
    priority: str | None
    blurb: str
    description: str = ""
    dataset: dict[str, str] = {}  # noqa: RUF012 - NamedTuple default, never mutated
    primary_metric: MetricDef | None = None
    secondary_metrics: tuple[MetricDef, ...] = ()
    scope: str = _DEFAULT_SCOPE

    def status_label(self) -> str:
        """Human label for the declared status (falls back to a de-underscored form)."""
        return _STATUS_BADGE.get(self.status, (self.status.replace("_", " "), ""))[0]

    def status_class(self) -> str:
        """CSS class for the declared status badge (empty for unknown statuses)."""
        return _STATUS_BADGE.get(self.status, ("", "status-wip"))[1]


def _resolve_manifest_path() -> Path | None:
    """Locate the committed task manifest (``leaderboard/tasks.json``).

    Walks up from this file until a ``leaderboard/tasks.json`` is found, mirroring the
    schema resolver's source-checkout fallback. Returns ``None`` if no manifest exists (the
    board then falls back to rendering only the tasks that have results).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "leaderboard" / _MANIFEST_NAME
        if candidate.is_file():
            return candidate
    return None


def _parse_metric_def(raw: object) -> MetricDef | None:
    """Parse a manifest metric object ``{name, definition}`` into a ``MetricDef``.

    Returns ``None`` unless the object is a dict carrying a non-empty ``name`` (the
    ``definition`` is optional and defaults to an empty string), so a malformed or absent
    entry is silently skipped rather than crashing the build.
    """
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    definition = raw.get("definition")
    return MetricDef(name=name, definition=str(definition) if definition else "")


def _parse_dataset(raw: object) -> dict[str, str]:
    """Parse a manifest ``dataset`` object into a flat name -> string-value map.

    Non-dict input yields an empty map; every value is coerced to ``str`` so the card
    renderer can treat them uniformly. ``name`` is kept alongside the card fields (used as
    the card's title).
    """
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def load_manifest(manifest_path: str | Path | None = None) -> dict[str, DeclaredTask]:
    """Load the declared-task manifest as ``task_id -> DeclaredTask`` (input order kept).

    ``manifest_path`` may point at a specific ``tasks.json``; when ``None`` the committed
    manifest is auto-located next to ``leaderboard/``. A missing or malformed manifest is a
    non-fatal warning and yields an empty mapping (the board then shows only tasks that have
    results). Entries missing an ``id`` are skipped; a missing/unknown ``status`` defaults
    to ``wip`` so the task renders as a WIP page rather than a broken empty table.
    """
    if manifest_path is None:
        resolved = _resolve_manifest_path()
    else:
        resolved = Path(manifest_path)
    if resolved is None or not resolved.is_file():
        if manifest_path is not None:
            _warn(f"manifest not found: {manifest_path}")
        return {}
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"skipping manifest {resolved}: could not read/parse JSON ({exc})")
        return {}
    entries = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        _warn(f"manifest {resolved} has no 'tasks' array; ignoring")
        return {}

    declared: dict[str, DeclaredTask] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("id")
        if not isinstance(task_id, str) or not task_id:
            _warn(f"manifest {resolved}: skipping entry without a string 'id'")
            continue
        status = entry.get("status")
        if status not in _STATUS_BADGE:
            status = _DEFAULT_STATUS
        scope = entry.get("scope")
        if scope not in _SCOPE_TITLES:
            scope = _DEFAULT_SCOPE
        priority = entry.get("priority")
        secondary_raw = entry.get("secondary_metrics")
        secondary = tuple(
            md
            for md in (
                _parse_metric_def(item)
                for item in (secondary_raw if isinstance(secondary_raw, list) else [])
            )
            if md is not None
        )
        declared[task_id] = DeclaredTask(
            id=task_id,
            title=str(entry.get("title") or TASK_TITLES.get(task_id, task_id)),
            status=str(status),
            priority=str(priority) if isinstance(priority, str) and priority else None,
            blurb=str(entry.get("blurb") or ""),
            description=str(entry.get("description") or ""),
            dataset=_parse_dataset(entry.get("dataset")),
            primary_metric=_parse_metric_def(entry.get("primary_metric")),
            secondary_metrics=secondary,
            scope=str(scope),
        )
    return declared


# --------------------------------------------------------------------------------------------------
# Loading + grouping
# --------------------------------------------------------------------------------------------------
def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Read and schema-validate every ``*.json`` under ``results_dir`` (recursively).

    Each file must be a JSON object that validates against ``result.schema.json``. Files
    that fail to parse or validate are SKIPPED with a stderr warning (never added to the
    board). Returns the list of valid result dicts in sorted-by-path order for stable,
    deterministic output.
    """
    validator = _load_validator()
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _warn(f"skipping {path}: could not read/parse JSON ({exc})")
            continue
        if not isinstance(document, dict):
            _warn(f"skipping {path}: top-level JSON is not an object")
            continue
        errors = sorted(validator.iter_errors(document), key=lambda err: err.json_path)
        if errors:
            first = errors[0]
            _warn(
                f"skipping {path}: fails result.schema.json "
                f"({first.json_path}: {first.message})"
            )
            continue
        rows.append(document)
    return rows


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group valid result rows by ``task.name``, preserving input order within a task."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_name = str(row["task"]["name"])
        grouped.setdefault(task_name, []).append(row)
    return grouped


# --------------------------------------------------------------------------------------------------
# Row accessors (all defensive; never assume a specific task/metric)
# --------------------------------------------------------------------------------------------------
def _primary_key(row: dict[str, Any]) -> str:
    """Return the row's primary (ranking) metric key (``metrics.primary``)."""
    return str(row["metrics"]["primary"])


def _primary_value(row: dict[str, Any]) -> float:
    """Return the row's primary-metric scalar (``metrics.values[metrics.primary]``).

    Guaranteed present by the schema (``primary`` must be a key of ``values``); we read it
    defensively and coerce to ``float`` for a total ordering.
    """
    values = row["metrics"]["values"]
    return float(values[_primary_key(row)])


def _scalar_values(row: dict[str, Any]) -> dict[str, float]:
    """Return this row's scalar metric map (``metrics.values``) as name -> float."""
    values = row["metrics"]["values"]
    return {str(k): float(v) for k, v in values.items()}


def _primary_ci(row: dict[str, Any]) -> tuple[float, float] | None:
    """Return the ``(ci_low, ci_high)`` confidence interval on the PRIMARY metric, or ``None``.

    Reads ``metrics.uncertainty[<primary>]`` (schema 1.2.0); absent on pre-1.2.0 rows and on
    literature rows that never got a bootstrap/backfill, in which case there is simply no
    interval to draw or compare.
    """
    uncertainty = row["metrics"].get("uncertainty")
    if not isinstance(uncertainty, dict):
        return None
    entry = uncertainty.get(_primary_key(row))
    if not isinstance(entry, dict):
        return None
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    lo_f, hi_f = float(lo), float(hi)
    return (lo_f, hi_f) if lo_f <= hi_f else (hi_f, lo_f)


def _cis_overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    """Whether two closed intervals overlap. Missing either interval -> no overlap claim.

    Two intervals ``[a_lo, a_hi]`` and ``[b_lo, b_hi]`` overlap iff ``a_lo <= b_hi`` and
    ``b_lo <= a_hi``. When either row lacks a CI we cannot assert a statistical tie, so we
    conservatively return ``False`` (the strict ranking stands, unannotated).
    """
    if a is None or b is None:
        return False
    return a[0] <= b[1] and b[0] <= a[1]


def _overlap_with_above(ordered: list[dict[str, Any]]) -> list[bool]:
    """Per-row flag: does this row's PRIMARY CI overlap the row directly above it?

    ``ordered`` is the already-ranked group (see :func:`_sort_rows`). The first row is never
    flagged (nothing above it). A flagged row is statistically indistinguishable from its
    predecessor on the primary metric -- the rank gap is within confidence-interval noise.
    The ordering is NOT changed: overlap only annotates, never reranks (ranking on noise
    would be dishonest), so the deterministic primary/trust/name order is preserved.
    """
    flags: list[bool] = [False] * len(ordered)
    for i in range(1, len(ordered)):
        flags[i] = _cis_overlap(_primary_ci(ordered[i]), _primary_ci(ordered[i - 1]))
    return flags


def _curves(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return this row's curve map (``metrics.curves``) as name -> list of points.

    The schema allows each curve to be an array of ``{x, y[, label]}`` points; some tooling
    may also nest them under ``{"points": [...]}``. Both shapes are normalised to the flat
    list-of-points form here so the plot code stays shape-agnostic. Absent/empty curves map
    to an empty dict (the group simply renders no plots).
    """
    metrics = row["metrics"]
    raw = metrics.get("curves")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for name, curve in raw.items():
        points: Any = curve
        if isinstance(curve, dict):
            points = curve.get("points", [])
        if not isinstance(points, list):
            continue
        clean = [p for p in points if isinstance(p, dict) and "x" in p and "y" in p]
        if clean:
            out[str(name)] = clean
    return out


def _regime_name(row: dict[str, Any]) -> str:
    """Return the declared regime name (D5: always present, never inferred)."""
    return str(row["regime"]["name"])


def _regime_label(row_or_name: dict[str, Any] | str) -> str:
    """Human label for a regime, expanding ``few_shot`` to include its ``k_shot``.

    Accepts either a full row (so ``k_shot`` can be read) or a bare regime name.
    """
    if isinstance(row_or_name, str):
        return REGIME_TITLES.get(row_or_name, row_or_name.replace("_", " "))
    name = _regime_name(row_or_name)
    label = REGIME_TITLES.get(name, name.replace("_", " "))
    if name == "few_shot":
        k = row_or_name["regime"].get("k_shot")
        if k is not None:
            return f"{label} (k={k})"
    return label


def _k_shot(row: dict[str, Any]) -> int | None:
    """Return the declared ``k_shot`` for a ``few_shot`` row, else ``None``.

    Two ``few_shot`` results with different ``k`` are NOT comparable (k=50 vs k=500 probe
    very different amounts of labelled data), so they must never share a group/table.
    """
    if _regime_name(row) != "few_shot":
        return None
    k = row["regime"].get("k_shot")
    return int(k) if k is not None else None


def _track_name(row: dict[str, Any]) -> str:
    """Return the row's evaluation track, defaulting to ``all``.

    ``track`` is free-form and OPTIONAL: it may live at ``eval.conditions.track`` or at
    ``split.track`` (checked in that order). SEI reports closed_set / cross_receiver /
    cross_day / open_set and detection reports detection / recognition as separate rows,
    while AMC/sensing typically omit it. A missing/empty track maps to the default ``all``
    bucket so single-track tasks still render.
    """
    eval_block = row.get("eval")
    conditions = eval_block.get("conditions", {}) if isinstance(eval_block, dict) else {}
    candidate = conditions.get("track") if isinstance(conditions, dict) else None
    if candidate is None:
        candidate = row["split"].get("track")
    if candidate is None:
        return _DEFAULT_TRACK
    text = str(candidate).strip()
    return text or _DEFAULT_TRACK


def _track_label(track: str) -> str:
    """Human label for a track (falls back to a de-underscored form)."""
    return TRACK_TITLES.get(track, track.replace("_", " "))


def _status(row: dict[str, Any]) -> str:
    """Return the verification status (``verified`` | ``self_reported``)."""
    return str(row["verification"]["status"])


def _family(row: dict[str, Any]) -> str | None:
    """Return the model family (``baseline`` | ``foundation``) if declared."""
    family = row["model"].get("family")
    return str(family) if family is not None else None


# --------------------------------------------------------------------------------------------------
# Ordering helpers
# --------------------------------------------------------------------------------------------------
def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by primary metric DESC, then verified-first, then model name.

    All v1 primary metrics are 'higher is better', so descending primary gives the board
    ranking; verified rows break ties ahead of self-reported ones, and the model name is a
    final deterministic tiebreak.

    Confidence intervals (schema 1.2.0 ``metrics.uncertainty``) DELIBERATELY do not enter
    this key. Reranking two rows because their CIs overlap would let statistical noise
    reshuffle the board (and would not even be a total order -- overlap is not transitive),
    so the strict primary/trust/name order stands. Overlap is surfaced instead as a
    non-reordering annotation on the rendered row (see :func:`_overlap_with_above` and
    :func:`_render_group_table`): the reader sees "within CI of the row above" without the
    rank silently changing on noise.
    """
    return sorted(
        rows,
        key=lambda r: (
            -_primary_value(r),
            _STATUS_TRUST_RANK.get(_status(r), len(_STATUS_TRUST_RANK)),
            str(r["model"]["name"]),
        ),
    )


def _task_sort_key(task: str) -> tuple[int, str]:
    """Order tasks by the fixed known order, unknown tasks last (alphabetical)."""
    if task in TASK_ORDER:
        return (TASK_ORDER.index(task), task)
    return (len(TASK_ORDER), task)


def _regime_sort_key(regime: str) -> tuple[int, str]:
    """Order regimes by the locked D5 order, unknown regimes last (alphabetical)."""
    if regime in REGIME_ORDER:
        return (REGIME_ORDER.index(regime), regime)
    return (len(REGIME_ORDER), regime)


def _track_sort_key(track: str) -> tuple[int, str]:
    """Order tracks with the default ``all`` bucket first, then alphabetically."""
    return (0 if track == _DEFAULT_TRACK else 1, track)


def _ordered_scalar_keys(rows: list[dict[str, Any]], primary_key: str) -> list[str]:
    """Discover every scalar metric across ``rows``: primary first, then the rest sorted.

    This is the core of the data-driven table: the column set is the UNION of every key in
    each row's ``metrics.values`` -- so a new task's metrics render automatically -- with the
    primary metric pinned to the front for emphasis.
    """
    keys: set[str] = set()
    for row in rows:
        keys.update(_scalar_values(row))
    rest = sorted(k for k in keys if k != primary_key)
    return ([primary_key] if primary_key in keys else []) + rest


def _ordered_curve_names(rows: list[dict[str, Any]]) -> list[str]:
    """Discover every curve metric name across ``rows`` (sorted, deduplicated)."""
    names: set[str] = set()
    for row in rows:
        names.update(_curves(row))
    return sorted(names)


# --------------------------------------------------------------------------------------------------
# Small formatting + escaping helpers
# --------------------------------------------------------------------------------------------------
def _esc(value: object) -> str:
    """HTML-escape a value for safe insertion into markup."""
    return html.escape(str(value), quote=True)


def _fmt_metric(value: float) -> str:
    """Format a metric scalar for the board (4 decimals, stable string)."""
    return f"{value:.4f}"


def _fmt_params(n_params: object) -> str:
    """Format a parameter count compactly (e.g. 289K, 2.5M), or an en-dash if absent."""
    if not isinstance(n_params, int):
        return "&ndash;"
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.1f}M"
    if n_params >= 1_000:
        return f"{n_params / 1_000:.0f}K"
    return str(n_params)


def _fmt_axis(value: float) -> str:
    """Format an axis tick label (drops a trailing ``.0`` for integer-valued ticks)."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


# --------------------------------------------------------------------------------------------------
# Inline-SVG line plot (stdlib only -- polylines computed from the curve points)
# --------------------------------------------------------------------------------------------------
def _render_bar(value: float, vmax: float) -> str:
    """Render a tiny horizontal bar (0..vmax) visualising a primary score."""
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    pct = f"{frac * 100:.1f}"
    return (
        '<span class="bar" role="img" '
        f'aria-label="{_esc(_fmt_metric(value))}">'
        f'<span class="bar-fill" style="width:{pct}%"></span></span>'
    )


def _render_curve_plot(
    curve_name: str,
    series: list[tuple[str, list[dict[str, Any]]]],
) -> str:
    """Render one inline-SVG line plot overlaying every model's curve in a group.

    ``series`` is a list of ``(model_name, points)`` where each point is a ``{x, y}`` dict.
    Axes, gridlines and a legend are drawn; each series gets a distinct color + dash pattern
    (cycled from ``_PLOT_SERIES_STYLES``) so lines stay distinguishable without color alone.
    Every x/y is computed here -- there is no JS and no external chart library.
    """
    # Collect the global x/y ranges across all series.
    xs = [float(p["x"]) for _, pts in series for p in pts]
    ys = [float(p["y"]) for _, pts in series for p in pts]
    if not xs or not ys:
        return ""
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1.0
    # Pad the y range a touch and clamp to sensible bounds (curves are usually 0..1 rates).
    yspan = (ymax - ymin) or 1.0
    ymin = min(ymin, ymin - 0.05 * yspan)
    ymax = max(ymax, ymax + 0.05 * yspan)
    if ymax == ymin:
        ymax = ymin + 1.0

    # SVG geometry (viewBox coordinates; scales responsively via CSS width:100%).
    width, height = 720, 340
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    parts: list[str] = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(curve_name)} line plot" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Gridlines + y-axis ticks (5 horizontal bands).
    n_yticks = 5
    for i in range(n_yticks + 1):
        yval = ymin + (ymax - ymin) * i / n_yticks
        y = sy(yval)
        parts.append(
            f'<line class="grid" x1="{pad_l:.1f}" y1="{y:.1f}" '
            f'x2="{pad_l + plot_w:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_esc(_fmt_axis(yval))}</text>'
        )
    # x-axis ticks (min .. max in a few steps).
    n_xticks = 5
    for i in range(n_xticks + 1):
        xval = xmin + (xmax - xmin) * i / n_xticks
        x = sx(xval)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{pad_t:.1f}" '
            f'x2="{x:.1f}" y2="{pad_t + plot_h:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 16:.1f}" '
            f'text-anchor="middle">{_esc(_fmt_axis(xval))}</text>'
        )

    # Axis frame.
    parts.append(
        f'<line class="axis" x1="{pad_l:.1f}" y1="{pad_t + plot_h:.1f}" '
        f'x2="{pad_l + plot_w:.1f}" y2="{pad_t + plot_h:.1f}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{pad_l:.1f}" y1="{pad_t:.1f}" '
        f'x2="{pad_l:.1f}" y2="{pad_t + plot_h:.1f}"/>'
    )

    # Series polylines.
    legend_items: list[str] = []
    for idx, (model_name, pts) in enumerate(series):
        color, dash = _PLOT_SERIES_STYLES[idx % len(_PLOT_SERIES_STYLES)]
        ordered = sorted(pts, key=lambda p: float(p["x"]))
        coords = " ".join(f"{sx(float(p['x'])):.1f},{sy(float(p['y'])):.1f}" for p in ordered)
        dash_attr = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr} points="{coords}"/>'
        )
        # Point markers for readability.
        for p in ordered:
            parts.append(
                f'<circle cx="{sx(float(p["x"])):.1f}" cy="{sy(float(p["y"])):.1f}" '
                f'r="2.2" fill="{color}"/>'
            )
        legend_items.append(
            '<span class="legend-item">'
            f'<svg class="legend-swatch" viewBox="0 0 24 10" aria-hidden="true">'
            f'<line x1="1" y1="5" x2="23" y2="5" stroke="{color}" stroke-width="2"'
            f"{dash_attr}/></svg>"
            f"<span>{_esc(model_name)}</span></span>"
        )

    parts.append("</svg>")
    legend = f'<div class="legend">{"".join(legend_items)}</div>'
    return (
        '<figure class="plot-figure">'
        f'<figcaption class="plot-title">{_esc(curve_name)}</figcaption>'
        f'{"".join(parts)}{legend}'
        "</figure>"
    )


# --------------------------------------------------------------------------------------------------
# Table rendering (a column per discovered scalar metric)
# --------------------------------------------------------------------------------------------------
def _render_family_chip(family: str | None) -> str:
    """Render the model-family chip (baseline=neutral, foundation=violet)."""
    if family is None:
        return ""
    text, css_class = _FAMILY_CHIP.get(family, (family, "chip-baseline"))
    return f'<span class="chip {css_class}">{_esc(text)}</span>'


def _render_badge(status: str) -> str:
    """Render the verification badge span for a status."""
    text, css_class = _BADGE.get(status, (status, "badge-self"))
    return f'<span class="badge {css_class}">{_esc(text)}</span>'


def _render_ci_note(row: dict[str, Any], overlaps_above: bool) -> str:
    """Render the sub-value CI annotation for the primary cell (empty when no CI).

    Shows the primary metric's ``[ci_low, ci_high]`` interval (schema 1.2.0) under the
    point estimate. When the interval overlaps the row directly above, an ``≈`` marker is
    added with a tooltip: the two rows are statistically indistinguishable on the primary
    metric, so the rank gap between them is within confidence-interval noise. The ordering
    itself is unchanged (see :func:`_sort_rows`).
    """
    ci = _primary_ci(row)
    if ci is None:
        return ""
    lo, hi = ci
    ci_text = f"[{_fmt_metric(lo)}, {_fmt_metric(hi)}]"
    if overlaps_above:
        marker = (
            '<span class="ci-tie" title="Within the confidence interval of the row above'
            ' &mdash; statistically indistinguishable on the primary metric.">&asymp;</span>'
        )
        return f'<span class="metric-ci overlap">{marker}{_esc(ci_text)}</span>'
    return f'<span class="metric-ci">{_esc(ci_text)}</span>'


def _render_row(
    rank: int,
    row: dict[str, Any],
    scalar_keys: list[str],
    primary_key: str,
    primary_max: float,
    *,
    overlaps_above: bool = False,
) -> str:
    """Render one ``<tr>``: rank, model (+family chip, +params), each scalar, status.

    A cell is rendered for EVERY discovered scalar metric so no metric is left out; a metric
    absent from this particular row shows an en-dash. The primary column carries the score
    bar and is visually emphasised, plus its confidence interval (schema 1.2.0) when known;
    ``overlaps_above`` flags a CI overlap with the preceding row (annotation only, never a
    reorder -- see :func:`_sort_rows`).
    """
    model = row["model"]
    name = _esc(model["name"])
    url = model.get("url")
    if isinstance(url, str) and url:
        name = f'<a href="{_esc(url)}">{name}</a>'
    chip = _render_family_chip(_family(row))
    params = _fmt_params(model.get("n_params"))

    values = _scalar_values(row)
    metric_cells: list[str] = []
    for key in scalar_keys:
        if key not in values:
            metric_cells.append('<td class="num">&ndash;</td>')
            continue
        formatted = _fmt_metric(values[key])
        if key == primary_key:
            bar = _render_bar(values[key], primary_max)
            ci_note = _render_ci_note(row, overlaps_above)
            cell_class = "num primary overlap" if overlaps_above else "num primary"
            metric_cells.append(
                f'<td class="{cell_class}"><span class="metric-val">{formatted}</span>'
                f"{ci_note}{bar}</td>"
            )
        else:
            metric_cells.append(f'<td class="num">{formatted}</td>')

    badge = _render_badge(_status(row))
    rank_html = f'<span class="rank-badge">{rank}</span>' if rank == 1 else str(rank)
    return (
        "<tr>"
        f'<td class="rank num">{rank_html}</td>'
        f'<td class="model"><span class="model-name">{name}</span>{chip}'
        f'<span class="params">{params}</span></td>'
        f"{''.join(metric_cells)}"
        f'<td class="status">{badge}</td>'
        "</tr>"
    )


def _render_group_table(
    regime: str,
    track: str,
    rows: list[dict[str, Any]],
    primary_key: str,
    scalar_keys: list[str],
) -> str:
    """Render the leaderboard table for one ``(regime, track)`` group.

    Columns: ``#``, ``Model`` (name + family chip + params), one column per discovered
    scalar metric (primary first + emphasised), ``Status``. Rows are sorted by the primary
    metric, descending. The table carries ``data-regime`` and ``data-track`` so the
    no-mixing invariant is checkable from the rendered HTML.
    """
    ordered = _sort_rows(rows)
    primary_max = max((_primary_value(r) for r in ordered), default=1.0)
    overlap_flags = _overlap_with_above(ordered)

    head_metric_cells = "".join(
        (
            f'<th class="num primary">{_esc(k)}<span class="col-note">primary</span></th>'
            if k == primary_key
            else f'<th class="num">{_esc(k)}</th>'
        )
        for k in scalar_keys
    )
    header = (
        "<thead><tr>"
        '<th class="rank">#</th><th class="model">Model</th>'
        f"{head_metric_cells}"
        '<th class="status">Status</th>'
        "</tr></thead>"
    )
    body_rows = "\n".join(
        _render_row(
            i,
            row,
            scalar_keys,
            primary_key,
            primary_max,
            overlaps_above=overlap_flags[i - 1],
        )
        for i, row in enumerate(ordered, start=1)
    )
    return (
        '<div class="table-scroll">'
        f'<table data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f"{header}<tbody>\n{body_rows}\n</tbody></table>"
        "</div>"
    )


def _render_group(
    regime: str,
    track: str,
    rows: list[dict[str, Any]],
    primary_key: str,
) -> str:
    """Render one ``(regime, track)`` group: its table plus one plot per curve metric.

    Genericity + the plot-OR-table-for-every-metric rule are realised here: the scalar keys
    are discovered from the group's rows (every scalar gets a column) and the curve names
    are discovered too (every curve gets an inline-SVG plot overlaying the group's models).
    Nothing here is task-specific, and because the group is a single (regime, track), no
    table or plot ever mixes two regimes nor two tracks.
    """
    scalar_keys = _ordered_scalar_keys(rows, primary_key)
    table = _render_group_table(regime, track, rows, primary_key, scalar_keys)

    # One inline-SVG plot per discovered curve metric (skipped gracefully if none).
    plots: list[str] = []
    for curve_name in _ordered_curve_names(rows):
        series: list[tuple[str, list[dict[str, Any]]]] = []
        for row in _sort_rows(rows):
            curves = _curves(row)
            if curve_name in curves:
                series.append((str(row["model"]["name"]), curves[curve_name]))
        plot = _render_curve_plot(curve_name, series)
        if plot:
            plots.append(plot)
    plots_html = f'<div class="plots">{"".join(plots)}</div>' if plots else ""

    # A clear label for the group. A single-track task (everything in the default 'all'
    # bucket) is labelled by regime only; multi-track tasks name the track too.
    regime_label = _regime_label(rows[0])
    if track == _DEFAULT_TRACK:
        heading = f"Regime &middot; {_esc(regime_label)}"
    else:
        heading = (
            f"Regime &middot; {_esc(regime_label)} &nbsp;/&nbsp; "
            f"Track &middot; {_esc(_track_label(track))}"
        )
    return (
        '<section class="group" '
        f'data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f'<h3 class="group-title">{heading}</h3>'
        f"{table}{plots_html}"
        "</section>"
    )


# --------------------------------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------------------------------
def _task_meta_line(task_name: str, rows: list[dict[str, Any]]) -> str:
    """Build the mono sub-line under a task heading (primary metric + row/model counts)."""
    primary = _primary_key(rows[0])
    n_models = len({str(r["model"]["name"]) for r in rows})
    datasets = sorted({str(r["dataset"]["name"]) for r in rows})
    n_rows = len(rows)
    bits = [
        f"primary = {primary}",
        f"{n_rows} result{'s' if n_rows != 1 else ''}",
        f"{n_models} model{'s' if n_models != 1 else ''}",
    ]
    if datasets:
        bits.insert(0, "datasets: " + ", ".join(datasets))
    return _esc(" · ".join(bits))


def _task_title(task_name: str, declared: dict[str, DeclaredTask]) -> str:
    """Best human title for a task: manifest title, then the static map, then the id."""
    entry = declared.get(task_name)
    if entry is not None:
        return entry.title
    return TASK_TITLES.get(task_name, task_name)


def _all_task_ids(
    grouped: dict[str, list[dict[str, Any]]],
    declared: dict[str, DeclaredTask],
) -> list[str]:
    """Return every task id to show on the board (declared UNION has-results), ordered.

    The manifest is additive, not a filter: a task that has result rows but is missing from
    the manifest is still listed (defensive), and every declared task is listed even with no
    results (it renders as WIP). Ordering follows the fixed known order then alphabetical.
    """
    ids = set(declared) | set(grouped)
    return sorted(ids, key=_task_sort_key)


def _render_status_badge(entry: DeclaredTask) -> str:
    """Render the declared-status badge (implemented | work in progress | planned)."""
    css_class = entry.status_class()
    label = entry.status_label()
    priority = f" &middot; {_esc(entry.priority)}" if entry.priority else ""
    return f'<span class="status-badge {css_class}">{_esc(label)}{priority}</span>'


def _render_metric_def(md: MetricDef, *, primary: bool) -> str:
    """Render one metric definition line (name + its definition; primary is emphasised)."""
    marker = '<span class="metric-def-tag">primary</span>' if primary else ""
    definition = f" &mdash; {_esc(md.definition)}" if md.definition else ""
    css = "metric-def metric-def-primary" if primary else "metric-def"
    return (
        f'<li class="{css}"><code class="metric-def-name">{_esc(md.name)}</code>'
        f"{marker}{definition}</li>"
    )


def _render_dataset_card(dataset: dict[str, str]) -> str:
    """Render the compact dataset card (a definition list of the present dataset fields).

    Only fields present in ``dataset`` render a row (from the fixed ``_DATASET_FIELDS``
    order), so a partial dataset object degrades cleanly. Returns an empty string if the
    object carries none of the card fields (a bare ``name`` alone still shows as the title).
    """
    name = dataset.get("name", "")
    rows: list[str] = []
    for key, label in _DATASET_FIELDS:
        value = dataset.get(key)
        if not value:
            continue
        rows.append(
            f'<div class="ds-row"><dt class="ds-key">{_esc(label)}</dt>'
            f'<dd class="ds-val">{_esc(value)}</dd></div>'
        )
    if not name and not rows:
        return ""
    title = (
        f'<p class="ds-title"><span class="ds-label">Dataset</span> '
        f"<strong>{_esc(name)}</strong></p>"
        if name
        else '<p class="ds-title"><span class="ds-label">Dataset</span></p>'
    )
    body = f'<dl class="ds-grid">{"".join(rows)}</dl>' if rows else ""
    return f'<div class="dataset-card">{title}{body}</div>'


def _render_metrics_block(entry: DeclaredTask) -> str:
    """Render the primary + secondary metric-definition block for a task header."""
    items: list[str] = []
    if entry.primary_metric is not None:
        items.append(_render_metric_def(entry.primary_metric, primary=True))
    for md in entry.secondary_metrics:
        items.append(_render_metric_def(md, primary=False))
    if not items:
        return ""
    return (
        '<div class="metrics-block"><p class="metrics-block-title">Metrics</p>'
        f'<ul class="metric-def-list">{"".join(items)}</ul></div>'
    )


def _render_task_header(
    entry: DeclaredTask | None, *, include_dataset_and_metrics: bool = True
) -> str:
    """Render the explanatory header (description + dataset card + metric defs) for a task.

    Every piece is optional and driven by the manifest, so an ``entry`` of ``None`` (task
    not in the manifest) or an entry missing any educational field simply omits that piece.
    Returns an empty string when there is nothing educational to show, so the generic
    leaderboard rendering below is never disturbed.

    ``include_dataset_and_metrics=False`` renders only the description callout, omitting the
    dataset-card/metrics-block grid -- used by the two-column task-page layout, which instead
    renders that same dataset card + metrics block (unchanged) further down the main column,
    alongside a new compact "Task details" sidebar summary (see ``_render_task_details_card``).
    """
    if entry is None:
        return ""
    parts: list[str] = []
    if entry.description:
        parts.append(
            '<div class="task-desc">'
            '<span class="task-desc-label">What is this task?</span>'
            f"<p>{_esc(entry.description)}</p>"
            "</div>"
        )
    if include_dataset_and_metrics:
        card = _render_dataset_card(entry.dataset)
        metrics = _render_metrics_block(entry)
        if card or metrics:
            parts.append(f'<div class="task-header-grid">{card}{metrics}</div>')
    if not parts:
        return ""
    return f'<section class="task-header">{"".join(parts)}</section>'


def _render_task_sidebar(
    nav_task_ids: list[str], declared: dict[str, DeclaredTask], current: str
) -> str:
    """Render the task-to-task sidebar, grouped by scope, current task highlighted.

    Purely additive: this is a NEW navigation surface inside the two-column task-page body.
    It does not replace or alter ``_task_nav``'s output (the top header's Home/task/Guide
    chips), so the tests pinning that nav's exact markup are unaffected.
    """
    buckets = _partition_by_scope(nav_task_ids, declared)
    groups: list[str] = []
    for scope in ("terrestrial_iq", "csi_sensing"):
        task_ids = buckets.get(scope)
        if not task_ids:
            continue
        links: list[str] = []
        for task_id in task_ids:
            entry = declared.get(task_id)
            title = _task_title(task_id, declared)
            dot_class = f"dot-{entry.status_class()}" if entry is not None else "dot-status-wip"
            priority = (
                f'<span class="sidebar-priority">{_esc(entry.priority)}</span>'
                if (entry is not None and entry.priority)
                else ""
            )
            active = " sidebar-task-active" if task_id == current else ""
            links.append(
                f'<a class="sidebar-task-link{active}" href="{_esc(task_id)}.html">'
                f'<span class="dot {dot_class}"></span>'
                f'<span class="sidebar-task-title">{_esc(title)}</span>{priority}'
                "</a>"
            )
        groups.append(
            f'<div class="sidebar-group"><h3>{_esc(_scope_label(scope))}</h3>{"".join(links)}</div>'
        )
    return f'<nav class="task-sidebar" aria-label="Task navigation">{"".join(groups)}</nav>'


def _render_task_details_card(
    entry: DeclaredTask | None, task_name: str, rows: list[dict[str, Any]] | None
) -> str:
    """Render the compact "Task details" sidebar card (status, dataset, metric, models).

    Returns an empty string for an undeclared task (``entry is None``), mirroring
    ``_render_task_header``'s optionality contract, so an undeclared-but-has-results page
    (manifest additive) still renders sensibly with no crash.
    """
    if entry is None:
        return ""
    rows = rows or []
    dataset_name = entry.dataset.get("name")
    primary_name = entry.primary_metric.name if entry.primary_metric is not None else None
    tracks = sorted({_track_name(r) for r in rows}) if rows else []
    n_models = len({str(r["model"]["name"]) for r in rows}) if rows else 0

    fields: list[tuple[str, str]] = [
        ("Status", entry.status_label()),
        ("Priority", entry.priority or "—"),
    ]
    if dataset_name:
        fields.append(("Dataset", dataset_name))
    if primary_name:
        fields.append(("Primary metric", primary_name))
    fields.append(("Track", ", ".join(_track_label(t) for t in tracks) if tracks else "—"))
    fields.append(("Models on board", str(n_models)))

    rows_html = "".join(
        f'<div class="details-row"><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>' for k, v in fields
    )
    return (
        '<div class="task-details-card"><h3>Task details</h3>'
        f'<dl class="details-grid">{rows_html}</dl></div>'
    )


def _render_submit_card() -> str:
    """Render the static "Submit a result" sidebar CTA (links to docs/SUBMISSION.md)."""
    return (
        '<div class="submit-card"><h3>Submit a result</h3>'
        "<p>Add a JSON result validated against result.schema.json and open a pull "
        "request. Rows are auto-ranked on merge.</p>"
        f'<a class="submit-cta" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Submission guide</a>'
        "</div>"
    )


#: A generic "empty tray" glyph for the no-baseline empty state (stdlib inline SVG, no new
#: binary assets; a distinct class from ``.plot`` keeps the "no plot on a WIP page" test true).
_EMPTY_STATE_SVG: str = (
    '<svg class="empty-glyph" viewBox="0 0 24 24" width="32" height="32" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="4"></rect>'
    '<line x1="8" y1="12" x2="16" y2="12"></line></svg>'
)


def render_wip_page(
    entry: DeclaredTask, nav_task_ids: list[str], declared: dict[str, DeclaredTask] | None = None
) -> str:
    """Render a minimal "work in progress" page for a declared task with no results.

    Shows the task title, a clear WIP badge/state and the manifest blurb -- but NO
    leaderboard table and NO plots, so a task without a baseline never renders a broken
    empty table. Kept deliberately spare; it is replaced by the full leaderboard the moment
    a valid ``result.json`` for the task lands. Uses the same two-column shell (sidebar +
    main + details/submit cards) as ``render_task_page``.
    """
    declared = declared if declared is not None else {entry.id: entry}
    title = entry.title
    badge = _render_status_badge(entry)
    header = _render_task_header(entry, include_dataset_and_metrics=False)
    dataset_card = _render_dataset_card(entry.dataset)
    metrics_block = _render_metrics_block(entry)
    blurb = f'<p class="wip-blurb">{_esc(entry.blurb)}</p>' if entry.blurb else ""
    sidebar = _render_task_sidebar(nav_task_ids, declared, entry.id)
    details_card = _render_task_details_card(entry, entry.id, None)
    submit_card = _render_submit_card()
    body = (
        '<section class="task">'
        f'<p class="breadcrumb"><a href="index.html">Tasks</a> / {_esc(title)}</p>'
        f'<h2 class="task-title">{_esc(title)}</h2>'
        f'<p class="task-meta">{badge}</p>'
        f"{header}"
        '<div class="task-layout">'
        f"{sidebar}"
        '<div class="task-main">'
        f"{dataset_card}{metrics_block}"
        '<div class="wip-card">'
        '<div class="empty-state-card">'
        f'<p class="wip-kicker">Work in progress</p>'
        f"{_EMPTY_STATE_SVG}"
        '<p class="empty-state-heading">No baseline submitted yet</p>'
        '<p class="note">This task is declared in the benchmark but has no submitted '
        "results on the board yet. A leaderboard (tables + plots) will appear here "
        "automatically once a valid result is added.</p>"
        f"{blurb}"
        f'<a class="submit-cta" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Read the submission guide</a>'
        "</div>"
        "</div>"
        "</div>"
        '<aside class="task-sidebar-right">'
        f"{details_card}{submit_card}"
        "</aside>"
        "</div>"
        "</section>"
    )
    page_title = f"{title} — RF-Benchmark-Hub"
    return _page(page_title, body, current=entry.id)


def render_task_page(
    task_name: str,
    rows: list[dict[str, Any]],
    nav_task_ids: list[str] | None = None,
    declared: dict[str, DeclaredTask] | None = None,
) -> str:
    """Render the full HTML page for one task.

    Rows are partitioned into ``(regime, k_shot, track)`` groups; each group renders one
    table (a column per discovered scalar metric) and one plot per discovered curve metric.
    A group is a single (regime, k_shot, track) -- for ``few_shot`` rows the shot count is
    part of the key too, since k=50 and k=500 are not a comparable regime (D5 extension) --
    so no table or plot ever mixes regimes, shot counts, nor tracks.
    Groups are ordered by regime (locked D5 order), then k_shot ascending, then track
    (``all`` first).
    """
    if not rows:
        raise ValueError(f"render_task_page called with no rows for task '{task_name}'")

    declared = declared or {}
    title = _task_title(task_name, declared)
    dataset_line = _task_meta_line(task_name, rows)
    entry = declared.get(task_name)
    header = _render_task_header(entry, include_dataset_and_metrics=False)
    dataset_card = _render_dataset_card(entry.dataset) if entry is not None else ""
    metrics_block = _render_metrics_block(entry) if entry is not None else ""

    # (regime, k_shot, track) -> rows, preserving input order within each leaf group.
    groups: dict[tuple[str, int | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((_regime_name(row), _k_shot(row), _track_name(row)), []).append(row)

    def _group_sort_key(rkt: tuple[str, int | None, str]) -> tuple[Any, int, Any]:
        regime, k_shot, track = rkt
        shot_rank = k_shot if k_shot is not None else -1
        return (_regime_sort_key(regime), shot_rank, _track_sort_key(track))

    ordered_keys = sorted(groups, key=_group_sort_key)
    sections = [
        _render_group(regime, track, group_rows, _primary_key(group_rows[0]))
        for (regime, _k, track) in ordered_keys
        for group_rows in (groups[(regime, _k, track)],)
    ]

    nav_ids = nav_task_ids if nav_task_ids is not None else [task_name]
    sidebar = _render_task_sidebar(nav_ids, declared, task_name)
    details_card = _render_task_details_card(entry, task_name, rows)
    submit_card = _render_submit_card()
    body = (
        '<section class="task">'
        f'<p class="breadcrumb"><a href="index.html">Tasks</a> / {_esc(title)}</p>'
        f'<h2 class="task-title">{_esc(title)}</h2>'
        f'<p class="task-meta">{dataset_line}</p>'
        f"{header}"
        '<div class="task-layout">'
        f"{sidebar}"
        '<div class="task-main">'
        f"{dataset_card}{metrics_block}"
        f'<p class="note">Each (regime, track) is ranked separately &mdash; a table or plot '
        "never mixes two regimes nor two tracks (protocol invariant). Badges mark "
        "maintainer-verified rows vs self-reported ones.</p>"
        f"{''.join(sections)}"
        "</div>"
        '<aside class="task-sidebar-right">'
        f"{details_card}{submit_card}"
        "</aside>"
        "</div>"
        "</section>"
    )
    page_title = f"{title} — RF-Benchmark-Hub"
    return _page(page_title, body, current=task_name)


def _best_summary(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Return ``(best_model, best_score_str, primary_key)`` for a task's rows."""
    best = _sort_rows(rows)[0]
    return (
        str(best["model"]["name"]),
        _fmt_metric(_primary_value(best)),
        _primary_key(best),
    )


def _render_result_card(
    task_name: str,
    rows: list[dict[str, Any]],
    declared: dict[str, DeclaredTask],
) -> str:
    """Render an index card for a task that HAS results (best-score summary)."""
    title = _task_title(task_name, declared)
    n_rows = len(rows)
    n_models = len({str(r["model"]["name"]) for r in rows})
    best_model, best_score, primary = _best_summary(rows)
    entry = declared.get(task_name)
    badge = _render_status_badge(entry) if entry is not None else ""
    status = entry.status if entry is not None else "implemented"
    blurb = (
        f'<span class="card-blurb">{_esc(entry.blurb)}</span>'
        if entry is not None and entry.blurb
        else ""
    )
    return (
        f'<a class="task-card hover-elevate" data-status="{_esc(status)}" '
        f'href="{_esc(task_name)}.html">'
        f'<span class="card-title">{_esc(title)}</span>{badge}'
        f"{blurb}"
        f'<span class="card-sub">{_esc(f"{n_rows} results · {n_models} models")}</span>'
        '<div class="card-best-box">'
        f'<span class="card-best">Best: <strong>{_esc(best_model)}</strong> '
        f"&middot; {_esc(primary)} = <strong>{_esc(best_score)}</strong></span>"
        "</div>"
        "</a>"
    )


def _render_wip_card(entry: DeclaredTask) -> str:
    """Render an index card for a declared task WITHOUT results (WIP state, no scores)."""
    badge = _render_status_badge(entry)
    blurb = f'<span class="card-blurb">{_esc(entry.blurb)}</span>' if entry.blurb else ""
    return (
        f'<a class="task-card task-card-wip hover-elevate" data-status="{_esc(entry.status)}" '
        f'href="{_esc(entry.id)}.html">'
        f'<span class="card-title">{_esc(entry.title)}</span>{badge}'
        f"{blurb}"
        '<div class="card-nobaseline-box"><span class="card-sub">no baseline yet</span></div>'
        "</a>"
    )


def _partition_by_scope(
    task_ids: list[str], declared: dict[str, DeclaredTask]
) -> dict[str, list[str]]:
    """Split ``task_ids`` into scope buckets (``terrestrial_iq``/``csi_sensing``), order kept.

    A task absent from ``declared`` (results-only, manifest additive) falls back to
    :data:`_DEFAULT_SCOPE` so it still lands in a section rather than being dropped.
    """
    buckets: dict[str, list[str]] = {}
    for task_id in task_ids:
        entry = declared.get(task_id)
        scope = entry.scope if entry is not None else _DEFAULT_SCOPE
        buckets.setdefault(scope, []).append(task_id)
    return buckets


def _compute_stats(
    grouped: dict[str, list[dict[str, Any]]], declared: dict[str, DeclaredTask]
) -> dict[str, int]:
    """Derive the homepage stats-row counts from data ``render_index`` already has.

    ``live`` counts every task (declared or not) that has at least one valid result --
    matching the additive-manifest invariant. ``eval_tracks`` counts distinct
    ``(task, regime, track)`` triples across every loaded result, a live measure of how many
    separate (never-mixed) leaderboard comparisons the board currently reports.
    """
    n_implemented = sum(1 for e in declared.values() if e.status == "implemented")
    n_live = sum(1 for rows in grouped.values() if rows)
    n_eval_groups = len(
        {
            (task, _regime_name(row), _track_name(row))
            for task, rows in grouped.items()
            for row in rows
        }
    )
    return {
        "tasks_defined": len(declared),
        "implemented": n_implemented,
        "live": n_live,
        "eval_tracks": n_eval_groups,
    }


def _render_stats_row(stats: dict[str, int]) -> str:
    """Render the homepage's 4 big-number stat cards."""
    items = (
        (stats["tasks_defined"], "Tasks defined"),
        (stats["implemented"], "Implemented"),
        (stats["live"], "Live leaderboard" if stats["live"] == 1 else "Live leaderboards"),
        (stats["eval_tracks"], "Evaluation tracks"),
    )
    cards = "".join(
        f'<div class="stat-card"><span class="stat-value">{n}</span>'
        f'<span class="stat-label">{_esc(label)}</span></div>'
        for n, label in items
    )
    return f'<div class="stats-row">{cards}</div>'


def _render_filter_bar() -> str:
    """Render the homepage's search input + status filter pills (wired by the inline JS)."""
    pills = (
        ("all", "All tasks", True),
        ("implemented", "Implemented", False),
        ("wip", "In progress", False),
        ("planned", "Planned", False),
    )
    pill_html = "".join(
        f'<button type="button" class="filter-pill{" filter-pill-active" if active else ""}" '
        f'data-filter="{value}">{_esc(label)}</button>'
        for value, label, active in pills
    )
    return (
        '<div class="filter-bar">'
        '<input type="search" id="task-search" class="search-input" '
        'placeholder="Search a task, dataset or metric...">'
        f'<div class="filter-pills">{pill_html}</div>'
        "</div>"
    )


def _render_index_sections(
    grouped: dict[str, list[dict[str, Any]]],
    declared: dict[str, DeclaredTask],
    ordered_tasks: list[str],
) -> str:
    """Render the homepage's task cards grouped into scope sections (skip empty sections)."""
    buckets = _partition_by_scope(ordered_tasks, declared)
    sections: list[str] = []
    for scope in ("terrestrial_iq", "csi_sensing"):
        task_ids = buckets.get(scope)
        if not task_ids:
            continue
        cards: list[str] = []
        for task_name in task_ids:
            rows = grouped.get(task_name)
            if rows:
                cards.append(_render_result_card(task_name, rows, declared))
            else:
                entry = declared.get(task_name)
                if entry is None:  # pragma: no cover - defensive; ids came from declared/grouped
                    continue
                cards.append(_render_wip_card(entry))
        sections.append(
            f'<section class="task-scope-section" data-scope="{scope}">'
            f'<h2 class="scope-heading">{_esc(_scope_label(scope))} '
            f'<span class="scope-count">({len(task_ids)})</span></h2>'
            f'<div class="card-grid">{"".join(cards)}</div>'
            "</section>"
        )
    return "".join(sections)


def render_index(
    grouped: dict[str, list[dict[str, Any]]],
    declared: dict[str, DeclaredTask] | None = None,
) -> str:
    """Render the ``index.html`` landing page: one card per DECLARED task, grouped by scope.

    Every declared task appears: tasks that have results get a best-score card linking to
    their full leaderboard; declared tasks without results get a WIP card (no scores) linking
    to a minimal WIP page. Any task that has results but is absent from the manifest is still
    shown (the manifest is additive, not a filter), so nothing silently drops off the board.
    Cards are grouped into two sections (Terrestrial IQ / CSI-RF-sensing) by each task's
    declared ``scope``; a section with zero tasks is omitted.
    """
    declared = declared or {}
    ordered_tasks = _all_task_ids(grouped, declared)
    sections_html = _render_index_sections(grouped, declared, ordered_tasks)

    if sections_html:
        stats = _compute_stats(grouped, declared)
        body = (
            '<section class="task">'
            '<p class="hero-eyebrow">Terrestrial RF &middot; Machine Learning</p>'
            '<h1 class="hero-title">RF machine-learning leaderboards</h1>'
            '<p class="hero-lead">Reproducible benchmarks for terrestrial RF '
            "machine-learning tasks, comparing specialised baselines against fine-tuned "
            "foundation models. Each task ranks submissions by its primary metric; regimes "
            "and tracks are never mixed in a comparison.</p>"
            f"{_render_stats_row(stats)}"
            f"{_render_filter_bar()}"
            f"{sections_html}"
            "</section>"
        )
    else:
        body = (
            '<section class="task">'
            '<p class="note">No tasks declared or results yet.</p>'
            "</section>"
        )
    return _page(
        "RF-Benchmark-Hub Leaderboard",
        body,
        current=None,
        extra_body=f"<script>{_JS}</script>" if sections_html else "",
    )


def _render_regimes_section() -> str:
    """Render the four evaluation regimes as a definition list (from ``_GUIDE``)."""
    rows: list[str] = []
    for name, definition in _GUIDE["regimes"]:
        rows.append(
            f'<div class="guide-def"><dt><code>{_esc(name)}</code></dt>'
            f"<dd>{_esc(definition)}</dd></div>"
        )
    return (
        '<section class="guide-section" id="regimes">'
        "<h2>Evaluation regimes</h2>"
        '<p class="note">The declared regime lives in every result.json and is never '
        "inferred; the board never mixes two regimes in one comparison.</p>"
        f'<dl class="guide-deflist">{"".join(rows)}</dl>'
        "</section>"
    )


def _render_glossary_section() -> str:
    """Render the metrics glossary: name + definition + an up/down arrow per metric.

    ``higher_is_better`` picks the arrow (▲ up = higher is better, ▼ down = lower is
    better) and a matching aria-label so the direction is not conveyed by glyph alone.
    """
    rows: list[str] = []
    for name, definition, higher in _GUIDE["metrics_glossary"]:
        if higher:
            arrow, label, css = "&#9650;", "higher is better", "arrow-up"
        else:
            arrow, label, css = "&#9660;", "lower is better", "arrow-down"
        rows.append(
            "<tr>"
            f'<td class="glossary-name"><code>{_esc(name)}</code></td>'
            f'<td class="glossary-dir"><span class="dir-arrow {css}" '
            f'role="img" aria-label="{_esc(label)}">{arrow}</span></td>'
            f'<td class="glossary-def">{_esc(definition)}</td>'
            "</tr>"
        )
    return (
        '<section class="guide-section" id="metrics-glossary">'
        "<h2>Metrics glossary</h2>"
        '<p class="note">The arrow marks the optimisation direction: '
        '<span class="dir-arrow arrow-up" aria-hidden="true">&#9650;</span> higher is better, '
        '<span class="dir-arrow arrow-down" aria-hidden="true">&#9660;</span> lower is '
        "better.</p>"
        '<table class="glossary"><thead><tr>'
        '<th>Metric</th><th class="glossary-dir">Dir.</th><th>Definition</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        "</section>"
    )


def render_guide() -> str:
    """Render the standalone Guide page (``guide.html``) from the shared ``_GUIDE`` content.

    Sections: what I/Q is, the four evaluation regimes, verified-vs-self_reported, the data
    policy, the split policy and a metrics glossary (name + definition + an up/down arrow for
    higher/lower-is-better). Self-contained -- shares the site theme and the same top nav
    (Tasks | Guide | Submit) as every other page.
    """
    body = (
        '<section class="task guide">'
        '<h1 class="task-title">Guide</h1>'
        '<p class="note">How to read this board: the signal representation, the evaluation '
        "regimes, what verification means, how data and splits are handled, and what each "
        "metric measures.</p>"
        '<section class="guide-section" id="what-is-iq">'
        "<h2>What is I/Q?</h2>"
        f'<p>{_esc(_GUIDE["iq_explainer"])}</p>'
        "</section>"
        f"{_render_regimes_section()}"
        '<section class="guide-section" id="verification">'
        "<h2>Verified vs self-reported</h2>"
        f'<p>{_esc(_GUIDE["verification"])}</p>'
        "</section>"
        '<section class="guide-section" id="data-policy">'
        "<h2>Data policy</h2>"
        f'<p>{_esc(_GUIDE["data_policy"])}</p>'
        "</section>"
        '<section class="guide-section" id="split-policy">'
        "<h2>Split policy</h2>"
        f'<p>{_esc(_GUIDE["split_policy"])}</p>'
        "</section>"
        f"{_render_glossary_section()}"
        "</section>"
    )
    return _page("Guide — RF-Benchmark-Hub", body, current=_GUIDE_SLUG)


def _top_nav(current: str | None) -> str:
    """Render the site-wide top nav: Tasks | Guide | Submit, plus a GitHub repo icon link.

    Replaces the old per-task chip list (Home + one chip per task name) -- task-to-task
    navigation now lives in each task page's own sidebar (see ``_render_task_sidebar``), so
    this bar only needs to say which TOP-LEVEL section of the site you're in.

    ``current`` is the Guide slug (``_GUIDE_SLUG``) on the Guide page, or anything else
    (a task id, or ``None`` for the index) everywhere else -- "Tasks" is active whenever
    "Guide" isn't, since every task/WIP page lives under the Tasks section.
    """
    guide_active = current == _GUIDE_SLUG
    tasks_class = "top-tab" if guide_active else "top-tab top-tab-active"
    guide_class = "top-tab top-tab-active" if guide_active else "top-tab"
    return (
        '<div class="top-tabs">'
        f'<a class="{tasks_class}" href="index.html">Tasks</a>'
        f'<a class="{guide_class}" href="{_GUIDE_SLUG}.html">Guide</a>'
        '<a class="top-tab" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Submit</a>'
        "</div>"
        '<a class="icon-link" aria-label="GitHub repository" target="_blank" rel="noopener" '
        f'href="{_esc(_REPO_URL)}">{_REPO_ICON_SVG}</a>'
    )


#: Google Fonts request for the board's typeface pair (Space Grotesk headings, IBM Plex Sans
#: body, IBM Plex Mono code/metrics) -- the ONE external network dependency this site has;
#: every selector falls back to system fonts (see ``--font-*`` in ``_CSS``) if it's blocked.
_GOOGLE_FONTS_LINK: str = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" '
    'rel="stylesheet">\n'
)


def _page(
    title: str,
    body: str,
    current: str | None,
    *,
    extra_body: str = "",
) -> str:
    """Assemble a complete standalone HTML page (header + nav + body + footer).

    ``current`` drives the site-wide top nav's active state (see ``_top_nav``): the Guide
    slug (``_GUIDE_SLUG``) on the Guide page, or anything else (a task id, or ``None`` for
    the index) everywhere else. ``extra_body`` renders just before ``</body>`` (homepage-only
    inline filter script, see ``render_index``); it defaults to empty so every other page
    (task pages, WIP pages, the guide) is unaffected.
    """
    task_nav = _top_nav(current)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"{_GOOGLE_FONTS_LINK}"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        '<header class="site-header">'
        '<div class="brand">'
        f"{_LOGO_SVG}"
        '<div class="brand-text">'
        '<span class="brand-name">RF-Benchmark-Hub</span>'
        '<span class="brand-tag">Reproducible leaderboards for terrestrial RF '
        "machine learning</span>"
        "</div></div>"
        f"{task_nav}"
        "</header>\n"
        f"<main>\n{body}\n</main>\n"
        '<footer class="site-footer"><p>Generated by leaderboard/site/generate.py '
        "&mdash; every row validated against result.schema.json. Charts are inline SVG "
        "computed here in Python; a Google Fonts link and a small homepage-only inline "
        "filter script are the only external/runtime additions.</p></footer>\n"
        f"{extra_body}"
        "</body>\n</html>\n"
    )


# --------------------------------------------------------------------------------------------------
# Theme (self-contained; light + dark via prefers-color-scheme; NO external assets)
# --------------------------------------------------------------------------------------------------
_LOGO_SVG = (
    '<svg class="logo" viewBox="0 0 32 32" aria-hidden="true" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<rect x="1" y="1" width="30" height="30" rx="7" fill="none" '
    'stroke="var(--accent)" stroke-width="2"/>'
    '<path d="M5 20 Q9 8 13 20 T21 20 T29 20" fill="none" stroke="var(--accent)" '
    'stroke-width="2" stroke-linecap="round"/>'
    "</svg>"
)

_CSS = """
:root {
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-2: #f7f8fa;
  --fg: #1a1c20;
  --muted: #5c6470;
  --line: #e2e5ea;
  --line-strong: #cbd0d8;
  --accent: #2f6bff;
  --accent-soft: #e8f0ff;
  --head: #f4f5f7;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif;
  --font-heading: "Space Grotesk", var(--font-body);
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --badge-verified-bg: #e6f6ea; --badge-verified-fg: #137333; --badge-verified-bd: #9fd8ae;
  --badge-self-bg: #fff4e5; --badge-self-fg: #a15c00; --badge-self-bd: #f0c891;
  --badge-paper-bg: #eaf1fb; --badge-paper-fg: #2158a0; --badge-paper-bd: #b9d2f0;
  --badge-paper-uncertain-bg: #f2eefb; --badge-paper-uncertain-fg: #6b4fa0;
  --badge-paper-uncertain-bd: #d6c8f0;
  --chip-baseline-bg: #eef0f3; --chip-baseline-fg: #444b56; --chip-baseline-bd: #d6dae1;
  --chip-foundation-bg: #f1e9fb; --chip-foundation-fg: #6b31c9; --chip-foundation-bd: #d9c4f4;
  --status-impl-bg: #e6f6ea; --status-impl-fg: #137333; --status-impl-bd: #9fd8ae;
  --status-wip-bg: #fdeede; --status-wip-fg: #9a5b00; --status-wip-bd: #f0c891;
  --status-planned-bg: #eef0f3; --status-planned-fg: #5c6470; --status-planned-bd: #d6dae1;
  --bar-track: #eef0f3; --bar-fill: #0b5fff;
  --grid: #edeff2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101317;
    --surface: #161a20;
    --surface-2: #1b2028;
    --fg: #e6e8ec;
    --muted: #9aa3b0;
    --line: #262c35;
    --line-strong: #333b46;
    --accent: #6f97ff;
    --accent-soft: #1a2740;
    --head: #1b2028;
    --badge-verified-bg: #12281a; --badge-verified-fg: #57cc7f; --badge-verified-bd: #2c5b3b;
    --badge-self-bg: #2e2410; --badge-self-fg: #e0a94b; --badge-self-bd: #5c4a1f;
    --badge-paper-bg: #16233a; --badge-paper-fg: #7fb0ea; --badge-paper-bd: #2c4a70;
    --badge-paper-uncertain-bg: #241c38; --badge-paper-uncertain-fg: #b79ce6;
    --badge-paper-uncertain-bd: #3f2f5c;
    --chip-baseline-bg: #20262e; --chip-baseline-fg: #b6bdc8; --chip-baseline-bd: #333b46;
    --chip-foundation-bg: #241a35; --chip-foundation-fg: #b892ec; --chip-foundation-bd: #4a3670;
    --status-impl-bg: #12281a; --status-impl-fg: #57cc7f; --status-impl-bd: #2c5b3b;
    --status-wip-bg: #2e2410; --status-wip-fg: #e0a94b; --status-wip-bd: #5c4a1f;
    --status-planned-bg: #20262e; --status-planned-fg: #9aa3b0; --status-planned-bd: #333b46;
    --bar-track: #20262e; --bar-fill: #5b8dff;
    --grid: #22282f;
  }
}
* { box-sizing: border-box; }
html { color-scheme: light dark; }
body {
  font-family: var(--font-body);
  color: var(--fg); background: var(--bg); margin: 0;
  line-height: 1.5; -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1, h2, h3, .task-title, .card-title, .brand-name, .group-title {
  font-family: var(--font-heading); font-weight: 600; letter-spacing: -0.01em;
}
.hover-elevate { transition: box-shadow .15s ease, border-color .15s ease; }
.hover-elevate:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
@media (prefers-color-scheme: dark) {
  .hover-elevate:hover { box-shadow: 0 2px 14px rgba(0,0,0,0.35); }
}

.site-header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.5rem;
  padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); background: var(--surface);
}
.brand { display: flex; align-items: center; gap: 0.6rem; }
.logo { width: 30px; height: 30px; flex: none; }
.brand-text { display: flex; flex-direction: column; }
.brand-name { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }
.brand-tag { color: var(--muted); font-size: 0.8rem; }
main { max-width: 1320px; margin: 0 auto; padding: 1.5rem 1.5rem 4rem; }
.task-title { font-size: 1.4rem; margin: 0.5rem 0 0.25rem; letter-spacing: -0.01em; }
.task-meta {
  font-family: var(--font-mono);
  color: var(--muted); font-size: 0.8rem; margin: 0 0 0.75rem;
}
.note { color: var(--muted); font-size: 0.85rem; margin: 0.25rem 0 1.25rem; }

.group {
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface);
  padding: 1rem 1.1rem 1.25rem; margin: 0 0 1.5rem;
}
.group-title {
  font-size: 1rem; margin: 0 0 0.75rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--line); font-weight: 600;
}

.table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; min-width: 480px; font-size: 0.9rem; }
th, td {
  border-bottom: 1px solid var(--line); padding: 0.5rem 0.7rem; text-align: left;
  vertical-align: middle;
}
thead th {
  background: var(--head); font-weight: 600; font-size: 0.82rem; color: var(--muted);
  border-bottom: 1px solid var(--line-strong); white-space: nowrap;
}
thead th:first-child { border-top-left-radius: 8px; }
thead th:last-child { border-top-right-radius: 8px; }
tbody tr:last-child td { border-bottom: none; }
th.rank, td.rank { width: 2rem; text-align: right; color: var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
th.primary, td.primary { color: var(--fg); }
td.num.primary .metric-val { font-weight: 700; }
.metric-ci {
  display: block; font-size: 0.66rem; font-weight: 500; color: var(--muted);
  font-variant-numeric: tabular-nums; margin-top: 0.1rem;
}
.metric-ci.overlap { color: var(--accent); }
.ci-tie { margin-right: 0.2rem; font-weight: 700; cursor: help; }
.col-note {
  display: block; font-size: 0.66rem; font-weight: 500; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.model { min-width: 12rem; }
.model-name { font-weight: 600; }
.params {
  display: inline-block; margin-left: 0.5rem; color: var(--muted); font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
}
.bar {
  display: block; height: 5px; width: 100%; max-width: 90px; margin: 0.25rem 0 0 auto;
  background: var(--bar-track); border-radius: 999px; overflow: hidden;
}
.bar-fill { display: block; height: 100%; background: var(--bar-fill); }

.badge, .chip {
  display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px;
  font-size: 0.74rem; font-weight: 600; white-space: nowrap; border: 1px solid transparent;
}
.badge-verified {
  background: var(--badge-verified-bg); color: var(--badge-verified-fg);
  border-color: var(--badge-verified-bd);
}
.badge-self {
  background: var(--badge-self-bg); color: var(--badge-self-fg);
  border-color: var(--badge-self-bd);
}
.badge-paper {
  background: var(--badge-paper-bg); color: var(--badge-paper-fg);
  border-color: var(--badge-paper-bd);
}
.badge-paper-uncertain {
  background: var(--badge-paper-uncertain-bg); color: var(--badge-paper-uncertain-fg);
  border-color: var(--badge-paper-uncertain-bd);
}
.chip { margin-left: 0.4rem; font-size: 0.68rem; padding: 0.02rem 0.45rem; }
.chip-baseline {
  background: var(--chip-baseline-bg); color: var(--chip-baseline-fg);
  border-color: var(--chip-baseline-bd);
}
.chip-foundation {
  background: var(--chip-foundation-bg); color: var(--chip-foundation-fg);
  border-color: var(--chip-foundation-bd);
}

.plots { margin-top: 1.25rem; display: flex; flex-direction: column; gap: 1.25rem; }
.plot-figure { margin: 0; }
.plot-title { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem; }
.plot {
  width: 100%; height: auto; background: var(--surface-2);
  border: 1px solid var(--line); border-radius: 8px;
}
.plot .grid { stroke: var(--grid); stroke-width: 1; }
.plot .axis { stroke: var(--line-strong); stroke-width: 1; }
.plot .tick { fill: var(--muted); font-size: 11px;
  font-family: var(--font-mono); }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; margin-top: 0.5rem; }
.legend-item {
  display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; color: var(--muted);
}
.legend-swatch { width: 24px; height: 10px; flex: none; }

.top-tabs { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-left: auto; }
.top-tab {
  font-size: 0.85rem; font-weight: 500; padding: 0.35rem 0.7rem; border-radius: 8px;
  color: var(--muted);
}
.top-tab:hover { color: var(--fg); text-decoration: none; background: var(--surface-2); }
.top-tab-active { color: var(--accent); font-weight: 600; }
.icon-link {
  display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; border-radius: 8px; color: var(--muted); margin-left: 0.5rem;
}
.icon-link:hover { color: var(--fg); background: var(--surface-2); text-decoration: none; }

.hero-eyebrow {
  color: var(--accent); font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; margin: 0.5rem 0 0.5rem;
}
.hero-title {
  font-size: 2.1rem; margin: 0 0 0.6rem; letter-spacing: -0.02em;
}
.hero-lead { color: var(--muted); font-size: 0.95rem; max-width: 68ch; margin: 0 0 1.5rem; }

.stats-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem; margin: 0 0 1.5rem;
}
.stat-card {
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface);
  padding: 0.9rem 1.1rem; display: flex; flex-direction: column; gap: 0.15rem;
}
.stat-value {
  font-family: var(--font-heading); font-size: 1.9rem; font-weight: 700; color: var(--fg);
}
.stat-label { color: var(--muted); font-size: 0.8rem; }

.filter-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin: 0 0 1.5rem;
}
.search-input {
  flex: 1 1 260px; min-width: 200px; padding: 0.5rem 0.85rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); color: var(--fg);
  font-family: var(--font-body); font-size: 0.88rem;
}
.search-input:focus { outline: none; border-color: var(--accent); }
.filter-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.filter-pill {
  font-family: var(--font-body); font-size: 0.82rem; padding: 0.35rem 0.8rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--surface);
  color: var(--fg); cursor: pointer;
}
.filter-pill:hover { border-color: var(--line-strong); }
.filter-pill-active {
  background: var(--fg); color: var(--bg); border-color: var(--fg);
}
@media (prefers-color-scheme: dark) {
  .filter-pill-active { background: var(--fg); color: var(--bg); }
}

.scope-heading {
  font-size: 1.15rem; margin: 1.75rem 0 0.9rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--line);
}
.scope-heading:first-of-type { margin-top: 0.5rem; }
.scope-count { color: var(--muted); font-weight: 400; font-size: 0.9rem; }

.card-best-box, .card-nobaseline-box {
  border-radius: 10px; padding: 0.5rem 0.7rem; margin-top: 0.25rem;
}
.card-best-box { background: var(--accent-soft); }
.card-nobaseline-box { background: var(--surface-2); border: 1px dashed var(--line-strong); }
.card-best-box .card-best { color: var(--fg); font-size: 0.85rem; }
.card-nobaseline-box .card-sub { color: var(--muted); }

.breadcrumb { color: var(--muted); font-size: 0.8rem; margin: 0 0 0.4rem; }
.breadcrumb a { color: var(--muted); }
.breadcrumb a:hover { color: var(--accent); }

.task-layout {
  display: grid; grid-template-columns: 200px minmax(0, 1fr) 240px; gap: 1.5rem;
  align-items: start; margin-top: 1rem;
}
@media (max-width: 1100px) {
  .task-layout { grid-template-columns: 200px minmax(0, 1fr); }
  .task-sidebar-right { grid-column: 1 / -1; }
}
@media (max-width: 900px) {
  .task-layout { grid-template-columns: 1fr; }
}
.task-main { min-width: 0; }

.task-sidebar { display: flex; flex-direction: column; gap: 1.25rem; }
.sidebar-group h3 {
  font-family: var(--font-heading); font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.5rem;
}
.sidebar-task-link {
  display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.4rem;
  border-radius: 8px; color: var(--fg); font-size: 0.85rem;
}
.sidebar-task-link:hover { background: var(--surface-2); text-decoration: none; }
.sidebar-task-active { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.sidebar-task-title { flex: 1 1 auto; }
.sidebar-priority { color: var(--muted); font-size: 0.72rem; }
.dot {
  width: 8px; height: 8px; border-radius: 999px; flex: none; background: var(--status-wip-fg);
}
.dot-status-implemented { background: var(--status-impl-fg); }
.dot-status-wip { background: var(--status-wip-fg); }
.dot-status-planned { background: var(--status-planned-fg); }

.task-sidebar-right { display: flex; flex-direction: column; gap: 1rem; }
.task-details-card, .submit-card {
  border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem;
}
.task-details-card { background: var(--surface); }
.task-details-card h3, .submit-card h3 {
  font-family: var(--font-heading); font-size: 0.9rem; margin: 0 0 0.75rem;
}
.details-grid { margin: 0; display: grid; gap: 0.5rem; }
.details-row {
  display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.82rem;
}
.details-row dt { color: var(--muted); margin: 0; }
.details-row dd { margin: 0; text-align: right; }
.submit-card {
  background: var(--accent-soft); border-color: var(--accent);
}
.submit-card p { font-size: 0.82rem; color: var(--fg); margin: 0 0 0.9rem; }
.submit-cta {
  display: inline-block; background: var(--accent); color: #fff; font-weight: 600;
  font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 8px;
}
.submit-cta:hover { text-decoration: none; opacity: 0.9; }

.empty-state-card {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  padding: 2rem 1.5rem;
}
.wip-kicker {
  color: var(--status-wip-fg); font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; margin: 0 0 0.75rem;
}
.empty-glyph { color: var(--muted); margin: 0 0 0.75rem; }
.empty-state-heading {
  font-family: var(--font-heading); font-size: 1.1rem; font-weight: 600; margin: 0 0 0.4rem;
}
.empty-state-card .note { max-width: 42ch; }
.empty-state-card .submit-cta { margin-top: 0.75rem; }

.rank-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 999px; background: var(--accent); color: #fff;
  font-size: 0.78rem; font-weight: 700;
}

.card-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem; margin-top: 0.5rem;
}
.task-card {
  display: flex; flex-direction: column; gap: 0.35rem; color: var(--fg);
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface);
  padding: 1rem 1.1rem; transition: box-shadow .15s ease, border-color .15s ease;
}
.task-card:hover {
  border-color: var(--accent); text-decoration: none; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
@media (prefers-color-scheme: dark) {
  .task-card:hover { box-shadow: 0 2px 14px rgba(0,0,0,0.35); }
}
.card-title { font-weight: 700; font-size: 1.05rem; }
.card-sub { color: var(--muted); font-size: 0.8rem;
  font-family: var(--font-mono); }
.card-best { font-size: 0.85rem; }
.card-blurb { color: var(--muted); font-size: 0.82rem; line-height: 1.4; }
.task-card-wip { border-style: dashed; }
.task-card-wip:hover { border-color: var(--status-wip-bd); }

.status-badge {
  display: inline-block; align-self: flex-start; padding: 0.08rem 0.55rem;
  border-radius: 999px; font-size: 0.72rem; font-weight: 600; white-space: nowrap;
  border: 1px solid transparent;
}
.status-implemented {
  background: var(--status-impl-bg); color: var(--status-impl-fg);
  border-color: var(--status-impl-bd);
}
.status-wip {
  background: var(--status-wip-bg); color: var(--status-wip-fg);
  border-color: var(--status-wip-bd);
}
.status-planned {
  background: var(--status-planned-bg); color: var(--status-planned-fg);
  border-color: var(--status-planned-bd);
}
.wip-card {
  border: 1px dashed var(--status-wip-bd); border-radius: 14px; background: var(--surface);
  padding: 1.1rem 1.25rem; margin: 0 0 1.5rem;
}
.wip-state { font-size: 1rem; margin: 0 0 0.5rem; }
.wip-blurb { color: var(--fg); font-size: 0.9rem; margin: 0.75rem 0 0; }

/* Explanatory task header (description + dataset card + metric definitions). */
.task-header { margin: 0 0 1.25rem; }
.task-desc {
  border: 1px solid var(--accent); border-left-width: 4px; border-radius: 10px;
  background: var(--accent-soft); padding: 0.85rem 1.1rem; margin: 0 0 1.1rem; max-width: 68ch;
}
.task-desc-label {
  display: block; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--accent); margin: 0 0 0.35rem;
}
.task-desc p { margin: 0; font-size: 1rem; line-height: 1.5; color: var(--fg); }
.task-header-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem; align-items: start;
}
.dataset-card, .metrics-block {
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface-2);
  padding: 0.9rem 1.1rem;
}
.ds-title { margin: 0 0 0.6rem; font-size: 0.95rem; }
.ds-label, .metrics-block-title {
  display: inline-block; font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--accent);
}
.metrics-block-title { margin: 0 0 0.5rem; }
.ds-grid { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 0.75rem; }
.ds-row { display: contents; }
.ds-key {
  color: var(--muted); font-size: 0.78rem; white-space: nowrap;
  font-family: var(--font-mono);
}
.ds-val { margin: 0; font-size: 0.82rem; color: var(--fg); }
.metric-def-list { margin: 0; padding: 0; list-style: none;
  display: flex; flex-direction: column; gap: 0.5rem; }
.metric-def { font-size: 0.82rem; color: var(--fg); }
.metric-def-name {
  font-family: var(--font-mono);
  font-size: 0.8rem; color: var(--accent);
}
.metric-def-tag {
  display: inline-block; margin-left: 0.4rem; padding: 0.02rem 0.4rem; border-radius: 999px;
  font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  background: var(--accent-soft); color: var(--accent); border: 1px solid var(--accent);
}

/* Guide page. */
.guide-section { margin: 1.5rem 0; }
.guide-section h2 {
  font-size: 1.15rem; margin: 0 0 0.5rem; padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--line);
}
.guide-section p { max-width: 74ch; font-size: 0.92rem; }
.guide-deflist { margin: 0.5rem 0 0; display: flex; flex-direction: column; gap: 0.75rem; }
.guide-def dt { margin: 0 0 0.15rem; }
.guide-def dt code {
  font-family: var(--font-mono);
  font-size: 0.85rem; color: var(--accent); font-weight: 600;
}
.guide-def dd { margin: 0; font-size: 0.9rem; color: var(--fg); max-width: 74ch; }
.glossary { margin-top: 0.75rem; }
.glossary-name code {
  font-family: var(--font-mono);
  font-size: 0.82rem; color: var(--accent);
}
.glossary-def { font-size: 0.88rem; color: var(--fg); }
.glossary-dir, th.glossary-dir { text-align: center; width: 3rem; white-space: nowrap; }
.dir-arrow { font-size: 0.9rem; line-height: 1; }
.arrow-up { color: var(--badge-verified-fg); }
.arrow-down { color: var(--badge-self-fg); }

.site-footer { border-top: 1px solid var(--line); background: var(--surface); }
.site-footer p { max-width: 1080px; margin: 0 auto; padding: 1rem 1.5rem;
  color: var(--muted); font-size: 0.78rem; }
"""

#: Homepage-only inline search/filter script -- vanilla JS, no dependencies, no build step.
#: Degrades gracefully with JS disabled: cards carry no default ``display:none``, so every
#: card stays visible and the search box / pills are simply inert. Injected via ``_page``'s
#: ``extra_body`` kwarg (``render_index`` only); every other page has zero ``<script>``.
_JS: str = """
(function () {
  var search = document.getElementById('task-search');
  var pills = document.querySelectorAll('.filter-pill');
  var cards = document.querySelectorAll('.task-card');
  var activeStatus = 'all';

  function applyFilters() {
    var query = (search && search.value || '').trim().toLowerCase();
    cards.forEach(function (card) {
      var matchesStatus = activeStatus === 'all' || card.dataset.status === activeStatus;
      var matchesQuery = !query || card.textContent.toLowerCase().indexOf(query) !== -1;
      card.style.display = (matchesStatus && matchesQuery) ? '' : 'none';
    });
  }

  if (search) { search.addEventListener('input', applyFilters); }
  pills.forEach(function (pill) {
    pill.addEventListener('click', function () {
      pills.forEach(function (p) { p.classList.remove('filter-pill-active'); });
      pill.classList.add('filter-pill-active');
      activeStatus = pill.dataset.filter;
      applyFilters();
    });
  });
})();
"""


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
def build_site(results_dir: str | Path, out_dir: str | Path) -> Path:
    """Build the static leaderboard site from ``results_dir`` into ``out_dir``.

    Reads and schema-validates every ``results_dir/**/*.json`` (invalid rows skipped with a
    warning), groups the valid rows by task, and reads the committed task manifest
    (``leaderboard/tasks.json``). Writes ``index.html`` -- a card for EVERY declared task --
    plus one ``<task>.html`` page per task: a full leaderboard for tasks that have results, a
    minimal "work in progress" page for declared tasks without results (never a broken empty
    table). Also writes ``guide.html`` (the shared educational Guide, linked from every page's
    nav). Returns the path to the written ``index.html``. Signature is intentionally unchanged
    (manifest is auto-located from the source tree).
    """
    results_path = Path(results_dir)
    out_path = Path(out_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"results directory does not exist: {results_path}")

    rows = load_results(results_path)
    grouped = group_by_task(rows)
    declared = load_manifest()
    nav_ids = _all_task_ids(grouped, declared)

    out_path.mkdir(parents=True, exist_ok=True)
    # Full leaderboard pages for tasks that have results.
    for task_name, task_rows in grouped.items():
        page = render_task_page(task_name, task_rows, nav_task_ids=nav_ids, declared=declared)
        (out_path / f"{task_name}.html").write_text(page, encoding="utf-8")
    # Minimal WIP pages for declared tasks that have NO results yet.
    for task_id, entry in declared.items():
        if grouped.get(task_id):
            continue
        page = render_wip_page(entry, nav_task_ids=nav_ids, declared=declared)
        (out_path / f"{task_id}.html").write_text(page, encoding="utf-8")

    # Shared educational Guide page (linked from the nav on every page).
    (out_path / f"{_GUIDE_SLUG}.html").write_text(render_guide(), encoding="utf-8")

    index_path = out_path / "index.html"
    index_path.write_text(render_index(grouped, declared), encoding="utf-8")
    return index_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments for the generator."""
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Render the RF-Benchmark-Hub static leaderboard from result JSONs.",
    )
    parser.add_argument(
        "--results",
        default="leaderboard/results",
        help="directory scanned recursively for result *.json (default: leaderboard/results)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory for the generated static site",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build the site and report where it landed."""
    args = _parse_args(argv)
    index_path = build_site(args.results, args.out)
    print(f"[generate] wrote leaderboard to {index_path.parent} (index: {index_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
