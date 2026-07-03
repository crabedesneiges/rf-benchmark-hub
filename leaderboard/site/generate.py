"""WP-50 -- static leaderboard site generator (design + genericity overhaul).

Reads every ``leaderboard/results/**/*.json`` row (each MUST validate against
``schemas/result.schema.json``), groups the valid rows by task, and renders a polished,
**fully static** HTML site into an ``--out`` directory. The generator is stdlib-only (no
Jinja, no Chart.js, no CDN, no runtime JS): pages are assembled with manual string
building + ``html.escape``, and every chart is an **inline SVG** whose polylines are
computed here in Python. The one non-stdlib dependency is ``jsonschema`` (a hard dep of
the harness), imported LAZILY so importing this module stays dependency-free.

The renderer is **data-driven, never task-specific**:

* The set of tasks shown is DECLARED in a committed manifest (``leaderboard/tasks.json``):
  every declared task appears on the board. Tasks that have valid result rows render as
  full leaderboards (tables + plots); declared tasks that have NO results (or no baseline
  yet) render a WIP card on the index and a minimal "Work in progress" task page -- never a
  broken empty table. Any task that has results but is missing from the manifest is still
  rendered (the manifest is additive, not a filter). The manifest supplies each task's
  title, ``status`` (implemented | wip | planned), ``priority`` (P1/P2/P3) and a short blurb.
* A fixed display order is applied to the known tasks (amc, sei, wideband_detection,
  spectrum_sensing, ...) then any others land alphabetically. No task is hardcoded into the
  rendering path.
* Per task, the SCALAR metrics are discovered from every key of ``metrics.values`` and the
  CURVE metrics from every key of ``metrics.curves``. SEI (rank1_accuracy/auroc/eer),
  detection (mAP/mAR/IoU), sensing (pd@pfa=0.1/latency + ROC) etc. therefore render
  automatically the moment their result JSONs appear -- nothing about their metric names
  is baked in.

Protocol invariants (docs/EVALUATION_PROTOCOL.md / D5), enforced structurally:

* One HTML page per task, plus an ``index.html`` landing page.
* Rows are partitioned into ``(regime, track)`` groups. Each group renders exactly one
  leaderboard TABLE (a column per scalar metric, primary first + emphasised) and one line
  PLOT per curve metric (overlaying every model in that group). A table or plot therefore
  NEVER compares across two regimes -- nor two tracks.
* ``track`` is read from ``eval.conditions.track`` or ``split.track`` (free-form/optional);
  rows without one land in a default ``all`` bucket so single-track tasks still render.
* A badge distinguishes ``verification.status`` ``verified`` from ``self_reported``; a chip
  distinguishes the model ``family`` ``baseline`` from ``foundation``.

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

#: Verification badge text/CSS-class per status.
_BADGE: dict[str, tuple[str, str]] = {
    "verified": ("verified", "badge-verified"),
    "self_reported": ("self reported", "badge-self"),
}

#: Family chip text/CSS-class per family.
_FAMILY_CHIP: dict[str, tuple[str, str]] = {
    "baseline": ("baseline", "chip-baseline"),
    "foundation": ("foundation", "chip-foundation"),
}

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
class DeclaredTask(NamedTuple):
    """One declared downstream task from the manifest (``leaderboard/tasks.json``).

    ``status`` is the declared build state: ``implemented`` (has or will have results),
    ``wip`` (task/track defined, baseline pending) or ``planned`` (blocked / out of the
    current scope). A task renders its full leaderboard only when it actually has result
    rows; otherwise -- whatever its declared status -- it renders a WIP page/card so the
    board never shows a broken empty table.

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
        priority = entry.get("priority")
        declared[task_id] = DeclaredTask(
            id=task_id,
            title=str(entry.get("title") or TASK_TITLES.get(task_id, task_id)),
            status=str(status),
            priority=str(priority) if isinstance(priority, str) and priority else None,
            blurb=str(entry.get("blurb") or ""),
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
    """
    return sorted(
        rows,
        key=lambda r: (
            -_primary_value(r),
            0 if _status(r) == "verified" else 1,
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


def _render_row(
    rank: int,
    row: dict[str, Any],
    scalar_keys: list[str],
    primary_key: str,
    primary_max: float,
) -> str:
    """Render one ``<tr>``: rank, model (+family chip, +params), each scalar, status.

    A cell is rendered for EVERY discovered scalar metric so no metric is left out; a metric
    absent from this particular row shows an en-dash. The primary column carries the score
    bar and is visually emphasised.
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
            metric_cells.append(
                f'<td class="num primary"><span class="metric-val">{formatted}</span>{bar}</td>'
            )
        else:
            metric_cells.append(f'<td class="num">{formatted}</td>')

    badge = _render_badge(_status(row))
    return (
        "<tr>"
        f'<td class="rank num">{rank}</td>'
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
        _render_row(i, row, scalar_keys, primary_key, primary_max)
        for i, row in enumerate(ordered, start=1)
    )
    return (
        f'<table data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f"{header}<tbody>\n{body_rows}\n</tbody></table>"
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


def render_wip_page(entry: DeclaredTask, nav_task_ids: list[str]) -> str:
    """Render a minimal "work in progress" page for a declared task with no results.

    Shows the task title, a clear WIP badge/state and the manifest blurb -- but NO
    leaderboard table and NO plots, so a task without a baseline never renders a broken
    empty table. Kept deliberately spare; it is replaced by the full leaderboard the moment
    a valid ``result.json`` for the task lands.
    """
    title = entry.title
    badge = _render_status_badge(entry)
    blurb = f'<p class="wip-blurb">{_esc(entry.blurb)}</p>' if entry.blurb else ""
    body = (
        '<section class="task">'
        f'<h2 class="task-title">{_esc(title)}</h2>'
        f'<p class="task-meta">{badge}</p>'
        '<div class="wip-card">'
        '<p class="wip-state"><strong>Work in progress</strong> &mdash; no baseline yet.</p>'
        '<p class="note">This task is declared in the benchmark but has no submitted '
        "results on the board yet. A leaderboard (tables + plots) will appear here "
        "automatically once a valid result is added.</p>"
        f"{blurb}"
        "</div>"
        "</section>"
    )
    page_title = f"{title} — RF-Benchmark-Hub"
    return _page(page_title, body, task_nav=_task_nav(nav_task_ids, entry.id))


def render_task_page(
    task_name: str,
    rows: list[dict[str, Any]],
    nav_task_ids: list[str] | None = None,
    declared: dict[str, DeclaredTask] | None = None,
) -> str:
    """Render the full HTML page for one task.

    Rows are partitioned into ``(regime, track)`` groups; each group renders one table
    (a column per discovered scalar metric) and one plot per discovered curve metric. A
    group is a single (regime, track), so no table or plot ever mixes regimes nor tracks.
    Groups are ordered by regime (locked D5 order) then track (``all`` first).
    """
    if not rows:
        raise ValueError(f"render_task_page called with no rows for task '{task_name}'")

    declared = declared or {}
    title = _task_title(task_name, declared)
    dataset_line = _task_meta_line(task_name, rows)

    # (regime, track) -> rows, preserving input order within each leaf group.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((_regime_name(row), _track_name(row)), []).append(row)

    ordered_keys = sorted(groups, key=lambda rt: (_regime_sort_key(rt[0]), _track_sort_key(rt[1])))
    sections = [
        _render_group(regime, track, group_rows, _primary_key(group_rows[0]))
        for (regime, track) in ordered_keys
        for group_rows in (groups[(regime, track)],)
    ]

    body = (
        '<section class="task">'
        f'<h2 class="task-title">{_esc(title)}</h2>'
        f'<p class="task-meta">{dataset_line}</p>'
        f'<p class="note">Each (regime, track) is ranked separately &mdash; a table or plot '
        "never mixes two regimes nor two tracks (protocol invariant). Badges mark "
        "maintainer-verified rows vs self-reported ones.</p>"
        f"{''.join(sections)}"
        "</section>"
    )
    page_title = f"{title} — RF-Benchmark-Hub"
    nav_ids = nav_task_ids if nav_task_ids is not None else [task_name]
    return _page(page_title, body, task_nav=_task_nav(nav_ids, task_name))


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
    return (
        f'<a class="task-card" href="{_esc(task_name)}.html">'
        f'<span class="card-title">{_esc(title)}</span>{badge}'
        f'<span class="card-sub">{_esc(f"{n_rows} results · {n_models} models")}</span>'
        f'<span class="card-best">Best: <strong>{_esc(best_model)}</strong> '
        f"&middot; {_esc(primary)} = <strong>{_esc(best_score)}</strong></span>"
        "</a>"
    )


def _render_wip_card(entry: DeclaredTask) -> str:
    """Render an index card for a declared task WITHOUT results (WIP state, no scores)."""
    badge = _render_status_badge(entry)
    blurb = f'<span class="card-blurb">{_esc(entry.blurb)}</span>' if entry.blurb else ""
    return (
        f'<a class="task-card task-card-wip" href="{_esc(entry.id)}.html">'
        f'<span class="card-title">{_esc(entry.title)}</span>{badge}'
        '<span class="card-sub">no baseline yet</span>'
        f"{blurb}"
        "</a>"
    )


def render_index(
    grouped: dict[str, list[dict[str, Any]]],
    declared: dict[str, DeclaredTask] | None = None,
) -> str:
    """Render the ``index.html`` landing page: one card per DECLARED task.

    Every declared task appears: tasks that have results get a best-score card linking to
    their full leaderboard; declared tasks without results get a WIP card (no scores) linking
    to a minimal WIP page. Any task that has results but is absent from the manifest is still
    shown (the manifest is additive, not a filter), so nothing silently drops off the board.
    """
    declared = declared or {}
    ordered_tasks = _all_task_ids(grouped, declared)
    cards: list[str] = []
    for task_name in ordered_tasks:
        rows = grouped.get(task_name)
        if rows:
            cards.append(_render_result_card(task_name, rows, declared))
        else:
            entry = declared.get(task_name)
            if entry is None:  # pragma: no cover - defensive; ids came from declared/grouped
                continue
            cards.append(_render_wip_card(entry))

    if cards:
        body = (
            '<section class="task">'
            '<p class="note">Reproducible benchmarks for terrestrial RF machine-learning '
            "tasks, comparing specialised baselines against fine-tuned foundation models. "
            "Each task ranks submissions by its primary metric; regimes and tracks are never "
            "mixed in a comparison. Tasks marked <em>work in progress</em> are declared but "
            "have no baseline on the board yet.</p>"
            f'<div class="card-grid">{"".join(cards)}</div>'
            "</section>"
        )
    else:
        body = (
            '<section class="task">'
            '<p class="note">No tasks declared or results yet.</p>'
            "</section>"
        )
    return _page("RF-Benchmark-Hub Leaderboard", body, task_nav=_task_nav(ordered_tasks, None))


def _task_nav(task_names: list[str], current: str | None) -> str:
    """Render the header task navigation chips (linking each task to its page)."""
    chips: list[str] = ['<a class="nav-chip" href="index.html">Home</a>']
    for task in sorted(task_names, key=_task_sort_key):
        title = TASK_TITLES.get(task, task)
        active = " nav-chip-active" if task == current else ""
        chips.append(f'<a class="nav-chip{active}" href="{_esc(task)}.html">{_esc(title)}</a>')
    return f'<nav class="nav">{"".join(chips)}</nav>'


def _page(title: str, body: str, task_nav: str) -> str:
    """Assemble a complete standalone HTML page (header + nav + body + footer)."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
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
        "&mdash; every row validated against result.schema.json. No runtime dependencies; "
        "charts are inline SVG.</p></footer>\n"
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
  --accent: #0b5fff;
  --accent-soft: #e8f0ff;
  --head: #f4f5f7;
  --badge-verified-bg: #e6f6ea; --badge-verified-fg: #137333; --badge-verified-bd: #9fd8ae;
  --badge-self-bg: #fff4e5; --badge-self-fg: #a15c00; --badge-self-bd: #f0c891;
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
    --accent: #5b8dff;
    --accent-soft: #1a2740;
    --head: #1b2028;
    --badge-verified-bg: #12281a; --badge-verified-fg: #57cc7f; --badge-verified-bd: #2c5b3b;
    --badge-self-bg: #2e2410; --badge-self-fg: #e0a94b; --badge-self-bd: #5c4a1f;
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
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--fg); background: var(--bg); margin: 0;
  line-height: 1.5; -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.site-header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.5rem;
  padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); background: var(--surface);
}
.brand { display: flex; align-items: center; gap: 0.6rem; }
.logo { width: 30px; height: 30px; flex: none; }
.brand-text { display: flex; flex-direction: column; }
.brand-name { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }
.brand-tag { color: var(--muted); font-size: 0.8rem; }
.nav { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-left: auto; }
.nav-chip {
  font-size: 0.82rem; padding: 0.25rem 0.7rem; border-radius: 999px;
  border: 1px solid var(--line); color: var(--fg); background: var(--surface-2);
}
.nav-chip:hover { border-color: var(--line-strong); text-decoration: none; }
.nav-chip-active {
  background: var(--accent-soft); border-color: var(--accent); color: var(--accent);
}

main { max-width: 1080px; margin: 0 auto; padding: 1.5rem 1.5rem 4rem; }
.task-title { font-size: 1.4rem; margin: 0.5rem 0 0.25rem; letter-spacing: -0.01em; }
.task-meta {
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  color: var(--muted); font-size: 0.8rem; margin: 0 0 0.75rem;
}
.note { color: var(--muted); font-size: 0.85rem; margin: 0.25rem 0 1.25rem; }

.group {
  border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
  padding: 1rem 1.1rem 1.25rem; margin: 0 0 1.5rem;
}
.group-title {
  font-size: 1rem; margin: 0 0 0.75rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--line); font-weight: 600;
}

table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
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
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; margin-top: 0.5rem; }
.legend-item {
  display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; color: var(--muted);
}
.legend-swatch { width: 24px; height: 10px; flex: none; }

.card-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem; margin-top: 0.5rem;
}
.task-card {
  display: flex; flex-direction: column; gap: 0.35rem; color: var(--fg);
  border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
  padding: 1rem 1.1rem;
}
.task-card:hover { border-color: var(--accent); text-decoration: none; }
.card-title { font-weight: 700; font-size: 1.05rem; }
.card-sub { color: var(--muted); font-size: 0.8rem;
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }
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
  border: 1px dashed var(--status-wip-bd); border-radius: 12px; background: var(--surface);
  padding: 1.1rem 1.25rem; margin: 0 0 1.5rem;
}
.wip-state { font-size: 1rem; margin: 0 0 0.5rem; }
.wip-blurb { color: var(--fg); font-size: 0.9rem; margin: 0.75rem 0 0; }

.site-footer { border-top: 1px solid var(--line); background: var(--surface); }
.site-footer p { max-width: 1080px; margin: 0 auto; padding: 1rem 1.5rem;
  color: var(--muted); font-size: 0.78rem; }
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
    table). Returns the path to the written ``index.html``. Signature is intentionally
    unchanged (manifest is auto-located from the source tree).
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
        page = render_wip_page(entry, nav_task_ids=nav_ids)
        (out_path / f"{task_id}.html").write_text(page, encoding="utf-8")

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
