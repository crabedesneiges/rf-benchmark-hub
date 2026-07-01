"""WP-50 -- static leaderboard site generator.

Reads every ``leaderboard/results/**/*.json`` row (each MUST validate against
``schemas/result.schema.json``), groups the valid rows by task, and renders a plain
**static HTML** site into an ``--out`` directory. No Jinja, no third-party templating:
a tiny stdlib ``string.Template`` layer plus manual escaping is all we use. The only
non-stdlib dependency is ``jsonschema`` (a hard dep of the harness), imported LAZILY so
``import`` of this module stays dependency-free.

Board contract (mirrors the frozen policy in docs/IMPLEMENTATION_PLAN.md / D5):

* One HTML page per task, plus an ``index.html`` linking to them.
* Each task table is sorted by that task's PRIMARY metric (``metrics.primary``),
  descending -- higher is better for every v1 primary (accuracy/rank1/auroc/mAP/pd).
* ``regime`` is an explicit column and the table is *grouped by regime*: a metric
  (comparison) column therefore only ever compares rows within a single regime, so the
  board NEVER mixes two regimes in one comparison column (D5 / risk guard-rail).
* A badge distinguishes ``verification.status`` ``verified`` from ``self_reported``.

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
from string import Template
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jsonschema import Draft202012Validator

# --------------------------------------------------------------------------------------------------
# Board presentation constants
# --------------------------------------------------------------------------------------------------

#: Human-readable task titles for the board (falls back to the raw task id).
TASK_TITLES: dict[str, str] = {
    "amc": "AMC -- Automatic Modulation Classification",
    "sei": "SEI -- RF Fingerprinting",
    "wideband_detection": "Wideband Detection",
    "spectrum_sensing": "Spectrum Sensing",
}

#: Stable display order of the four locked regimes (D5).
REGIME_ORDER: tuple[str, ...] = (
    "from_scratch",
    "full_finetune",
    "linear_probe",
    "few_shot",
)

#: Verification badge text/CSS-class per status.
_BADGE: dict[str, tuple[str, str]] = {
    "verified": ("verified", "badge-verified"),
    "self_reported": ("self-reported", "badge-self"),
}


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
# Row helpers
# --------------------------------------------------------------------------------------------------
def _primary_value(row: dict[str, Any]) -> float:
    """Return the row's primary-metric scalar (``metrics.values[metrics.primary]``).

    Guaranteed present by the schema (``primary`` must be a key of ``values``); we read it
    defensively and coerce to ``float`` for a total ordering.
    """
    metrics = row["metrics"]
    primary_key = str(metrics["primary"])
    return float(metrics["values"][primary_key])


def _regime_name(row: dict[str, Any]) -> str:
    """Return the declared regime name (D5: always present, never inferred)."""
    return str(row["regime"]["name"])


def _regime_label(row: dict[str, Any]) -> str:
    """Human label for a regime, expanding ``few_shot`` to include its ``k_shot``."""
    name = _regime_name(row)
    if name == "few_shot":
        k = row["regime"].get("k_shot")
        if k is not None:
            return f"few_shot(k={k})"
    return name


def _status(row: dict[str, Any]) -> str:
    """Return the verification status (``verified`` | ``self_reported``)."""
    return str(row["verification"]["status"])


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


def _regime_sort_key(regime: str) -> tuple[int, str]:
    """Order regimes by the locked D5 order, unknown regimes last (alphabetical)."""
    if regime in REGIME_ORDER:
        return (REGIME_ORDER.index(regime), regime)
    return (len(REGIME_ORDER), regime)


# --------------------------------------------------------------------------------------------------
# HTML rendering (stdlib string.Template + manual escaping; NO jinja)
# --------------------------------------------------------------------------------------------------
def _esc(value: object) -> str:
    """HTML-escape a value for safe insertion into markup."""
    return html.escape(str(value), quote=True)


def _fmt_metric(value: float) -> str:
    """Format a metric scalar for the board (4 decimals, trims to a stable string)."""
    return f"{value:.4f}"


_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>$css</style>
</head>
<body>
<header><h1>$heading</h1></header>
<main>
$body
</main>
<footer><p>Generated by leaderboard/site/generate.py -- rows validated against \
result.schema.json.</p></footer>
</body>
</html>
""")

_CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#ddd; --head:#f4f4f6; --accent:#0b5fff; }
* { box-sizing:border-box; }
body { font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg);
       margin:0; padding:0 1.5rem 3rem; line-height:1.45; }
header h1 { font-size:1.5rem; margin:1.5rem 0 0.5rem; }
main { max-width:1000px; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
table { border-collapse:collapse; width:100%; margin:0.5rem 0 2rem; font-size:0.92rem; }
caption { text-align:left; font-weight:600; margin-bottom:0.35rem; }
th, td { border:1px solid var(--line); padding:0.4rem 0.6rem; text-align:left; }
thead th { background:var(--head); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.primary td.num.primary { font-weight:700; }
.badge { display:inline-block; padding:0.05rem 0.5rem; border-radius:999px; font-size:0.78rem;
         font-weight:600; white-space:nowrap; }
.badge-verified { background:#e6f6ea; color:#137333; border:1px solid #9fd8ae; }
.badge-self { background:#fff4e5; color:#a15c00; border:1px solid #f0c891; }
.regime-group { margin:0 0 1.5rem; }
.regime-group h3 { font-size:1.05rem; margin:1rem 0 0.25rem; }
.note { color:var(--muted); font-size:0.85rem; }
ul.tasks { list-style:none; padding:0; }
ul.tasks li { margin:0.4rem 0; font-size:1.05rem; }
footer p { color:var(--muted); font-size:0.8rem; margin-top:2rem; }
"""


def _render_badge(status: str) -> str:
    """Render the verification badge span for a status."""
    text, css_class = _BADGE.get(status, (status, "badge-self"))
    return f'<span class="badge {css_class}">{_esc(text)}</span>'


def _render_row(row: dict[str, Any], primary_key: str) -> str:
    """Render one ``<tr>`` for a task table (rank filled in by the caller's numbering).

    Columns: model, params, regime, primary metric (the only cross-row comparison column,
    kept within a single regime group), other scalar metrics, verification badge.
    """
    model = row["model"]
    name = _esc(model["name"])
    url = model.get("url")
    if isinstance(url, str) and url:
        name = f'<a href="{_esc(url)}">{name}</a>'
    n_params = model.get("n_params")
    params_cell = f"{n_params:,}" if isinstance(n_params, int) else "&ndash;"

    values = row["metrics"]["values"]
    primary = _fmt_metric(float(values[primary_key]))
    other_keys = sorted(k for k in values if k != primary_key)
    other_cells = "".join(
        f'<td class="num">{_fmt_metric(float(values[k]))}</td>' for k in other_keys
    )

    badge = _render_badge(_status(row))
    return (
        "<tr>"
        f"<td>{name}</td>"
        f'<td class="num">{params_cell}</td>'
        f"<td>{_esc(_regime_label(row))}</td>"
        f'<td class="num primary">{primary}</td>'
        f"{other_cells}"
        f"<td>{badge}</td>"
        "</tr>"
    )


def _render_regime_table(
    regime: str,
    rows: list[dict[str, Any]],
    primary_key: str,
) -> str:
    """Render a per-regime sub-table (header + sorted rows).

    Grouping by regime is what guarantees D5: the primary/other metric columns only ever
    compare rows *inside one regime*, so a comparison column never mixes regimes.
    """
    ordered = _sort_rows(rows)
    # Union of non-primary scalar keys across this regime's rows, for stable columns.
    other_keys: list[str] = sorted(
        {k for row in ordered for k in row["metrics"]["values"] if k != primary_key}
    )
    head_cells = "".join(f"<th>{_esc(k)}</th>" for k in other_keys)
    header = (
        "<thead><tr>"
        "<th>Model</th><th>Params</th><th>Regime</th>"
        f"<th>{_esc(primary_key)} (primary)</th>{head_cells}<th>Status</th>"
        "</tr></thead>"
    )
    body_rows = "\n".join(_render_row(row, primary_key) for row in ordered)
    label = _regime_label(ordered[0]) if regime == "few_shot" else regime
    return (
        '<section class="regime-group">'
        f"<h3>Regime: {_esc(label)}</h3>"
        f'<table data-regime="{_esc(regime)}">'
        f"<caption>Ranked by <code>{_esc(primary_key)}</code> (descending).</caption>"
        f"{header}<tbody>\n{body_rows}\n</tbody></table>"
        "</section>"
    )


def render_task_page(task_name: str, rows: list[dict[str, Any]]) -> str:
    """Render the full HTML page for one task.

    The primary metric key is taken from the rows (all rows of a task share it by
    protocol). Rows are grouped by regime (D5) and each group is a table sorted by the
    primary metric, descending. A verification badge is shown per row.
    """
    if not rows:
        raise ValueError(f"render_task_page called with no rows for task '{task_name}'")

    primary_key = str(rows[0]["metrics"]["primary"])
    title = TASK_TITLES.get(task_name, task_name)

    by_regime: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(_regime_name(row), []).append(row)

    sections: list[str] = [
        '<p class="note">Each regime is ranked separately -- comparison columns never '
        "mix regimes (D5). Badges mark maintainer-<strong>verified</strong> rows vs "
        "self-reported ones.</p>",
        '<p class="note"><a href="index.html">&larr; all tasks</a></p>',
    ]
    for regime in sorted(by_regime, key=_regime_sort_key):
        sections.append(_render_regime_table(regime, by_regime[regime], primary_key))

    return _PAGE_TEMPLATE.substitute(
        title=f"{title} -- RF-Benchmark-Hub",
        heading=_esc(title),
        css=_CSS,
        body="\n".join(sections),
    )


def render_index(grouped: dict[str, list[dict[str, Any]]]) -> str:
    """Render the ``index.html`` linking to every task page with a row count."""
    items: list[str] = []
    for task_name in sorted(grouped, key=lambda t: TASK_TITLES.get(t, t)):
        n = len(grouped[task_name])
        title = TASK_TITLES.get(task_name, task_name)
        plural = "row" if n == 1 else "rows"
        items.append(
            f'<li><a href="{_esc(task_name)}.html">{_esc(title)}</a> '
            f'<span class="note">({n} {plural})</span></li>'
        )
    body = (
        '<p class="note">Reproducible benchmarks for terrestrial RF ML tasks. '
        "Each task ranks submissions by its primary metric; regimes are never mixed in a "
        "comparison column.</p>"
        f'<ul class="tasks">\n{chr(10).join(items)}\n</ul>'
        if items
        else '<p class="note">No valid results yet.</p>'
    )
    return _PAGE_TEMPLATE.substitute(
        title="RF-Benchmark-Hub Leaderboard",
        heading="RF-Benchmark-Hub Leaderboard",
        css=_CSS,
        body=body,
    )


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
def build_site(results_dir: str | Path, out_dir: str | Path) -> Path:
    """Build the static leaderboard site from ``results_dir`` into ``out_dir``.

    Reads and schema-validates every ``results_dir/**/*.json`` (invalid rows skipped with
    a warning), groups the valid rows by task, and writes ``index.html`` plus one
    ``<task>.html`` page per task into ``out_dir`` (created if absent). Returns the path to
    the written ``index.html``.
    """
    results_path = Path(results_dir)
    out_path = Path(out_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"results directory does not exist: {results_path}")

    rows = load_results(results_path)
    grouped = group_by_task(rows)

    out_path.mkdir(parents=True, exist_ok=True)
    for task_name, task_rows in grouped.items():
        page = render_task_page(task_name, task_rows)
        (out_path / f"{task_name}.html").write_text(page, encoding="utf-8")

    index_path = out_path / "index.html"
    index_path.write_text(render_index(grouped), encoding="utf-8")
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
