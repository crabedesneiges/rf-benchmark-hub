"""WP-50 tests -- static leaderboard site generator (generic, task-agnostic rewrite).

Exercises ``leaderboard.site.generate.build_site`` end to end on synthetic, in-tree
result fixtures written to ``tmp_path`` (no network, no heavy deps -- only ``jsonschema``,
a hard dep of the harness, is used, and only transitively via the generator). The tests
assert the acceptance criteria for the generic generator:

* an ``index.html`` and one ``<task>.html`` page are produced (one per task with >=1 valid
  row); a task whose every row is invalid produces NO page,
* GENERIC per-metric rendering: every scalar in ``metrics.values`` gets its OWN table
  column (not just the primary one),
* every curve in ``metrics.curves`` yields an inline ``<svg class="plot">`` on the page,
* a task carrying only scalar metrics (no curves) still renders its table (no plot, no
  crash),
* rows are ordered by the task's PRIMARY metric (descending) within a (regime, track)
  group,
* the no-mixing invariant (D5): each ``<table>`` carries exactly one ``data-regime`` /
  ``data-track`` pair; two rows sharing a regime but declaring different tracks land in
  SEPARATE tables, and two different regimes never share a table,
* both verification states surface as distinguishable badges,
* invalid result JSONs are skipped (not rendered) instead of crashing the build,
* the EDUCATIONAL header (description + dataset card + metric definitions) is rendered on a
  task page (and on a WIP page) from the manifest's optional fields,
* a ``guide.html`` page is written, carries the "What is I/Q?" section + the metrics glossary
  (with up/down arrows), and is linked from every page's nav.

The generator module lives at ``leaderboard/site/generate.py``; it is loaded by file
path so the test does not depend on ``leaderboard`` being an importable package.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_PATH = REPO_ROOT / "leaderboard" / "site" / "generate.py"


def _load_generate() -> ModuleType:
    """Import ``leaderboard/site/generate.py`` by path (no package install needed).

    The loaded module is registered in ``sys.modules`` under its spec name -- standard
    practice for a path-based import so the module can resolve itself by name if needed.
    """
    spec = importlib.util.spec_from_file_location("lb_generate", GENERATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_generate()


# --------------------------------------------------------------------------------------------------
# Fixtures: minimal, schema-valid result rows built in-memory (pure stdlib).
# --------------------------------------------------------------------------------------------------
def _amc_row(
    result_id: str,
    model_name: str,
    regime: str,
    accuracy: float,
    status: str,
    *,
    family: str = "baseline",
) -> dict[str, Any]:
    """Build a schema-valid AMC result row (primary metric = accuracy_overall)."""
    verification: dict[str, Any] = {"status": status}
    if status == "verified":
        verification.update(
            verified_by="tester",
            verified_date="2026-06-30",
            verified_hardware="4x NVIDIA GB200",
            method="eval_only",
        )
    return {
        "schema_version": "1.0.0",
        "result_id": result_id,
        "task": {"name": "amc", "version": "v1"},
        "model": {"name": model_name, "family": family, "n_params": 100000},
        "regime": {"name": regime},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "amc-radioml2016-strat-snr-8010-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": "sha256:" + "0" * 64,
        },
        "metrics": {
            "primary": "accuracy_overall",
            "values": {"accuracy_overall": accuracy, "macro_f1": accuracy - 0.02},
        },
        "verification": verification,
    }


def _sei_row(
    result_id: str,
    model_name: str,
    track: str,
    rank1: float,
    *,
    values: dict[str, float] | None = None,
    primary: str = "rank1_accuracy",
    curves: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a schema-valid SEI result row.

    ``values`` overrides the scalar metric map (defaults to a single ``rank1_accuracy``);
    ``curves`` optionally attaches ``metrics.curves`` for the inline-SVG plot tests.
    """
    metric_values = values if values is not None else {"rank1_accuracy": rank1}
    metrics: dict[str, Any] = {"primary": primary, "values": metric_values}
    if curves is not None:
        metrics["curves"] = curves
    return {
        "schema_version": "1.0.0",
        "result_id": result_id,
        "task": {"name": "sei", "version": "v1"},
        "model": {"name": model_name, "family": "baseline", "n_params": 200000},
        "regime": {"name": "from_scratch"},
        "dataset": {"name": "wisig"},
        "split": {
            "canonical_split_id": f"sei-wisig-{track.replace('_', '-')}-seed42-v1",
            "name": "test",
            "track": track,
            "seed": 42,
            "checksum": "sha256:" + "1" * 64,
        },
        "metrics": metrics,
        "verification": {"status": "self_reported"},
    }


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _make_results_tree(root: Path) -> None:
    """Populate a results tree with a spread of tasks/regimes/statuses."""
    # AMC linear_probe: two rows so within-regime ordering is testable.
    _write(
        root / "amc" / "a.json", _amc_row("row-a", "iqfm", "linear_probe", 0.71, "self_reported")
    )
    _write(root / "amc" / "b.json", _amc_row("row-b", "mcldnn", "linear_probe", 0.52, "verified"))
    # AMC full_finetune: a different regime -> a separate comparison group.
    _write(root / "amc" / "c.json", _amc_row("row-c", "mcldnn", "full_finetune", 0.61, "verified"))
    # SEI: separate task, separate page.
    _write(root / "sei" / "d.json", _sei_row("row-d", "wisig-cnn", "closed_set", 0.94))


# --------------------------------------------------------------------------------------------------
# Minimal HTML parsing helpers (stdlib html.parser -- no bs4/lxml).
# --------------------------------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Collect, per ``<table data-regime=R data-track=T>``, its ordered body model names.

    The generic table has NO per-row regime cell: the no-mixing invariant is expressed
    entirely by each ``<table>``'s ``data-regime`` / ``data-track`` attributes (a table is
    exactly one (regime, track) group). This parser therefore keys tables by that
    ``(track, regime)`` pair and records, in document order, the ``<span class="model-name">``
    of every body row -- enough to assert (a) two regimes never share a table, (b) two tracks
    never share a table, and (c) the within-table row order (primary metric, descending).
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_key: tuple[str, str] | None = None
        self._in_body = False
        self._capture_model = False
        self._buffer: list[str] = []
        # (track, regime) -> list of body-row model names, in document order.
        self.rows_by_table: dict[tuple[str, str], list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "table":
            regime = attr.get("data-regime")
            if regime is not None:
                track = attr.get("data-track") or "all"
                self._current_key = (track, regime)
                self.rows_by_table.setdefault(self._current_key, [])
        elif tag == "tbody":
            self._in_body = True
        elif (
            tag == "span"
            and self._in_body
            and self._current_key is not None
            and attr.get("class") == "model-name"
        ):
            self._capture_model = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._capture_model:
            name = "".join(self._buffer).strip()
            assert self._current_key is not None
            self.rows_by_table[self._current_key].append(name)
            self._capture_model = False
        elif tag == "tbody":
            self._in_body = False
        elif tag == "table":
            self._current_key = None

    def handle_data(self, data: str) -> None:
        if self._capture_model:
            self._buffer.append(data)


def _tables_by_track_regime(html_text: str) -> dict[tuple[str, str], list[str]]:
    """Return rendered tables keyed by their ``(track, regime)`` pair -> ordered models."""
    parser = _TableParser()
    parser.feed(html_text)
    return parser.rows_by_table


def _regime_tables(html_text: str) -> dict[str, list[str]]:
    """Return tables keyed by regime, asserting a single track per regime.

    A convenience wrapper for the single-track AMC fixtures where each regime appears in
    exactly one track; asserts that assumption so multi-track pages use the richer
    ``_tables_by_track_regime`` view instead.
    """
    by_regime: dict[str, list[str]] = {}
    for (_track, regime), models in _tables_by_track_regime(html_text).items():
        assert regime not in by_regime, f"regime {regime} spans multiple tracks; use pair view"
        by_regime[regime] = models
    return by_regime


def _table_data_regimes(html_text: str) -> list[str]:
    """Return the ``data-regime`` of every ``<table ...>`` tag, in document order.

    Each ``<table>`` must declare exactly ONE ``data-regime`` -- greping the raw opening
    tags asserts that no table blends two regimes at the tag level.
    """
    regimes: list[str] = []
    for tag in re.findall(r"<table\b[^>]*>", html_text):
        found = re.findall(r'data-regime="([^"]*)"', tag)
        assert len(found) == 1, f"table tag has {len(found)} data-regime attrs: {tag}"
        regimes.append(found[0])
    return regimes


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------
def test_build_site_returns_index_and_pages(tmp_path: Path) -> None:
    """build_site writes index.html plus one page per task and returns the index path."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)

    index_path = generate.build_site(results, out)

    assert index_path == out / "index.html"
    assert index_path.is_file()
    assert (out / "amc.html").is_file()
    assert (out / "sei.html").is_file()

    index_html = index_path.read_text(encoding="utf-8")
    assert "amc.html" in index_html
    assert "sei.html" in index_html


def test_build_site_requires_existing_results_dir(tmp_path: Path) -> None:
    """A missing results directory raises FileNotFoundError (never a silent empty build)."""
    missing = tmp_path / "does-not-exist"
    try:
        generate.build_site(missing, tmp_path / "site")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("build_site should raise FileNotFoundError for a missing results dir")


def test_expected_model_rows_present(tmp_path: Path) -> None:
    """Every submitted model appears on its task page."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    for model in ("iqfm", "mcldnn"):
        assert model in amc_html
    sei_html = (out / "sei.html").read_text(encoding="utf-8")
    assert "wisig-cnn" in sei_html


def test_scalar_metrics_get_one_column_each(tmp_path: Path) -> None:
    """GENERIC rendering: every scalar in metrics.values gets its OWN table column.

    A synthetic SEI row carries three scalars (rank1_accuracy, auroc, eer); the rendered
    table must expose a column HEADER for each -- not only the primary -- and the primary
    column is the one flagged with the ``primary`` col-note (exactly once).
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "sei" / "multi.json",
        _sei_row(
            "row-multi",
            "sei-net",
            "open_set",
            0.0,
            primary="rank1_accuracy",
            values={"rank1_accuracy": 0.81, "auroc": 0.90, "eer": 0.07},
        ),
    )

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    # One header cell per distinct scalar metric (primary uses `<th class="num primary"`,
    # the rest `<th class="num"`; both share the `<th class="num` prefix).
    assert sei_html.count('<th class="num') == 3
    # Each metric name appears as a header cell.
    for metric in ("rank1_accuracy", "auroc", "eer"):
        assert f">{metric}<" in sei_html
    # The primary column is flagged exactly once.
    assert sei_html.count('<span class="col-note">primary</span>') == 1
    # The primary header carries the metric name + the primary marker together.
    assert '<th class="num primary">rank1_accuracy<span class="col-note">primary</span>' in sei_html


def test_missing_metric_renders_en_dash(tmp_path: Path) -> None:
    """A metric present on one row but absent on another shows an en-dash for the gap.

    The column set is the UNION of every row's metric keys; a row lacking a discovered
    metric must still render a cell (en-dash) so columns stay aligned.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    # Same task/regime/track; one row has auroc, the other does not.
    _write(
        results / "sei" / "full.json",
        _sei_row(
            "row-full",
            "with-auroc",
            "closed_set",
            0.0,
            values={"rank1_accuracy": 0.90, "auroc": 0.88},
        ),
    )
    _write(
        results / "sei" / "partial.json",
        _sei_row("row-partial", "no-auroc", "closed_set", 0.70),
    )

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    # auroc is a discovered column; the row lacking it renders an en-dash cell.
    assert ">auroc<" in sei_html
    assert '<td class="num">&ndash;</td>' in sei_html


def test_curve_metric_yields_inline_svg_plot(tmp_path: Path) -> None:
    """Each curve in metrics.curves becomes one inline <svg class="plot"> on the page."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    curve = [
        {"x": -20.0, "y": 0.10},
        {"x": 0.0, "y": 0.55},
        {"x": 18.0, "y": 0.95},
    ]
    _write(
        results / "sei" / "curve.json",
        _sei_row(
            "row-curve",
            "curve-net",
            "closed_set",
            0.80,
            curves={"accuracy_vs_snr": curve},
        ),
    )

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    # Exactly one curve -> exactly one plot SVG, tagged with its aria-label + figure caption.
    assert sei_html.count('class="plot"') == 1
    assert 'aria-label="accuracy_vs_snr line plot"' in sei_html
    assert '<figcaption class="plot-title">accuracy_vs_snr</figcaption>' in sei_html
    # A polyline is drawn for the model series.
    assert "<polyline" in sei_html


def test_scalar_only_task_has_no_plot(tmp_path: Path) -> None:
    """A task whose rows carry only scalar metrics renders its table but zero plots."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "sei" / "scalar.json",
        _sei_row("row-scalar", "scalar-net", "closed_set", 0.77),
    )

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    # Table rendered (has the model + a primary column), and NO plot SVG at all.
    assert "scalar-net" in sei_html
    assert 'class="plot"' not in sei_html
    assert '<div class="plots">' not in sei_html


def test_rows_sorted_by_primary_descending_within_group(tmp_path: Path) -> None:
    """Within a (regime, track) group, rows are ordered by the primary metric descending."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    tables = _regime_tables(amc_html)
    # In the linear_probe table, iqfm (0.71) must be rendered before mcldnn (0.52).
    assert tables["linear_probe"] == ["iqfm", "mcldnn"]

    # And the ordering is reflected by the sort helper directly.
    rows = generate.load_results(results)
    lp = [r for r in rows if r["task"]["name"] == "amc" and r["regime"]["name"] == "linear_probe"]
    ordered = generate._sort_rows(lp)
    primaries = [generate._primary_value(r) for r in ordered]
    assert primaries == sorted(primaries, reverse=True)
    assert primaries[0] > primaries[-1]


def test_verified_and_self_reported_badges_distinct(tmp_path: Path) -> None:
    """Both verification states surface as distinguishable badges."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "badge-verified" in amc_html
    assert ">verified<" in amc_html
    assert "badge-self" in amc_html
    assert ">self reported<" in amc_html
    # The two badge classes are genuinely different markers.
    assert amc_html.count("badge-verified") != 0
    assert amc_html.count("badge-self") != 0


def test_each_table_carries_exactly_one_regime(tmp_path: Path) -> None:
    """D5: every <table> tag declares exactly one data-regime; two regimes never share one.

    The AMC page has a linear_probe group (2 rows) and a full_finetune group (1 row); each
    renders its OWN table, and no table blends the two regimes.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")

    # Every table tag carries exactly one data-regime (asserted inside the helper), and the
    # AMC page has exactly the two expected regime tables.
    assert sorted(_table_data_regimes(amc_html)) == ["full_finetune", "linear_probe"]

    tables = _regime_tables(amc_html)
    assert set(tables) == {"linear_probe", "full_finetune"}
    # linear_probe holds exactly its two models; full_finetune holds its one.
    assert tables["linear_probe"] == ["iqfm", "mcldnn"]
    assert tables["full_finetune"] == ["mcldnn"]


def test_same_regime_different_track_split_into_separate_tables(tmp_path: Path) -> None:
    """Two rows sharing a regime but declaring different tracks never share a table.

    SEI reports closed_set / cross_receiver / cross_day separately; both fixture rows use
    regime ``from_scratch`` but different ``split.track``, so they must land in two distinct
    tables -- their primary-metric column must never compare across tracks.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    # Same regime (from_scratch, per _sei_row), same model name, DIFFERENT track.
    _write(results / "sei" / "closed.json", _sei_row("row-cs", "wisig-cnn", "closed_set", 0.94))
    _write(results / "sei" / "xrx.json", _sei_row("row-xr", "wisig-cnn", "cross_receiver", 0.60))

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    tables = _tables_by_track_regime(sei_html)
    # Two tables, one per track, both scoped to the same regime.
    assert set(tables) == {("closed_set", "from_scratch"), ("cross_receiver", "from_scratch")}
    # Each track table holds exactly its own row -- tracks are never blended.
    assert tables[("closed_set", "from_scratch")] == ["wisig-cnn"]
    assert tables[("cross_receiver", "from_scratch")] == ["wisig-cnn"]
    # Both tables share the SAME regime yet are distinct tables (no mixing).
    assert _table_data_regimes(sei_html) == ["from_scratch", "from_scratch"]
    # The page tags the tracks distinctly.
    assert 'data-track="closed_set"' in sei_html
    assert 'data-track="cross_receiver"' in sei_html


def test_missing_track_falls_back_to_default_all_bucket(tmp_path: Path) -> None:
    """A row without split.track renders fine under the default 'all' track bucket.

    AMC omits ``track``; the page must still render and its tables key under track ``all``
    (kept label-free for single-track tasks), so existing fixtures keep working.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json", _amc_row("row-a", "iqfm", "linear_probe", 0.71, "self_reported")
    )

    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")

    tables = _tables_by_track_regime(amc_html)
    assert set(tables) == {("all", "linear_probe")}
    assert tables[("all", "linear_probe")] == ["iqfm"]
    assert 'data-track="all"' in amc_html


def test_few_shot_regime_label_carries_k(tmp_path: Path) -> None:
    """A few_shot row is labelled with its k (spaced label), table tagged data-regime=few_shot."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    row = _amc_row("row-fs", "iqfm", "few_shot", 0.40, "self_reported", family="foundation")
    row["regime"]["k_shot"] = 5
    _write(results / "amc" / "fs.json", row)

    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    # REGIME_TITLES spaces the label; few_shot expands to include k.
    assert "few shot (k=5)" in amc_html
    assert 'data-regime="few_shot"' in amc_html
    # The foundation family surfaces a chip.
    assert "chip-foundation" in amc_html


def test_invalid_result_is_skipped(tmp_path: Path) -> None:
    """A schema-invalid JSON is skipped (warned), not rendered, and does not crash."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    # Missing required 'verification' -> invalid against result.schema.json.
    bad = _amc_row("row-bad", "ghost-model", "linear_probe", 0.99, "self_reported")
    del bad["verification"]
    _write(results / "amc" / "bad.json", bad)
    # Also a non-JSON file must not break the scan.
    (results / "amc" / "notjson.json").write_text("{ not valid json", encoding="utf-8")

    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "ghost-model" not in amc_html


def test_declared_task_with_only_invalid_rows_renders_wip_page(tmp_path: Path) -> None:
    """A DECLARED task whose every row is invalid renders a WIP page, not a broken table.

    ``wideband_detection`` is declared in the manifest; when its only submitted row is
    schema-invalid (so it has zero valid results), the build must still write its page as a
    WIP page (clear WIP state, no results table) rather than dropping it or crashing.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json", _amc_row("row-a", "iqfm", "linear_probe", 0.71, "self_reported")
    )
    bad = _amc_row("row-bad", "ghost", "linear_probe", 0.5, "self_reported")
    bad["task"] = {"name": "wideband_detection", "version": "v1"}
    del bad["verification"]
    _write(results / "wideband_detection" / "bad.json", bad)

    generate.build_site(results, out)

    assert (out / "index.html").is_file()
    assert (out / "amc.html").is_file()
    # The declared task still gets a page -- a WIP page (no results, no crash).
    wb_page = out / "wideband_detection.html"
    assert wb_page.is_file()
    wb_html = wb_page.read_text(encoding="utf-8")
    assert "Work in progress" in wb_html
    assert "ghost" not in wb_html
    assert "<table" not in wb_html


def test_load_results_only_returns_valid_rows(tmp_path: Path) -> None:
    """load_results validates against the frozen schema and drops invalid rows."""
    results = tmp_path / "results"
    _make_results_tree(results)
    rows = generate.load_results(results)
    # 4 valid fixtures.
    assert len(rows) == 4
    assert {r["result_id"] for r in rows} == {"row-a", "row-b", "row-c", "row-d"}


# --------------------------------------------------------------------------------------------------
# Declared-task manifest: every declared task appears; WIP tasks render WIP (no results table).
# --------------------------------------------------------------------------------------------------
#: The canonical downstream tasks (docs/DOWNSTREAM_TASKS.md); the committed manifest declares all.
_CANONICAL_TASKS: tuple[str, ...] = (
    "amc",
    "sei",
    "wideband_detection",
    "spectrum_sensing",
    "interference_id",
    "protocol_tech_id",
    "beam_prediction",
    "direction_finding",
    "los_nlos",
    "positioning",
    "har",
    "channel_estimation",
    "snr_mobility_recognition",
)


def test_committed_manifest_declares_every_canonical_task() -> None:
    """The committed leaderboard/tasks.json declares every canonical downstream task."""
    declared = generate.load_manifest()
    assert set(declared) == set(_CANONICAL_TASKS)
    # Each entry carries a recognised status and a non-empty title.
    for entry in declared.values():
        assert entry.status in {"implemented", "wip", "planned"}
        assert entry.title


def test_committed_manifest_status_matches_committed_results() -> None:
    """Honesty guard: any task with a REAL committed result must be declared 'implemented'.

    Catches the class of staleness where a task lands a baseline on the board but its manifest
    status is left at 'wip'/'planned' (so the live site mislabels a working leaderboard as
    work-in-progress). A task with zero committed results may be any status; a task WITH results
    must not be wip/planned.
    """
    declared = generate.load_manifest()
    results_dir = REPO_ROOT / "leaderboard" / "results"
    tasks_with_results = {
        p.parent.name for p in results_dir.rglob("*.json") if p.name != ".gitkeep"
    }
    for task_id in tasks_with_results:
        entry = declared.get(task_id)
        if entry is None:  # undeclared-but-has-results is allowed (manifest is additive)
            continue
        assert entry.status == "implemented", (
            f"{task_id} has committed results on the board but is declared "
            f"status={entry.status!r}; a task with a real baseline must be 'implemented'."
        )


def test_index_lists_every_declared_task(tmp_path: Path) -> None:
    """Every DECLARED task appears on the index -- implemented ones AND WIP ones.

    Only ``amc`` + ``sei`` have result fixtures here; the index must still surface a card
    (and a nav link -> a page) for every other declared task as work-in-progress.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    for task in _CANONICAL_TASKS:
        # A card/nav link to each declared task's page is present on the index.
        assert f"{task}.html" in index_html
        # And its page file was written (full leaderboard or WIP page).
        assert (out / f"{task}.html").is_file()
    # The index advertises the work-in-progress state for tasks lacking a baseline.
    assert "work in progress" in index_html


def test_wip_task_renders_badge_and_no_results_table(tmp_path: Path) -> None:
    """A declared task WITHOUT results renders a WIP badge + NO results table, no crash.

    ``spectrum_sensing`` is declared (status wip) but has no result fixtures here: its page
    must show the WIP badge/state and MUST NOT contain a leaderboard ``<table>`` (a broken
    empty table is the exact failure mode we are avoiding).
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    page = out / "spectrum_sensing.html"
    assert page.is_file()
    html_text = page.read_text(encoding="utf-8")
    # Clear WIP state + badge, and no results table at all.
    assert "Work in progress" in html_text
    assert "status-wip" in html_text
    assert "<table" not in html_text
    assert 'class="plot"' not in html_text


def test_implemented_task_with_results_still_renders_tables_and_plots(tmp_path: Path) -> None:
    """A declared+implemented task that HAS results renders its tables/plots as before.

    ``amc`` is declared implemented AND has fixtures with an ``accuracy_vs_snr`` curve, so
    its page must carry a real leaderboard ``<table>`` and an inline SVG plot -- the manifest
    must not downgrade a task that has data to a WIP page.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    # Give amc a curve so a plot is expected.
    curved = _amc_row("row-curve", "amc-curve", "linear_probe", 0.66, "self_reported")
    curved["metrics"]["curves"] = {
        "accuracy_vs_snr": [{"x": -20.0, "y": 0.1}, {"x": 18.0, "y": 0.9}]
    }
    _write(results / "amc" / "curve.json", curved)

    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "<table" in amc_html
    assert 'class="plot"' in amc_html
    assert "Work in progress" not in amc_html


def test_render_wip_page_is_self_contained_and_tableless() -> None:
    """render_wip_page produces a valid standalone page with a WIP badge and no table."""
    entry = generate.DeclaredTask(
        id="direction_finding",
        title="Direction finding",
        status="planned",
        priority="P1",
        blurb="Angle-of-arrival estimation; blocked on a public dataset.",
    )
    html_text = generate.render_wip_page(entry, ["amc", "direction_finding"])
    assert "<!DOCTYPE html>" in html_text
    assert "Direction finding" in html_text
    assert "status-planned" in html_text
    assert "planned" in html_text
    assert "P1" in html_text
    assert "Work in progress" in html_text
    assert "<table" not in html_text


def test_undeclared_task_with_results_is_still_rendered(tmp_path: Path) -> None:
    """A task with results but ABSENT from the manifest is still rendered (manifest additive).

    ``interference_id`` is declared wip; here we give it a valid result. Even though the
    manifest declares it WIP, having a real result promotes it to a full leaderboard page --
    the manifest never suppresses tasks that have data.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    row = _amc_row("row-if", "interf-net", "linear_probe", 0.63, "self_reported")
    row["task"] = {"name": "interference_id", "version": "v1"}
    _write(results / "interference_id" / "a.json", row)

    generate.build_site(results, out)
    page = out / "interference_id.html"
    assert page.is_file()
    html_text = page.read_text(encoding="utf-8")
    assert "interf-net" in html_text
    assert "<table" in html_text
    assert "Work in progress" not in html_text


def test_load_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing manifest path is a non-fatal warning yielding an empty mapping."""
    assert generate.load_manifest(tmp_path / "no-such-manifest.json") == {}


def test_load_manifest_bad_status_defaults_to_wip(tmp_path: Path) -> None:
    """An entry with an unknown status is coerced to wip (renders as a WIP page)."""
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps({"tasks": [{"id": "amc", "title": "AMC", "status": "bogus"}]}),
        encoding="utf-8",
    )
    declared = generate.load_manifest(manifest)
    assert declared["amc"].status == "wip"


def test_load_manifest_scope_defaults_to_terrestrial_iq(tmp_path: Path) -> None:
    """An entry with no 'scope' key (or an unknown value) defaults to terrestrial_iq."""
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "amc", "title": "AMC", "status": "implemented"},
                    {"id": "har", "title": "HAR", "status": "planned", "scope": "bogus"},
                ]
            }
        ),
        encoding="utf-8",
    )
    declared = generate.load_manifest(manifest)
    assert declared["amc"].scope == "terrestrial_iq"
    assert declared["har"].scope == "terrestrial_iq"


def test_committed_manifest_scope_matches_downstream_tasks_doc() -> None:
    """The committed manifest's scope split matches DOWNSTREAM_TASKS.md's CSI/terrestrial split."""
    declared = generate.load_manifest()
    csi_ids = {t for t, e in declared.items() if e.scope == "csi_sensing"}
    assert csi_ids == {
        "beam_prediction",
        "direction_finding",
        "los_nlos",
        "positioning",
        "har",
        "channel_estimation",
    }
    terrestrial_ids = {t for t, e in declared.items() if e.scope == "terrestrial_iq"}
    assert terrestrial_ids == set(declared) - csi_ids


# --------------------------------------------------------------------------------------------------
# Educational content: per-task explanatory header + shared Guide page.
# --------------------------------------------------------------------------------------------------
def test_committed_manifest_carries_educational_fields() -> None:
    """The committed manifest merges description + dataset card + metric defs into each task."""
    declared = generate.load_manifest()
    amc = declared["amc"]
    # Description + dataset card (name + fields) + primary/secondary metric definitions.
    assert "modulation scheme" in amc.description
    assert amc.dataset.get("name") == "RadioML 2016.10a"
    assert "raw IQ" in amc.dataset.get("modality", "")
    assert amc.primary_metric is not None
    assert amc.primary_metric.name == "accuracy_overall"
    assert "SNR" in amc.primary_metric.definition
    secondary_names = {md.name for md in amc.secondary_metrics}
    assert {"accuracy_vs_snr", "macro_f1"} <= secondary_names
    # A regression-metric task still parses (primary present, no secondary metrics).
    pos = declared["positioning"]
    assert pos.primary_metric is not None
    assert pos.primary_metric.name == "mean_positioning_error"
    assert pos.secondary_metrics == ()


def test_load_manifest_educational_fields_optional(tmp_path: Path) -> None:
    """A manifest entry WITHOUT educational fields loads with empty/None educational data."""
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps({"tasks": [{"id": "amc", "title": "AMC", "status": "implemented"}]}),
        encoding="utf-8",
    )
    declared = generate.load_manifest(manifest)
    amc = declared["amc"]
    assert amc.description == ""
    assert amc.dataset == {}
    assert amc.primary_metric is None
    assert amc.secondary_metrics == ()


def test_task_page_renders_dataset_card_and_metric_defs(tmp_path: Path) -> None:
    """A task page renders its explanatory header: description + dataset card + metric defs.

    The header is driven by the COMMITTED manifest (auto-located from the source tree), so
    building AMC with a real result row must surface AMC's description, its RadioML dataset
    card and the accuracy_overall primary-metric definition above the leaderboard tables.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    # The explanatory header section + a dataset card with the RadioML name and fields.
    assert '<section class="task-header">' in amc_html
    assert '<div class="dataset-card">' in amc_html
    assert "RadioML 2016.10a" in amc_html
    assert ">modality<" in amc_html
    assert ">license<" in amc_html
    # The primary metric definition renders (name + primary tag + its definition text).
    assert '<code class="metric-def-name">accuracy_overall</code>' in amc_html
    assert '<span class="metric-def-tag">primary</span>' in amc_html
    assert "no high-SNR cherry-picking" in amc_html
    # A secondary metric definition renders too.
    assert '<code class="metric-def-name">macro_f1</code>' in amc_html
    # The header sits ABOVE the results (before the first leaderboard table).
    assert amc_html.index('<section class="task-header">') < amc_html.index("<table")


def test_wip_page_renders_educational_header(tmp_path: Path) -> None:
    """A WIP/planned task page still shows the explanatory header (informative with no results).

    ``spectrum_sensing`` is declared wip with educational content but has no result fixtures;
    its page must carry the description + dataset card + primary metric definition ABOVE the
    WIP card, yet still contain no results table.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    page = (out / "spectrum_sensing.html").read_text(encoding="utf-8")
    assert '<section class="task-header">' in page
    assert "DeepSense" in page
    assert '<code class="metric-def-name">pd@pfa=0.1</code>' in page
    # Header is above the WIP card, and there is still no results table.
    assert page.index('<section class="task-header">') < page.index('<div class="wip-card">')
    assert "<table" not in page


def test_task_header_omitted_for_undeclared_task(tmp_path: Path) -> None:
    """A task with results but ABSENT from the manifest renders no educational header.

    ``render_task_page`` with an empty ``declared`` map must not invent a header (the header
    is purely manifest-driven), so the generic leaderboard rendering is untouched.
    """
    row = _amc_row("row-x", "x-net", "linear_probe", 0.5, "self_reported")
    html_text = generate.render_task_page("amc", [row], nav_task_ids=["amc"], declared={})
    assert '<section class="task-header">' not in html_text
    # The generic table still renders.
    assert "<table" in html_text
    assert "x-net" in html_text


# --------------------------------------------------------------------------------------------------
# Task-page two-column layout: sidebar, details/submit cards, dataset-card relocation.
# --------------------------------------------------------------------------------------------------
def test_task_page_has_sidebar_grouped_by_scope(tmp_path: Path) -> None:
    """The task page's sidebar groups every nav task by scope (Terrestrial IQ / CSI-sensing)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert '<nav class="task-sidebar"' in amc_html
    assert 'href="har.html"' in amc_html  # a CSI-sensing task listed in the sidebar
    assert 'href="sei.html"' in amc_html  # a terrestrial-IQ task listed in the sidebar


def test_task_sidebar_marks_current_task_active(tmp_path: Path) -> None:
    """The sidebar highlights the current task's own link."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    sidebar = amc_html[amc_html.index('<nav class="task-sidebar"') :]
    assert 'class="sidebar-task-link sidebar-task-active" href="amc.html"' in sidebar
    assert 'class="sidebar-task-link sidebar-task-active" href="sei.html"' not in sidebar


def test_task_details_card_renders_status_and_dataset(tmp_path: Path) -> None:
    """The compact 'Task details' sidebar card shows dataset + primary metric for AMC."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert '<div class="task-details-card">' in amc_html
    assert "RadioML 2016.10a" in amc_html
    assert "accuracy_overall" in amc_html


def test_submit_card_links_to_submission_guide(tmp_path: Path) -> None:
    """Both the top-nav Submit tab and the task-page sidebar CTA link SUBMISSION.md."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "docs/SUBMISSION.md" in index_html
    assert "docs/SUBMISSION.md" in amc_html
    assert '<div class="submit-card">' in amc_html


def test_wip_page_empty_state_has_no_table_or_plot_svg(tmp_path: Path) -> None:
    """The redesigned empty state still carries zero <table>/plot SVG, and a distinct glyph."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    page = (out / "spectrum_sensing.html").read_text(encoding="utf-8")
    assert "<table" not in page
    assert 'class="plot"' not in page
    assert 'class="empty-glyph"' in page
    assert "No baseline submitted yet" in page


def test_dataset_card_and_metrics_block_still_render_in_main_column(tmp_path: Path) -> None:
    """Regression guard: the full dataset-card/metrics-block still render on amc.html.

    They moved from the hero header into the main column (below the description, above the
    leaderboard) to make room for the new compact sidebar summary -- nothing was deleted.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert '<div class="dataset-card">' in amc_html
    assert '<div class="metrics-block">' in amc_html
    main_pos = amc_html.index('<div class="task-main">')
    dataset_pos = amc_html.index('<div class="dataset-card">')
    table_pos = amc_html.index("<table")
    assert main_pos < dataset_pos < table_pos


def test_rank_one_gets_rank_badge(tmp_path: Path) -> None:
    """The #1 row in a leaderboard table gets a filled rank badge, others stay plain numbers."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert '<span class="rank-badge">1</span>' in amc_html


def test_guide_page_written_with_iq_and_glossary(tmp_path: Path) -> None:
    """build_site writes guide.html with the I/Q section and the metrics glossary.

    The Guide must carry the "What is I/Q?" explainer, the four evaluation regimes, the
    verified-vs-self_reported note, the data + split policies and a metrics glossary that
    lists each metric with an up/down (higher/lower-is-better) arrow.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    guide = out / "guide.html"
    assert guide.is_file()
    guide_html = guide.read_text(encoding="utf-8")
    # What is I/Q? section + its explainer.
    assert "What is I/Q?" in guide_html
    assert "in-phase (I)" in guide_html
    # The four evaluation regimes are each defined.
    for regime in ("from_scratch", "linear_probe", "full_finetune", "few_shot"):
        assert f"<code>{regime}</code>" in guide_html
    # Verification, data + split policies.
    assert "Verified vs self-reported" in guide_html
    assert "Data policy" in guide_html
    assert "Split policy" in guide_html
    # Metrics glossary with named metrics AND both arrow directions.
    assert "Metrics glossary" in guide_html
    assert "<code>accuracy_overall</code>" in guide_html
    assert "<code>eer</code>" in guide_html
    # Up arrow (higher-is-better) and down arrow (lower-is-better) both appear.
    assert "&#9650;" in guide_html  # ▲
    assert "&#9660;" in guide_html  # ▼
    assert "higher is better" in guide_html
    assert "lower is better" in guide_html


def test_guide_linked_in_nav_on_every_page(tmp_path: Path) -> None:
    """Every generated page's top nav links the Guide; the Guide page marks its own tab active."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    # Index, a full task page, and a WIP page all link guide.html in the top nav.
    for name in ("index.html", "amc.html", "spectrum_sensing.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert 'href="guide.html">Guide</a>' in page
        # "Tasks" is the active tab everywhere except the Guide page itself.
        assert 'class="top-tab top-tab-active" href="index.html">Tasks</a>' in page
    # On the Guide page itself, the Guide tab is the active one (not Tasks).
    guide_html = (out / "guide.html").read_text(encoding="utf-8")
    assert 'class="top-tab top-tab-active" href="guide.html">Guide</a>' in guide_html
    assert 'class="top-tab top-tab-active" href="index.html">Tasks</a>' not in guide_html


def test_top_nav_has_only_tasks_guide_submit_and_repo_icon(tmp_path: Path) -> None:
    """The per-task chip list is gone: the top nav is Tasks/Guide/Submit + the repo icon."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    for name in ("index.html", "amc.html", "spectrum_sensing.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert 'class="top-tabs"' in page
        assert "docs/SUBMISSION.md" in page
        assert 'class="icon-link"' in page
        # No more per-task nav chips (the old design this replaces).
        assert "nav-chip" not in page


def test_google_fonts_link_present_on_every_page(tmp_path: Path) -> None:
    """Every page loads the board's Space Grotesk / IBM Plex Google Fonts link once."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    for name in ("index.html", "amc.html", "spectrum_sensing.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert "fonts.googleapis.com" in page
        assert "Space+Grotesk" in page


# --------------------------------------------------------------------------------------------------
# Homepage redesign: stats row, scope sections, filter bar, search/filter script.
# --------------------------------------------------------------------------------------------------
def test_index_stats_row_counts_match_manifest_and_results(tmp_path: Path) -> None:
    """The homepage's 4 stat numbers are computed live from the manifest + loaded results."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    declared = generate.load_manifest()
    grouped: dict[str, list[dict]] = {}
    for row in generate.load_results(results):
        grouped.setdefault(row["task"]["name"], []).append(row)
    expected = generate._compute_stats(grouped, declared)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    for value in expected.values():
        assert f'<span class="stat-value">{value}</span>' in index_html
    assert expected["tasks_defined"] == len(declared)
    assert expected["live"] == 2  # amc + sei both have >=1 valid result in the fixture tree


def test_index_has_two_scope_sections_correctly_populated(tmp_path: Path) -> None:
    """The homepage splits cards into Terrestrial IQ / CSI-RF-sensing sections by scope."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    # Restrict position checks to the <main> body (the header's nav chips also link every
    # task, in a different, unrelated order).
    main_html = index_html[index_html.index("<main>") :]
    assert 'data-scope="terrestrial_iq"' in main_html
    assert 'data-scope="csi_sensing"' in main_html
    terrestrial_pos = main_html.index('data-scope="terrestrial_iq"')
    csi_pos = main_html.index('data-scope="csi_sensing"')
    amc_pos = main_html.index("amc.html")
    har_pos = main_html.index("har.html")
    # amc (terrestrial) sits within the terrestrial section, before the CSI section starts;
    # har (csi_sensing) sits at/after the CSI section heading.
    assert terrestrial_pos < amc_pos < csi_pos <= har_pos


def test_task_card_carries_data_status(tmp_path: Path) -> None:
    """Each task card exposes its declared status via a data-status attribute (JS filter hook)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert 'data-status="implemented"' in index_html  # amc
    assert 'data-status="wip"' in index_html  # spectrum_sensing / interference_id / ...
    assert 'data-status="planned"' in index_html  # beam_prediction / har / ...


def test_filter_bar_and_search_input_present(tmp_path: Path) -> None:
    """The homepage renders the search box and all 4 status filter pills."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="task-search"' in index_html
    for value in ("all", "implemented", "wip", "planned"):
        assert f'data-filter="{value}"' in index_html


def test_index_has_inline_filter_script(tmp_path: Path) -> None:
    """The homepage embeds the vanilla-JS search/filter script inline."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert "<script>" in index_html
    assert "getElementById('task-search')" in index_html
    assert "filter-pill" in index_html


def test_task_and_guide_pages_have_no_inline_script(tmp_path: Path) -> None:
    """The search/filter script is homepage-only -- every other page stays zero-JS."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    for name in ("amc.html", "spectrum_sensing.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert "<script>" not in page


def test_submit_tab_links_to_submission_guide(tmp_path: Path) -> None:
    """The homepage's Submit tab links to docs/SUBMISSION.md on GitHub (no submit.html page)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert "docs/SUBMISSION.md" in index_html
    assert not (out / "submit.html").exists()


def test_render_guide_is_self_contained() -> None:
    """render_guide produces a standalone HTML page (doctype + theme) with the I/Q section."""
    html_text = generate.render_guide()
    assert "<!DOCTYPE html>" in html_text
    assert "Guide — RF-Benchmark-Hub" in html_text
    assert "What is I/Q?" in html_text
    assert "Metrics glossary" in html_text
    # It shares the site-wide top nav (Tasks | Guide | Submit).
    assert 'href="index.html">Tasks</a>' in html_text
    assert 'href="guide.html">Guide</a>' in html_text
