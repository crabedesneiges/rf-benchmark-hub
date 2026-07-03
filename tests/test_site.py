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
* invalid result JSONs are skipped (not rendered) instead of crashing the build.

The generator module lives at ``leaderboard/site/generate.py``; it is loaded by file
path so the test does not depend on ``leaderboard`` being an importable package.
"""

from __future__ import annotations

import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_PATH = REPO_ROOT / "leaderboard" / "site" / "generate.py"


def _load_generate() -> ModuleType:
    """Import ``leaderboard/site/generate.py`` by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location("lb_generate", GENERATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def test_task_with_only_invalid_rows_produces_no_page(tmp_path: Path) -> None:
    """A task whose every row is invalid yields NO <task>.html (index still written)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    # One valid AMC row so the build has content, plus a wideband_detection task whose only
    # row is invalid (missing verification) -> no page for it.
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
    assert not (out / "wideband_detection.html").exists()


def test_load_results_only_returns_valid_rows(tmp_path: Path) -> None:
    """load_results validates against the frozen schema and drops invalid rows."""
    results = tmp_path / "results"
    _make_results_tree(results)
    rows = generate.load_results(results)
    # 4 valid fixtures.
    assert len(rows) == 4
    assert {r["result_id"] for r in rows} == {"row-a", "row-b", "row-c", "row-d"}
