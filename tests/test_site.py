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


def test_index_showcases_verified_tier(tmp_path: Path) -> None:
    """The landing page surfaces verified coverage: a global 'Verified scores' stat card and a
    per-task 'N/M verified' chip. The sample tree has 2 verified rows (of 3 amc + 1 sei)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)

    index_html = generate.build_site(results, out).read_text(encoding="utf-8")

    # Global stat: exactly the 2 verified rows across the board, in the highlighted card.
    assert "Verified scores" in index_html
    assert 'stat-card-verified"><span class="stat-value">2<' in index_html
    # Per-task coverage chips: amc has 2 of its 3 rows verified; sei has 0 of 1 (muted variant).
    assert "2/3 verified" in index_html
    assert "0/1 verified" in index_html
    assert "card-verified-cov-zero" in index_html


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

    # One header cell per distinct scalar metric (each carries a data-metric hook); the Size
    # column also renders a `<th class="num size"` but is NOT a metric column (no data-metric),
    # so counting data-metric headers keeps the "one column per scalar" invariant unambiguous.
    assert sei_html.count('<th class="num size" data-sortable data-sort="num"') == 1
    assert len(re.findall(r'<th class="num[^"]*" data-sortable data-metric="', sei_html)) == 3
    # Each metric name appears as a header cell.
    for metric in ("rank1_accuracy", "auroc", "eer"):
        assert f">{metric}<" in sei_html
    # The primary column is flagged exactly once.
    assert sei_html.count('<span class="col-note">primary</span>') == 1
    # The primary header carries the metric name + the primary marker together. The header now
    # also carries the interactive-sort hooks (data-sortable/data-metric/aria-sort/tabindex),
    # but keeps class="num primary" FIRST and the >rank1_accuracy<span class="col-note">primary
    # substring intact.
    assert '<th class="num primary" data-sortable data-metric="rank1_accuracy"' in sei_html
    assert '>rank1_accuracy<span class="col-note">primary</span>' in sei_html


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


def test_scalar_only_task_has_bar_charts_but_no_curve(tmp_path: Path) -> None:
    """A scalar-only task renders its table + a bar chart per metric, but no 2-D line plot."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "sei" / "scalar.json",
        _sei_row("row-scalar", "scalar-net", "closed_set", 0.77),
    )

    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")

    # Table rendered, plus a per-model bar chart for the scalar metric -- but NO curve/line plot.
    assert "scalar-net" in sei_html
    assert 'class="plot barplot"' in sei_html
    assert "rank1_accuracy by model" in sei_html
    assert "line plot" not in sei_html
    assert '<polyline fill="none"' not in sei_html


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


# --------------------------------------------------------------------------------------------------
# Confidence intervals + CI-overlap tie annotation (schema 1.2.0 metrics.uncertainty)
# --------------------------------------------------------------------------------------------------
def _with_ci(row: dict[str, Any], ci_low: float, ci_high: float) -> dict[str, Any]:
    """Attach a primary-metric confidence interval to an in-memory AMC/SEI row."""
    primary = row["metrics"]["primary"]
    row["schema_version"] = "1.2.0"
    row["metrics"].setdefault("uncertainty", {})[primary] = {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": "bootstrap_percentile",
        "confidence": 0.95,
        "n_resamples": 1000,
    }
    return row


def test_primary_ci_and_overlap_helpers() -> None:
    """``_primary_ci`` reads the interval; ``_cis_overlap`` is symmetric and correct."""
    row = _with_ci(_amc_row("r", "m", "linear_probe", 0.60, "self_reported"), 0.58, 0.62)
    assert generate._primary_ci(row) == (0.58, 0.62)
    # No uncertainty block -> None (pre-1.2.0 / literature rows).
    bare = _amc_row("r2", "m2", "linear_probe", 0.60, "self_reported")
    assert generate._primary_ci(bare) is None

    assert generate._cis_overlap((0.58, 0.62), (0.60, 0.65)) is True
    assert generate._cis_overlap((0.58, 0.62), (0.63, 0.70)) is False
    # A missing interval can never claim an overlap.
    assert generate._cis_overlap((0.58, 0.62), None) is False


def test_overlap_flag_does_not_reorder_rows() -> None:
    """CI overlap annotates but NEVER changes the strict primary/trust/name ordering."""
    # Two rows whose CIs overlap; the higher point estimate must still rank first.
    hi = _with_ci(_amc_row("hi", "alpha", "linear_probe", 0.605, "self_reported"), 0.59, 0.62)
    lo = _with_ci(_amc_row("lo", "beta", "linear_probe", 0.600, "self_reported"), 0.585, 0.615)
    ordered = generate._sort_rows([lo, hi])
    assert [r["model"]["name"] for r in ordered] == ["alpha", "beta"]  # order by point estimate

    flags = generate._overlap_with_above(ordered)
    assert flags == [False, True]  # first never flagged; second overlaps the first


def test_non_overlapping_cis_are_not_flagged() -> None:
    """Well-separated intervals produce no overlap annotation."""
    top = _with_ci(_amc_row("t", "alpha", "linear_probe", 0.70, "self_reported"), 0.68, 0.72)
    bot = _with_ci(_amc_row("b", "beta", "linear_probe", 0.55, "self_reported"), 0.53, 0.57)
    ordered = generate._sort_rows([top, bot])
    assert generate._overlap_with_above(ordered) == [False, False]


def test_overlap_renders_tie_marker_and_ci_note(tmp_path: Path) -> None:
    """Overlapping CIs surface an ≈ tie marker + CI text in the rendered primary cell."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "hi.json",
        _with_ci(_amc_row("hi", "alpha", "linear_probe", 0.605, "self_reported"), 0.59, 0.62),
    )
    _write(
        results / "amc" / "lo.json",
        _with_ci(_amc_row("lo", "beta", "linear_probe", 0.600, "self_reported"), 0.585, 0.615),
    )
    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")

    # The CI text appears for both rows; the ≈ tie marker appears for the overlapping one.
    assert 'class="metric-ci' in amc_html
    assert "&asymp;" in amc_html  # rendered ONLY in a row, never in the CSS block
    assert 'class="ci-tie"' in amc_html
    assert amc_html.count("&asymp;") == 1  # exactly the single overlapping row


def test_no_ci_note_without_uncertainty(tmp_path: Path) -> None:
    """Rows without an uncertainty block render no CI text and no tie marker."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)  # the default fixtures carry no uncertainty
    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "&asymp;" not in amc_html  # no rendered tie markers
    assert 'class="metric-ci"' not in amc_html  # no per-row CI text spans


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
    "snr_estimation",
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
    """The interactive board script is scoped to FULL leaderboard pages only.

    A full leaderboard page (amc, which has tables + charts) now carries the inline board
    script (sort / filter / hover). A WIP page (spectrum_sensing, no board) and the shared
    Guide page have no board to drive, so they carry only the head theme-boot script.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    # WIP + Guide pages have no board to drive -> only the head theme-boot script.
    for name in ("spectrum_sensing.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert page.count("<script>") == 1
        assert "rfb-theme" in page
        assert "sortTable" not in page
    # The full amc leaderboard page adds the board script on top of the theme boot.
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert amc_html.count("<script>") == 2
    # ...but still never contains the "line plot" substring in that script/CSS/controls.
    assert "line plot" not in amc_html.split("</table>")[-1]


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


# --------------------------------------------------------------------------------------------------
# J4 -- contamination badge (pretraining.overlap_with_eval) + lower-is-better regression rendering
# --------------------------------------------------------------------------------------------------
def _with_pretraining(
    row: dict[str, Any], overlap: str, *, note: str | None = None
) -> dict[str, Any]:
    """Attach a schema-1.2.0 ``pretraining`` block with the given overlap to an in-memory row."""
    row["schema_version"] = "1.2.0"
    row["model"]["family"] = "foundation"
    block: dict[str, Any] = {
        "pretrain_datasets": ["radioml_2016_10a"],
        "overlap_with_eval": overlap,
    }
    if note is not None:
        block["disclosure_note"] = note
    row["pretraining"] = block
    return row


def _snr_row(
    result_id: str,
    model_name: str,
    rmse: float,
    mae: float,
    status: str = "self_reported",
) -> dict[str, Any]:
    """Build a genuine ``snr_estimation`` row with the lower-is-better regression metrics.

    ``snr_estimation`` is now a valid ``task.name`` schema enum value, so ``load_results``
    accepts the fixture end-to-end and the row renders on its own board page. The direction
    logic keys off the METRIC name (``rmse_db``), so this exercises the lower-is-better sort/bar
    path on the real task.
    """
    return {
        "schema_version": "1.2.0",
        "result_id": result_id,
        "task": {"name": "snr_estimation", "version": "v1"},
        "model": {"name": model_name, "family": "baseline", "n_params": 100000},
        "regime": {"name": "from_scratch"},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "snr-radioml2016-strat-snr-8010-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": "sha256:" + "0" * 64,
        },
        "metrics": {"primary": "rmse_db", "values": {"rmse_db": rmse, "mae_db": mae}},
        "verification": {"status": status},
    }


def test_lower_is_better_metric_direction_helpers() -> None:
    """``rmse_db``/``mae_db`` are lower-is-better; classification metrics are not."""
    assert generate._is_lower_better("rmse_db") is True
    assert generate._is_lower_better("mae_db") is True
    assert generate._is_lower_better("accuracy_overall") is False
    assert generate._is_lower_better("rank1_accuracy") is False


def test_rmse_rows_ranked_ascending_smallest_error_first() -> None:
    """For a lower-is-better primary, the SMALLEST error ranks first (ascending)."""
    hi_err = _snr_row("hi", "worse", rmse=6.0, mae=5.0)
    lo_err = _snr_row("lo", "better", rmse=2.0, mae=1.5)
    ordered = generate._sort_rows([hi_err, lo_err])
    assert [r["model"]["name"] for r in ordered] == ["better", "worse"]
    # The higher-is-better path is unchanged: accuracy still ranks descending.
    acc_hi = _amc_row("a", "top", "from_scratch", 0.70, "self_reported")
    acc_lo = _amc_row("b", "bot", "from_scratch", 0.55, "self_reported")
    acc_ordered = generate._sort_rows([acc_lo, acc_hi])
    assert [r["model"]["name"] for r in acc_ordered] == ["top", "bot"]


def test_lower_is_better_bar_fills_inversely() -> None:
    """The score bar of the best (smallest-error) row fills MORE than the worst row's."""
    # vmax = 6.0 (the larger error). lower-is-better -> best fill = 1 - 2/6 = 0.667, worst = 0.0.
    best_bar = generate._render_bar(2.0, 6.0, lower_is_better=True)
    worst_bar = generate._render_bar(6.0, 6.0, lower_is_better=True)
    assert "width:66.7%" in best_bar
    assert "width:0.0%" in worst_bar
    # Higher-is-better keeps the direct fill (regression guard on the default path).
    assert "width:100.0%" in generate._render_bar(6.0, 6.0)


def test_contamination_badge_rendered_per_overlap_value(tmp_path: Path) -> None:
    """Each overlap value renders its own contamination badge class + label on the page."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "clean.json",
        _with_pretraining(
            _amc_row("clean", "fm-clean", "linear_probe", 0.62, "self_reported"), "none"
        ),
    )
    _write(
        results / "amc" / "unk.json",
        _with_pretraining(
            _amc_row("unk", "fm-unk", "linear_probe", 0.60, "self_reported"), "unknown"
        ),
    )
    _write(
        results / "amc" / "bad.json",
        _with_pretraining(
            _amc_row("bad", "fm-bad", "linear_probe", 0.58, "self_reported"),
            "confirmed",
            note="pretrained on RadioML incl. the eval split",
        ),
    )
    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")

    assert 'class="badge badge-overlap-none"' in amc_html and ">clean<" in amc_html
    assert 'class="badge badge-overlap-unknown"' in amc_html and ">overlap unknown<" in amc_html
    assert "badge badge-overlap-confirmed" in amc_html and ">contaminated<" in amc_html
    # The disclosure_note becomes the badge tooltip.
    assert 'title="pretrained on RadioML incl. the eval split"' in amc_html


def test_no_contamination_badge_without_pretraining(tmp_path: Path) -> None:
    """Rows without a pretraining block render NO contamination badge (byte-identical to before)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)  # baseline rows, no pretraining
    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    # No RENDERED badge span (the CSS variables/classes always exist in the theme block; a
    # rendered badge is a ``<span class="badge badge-overlap-...``).
    assert 'class="badge badge-overlap-none"' not in amc_html
    assert 'class="badge badge-overlap-unknown"' not in amc_html
    assert 'class="badge badge-overlap-confirmed"' not in amc_html


def test_overlap_badge_helpers_direct() -> None:
    """``_overlap_status``/``_render_overlap_badge`` read the block and default safely."""
    row = _with_pretraining(_amc_row("r", "m", "linear_probe", 0.6, "self_reported"), "confirmed")
    assert generate._overlap_status(row) == "confirmed"
    assert "badge-overlap-confirmed" in generate._render_overlap_badge(row)
    # A bare row (no pretraining) yields no badge.
    bare = _amc_row("r2", "m2", "linear_probe", 0.6, "self_reported")
    assert generate._overlap_status(bare) is None
    assert generate._render_overlap_badge(bare) == ""


def test_guide_documents_contamination_and_regression_metrics() -> None:
    """The Guide explains the contamination badge and lists the regression metrics."""
    html_text = generate.render_guide()
    assert "Contamination badge" in html_text
    # All three badge states are shown as rendered legend chips.
    assert 'class="badge badge-overlap-none"' in html_text
    assert 'class="badge badge-overlap-unknown"' in html_text
    assert 'class="badge badge-overlap-confirmed"' in html_text
    # The regression metrics appear in the glossary with a lower-is-better (down) arrow.
    assert "rmse_db" in html_text
    assert "mae_db" in html_text


def test_snr_estimation_is_a_known_ordered_task() -> None:
    """The generator knows snr_estimation's title and fixes its board order (site support)."""
    assert generate.TASK_TITLES["snr_estimation"] == "SNR estimation"
    assert "snr_estimation" in generate.TASK_ORDER
    # It sorts after the P1/P2 classification tasks but is a KNOWN (non-alphabetical) task.
    key = generate._task_sort_key("snr_estimation")
    assert key[0] < len(generate.TASK_ORDER)


def test_snr_estimation_page_renders_end_to_end(tmp_path: Path) -> None:
    """A real snr_estimation result.json passes load_results and renders its own board page,
    with the smallest-error row ranked first (ascending, lower-is-better)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "snr_estimation" / "worse.json",
        _snr_row("hi", "big-error", rmse=6.0, mae=5.0),
    )
    _write(
        results / "snr_estimation" / "better.json",
        _snr_row("lo", "small-error", rmse=2.0, mae=1.5),
    )
    generate.build_site(results, out)

    page = out / "snr_estimation.html"
    assert page.is_file()  # the row survived load_results validation (enum now accepts the task)
    html = page.read_text(encoding="utf-8")
    assert "SNR estimation" in html
    assert "big-error" in html and "small-error" in html
    # Ascending: the smaller-error model appears before the larger-error one in the table.
    assert html.index("small-error") < html.index("big-error")


# --------------------------------------------------------------------------------------------------
# Methods page: per-method explanations extracted from docstrings + no-paper name linking
# --------------------------------------------------------------------------------------------------
def test_extract_method_docs_covers_no_paper_baselines() -> None:
    """_extract_method_docs pulls each registered model's docstring from source (no torch)."""
    docs = generate._extract_method_docs()
    for name in (
        "mean_snr",
        "snr_moment_ridge",
        "hoc_lr",
        "majority_class",
        "chance",
        "cldnn",
        "complex_cnn",
        "resnet1d_sei",
        "wisig_cnn_paper",
    ):
        assert name in docs, name
        assert docs[name].doc.strip()  # a non-empty explanation
        assert docs[name].source.startswith("rfbench/models/")
    # The explanation is faithful to the implementation (hoc_lr really is a logistic regression).
    assert "logistic regression" in docs["hoc_lr"].doc.lower()


def test_render_docstring_bullets_code_and_roles() -> None:
    """_render_docstring: bullets -> <ul>, code/roles -> <code>, wrapped lines folded, no rst."""
    text = (
        "A short intro paragraph with ``code`` and a :meth:`fit` role.\n\n"
        "* first bullet with a continuation\n"
        "  line folded in\n"
        "* second bullet with :class:`~a.b.Thing`\n"
    )
    html = generate._render_docstring(text)
    assert "<ul>" in html and html.count("<li>") == 2
    assert "<code>code</code>" in html
    assert "<code>fit</code>" in html and "<code>Thing</code>" in html  # role -> referent tail
    assert ":meth:" not in html and ":class:" not in html and "``" not in html  # no rst leak
    assert "continuation line folded in" in html  # wrapped bullet line absorbed


def test_no_paper_methods_link_to_methods_page(tmp_path: Path) -> None:
    """A no-paper method links to methods.html#<name>; a paper method keeps its external url."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "hoc.json",
        _amc_row("hoc-row", "hoc_lr", "from_scratch", 0.30, "self_reported"),
    )
    paper = _amc_row("mcldnn-row", "mcldnn", "from_scratch", 0.60, "self_reported")
    paper["model"]["url"] = "https://doi.org/10.1109/LWC.2020.2999453"
    _write(results / "amc" / "mcldnn.json", paper)
    generate.build_site(results, out)

    methods = (out / "methods.html").read_text(encoding="utf-8")
    assert 'id="hoc_lr"' in methods and "logistic regression" in methods.lower()
    assert 'href="methods.html">Methods' in (out / "index.html").read_text(encoding="utf-8")

    amc = (out / "amc.html").read_text(encoding="utf-8")
    assert '<a href="methods.html#hoc_lr">hoc_lr</a>' in amc  # no-paper -> internal explanation
    assert (
        '<a href="https://doi.org/10.1109/LWC.2020.2999453">mcldnn</a>' in amc
    )  # paper -> external


def test_render_docstring_linkifies_arxiv_and_doi() -> None:
    """arXiv ids and DOIs in a docstring become clickable paper links."""
    html = generate._render_docstring(
        "See arXiv:1905.09388 and DOI 10.1109/ACCESS.2022.3154790 for the architecture."
    )
    assert 'href="https://arxiv.org/abs/1905.09388"' in html
    assert 'href="https://doi.org/10.1109/ACCESS.2022.3154790"' in html


def test_methods_page_shows_paper_refs_and_no_paper_note(tmp_path: Path) -> None:
    """The Methods page surfaces a paper method's references and marks a no-paper method as such."""
    import re

    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "hoc.json",
        _amc_row("hoc-row", "hoc_lr", "from_scratch", 0.3, "self_reported"),
    )
    generate.build_site(results, out)
    methods = (out / "methods.html").read_text(encoding="utf-8")

    # A paper method (complex_cnn cites arXiv in its module docstring) gets a clickable reference.
    assert "Paper / references:" in methods
    assert 'href="https://arxiv.org/abs/1905.09388"' in methods
    # A no-paper method (mean_snr) carries the no-paper note instead.
    section = re.search(
        r'id="mean_snr">.*?(?=<section class="guide-section method"|</section></section>)',
        methods,
        re.S,
    )
    assert section is not None and "No published paper" in section.group(0)


def test_scalar_metric_gets_bar_chart_with_ci_whiskers(tmp_path: Path) -> None:
    """Each scalar metric (no 2-D curve) renders a bar chart; a row's CI -> error-bar whiskers."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    r1 = _amc_row("bar-a", "top", "from_scratch", 0.70, "self_reported")
    r1["schema_version"] = "1.2.0"
    r1["metrics"]["uncertainty"] = {
        "accuracy_overall": {"ci_low": 0.68, "ci_high": 0.72, "method": "bootstrap_percentile"}
    }
    r2 = _amc_row("bar-b", "bot", "from_scratch", 0.55, "self_reported")
    _write(results / "amc" / "a.json", r1)
    _write(results / "amc" / "b.json", r2)
    generate.build_site(results, out)

    amc = (out / "amc.html").read_text(encoding="utf-8")
    assert 'class="plot barplot"' in amc  # a bar chart is rendered for the scalar metric
    assert 'class="barplot-bar"' in amc  # rect uses a scoped class (not the table .bar)
    assert "accuracy_overall by model" in amc
    assert 'class="errbar"' in amc  # the CI on r1 draws whisker error bars
    assert "whiskers = confidence interval" in amc


def test_render_bar_chart_empty_without_metric() -> None:
    """A metric absent from every row yields no chart (no broken SVG)."""
    rows = [_amc_row("z", "m", "from_scratch", 0.5, "self_reported")]
    assert generate._render_bar_chart("does_not_exist", rows) == ""


def test_curve_with_ci_renders_shaded_band(tmp_path: Path) -> None:
    """A curve whose points carry y_low/y_high renders a shaded uncertainty band (ci-band)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    curve = [
        {"x": -10.0, "y": 0.30, "y_low": 0.25, "y_high": 0.35},
        {"x": 10.0, "y": 0.80, "y_low": 0.77, "y_high": 0.83},
    ]
    _write(
        results / "sei" / "c.json",
        _sei_row("curve-row", "net", "closed_set", 0.6, curves={"accuracy_vs_snr": curve}),
    )
    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")
    assert 'class="ci-band"' in sei_html  # the shaded uncertainty envelope
    assert 'aria-label="accuracy_vs_snr line plot"' in sei_html  # the line is still drawn


# --------------------------------------------------------------------------------------------------
# Adversarial-review regressions: XSS, deterministic hues, palette contrast, and the interactive
# markup <-> board-JS contract (sort / filter / tooltip / legend hooks).
# --------------------------------------------------------------------------------------------------
def _rel_luminance(hex_color: str) -> float:
    """WCAG relative luminance of an ``#rrggbb`` colour."""

    def _lin(channel: int) -> float:
        c = channel / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(lum_a: float, lum_b: float) -> float:
    hi, lo = max(lum_a, lum_b), min(lum_a, lum_b)
    return (hi + 0.05) / (lo + 0.05)


def test_tooltip_never_uses_innerHTML_on_dynamic_values() -> None:
    """The shared chart tooltip must build DOM nodes + textContent, never innerHTML.

    ``data-model``/``data-metric`` are contributor-controlled (result.json). ``getAttribute``
    returns the DECODED string, so any ``innerHTML`` sink would re-parse ``<img onerror=...>``
    into live markup (stored DOM XSS). Guard the tool never regresses to a string-HTML sink.
    """
    js = generate.render_scripts()
    assert "innerHTML" not in js
    assert "textContent" in js  # the tooltip is built from text nodes


def test_malicious_model_name_never_renders_live_markup(tmp_path: Path) -> None:
    """A model name containing an HTML payload is escaped everywhere it lands in the page.

    ``model.name`` has no schema pattern (only length 1..128), so a valid result.json can carry
    ``<img src=x onerror=alert(1)>``. It must appear ONLY as escaped text (``&lt;img ...``),
    never as a live ``<img ... onerror=...>`` tag, in any data-* attribute or cell.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    payload = "<img src=x onerror=alert(1)>"
    curve = [{"x": -5.0, "y": 0.4}, {"x": 5.0, "y": 0.7}]
    _write(
        results / "sei" / "x.json",
        _sei_row("xss-row", payload, "closed_set", 0.5, curves={"accuracy_vs_snr": curve}),
    )
    generate.build_site(results, out)
    sei_html = (out / "sei.html").read_text(encoding="utf-8")
    # No LIVE tag: the unescaped '<img ... onerror' opening must never appear (the escaped form
    # '&lt;img ... onerror=alert(1)&gt;' is inert text and is fine to contain the handler string).
    assert "<img src=x onerror" not in sei_html
    # ...but the name is still present, HTML-escaped (angle brackets neutralised).
    assert "&lt;img src=x onerror=alert(1)&gt;" in sei_html


def test_model_hue_is_stable_deterministic_and_valid() -> None:
    """``_model_hue`` is a pure, deterministic ``str -> #rrggbb`` map (crc32, not salted hash).

    Locks (a) idempotence within a run, (b) an exact value so a swap to the salted built-in
    ``hash`` (which breaks cross-run determinism) is caught, and (c) the hex format.
    """
    hex_re = re.compile(r"^#[0-9a-f]{6}$")
    for name in ("iqfm", "mcldnn", "wisig-cnn", "snr_cnn", "chance"):
        first = generate._model_hue(name)
        assert first == generate._model_hue(name)  # stable within a run
        assert hex_re.match(first), first
        assert first in generate._MODEL_PALETTE
    # Exact locked mapping -> guards the crc32-based determinism (would change under salted hash).
    assert generate._model_hue("iqfm") == "#d55a4e"
    # The same model gets the same hue in a line plot AND a bar chart (colour == identity).
    line = generate._render_curve_plot(
        "accuracy_vs_snr", [("iqfm", "baseline", [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.6}])]
    )
    assert 'stroke="#d55a4e"' in line
    bar = generate._render_bar_chart(
        "accuracy_overall", [_amc_row("h", "iqfm", "from_scratch", 0.5, "self_reported")]
    )
    assert "fill:#d55a4e" in bar


def test_palette_hues_meet_3to1_contrast_on_plot_background() -> None:
    """Every series hue clears WCAG 1.4.11 (>= 3:1) against the ACTUAL plot bg ``--surface-2``.

    Curves/markers are informational graphic objects rendered on ``.plot { background:
    var(--surface-2) }`` -- so each hue must be perceivable on that ground in BOTH themes, not
    merely distinguishable from its neighbours. The ``--surface-2`` luminances below are derived
    from the token's two oklch values (light ``0.975 0.004 250`` / dark ``0.25 0.016 260``).
    """
    surface2_light_lum = 0.9272
    surface2_dark_lum = 0.0156
    for hue in generate._MODEL_PALETTE:
        lum = _rel_luminance(hue)
        assert _contrast(lum, surface2_light_lum) >= 3.0, f"{hue} fails on light --surface-2"
        assert _contrast(lum, surface2_dark_lum) >= 3.0, f"{hue} fails on dark --surface-2"


def test_group_table_carries_interactive_hooks(tmp_path: Path) -> None:
    """Every hook the board JS reads for sort/filter is present on the rendered table markup."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json", _amc_row("row-a", "iqfm", "linear_probe", 0.71, "self_reported")
    )
    _write(
        results / "amc" / "b.json", _amc_row("row-b", "mcldnn", "linear_probe", 0.52, "verified")
    )
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")
    # The table advertises its (dataset, regime, track) group so the no-mixing invariant is
    # checkable (dataset added so two datasets of one task never share a table).
    assert "<table data-leaderboard data-dataset=" in amc
    assert "data-regime=" in amc and "data-track=" in amc
    # Each model row carries data-model + data-family + a data-verified consistent with status.
    assert 'data-model="iqfm"' in amc and 'data-family="baseline"' in amc
    assert 'data-model="mcldnn"' in amc
    assert 'data-verified="true"' in amc  # the verified row
    assert 'data-verified="false"' in amc  # the self_reported row
    # Metric cells expose data-value for the numeric sort.
    assert 'data-value="0.71' in amc or 'data-value="0.71"' in amc
    # Exactly one no-match row per table (one linear_probe table here), hidden by default.
    assert amc.count('class="no-match-row"') == 1
    assert '<tr class="no-match-row" hidden>' in amc


def test_board_controls_present_on_full_page_only(tmp_path: Path) -> None:
    """The controls bar (search / verified / family) is on result pages, not WIP/guide pages."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json", _amc_row("row-a", "iqfm", "linear_probe", 0.7, "self_reported")
    )
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")
    assert 'id="board-search"' in amc
    assert 'id="board-verified-only"' in amc
    # Three segmented buttons (all | baseline | foundation); count the <button> markup, not the
    # bare "board-seg" prefix (which also matches the ".board-segmented" wrapper).
    assert amc.count('type="button" class="board-seg') == 3
    for fam in ("all", "baseline", "foundation"):
        assert f'data-family="{fam}"' in amc
    # The board script is injected on this page (so the controls become live).
    assert "js-on" in amc
    # A declared-but-resultless task renders a WIP page with NO controls and NO script.
    guide = (out / "guide.html").read_text(encoding="utf-8")
    assert 'id="board-search"' not in guide
    assert "sortTable" not in guide  # no board script; only the theme boot


def test_chart_points_and_bars_carry_tooltip_data(tmp_path: Path) -> None:
    """Curve points and bar rects expose the data-* the tooltip reads, plus focusability."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    curve = [
        {"x": -10.0, "y": 0.30, "y_low": 0.25, "y_high": 0.35},
        {"x": 10.0, "y": 0.80, "y_low": 0.77, "y_high": 0.83},
    ]
    _write(
        results / "sei" / "c.json",
        _sei_row("pt-row", "net", "closed_set", 0.6, curves={"accuracy_vs_snr": curve}),
    )
    generate.build_site(results, out)
    sei = (out / "sei.html").read_text(encoding="utf-8")
    # Curve points: model + x + y hooks, CI bounds (present because the curve carries y_low/high),
    # keyboard-focusable, and a group (not opaque img) container so points reach the AT.
    assert 'data-model="net"' in sei
    assert "data-x=" in sei and "data-y=" in sei
    assert "data-ci-low=" in sei and "data-ci-high=" in sei
    assert 'class="pt"' in sei and 'tabindex="0"' in sei
    assert '<svg class="plot" viewBox' in sei and 'role="group"' in sei

    # A scalar-only bar chart: bar rects carry data-metric + data-value (+ no CI here).
    results2 = tmp_path / "results2"
    out2 = tmp_path / "site2"
    _write(
        results2 / "amc" / "a.json", _amc_row("bar-row", "m", "from_scratch", 0.5, "self_reported")
    )
    generate.build_site(results2, out2)
    amc = (out2 / "amc.html").read_text(encoding="utf-8")
    assert 'class="barplot-bar"' in amc
    assert 'data-metric="accuracy_overall"' in amc and "data-value=" in amc
    assert 'class="plot barplot"' in amc and 'role="group"' in amc


def test_legend_items_match_series(tmp_path: Path) -> None:
    """Every ``g.series[data-series=X]`` in a plot has a matching ``legend-item[data-series=X]``."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    curve_a = [{"x": 0.0, "y": 0.4}, {"x": 1.0, "y": 0.6}]
    curve_b = [{"x": 0.0, "y": 0.3}, {"x": 1.0, "y": 0.5}]
    _write(
        results / "sei" / "a.json",
        _sei_row("s-a", "alpha", "closed_set", 0.6, curves={"accuracy_vs_snr": curve_a}),
    )
    _write(
        results / "sei" / "b.json",
        _sei_row("s-b", "beta", "closed_set", 0.5, curves={"accuracy_vs_snr": curve_b}),
    )
    generate.build_site(results, out)
    sei = (out / "sei.html").read_text(encoding="utf-8")
    series = set(re.findall(r'<g class="series" data-series="([^"]+)"', sei))
    legend = set(re.findall(r'class="legend-item" data-series="([^"]+)"', sei))
    assert series == legend == {"alpha", "beta"}


def test_sortable_affordance_is_gated_behind_js() -> None:
    """The sort caret + pointer + hover are CSS-hidden until ``body.js-on`` (no dead affordance).

    Headers still carry ``data-sortable``/``tabindex`` in markup (the JS binds click/keydown), but
    with JS off they must not advertise a sort they cannot perform.
    """
    css = generate.render_styles()
    # Default (JS off): the caret is hidden and the header is not a pointer.
    assert ".sort-caret {\n  display: none;" in css
    assert "th[data-sortable] { position: relative; user-select: none; cursor: default; }" in css
    # JS on: the caret + pointer + hover affordance are revealed, exactly like .board-controls.
    assert "body.js-on .sort-caret { display: inline-block; }" in css
    assert "body.js-on th[data-sortable] { cursor: pointer; }" in css
    assert "body.js-on th[data-sortable]:hover { color: var(--fg); }" in css


def test_svg_focus_indicators_have_stroke_fallback() -> None:
    """Focusable SVG points/bars pair ``outline`` with a ``stroke`` ring (WebKit paints stroke)."""
    css = generate.render_styles()
    assert ".plot .pt:focus-visible { outline: 2px solid var(--focus); stroke: var(--focus)" in css
    assert "stroke-width: 2" in css.split(".plot .barplot-bar:focus-visible")[1][:120]


def test_home_search_and_pills_have_a_focus_ring() -> None:
    """The homepage search box keeps a 2px focus ring (never ``outline: none``); pills get one."""
    css = generate.render_styles()
    assert ".search-input:focus { outline: none;" not in css
    assert ".search-input:focus-visible {\n  outline: 2px solid var(--focus);" in css
    assert (
        ".filter-pill:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }"
        in css
    )


# --------------------------------------------------------------------------------------------------
# Foundation Models page -- dedicated foundation-only podiums (leaderboard/site/generate.py
# render_foundation) + the global cumulative medal table. All rows are 100% synthetic/in-tree;
# no task or model name used below is hardcoded anywhere in the generator.
# --------------------------------------------------------------------------------------------------
def _foundation_row(
    result_id: str,
    task: str,
    model_name: str,
    regime: str,
    value: float,
    *,
    status: str = "self_reported",
    k_shot: int | None = None,
    track: str | None = None,
    primary: str = "accuracy_overall",
    url: str | None = None,
) -> dict[str, Any]:
    """Build a schema-valid FOUNDATION-family (``model.family == "foundation"``) result row.

    ``task`` must be one of the schema's registered task ids (``amc``, ``sei``,
    ``wideband_detection``, ``spectrum_sensing``, ``interference_id``, ``protocol_tech_id``,
    ``snr_estimation``) -- ``task.name`` is a locked enum, not free-form.
    """
    regime_block: dict[str, Any] = {"name": regime}
    if k_shot is not None:
        regime_block["k_shot"] = k_shot
    model: dict[str, Any] = {"name": model_name, "family": "foundation", "pretrained": True}
    if url is not None:
        model["url"] = url
    split_slug = task.replace("_", "-")  # canonical_split_id forbids underscores.
    split: dict[str, Any] = {
        "canonical_split_id": f"{split_slug}-synth-8010-seed42-v1",
        "name": "test",
        "seed": 42,
        "checksum": "sha256:" + "2" * 64,
    }
    if track is not None:
        split["track"] = track
    return {
        "schema_version": "1.0.0",
        "result_id": result_id,
        "task": {"name": task, "version": "v1"},
        "model": model,
        "regime": regime_block,
        "dataset": {"name": f"{task}_synth"},
        "split": split,
        "metrics": {"primary": primary, "values": {primary: value}},
        "verification": {"status": status},
    }


def test_regime_track_sort_key_orders_regime_then_kshot_then_track() -> None:
    """The hoisted ``_regime_track_sort_key`` orders (regime, k_shot, track) exactly as
    ``render_task_page`` always has -- D5 regime order, ``k_shot`` ascending, then track (default
    ``all`` bucket first) -- proving the Foundation page's grouping matches the task pages'."""
    keys = [
        ("full_finetune", None, "all"),
        ("from_scratch", None, "all"),
        ("few_shot", 50, "all"),
        ("few_shot", 5, "all"),
        ("linear_probe", None, "z_track"),
        ("linear_probe", None, "all"),
    ]
    ordered = sorted(keys, key=generate._regime_track_sort_key)
    assert ordered == [
        ("from_scratch", None, "all"),
        ("full_finetune", None, "all"),
        ("linear_probe", None, "all"),
        ("linear_probe", None, "z_track"),
        ("few_shot", 5, "all"),
        ("few_shot", 50, "all"),
    ]


def test_foundation_page_linked_in_nav_and_tab_active_only_there(tmp_path: Path) -> None:
    """Every page's top nav links foundation.html; only foundation.html marks that tab active."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    for name in ("index.html", "amc.html", "spectrum_sensing.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert 'href="foundation.html">Foundation</a>' in page
        assert 'top-tab top-tab-active" href="foundation.html">Foundation</a>' not in page

    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")
    assert 'top-tab top-tab-active" href="foundation.html">Foundation</a>' in foundation_html
    assert 'top-tab top-tab-active" href="index.html">Tasks</a>' not in foundation_html


def test_foundation_page_empty_state_when_no_foundation_rows(tmp_path: Path) -> None:
    """With zero ``model.family == "foundation"`` rows anywhere, foundation.html shows the WIP
    card instead of an empty/broken table -- same graceful-degradation philosophy as a task's
    own WIP page."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)  # baselines only, no foundation rows.
    generate.build_site(results, out)

    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")
    assert "No foundation-model results yet" in foundation_html
    assert "Work in progress" in foundation_html
    assert 'class="podium-table"' not in foundation_html


def test_foundation_ranking_excludes_baselines_from_medals(tmp_path: Path) -> None:
    """A baseline row scoring higher than every foundation row must not win the Foundation
    page's medal, nor even appear on the page: ranking is scoped to ``family == "foundation"``."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "baseline.json",
        _amc_row("row-base", "mcldnn", "linear_probe", 0.99, "verified"),
    )
    _write(
        results / "amc" / "f1.json",
        _foundation_row("row-f1", "amc", "iqfm-base", "linear_probe", 0.80),
    )
    _write(
        results / "amc" / "f2.json",
        _foundation_row(
            "row-f2",
            "amc",
            "wirelessjepa-paper",
            "linear_probe",
            0.75,
            status="from_paper_uncertain",
        ),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    # The baseline never enters the foundation RANKING tbody (the medal table), even though it
    # scores higher: locate the task SECTION (not a selector button) via its unique class prefix.
    section = foundation_html[foundation_html.index('foundation-task" data-task="amc"') :]
    tbody = section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    trs = re.findall(r"<tr>.*?</tr>", tbody, re.S)
    assert len(trs) == 2
    assert "\U0001f947" in trs[0] and "iqfm-base" in trs[0]
    assert "\U0001f948" in trs[1] and "wirelessjepa-paper" in trs[1]
    assert "mcldnn" not in tbody  # the specialist is not in the FM ranking/medals ...
    # ... but it DOES appear as the labelled best-baseline REFERENCE (the vs-baselines feature).
    ref = section.split("data-baseline-ref", 1)[1].split("</div>", 1)[0]
    assert "mcldnn" in ref
    assert "best baseline" in ref


def test_single_competitor_group_gets_gold_and_beyond_third_has_no_medal(tmp_path: Path) -> None:
    """A (regime, track) group with exactly one foundation competitor still awards gold (expected
    given today's sparse submissions); a 4th-place competitor in a 4-way group gets no medal."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "sei" / "solo.json",
        _foundation_row("row-solo", "sei", "iqfm-base", "full_finetune", 0.5),
    )
    for i, (name, val) in enumerate([("m1", 0.9), ("m2", 0.8), ("m3", 0.7), ("m4", 0.6)]):
        _write(
            results / "wideband_detection" / f"r{i}.json",
            _foundation_row(f"row-r{i}", "wideband_detection", name, "full_finetune", val),
        )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    # TASK_ORDER puts "sei" before "wideband_detection" -> sei's section comes first.
    # Target the task SECTIONS (unique class prefix), not the new selector buttons that also
    # carry data-task.
    sei_start = foundation_html.index('foundation-task" data-task="sei"')
    wd_start = foundation_html.index('foundation-task" data-task="wideband_detection"')
    sei_section = foundation_html[sei_start:wd_start]
    solo_tbody = sei_section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert "\U0001f947" in solo_tbody  # sole competitor in its group still gets gold.

    wd_section = foundation_html[wd_start:]
    tbody = wd_section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    trs = re.findall(r"<tr>.*?</tr>", tbody, re.S)
    assert len(trs) == 4
    assert "\U0001f947" in trs[0]
    assert "\U0001f948" in trs[1]
    assert "\U0001f949" in trs[2]
    assert "\U0001f947" not in trs[3] and "\U0001f948" not in trs[3] and "\U0001f949" not in trs[3]
    assert '<td class="rank num">4</td>' in trs[3]


def test_global_podium_caps_one_medal_per_task_across_groups(tmp_path: Path) -> None:
    """Global cumulative podium: best rank per task, even across multiple (regime, track)
    groups within that task -- a competitor winning gold in TWO tracks of the SAME task is
    credited only ONE gold for that task, not two."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json",
        _foundation_row("row-a", "amc", "iqfm-base", "full_finetune", 0.9, track="track_a"),
    )
    _write(
        results / "amc" / "b.json",
        _foundation_row("row-b", "amc", "iqfm-base", "full_finetune", 0.7, track="track_b"),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    podium_start = foundation_html.index("global-podium")
    podium_end = foundation_html.index("</section>", podium_start)
    podium = foundation_html[podium_start:podium_end]
    trs = re.findall(r"<tr>.*?</tr>", podium.split("<tbody>", 1)[1], re.S)
    assert len(trs) == 1
    assert "iqfm-base" in trs[0]
    assert "&times;1" in trs[0]  # exactly ONE gold credited for the task, not two.


def test_global_podium_treats_each_result_row_as_its_own_competitor(tmp_path: Path) -> None:
    """``iqfm-base`` and ``iqfm-paper`` are two distinct result rows/competitors on the global
    podium -- the ONE table on this board that deliberately departs from the "never mix two
    regimes/tiers in one table" rule enforced everywhere else. Sorted gold desc, then silver
    desc, then bronze desc, with name as the final deterministic tiebreak."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "base.json",
        _foundation_row("row-1", "amc", "iqfm-base", "linear_probe", 0.9),
    )
    _write(
        results / "amc" / "paper.json",
        _foundation_row("row-2", "amc", "iqfm-paper", "linear_probe", 0.8, status="from_paper"),
    )
    _write(
        results / "sei" / "base.json",
        _foundation_row("row-3", "sei", "iqfm-base", "linear_probe", 0.5),
    )
    _write(
        results / "sei" / "paper.json",
        _foundation_row("row-4", "sei", "iqfm-paper", "linear_probe", 0.6, status="from_paper"),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    podium_start = foundation_html.index("global-podium")
    podium_end = foundation_html.index("</section>", podium_start)
    podium = foundation_html[podium_start:podium_end]
    trs = re.findall(r"<tr>.*?</tr>", podium.split("<tbody>", 1)[1], re.S)
    assert len(trs) == 2
    # Both distinct competitors appear as their OWN row (never merged into one "iqfm" identity).
    assert '<span class="model-name">iqfm-base</span>' in trs[0] + trs[1]
    assert '<span class="model-name">iqfm-paper</span>' in trs[0] + trs[1]
    # Both tally 1 gold + 1 silver (the two tasks swap the winner).
    for tr in trs:
        assert "&times;1" in tr
    assert "iqfm-base" in trs[0]  # exact tie on medal counts -> alphabetical tiebreak.


def test_scatter_skipped_for_single_point_task(tmp_path: Path) -> None:
    """A task with fewer than 2 total foundation points renders no scatter plot at all (same
    graceful-degradation philosophy as the WIP cards -- no empty/misleading graph)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "spectrum_sensing" / "a.json",
        _foundation_row("row-a", "spectrum_sensing", "iqfm-base", "linear_probe", 0.8),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")
    assert 'class="plot"' not in foundation_html


def test_scatter_orders_adaptation_cost_axis_and_subsorts_few_shot_by_k(tmp_path: Path) -> None:
    """The frugality x-axis orders few_shot (sub-sorted by k ascending) -> linear_probe ->
    full_finetune -- a categorical adaptation-cost ordering, never an invented numeric FLOPs
    axis the schema doesn't track."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    for i, (regime, k) in enumerate(
        [("full_finetune", None), ("linear_probe", None), ("few_shot", 50), ("few_shot", 5)]
    ):
        _write(
            results / "snr_estimation" / f"r{i}.json",
            _foundation_row(
                f"row-{i}", "snr_estimation", "wirelessjepa-paper", regime, 0.5 + i * 0.05, k_shot=k
            ),
        )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    section_start = foundation_html.index('data-task="snr_estimation"')
    svg_start = foundation_html.index('class="plot"', section_start)
    svg_end = foundation_html.index("</svg>", svg_start)
    svg = foundation_html[svg_start:svg_end]

    pos_k5 = svg.index("few shot (k=5)")
    pos_k50 = svg.index("few shot (k=50)")
    pos_linear = svg.index("linear probe")
    pos_full = svg.index("full finetune")
    assert pos_k5 < pos_k50 < pos_linear < pos_full


def test_foundation_rows_reuse_existing_verification_badge_and_show_regime_track_chips(
    tmp_path: Path,
) -> None:
    """Each foundation row shows a regime chip, a track chip (when non-default), and the SAME
    verification-tier badge markup used on the per-task pages (reused, never reimplemented)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "protocol_tech_id" / "a.json",
        _foundation_row(
            "row-a",
            "protocol_tech_id",
            "iqfm-base",
            "few_shot",
            0.8,
            k_shot=10,
            status="from_paper_uncertain",
            track="cross_room",
        ),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")

    assert generate._render_badge("from_paper_uncertain") in foundation_html
    assert 'class="chip chip-regime"' in foundation_html
    assert 'class="chip chip-track"' in foundation_html


def test_foundation_page_has_only_minimal_gated_script(tmp_path: Path) -> None:
    """foundation.html carries the head theme-boot script PLUS exactly one small gated script
    (the task selector + "Include baselines" toggle) -- and never the full board script.

    So: exactly two ``<script>`` blocks, no ``sortTable`` (not the board script), and the
    no-JS degradation holds -- neither a ``.foundation-task`` section nor a baseline reference
    is hidden by default CSS, so with JS off every section AND the baseline reference stay
    visible; only the controls bar is CSS-hidden until ``body.js-on``.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    _write(
        results / "amc" / "found.json",
        _foundation_row("row-f", "amc", "iqfm-base", "linear_probe", 0.8),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")
    # Head theme boot + the one minimal gated foundation script = 2, and NOT the board script.
    assert foundation_html.count("<script>") == 2
    assert "sortTable" not in foundation_html
    # The controls bar is gated (hidden until body.js-on); no default rule hides a section.
    css = generate.render_styles()
    assert ".foundation-controls { display: none; }" in css
    assert "body.js-on .foundation-controls { display: flex; }" in css
    assert ".foundation-task { display: none" not in css  # no section is hidden by default
    # With JS off the baseline reference is present and VISIBLE (no default is-hidden on it).
    assert 'class="baseline-reference" data-baseline-ref' in foundation_html
    assert 'class="baseline-reference is-hidden"' not in foundation_html


def test_render_foundation_is_self_contained() -> None:
    """render_foundation produces a standalone HTML page sharing the site-wide nav (mirrors
    test_render_guide_is_self_contained's direct-call pattern)."""
    declared = generate.load_manifest()
    grouped: dict[str, list[dict[str, Any]]] = {
        "amc": [_foundation_row("row-a", "amc", "iqfm-base", "linear_probe", 0.8)]
    }
    html_text = generate.render_foundation(grouped, declared)
    assert "<!DOCTYPE html>" in html_text
    assert "Foundation Models — RF-Benchmark-Hub" in html_text
    assert 'href="index.html">Tasks</a>' in html_text
    assert 'href="foundation.html">Foundation</a>' in html_text


def test_foundation_page_css_present() -> None:
    """The Foundation page's chip/medal/podium CSS is emitted once in the shared stylesheet."""
    css = generate.render_styles()
    assert ".chip-regime {" in css
    assert ".chip-track {" in css
    assert ".medal { font-size: 1.1rem; }" in css
    assert ".global-podium { margin-bottom: 2rem; }" in css


# --------------------------------------------------------------------------------------------------
# Agent 1: model size + n_flops compute proxy + the size/perf Pareto scatter
# (schema 1.3.0 `model.n_flops`; generate.py _render_size_cell / _pareto_frontier /
# _render_pareto_scatter). All rows below are synthetic/in-tree.
# --------------------------------------------------------------------------------------------------
def _sized_amc_row(
    result_id: str,
    model_name: str,
    accuracy: float,
    *,
    n_params: int | None = None,
    n_flops: int | None = None,
    family: str = "baseline",
    regime: str = "linear_probe",
) -> dict[str, Any]:
    """An AMC row with explicit control over ``model.n_params`` / ``model.n_flops`` (or neither)."""
    model: dict[str, Any] = {"name": model_name, "family": family}
    if n_params is not None:
        model["n_params"] = n_params
    if n_flops is not None:
        model["n_flops"] = n_flops
    schema_version = "1.3.0" if n_flops is not None else "1.0.0"
    return {
        "schema_version": schema_version,
        "result_id": result_id,
        "task": {"name": "amc", "version": "v1"},
        "model": model,
        "regime": {"name": regime},
        "dataset": {"name": "radioml_2016_10a"},
        "split": {
            "canonical_split_id": "amc-radioml2016-strat-snr-8010-seed42-v1",
            "name": "test",
            "seed": 42,
            "checksum": "sha256:" + "0" * 64,
        },
        "metrics": {"primary": "accuracy_overall", "values": {"accuracy_overall": accuracy}},
        "verification": {"status": "self_reported"},
    }


def test_size_column_present_and_sortable_on_leaderboard_table(tmp_path: Path) -> None:
    """The leaderboard table carries a sortable Size column; its cell shows params (and a muted
    FLOPs sub-line when n_flops is present) and carries data-value = n_flops if present else
    n_params, so the generic board sort orders by compute/size."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json",
        _sized_amc_row("row-a", "big", 0.80, n_params=2_300_000, n_flops=1_200_000_000),
    )
    _write(results / "amc" / "b.json", _sized_amc_row("row-b", "small", 0.70, n_params=90_000))
    _write(results / "amc" / "c.json", _sized_amc_row("row-c", "unsized", 0.60))
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")

    # A sortable Size header (num + data-sortable + data-sort="num"), exactly one per table.
    assert (
        '<th class="num size" data-sortable data-sort="num" aria-sort="none" tabindex="0">Size'
        in amc
    )
    # params on the main line; FLOPs on the muted sub-line only for the row that declares them.
    assert '<span class="size-params">2.3M</span>' in amc
    assert '<span class="size-flops">1.2G FLOPs</span>' in amc
    assert '<span class="size-params">90K</span>' in amc
    # data-value = n_flops when present, else n_params; the unsized row gets a muted en-dash cell.
    assert 'data-value="1200000000"' in amc  # FLOPs preferred over params for sorting
    assert 'data-value="90000"' in amc
    assert '<td class="num size"><span class="size-params">&ndash;</span></td>' in amc
    # The inline params span next to the model name is gone (params moved into the Size column).
    assert 'class="params"' not in amc


def test_size_column_present_on_foundation_table(tmp_path: Path) -> None:
    """The foundation mini-table also renders a Size column with the same params/FLOPs cell."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "f.json",
        {
            **_foundation_row("row-f", "amc", "iqfm-base", "linear_probe", 0.8),
            "schema_version": "1.3.0",
        },
    )
    # give the foundation row a size
    row = json.loads((results / "amc" / "f.json").read_text())
    row["model"]["n_params"] = 90_000
    _write(results / "amc" / "f.json", row)
    generate.build_site(results, out)
    foundation = (out / "foundation.html").read_text(encoding="utf-8")
    assert '<th class="num size" data-sortable data-sort="num"' in foundation
    assert '<span class="size-params">90K</span>' in foundation


def test_pareto_scatter_present_when_two_sized_points(tmp_path: Path) -> None:
    """A task with >=2 models carrying size data renders the size/perf Pareto scatter (labelled
    an efficiency/reference view spanning regimes, NOT a ranking)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(results / "amc" / "a.json", _sized_amc_row("row-a", "big", 0.80, n_params=2_300_000))
    _write(results / "amc" / "b.json", _sized_amc_row("row-b", "small", 0.70, n_params=90_100))
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")

    assert 'aria-label="accuracy_overall vs model size scatter"' in amc
    assert 'class="pareto-front"' in amc  # the frontier staircase is drawn
    assert "parameters (log scale)" in amc  # axis names the size proxy in use
    # Explicit efficiency/reference labelling (spans regimes, not a ranking).
    assert "Efficiency / reference view" in amc


def test_pareto_scatter_uses_flops_axis_when_any_row_declares_it(tmp_path: Path) -> None:
    """When ANY row declares n_flops the scatter's X axis switches to FLOPs (the
    hardware-independent compute proxy), naming it in the axis title."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "a.json",
        _sized_amc_row("row-a", "big", 0.80, n_params=2_300_000, n_flops=1_200_000_000),
    )
    _write(results / "amc" / "b.json", _sized_amc_row("row-b", "small", 0.70, n_flops=90_000_000))
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")
    assert "FLOPs (log scale)" in amc
    assert "parameters (log scale)" not in amc


def test_pareto_scatter_skipped_below_two_sized_points(tmp_path: Path) -> None:
    """A task with fewer than 2 sized models renders no Pareto scatter (graceful degradation);
    a size-less point is dropped, not silently placed at an arbitrary size."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(results / "amc" / "a.json", _sized_amc_row("row-a", "sized", 0.80, n_params=90_100))
    _write(results / "amc" / "b.json", _sized_amc_row("row-b", "unsized", 0.70))
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")
    assert "vs model size scatter" not in amc


def test_pareto_scatter_on_foundation_page_when_sized(tmp_path: Path) -> None:
    """The Pareto scatter also renders on the foundation page for a task whose foundation rows
    carry size data (>=2), and foundation markers read apart via a dashed outline."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    for rid, name, val, npar in (
        ("f1", "iqfm-base", 0.80, 90_100),
        ("f2", "wjepa", 0.75, 2_300_000),
    ):
        row = _foundation_row(rid, "amc", name, "linear_probe", val)
        row["schema_version"] = "1.3.0"
        row["model"]["n_params"] = npar
        _write(results / "amc" / f"{rid}.json", row)
    generate.build_site(results, out)
    foundation = (out / "foundation.html").read_text(encoding="utf-8")
    assert 'aria-label="accuracy_overall vs model size scatter"' in foundation
    # foundation family => dashed marker outline (a non-colour family channel).
    assert 'stroke-dasharray="7 4"' in foundation


def test_pareto_frontier_dominance_and_ties() -> None:
    """`_pareto_frontier` returns the non-dominated set sorted by x, keeps exact ties, handles a
    single point, and inverts correctly for a lower-is-better y."""
    P = generate.ParetoPoint
    # Higher-is-better y: 'a' cheapest, 'b' best score; 'c' is dominated by 'b' (>= x, <= y).
    fr = generate._pareto_frontier([P(1.0, 0.5, "a"), P(2.0, 0.9, "b"), P(3.0, 0.7, "c")])
    assert [p.label for p in fr] == ["a", "b"]  # sorted by x ascending
    # A strictly dominated point drops out: 'lo' has more params AND worse score than 'hi'.
    fr2 = generate._pareto_frontier([P(10.0, 0.9, "hi"), P(20.0, 0.6, "lo")])
    assert [p.label for p in fr2] == ["hi"]
    # Exact (x, y) ties do not dominate one another -> both are kept.
    tied = generate._pareto_frontier([P(1.0, 0.5, "a"), P(1.0, 0.5, "dup"), P(2.0, 0.4, "b")])
    assert sorted(p.label for p in tied) == ["a", "dup"]
    # Single point is trivially non-dominated.
    assert [p.label for p in generate._pareto_frontier([P(5.0, 0.3, "solo")])] == ["solo"]
    # Lower-is-better y (a regression error): smaller y is better.
    lb = generate._pareto_frontier(
        [P(1.0, 2.0, "a"), P(2.0, 1.0, "b"), P(3.0, 3.0, "c")], y_lower_is_better=True
    )
    assert [p.label for p in lb] == ["a", "b"]  # 'c' dominated (more params, larger error)


def test_pareto_scatter_respects_lower_is_better_metric(tmp_path: Path) -> None:
    """On a lower-is-better task (snr_estimation rmse_db) the scatter labels the perf direction
    as 'lower is better' -- the Pareto direction follows the metric, not the task."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    for i, (name, rmse, npar) in enumerate([("small", 3.0, 50_000), ("big", 2.0, 5_000_000)]):
        _write(
            results / "snr_estimation" / f"r{i}.json",
            {
                "schema_version": "1.0.0",
                "result_id": f"row-{i}",
                "task": {"name": "snr_estimation", "version": "v1"},
                "model": {"name": name, "family": "baseline", "n_params": npar},
                "regime": {"name": "from_scratch"},
                "dataset": {"name": "radioml_2016_10a"},
                "split": {
                    "canonical_split_id": "snr-estimation-synth-8010-seed42-v1",
                    "name": "test",
                    "seed": 42,
                    "checksum": "sha256:" + "3" * 64,
                },
                "metrics": {"primary": "rmse_db", "values": {"rmse_db": rmse}},
                "verification": {"status": "self_reported"},
            },
        )
    generate.build_site(results, out)
    snr = (out / "snr_estimation.html").read_text(encoding="utf-8")
    # The figcaption (which carries the direction arrow) precedes the svg; slice from it.
    start = snr.index('<figcaption class="plot-title">rmse_db vs model size')
    figure = snr[start : snr.index("</figure>", start)]
    assert "&darr; lower is better" in figure
    assert 'aria-label="rmse_db vs model size scatter"' in figure


def test_guide_documents_size_and_compute(tmp_path: Path) -> None:
    """The Guide has a 'Model size & compute' section (id=size-compute) explaining FLOPs as the
    hardware-independent proxy vs the hardware-specific inference_latency_ms, and the glossary
    defines n_params and n_flops."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)
    guide = (out / "guide.html").read_text(encoding="utf-8")
    assert 'id="size-compute"' in guide
    assert "Model size &amp; compute" in guide
    assert "FLOPs" in guide
    assert "hardware-independent" in guide
    assert "inference_latency_ms" in guide
    # Glossary rows for the two size/compute quantities.
    assert ">n_params<" in guide
    assert ">n_flops<" in guide


# --- Sprint 2 (adoption) additions ---
def test_hero_has_cta_row(tmp_path: Path) -> None:
    """The homepage hero exposes Submit / Guide / GitHub calls-to-action (usable with JS off)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert '<div class="hero-cta">' in index_html
    assert generate._SUBMISSION_GUIDE_URL in index_html
    assert 'href="guide.html"' in index_html
    assert generate._REPO_URL in index_html


def test_submit_card_shows_real_command(tmp_path: Path) -> None:
    """The submit card carries the real self-serve command in a selectable <pre> (no JS needed)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert '<pre class="cmd">' in amc_html
    assert "rfbench eval" in amc_html


def test_head_has_meta_and_favicon(tmp_path: Path) -> None:
    """Every page's <head> carries a description, Open Graph title and an inline SVG favicon."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert '<meta name="description"' in index_html
    assert '<meta property="og:title"' in index_html
    assert 'rel="icon" href="data:image/svg+xml,' in index_html


def test_task_page_meta_description_is_page_specific() -> None:
    """A declared task's meta description prefers its own blurb over the site tagline."""
    entry = generate.DeclaredTask(
        id="amc",
        title="Automatic modulation classification",
        status="implemented",
        priority="P1",
        blurb="Predict the modulation scheme of a raw-IQ window.",
    )
    desc = generate._page_description_for(entry, entry.title)
    assert desc == entry.blurb


def test_mobile_layout_puts_leaderboard_first() -> None:
    """At <=900px the leaderboard (task-main) is ordered before the task-nav sidebar."""
    css = generate.render_styles()
    idx = css.index("@media (max-width: 900px)")
    block = css[idx : idx + 200]
    assert ".task-main { order: -1; }" in block


# --- UI fixes: tier-legend collision, dataset/metrics spacing, theme toggle ---
def test_tier_legend_uses_dedicated_classes() -> None:
    """Tier-legend pills use tl-* classes, never the chart-legend legend-item/swatch names.

    The chart legend pins .legend-swatch to a fixed 24x10 box; reusing that class for the
    tier pills clobbered their layout (overlapping text). The namespaces must stay distinct.
    """
    legend = generate._render_tier_legend()
    assert 'class="tl-swatch' in legend
    assert 'class="tl-item"' in legend
    assert "legend-swatch" not in legend
    assert "legend-item" not in legend
    css = generate.render_styles()
    assert ".tl-swatch" in css
    # The chart-legend swatch rule is untouched.
    assert ".legend-swatch { width: 24px; height: 10px; flex: none; }" in css


def test_task_page_wraps_dataset_and_metrics_in_grid(tmp_path: Path) -> None:
    """dataset-card + metrics-block sit in a task-header-grid wrapper (gap between boxes)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    main = amc_html.split('<div class="task-main">', 1)[1]
    assert main.startswith('<div class="task-header-grid">')
    assert ".task-main > .task-header-grid { margin: 0 0 1.1rem; }" in amc_html


def test_theme_toggle_present_and_progressive(tmp_path: Path) -> None:
    """Every page ships the theme boot + a toggle button hidden until JS marks html.theme-js."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    for name in ("index.html", "amc.html", "guide.html"):
        page = (out / name).read_text(encoding="utf-8")
        assert 'class="icon-link theme-toggle"' in page
        assert "rfb-theme" in page
    css = generate.render_styles()
    assert ".theme-toggle { display: none;" in css
    assert "html.theme-js .theme-toggle { display: inline-flex; }" in css
    assert ':root[data-theme="dark"]' in css
    assert ':root:not([data-theme="light"])' in css


# --------------------------------------------------------------------------------------------------
# Agent 2: dataset selector (task pages), Foundation task selector, vs-baselines reference/toggle.
# All rows below are synthetic/in-tree; no task/model/dataset name is hardcoded in the generator.
# --------------------------------------------------------------------------------------------------
def _amc_row_on(result_id: str, model_name: str, accuracy: float, dataset: str) -> dict[str, Any]:
    """An AMC row placed on an explicit dataset (for the multi-dataset selector tests)."""
    row = _amc_row(result_id, model_name, "linear_probe", accuracy, "self_reported")
    row["dataset"] = {"name": dataset}
    return row


def test_multi_dataset_task_renders_dataset_selector_and_keeps_all_groups(
    tmp_path: Path,
) -> None:
    """A task spanning >1 dataset gets a segmented dataset selector; every dataset's .group
    section is present and carries data-dataset. Progressive enhancement: the selector bar is
    CSS-hidden until body.js-on and NO group is hidden by default CSS (JS-off shows all)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(results / "amc" / "a.json", _amc_row_on("row-a", "iqfm", 0.71, "radioml_2016_10a"))
    _write(results / "amc" / "b.json", _amc_row_on("row-b", "mcldnn", 0.66, "radioml_2018_01a"))
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")

    assert 'class="dataset-selector"' in amc
    assert 'data-dataset="radioml_2016_10a"' in amc
    assert 'data-dataset="radioml_2018_01a"' in amc
    # Both dataset groups are rendered (each is its own .group with data-dataset).
    assert 'section class="group" data-dataset="radioml_2016_10a"' in amc
    assert 'section class="group" data-dataset="radioml_2018_01a"' in amc
    # The first dataset button is the default-selected one.
    assert (
        '<button type="button" class="board-seg board-seg-active" '
        'data-dataset="radioml_2016_10a" aria-pressed="true">' in amc
    )
    # Progressive enhancement + no-JS degradation.
    css = generate.render_styles()
    assert ".dataset-selector, .foundation-controls { display: none; }" in css
    assert "body.js-on .dataset-selector { display: flex; }" in css
    assert ".group { display: none" not in css  # a group is NEVER hidden by default CSS


def test_single_dataset_task_renders_no_dataset_selector(tmp_path: Path) -> None:
    """A single-dataset task page renders NO dataset selector (unchanged, clean)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)  # amc rows here all share radioml_2016_10a
    generate.build_site(results, out)
    amc = (out / "amc.html").read_text(encoding="utf-8")
    sei = (out / "sei.html").read_text(encoding="utf-8")
    assert 'class="dataset-selector"' not in amc
    assert 'class="dataset-selector"' not in sei


def test_foundation_page_has_task_selector_with_all_option(tmp_path: Path) -> None:
    """foundation.html carries a task selector: an 'All' option plus one button per task that
    has foundation results (each targeting a .foundation-task via data-task). All is default."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "f.json",
        _foundation_row("row-fa", "amc", "iqfm-base", "linear_probe", 0.8),
    )
    _write(
        results / "sei" / "f.json",
        _foundation_row("row-fs", "sei", "iqfm-base", "linear_probe", 0.7),
    )
    generate.build_site(results, out)
    foundation = (out / "foundation.html").read_text(encoding="utf-8")

    assert 'class="foundation-controls"' in foundation
    assert (
        '<button type="button" class="board-seg board-seg-active" data-task="all" '
        'aria-pressed="true">All</button>' in foundation
    )
    # A selector button per task with foundation results (targets the .foundation-task section).
    assert '<button type="button" class="board-seg" data-task="amc"' in foundation
    assert '<button type="button" class="board-seg" data-task="sei"' in foundation
    # The task selector's own script shows only the picked section (no board sort/filter script).
    assert "getAttribute('data-task')" in foundation
    assert "sortTable" not in foundation


def test_foundation_page_include_baselines_toggle_and_reference(tmp_path: Path) -> None:
    """The Foundation page carries an 'Include baselines' toggle and, for a task with baseline
    rows, a labelled best-baseline REFERENCE (one per track) that is NOT merged into the FM
    ranking, plus a baseline-inclusive Pareto scatter. The reference is VISIBLE with JS off."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _write(
        results / "amc" / "base.json",
        _sized_amc_row("row-base", "mcldnn", 0.99, n_params=2_000_000, regime="from_scratch"),
    )
    row = _foundation_row("row-f", "amc", "iqfm-base", "linear_probe", 0.80)
    row["schema_version"] = "1.3.0"
    row["model"]["n_params"] = 90_000
    _write(results / "amc" / "f.json", row)
    generate.build_site(results, out)
    foundation = (out / "foundation.html").read_text(encoding="utf-8")

    # The toggle (a .board-toggle checkbox, like "Verified only").
    assert 'id="foundation-baselines"' in foundation
    assert ">Include baselines<" in foundation
    # A best-baseline reference block, clearly labelled a reference (not merged with the FMs).
    assert "data-baseline-ref" in foundation
    assert "best baseline" in foundation
    assert "reference, not ranked with the foundation models" in foundation
    # The best baseline is named in the reference (family chip "baseline" + its regime + score).
    section = foundation[foundation.index('foundation-task" data-task="amc"') :]
    ref = section.split("data-baseline-ref", 1)[1].split("</div>", 1)[0]
    assert "mcldnn" in ref
    assert 'class="chip chip-baseline"' in ref
    # ...but NOT in the FM ranking tbody (invariant: baselines never in the FM ranking/medals).
    tbody = section.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert "mcldnn" not in tbody
    # A baseline-inclusive Pareto scatter is present (efficiency/reference, spanning regimes).
    assert "data-baseline-pareto" in foundation
    # Progressive enhancement: the baseline surfaces render VISIBLE (no default is-hidden).
    assert 'class="baseline-reference" data-baseline-ref' in foundation
    assert (
        "is-hidden"
        not in foundation.split("data-baseline-ref", 1)[0].rsplit("baseline-reference", 1)[-1]
    )


def test_best_baseline_per_track_picks_top_specialist(tmp_path: Path) -> None:
    """_best_baseline_per_track returns the best specialist per track (top of _sort_rows),
    one entry per distinct track, ordered with the default 'all' bucket first."""
    rows = [
        _sei_row("r1", "weak", "closed_set", 0.50),
        _sei_row("r2", "strong", "closed_set", 0.94),
        _sei_row("r3", "solo", "cross_receiver", 0.60),
    ]
    best = generate._best_baseline_per_track(rows)
    tracks = [t for t, _r in best]
    assert tracks == ["closed_set", "cross_receiver"]
    by_track = {t: r for t, r in best}
    assert by_track["closed_set"]["model"]["name"] == "strong"  # top specialist of the track
    assert by_track["cross_receiver"]["model"]["name"] == "solo"


# --- Pareto rendering fixes ---
def test_axis_ticks_are_short() -> None:
    """Axis tick labels use <=3 significant figures (no 6-decimal overflow like 0.654866)."""
    assert generate._fmt_axis(0.6548659) == "0.655"
    assert generate._fmt_axis(0.13) == "0.13"
    assert generate._fmt_axis(5.0) == "5"


def test_foundation_scatter_points_get_a_tooltip(tmp_path: Path) -> None:
    """foundation.html wires the shared tooltip to its scatter points (no board script)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    _write(
        results / "amc" / "found.json",
        _foundation_row("row-f", "amc", "iqfm-base", "linear_probe", 0.8),
    )
    generate.build_site(results, out)
    foundation_html = (out / "foundation.html").read_text(encoding="utf-8")
    assert "chart-tooltip" in foundation_html
    assert ".pt[data-model], .barplot-bar[data-model]" in foundation_html


# --- per-dataset Pareto ---
def test_pareto_scatter_is_per_dataset_and_filterable(tmp_path: Path) -> None:
    """A multi-dataset task renders one size/perf scatter PER dataset, each tagged data-dataset."""
    import re

    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    # give the amc task two datasets, each with >=2 sized models (so both scatters render)
    for ds in ("radioml_2016_10a", "radioml_2018_01a"):
        for i in range(2):
            row = _amc_row(
                f"{ds}-{i}", f"{ds}-m{i}", "from_scratch", 0.7 + i * 0.05, "self_reported"
            )
            row["dataset"] = {"name": ds}
            row["model"]["n_params"] = 1000 * (i + 1)
            _write(results / "amc" / f"pareto_{ds}_{i}.json", row)
    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    tagged = re.findall(r'<section class="efficiency-section" data-dataset="([^"]+)"', amc_html)
    assert set(tagged) >= {"radioml_2016_10a", "radioml_2018_01a"}  # one per dataset, tagged
    assert "section.efficiency-section[data-dataset]" in amc_html  # selector filters them
