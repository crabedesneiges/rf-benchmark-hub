"""WP-50 tests -- static leaderboard site generator.

Exercises ``leaderboard.site.generate.build_site`` end to end on synthetic, in-tree
result fixtures written to ``tmp_path`` (no network, no heavy deps -- only ``jsonschema``,
a hard dep of the harness, is used, and only transitively via the generator). The tests
assert the acceptance criteria for WP-50:

* an ``index.html`` and one ``<task>.html`` page are produced,
* the expected model rows appear on the task page,
* rows are ordered by the task's PRIMARY metric (descending) within a regime,
* the maintainer ``verified`` badge is rendered (vs ``self-reported``),
* a comparison column NEVER mixes two regimes (D5): every row inside a
  ``<table data-regime="R">`` really has regime ``R``,
* a comparison table NEVER mixes two tracks: two rows sharing a regime but declaring
  different ``split.track`` land in SEPARATE tables (SEI / detection tracks are reported
  separately),
* invalid result JSONs are skipped (not rendered) instead of crashing the build.

The generator module lives at ``leaderboard/site/generate.py``; it is loaded by file
path so the test does not depend on ``leaderboard`` being an importable package.
"""

from __future__ import annotations

import importlib.util
import json
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


def _sei_row(result_id: str, model_name: str, track: str, rank1: float) -> dict[str, Any]:
    """Build a schema-valid SEI result row (primary metric = rank1_accuracy)."""
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
        "metrics": {"primary": "rank1_accuracy", "values": {"rank1_accuracy": rank1}},
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
class _RegimeTableParser(HTMLParser):
    """Collect, per ``<table data-track=T data-regime=R>``, the (model, regime) body rows.

    The Model cell is the 1st ``<td>`` and the Regime cell the 3rd (Model, Params, Regime,
    ...), captured in document order. Tables are keyed by the ``(track, regime)`` pair so
    two same-regime/different-track tables never collide. Lets a test assert (a) that every
    row inside a table really carries that table's regime -- the comparison column never
    blends regimes (D5) -- (b) that different tracks land in separate tables (SEI /
    detection reported separately), and (c) the within-table row order (primary metric).
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_key: tuple[str, str] | None = None
        self._in_body = False
        self._td_index = -1
        self._capture = False
        self._buffer: list[str] = []
        self._row_model: str = ""
        # (track, regime) -> list of per-row (model, regime) cell texts, in document order
        self.rows_by_table: dict[tuple[str, str], list[tuple[str, str]]] = {}

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
        elif tag == "tr" and self._in_body:
            self._td_index = 0
            self._row_model = ""
        elif tag == "td" and self._in_body and self._td_index >= 0:
            self._td_index += 1
            # Capture the Model (1st) and Regime (3rd) cells.
            if self._td_index in (1, 3):
                self._capture = True
                self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._capture:
            text = "".join(self._buffer).strip()
            if self._td_index == 1:
                self._row_model = text
            elif self._td_index == 3 and self._current_key is not None:
                self.rows_by_table[self._current_key].append((self._row_model, text))
            self._capture = False
        elif tag == "tbody":
            self._in_body = False
        elif tag == "table":
            self._current_key = None

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)


def _tables_by_track_regime(html_text: str) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Return rendered tables keyed by their ``(track, regime)`` pair."""
    parser = _RegimeTableParser()
    parser.feed(html_text)
    return parser.rows_by_table


def _regime_tables(html_text: str) -> dict[str, list[tuple[str, str]]]:
    """Return rendered tables keyed by regime, assuming a single track per regime.

    A convenience wrapper for the single-track AMC fixtures where each regime appears in
    exactly one track; asserts that assumption so multi-track pages use the richer
    ``_tables_by_track_regime`` view instead.
    """
    by_regime: dict[str, list[tuple[str, str]]] = {}
    for (_track, regime), cells in _tables_by_track_regime(html_text).items():
        assert regime not in by_regime, f"regime {regime} spans multiple tracks; use pair view"
        by_regime[regime] = cells
    return by_regime


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


def test_rows_sorted_by_primary_descending_within_regime(tmp_path: Path) -> None:
    """Within a regime, rows are ordered by the primary metric (descending)."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    tables = _regime_tables(amc_html)
    # Within the linear_probe table, iqfm (0.71) must be rendered before mcldnn (0.52).
    lp_models = [model for model, _ in tables["linear_probe"]]
    assert lp_models == ["iqfm", "mcldnn"]

    # And the ordering is reflected by the sort helper directly.
    rows = generate.load_results(results)
    lp = [r for r in rows if r["task"]["name"] == "amc" and r["regime"]["name"] == "linear_probe"]
    ordered = generate._sort_rows(lp)
    primaries = [generate._primary_value(r) for r in ordered]
    assert primaries == sorted(primaries, reverse=True)
    assert primaries[0] > primaries[-1]


def test_verified_badge_rendered(tmp_path: Path) -> None:
    """Both verification states surface as distinct badges."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "badge-verified" in amc_html
    assert ">verified<" in amc_html
    assert "badge-self" in amc_html
    assert "self-reported" in amc_html


def test_comparison_column_never_mixes_regimes(tmp_path: Path) -> None:
    """D5: every row inside a regime-tagged table carries exactly that regime.

    The AMC page has a linear_probe table (2 rows) and a full_finetune table (1 row);
    neither table may contain a row from the other regime.
    """
    results = tmp_path / "results"
    out = tmp_path / "site"
    _make_results_tree(results)
    generate.build_site(results, out)

    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    tables = _regime_tables(amc_html)

    assert set(tables) == {"linear_probe", "full_finetune"}
    # No table may mix regimes: every row's regime cell equals the table's data-regime tag.
    for regime, cells in tables.items():
        assert cells, f"table {regime} rendered no rows"
        regimes_in_table = {cell_regime for _, cell_regime in cells}
        assert regimes_in_table == {regime}, f"table {regime} mixed regimes: {cells}"
    # The linear_probe table holds exactly its two rows; full_finetune holds its one.
    assert len(tables["linear_probe"]) == 2
    assert len(tables["full_finetune"]) == 1


def test_same_regime_different_track_split_into_separate_tables(tmp_path: Path) -> None:
    """Two rows sharing a regime but declaring different tracks never share a table.

    SEI reports closed_set / cross_receiver / cross_day separately (docs/
    EVALUATION_PROTOCOL.md); both fixture rows use regime ``from_scratch`` but different
    ``split.track``, so they must land in two distinct tables -- their primary-metric
    column must never compare across tracks.
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
    assert tables[("closed_set", "from_scratch")] == [("wisig-cnn", "from_scratch")]
    assert tables[("cross_receiver", "from_scratch")] == [("wisig-cnn", "from_scratch")]
    # The page separates the tracks with labelled sections.
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
    assert tables[("all", "linear_probe")] == [("iqfm", "linear_probe")]
    # A single-track task stays label-free: no visible "Track:" heading.
    assert "Track:" not in amc_html


def test_few_shot_regime_labels_carry_k(tmp_path: Path) -> None:
    """A few_shot row is labelled with its k, and its table is tagged data-regime=few_shot."""
    results = tmp_path / "results"
    out = tmp_path / "site"
    row = _amc_row("row-fs", "iqfm", "few_shot", 0.40, "self_reported", family="foundation")
    row["regime"]["k_shot"] = 5
    _write(results / "amc" / "fs.json", row)

    generate.build_site(results, out)
    amc_html = (out / "amc.html").read_text(encoding="utf-8")
    assert "few_shot(k=5)" in amc_html
    assert 'data-regime="few_shot"' in amc_html


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


def test_load_results_only_returns_valid_rows(tmp_path: Path) -> None:
    """load_results validates against the frozen schema and drops invalid rows."""
    results = tmp_path / "results"
    _make_results_tree(results)
    rows = generate.load_results(results)
    # 4 valid fixtures.
    assert len(rows) == 4
    assert {r["result_id"] for r in rows} == {"row-a", "row-b", "row-c", "row-d"}
