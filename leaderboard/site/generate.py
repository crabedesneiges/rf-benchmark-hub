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
disabled) plus a tiny head theme-boot on every page (the dark/light toggle; hidden with JS
off, when the OS scheme simply applies) -- pages without a board carry no other script.

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
import ast
import html
import json
import math
import re
import sys
import zlib
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote

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
    "snr_estimation": "SNR estimation",
}

#: Sensible fixed order for the known tasks; unknown tasks sort alphabetically AFTER these.
TASK_ORDER: tuple[str, ...] = (
    "amc",
    "sei",
    "wideband_detection",
    "spectrum_sensing",
    "interference_id",
    "protocol_tech_id",
    "snr_estimation",
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

#: Metric keys where a LOWER value is better (regression error metrics), keyed by the primary
#: metric name. Every classification metric on the board is higher-is-better, so this set is
#: empty for them and the default DESC ranking is unchanged; ``snr_estimation``'s ``rmse_db`` /
#: ``mae_db`` (dB error, 0 == perfect) rank ASCENDING and their score bar fills inversely.
#: Direction is a property of the METRIC name, not the task, so a shared metric ranks the same
#: way wherever it appears.
LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({"rmse_db", "mae_db"})

#: Contamination-badge text/CSS-class per ``pretraining.overlap_with_eval`` value (schema
#: 1.2.0). Rendered only for rows that DECLARE a ``pretraining`` block (foundation models with
#: disclosed provenance); rows without it render no contamination badge and are byte-identical
#: to before. ``none`` = disjoint by construction (green), ``unknown`` = not audited (amber),
#: ``confirmed`` = known overlap (red).
_OVERLAP_BADGE: dict[str, tuple[str, str]] = {
    "none": ("clean", "badge-overlap-none"),
    "unknown": ("overlap unknown", "badge-overlap-unknown"),
    "confirmed": ("contaminated", "badge-overlap-confirmed"),
}

#: Short (<=6-word) glosses keyed by verification status, used by the tier legend. The visible
#: label text still comes from ``_BADGE`` (never hardcoded here); this only adds a one-line hint.
_BADGE_GLOSS: dict[str, str] = {
    "verified": "maintainer re-ran the harness",
    "self_reported": "author-submitted, not re-run",
    "from_paper": "cited from the paper, split confirmed",
    "from_paper_uncertain": "cited; split/protocol not confirmed",
}

#: Short glosses keyed by model family, used by the tier legend (labels come from ``_FAMILY_CHIP``).
_FAMILY_GLOSS: dict[str, str] = {
    "baseline": "task-specific specialised model",
    "foundation": "pretrained, then adapted",
}

#: Short glosses keyed by ``pretraining.overlap_with_eval``, used by the tier legend (labels come
#: from ``_OVERLAP_BADGE``).
_OVERLAP_GLOSS: dict[str, str] = {
    "none": "eval data disjoint from pretraining",
    "unknown": "overlap not audited",
    "confirmed": "known pretrain/eval overlap",
}


def _is_lower_better(metric_key: str) -> bool:
    """Whether ``metric_key`` ranks ascending (lower == better), e.g. a regression error."""
    return metric_key in LOWER_IS_BETTER_METRICS


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
        (
            "rmse_db",
            "Root-mean-square error in dB for SNR estimation (regression): the square root of "
            "the mean squared gap between the predicted and true SNR over the full SNR range. "
            "Primary snr_estimation metric; lower is better (0 dB is perfect), and it penalises "
            "large errors (e.g. failures at low SNR) more than MAE.",
            False,
        ),
        (
            "mae_db",
            "Mean absolute error in dB for SNR estimation: the average absolute gap between the "
            "predicted and true SNR. Secondary snr_estimation metric; lower is better.",
            False,
        ),
        (
            "n_params",
            "Total trainable parameter count of the evaluated model (including any adapter head) "
            "-- its capacity. Shown in the Size column; a smaller model at equal accuracy is the "
            "cheaper, more deployable choice, so lower is the frugal direction.",
            False,
        ),
        (
            "n_flops",
            "FLOPs (or MACs) for a single forward pass at the task's fixed input size -- a "
            "hardware-independent compute proxy that is deterministic and comparable across "
            "GPUs/CPUs. When declared it drives the Size column's sub-line and the size axis of "
            "the size/perf Pareto scatter; lower means less compute per inference.",
            False,
        ),
    ),
    "size_compute": (
        "Two model-cost signals sit beside every score. n_params is the model's capacity (its "
        "total trainable parameter count, adapter head included) -- more parameters can fit more, "
        "but cost more memory and, roughly, more compute. n_flops (or MACs) counts the arithmetic "
        "of a single forward pass at the task's FIXED input size: it is a HARDWARE-INDEPENDENT "
        "compute proxy -- deterministic and directly comparable across GPUs, CPUs and runs, "
        "because it never touches a clock. inference_latency_ms is the measured complement: an "
        "actual wall-clock time, so it is faster to read but HARDWARE-SPECIFIC (a GB200 and a "
        "laptop CPU report wildly different numbers for the same model), which is why a latency "
        "figure must ALWAYS be reported with the hardware it was measured on. On the board, "
        "n_params always shows in the Size column, n_flops adds a muted FLOPs sub-line when "
        "declared, and the size/perf Pareto scatter plots accuracy against size (FLOPs when any "
        "row has them, else parameters) on a log axis -- an efficiency view spanning regimes, not "
        "a ranking."
    ),
    "contamination": (
        "Foundation-model rows may disclose what their backbone was pretrained on and whether "
        "that data overlaps the evaluation split (schema field pretraining.overlap_with_eval). "
        "The board surfaces this as a contamination badge next to the model name: clean means "
        "the pretraining data is disjoint from the eval split by construction; overlap unknown "
        "means the overlap was not audited; contaminated means a known overlap, so the score "
        "may be inflated by having seen (some of) the test signals during pretraining. Rows "
        "with no pretraining disclosure (every specialized baseline, which is trained from "
        "scratch on the task split) carry no badge at all. The badge never changes the ranking; "
        "it is a disclosure the reader weighs, especially for an FM evaluated on a dataset it "
        "was pretrained on (e.g. an RF FM pretrained on RadioML then probed on our RadioML "
        "SNR-estimation split)."
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

#: One-line site tagline, reused as the ``brand-tag`` line and the default ``<meta>`` /
#: Open Graph description for any page that does not supply a more specific one.
_SITE_TAGLINE: str = "Reproducible leaderboards for terrestrial RF machine learning"

#: A generic "code repository" glyph (NOT the GitHub Octocat, to avoid any trademark/asset
#: concern) used as the homepage's repo-link icon.
_REPO_ICON_SVG: str = (
    '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><polyline points="8 6 2 12 8 18"></polyline>'
    '<polyline points="16 6 22 12 16 18"></polyline></svg>'
)

#: Perceptually-spaced hue palette (hex), robust in both light and dark themes. A model's
#: colour is a DETERMINISTIC pick from this palette (``zlib.crc32`` of its name), so the SAME
#: model keeps the SAME hue on every chart and across every run -- colour therefore carries
#: meaning (identity), never a per-run accident. ``zlib.crc32`` (not the salted built-in
#: ``hash``) guarantees the mapping is stable between interpreter runs (idempotent output).
#: Every hue is calibrated to clear WCAG 1.4.11 non-text contrast (>= 3:1) against the ACTUAL
#: plot background ``--surface-2`` in BOTH themes (off-white in light, near-black in dark) --
#: curves/markers are informational graphic objects, so each must be perceivable on its own
#: ground, not merely distinguishable from its neighbours. ``tests/test_site.py`` recomputes
#: the ratio for every hue against both oklch ``--surface-2`` values and asserts >= 3.0.
_MODEL_PALETTE: tuple[str, ...] = (
    "#2f6bff",  # blue
    "#d1495b",  # rose
    "#0f9d8f",  # teal
    "#a45fc9",  # violet
    "#a86a09",  # amber
    "#2c7fb8",  # steel
    "#d55a4e",  # brick
    "#16a085",  # green
    "#d81b60",  # magenta
    "#5c6bc0",  # indigo
    "#00838f",  # cyan
    "#548a2b",  # lime
)

#: Family -> line dash pattern (a non-colour channel): ``baseline`` is a SOLID line, a
#: ``foundation`` model is DASHED, so the two families read apart even in greyscale.
_FAMILY_DASH: dict[str, str] = {
    "baseline": "none",
    "foundation": "7 4",
}

#: Marker shapes cycled by a series' index in its group (a THIRD, shape-based channel on top
#: of hue + dash, for colour-blind accessibility): circle, square, triangle, diamond.
_MARKER_SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "diamond")


def _model_hue(name: str) -> str:
    """Deterministic palette colour for a model name (stable across runs and charts).

    Uses ``zlib.crc32`` (NOT the salted built-in ``hash``) so the name -> colour mapping is
    identical on every interpreter run, keeping the generated site byte-for-byte reproducible.
    """
    return _MODEL_PALETTE[zlib.crc32(name.encode("utf-8")) % len(_MODEL_PALETTE)]


def _family_dash(family: str | None) -> str:
    """Line dash-array for a model family (baseline solid, foundation dashed), else solid."""
    return _FAMILY_DASH.get(family or "", "none")


def _marker_shape(index: int) -> str:
    """Marker shape cycled by a series' index within its group (3rd, non-colour channel)."""
    return _MARKER_SHAPES[index % len(_MARKER_SHAPES)]


def _marker_svg(shape: str, cx: float, cy: float, r: float, fill: str, extra: str = "") -> str:
    """Render one marker of ``shape`` centred at ``(cx, cy)`` with radius ``r`` (filled ``fill``).

    ``extra`` is appended verbatim to the element's attributes (used for the data-* hooks and
    the ``<title>`` child are handled by the caller). Falls back to a circle for an unknown shape.
    """
    if shape == "square":
        s = r * 1.7
        return f'<rect x="{cx - s / 2:.1f}" y="{cy - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{fill}"{extra}'  # noqa: E501
    if shape == "triangle":
        h = r * 1.9
        pts = f"{cx:.1f},{cy - h:.1f} {cx - h:.1f},{cy + h * 0.75:.1f} {cx + h:.1f},{cy + h * 0.75:.1f}"  # noqa: E501
        return f'<polygon points="{pts}" fill="{fill}"{extra}'
    if shape == "diamond":
        d = r * 1.7
        pts = f"{cx:.1f},{cy - d:.1f} {cx + d:.1f},{cy:.1f} {cx:.1f},{cy + d:.1f} {cx - d:.1f},{cy:.1f}"  # noqa: E501
        return f'<polygon points="{pts}" fill="{fill}"{extra}'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{extra}'


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


def _dataset_name(row: dict[str, Any]) -> str:
    """Return the row's dataset id (``dataset.name``).

    Part of the group key so two DATASETS of the same task never share a leaderboard table/plot
    (e.g. AMC on RadioML 2016.10a (11-class) vs 2018.01a (24-class), or SEI on WiSig vs ORACLE) --
    comparing accuracies across different datasets/class-counts would be as meaningless as mixing
    two regimes, so the board keeps them in separate groups.
    """
    return str(row["dataset"]["name"])


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


def _foundation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to rows whose declared model family is ``foundation`` (baselines dropped)."""
    return [r for r in rows if _family(r) == "foundation"]


# --------------------------------------------------------------------------------------------------
# Ordering helpers
# --------------------------------------------------------------------------------------------------
def _rank_value(row: dict[str, Any]) -> float:
    """Signed primary value for a DESCENDING sort: negate higher-is-better, keep lower-is-better.

    Most metrics are higher-is-better, so ``-value`` puts the best first; a regression error
    metric (``rmse_db``/``mae_db``) is lower-is-better, so ``+value`` puts the smallest error
    first. All rows in a ranked group share one task/primary metric, so the direction is
    consistent within a table. Keeps the board's single ``sorted(...)`` call and its total
    order intact.
    """
    value = _primary_value(row)
    return value if _is_lower_better(_primary_key(row)) else -value


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by primary metric (best first), then verified-first, then model name.

    'Best first' is DESC for the classification metrics (all higher-is-better) and ASC for the
    regression error metrics (``rmse_db``/``mae_db``, lower-is-better) -- both expressed through
    :func:`_rank_value`. Verified rows break ties ahead of self-reported ones, and the model
    name is a final deterministic tiebreak.

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
            _rank_value(r),
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


def _regime_track_sort_key(rkt: tuple[str, int | None, str]) -> tuple[Any, int, Any]:
    """Order a ``(regime, k_shot, track)`` group key: regime (D5 order), then k_shot
    ascending, then track (default bucket first). Shared by :func:`render_task_page` and the
    Foundation Models page so both partition a task's rows into groups in the same order.
    """
    regime, k_shot, track = rkt
    shot_rank = k_shot if k_shot is not None else -1
    return (_regime_sort_key(regime), shot_rank, _track_sort_key(track))


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


def _fmt_flops(n_flops: object) -> str:
    """Format a FLOPs/MACs count compactly with an SI suffix (e.g. 90.1M, 1.2G).

    Mirrors :func:`_fmt_params`' human units but adds a Giga (``G``) tier, since a single
    forward pass is routinely billions of FLOPs. Returns an en-dash for a missing/non-integer
    value so a caller can fall back gracefully. The trailing ``FLOPs`` word is added by the
    caller (the sub-line), not here, so this stays a pure number formatter.
    """
    if not isinstance(n_flops, int):
        return "&ndash;"
    if n_flops >= 1_000_000_000:
        return f"{n_flops / 1_000_000_000:.1f}G"
    if n_flops >= 1_000_000:
        return f"{n_flops / 1_000_000:.1f}M"
    if n_flops >= 1_000:
        return f"{n_flops / 1_000:.0f}K"
    return str(n_flops)


def _model_n_params(row: dict[str, Any]) -> int | None:
    """Return the row's ``model.n_params`` when it is a non-negative int, else ``None``."""
    value = row["model"].get("n_params")
    return value if isinstance(value, int) and value >= 0 else None


def _model_n_flops(row: dict[str, Any]) -> int | None:
    """Return the row's ``model.n_flops`` compute proxy (schema 1.3.0) when present, else ``None``.

    ``n_flops`` is the hardware-independent FLOPs/MACs count for one forward pass at the task's
    fixed input size. Optional and absent on every current row -- returns ``None`` unless the
    field is a non-negative int, so size rendering degrades to ``n_params`` (or an en-dash).
    """
    value = row["model"].get("n_flops")
    return value if isinstance(value, int) and value >= 0 else None


def _size_sort_value(row: dict[str, Any]) -> int | None:
    """The numeric value the Size column sorts on: ``n_flops`` if present, else ``n_params``.

    A compute proxy (FLOPs) is preferred over raw capacity (params) when available, so the
    generic board sort on the Size column orders by compute where declared and by parameter
    count otherwise. ``None`` when the row declares neither (its cell sorts last / as -inf).
    """
    flops = _model_n_flops(row)
    return flops if flops is not None else _model_n_params(row)


def _render_size_cell(row: dict[str, Any]) -> str:
    """Render the sortable Size ``<td>``: params (main line) + FLOPs (muted sub-line) if any.

    The cell always shows the parameter count via :func:`_fmt_params` (an en-dash when absent);
    when ``model.n_flops`` is declared a muted second line shows the FLOPs (e.g. ``1.2G FLOPs``).
    The ``data-value`` attribute carries the numeric sort key (:func:`_size_sort_value`) -- FLOPs
    when present, else params -- so the generic board sort (``data-sort="num"`` reading
    ``data-value``) orders the column by compute/size. When neither is present the cell shows a
    muted en-dash and carries NO ``data-value`` (it sorts last, like an empty metric cell).
    """
    n_params = _model_n_params(row)
    n_flops = _model_n_flops(row)
    sort_value = _size_sort_value(row)
    value_attr = f' data-value="{sort_value}"' if sort_value is not None else ""
    if n_params is None and n_flops is None:
        return f'<td class="num size"{value_attr}><span class="size-params">&ndash;</span></td>'
    params_line = f'<span class="size-params">{_fmt_params(n_params)}</span>'
    flops_line = (
        f'<span class="size-flops">{_fmt_flops(n_flops)} FLOPs</span>'
        if n_flops is not None
        else ""
    )
    return f'<td class="num size"{value_attr}>{params_line}{flops_line}</td>'


def _fmt_axis(value: float) -> str:
    """Format an axis tick label (drops a trailing ``.0`` for integer-valued ticks)."""
    if value == int(value):
        return str(int(value))
    return f"{value:.3g}"


def _axis_titles(curve_name: str) -> tuple[str | None, str | None]:
    """Derive ``(y_title, x_title)`` GENERICALLY from a ``"<y>_vs_<x>"`` curve name.

    ``accuracy_vs_snr`` -> ``("accuracy", "snr")``; a name with no ``_vs_`` yields
    ``(None, None)`` (no axis title). Nothing task/metric-specific is hardcoded -- the labels
    are just the two halves of the name split on the first ``_vs_``.
    """
    parts = curve_name.split("_vs_", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return (None, None)
    return (parts[0], parts[1])


# --------------------------------------------------------------------------------------------------
# Inline-SVG line plot (stdlib only -- polylines computed from the curve points)
# --------------------------------------------------------------------------------------------------
def _render_bar(value: float, vmax: float, *, lower_is_better: bool = False) -> str:
    """Render a tiny horizontal bar (0..vmax) visualising a primary score.

    For a higher-is-better metric the bar fills ``value / vmax`` (bigger score = fuller bar).
    For a lower-is-better metric (a regression error) the fill is inverted to ``1 - value/vmax``
    so the BEST row (smallest error) still shows the fullest bar -- the visual "more is better"
    reading holds for both metric directions.
    """
    if vmax <= 0:
        frac = 0.0
    else:
        ratio = max(0.0, min(1.0, value / vmax))
        frac = (1.0 - ratio) if lower_is_better else ratio
    pct = f"{frac * 100:.1f}"
    return (
        '<span class="bar" role="img" '
        f'aria-label="{_esc(_fmt_metric(value))}">'
        f'<span class="bar-fill" style="width:{pct}%"></span></span>'
    )


def _render_curve_plot(
    curve_name: str,
    series: list[tuple[str, str | None, list[dict[str, Any]]]],
) -> str:
    """Render one inline-SVG line plot overlaying every model's curve in a group.

    ``series`` is a list of ``(model_name, family, points)`` where each point is a ``{x, y}``
    dict (optionally ``y_low``/``y_high`` for a per-point CI). A model's colour is its stable
    hue (:func:`_model_hue`), its line style keys off the family (baseline solid / foundation
    dashed), and a marker shape cycled by series index gives a third, non-colour channel. Axis
    titles are derived generically from a ``"<y>_vs_<x>"`` curve name. Each series is wrapped in
    a ``<g class="series" data-series>`` so the inline legend buttons can toggle it; every point
    carries data-* hooks + a native ``<title>`` so tooltips work with OR without JS. Everything
    is computed here -- no JS drawing, no external chart library.
    """
    # Collect the global x/y ranges across all series (including any uncertainty band bounds).
    xs = [float(p["x"]) for _, _fam, pts in series for p in pts]
    ys: list[float] = []
    for _, _fam, pts in series:
        for p in pts:
            ys.append(float(p["y"]))
            if "y_low" in p:
                ys.append(float(p["y_low"]))
            if "y_high" in p:
                ys.append(float(p["y_high"]))
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
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_title, x_title = _axis_titles(curve_name)

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    parts: list[str] = [
        # role="group" (not "img"): an img exposes an OPAQUE subtree, which would hide the
        # focusable per-point children (their aria-labels would never reach the AT). A group keeps
        # the aria-label as the accessible name while leaving the points explorable by keyboard.
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="group" '
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
    # Axis titles, derived generically from the "<y>_vs_<x>" curve name (omitted otherwise).
    if x_title:
        parts.append(
            f'<text class="axis-title" x="{pad_l + plot_w / 2:.1f}" '
            f'y="{height - 6:.1f}" text-anchor="middle">{_esc(x_title)}</text>'
        )
    if y_title:
        ty = pad_t + plot_h / 2
        parts.append(
            f'<text class="axis-title" x="14" y="{ty:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 14 {ty:.1f})">{_esc(y_title)}</text>'
        )

    def _ci_text(p: dict[str, Any]) -> str:
        if "y_low" in p and "y_high" in p:
            return f" (CI [{_fmt_metric(float(p['y_low']))}, {_fmt_metric(float(p['y_high']))}])"
        return ""

    # Series polylines.
    legend_items: list[str] = []
    for idx, (model_name, family, pts) in enumerate(series):
        color = _model_hue(model_name)
        dash = _family_dash(family)
        shape = _marker_shape(idx)
        ordered = sorted(pts, key=lambda p: float(p["x"]))
        dash_attr = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        series_parts: list[str] = [f'<g class="series" data-series="{_esc(model_name)}">']
        # Uncertainty band: a shaded envelope from y_high (left->right) back along y_low
        # (right->left), drawn BEHIND the line, when every point carries a per-point CI.
        if ordered and all("y_low" in p and "y_high" in p for p in ordered):
            up = " ".join(f"{sx(float(p['x'])):.1f},{sy(float(p['y_high'])):.1f}" for p in ordered)
            down = " ".join(
                f"{sx(float(p['x'])):.1f},{sy(float(p['y_low'])):.1f}" for p in reversed(ordered)
            )
            series_parts.append(
                f'<polygon class="ci-band" fill="{color}" fill-opacity="0.14" '
                f'stroke="none" points="{up} {down}"/>'
            )
        coords = " ".join(f"{sx(float(p['x'])):.1f},{sy(float(p['y'])):.1f}" for p in ordered)
        series_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr} points="{coords}"/>'
        )
        # Point markers: hoverable/focusable, with data-* hooks + a native <title> fallback.
        for p in ordered:
            px, py = float(p["x"]), float(p["y"])
            cx, cy = sx(px), sy(py)
            ci_attr = ""
            if "y_low" in p and "y_high" in p:
                ci_attr = (
                    f' data-ci-low="{_fmt_metric(float(p["y_low"]))}"'
                    f' data-ci-high="{_fmt_metric(float(p["y_high"]))}"'
                )
            title = _esc(f"{model_name}: {_fmt_axis(px)} → {_fmt_metric(py)}{_ci_text(p)}")
            extra = (
                f' class="pt" data-model="{_esc(model_name)}" data-x="{_fmt_axis(px)}"'
                f' data-y="{_fmt_metric(py)}"{ci_attr} tabindex="0" role="img"'
                f' aria-label="{title}">'
                f"<title>{title}</title>"
            )
            elem = _marker_svg(shape, cx, cy, 2.8, color, extra)
            close = (
                "</rect>"
                if shape == "square"
                else ("</circle>" if shape == "circle" else "</polygon>")
            )
            series_parts.append(elem + close)
        series_parts.append("</g>")
        parts.append("".join(series_parts))
        legend_items.append(
            '<button type="button" class="legend-item" '
            f'data-series="{_esc(model_name)}" aria-pressed="false">'
            f'<svg class="legend-swatch" viewBox="0 0 24 10" aria-hidden="true">'
            f'<line x1="1" y1="5" x2="23" y2="5" stroke="{color}" stroke-width="2"'
            f"{dash_attr}/>"
            f"{_marker_svg(shape, 12, 5, 2.6, color, '/>')}"
            "</svg>"
            f"<span>{_esc(model_name)}</span></button>"
        )

    parts.append("</svg>")
    legend = f'<div class="legend">{"".join(legend_items)}</div>'
    return (
        '<figure class="plot-figure">'
        f'<figcaption class="plot-title">{_esc(curve_name)}</figcaption>'
        f'{"".join(parts)}{legend}'
        "</figure>"
    )


def _svg_line(cls: str, x1: float, y1: float, x2: float, y2: float) -> str:
    """One ``<line>`` SVG element (helper to keep the plot builders short)."""
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


def _metric_uncertainty(row: dict[str, Any], key: str) -> tuple[float, float] | None:
    """Return ``(ci_low, ci_high)`` for ``key`` from ``metrics.uncertainty`` (1.2.0), else None."""
    unc = row["metrics"].get("uncertainty")
    if not isinstance(unc, dict):
        return None
    entry = unc.get(key)
    if isinstance(entry, dict) and "ci_low" in entry and "ci_high" in entry:
        return float(entry["ci_low"]), float(entry["ci_high"])
    return None


class ParetoPoint(NamedTuple):
    """One point for the size/perf Pareto analysis: cost ``x`` (lower=better) + perf ``y``.

    ``label`` (a model name) rides along so :func:`_pareto_frontier` can return the identity of
    each non-dominated point without the caller re-matching on floats.
    """

    x: float  # cost proxy (model size / FLOPs) -- LOWER is better
    y: float  # performance (the group's primary metric)
    label: str  # model name (carried through, never used in dominance)


def _pareto_frontier(
    points: list[ParetoPoint], *, y_lower_is_better: bool = False
) -> list[ParetoPoint]:
    """Return the non-dominated (Pareto-optimal) subset of ``points``, sorted by ``x`` ascending.

    Cost ``x`` is ALWAYS lower-is-better (a smaller model / fewer FLOPs is cheaper). The
    performance direction is set by ``y_lower_is_better`` -- ``False`` (default) treats a HIGHER
    ``y`` as better (accuracy-like metrics), ``True`` treats a LOWER ``y`` as better (a
    regression error such as ``rmse_db``).

    A point ``p`` is DOMINATED iff some other point ``q`` is at-least-as-good on BOTH axes
    (``q.x <= p.x`` and ``q`` at-least-as-good in ``y``) with at least ONE strict inequality;
    the frontier is every point that is NOT dominated. Duplicate ``(x, y)`` points do not
    dominate one another (no strict inequality), so exact ties are all kept. Pure and
    side-effect-free; sorted by ``x`` so the caller can draw a monotone staircase.
    """

    def _y_at_least_as_good(a: float, b: float) -> bool:
        """Is ``a`` at least as good as ``b`` in the perf axis (per the direction flag)?"""
        return a <= b if y_lower_is_better else a >= b

    def _y_strictly_better(a: float, b: float) -> bool:
        return a < b if y_lower_is_better else a > b

    frontier: list[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_or_equal = q.x <= p.x and _y_at_least_as_good(q.y, p.y)
            strictly_better = q.x < p.x or _y_strictly_better(q.y, p.y)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda pt: pt.x)


def _render_bar_chart(metric_key: str, rows: list[dict[str, Any]]) -> str:
    """Render an inline-SVG bar chart for one scalar metric: X = model, Y = performance.

    The per-model view a scalar metric has no 2-D curve for. Bars are ordered best-first
    (ascending for lower-is-better metrics, else descending); when a row carries
    ``metrics.uncertainty[metric_key]`` (schema 1.2.0) a capped **error bar** spans its
    ``[ci_low, ci_high]``, so the confidence interval is shown right on the plot. Pure-stdlib
    SVG -- no JS, no chart library (mirrors :func:`_render_curve_plot`).
    """
    lower = _is_lower_better(metric_key)
    entries: list[tuple[str, float, tuple[float, float] | None, str | None]] = []
    for row in rows:
        values = _scalar_values(row)
        if metric_key not in values:
            continue
        entries.append(
            (
                str(row["model"]["name"]),
                float(values[metric_key]),
                _metric_uncertainty(row, metric_key),
                _family(row),
            )
        )
    if not entries:
        return ""
    entries.sort(key=lambda e: e[1], reverse=not lower)  # best model first

    hi_vals = [e[1] for e in entries] + [ci[1] for e in entries if (ci := e[2]) is not None]
    lo_vals = [e[1] for e in entries] + [ci[0] for e in entries if (ci := e[2]) is not None]
    ymax = max(hi_vals) * 1.08
    ymin = min(0.0, min(lo_vals))
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 720, 320
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 80  # extra bottom room for rotated model labels
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    slot = plot_w / len(entries)
    bar_w = min(52.0, slot * 0.62)

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    parts: list[str] = [
        # role="group" (not "img"): keeps the focusable per-bar children exposed to the AT while
        # the aria-label names the chart (see the line-plot container for the full rationale).
        f'<svg class="plot barplot" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="{_esc(metric_key)} by model bar chart" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
    ]
    for i in range(6):
        yval = ymin + (ymax - ymin) * i / 5
        y = sy(yval)
        parts.append(_svg_line("grid", pad_l, y, pad_l + plot_w, y))
        parts.append(
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_esc(_fmt_axis(yval))}</text>'
        )
    baseline = sy(max(ymin, 0.0))
    for idx, (name, val, ci, family) in enumerate(entries):
        cx = pad_l + slot * (idx + 0.5)
        top = sy(val)
        color = _model_hue(name)
        ci_attr = ""
        ci_txt = ""
        if ci is not None:
            ci_attr = f' data-ci-low="{_fmt_metric(ci[0])}" data-ci-high="{_fmt_metric(ci[1])}"'
            ci_txt = f" (CI [{_fmt_metric(ci[0])}, {_fmt_metric(ci[1])}])"
        title = _esc(f"{name}: {metric_key} = {_fmt_metric(val)}{ci_txt}")
        fam_attr = f' data-family="{_esc(family)}"' if family else ""
        parts.append(
            f'<rect class="barplot-bar" x="{cx - bar_w / 2:.1f}" y="{min(top, baseline):.1f}" '
            f'width="{bar_w:.1f}" height="{abs(baseline - top):.1f}" rx="2" '
            f'style="fill:{color}" data-model="{_esc(name)}"{fam_attr} '
            f'data-metric="{_esc(metric_key)}" data-value="{_fmt_metric(val)}"{ci_attr} '
            f'tabindex="0" role="img" aria-label="{title}"><title>{title}</title></rect>'
        )
        parts.append(
            f'<text class="bar-val" x="{cx:.1f}" y="{top - 5:.1f}" '
            f'text-anchor="middle">{_esc(_fmt_metric(val))}</text>'
        )
        if ci is not None and ci[1] > ci[0]:
            ytop, ybot = sy(ci[1]), sy(ci[0])
            cap = min(6.0, bar_w / 3)
            parts.append(_svg_line("errbar", cx, ytop, cx, ybot))
            parts.append(_svg_line("errbar", cx - cap, ytop, cx + cap, ytop))
            parts.append(_svg_line("errbar", cx - cap, ybot, cx + cap, ybot))
        label_y = pad_t + plot_h + 14
        parts.append(
            f'<text class="bar-label" x="{cx:.1f}" y="{label_y:.1f}" text-anchor="end" '
            f'transform="rotate(-30 {cx:.1f} {label_y:.1f})">{_esc(name)}</text>'
        )
    parts.append(_svg_line("axis", pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h))
    parts.append(_svg_line("axis", pad_l, pad_t, pad_l, pad_t + plot_h))
    # Y-axis title = the metric (X is simply the model, labelled per-bar below).
    ty = pad_t + plot_h / 2
    parts.append(
        f'<text class="axis-title" x="14" y="{ty:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {ty:.1f})">{_esc(metric_key)}</text>'
    )
    parts.append("</svg>")
    arrow = "&darr; lower is better" if lower else "&uarr; higher is better"
    has_ci = any(e[2] is not None for e in entries)
    ci_hint = (
        ' <span class="plot-dir">&middot; whiskers = confidence interval</span>' if has_ci else ""
    )
    return (
        '<figure class="plot-figure">'
        f'<figcaption class="plot-title">{_esc(metric_key)} by model '
        f'<span class="plot-dir">{arrow}</span>{ci_hint}</figcaption>'
        f'{"".join(parts)}'
        "</figure>"
    )


def _render_pareto_scatter(rows: list[dict[str, Any]]) -> str:
    """Render the size/perf Pareto scatter: X = model size (log), Y = the primary metric.

    THE headline efficiency view: one marker per model, X = a compute/size cost on a LOG scale
    (``model.n_flops`` if ANY row declares it, else ``model.n_params`` -- the axis title names
    which), Y = the group's primary metric (direction respects :func:`_is_lower_better`). Markers
    reuse the board's stable channels (:func:`_model_hue` + :func:`_marker_shape` +
    :func:`_family_dash`) so baseline vs foundation read apart, and carry the same ``data-*``
    tooltip hooks as the other charts. The Pareto FRONTIER (:func:`_pareto_frontier`: minimise
    size, optimise perf) is drawn as a staircase polyline.

    Rows with NO size data are dropped (a size-less point cannot be placed on a size axis); when
    any are dropped a caption note names how many (no silent truncation). The whole plot is
    SKIPPED (returns ``""``) when fewer than 2 points carry size -- a single point cannot show a
    tradeoff. This view intentionally SPANS regimes/tracks (an efficiency/reference view, not a
    ranking), which the caption states explicitly.
    """
    if not rows:
        return ""
    primary_key = _primary_key(rows[0])
    lower = _is_lower_better(primary_key)
    # X axis = FLOPs when ANY row declares it (a hardware-independent compute proxy), else params.
    use_flops = any(_model_n_flops(r) is not None for r in rows)

    def _size_of(row: dict[str, Any]) -> int | None:
        return _model_n_flops(row) if use_flops else _model_n_params(row)

    sized: list[tuple[str, int, float, str | None]] = []
    dropped = 0
    for row in rows:
        size = _size_of(row)
        # A size of 0 cannot sit on a log axis (log10(0) is undefined) -> treat it as unsized.
        if size is None or size <= 0:
            dropped += 1
            continue
        sized.append(
            (str(row["model"]["name"]), size, _primary_value(row), _family(row)),
        )
    if dropped:
        _warn(
            f"pareto scatter ({primary_key}): dropped {dropped} point(s) with no positive "
            f"{'n_flops' if use_flops else 'n_params'} size data"
        )
    if len(sized) < 2:
        return ""

    axis_label = "FLOPs (log scale)" if use_flops else "parameters (log scale)"
    frontier = _pareto_frontier(
        [ParetoPoint(x=float(s), y=v, label=n) for n, s, v, _fam in sized],
        y_lower_is_better=lower,
    )

    log_xs = [math.log10(s) for _n, s, _v, _f in sized]
    xmin, xmax = min(log_xs), max(log_xs)
    if xmax == xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    else:
        pad = (xmax - xmin) * 0.06
        xmin, xmax = xmin - pad, xmax + pad
    values = [v for _n, _s, v, _f in sized]
    ymax = max(values) * 1.08
    ymin = min(0.0, min(values))
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 720, 340
    pad_l, pad_r, pad_t, pad_b = 66, 16, 20, 56
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(log_x: float) -> float:
        return pad_l + (log_x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    parts: list[str] = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="{_esc(primary_key)} vs model size scatter" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
    ]
    # Y gridlines + ticks.
    for i in range(6):
        yval = ymin + (ymax - ymin) * i / 5
        y = sy(yval)
        parts.append(_svg_line("grid", pad_l, y, pad_l + plot_w, y))
        parts.append(
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_esc(_fmt_axis(yval))}</text>'
        )
    # X gridlines + ticks at decade (integer log10) boundaries within range, labelled in size units.
    first_decade = math.ceil(xmin)
    last_decade = math.floor(xmax)
    decades = list(range(int(first_decade), int(last_decade) + 1))
    tick_logs = [float(d) for d in decades] if decades else [xmin, xmax]
    for log_x in tick_logs:
        x = sx(log_x)
        parts.append(_svg_line("grid", x, pad_t, x, pad_t + plot_h))
        size_val = int(round(10**log_x))
        label = _fmt_flops(size_val) if use_flops else _fmt_params(size_val)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 16:.1f}" '
            f'text-anchor="middle">{label}</text>'
        )
    parts.append(_svg_line("axis", pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h))
    parts.append(_svg_line("axis", pad_l, pad_t, pad_l, pad_t + plot_h))
    ty = pad_t + plot_h / 2
    parts.append(
        f'<text class="axis-title" x="14" y="{ty:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {ty:.1f})">{_esc(primary_key)}</text>'
    )
    parts.append(
        f'<text class="axis-title" x="{pad_l + plot_w / 2:.1f}" y="{height - 4:.1f}" '
        f'text-anchor="middle">{_esc(axis_label)}</text>'
    )

    # Pareto frontier staircase (drawn behind the markers). A staircase between two frontier
    # points steps horizontally at the better perf then vertically, so it never implies an
    # unobserved point is achievable.
    if len(frontier) >= 2:
        stair: list[str] = []
        prev: ParetoPoint | None = None
        for pt in frontier:
            px, py = sx(math.log10(pt.x)), sy(pt.y)
            if prev is None:
                stair.append(f"{px:.1f},{py:.1f}")
            else:
                stair.append(f"{px:.1f},{sy(prev.y):.1f}")
                stair.append(f"{px:.1f},{py:.1f}")
            prev = pt
        parts.append(
            f'<polyline class="pareto-front" fill="none" stroke="var(--accent)" '
            f'stroke-width="1.5" stroke-dasharray="4 3" points="{" ".join(stair)}"/>'
        )

    model_order = sorted({n for n, _s, _v, _f in sized})
    shape_for_model = {name: _marker_shape(i) for i, name in enumerate(model_order)}
    for name, size, val, family in sized:
        cx, cy = sx(math.log10(size)), sy(val)
        color = _model_hue(name)
        shape = shape_for_model[name]
        # Family as a non-colour channel (mirrors the curve plots): a foundation model gets a
        # dashed marker outline, a baseline none, so the two families read apart in greyscale.
        dash = _family_dash(family)
        fam_stroke = (
            ""
            if dash == "none"
            else f' stroke="{color}" stroke-width="1" stroke-dasharray="{dash}"'
        )
        size_label = _fmt_flops(size) if use_flops else _fmt_params(size)
        unit = "FLOPs" if use_flops else "params"
        title = _esc(f"{name}: {size_label} {unit} → {_fmt_metric(val)}")
        fam_attr = f' data-family="{_esc(family)}"' if family else ""
        extra = (
            f' class="pt" data-model="{_esc(name)}"{fam_attr}'
            f' data-x="{_esc(f"{size_label} {unit}")}"'
            f' data-metric="{_esc(primary_key)}" data-y="{_fmt_metric(val)}"'
            f'{fam_stroke} tabindex="0" role="img" aria-label="{title}">'
            f"<title>{title}</title>"
        )
        elem = _marker_svg(shape, cx, cy, 4.0, color, extra)
        close = (
            "</rect>" if shape == "square" else ("</circle>" if shape == "circle" else "</polygon>")
        )
        parts.append(elem + close)
    parts.append("</svg>")

    legend_items = "".join(
        '<span class="legend-item">'
        '<svg class="legend-swatch" viewBox="0 0 24 10" aria-hidden="true">'
        f"{_marker_svg(shape_for_model[name], 12, 5, 3.0, _model_hue(name), '/>')}"
        "</svg>"
        f"<span>{_esc(name)}</span></span>"
        for name in model_order
    )
    legend = f'<div class="legend">{legend_items}</div>'
    arrow = "&darr; lower is better" if lower else "&uarr; higher is better"
    dropped_note = (
        f' <span class="plot-dir">&middot; {dropped} model(s) without size data omitted</span>'
        if dropped
        else ""
    )
    return (
        '<figure class="plot-figure">'
        f'<figcaption class="plot-title">{_esc(primary_key)} vs model size '
        f'<span class="plot-dir">{arrow}</span>{dropped_note}</figcaption>'
        f'{"".join(parts)}{legend}'
        '<p class="note plot-note">Efficiency / reference view &mdash; spans every regime and '
        "track (not a ranking). The dashed staircase is the size/perf Pareto frontier.</p>"
        "</figure>"
    )


# --------------------------------------------------------------------------------------------------
# Table rendering (a column per discovered scalar metric)
#: Cap for the SHORT badge tooltip (the full note lives in the provenance <details> block).
_BADGE_NOTE_MAX: int = 160

#: The sort-direction caret appended to a sortable ``<th>`` (CSS-hidden until ``body.js-on`` --
#: no dead affordance with JS off). Shared by the leaderboard tables and the foundation Size
#: header so both stamp identical markup.
_SORT_CARET: str = '<span class="sort-caret" aria-hidden="true"></span>'


def _shorten(text: str, limit: int = _BADGE_NOTE_MAX) -> str:
    """Collapse whitespace and truncate ``text`` to a short one-line tooltip (adds an ellipsis).

    Used for the verification-note ``title=`` on a badge: the full, wrapping note is rendered
    in the provenance block, so the hover tooltip only needs a readable teaser.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------------------------------
def _render_family_chip(family: str | None) -> str:
    """Render the model-family chip (baseline=neutral, foundation=violet)."""
    if family is None:
        return ""
    text, css_class = _FAMILY_CHIP.get(family, (family, "chip-baseline"))
    return f'<span class="chip {css_class}">{_esc(text)}</span>'


def _render_badge(status: str, note: str | None = None) -> str:
    """Render the verification badge span for a status.

    When ``note`` is non-empty it becomes a (short, escaped) ``title=`` tooltip on the badge
    so the provenance/verification note is discoverable on hover; ``None``/empty leaves the
    badge byte-identical to before, so every other call site keeps working unchanged.
    """
    text, css_class = _BADGE.get(status, (status, "badge-self"))
    title = f' title="{_esc(_shorten(note))}"' if isinstance(note, str) and note.strip() else ""
    return f'<span class="badge {css_class}"{title}>{_esc(text)}</span>'


def _render_tier_legend() -> str:
    """Render the compact legend decoding the board's verification / family / contamination keys.

    Data-driven: the badge and chip labels come from ``_BADGE`` / ``_FAMILY_CHIP`` /
    ``_OVERLAP_BADGE`` (never hardcoded), each paired with its short gloss from the ``*_GLOSS``
    maps and given a ``title=`` for hover. Lists the 4 verification tiers, the 2 model families,
    and the 3 contamination states, then a trailing "Full guide ->" link to
    ``guide.html#verification``. Fully usable with JS off (plain spans + an ``<a>``). Rendered in
    the index hero and under the task-page invariant note.

    The swatches carry a dedicated ``tl-swatch`` class -- NOT the chart-legend
    ``legend-item``/``legend-swatch`` classes (whose fixed 24x10 swatch sizing would clobber
    the pills) and NOT the live ``badge``/``chip`` class, so a legend sample is never
    mistaken for a real row badge -- e.g. the contamination legend does not emit a
    ``class="badge badge-overlap-*"`` row-badge string on a page whose rows declare no
    pretraining.
    """

    def _item(label: str, css_class: str, gloss: str) -> str:
        gloss_attr = f' title="{_esc(gloss)}"' if gloss else ""
        return (
            f'<span class="tl-item"{gloss_attr}>'
            f'<span class="tl-swatch {css_class}">{_esc(label)}</span>'
            f'<span class="tl-gloss">{_esc(gloss)}</span>'
            "</span>"
        )

    badges = "".join(
        _item(text, css_class, _BADGE_GLOSS.get(key, ""))
        for key, (text, css_class) in _BADGE.items()
    )
    families = "".join(
        _item(text, css_class, _FAMILY_GLOSS.get(key, ""))
        for key, (text, css_class) in _FAMILY_CHIP.items()
    )
    overlaps = "".join(
        _item(text, css_class, _OVERLAP_GLOSS.get(key, ""))
        for key, (text, css_class) in _OVERLAP_BADGE.items()
    )
    guide_link = f'<a class="tl-guide" href="{_GUIDE_SLUG}.html#verification">Full guide &rarr;</a>'
    return (
        '<div class="tier-legend" aria-label="How to read the board">'
        f'<span class="tl-group">{badges}</span>'
        f'<span class="tl-group">{families}</span>'
        f'<span class="tl-group">{overlaps}</span>'
        f"{guide_link}"
        "</div>"
    )


def _render_hero_cta() -> str:
    """Render the homepage hero's call-to-action row (submit / guide / repo).

    The primary action reuses the ``submit-cta`` button and links to the external submission
    guide; the two secondary actions are outline buttons to the on-site Guide and the GitHub
    repository. All three are plain ``<a>`` links, so the row is fully usable with JS off.
    """
    return (
        '<div class="hero-cta">'
        '<a class="submit-cta" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Submit a result</a>'
        f'<a class="hero-cta-secondary" href="{_GUIDE_SLUG}.html">How it works</a>'
        '<a class="hero-cta-secondary" target="_blank" rel="noopener" '
        f'href="{_esc(_REPO_URL)}">GitHub</a>'
        "</div>"
    )


def _overlap_status(row: dict[str, Any]) -> str | None:
    """Return the row's ``pretraining.overlap_with_eval`` value, or ``None`` if undeclared.

    Only foundation-model rows that disclose a ``pretraining`` block (schema 1.2.0) carry this;
    a baseline ``from_scratch`` row -- or any row without the block -- returns ``None`` so no
    contamination badge is rendered and its markup stays byte-identical to before.
    """
    pretraining = row.get("pretraining")
    if not isinstance(pretraining, dict):
        return None
    overlap = pretraining.get("overlap_with_eval")
    return overlap if isinstance(overlap, str) else None


def _render_overlap_badge(row: dict[str, Any]) -> str:
    """Render the contamination badge for a row's pretraining/eval overlap (empty if undeclared).

    Reads ``pretraining.overlap_with_eval`` (schema 1.2.0): ``none`` -> a neutral/green "clean"
    badge, ``unknown`` -> an amber "overlap unknown" badge, ``confirmed`` -> a red "contaminated"
    badge. Rows without a ``pretraining`` block (every current baseline row) render nothing, so
    the contamination surface is purely additive and never alters existing rows. The
    ``disclosure_note`` becomes the badge tooltip when present.
    """
    overlap = _overlap_status(row)
    if overlap is None:
        return ""
    text, css_class = _OVERLAP_BADGE.get(overlap, (overlap, "badge-overlap-unknown"))
    pretraining = row.get("pretraining")
    note = pretraining.get("disclosure_note") if isinstance(pretraining, dict) else None
    title = f' title="{_esc(str(note))}"' if isinstance(note, str) and note else ""
    return f'<span class="badge {css_class}"{title}>{_esc(text)}</span>'


def _ci_nature_title(row: dict[str, Any]) -> str:
    """Explain what the primary interval IS, for the ``.metric-ci`` ``title=`` (empty if none).

    Reads ``metrics.uncertainty[<primary>]`` (schema 1.2.0). A ``multi_seed_std`` interval is a
    DESCRIPTIVE std over seeds -- NOT a 95% CI -- so we say so explicitly (naming the seed count
    when present). Any other method falls back to the entry's own ``note``, else a generic
    "confidence interval [method]" label. Returns ``""`` when there is no uncertainty entry, so
    the caller adds no ``title=`` and the markup stays unchanged for CI-only rows.
    """
    uncertainty = row["metrics"].get("uncertainty")
    if not isinstance(uncertainty, dict):
        return ""
    entry = uncertainty.get(_primary_key(row))
    if not isinstance(entry, dict):
        return ""
    method = entry.get("method")
    if method == "multi_seed_std":
        n_seeds = entry.get("n_seeds")
        scope = f"{n_seeds} seeds" if isinstance(n_seeds, int) else "seeds"
        return f"±1σ over {scope} (descriptive std, not a 95% CI)"
    note = entry.get("note")
    if isinstance(note, str) and note.strip():
        return " ".join(note.split())
    if isinstance(method, str) and method:
        return f"confidence interval [{method}]"
    return "confidence interval"


def _render_ci_note(row: dict[str, Any], overlaps_above: bool) -> str:
    """Render the sub-value CI annotation for the primary cell (empty when no CI).

    Shows the primary metric's ``[ci_low, ci_high]`` interval (schema 1.2.0) under the
    point estimate. A ``title=`` (from :func:`_ci_nature_title`) states what the interval IS
    (e.g. a descriptive multi-seed std vs a 95% CI) so a skeptic does not misread it. When the
    interval overlaps the row directly above, an ``≈`` marker is added with a tooltip: the two
    rows are statistically indistinguishable on the primary metric, so the rank gap between
    them is within confidence-interval noise. The ordering itself is unchanged (see
    :func:`_sort_rows`).
    """
    ci = _primary_ci(row)
    if ci is None:
        return ""
    lo, hi = ci
    ci_text = f"[{_fmt_metric(lo)}, {_fmt_metric(hi)}]"
    nature = _ci_nature_title(row)
    nature_attr = f' title="{_esc(nature)}"' if nature else ""
    if overlaps_above:
        marker = (
            '<span class="ci-tie" title="Within the confidence interval of the row above'
            ' &mdash; statistically indistinguishable on the primary metric.">&asymp;</span>'
        )
        return f'<span class="metric-ci overlap"{nature_attr}>{marker}{_esc(ci_text)}</span>'
    return f'<span class="metric-ci"{nature_attr}>{_esc(ci_text)}</span>'


def _render_row(
    rank: int,
    row: dict[str, Any],
    scalar_keys: list[str],
    primary_key: str,
    primary_max: float,
    *,
    overlaps_above: bool = False,
) -> str:
    """Render one ``<tr>``: rank, model (+family chip), size, each scalar, status.

    A cell is rendered for EVERY discovered scalar metric so no metric is left out; a metric
    absent from this particular row shows an en-dash. The primary column carries the score
    bar and is visually emphasised, plus its confidence interval (schema 1.2.0) when known;
    ``overlaps_above`` flags a CI overlap with the preceding row (annotation only, never a
    reorder -- see :func:`_sort_rows`). Model size/compute (params + optional n_flops) lives in
    its OWN sortable Size column (see :func:`_render_size_cell`), keeping the model cell clean.
    """
    model = row["model"]
    name = _esc(model["name"])
    # A published-paper method links to its paper (model.url); a no-paper method links to its
    # implementation-faithful explanation on the Methods page (methods.html#<name>).
    url = model.get("url") or _method_anchor(model["name"])
    if isinstance(url, str) and url:
        name = f'<a href="{_esc(url)}">{name}</a>'
    chip = _render_family_chip(_family(row))
    overlap_badge = _render_overlap_badge(row)
    size_cell = _render_size_cell(row)
    lower_is_better = _is_lower_better(primary_key)

    values = _scalar_values(row)
    metric_cells: list[str] = []
    for key in scalar_keys:
        if key not in values:
            metric_cells.append('<td class="num">&ndash;</td>')
            continue
        formatted = _fmt_metric(values[key])
        value_attr = f' data-value="{formatted}"'
        if key == primary_key:
            bar = _render_bar(values[key], primary_max, lower_is_better=lower_is_better)
            ci_note = _render_ci_note(row, overlaps_above)
            cell_class = "num primary overlap" if overlaps_above else "num primary"
            metric_cells.append(
                f'<td class="{cell_class}"{value_attr}><span class="metric-val">{formatted}</span>'
                f"{ci_note}{bar}</td>"
            )
        else:
            metric_cells.append(f'<td class="num"{value_attr}>{formatted}</td>')

    badge = _render_badge(_status(row), (row.get("verification") or {}).get("note"))
    rank_html = f'<span class="rank-badge">{rank}</span>' if rank == 1 else str(rank)
    family = _family(row)
    family_attr = f' data-family="{_esc(family)}"' if family else ""
    verified = "true" if _status(row) == "verified" else "false"
    return (
        f'<tr data-model="{_esc(str(model["name"]))}"{family_attr} data-verified="{verified}">'
        f'<td class="rank num">{rank_html}</td>'
        f'<td class="model"><span class="model-name">{name}</span>{chip}{overlap_badge}</td>'
        f"{size_cell}"
        f"{''.join(metric_cells)}"
        f'<td class="status">{badge}</td>'
        "</tr>"
    )


def _render_group_table(
    dataset: str,
    regime: str,
    track: str,
    rows: list[dict[str, Any]],
    primary_key: str,
    scalar_keys: list[str],
) -> str:
    """Render the leaderboard table for one ``(dataset, regime, track)`` group.

    Columns: ``#``, ``Model`` (name + family chip + params), one column per discovered
    scalar metric (primary first + emphasised), ``Status``. Rows are sorted by the primary
    metric, descending. The table carries ``data-regime`` and ``data-track`` so the
    no-mixing invariant is checkable from the rendered HTML.
    """
    ordered = _sort_rows(rows)
    primary_max = max((_primary_value(r) for r in ordered), default=1.0)
    overlap_flags = _overlap_with_above(ordered)

    caret = _SORT_CARET
    head_metric_cells = "".join(
        (
            f'<th class="num primary" data-sortable data-metric="{_esc(k)}" data-sort="num" '
            f'aria-sort="none" tabindex="0">{_esc(k)}'
            f'<span class="col-note">primary</span>{caret}</th>'
            if k == primary_key
            else f'<th class="num" data-sortable data-metric="{_esc(k)}" data-sort="num" '
            f'aria-sort="none" tabindex="0">{_esc(k)}{caret}</th>'
        )
        for k in scalar_keys
    )
    header = (
        "<thead><tr>"
        '<th class="rank">#</th>'
        f'<th class="model" data-sortable data-sort="text" aria-sort="none" tabindex="0">'
        f"Model{caret}</th>"
        f'<th class="num size" data-sortable data-sort="num" aria-sort="none" tabindex="0">'
        f"Size{caret}</th>"
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
    no_match = (
        '<tr class="no-match-row" hidden><td colspan="99" class="no-match">'
        "No model matches the current filter.</td></tr>"
    )
    return (
        '<div class="table-scroll">'
        f'<table data-leaderboard data-dataset="{_esc(dataset)}" '
        f'data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f"{header}<tbody>\n{body_rows}\n{no_match}</tbody></table>"
        "</div>"
    )


def _render_provenance(rows: list[dict[str, Any]]) -> str:
    """Render a ``<details class="provenance">`` block listing every row's verification note.

    Only rows whose ``verification.note`` is non-empty are listed; each entry shows the model
    name (a clickable link -- ``model.url`` when present, else the in-site methods anchor via
    :func:`_method_anchor`), its tier badge, and the FULL note text (escaped, wrapping). When
    no row in the group carries a note, returns ``""`` (renders nothing) so single-baseline /
    note-free groups stay clean. Rows are shown in the group's ranked order for consistency
    with the table above.
    """
    entries: list[str] = []
    for row in _sort_rows(rows):
        note = (row.get("verification") or {}).get("note")
        if not (isinstance(note, str) and note.strip()):
            continue
        model = row["model"]
        name = _esc(str(model["name"]))
        url = model.get("url") or _method_anchor(str(model["name"]))
        name_html = f'<a href="{_esc(url)}">{name}</a>' if isinstance(url, str) and url else name
        badge = _render_badge(_status(row))
        entries.append(
            '<li class="provenance-item">'
            f'<span class="provenance-head"><span class="provenance-model">{name_html}</span>'
            f"{badge}</span>"
            f'<p class="provenance-note">{_esc(note)}</p>'
            "</li>"
        )
    if not entries:
        return ""
    count = len(entries)
    summary = f"Provenance &amp; notes ({count})" if count != 1 else "Provenance &amp; notes"
    return (
        '<details class="provenance">'
        f'<summary class="provenance-summary">{summary}</summary>'
        f'<ul class="provenance-list">{"".join(entries)}</ul>'
        "</details>"
    )


def _render_group(
    dataset: str,
    regime: str,
    track: str,
    rows: list[dict[str, Any]],
    primary_key: str,
    *,
    show_dataset: bool = False,
) -> str:
    """Render one ``(dataset, regime, track)`` group: its table plus one plot per curve metric.

    Genericity + the plot-OR-table-for-every-metric rule are realised here: the scalar keys
    are discovered from the group's rows (every scalar gets a column) and the curve names
    are discovered too (every curve gets an inline-SVG plot overlaying the group's models).
    Nothing here is task-specific, and because the group is a single (dataset, regime, track), no
    table or plot ever mixes two datasets, two regimes nor two tracks. ``show_dataset`` names the
    dataset in the heading only when the task has more than one (single-dataset pages stay clean).
    """
    scalar_keys = _ordered_scalar_keys(rows, primary_key)
    table = _render_group_table(dataset, regime, track, rows, primary_key, scalar_keys)
    # Provenance/notes block (empty when no row in the group carries a verification note), placed
    # right after the table and before the plots so a reader sees the caveats next to the ranking.
    provenance = _render_provenance(rows)

    # One inline-SVG plot per discovered curve metric (skipped gracefully if none).
    plots: list[str] = []
    for curve_name in _ordered_curve_names(rows):
        series: list[tuple[str, str | None, list[dict[str, Any]]]] = []
        for row in _sort_rows(rows):
            curves = _curves(row)
            if curve_name in curves:
                series.append((str(row["model"]["name"]), _family(row), curves[curve_name]))
        plot = _render_curve_plot(curve_name, series)
        if plot:
            plots.append(plot)
    # A bar chart per scalar metric (X = model, Y = performance, with CI whiskers) -- the
    # per-model comparison every scalar metric (which has no 2-D curve of its own) otherwise lacks.
    for scalar_key in scalar_keys:
        bar = _render_bar_chart(scalar_key, rows)
        if bar:
            plots.append(bar)
    plots_html = f'<div class="plots">{"".join(plots)}</div>' if plots else ""

    # A clear label for the group. The dataset is named only on multi-dataset tasks; a single-track
    # task (everything in the default 'all' bucket) is labelled by regime only; multi-track tasks
    # name the track too. Regime and track render as chips (reusing the foundation page's
    # .chip-regime / .chip-track classes); the dataset stays plain text so it reads as a title.
    regime_label = _regime_label(rows[0])
    parts = []
    if show_dataset:
        parts.append(f'<span class="group-dataset">Dataset &middot; {_esc(dataset)}</span>')
    parts.append(f'<span class="chip chip-regime">Regime &middot; {_esc(regime_label)}</span>')
    if track != _DEFAULT_TRACK:
        parts.append(
            f'<span class="chip chip-track">Track &middot; {_esc(_track_label(track))}</span>'
        )
    heading = "".join(parts)
    return (
        '<section class="group" '
        f'data-dataset="{_esc(dataset)}" data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f'<h3 class="group-title">{heading}</h3>'
        f"{table}{provenance}{plots_html}"
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
    """Render the static "Submit a result" sidebar CTA (links to docs/SUBMISSION.md).

    Shows the real self-serve command flow (docs/SUBMISSION.md) in a selectable ``<pre>``
    (readable with JavaScript off) so a would-be submitter sees the exact steps in place.
    """
    commands = """# 1. run the eval (emits result.json, marked self_reported)
rfbench eval <task> --model <name> --regime <regime>
# 2. add leaderboard/results/<task>/<name>.json, then open a PR"""
    return (
        '<div class="submit-card"><h3>Submit a result</h3>'
        "<p>Run the eval, add the JSON result (validated against result.schema.json) "
        "and open a pull request. Rows are auto-ranked on merge.</p>"
        f'<pre class="cmd">{_esc(commands)}</pre>'
        f'<a class="submit-cta" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Submission guide</a>'
        "</div>"
    )


def _page_description_for(entry: DeclaredTask | None, title: str) -> str:
    """Return a page-specific meta description for a task page.

    Prefers the task's committed one-paragraph ``description``, then its short ``blurb``;
    falls back to a generic, data-driven sentence built from the task ``title`` so an
    undeclared task still gets a sensible (non-empty) description.
    """
    if entry is not None and entry.description:
        return entry.description
    if entry is not None and entry.blurb:
        return entry.blurb
    return f"{title} leaderboard on RF-Benchmark-Hub — {_SITE_TAGLINE}."


def _render_dataset_selector(datasets: list[str]) -> str:
    """Render the multi-dataset segmented selector for a task page (>1 dataset only).

    A segmented button group (same look/markup family as ``_render_board_controls``'s family
    control) listing every dataset of the task; the board script shows only the ``.group``
    sections whose ``data-dataset`` matches the picked one and hides the rest. Progressive
    enhancement: the whole ``.dataset-selector`` bar is CSS-hidden until ``body.js-on`` (no dead
    control with JS off) AND, crucially, with JS off NO group is hidden -- the selector never
    emits a default-hidden rule, so every dataset section stays visible. The first dataset is the
    default active button. Callers pass this only when ``len(datasets) > 1``; a single-dataset
    task renders no selector at all.
    """
    buttons: list[str] = []
    for i, dataset in enumerate(datasets):
        active = " board-seg-active" if i == 0 else ""
        pressed = "true" if i == 0 else "false"
        buttons.append(
            f'<button type="button" class="board-seg{active}" '
            f'data-dataset="{_esc(dataset)}" aria-pressed="{pressed}">{_esc(dataset)}</button>'
        )
    return (
        '<div class="dataset-selector" role="region" aria-label="Dataset selector">'
        '<span class="dataset-selector-label">Dataset</span>'
        '<div class="board-segmented" role="group" aria-label="Choose dataset">'
        f'{"".join(buttons)}'
        "</div>"
        "</div>"
    )


def _render_board_controls() -> str:
    """Render the per-task-page interactive controls bar (search + verified toggle + family).

    Progressive enhancement: the bar is ``display:none`` until the board script adds
    ``body.js-on`` (so no dead control shows when JS is off). Every control is a hook the
    generic board script reads by attribute -- it filters the ``tr[data-model]`` rows of EVERY
    table on the page (masking rows never violates the one-regime-per-table invariant) and does
    not couple to any task. Deliberately free of the substring "line plot" so the scalar-only
    page test (which asserts that substring never appears when no curve is drawn) stays true.
    """
    return (
        '<div class="board-controls" role="region" aria-label="Leaderboard controls">'
        '<input type="search" class="board-search" id="board-search" '
        'placeholder="Filter models by name..." aria-label="Filter models by name">'
        '<label class="board-toggle"><input type="checkbox" id="board-verified-only">'
        "<span>Verified only</span></label>"
        '<div class="board-segmented" role="group" aria-label="Model family">'
        '<button type="button" class="board-seg board-seg-active" data-family="all" '
        'aria-pressed="true">All</button>'
        '<button type="button" class="board-seg" data-family="baseline" '
        'aria-pressed="false">Baseline</button>'
        '<button type="button" class="board-seg" data-family="foundation" '
        'aria-pressed="false">Foundation</button>'
        "</div>"
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
    header_grid = (
        f'<div class="task-header-grid">{dataset_card}{metrics_block}</div>'
        if (dataset_card or metrics_block)
        else ""
    )
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
        f"{header_grid}"
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
    return _page(
        page_title, body, current=entry.id, description=_page_description_for(entry, title)
    )


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
    header_grid = (
        f'<div class="task-header-grid">{dataset_card}{metrics_block}</div>'
        if (dataset_card or metrics_block)
        else ""
    )

    # (dataset, regime, k_shot, track) -> rows, preserving input order within each leaf group.
    # Dataset is part of the key so two datasets of one task never share a table/plot.
    groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (_dataset_name(row), _regime_name(row), _k_shot(row), _track_name(row))
        groups.setdefault(key, []).append(row)
    multi_dataset = len({dataset for dataset, _r, _k, _t in groups}) > 1

    def _dataset_group_sort_key(k: tuple[str, str, int | None, str]) -> tuple[Any, ...]:
        dataset, regime, k_shot, track = k
        return (dataset, *_regime_track_sort_key((regime, k_shot, track)))

    ordered_keys = sorted(groups, key=_dataset_group_sort_key)
    sections = [
        _render_group(
            dataset,
            regime,
            track,
            groups[key],
            _primary_key(groups[key][0]),
            show_dataset=multi_dataset,
        )
        for key in ordered_keys
        for (dataset, regime, _k, track) in (key,)
    ]

    nav_ids = nav_task_ids if nav_task_ids is not None else [task_name]
    sidebar = _render_task_sidebar(nav_ids, declared, task_name)
    details_card = _render_task_details_card(entry, task_name, rows)
    submit_card = _render_submit_card()
    controls = _render_board_controls()
    # A dataset selector ONLY when the task spans more than one dataset (segmented control; the
    # board script filters the .group sections by data-dataset). Single-dataset tasks: no selector.
    dataset_selector = (
        _render_dataset_selector(sorted({dataset for dataset, _r, _k, _t in groups}))
        if multi_dataset
        else ""
    )
    # One size/perf Pareto scatter over EVERY model of the task (efficiency/reference view that
    # deliberately spans regimes/tracks -- clearly labelled as such in its own caption). Skipped
    # when fewer than 2 models carry size data.
    pareto = _render_pareto_scatter(rows)
    pareto_html = (
        f'<section class="efficiency-section"><h3 class="group-title">Size vs performance</h3>'
        f'<div class="plots">{pareto}</div></section>'
        if pareto
        else ""
    )
    body = (
        '<section class="task">'
        f'<p class="breadcrumb"><a href="index.html">Tasks</a> / {_esc(title)}</p>'
        f'<h2 class="task-title">{_esc(title)}</h2>'
        f'<p class="task-meta">{dataset_line}</p>'
        f"{header}"
        '<div class="task-layout">'
        f"{sidebar}"
        '<div class="task-main">'
        f"{header_grid}"
        f'<p class="note">Each (regime, track) is ranked separately &mdash; a table or plot '
        "never mixes two regimes nor two tracks (protocol invariant). Badges mark "
        "maintainer-verified rows vs self-reported ones.</p>"
        f"{_render_tier_legend()}"
        f"{controls}"
        f"{dataset_selector}"
        f"{''.join(sections)}"
        f"{pareto_html}"
        "</div>"
        '<aside class="task-sidebar-right">'
        f"{details_card}{submit_card}"
        "</aside>"
        "</div>"
        "</section>"
    )
    page_title = f"{title} — RF-Benchmark-Hub"
    page_desc = _page_description_for(entry, title)
    return _page(
        page_title,
        body,
        current=task_name,
        description=page_desc,
        extra_body=f"<script>{render_scripts()}</script>",
    )


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
    n_verified = sum(1 for r in rows if _status(r) == "verified")
    best_model, best_score, primary = _best_summary(rows)
    entry = declared.get(task_name)
    badge = _render_status_badge(entry) if entry is not None else ""
    status = entry.status if entry is not None else "implemented"
    blurb = (
        f'<span class="card-blurb">{_esc(entry.blurb)}</span>'
        if entry is not None and entry.blurb
        else ""
    )
    priority = entry.priority if (entry is not None and entry.priority) else ""
    verified_frac = f"{n_verified / n_rows:.4f}" if n_rows else "0"
    return (
        f'<a class="task-card hover-elevate" data-status="{_esc(status)}" '
        f'data-priority="{_esc(priority)}" data-results="{n_rows}" '
        f'data-verified="{verified_frac}" '
        f'href="{_esc(task_name)}.html">'
        f'<span class="card-title">{_esc(title)}</span>{badge}'
        f"{blurb}"
        f'<span class="card-sub">{_esc(f"{n_rows} results · {n_models} models")}'
        f'<span class="card-verified-cov{"" if n_verified else " card-verified-cov-zero"}" '
        f'title="maintainer-verified re-runs on this task">{n_verified}/{n_rows} verified</span>'
        "</span>"
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
        f'data-priority="{_esc(entry.priority or "")}" data-results="0" data-verified="0" '
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
    n_verified = sum(1 for rows in grouped.values() for row in rows if _status(row) == "verified")
    return {
        "tasks_defined": len(declared),
        "implemented": n_implemented,
        "live": n_live,
        "eval_tracks": n_eval_groups,
        "verified": n_verified,
    }


def _render_stats_row(stats: dict[str, int]) -> str:
    """Render the homepage's 4 big-number stat cards."""
    items = (
        (stats["tasks_defined"], "Tasks defined", ""),
        (stats["implemented"], "Implemented", ""),
        (stats["live"], "Live leaderboard" if stats["live"] == 1 else "Live leaderboards", ""),
        (stats["eval_tracks"], "Evaluation tracks", ""),
        (stats["verified"], "Verified scores", " stat-card-verified"),
    )
    cards = "".join(
        f'<div class="stat-card{extra}"><span class="stat-value">{n}</span>'
        f'<span class="stat-label">{_esc(label)}</span></div>'
        for n, label, extra in items
    )
    return f'<div class="stats-row">{cards}</div>'


def _render_filter_bar() -> str:
    """Render the homepage's search input + status filter pills + card sort (wired by inline JS).

    The sort ``<select>`` is CSS-hidden until the script adds ``body.js-on`` (progressive
    enhancement, no dead control with JS off). The pills (All/Implemented/In progress/Planned)
    are unchanged. Sorting acts only WITHIN each ``.card-grid`` (never moves a card across a
    scope section), keyed off the ``data-priority``/``data-results``/``data-verified`` hooks the
    cards carry.
    """
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
    sort_options = (
        ("default", "Default order"),
        ("priority", "Priority (P1 first)"),
        ("results", "Most results"),
        ("verified", "% verified"),
    )
    option_html = "".join(
        f'<option value="{value}">{_esc(label)}</option>' for value, label in sort_options
    )
    return (
        '<div class="filter-bar">'
        '<input type="search" id="task-search" class="search-input" '
        'placeholder="Search a task, dataset or metric...">'
        f'<div class="filter-pills">{pill_html}</div>'
        '<div class="home-sort"><label class="home-sort-label" for="task-sort">Sort</label>'
        f'<select id="task-sort">{option_html}</select></div>'
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
            "foundation models. Built for ML and RF researchers: each task ranks submissions "
            "by its primary metric, and regimes and tracks are never mixed in a "
            "comparison.</p>"
            f"{_render_hero_cta()}"
            f"{_render_tier_legend()}"
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


# --------------------------------------------------------------------------------------------------
# Methods page -- per-method explanations extracted (faithfully) from the model docstrings
# --------------------------------------------------------------------------------------------------
_METHODS_SLUG: str = "methods"

#: Slug for the dedicated Foundation Models page (``foundation.html``, see ``render_foundation``).
_FOUNDATION_SLUG: str = "foundation"


class _MethodDoc(NamedTuple):
    """One registered model's explanation, read from its source WITHOUT importing torch/numpy."""

    name: str  # the @register_model id (== result.json model.name)
    family: str  # "baseline" | "foundation" | ...
    doc: str  # the class docstring, verbatim (the implementation-faithful explanation)
    source: str  # repo-relative source path (e.g. rfbench/models/baselines/hoc_amc.py)


def _models_root() -> Path | None:
    """Locate ``rfbench/models`` by walking up from this file (source checkout)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "rfbench" / "models"
        if candidate.is_dir():
            return candidate
    return None


def _const_strings(module: ast.Module) -> dict[str, str]:
    """Map module-level ``NAME = "literal"`` assignments, to resolve ``@register_model(NAME)``."""
    consts: dict[str, str] = {}
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value.value
    return consts


def _register_model_name(decorator: ast.expr, consts: dict[str, str]) -> str | None:
    """Return the id passed to ``@register_model(...)`` (literal or module const), else ``None``."""
    if not (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "register_model"
        and decorator.args
    ):
        return None
    arg = decorator.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return consts.get(arg.id)
    return None


def _class_family(cls: ast.ClassDef) -> str:
    """Read the class-body ``family = "..."`` (annotated or plain); default ``"baseline"``."""
    for node in cls.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        value = getattr(node, "value", None)
        if target == "family" and isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return "baseline"


def _extract_method_docs() -> dict[str, _MethodDoc]:
    """Extract ``{registered_name: _MethodDoc}`` for every ``@register_model`` class via ``ast``.

    Parses the model source files (``rfbench/models/**/*.py``) WITHOUT importing them, so the
    dependency-free site build never pulls in torch/numpy. The explanation is each class's own
    docstring -- authored alongside the implementation, so it is faithful by construction.
    """
    root = _models_root()
    if root is None:
        return {}
    repo_root = root.parents[1]  # .../rfbench/models -> repo root
    docs: dict[str, _MethodDoc] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        consts = _const_strings(module)
        rel = str(path.relative_to(repo_root))
        module_doc = (ast.get_docstring(module) or "").strip()
        # Collect the registered model classes in this file.
        registered: list[tuple[str, ast.ClassDef]] = []
        for node in module.body:
            if not isinstance(node, ast.ClassDef) or not ast.get_docstring(node):
                continue
            for decorator in node.decorator_list:
                name = _register_model_name(decorator, consts)
                if name:
                    registered.append((name, node))
                    break
        # For a single-model file the module docstring is *about that model* (it holds the paper
        # citation + architecture rationale), so prepend it to the class docstring for a fuller,
        # still-faithful description. Multi-model files (e.g. the trivial floors) keep class docs
        # only, to avoid duplicating a shared module preamble across their sections.
        for name, node in registered:
            class_doc = (ast.get_docstring(node) or "").strip()
            if len(registered) == 1 and module_doc:
                full = f"{module_doc}\n\n{class_doc}"
            else:
                full = class_doc
            docs[name] = _MethodDoc(name, _class_family(node), full, rel)
    return dict(sorted(docs.items()))


#: Populated once at the start of :func:`build_site` (the source tree is stable mid-run).
_METHOD_DOCS: dict[str, _MethodDoc] = {}


def _method_anchor(name: str) -> str | None:
    """Return the in-site Methods anchor (``methods.html#name``) for ``name`` if one exists."""
    return f"{_METHODS_SLUG}.html#{name}" if name in _METHOD_DOCS else None


_RST_ROLE = re.compile(r":[a-z:]+:`~?([^`]+)`")  # :class:`~a.b.C` / :meth:`foo` -> the referent
_CODE_SPAN = re.compile(r"``([^`]+)``")
_ARXIV_RE = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s\"'<>)\]]+)")

#: model.name -> paper URL (result.json ``model.url``); populated at the start of build_site so the
#: Methods page can surface each method's paper reference alongside the parsed arXiv/DOI citations.
_MODEL_PAPER_URLS: dict[str, str] = {}


def _arxiv_link(match: re.Match[str]) -> str:
    """Turn an ``arXiv:<id>[vN]`` citation into a clickable abstract link."""
    ident = match.group(1) + (match.group(2) or "")
    return (
        f'<a target="_blank" rel="noopener" href="https://arxiv.org/abs/{ident}">'
        f"arXiv:{ident}</a>"
    )


def _doi_link(match: re.Match[str]) -> str:
    """Turn a bare ``10.xxxx/...`` DOI into a clickable doi.org link (trailing punctuation cut)."""
    doi = match.group(1).rstrip(".,;)")
    return f'<a target="_blank" rel="noopener" href="https://doi.org/{doi}">{doi}</a>'


def _paper_label(url: str) -> str:
    """A short human label for a paper URL (arXiv id / DOI / bare URL)."""
    arxiv = re.search(r"arxiv\.org/abs/([\w.]+)", url)
    if arxiv:
        return f"arXiv:{arxiv.group(1)}"
    doi = re.search(r"doi\.org/(10\.\S+)", url)
    if doi:
        return f"DOI {doi.group(1)}"
    return url


def _paper_links_for(md: _MethodDoc, model_url: str | None) -> list[tuple[str, str]]:
    """Collect a method's paper links: arXiv/DOI cited in its docstring + the result.json url."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(label: str, url: str) -> None:
        if url not in seen:
            seen.add(url)
            links.append((label, url))

    for hit in _ARXIV_RE.finditer(md.doc):
        ident = hit.group(1) + (hit.group(2) or "")
        _add(f"arXiv:{ident}", f"https://arxiv.org/abs/{ident}")
    for hit in _DOI_RE.finditer(md.doc):
        doi = hit.group(1).rstrip(".,;)")
        _add(f"DOI {doi}", f"https://doi.org/{doi}")
    if model_url:
        _add(_paper_label(model_url), model_url)
    return links


def _inline_doc(text: str) -> str:
    """Inline-format one paragraph of (already-joined) docstring text -> escaped HTML.

    Code spans / cross-reference roles become ``<code>``, ``**bold**``/``*emph*`` their tags, and
    **arXiv ids / DOIs become clickable paper links** so a method's architecture references are
    reachable straight from its description.
    """
    out = _esc(text)
    out = _RST_ROLE.sub(lambda m: f"<code>{_esc(m.group(1).split('.')[-1])}</code>", out)
    out = _CODE_SPAN.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _ARXIV_RE.sub(_arxiv_link, out)
    out = _DOI_RE.sub(_doi_link, out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    return out


def _render_docstring(text: str) -> str:
    """Render a Python docstring as readable HTML (a small, SAFE reStructuredText subset).

    Faithful passthrough of the authored text: paragraphs split on blank lines, ``*``/``-``
    bullet blocks become lists, ````code```` spans, ``**bold**``/``*emph*``, and cross-reference
    roles (``:class:`~mod.Name```) reduced to a ``<code>`` of the referent. Everything is
    HTML-escaped before any markup is added, so no docstring can inject HTML.
    """
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # A block is a bullet list iff its FIRST line is a bullet; wrapped continuation lines
        # (indented, not starting with a bullet) are folded into the current item.
        if lines[0].lstrip().startswith(("* ", "- ")):
            items: list[str] = []
            current: str | None = None
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith(("* ", "- ")):
                    if current is not None:
                        items.append(current)
                    current = stripped[2:].strip()
                elif current is not None:
                    current = f"{current} {stripped.strip()}"
            if current is not None:
                items.append(current)
            out.append("<ul>" + "".join(f"<li>{_inline_doc(it)}</li>" for it in items) + "</ul>")
        else:
            out.append(f"<p>{_inline_doc(' '.join(ln.strip() for ln in lines))}</p>")
    return "".join(out)


def render_methods_page() -> str:
    """Render ``methods.html``: one anchored section per registered model, from its docstring.

    The internal explanation every no-paper method on the board links to (``methods.html#name``).
    Faithful to the implementation by construction (the text IS the class docstring), with a
    link to the source file so a reader can jump straight to the code.
    """
    docs = _METHOD_DOCS or _extract_method_docs()
    sections: list[str] = []
    for name, md in docs.items():
        src_url = f"{_REPO_URL}/blob/main/{md.source}"
        links = _paper_links_for(md, _MODEL_PAPER_URLS.get(name))
        if links:
            refs = " &middot; ".join(
                f'<a target="_blank" rel="noopener" href="{_esc(url)}">{_esc(label)}</a>'
                for label, url in links
            )
            refs_html = f'<p class="note">Paper / references: {refs}</p>'
        else:
            refs_html = (
                '<p class="note">No published paper &mdash; the description below is the full, '
                "implementation-faithful reference (a from-source method, reproducible from the "
                "code alone).</p>"
            )
        sections.append(
            f'<section class="guide-section method" id="{_esc(name)}">'
            f"<h2><code>{_esc(name)}</code> {_render_family_chip(md.family)}</h2>"
            f"{refs_html}"
            f'<p class="note">Source: <a target="_blank" rel="noopener" '
            f'href="{_esc(src_url)}">{_esc(md.source)}</a></p>'
            f"{_render_docstring(md.doc)}"
            "</section>"
        )
    body = (
        '<section class="task guide">'
        '<h1 class="task-title">Methods</h1>'
        '<p class="note">The architecture and algorithm of every model on the board -- extracted '
        "verbatim from the implementation docstrings (faithful by construction). Each entry links "
        "its paper(s) when one exists (arXiv / DOI, also clickable inline); a method with no paper "
        "(e.g. the non-deep DSP baselines) carries its full description here instead.</p>"
        + "".join(sections)
        + "</section>"
    )
    return _page("Methods — RF-Benchmark-Hub", body, current=_METHODS_SLUG)


# --------------------------------------------------------------------------------------------------
# Foundation Models page -- foundation-only per-task podiums + a global cumulative medal table.
#
# Gives foundation-model (``model.family == "foundation"``) submissions dedicated prominence
# instead of being buried inside the per-task tables next to baselines. Everything here is
# 100% data-driven off ``leaderboard/results/**/*.json`` (no task or model name is ever
# hardcoded) and NEVER touches ``render_task_page``'s own rendering, which stays byte-identical.
# --------------------------------------------------------------------------------------------------
#: Gold/silver/bronze glyphs, indexed 0..2 (rank 1..3). A shared constant so the per-task mini
#: tables and the global cumulative podium never draw different medals for the same rank.
_MEDALS: tuple[str, ...] = ("\U0001f947", "\U0001f948", "\U0001f949")

#: Adaptation-cost proxy order for the scatter's X axis: a CATEGORICAL ordinal (NOT an
#: invented numeric FLOPs/compute-cost value -- the schema does not track compute cost)
#: derived from ``regime.name``. ``zero_shot`` needs no labelled adaptation data at all;
#: ``few_shot`` needs k labelled examples (sub-ordered by k ascending, see
#: :func:`_frugality_sort_key`); ``linear_probe`` fits a probe head; ``full_finetune`` updates
#: every weight. ``zero_shot`` is not a valid ``regime.name`` in the current schema enum (D5:
#: only from_scratch/full_finetune/linear_probe/few_shot) -- it is listed here so the axis is
#: forward-compatible the moment the schema grows that value; today the bucket simply stays
#: empty. Any unlisted regime (e.g. ``from_scratch``, which is not an "adapted" foundation
#: model) sorts after every listed value.
_FRUGALITY_REGIME_ORDER: tuple[str, ...] = (
    "zero_shot",
    "few_shot",
    "linear_probe",
    "full_finetune",
)


def _medal(rank: int) -> str:
    """Medal glyph for rank 1..3 (gold/silver/bronze); empty for anything past 3rd."""
    return _MEDALS[rank - 1] if 1 <= rank <= 3 else ""


def _frugality_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    """Ordinal position of a foundation row along the adaptation-cost axis.

    ``(regime rank, k_shot)`` -- the regime rank from :data:`_FRUGALITY_REGIME_ORDER`, with
    ``few_shot`` rows additionally sub-ordered by ``k_shot`` ascending (fewer labelled
    examples = cheaper adaptation). Non-``few_shot`` regimes carry ``k_shot = -1`` so they
    never interleave with the few_shot sub-order.
    """
    regime = _regime_name(row)
    rank = (
        _FRUGALITY_REGIME_ORDER.index(regime)
        if regime in _FRUGALITY_REGIME_ORDER
        else len(_FRUGALITY_REGIME_ORDER)
    )
    k = _k_shot(row)
    return (rank, k if k is not None else -1)


def _ranked_foundation_rows(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Rank a single ``(regime, k_shot, track)`` group of FOUNDATION-only rows, 1-based, best
    first.

    Reuses :func:`_sort_rows` (same primary-metric direction + trust-tier + name tiebreak as
    every table on the board), so a competitor's rank here matches what it would get if this
    group were compared against :func:`_render_group_table`'s ordering restricted to
    foundation rows. Shared by the per-task mini-tables and the global cumulative podium so
    the two never disagree on who ranked where.
    """
    return list(enumerate(_sort_rows(rows), start=1))


def _render_medal_rank(rank: int) -> str:
    """Render a mini-table's rank cell: a medal glyph for the top 3, a plain number after."""
    medal = _medal(rank)
    if medal:
        return f'<span class="medal" role="img" aria-label="rank {rank}">{medal}</span>'
    return str(rank)


def _render_track_chip(track: str) -> str:
    """Small chip for a track label; omitted for the default ``all`` bucket (not a real track)."""
    if track == _DEFAULT_TRACK:
        return ""
    return f'<span class="chip chip-track">{_esc(_track_label(track))}</span>'


def _render_foundation_group_table(
    regime: str,
    track: str,
    rows: list[dict[str, Any]],
    primary_key: str,
) -> str:
    """Render one foundation-only mini leaderboard for a ``(regime, k_shot, track)`` group.

    ``rows`` are ALREADY filtered to ``model.family == "foundation"`` (see
    :func:`_foundation_rows`) -- baselines never enter this ranking. The top 3 get a medal (a
    single-competitor group still awards gold -- expected with today's sparse foundation-model
    submissions), ranks below 3rd get a plain number. Mirrors :func:`_render_group_table`'s
    markup so the two stay visually consistent, but drops the per-metric column sprawl (one
    primary-metric column only, since a mini-table is scoped to one task's primary metric) and
    the family chip (every row here is foundation by construction).
    """
    ranked = _ranked_foundation_rows(rows)
    body_rows: list[str] = []
    for rank, row in ranked:
        model = row["model"]
        name = _esc(str(model["name"]))
        url = model.get("url") or _method_anchor(str(model["name"]))
        if isinstance(url, str) and url:
            name = f'<a href="{_esc(url)}">{name}</a>'
        value = _fmt_metric(_primary_value(row))
        tags = (
            f'<span class="chip chip-regime">{_esc(_regime_label(row))}</span>'
            f"{_render_track_chip(track)}"
            f"{_render_badge(_status(row))}"
        )
        body_rows.append(
            "<tr>"
            f'<td class="rank num">{_render_medal_rank(rank)}</td>'
            f'<td class="model"><span class="model-name">{name}</span></td>'
            f"{_render_size_cell(row)}"
            f'<td class="tags">{tags}</td>'
            f'<td class="num primary"><span class="metric-val">{value}</span></td>'
            "</tr>"
        )
    header = (
        "<thead><tr>"
        '<th class="rank">#</th>'
        '<th class="model">Model</th>'
        '<th class="num size" data-sortable data-sort="num" aria-sort="none" tabindex="0">'
        f"Size{_SORT_CARET}</th>"
        '<th class="tags">Regime / Track / Status</th>'
        f'<th class="num primary">{_esc(primary_key)}</th>'
        "</tr></thead>"
    )
    return (
        '<div class="table-scroll">'
        f'<table class="podium-table" data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f"{header}<tbody>\n{''.join(body_rows)}\n</tbody></table>"
        "</div>"
    )


def _render_foundation_group(regime: str, track: str, rows: list[dict[str, Any]]) -> str:
    """One foundation-only ``(regime, track)`` mini-leaderboard: heading + ranked table.

    Mirrors :func:`_render_group`'s heading convention (regime-only when the task has no real
    track, else "Regime · X / Track · Y") so a reader used to the per-task pages recognises
    the same grouping here.
    """
    primary_key = _primary_key(rows[0])
    table = _render_foundation_group_table(regime, track, rows, primary_key)
    regime_label = _regime_label(rows[0])
    if track == _DEFAULT_TRACK:
        heading = f"Regime &middot; {_esc(regime_label)}"
    else:
        heading = (
            f"Regime &middot; {_esc(regime_label)} &nbsp;/&nbsp; "
            f"Track &middot; {_esc(_track_label(track))}"
        )
    return (
        '<section class="group podium-group" '
        f'data-regime="{_esc(regime)}" data-track="{_esc(track)}">'
        f'<h4 class="group-title">{heading}</h4>'
        f"{table}"
        "</section>"
    )


def _render_foundation_scatter(rows: list[dict[str, Any]]) -> str:
    """Per-task scatter: Y = primary metric, X = a CATEGORICAL adaptation-cost axis.

    ``rows`` are a task's foundation rows across EVERY (regime, track) group -- the one chart
    on this page that intentionally spans regimes, since the tradeoff it visualises (score vs
    how much labelled data the adaptation regime costs) only exists across regimes. X is not a
    numeric FLOPs/compute-cost value (the schema does not track compute cost): it is the
    ordinal position of the row's ``(regime, k_shot)`` along ``zero_shot -> few_shot`` (k
    ascending) ``-> linear_probe -> full_finetune`` (:func:`_frugality_sort_key`), the only
    adaptation-cost proxy this board can actually derive from the schema. Skipped entirely
    (graceful degradation, same philosophy as the WIP empty-state card) when the task has
    fewer than 2 foundation points -- a single point cannot show a trend and would just be a
    misleading plot.
    """
    if len(rows) < 2:
        return ""
    primary_key = _primary_key(rows[0])
    lower = _is_lower_better(primary_key)

    # Distinct (regime, k_shot) categories actually present, in frugality order.
    categories: dict[tuple[int, int], tuple[str, int | None]] = {}
    for row in rows:
        categories.setdefault(_frugality_sort_key(row), (_regime_name(row), _k_shot(row)))
    cat_keys = sorted(categories)
    cat_index = {key: i for i, key in enumerate(cat_keys)}
    n_cats = len(cat_keys)

    values = [_primary_value(r) for r in rows]
    ymax = max(values) * 1.08
    ymin = min(0.0, min(values))
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 720, 320
    pad_l, pad_r, pad_t, pad_b = 56, 16, 20, 60
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    slot = plot_w / n_cats

    def sy(y: float) -> float:
        return pad_t + (ymax - y) / (ymax - ymin) * plot_h

    def sx(idx: int) -> float:
        return pad_l + slot * (idx + 0.5)

    parts: list[str] = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="{_esc(primary_key)} vs adaptation cost scatter" '
        f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
    ]
    for i in range(6):
        yval = ymin + (ymax - ymin) * i / 5
        y = sy(yval)
        parts.append(_svg_line("grid", pad_l, y, pad_l + plot_w, y))
        parts.append(
            f'<text class="tick" x="{pad_l - 6:.1f}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_esc(_fmt_axis(yval))}</text>'
        )
    for key, idx in cat_index.items():
        regime, k = categories[key]
        label = _regime_label({"regime": {"name": regime, "k_shot": k}})
        x = sx(idx)
        label_y = pad_t + plot_h + 16
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="middle">{_esc(label)}</text>'
        )
    parts.append(_svg_line("axis", pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h))
    parts.append(_svg_line("axis", pad_l, pad_t, pad_l, pad_t + plot_h))
    ty = pad_t + plot_h / 2
    parts.append(
        f'<text class="axis-title" x="14" y="{ty:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {ty:.1f})">{_esc(primary_key)}</text>'
    )
    parts.append(
        f'<text class="axis-title" x="{pad_l + plot_w / 2:.1f}" y="{height - 4:.1f}" '
        'text-anchor="middle">adaptation cost</text>'
    )

    # One marker per row; colour = stable model hue, shape cycled per distinct competitor (a
    # non-colour channel), mirroring the curve/bar plots elsewhere on the board.
    model_order = sorted({str(r["model"]["name"]) for r in rows})
    shape_for_model = {name: _marker_shape(i) for i, name in enumerate(model_order)}
    for row in rows:
        name = str(row["model"]["name"])
        key = _frugality_sort_key(row)
        x = sx(cat_index[key])
        val = _primary_value(row)
        y = sy(val)
        color = _model_hue(name)
        shape = shape_for_model[name]
        regime, k = categories[key]
        cat_label = _regime_label({"regime": {"name": regime, "k_shot": k}})
        title = _esc(f"{name}: {cat_label} → {_fmt_metric(val)}")
        extra = (
            f' class="pt" data-model="{_esc(name)}" data-x="{_esc(cat_label)}"'
            f' data-y="{_fmt_metric(val)}" tabindex="0" role="img"'
            f' aria-label="{title}">'
            f"<title>{title}</title>"
        )
        elem = _marker_svg(shape, x, y, 4.0, color, extra)
        close = (
            "</rect>" if shape == "square" else ("</circle>" if shape == "circle" else "</polygon>")
        )
        parts.append(elem + close)
    parts.append("</svg>")

    legend_items = "".join(
        '<span class="legend-item">'
        '<svg class="legend-swatch" viewBox="0 0 24 10" aria-hidden="true">'
        f"{_marker_svg(shape_for_model[name], 12, 5, 3.0, _model_hue(name), '/>')}"
        "</svg>"
        f"<span>{_esc(name)}</span></span>"
        for name in model_order
    )
    legend = f'<div class="legend">{legend_items}</div>'
    arrow = "&darr; lower is better" if lower else "&uarr; higher is better"
    return (
        '<figure class="plot-figure">'
        f'<figcaption class="plot-title">{_esc(primary_key)} vs adaptation cost '
        f'<span class="plot-dir">{arrow}</span></figcaption>'
        f'{"".join(parts)}{legend}'
        "</figure>"
    )


def _best_baseline_per_track(
    baseline_rows: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(track, best_row)`` for each track present among ``baseline_rows``, ordered.

    The best baseline of a track is the top row of :func:`_sort_rows` restricted to that track
    (same primary-metric direction + trust tiebreak the board uses everywhere). Tracks are
    ordered with the default ``all`` bucket first, then alphabetically (:func:`_track_sort_key`).
    This is a per-track REFERENCE (one specialist number to beat), never a ranking merged with
    the foundation competitors.
    """
    by_track: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_rows:
        by_track.setdefault(_track_name(row), []).append(row)
    out: list[tuple[str, dict[str, Any]]] = []
    for track in sorted(by_track, key=_track_sort_key):
        out.append((track, _sort_rows(by_track[track])[0]))
    return out


def _render_baseline_reference(baseline_rows: list[dict[str, Any]]) -> str:
    """Render the best-baseline REFERENCE block for a task (one line per track).

    Shown on the Foundation page when "Include baselines" is on: for each track, the single best
    specialist result (family chip ``baseline``, its regime, its primary score) as the number the
    foundation models are measured against. Explicitly labelled a REFERENCE, NOT a ranking merged
    with the FMs -- it is a cross-regime specialist yardstick, deliberately outside the FM podium.
    Returns ``""`` when the task has no baseline rows. Rendered VISIBLE by default so the info is
    present with JS off; the toggle hides it (JS on + unchecked) via the ``is-hidden`` class.
    """
    best = _best_baseline_per_track(baseline_rows)
    if not best:
        return ""
    lines: list[str] = []
    for track, row in best:
        model = row["model"]
        name = _esc(str(model["name"]))
        url = model.get("url") or _method_anchor(str(model["name"]))
        if isinstance(url, str) and url:
            name = f'<a href="{_esc(url)}">{name}</a>'
        track_chip = _render_track_chip(track)
        value = _fmt_metric(_primary_value(row))
        primary = _esc(_primary_key(row))
        lines.append(
            '<li class="baseline-ref-row">'
            '<span class="baseline-ref-tag">best baseline</span>'
            f"{_render_family_chip('baseline')}"
            f'<span class="chip chip-regime">{_esc(_regime_label(row))}</span>'
            f"{track_chip}"
            f'<span class="baseline-ref-model model-name">{name}</span>'
            f'<span class="baseline-ref-score">{primary} = <strong>{value}</strong></span>'
            "</li>"
        )
    return (
        '<div class="baseline-reference" data-baseline-ref>'
        '<p class="baseline-reference-title">Best baseline &mdash; reference, not ranked '
        "with the foundation models</p>"
        '<p class="note">A cross-regime specialist yardstick per track (the number to beat), '
        "shown for context. It is a REFERENCE only &mdash; deliberately outside the foundation "
        "podium above, never merged into that ranking.</p>"
        f'<ul class="baseline-reference-list">{"".join(lines)}</ul>'
        "</div>"
    )


def _render_foundation_task_section(
    task_name: str,
    rows: list[dict[str, Any]],
    declared: dict[str, DeclaredTask],
    baseline_rows: list[dict[str, Any]] | None = None,
) -> str:
    """One task's foundation-only section: stacked ``(regime, track)`` mini-tables + a scatter.

    ``rows`` are this task's ALREADY-FILTERED foundation rows (:func:`_foundation_rows`).
    Groups are partitioned and ordered exactly like :func:`render_task_page` (same
    :func:`_regime_track_sort_key`), so a group here never mixes two regimes nor two tracks --
    every group renders its OWN mini-table, stacked vertically, never merged into one table.

    ``baseline_rows`` (this task's ``family != "foundation"`` rows) drive the vs-baselines
    surfaces added by the "Include baselines" toggle: a per-track best-baseline REFERENCE block
    (:func:`_render_baseline_reference`) and baseline points in the size/perf Pareto scatter
    (which reuses :func:`_render_pareto_scatter` with foundation+baseline rows -- markers already
    distinguish family). Both are cross-regime efficiency/reference views, never rankings merged
    with the FM podium; both render VISIBLE by default (usable with JS off) and are hidden by the
    toggle when JS is on and the box is unchecked. When there are no baseline rows nothing extra
    is emitted, so the section is byte-identical to before for a baseline-less task.
    """
    baseline_rows = baseline_rows or []
    groups: dict[tuple[str, int | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((_regime_name(row), _k_shot(row), _track_name(row)), []).append(row)
    ordered_keys = sorted(groups, key=_regime_track_sort_key)
    group_sections = "".join(
        _render_foundation_group(regime, track, groups[(regime, k, track)])
        for (regime, k, track) in ordered_keys
    )
    baseline_reference = _render_baseline_reference(baseline_rows)
    # Two complementary numeric views, both spanning regimes (efficiency/reference, not rankings):
    # the existing frugality scatter (adaptation cost) + the size/perf Pareto scatter (compute
    # cost). The Pareto scatter is skipped when fewer than 2 foundation rows carry size data.
    scatter = _render_foundation_scatter(rows)
    pareto = _render_pareto_scatter(rows)
    plots = "".join(p for p in (scatter, pareto) if p)
    scatter_html = f'<div class="plots">{plots}</div>' if plots else ""
    # A SECOND Pareto scatter that also plots the baseline specialists (family markers already
    # read apart), so the FM-vs-specialist size/perf tradeoff is visible. Hidden by default when
    # JS is on and the toggle is unchecked; shown with JS off (info present without JS). Skipped
    # when fewer than 2 foundation+baseline rows carry size data (same graceful-degradation rule).
    pareto_with_baselines = _render_pareto_scatter(rows + baseline_rows)
    baseline_pareto_html = (
        '<div class="plots baseline-pareto" data-baseline-pareto>'
        '<p class="note">Size vs performance with the baseline specialists included '
        "(efficiency/reference view spanning regimes, not a ranking).</p>"
        f"{pareto_with_baselines}</div>"
        if (baseline_rows and pareto_with_baselines)
        else ""
    )
    title = _task_title(task_name, declared)
    return (
        f'<section class="foundation-task" data-task="{_esc(task_name)}">'
        f'<h3 class="foundation-task-title">'
        f'<a href="{_esc(task_name)}.html">{_esc(title)}</a></h3>'
        f"{group_sections}{baseline_reference}{scatter_html}{baseline_pareto_html}"
        "</section>"
    )


def _render_global_podium(grouped: dict[str, list[dict[str, Any]]]) -> str:
    """Cumulative medal table across EVERY task: one row per distinct foundation result row.

    DELIBERATE, EXPLICIT exception to the board's "never mix two regimes/tiers in one table"
    rule -- scoped to THIS one summary table only, nowhere else on the site. Every individual
    result row is its OWN competitor here: ``iqfm-base`` and ``iqfm-paper`` are NOT merged
    into one "IQFM" identity, even though that means the same underlying model can appear as
    two separate rows -- collapsing them would silently hide which exact regime/tier earned
    the medal.

    For each competitor, the BEST rank it achieved in a task counts once for that task (max
    one medal per task, even when the task has several (regime, track) groups -- its best
    result among ALL of that task's groups is what's kept); medals are then summed across
    every task and the table sorts gold desc, then silver desc, then bronze desc.
    """
    best_rank_per_task: dict[str, dict[str, int]] = {}
    for task_name, rows in grouped.items():
        foundation_rows = _foundation_rows(rows)
        if not foundation_rows:
            continue
        groups: dict[tuple[str, int | None, str], list[dict[str, Any]]] = {}
        for row in foundation_rows:
            groups.setdefault((_regime_name(row), _k_shot(row), _track_name(row)), []).append(row)
        for group_rows in groups.values():
            for rank, row in _ranked_foundation_rows(group_rows):
                competitor = str(row["model"]["name"])
                per_task = best_rank_per_task.setdefault(competitor, {})
                if per_task.get(task_name) is None or rank < per_task[task_name]:
                    per_task[task_name] = rank

    tallies: list[tuple[str, int, int, int]] = []
    for competitor, per_task in best_rank_per_task.items():
        gold = sum(1 for r in per_task.values() if r == 1)
        silver = sum(1 for r in per_task.values() if r == 2)
        bronze = sum(1 for r in per_task.values() if r == 3)
        if gold or silver or bronze:
            tallies.append((competitor, gold, silver, bronze))
    if not tallies:
        return ""
    tallies.sort(key=lambda t: (-t[1], -t[2], -t[3], t[0]))

    body_rows = "".join(
        "<tr>"
        f'<td class="model"><span class="model-name">{_esc(name)}</span></td>'
        f'<td class="num">{_medal(1)} &times;{gold}</td>'
        f'<td class="num">{_medal(2)} &times;{silver}</td>'
        f'<td class="num">{_medal(3)} &times;{bronze}</td>'
        "</tr>"
        for name, gold, silver, bronze in tallies
    )
    return (
        '<section class="group podium-group global-podium">'
        '<h2 class="group-title">Global cumulative podium</h2>'
        '<p class="note">Best rank per task (max one medal per task), summed across every '
        "task. Each result row is its OWN competitor here &mdash; e.g. <code>iqfm-base</code> "
        "and <code>iqfm-paper</code> are counted separately, never merged &mdash; the one "
        "table on this board that deliberately departs from the one-regime-per-table rule "
        "used everywhere else.</p>"
        '<div class="table-scroll"><table class="podium-table">'
        f'<thead><tr><th class="model">Model</th><th class="num">{_medal(1)}</th>'
        f'<th class="num">{_medal(2)}</th><th class="num">{_medal(3)}</th></tr></thead>'
        f"<tbody>{body_rows}</tbody></table></div>"
        "</section>"
    )


def _render_foundation_task_selector(task_ids: list[str], declared: dict[str, DeclaredTask]) -> str:
    """Render the Foundation page's task selector (an "All" option + one per task with results).

    A segmented control (same look/markup family as the task-page controls) listing every task
    that has foundation results; the foundation script shows only the ``.foundation-task`` section
    whose ``data-task`` matches the pick and hides the rest, with "All" showing everything.
    Progressive enhancement: the whole ``.foundation-controls`` bar is CSS-hidden until
    ``body.js-on`` and, with JS off, NO section is hidden (the control emits no default-hidden
    rule), so every task section stays visible. "All" is the default active button.
    """
    buttons: list[str] = [
        '<button type="button" class="board-seg board-seg-active" data-task="all" '
        'aria-pressed="true">All</button>'
    ]
    for task_id in task_ids:
        title = _task_title(task_id, declared)
        buttons.append(
            f'<button type="button" class="board-seg" data-task="{_esc(task_id)}" '
            f'aria-pressed="false">{_esc(title)}</button>'
        )
    return (
        '<div class="foundation-controls" role="region" aria-label="Foundation controls">'
        '<div class="foundation-task-select">'
        '<span class="dataset-selector-label">Task</span>'
        '<div class="board-segmented" role="group" aria-label="Choose task">'
        f'{"".join(buttons)}'
        "</div></div>"
        '<label class="board-toggle"><input type="checkbox" id="foundation-baselines">'
        "<span>Include baselines</span></label>"
        "</div>"
    )


#: The Foundation page's SMALL gated script -- vanilla JS, no dependencies. It adds
#: ``body.js-on`` (revealing the CSS-hidden controls), wires the task selector (show only the
#: picked ``.foundation-task`` / all), and the "Include baselines" toggle (show/hide the
#: best-baseline reference blocks + the baseline-inclusive Pareto scatters, which render VISIBLE
#: by default so the info is there with JS off). It is NOT the board script: no table sort /
#: row filter / legend toggle, and it never contains the substring "sortTable".
_FOUNDATION_JS: str = """
(function () {
  document.body.classList.add('js-on');

  var sections = document.querySelectorAll('.foundation-task');
  var taskButtons = document.querySelectorAll('.foundation-task-select .board-seg');
  taskButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      taskButtons.forEach(function (b) {
        b.classList.remove('board-seg-active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('board-seg-active');
      btn.setAttribute('aria-pressed', 'true');
      var pick = btn.getAttribute('data-task') || 'all';
      sections.forEach(function (sec) {
        sec.hidden = !(pick === 'all' || sec.getAttribute('data-task') === pick);
      });
    });
  });

  var baselineToggle = document.getElementById('foundation-baselines');
  var baselineBits = document.querySelectorAll('[data-baseline-ref], [data-baseline-pareto]');
  function applyBaselines() {
    var on = baselineToggle && baselineToggle.checked;
    baselineBits.forEach(function (el) { el.classList.toggle('is-hidden', !on); });
  }
  if (baselineToggle) { baselineToggle.addEventListener('change', applyBaselines); }
  applyBaselines();  // default unchecked -> hide the baseline surfaces once JS is on

  // Shared floating tooltip for this page's scatter points (foundation.html loads no board
  // script, so the points would otherwise have only the native <title>). Mirrors the board JS.
  var tip = document.createElement('div');
  tip.className = 'chart-tooltip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  function showTip(el) {
    var model = el.getAttribute('data-model');
    if (!model) { return; }
    var x = el.getAttribute('data-x');
    var metric = el.getAttribute('data-metric');
    var y = el.getAttribute('data-y');
    tip.textContent = '';
    var nm = document.createElement('span');
    nm.className = 'tt-model';
    nm.textContent = model;
    tip.appendChild(nm);
    if (x !== null) {
      tip.appendChild(document.createElement('br'));
      tip.appendChild(document.createTextNode('x = ' + x));
    }
    tip.appendChild(document.createElement('br'));
    tip.appendChild(document.createTextNode(
      (metric ? metric + ' = ' : '') + (y !== null ? y : '')));
    var box = el.getBoundingClientRect();
    tip.style.left = (box.left + box.width / 2) + 'px';
    tip.style.top = (box.top - 8) + 'px';
    tip.style.transform = 'translate(-50%, -100%)';
    tip.classList.add('visible');
  }
  function hideTip() { tip.classList.remove('visible'); }
  document.querySelectorAll('.pt[data-model], .barplot-bar[data-model]').forEach(function (el) {
    el.addEventListener('mouseenter', function () { showTip(el); });
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('focus', function () { showTip(el); });
    el.addEventListener('blur', hideTip);
  });
})();
"""


def render_foundation_scripts() -> str:
    """Return the Foundation page's minimal gated script (``_FOUNDATION_JS``).

    A named accessor mirroring :func:`render_scripts`; injected verbatim into foundation.html's
    ``extra_body`` (inside a ``<script>`` tag). Constant content -> deterministic build.
    """
    return _FOUNDATION_JS


def render_foundation(
    grouped: dict[str, list[dict[str, Any]]], declared: dict[str, DeclaredTask]
) -> str:
    """Render the standalone Foundation Models page (``foundation.html``).

    Gives foundation-model (``model.family == "foundation"``) submissions dedicated
    prominence instead of being buried inside per-task tables next to baselines. For every
    task with >=1 foundation row: every (regime, track) group renders its OWN mini
    leaderboard ranking ONLY foundation competitors (:func:`_render_foundation_task_section`),
    plus a frugality scatter (:func:`_render_foundation_scatter`, skipped when the task has
    fewer than 2 foundation points). A global cumulative podium (:func:`_render_global_podium`)
    sums medals across every task -- the one table on this page that deliberately treats each
    result row as its own competitor (see its docstring).

    100% data-driven: no task or model name is ever hardcoded, so the page empties/fills
    itself automatically as ``leaderboard/results/**/*.json`` changes, exactly like the rest
    of the generator. Never touches :func:`render_task_page`'s own rendering, which stays
    byte-identical to before.
    """
    intro = (
        '<p class="note">Dedicated leaderboards for foundation-model submissions '
        '(<code>model.family == "foundation"</code>), ranked separately from the baselines. '
        "Each task's (regime, track) groups are ranked independently &mdash; a mini-table "
        "never mixes two regimes nor two tracks (same protocol invariant as the rest of the "
        "board).</p>"
    )

    task_sections_list: list[str] = []
    task_ids_with_foundation: list[str] = []
    for task_name in sorted(grouped, key=_task_sort_key):
        foundation_rows = _foundation_rows(grouped[task_name])
        if not foundation_rows:
            continue
        task_ids_with_foundation.append(task_name)
        baseline_rows = [r for r in grouped[task_name] if _family(r) != "foundation"]
        task_sections_list.append(
            _render_foundation_task_section(task_name, foundation_rows, declared, baseline_rows)
        )

    if not task_sections_list:
        body = (
            '<section class="task guide">'
            '<h1 class="task-title">Foundation Models</h1>'
            f"{intro}"
            '<div class="wip-card"><div class="empty-state-card">'
            '<p class="wip-kicker">Work in progress</p>'
            f"{_EMPTY_STATE_SVG}"
            '<p class="empty-state-heading">No foundation-model results yet</p>'
            '<p class="note">This page fills itself in automatically once a foundation-model '
            "result is submitted.</p>"
            "</div></div>"
            "</section>"
        )
        return _page("Foundation Models — RF-Benchmark-Hub", body, current=_FOUNDATION_SLUG)

    podium = _render_global_podium(grouped)
    controls = _render_foundation_task_selector(task_ids_with_foundation, declared)
    body = (
        '<section class="task guide">'
        '<h1 class="task-title">Foundation Models</h1>'
        f"{intro}"
        f"{controls}"
        f"{podium}"
        f"{''.join(task_sections_list)}"
        "</section>"
    )
    return _page(
        "Foundation Models — RF-Benchmark-Hub",
        body,
        current=_FOUNDATION_SLUG,
        extra_body=f"<script>{render_foundation_scripts()}</script>",
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
        '<section class="guide-section" id="contamination">'
        "<h2>Contamination badge</h2>"
        f'<p>{_esc(_GUIDE["contamination"])}</p>'
        '<p class="note">'
        '<span class="badge badge-overlap-none">clean</span> disjoint by construction &middot; '
        '<span class="badge badge-overlap-unknown">overlap unknown</span> not audited &middot; '
        '<span class="badge badge-overlap-confirmed">contaminated</span> known overlap'
        "</p>"
        "</section>"
        '<section class="guide-section" id="size-compute">'
        "<h2>Model size &amp; compute</h2>"
        f'<p>{_esc(_GUIDE["size_compute"])}</p>'
        "</section>"
        f"{_render_glossary_section()}"
        "</section>"
    )
    return _page("Guide — RF-Benchmark-Hub", body, current=_GUIDE_SLUG)


def _top_nav(current: str | None) -> str:
    """Render the site-wide top nav: Tasks | Guide | Submit, repo icon and theme toggle.

    Replaces the old per-task chip list (Home + one chip per task name) -- task-to-task
    navigation now lives in each task page's own sidebar (see ``_render_task_sidebar``), so
    this bar only needs to say which TOP-LEVEL section of the site you're in.

    ``current`` is the Guide slug (``_GUIDE_SLUG``) on the Guide page, or anything else
    (a task id, or ``None`` for the index) everywhere else -- "Tasks" is active whenever
    "Guide" isn't, since every task/WIP page lives under the Tasks section.
    """
    methods_active = current == _METHODS_SLUG
    guide_active = current == _GUIDE_SLUG
    foundation_active = current == _FOUNDATION_SLUG
    active = "top-tab top-tab-active"
    tasks_class = "top-tab" if (guide_active or methods_active or foundation_active) else active
    foundation_class = active if foundation_active else "top-tab"
    guide_class = active if guide_active else "top-tab"
    methods_class = active if methods_active else "top-tab"
    return (
        '<div class="top-tabs">'
        f'<a class="{tasks_class}" href="index.html">Tasks</a>'
        f'<a class="{foundation_class}" href="{_FOUNDATION_SLUG}.html">Foundation</a>'
        f'<a class="{guide_class}" href="{_GUIDE_SLUG}.html">Guide</a>'
        f'<a class="{methods_class}" href="{_METHODS_SLUG}.html">Methods</a>'
        '<a class="top-tab" target="_blank" rel="noopener" '
        f'href="{_esc(_SUBMISSION_GUIDE_URL)}">Submit</a>'
        "</div>"
        '<a class="icon-link" aria-label="GitHub repository" target="_blank" rel="noopener" '
        f'href="{_esc(_REPO_URL)}">{_REPO_ICON_SVG}</a>'
        '<button type="button" class="icon-link theme-toggle" '
        f'aria-label="Switch between light and dark theme">{_SUN_SVG}{_MOON_SVG}</button>'
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


def _favicon_data_uri() -> str:
    """Return a ``data:image/svg+xml`` favicon URI reusing the ``_LOGO_SVG`` shape.

    The logo markup is URL-encoded (no binary/external asset), and the theme
    ``var(--accent)`` strokes are pinned to a concrete colour so the icon renders
    standalone (outside the page's CSS scope, where custom properties are unavailable).
    """
    svg = _LOGO_SVG.replace('class="logo" ', "").replace("var(--accent)", "#2f6fed")
    return "data:image/svg+xml," + quote(svg, safe="")


#: Minimal theme-boot script injected into EVERY page's <head> (before first paint, so a
#: stored preference never flashes the wrong scheme). It applies the localStorage override to
#: ``<html data-theme=...>``, marks JS availability (``html.theme-js`` reveals the toggle
#: button -- progressive enhancement: with JS off the button stays hidden and the OS scheme
#: applies), and wires the header toggle via event delegation (the button renders later).
_THEME_JS: str = (
    "(function(){var d=document.documentElement;"
    'try{var t=localStorage.getItem("rfb-theme");'
    'if(t==="light"||t==="dark"){d.setAttribute("data-theme",t);}}catch(e){}'
    'd.classList.add("theme-js");'
    'document.addEventListener("click",function(e){'
    'var b=e.target&&e.target.closest?e.target.closest(".theme-toggle"):null;'
    "if(!b){return;}"
    'var cur=d.getAttribute("data-theme")||'
    '(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?'
    '"dark":"light");'
    'var next=cur==="dark"?"light":"dark";'
    'd.setAttribute("data-theme",next);'
    'try{localStorage.setItem("rfb-theme",next);}catch(e2){}});})();'
)

#: Sun / moon glyphs for the header theme toggle (inline stroke SVGs, no binary assets); the
#: stylesheet shows exactly one of the pair for the EFFECTIVE scheme (forced attr or OS).
_SUN_SVG: str = (
    '<svg class="icon-sun" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="4"></circle>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
    'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>'
)
_MOON_SVG: str = (
    '<svg class="icon-moon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>'
)


def _page(
    title: str,
    body: str,
    current: str | None,
    *,
    description: str = _SITE_TAGLINE,
    extra_body: str = "",
) -> str:
    """Assemble a complete standalone HTML page (header + nav + body + footer).

    ``current`` drives the site-wide top nav's active state (see ``_top_nav``): the Guide
    slug (``_GUIDE_SLUG``) on the Guide page, or anything else (a task id, or ``None`` for
    the index) everywhere else. ``description`` (keyword-only, defaults to the site tagline)
    fills the ``<meta name="description">`` and Open Graph / Twitter card tags so callers can
    pass a page-specific blurb. ``extra_body`` renders just before ``</body>`` (homepage-only
    inline filter script, see ``render_index``); it defaults to empty so every other page
    (task pages, WIP pages, the guide) is unaffected.
    """
    task_nav = _top_nav(current)
    desc = description or _SITE_TAGLINE
    meta = (
        f'<meta name="description" content="{_esc(desc)}">\n'
        f'<meta property="og:title" content="{_esc(title)}">\n'
        f'<meta property="og:description" content="{_esc(desc)}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta name="twitter:card" content="summary">\n'
        f'<link rel="icon" href="{_esc(_favicon_data_uri())}">\n'
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"{meta}"
        f"<script>{_THEME_JS}</script>\n"
        f"{_GOOGLE_FONTS_LINK}"
        f"<style>{render_styles()}</style>\n"
        "</head>\n<body>\n"
        '<header class="site-header">'
        '<div class="brand">'
        f"{_LOGO_SVG}"
        '<div class="brand-text">'
        '<span class="brand-name">RF-Benchmark-Hub</span>'
        f'<span class="brand-tag">{_esc(_SITE_TAGLINE)}</span>'
        "</div></div>"
        f"{task_nav}"
        "</header>\n"
        f"<main>\n{body}\n</main>\n"
        '<footer class="site-footer"><p>Generated by leaderboard/site/generate.py '
        "&mdash; every row validated against result.schema.json. Charts are inline SVG "
        "computed here in Python; a Google Fonts link and a small inline interactivity "
        "script (sort / filter / hover / theme) are the only external/runtime additions, and the "
        "board stays fully readable with JavaScript disabled.</p></footer>\n"
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

#: Dark-scheme design tokens, emitted TWICE by :func:`render_styles` -- once inside the
#: ``prefers-color-scheme: dark`` media query (OS preference, unless the user forced
#: light via the header toggle) and once under ``:root[data-theme="dark"]`` (forced
#: dark) -- so the dark palette lives in one single place.
_DARK_TOKENS: str = """    --bg: oklch(0.18 0.012 260);
    --surface: oklch(0.22 0.014 260);
    --surface-2: oklch(0.25 0.016 260);
    --fg: oklch(0.93 0.006 260);
    --muted: oklch(0.7 0.02 260);
    --line: oklch(0.31 0.014 260);
    --line-strong: oklch(0.38 0.016 260);
    --accent: oklch(0.72 0.15 260);
    --accent-2: oklch(0.74 0.11 200);
    --accent-soft: oklch(0.3 0.05 260);
    --head: oklch(0.25 0.016 260);
    --focus: oklch(0.72 0.15 260);
    --tooltip-bg: oklch(0.32 0.016 260);
    --tooltip-fg: oklch(0.96 0.006 260);
    --badge-verified-bg: oklch(0.3 0.06 150); --badge-verified-fg: oklch(0.8 0.14 150);
    --badge-verified-bd: oklch(0.44 0.08 150);
    --badge-self-bg: oklch(0.32 0.06 70); --badge-self-fg: oklch(0.82 0.13 75);
    --badge-self-bd: oklch(0.45 0.07 70);
    --badge-paper-bg: oklch(0.3 0.06 255); --badge-paper-fg: oklch(0.8 0.12 255);
    --badge-paper-bd: oklch(0.44 0.09 255);
    --badge-paper-uncertain-bg: oklch(0.31 0.06 300);
    --badge-paper-uncertain-fg: oklch(0.82 0.11 300);
    --badge-paper-uncertain-bd: oklch(0.45 0.08 300);
    --badge-overlap-none-bg: oklch(0.3 0.06 150); --badge-overlap-none-fg: oklch(0.8 0.14 150);
    --badge-overlap-none-bd: oklch(0.44 0.08 150);
    --badge-overlap-unknown-bg: oklch(0.32 0.06 70);
    --badge-overlap-unknown-fg: oklch(0.82 0.13 75);
    --badge-overlap-unknown-bd: oklch(0.45 0.07 70);
    --badge-overlap-confirmed-bg: oklch(0.32 0.08 25);
    --badge-overlap-confirmed-fg: oklch(0.8 0.14 25);
    --badge-overlap-confirmed-bd: oklch(0.46 0.1 25);
    --chip-baseline-bg: oklch(0.28 0.012 260); --chip-baseline-fg: oklch(0.78 0.02 260);
    --chip-baseline-bd: oklch(0.38 0.016 260);
    --chip-foundation-bg: oklch(0.3 0.06 300); --chip-foundation-fg: oklch(0.82 0.12 300);
    --chip-foundation-bd: oklch(0.46 0.09 300);
    --status-impl-bg: oklch(0.3 0.06 150); --status-impl-fg: oklch(0.8 0.14 150);
    --status-impl-bd: oklch(0.44 0.08 150);
    --status-wip-bg: oklch(0.32 0.06 70); --status-wip-fg: oklch(0.82 0.13 75);
    --status-wip-bd: oklch(0.45 0.07 70);
    --status-planned-bg: oklch(0.28 0.012 260); --status-planned-fg: oklch(0.7 0.02 260);
    --status-planned-bd: oklch(0.38 0.016 260);
    --bar-track: oklch(0.3 0.014 260); --bar-fill: oklch(0.7 0.15 260);
    --grid: oklch(0.29 0.014 260);
    --shadow-hover: 0 2px 14px rgba(0,0,0,0.35);"""

_CSS = """
/* Theme tokens in oklch (perceptually uniform): neutrals are desaturated (near-zero chroma,
   varying lightness), accents hold a constant L/C with only the hue turning per semantic
   (blue=accent, green=positive, amber=caution, red=danger, violet=info). All token NAMES are
   preserved so every existing rule + test keeps working; contrast targets AA in both schemes. */
:root {
  --bg: oklch(1 0 0);
  --surface: oklch(1 0 0);
  --surface-2: oklch(0.975 0.004 250);
  --fg: oklch(0.24 0.012 260);
  --muted: oklch(0.5 0.02 260);
  --line: oklch(0.92 0.006 260);
  --line-strong: oklch(0.85 0.01 260);
  --accent: oklch(0.53 0.2 260);
  --accent-2: oklch(0.58 0.13 200);
  --accent-soft: oklch(0.96 0.03 260);
  --head: oklch(0.965 0.004 250);
  --focus: oklch(0.53 0.2 260);
  --radius: 12px;
  --tooltip-bg: oklch(0.26 0.015 260);
  --tooltip-fg: oklch(0.97 0.004 260);
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif;
  --font-heading: "Space Grotesk", var(--font-body);
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --badge-verified-bg: oklch(0.95 0.04 150); --badge-verified-fg: oklch(0.46 0.12 150);
  --badge-verified-bd: oklch(0.82 0.08 150);
  --badge-self-bg: oklch(0.96 0.045 75); --badge-self-fg: oklch(0.52 0.11 65);
  --badge-self-bd: oklch(0.84 0.09 75);
  --badge-paper-bg: oklch(0.95 0.035 255); --badge-paper-fg: oklch(0.47 0.13 255);
  --badge-paper-bd: oklch(0.83 0.07 255);
  --badge-paper-uncertain-bg: oklch(0.95 0.035 300);
  --badge-paper-uncertain-fg: oklch(0.5 0.13 300);
  --badge-paper-uncertain-bd: oklch(0.84 0.07 300);
  --badge-overlap-none-bg: oklch(0.95 0.04 150); --badge-overlap-none-fg: oklch(0.46 0.12 150);
  --badge-overlap-none-bd: oklch(0.82 0.08 150);
  --badge-overlap-unknown-bg: oklch(0.96 0.045 75);
  --badge-overlap-unknown-fg: oklch(0.52 0.11 65);
  --badge-overlap-unknown-bd: oklch(0.84 0.09 75);
  --badge-overlap-confirmed-bg: oklch(0.95 0.04 25);
  --badge-overlap-confirmed-fg: oklch(0.52 0.19 25);
  --badge-overlap-confirmed-bd: oklch(0.83 0.09 25);
  --chip-baseline-bg: oklch(0.95 0.006 260); --chip-baseline-fg: oklch(0.42 0.02 260);
  --chip-baseline-bd: oklch(0.87 0.01 260);
  --chip-foundation-bg: oklch(0.95 0.035 300); --chip-foundation-fg: oklch(0.5 0.16 300);
  --chip-foundation-bd: oklch(0.85 0.08 300);
  --status-impl-bg: oklch(0.95 0.04 150); --status-impl-fg: oklch(0.46 0.12 150);
  --status-impl-bd: oklch(0.82 0.08 150);
  --status-wip-bg: oklch(0.96 0.045 75); --status-wip-fg: oklch(0.52 0.12 65);
  --status-wip-bd: oklch(0.84 0.09 75);
  --status-planned-bg: oklch(0.95 0.006 260); --status-planned-fg: oklch(0.5 0.02 260);
  --status-planned-bd: oklch(0.87 0.01 260);
  --bar-track: oklch(0.94 0.006 260); --bar-fill: oklch(0.55 0.2 260);
  --grid: oklch(0.945 0.005 260);
  --shadow-hover: 0 2px 10px rgba(0,0,0,0.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
__DARK_TOKENS__
  }
}
:root[data-theme="dark"] {
__DARK_TOKENS__
}
* { box-sizing: border-box; }
html { color-scheme: light dark; }
html[data-theme="light"] { color-scheme: light; }
html[data-theme="dark"] { color-scheme: dark; }
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
.hover-elevate:hover { box-shadow: var(--shadow-hover); }

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
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem;
}
.group-title > *:first-child { margin-left: 0; }
.group-dataset { color: var(--fg); }

/* Provenance / verification notes: a collapsed <details> under a group's table. Muted, small,
   fully readable with JS off; each entry links the model, shows its tier badge, and the note. */
.provenance { margin: 0.75rem 0 0; }
.provenance-summary {
  color: var(--muted); font-size: 0.8rem; cursor: pointer; font-weight: 600;
  list-style-position: inside;
}
.provenance-summary:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.provenance-list { list-style: none; margin: 0.6rem 0 0; padding: 0; }
.provenance-item {
  border-top: 1px solid var(--line); padding: 0.6rem 0 0; margin-top: 0.6rem;
}
.provenance-item:first-child { border-top: none; padding-top: 0; margin-top: 0; }
.provenance-head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
.provenance-head > *:first-child { margin-left: 0; }
.provenance-model { font-weight: 600; }
.provenance-note {
  color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin: 0.35rem 0 0;
  max-width: 78ch; overflow-wrap: anywhere;
}

/* Tier legend: a compact strip decoding the board's verification / family / contamination keys.
   Wraps gracefully; each item pairs a colour swatch (a legend sample, not a live row badge) with
   a short gloss. Fully static -- no JS needed to read it. */
.tier-legend {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem 0.9rem;
  margin: 0 0 1.25rem; font-size: 0.78rem; color: var(--muted);
}
.tl-group {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.65rem;
  padding-right: 0.9rem; border-right: 1px solid var(--line);
}
.tl-item { display: inline-flex; align-items: center; gap: 0.32rem; }
.tl-swatch { font-size: 0.68rem; padding: 0.02rem 0.45rem; }
.tl-gloss { color: var(--muted); }
.tl-guide { font-weight: 600; white-space: nowrap; }

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
td.num.size, th.num.size { white-space: nowrap; }
.size-params {
  display: block; color: var(--fg); font-size: 0.82rem; font-variant-numeric: tabular-nums;
}
.size-flops {
  display: block; margin-top: 0.05rem; color: var(--muted); font-size: 0.66rem;
  font-variant-numeric: tabular-nums;
}
.bar {
  display: block; height: 5px; width: 100%; max-width: 90px; margin: 0.25rem 0 0 auto;
  background: var(--bar-track); border-radius: 999px; overflow: hidden;
}
.bar-fill { display: block; height: 100%; background: var(--bar-fill); }

.badge, .chip, .tl-swatch {
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
.badge-overlap-none {
  margin-left: 0.4rem;
  background: var(--badge-overlap-none-bg); color: var(--badge-overlap-none-fg);
  border-color: var(--badge-overlap-none-bd);
}
.badge-overlap-unknown {
  margin-left: 0.4rem;
  background: var(--badge-overlap-unknown-bg); color: var(--badge-overlap-unknown-fg);
  border-color: var(--badge-overlap-unknown-bd);
}
.badge-overlap-confirmed {
  margin-left: 0.4rem;
  background: var(--badge-overlap-confirmed-bg); color: var(--badge-overlap-confirmed-fg);
  border-color: var(--badge-overlap-confirmed-bd);
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
.plot .barplot-bar { fill: var(--accent); }
.plot .errbar { stroke: var(--fg); stroke-width: 1.4; }
.plot .bar-val { fill: var(--fg); font-size: 10px; font-family: var(--font-mono); }
.plot .bar-label { fill: var(--muted); font-size: 11px; font-family: var(--font-mono); }
.plot .pareto-front { opacity: 0.85; }
.plot-dir { color: var(--muted); font-weight: 400; font-size: 0.78rem; }
.plot-note { margin-top: 0.4rem; font-size: 0.76rem; }
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
.theme-toggle { display: none; border: none; background: none; padding: 0; cursor: pointer; }
html.theme-js .theme-toggle { display: inline-flex; }
.theme-toggle .icon-sun { display: none; }
html[data-theme="dark"] .theme-toggle .icon-sun { display: block; }
html[data-theme="dark"] .theme-toggle .icon-moon { display: none; }
@media (prefers-color-scheme: dark) {
  html:not([data-theme="light"]) .theme-toggle .icon-sun { display: block; }
  html:not([data-theme="light"]) .theme-toggle .icon-moon { display: none; }
}

.hero-eyebrow {
  color: var(--accent); font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; margin: 0.5rem 0 0.5rem;
}
.hero-title {
  font-size: 2.1rem; margin: 0 0 0.6rem; letter-spacing: -0.02em;
}
.hero-lead { color: var(--muted); font-size: 0.95rem; max-width: 68ch; margin: 0 0 1.5rem; }
.hero-cta { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0 0 1.5rem; }
.hero-cta-secondary {
  display: inline-block; font-size: 0.85rem; font-weight: 600; padding: 0.5rem 0.9rem;
  border-radius: 8px; border: 1px solid var(--line-strong); color: var(--fg);
}
.hero-cta-secondary:hover {
  text-decoration: none; border-color: var(--accent); color: var(--accent);
}

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
.stat-card-verified {
  border-color: var(--badge-verified-bd); background: var(--badge-verified-bg);
}
.stat-card-verified .stat-value { color: var(--badge-verified-fg); }
.card-verified-cov {
  display: inline-block; margin-left: 0.5rem; padding: 0.05rem 0.4rem; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; background: var(--badge-verified-bg);
  color: var(--badge-verified-fg); border: 1px solid var(--badge-verified-bd);
}
.card-verified-cov-zero {
  background: transparent; color: var(--muted); border-color: var(--line); font-weight: 500;
}

.filter-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin: 0 0 1.5rem;
}
.search-input {
  flex: 1 1 260px; min-width: 200px; padding: 0.5rem 0.85rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); color: var(--fg);
  font-family: var(--font-body); font-size: 0.88rem;
}
/* Keep the same 2px focus ring the rest of the site uses (never `outline: none`); the accent
   border stays as an extra cue. */
.search-input:focus-visible {
  outline: 2px solid var(--focus); outline-offset: 1px; border-color: var(--accent);
}
.filter-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.filter-pill {
  font-family: var(--font-body); font-size: 0.82rem; padding: 0.35rem 0.8rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--surface);
  color: var(--fg); cursor: pointer;
}
.filter-pill:hover { border-color: var(--line-strong); }
.filter-pill:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
.filter-pill-active {
  background: var(--fg); color: var(--bg); border-color: var(--fg);
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
  .task-main { order: -1; }
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
.cmd {
  font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.5; color: var(--fg);
  background: var(--surface-2); border: 1px solid var(--line); border-radius: 8px;
  padding: 0.6rem 0.7rem; margin: 0 0 0.9rem; white-space: pre-wrap; overflow-wrap: anywhere;
}

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
  border-color: var(--accent); text-decoration: none; box-shadow: var(--shadow-hover);
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
.task-main > .task-header-grid { margin: 0 0 1.1rem; }
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

/* Progressive enhancement: interactivity controls are hidden until the board script marks
   <body class="js-on">, so no dead control shows when JavaScript is off. */
.board-controls { display: none; }
body.js-on .board-controls { display: flex; }
body.js-on .home-sort { display: inline-flex; }
.home-sort { display: none; }

/* Per-task-page interactive controls bar (search + verified toggle + family segmented). */
.board-controls {
  flex-wrap: wrap; align-items: center; gap: 0.6rem 0.9rem; margin: 0 0 1.25rem;
}
.board-search {
  flex: 1 1 220px; min-width: 160px; padding: 0.45rem 0.8rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); color: var(--fg);
  font-family: var(--font-body); font-size: 0.85rem;
}
.board-search:focus {
  outline: 2px solid var(--focus); outline-offset: 1px; border-color: var(--accent);
}
.board-toggle {
  display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.82rem;
  color: var(--muted); cursor: pointer; user-select: none;
}
.board-toggle input { accent-color: var(--accent); }
.board-segmented {
  display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden;
}
.board-seg {
  font-family: var(--font-body); font-size: 0.8rem; padding: 0.35rem 0.75rem; border: none;
  background: var(--surface); color: var(--muted); cursor: pointer;
}
.board-seg + .board-seg { border-left: 1px solid var(--line); }
.board-seg:hover { color: var(--fg); background: var(--surface-2); }
.board-seg-active { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.board-seg:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
.no-match td { color: var(--muted); font-style: italic; text-align: center; }

/* Multi-dataset selector (task pages) + Foundation task selector / baselines toggle.
   Progressive enhancement: both bars are hidden until body.js-on (no dead control with JS off),
   and NEITHER emits any rule that hides a .group / .foundation-task by default -- so with JS off
   every dataset group and every task section stays visible. */
.dataset-selector, .foundation-controls { display: none; }
body.js-on .dataset-selector { display: flex; }
body.js-on .foundation-controls { display: flex; }
.dataset-selector, .foundation-controls {
  flex-wrap: wrap; align-items: center; gap: 0.5rem 0.9rem; margin: 0 0 1.25rem;
}
.dataset-selector-label {
  font-size: 0.8rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.03em;
}
.dataset-selector .board-seg[aria-pressed="true"],
.foundation-task-select .board-seg[aria-pressed="true"] {
  background: var(--accent-soft); color: var(--accent); font-weight: 600;
}
.foundation-task-select { display: inline-flex; align-items: center; gap: 0.5rem; }

/* Best-baseline reference block (Foundation page). Rendered VISIBLE by default (usable with JS
   off); the "Include baselines" toggle hides it via .is-hidden when JS is on and unchecked. */
.is-hidden { display: none; }
.baseline-reference {
  margin: 0.75rem 0 1.25rem; padding: 0.85rem 1rem; border: 1px dashed var(--line-strong);
  border-radius: var(--radius); background: var(--surface-2);
}
.baseline-reference-title {
  margin: 0 0 0.35rem; font-family: var(--font-heading); font-weight: 600; font-size: 0.92rem;
}
.baseline-reference-list { list-style: none; margin: 0.5rem 0 0; padding: 0; }
.baseline-ref-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.5rem;
  padding: 0.3rem 0; border-top: 1px solid var(--line);
}
.baseline-ref-row:first-child { border-top: none; }
.baseline-ref-tag {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
  color: var(--muted);
}
.baseline-ref-model { font-family: var(--font-mono); font-size: 0.85rem; }
.baseline-ref-score { margin-left: auto; font-family: var(--font-mono); font-size: 0.85rem; }
.baseline-pareto { margin-top: 0.5rem; }

/* Sortable table headers: the caret + pointer + hover affordance are GATED behind body.js-on,
   exactly like .board-controls -- with JS off the click/keydown handlers never bind, so the
   header must not advertise a sort it cannot perform (no dead affordance). The focus ring is
   kept unconditionally so keyboard users still get a visible focus outline. */
th[data-sortable] { position: relative; user-select: none; cursor: default; }
th[data-sortable]:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
body.js-on th[data-sortable] { cursor: pointer; }
body.js-on th[data-sortable]:hover { color: var(--fg); }
.sort-caret {
  display: none; width: 0.7em; margin-left: 0.3em; opacity: 0.35;
  font-size: 0.85em; vertical-align: middle;
}
body.js-on .sort-caret { display: inline-block; }
.sort-caret::after { content: "\\2195"; }
th[aria-sort="ascending"] .sort-caret { opacity: 1; }
th[aria-sort="ascending"] .sort-caret::after { content: "\\2191"; }
th[aria-sort="descending"] .sort-caret { opacity: 1; }
th[aria-sort="descending"] .sort-caret::after { content: "\\2193"; }
th[aria-sort="ascending"], th[aria-sort="descending"] { color: var(--accent); }

/* Chart axis titles + interactive series/point/legend affordances. */
.plot .axis-title {
  fill: var(--muted); font-size: 11px; font-family: var(--font-heading); font-weight: 600;
}
.plot .pt { cursor: pointer; transition: r .1s ease; }
/* SVG `outline` is not painted by WebKit/Safari on graphic elements; pair it with a `stroke`
   ring (rendered by every engine) so the focus indicator stays visible everywhere (WCAG 2.4.7). */
.plot .pt:focus-visible { outline: 2px solid var(--focus); stroke: var(--focus); stroke-width: 2; }
g.series.series-off { display: none; }
g.series.series-hi polyline { stroke-width: 3.5; }
g.series.series-hi .pt { stroke: var(--fg); stroke-width: 1; }
.plot .barplot-bar { cursor: pointer; }
/* Same WebKit `outline`-not-painted fallback as the points: add a stroke ring (WCAG 2.4.7). */
.plot .barplot-bar:focus-visible {
  outline: 2px solid var(--focus); stroke: var(--focus); stroke-width: 2;
}
.legend-item {
  cursor: pointer; background: none; border: 1px solid transparent; border-radius: 8px;
  padding: 0.15rem 0.4rem; font-family: var(--font-body);
}
.legend-item[aria-pressed="true"] { opacity: 0.4; }
.legend-item:hover, .legend-item.legend-hover { background: var(--surface-2); }
.legend-item:focus-visible { outline: 2px solid var(--focus); }

/* Single shared floating tooltip (created once by the board script). */
.chart-tooltip {
  position: fixed; z-index: 40; pointer-events: none; max-width: 260px;
  background: var(--tooltip-bg); color: var(--tooltip-fg);
  border-radius: 8px; padding: 0.4rem 0.6rem; font-size: 0.78rem; line-height: 1.35;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25); opacity: 0; transition: opacity .08s ease;
}
.chart-tooltip.visible { opacity: 1; }
.chart-tooltip .tt-model { font-weight: 700; }
.chart-tooltip .tt-ci { color: var(--tooltip-fg); opacity: 0.75; }

/* Home card sort control (segmented alongside the existing status pills). */
.home-sort { align-items: center; gap: 0.4rem; margin-left: auto; }
.home-sort-label { color: var(--muted); font-size: 0.8rem; }
.home-sort select {
  font-family: var(--font-body); font-size: 0.82rem; padding: 0.35rem 0.6rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--surface);
  color: var(--fg); cursor: pointer;
}
.home-sort select:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }

/* Foundation Models page: per-task foundation-only podiums + the global cumulative table. */
.chip-regime {
  background: var(--surface-2); color: var(--muted); border-color: var(--line-strong);
}
.chip-track {
  background: var(--accent-soft); color: var(--accent); border-color: var(--accent);
}
td.tags { display: flex; flex-wrap: wrap; gap: 0.3rem 0.4rem; align-items: center; }
td.tags > *:first-child { margin-left: 0; }
.medal { font-size: 1.1rem; }
.foundation-task { margin: 0 0 2.5rem; }
.foundation-task-title { font-size: 1.15rem; margin: 0 0 0.75rem; }
.foundation-task-title a { color: var(--fg); }
.global-podium { margin-bottom: 2rem; }

/* Respect a user's reduced-motion preference: neutralise the (small) hover/tooltip
   transitions -- WCAG 2.3.3. Focus rings and layout are unaffected. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition-duration: 0.001ms !important;
    animation-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}
"""


def render_styles() -> str:
    """Return the site's full CSS (the ``_CSS`` theme + component rules) as one string.

    A thin accessor so callers (``_page``) don't reach into the module constant directly and
    the stylesheet has a single named entry point. The content is a constant, so the output is
    deterministic (idempotent build).
    """
    return _CSS.replace("__DARK_TOKENS__", _DARK_TOKENS)


#: The per-task-page interactive board script -- vanilla JS, no dependencies, no build step.
#: GENERIC + data-driven: it reads the semantic ``data-*`` hooks the generator emits and never
#: names a task or metric. It (1) sorts a table's OWN tbody on a header click/Enter/Space
#: (numeric via ``data-value``, alpha for the Model column; stable; caret + aria-sort updated),
#: (2) filters ``tr[data-model]`` across every table from the controls bar (search / verified /
#: family; masking rows preserves the one-regime-per-table invariant), (3) toggles chart series
#: via the legend buttons, and (4) shows a single shared tooltip on point/bar hover+focus.
#: Progressive enhancement: it adds ``body.js-on`` FIRST (revealing the controls, which are CSS-
#: hidden otherwise), so with JS off the tables + charts stay fully readable and no dead control
#: is shown. NOTE: this text must never contain the substring "line plot" (a scalar-only page
#: asserts that substring is absent when it draws no curve).
_BOARD_JS: str = """
(function () {
  document.body.classList.add('js-on');

  // ---- Sortable tables (each table sorts only its OWN tbody = one (regime, track) group) ----
  function cellValue(row, index, kind) {
    var cell = row.children[index];
    if (!cell) { return kind === 'num' ? -Infinity : ''; }
    if (kind === 'num') {
      var raw = cell.getAttribute('data-value');
      if (raw === null) { return -Infinity; }
      var num = parseFloat(raw);
      return isNaN(num) ? -Infinity : num;
    }
    // Text key = the clean model name only (a .model-name span), never the whole cell -- the
    // family chip / params / badges in the Model cell must not pollute the alphabetical sort.
    var nameEl = cell.querySelector('.model-name');
    return ((nameEl ? nameEl.textContent : cell.textContent) || '').trim().toLowerCase();
  }
  function sortTable(table, th) {
    var headRow = th.parentNode;
    var index = Array.prototype.indexOf.call(headRow.children, th);
    var kind = th.getAttribute('data-sort') || 'text';
    var current = th.getAttribute('aria-sort');
    var dir = current === 'ascending' ? 'descending' : 'ascending';
    var tbody = table.tBodies[0];
    if (!tbody) { return; }
    var rows = Array.prototype.filter.call(tbody.rows, function (r) {
      return r.hasAttribute('data-model');
    });
    var decorated = rows.map(function (r, i) {
      return { r: r, i: i, v: cellValue(r, index, kind) };
    });
    decorated.sort(function (a, b) {
      if (a.v < b.v) { return dir === 'ascending' ? -1 : 1; }
      if (a.v > b.v) { return dir === 'ascending' ? 1 : -1; }
      return a.i - b.i;  // stable
    });
    decorated.forEach(function (d) { tbody.appendChild(d.r); });
    var extra = Array.prototype.filter.call(tbody.rows, function (r) {
      return !r.hasAttribute('data-model');
    });
    extra.forEach(function (r) { tbody.appendChild(r); });  // keep the no-match row last
    headRow.querySelectorAll('th[data-sortable]').forEach(function (h) {
      h.setAttribute('aria-sort', 'none');
    });
    th.setAttribute('aria-sort', dir);
  }
  document.querySelectorAll('table[data-leaderboard] th[data-sortable]').forEach(function (th) {
    var table = th.closest('table');
    th.addEventListener('click', function () { sortTable(table, th); });
    th.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sortTable(table, th); }
    });
  });

  // ---- Controls bar: filter rows across every table on the page ----
  var search = document.getElementById('board-search');
  var verifiedOnly = document.getElementById('board-verified-only');
  var segButtons = document.querySelectorAll('.board-seg');
  var family = 'all';
  var tables = document.querySelectorAll('table[data-leaderboard]');
  function applyRowFilters() {
    var q = (search && search.value || '').trim().toLowerCase();
    tables.forEach(function (table) {
      var tbody = table.tBodies[0];
      if (!tbody) { return; }
      var shown = 0;
      Array.prototype.forEach.call(tbody.rows, function (row) {
        if (!row.hasAttribute('data-model')) { return; }
        var name = (row.getAttribute('data-model') || '').toLowerCase();
        var fam = row.getAttribute('data-family') || '';
        var ver = row.getAttribute('data-verified') === 'true';
        var ok = (!q || name.indexOf(q) !== -1)
          && (family === 'all' || fam === family)
          && (!(verifiedOnly && verifiedOnly.checked) || ver);
        row.hidden = !ok;
        if (ok) { shown++; }
      });
      var note = tbody.querySelector('.no-match-row');
      if (note) { note.hidden = shown !== 0; }
    });
  }
  if (search) { search.addEventListener('input', applyRowFilters); }
  if (verifiedOnly) { verifiedOnly.addEventListener('change', applyRowFilters); }
  segButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      segButtons.forEach(function (b) {
        b.classList.remove('board-seg-active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('board-seg-active');
      btn.setAttribute('aria-pressed', 'true');
      family = btn.getAttribute('data-family') || 'all';
      applyRowFilters();
    });
  });

  // ---- Multi-dataset selector: show only the picked dataset's .group sections ----
  // Rendered ONLY when the task has >1 dataset (else the selector is absent). With JS off every
  // .group stays visible (no default-hidden CSS); this handler just narrows the view when JS is on.
  var datasetButtons = document.querySelectorAll('.dataset-selector .board-seg');
  var groupSections = document.querySelectorAll('section.group[data-dataset]');
  datasetButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      datasetButtons.forEach(function (b) {
        b.classList.remove('board-seg-active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('board-seg-active');
      btn.setAttribute('aria-pressed', 'true');
      var pick = btn.getAttribute('data-dataset');
      groupSections.forEach(function (sec) {
        sec.hidden = sec.getAttribute('data-dataset') !== pick;
      });
    });
  });
  // Apply the default (first dataset) selection once, so a multi-dataset page opens focused.
  if (datasetButtons.length > 1) {
    var firstActive = document.querySelector('.dataset-selector .board-seg-active');
    if (firstActive) {
      var initPick = firstActive.getAttribute('data-dataset');
      groupSections.forEach(function (sec) {
        sec.hidden = sec.getAttribute('data-dataset') !== initPick;
      });
    }
  }

  // ---- Chart legend toggles (hide/show a series) + hover highlight ----
  function seriesFor(btn) {
    var name = btn.getAttribute('data-series');
    var svg = btn.closest('.plot-figure').querySelector('svg.plot');
    if (!svg) { return null; }
    var groups = svg.querySelectorAll('g.series[data-series]');
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].getAttribute('data-series') === name) { return groups[i]; }
    }
    return null;
  }
  document.querySelectorAll('.legend-item[data-series]').forEach(function (btn) {
    var group = seriesFor(btn);
    btn.addEventListener('click', function () {
      if (!group) { return; }
      var off = group.classList.toggle('series-off');
      btn.setAttribute('aria-pressed', off ? 'true' : 'false');
    });
    btn.addEventListener('mouseenter', function () {
      if (group) { group.classList.add('series-hi'); }
    });
    btn.addEventListener('mouseleave', function () {
      if (group) { group.classList.remove('series-hi'); }
    });
  });

  // ---- Single shared tooltip for points + bars (hover AND focus) ----
  var tip = document.createElement('div');
  tip.className = 'chart-tooltip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  function showTip(el) {
    var model = el.getAttribute('data-model');
    if (!model) { return; }
    var x = el.getAttribute('data-x');
    var metric = el.getAttribute('data-metric');
    var y = el.getAttribute('data-y');
    var lo = el.getAttribute('data-ci-low');
    var hi = el.getAttribute('data-ci-high');
    // Build the tooltip from DOM nodes + textContent -- never an HTML-string sink: data-model /
    // data-metric are contributor-controlled (result.json) and getAttribute returns the DECODED
    // value, so re-parsing it as markup would turn '<img onerror=...>' into live DOM (stored XSS).
    tip.textContent = '';
    var name = document.createElement('span');
    name.className = 'tt-model';
    name.textContent = model;
    tip.appendChild(name);
    if (x !== null) {
      tip.appendChild(document.createElement('br'));
      tip.appendChild(document.createTextNode('x = ' + x));
    }
    tip.appendChild(document.createElement('br'));
    var yline = (metric ? metric + ' = ' : '') + (y !== null ? y : '');
    tip.appendChild(document.createTextNode(yline));
    if (lo !== null && hi !== null) {
      var ci = document.createElement('span');
      ci.className = 'tt-ci';
      ci.textContent = ' CI [' + lo + ', ' + hi + ']';
      tip.appendChild(ci);
    }
    var box = el.getBoundingClientRect();
    tip.style.left = (box.left + box.width / 2) + 'px';
    tip.style.top = (box.top - 8) + 'px';
    tip.style.transform = 'translate(-50%, -100%)';
    tip.classList.add('visible');
  }
  function hideTip() { tip.classList.remove('visible'); }
  document.querySelectorAll('.pt[data-model], .barplot-bar[data-model]').forEach(function (el) {
    el.addEventListener('mouseenter', function () { showTip(el); });
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('focus', function () { showTip(el); });
    el.addEventListener('blur', hideTip);
  });
})();
"""


def render_scripts() -> str:
    """Return the per-task-page interactive board script (``_BOARD_JS``) as a string.

    A named accessor mirroring :func:`render_styles`; the script is injected verbatim into the
    task page's ``extra_body`` (inside a ``<script>`` tag). Constant content -> deterministic.
    """
    return _BOARD_JS


#: Homepage-only inline search/filter script -- vanilla JS, no dependencies, no build step.
#: Degrades gracefully with JS disabled: cards carry no default ``display:none``, so every
#: card stays visible and the search box / pills are simply inert. Injected via ``_page``'s
#: ``extra_body`` kwarg (``render_index`` only); every other page has zero ``<script>``.
_JS: str = """
(function () {
  document.body.classList.add('js-on');
  var search = document.getElementById('task-search');
  var pills = document.querySelectorAll('.filter-pill');
  var sortSelect = document.getElementById('task-sort');
  var cards = document.querySelectorAll('.task-card');
  var grids = document.querySelectorAll('.card-grid');
  var activeStatus = 'all';

  function applyFilters() {
    var query = (search && search.value || '').trim().toLowerCase();
    cards.forEach(function (card) {
      var matchesStatus = activeStatus === 'all' || card.dataset.status === activeStatus;
      var matchesQuery = !query || card.textContent.toLowerCase().indexOf(query) !== -1;
      card.style.display = (matchesStatus && matchesQuery) ? '' : 'none';
    });
  }

  function priorityRank(card) {
    var p = (card.getAttribute('data-priority') || '').toUpperCase();
    var m = p.match(/P(\\d+)/);
    return m ? parseInt(m[1], 10) : 99;  // P1 first; unprioritised last
  }
  function num(card, attr) { return parseFloat(card.getAttribute(attr) || '0') || 0; }
  function sortCards() {
    var mode = sortSelect ? sortSelect.value : 'default';
    // Sort WITHIN each grid only -- a card never leaves its scope section.
    grids.forEach(function (grid) {
      var items = Array.prototype.slice.call(grid.querySelectorAll('.task-card'));
      var decorated = items.map(function (c, i) { return { c: c, i: i }; });
      decorated.sort(function (a, b) {
        var d = 0;
        if (mode === 'priority') { d = priorityRank(a.c) - priorityRank(b.c); }
        else if (mode === 'results') { d = num(b.c, 'data-results') - num(a.c, 'data-results'); }
        else if (mode === 'verified') { d = num(b.c, 'data-verified') - num(a.c, 'data-verified'); }
        return d !== 0 ? d : a.i - b.i;  // stable, and 'default' keeps DOM order
      });
      decorated.forEach(function (d) { grid.appendChild(d.c); });
    });
  }

  if (search) { search.addEventListener('input', applyFilters); }
  if (sortSelect) { sortSelect.addEventListener('change', sortCards); }
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

    # Extract the per-method explanations ONCE (ast over the model sources, no torch import) so the
    # leaderboard can link every no-paper method to its docstring on the Methods page.
    global _METHOD_DOCS, _MODEL_PAPER_URLS
    _METHOD_DOCS = _extract_method_docs()

    rows = load_results(results_path)
    # Map each model to its result.json paper URL (if any) so the Methods page can surface it next
    # to the arXiv/DOI citations parsed from the docstring.
    _MODEL_PAPER_URLS = {
        row["model"]["name"]: row["model"]["url"]
        for row in rows
        if isinstance(row.get("model"), dict) and row["model"].get("url")
    }
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

    # Shared educational Guide + Methods pages (linked from the nav on every page).
    (out_path / f"{_GUIDE_SLUG}.html").write_text(render_guide(), encoding="utf-8")
    (out_path / f"{_METHODS_SLUG}.html").write_text(render_methods_page(), encoding="utf-8")
    (out_path / f"{_FOUNDATION_SLUG}.html").write_text(
        render_foundation(grouped, declared), encoding="utf-8"
    )

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
