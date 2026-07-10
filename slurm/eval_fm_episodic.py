"""Episodic few-shot evaluation of a foundation model on AMC -> one aggregated board row.

The ``few_shot(k)`` regime is, per ``docs/EVALUATION_PROTOCOL.md`` ("Statistical rigor &
uncertainty"), an N>=10 *episodic* protocol: for a fixed shot count ``k`` we draw ``N``
independent k-per-class support sets (seeds ``42..42+N-1``), fit a frozen-backbone head on
each, evaluate every episode on the FULL test split, and report the mean scalar metrics
plus a descriptive spread across episodes. A single ``FewShotAdapter`` is mono-episode by
design (one instance == one seed == one declared regime row); this module is the
aggregation layer that turns ``N`` such episodes into ONE ``result.json`` board row.

Design (keeps :func:`rfbench.core.evaluate.evaluate` the CANONICAL writer):

1. For each seed we build ``FewShotAdapter(k, fresh_head, seed=seed)``, fit it on the
   (materialised-once) train split, wrap it in the same ``_AdaptedModel`` bridge
   :func:`~rfbench.models.foundation.base.run_regime` uses, and call ``evaluate(...)`` with
   a per-seed staging ``out_path``. That per-seed file is a fully schema-valid row carrying
   its OWN bootstrap CI (from ``evaluate``) -- the staging artifacts the conventions want
   kept under ``$WORK/logs/multiseed/``.
2. We aggregate the ``metrics.values`` across the ``N`` staged episodes: each scalar
   metric's board value is the mean over episodes, and its ``metrics.uncertainty`` entry is
   the DESCRIPTIVE +/- 1 sample-stdev (``statistics.stdev``, ddof = N-1) turned into an
   interval, ``method = "multi_seed_std"``, ``n_episodes = N``. This is explicitly NOT a 95%
   confidence interval (the note says so); the per-episode bootstrap CIs remain available in
   the staging files.
3. The board row is assembled from one episode's schema-valid document (so the split /
   task / model / regime blocks are verbatim what ``evaluate`` emits), with ``metrics``
   overwritten by the aggregate and the single-episode ``curves`` dropped (an aggregated row
   must not advertise one episode's accuracy-vs-SNR curve as if it were the mean). The
   mutated document is RE-VALIDATED against ``schemas/result.schema.json`` before it lands,
   mirroring ``scripts/backfill_uncertainty.py``.

This module imports ``torch`` transitively (through the FM backbone at fit/eval time) and is
meant to run on the cluster ARM/GPU node; it is NOT importable on the dependency-free
frontend. Only the aggregation helpers (:func:`aggregate_episode_values`,
:func:`build_board_row`) are pure-stdlib and unit-testable in isolation.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger("rfbench.eval_fm_episodic")

#: Protocol-mandated episode count for a few-shot report (EVALUATION_PROTOCOL.md, N >= 10).
N_EPISODES = 10

#: Base seed for the episode range; episodes use seeds ``BASE_SEED .. BASE_SEED+N-1``
#: (42..51 for N == 10), mirroring ``rfbench.regimes.few_shot.DEFAULT_FEW_SHOT_SEED``.
BASE_SEED = 42

#: The aggregate uncertainty method (schema 1.2.0 enum): a std across episodes, NOT a
#: resampling CI. Kept in lockstep with ``schemas/result.schema.json``.
_MULTI_SEED_METHOD = "multi_seed_std"

#: Descriptive-spread note attached to every aggregated metric's uncertainty block. Spells
#: out that this is a +/- 1 sample-stdev DESCRIPTIVE spread over the episode seeds, NOT a 95%
#: confidence interval, and that per-episode bootstrap CIs remain in the staging files.
_SPREAD_NOTE = (
    "plus/moins 1 ecart-type DESCRIPTIF sur les {n} episodes few-shot "
    "(seeds {lo}..{hi}, statistics.stdev ddof=n-1), PAS un intervalle de confiance a 95% ; "
    "les IC bootstrap par episode restent disponibles dans les fichiers de staging "
    "$WORK/logs/multiseed/<task>/k<k>/<model>-seed<seed>.json"
)

#: Metric keys whose values are genuine proportions in [0, 1] (AMC accuracy / macro-F1);
#: their descriptive interval bounds are clamped to [0, 1] so the aggregated row stays valid
#: against the schema's ``minimum: 0 / maximum: 1`` on these keys.
_PROPORTION_METRICS = frozenset(
    {"accuracy_overall", "macro_f1", "rank1_accuracy", "auroc", "eer", "mAP", "mAR", "IoU"}
)


def _clamp_unit(value: float, key: str) -> float:
    """Clamp ``value`` to [0, 1] iff ``key`` is a bounded-proportion metric, else pass through.

    The mean +/- 1 stdev bounds of a proportion metric can fall marginally outside [0, 1]
    (e.g. a near-perfect accuracy with a small spread), which the schema rejects for the
    proportion keys. Clamping keeps the descriptive interval honest (it never widens it) and
    schema-valid; non-proportion metrics (e.g. ``latency_ms``) are returned untouched.
    """
    if key not in _PROPORTION_METRICS:
        return value
    return max(0.0, min(1.0, value))


def aggregate_episode_values(
    per_episode_values: list[dict[str, float]],
    *,
    n_episodes: int = N_EPISODES,
    base_seed: int = BASE_SEED,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Aggregate per-episode scalar metrics into board ``values`` + ``uncertainty`` blocks.

    ``per_episode_values`` is one ``metrics.values`` dict per episode (as returned by
    :func:`rfbench.core.evaluate.evaluate`), all sharing the same metric keys. For every key
    present in EVERY episode this returns:

    * ``values[key]``  = the mean over episodes;
    * ``uncertainty[key]`` = ``{ci_low, ci_high, method="multi_seed_std", n_episodes, note}``
      where ``ci_low = mean - stdev`` and ``ci_high = mean + stdev`` (``statistics.stdev``,
      i.e. the ddof = n-1 sample standard deviation), clamped to [0, 1] for proportion
      metrics. No ``confidence`` field is emitted -- a +/- 1 stdev spread is NOT a 95%
      interval, and the ``note`` says so explicitly.

    Raises ``ValueError`` on an empty episode list or a metric key that is not shared by all
    episodes (a partial aggregate would silently drop metrics).
    """
    if not per_episode_values:
        raise ValueError("no episodes to aggregate")
    if len(per_episode_values) < 2:
        raise ValueError(
            f"multi_seed_std needs >= 2 episodes to compute a stdev, got {len(per_episode_values)}"
        )

    shared_keys = set(per_episode_values[0])
    for episode in per_episode_values[1:]:
        shared_keys &= set(episode)
    if not shared_keys:
        raise ValueError("episodes share no common metric key to aggregate")

    n = len(per_episode_values)
    note = _SPREAD_NOTE.format(n=n, lo=base_seed, hi=base_seed + n_episodes - 1)

    values: dict[str, float] = {}
    uncertainty: dict[str, dict[str, Any]] = {}
    for key in sorted(shared_keys):
        draws = [float(episode[key]) for episode in per_episode_values]
        mean = statistics.fmean(draws)
        stdev = statistics.stdev(draws)  # sample stdev, ddof = n - 1
        values[key] = mean
        uncertainty[key] = {
            "ci_low": _clamp_unit(mean - stdev, key),
            "ci_high": _clamp_unit(mean + stdev, key),
            "method": _MULTI_SEED_METHOD,
            "n_episodes": n_episodes,
            "note": note,
        }
    return values, uncertainty


def build_board_row(
    template: dict[str, Any],
    values: dict[str, float],
    uncertainty: dict[str, dict[str, Any]],
    *,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Assemble the aggregated ``few_shot`` board row from a single-episode ``template``.

    ``template`` is one episode's schema-valid ``result.json`` dict (as returned by
    ``evaluate``); its ``task`` / ``model`` / ``regime`` / ``dataset`` / ``split`` /
    ``verification`` blocks are reused verbatim (the regime -- including ``k_shot`` -- is
    already declared correctly by ``evaluate``). Only ``metrics`` is rewritten:

    * ``metrics.values``      -> the episode-mean ``values``;
    * ``metrics.uncertainty`` -> the ``multi_seed_std`` ``uncertainty`` block;
    * ``metrics.curves``      -> DROPPED (a single episode's accuracy-vs-SNR curve must not
      masquerade as the mean on an aggregated row).

    ``metrics.primary`` is preserved and, as a guard, must remain a key of the aggregated
    ``values`` (else the board could point at a metric we did not aggregate). The reported
    ``environment.seed`` is set to ``base_seed`` (the first episode's seed) so the row's
    fingerprint names the base of the episode range rather than the last episode's seed.

    The returned dict is a DEEP-ish copy (``metrics`` is rebuilt, other blocks are shared
    references from ``template``); callers validate it before writing. Raises ``ValueError``
    if the template's primary metric is absent from ``values``.
    """
    metrics_in = template["metrics"]
    primary = str(metrics_in["primary"])
    if primary not in values:
        raise ValueError(
            f"primary metric '{primary}' absent from aggregated values "
            f"(got: {', '.join(sorted(values)) or '<none>'})"
        )

    row: dict[str, Any] = dict(template)
    row["metrics"] = {
        "primary": primary,
        "values": values,
        "uncertainty": uncertainty,
    }
    # Aggregated-row fingerprint names the base seed of the episode range, not the last one.
    environment = dict(template.get("environment", {}))
    environment["seed"] = base_seed
    row["environment"] = environment
    return row


def _repo_schema_path() -> Path:
    """Locate ``schemas/result.schema.json`` in THIS script's repo checkout.

    Deliberately resolves relative to this file (like
    ``scripts/backfill_uncertainty.py``) rather than to the imported ``rfbench`` package: an
    editable install may point ``rfbench`` at a DIFFERENT worktree whose schema predates
    1.2.0, which would spuriously reject the (valid) aggregated row. We always validate
    against the schema living next to the results we are writing.

    Raises ``RuntimeError`` if the repo schema cannot be found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schemas" / "result.schema.json"
        if candidate.is_file():
            return candidate
    raise RuntimeError("could not locate schemas/result.schema.json next to this script")


def _validate_or_raise(document: dict[str, Any]) -> None:
    """Validate ``document`` against this repo's ``result.schema.json`` or raise.

    Pins the schema to this repo (see :func:`_repo_schema_path`) so the check is independent
    of where ``rfbench`` happens to be installed -- mirrors
    ``scripts/backfill_uncertainty._validate_or_raise``.
    """
    from jsonschema import Draft202012Validator

    schema = json.loads(_repo_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


def run_episodes(
    model_name: str,
    k: int,
    staging_dir: Path,
    *,
    task_name: str = "amc",
    dataset_name: str = "radioml_2016_10a",
    split: str = "test",
    n_episodes: int = N_EPISODES,
    base_seed: int = BASE_SEED,
) -> list[dict[str, float]]:
    """Fit + evaluate ``n_episodes`` few-shot draws, staging one ``result.json`` per seed.

    For each seed in ``range(base_seed, base_seed + n_episodes)``: builds a fresh
    ``FewShotAdapter(k, head, seed=seed)`` with a FRESH board head per episode (so no head
    state leaks across seeds), fits it on the materialised-once train split, wraps it in the
    ``_AdaptedModel`` bridge, and calls :func:`rfbench.core.evaluate.evaluate` writing to
    ``staging_dir/<model>-seed<seed>.json``. Returns the list of per-episode
    ``metrics.values`` dicts (in seed order) for :func:`aggregate_episode_values`.

    Heavy imports (torch-backed FM, evaluate) are LOCAL so importing this module stays cheap
    on the frontend; only ``run_episodes`` needs the cluster environment.
    """
    # Local imports: these pull in the FM backbone / evaluate machinery and (transitively)
    # torch, which only exists on the ARM/GPU node.
    import rfbench.models.foundation  # noqa: F401  (registers 'dummy-fm' + 'iqfm-base')
    import rfbench.models.foundation.lwm_spectro  # noqa: F401  (registers 'lwm-spectro')
    import rfbench.tasks.amc  # noqa: F401  (registers 'amc')
    from rfbench.core.evaluate import evaluate
    from rfbench.core.registry import MODELS, get_task
    from rfbench.models.foundation.base import _AdaptedModel, _default_probe_head
    from rfbench.regimes.few_shot import FewShotAdapter

    task = get_task(task_name)
    fm = MODELS.get(model_name)()
    ds = next(d for d in task.datasets() if d.name == dataset_name)

    logger.info(
        "few-shot k=%d on %s/%s: materialising train split once for %d episodes (seeds %d..%d)",
        k,
        task_name,
        ds.name,
        n_episodes,
        base_seed,
        base_seed + n_episodes - 1,
    )
    train = list(ds.load("train"))

    staging_dir.mkdir(parents=True, exist_ok=True)
    per_episode_values: list[dict[str, float]] = []
    for seed in range(base_seed, base_seed + n_episodes):
        # Fresh head per episode so no fitted state leaks from the previous seed's draw.
        adapter = FewShotAdapter(k, _default_probe_head(), seed=seed)
        state = adapter.fit(fm, train)
        adapted = _AdaptedModel(fm, adapter, state)
        stage_path = staging_dir / f"{model_name}-seed{seed}.json"
        logger.info("episode seed=%d: fitting head + evaluating on %s split...", seed, split)
        res = evaluate(adapted, task, split, adapted.regime, dataset=ds.name, out_path=stage_path)
        values = {k_: float(v) for k_, v in res["metrics"]["values"].items()}
        per_episode_values.append(values)
        logger.info("episode seed=%d -> %s (staged: %s)", seed, values, stage_path)
    return per_episode_values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registered FM name, e.g. 'iqfm-base'")
    parser.add_argument("--k-shot", type=int, required=True, help="examples per class (k >= 1)")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="board row path, e.g. leaderboard/results/amc/<model>-few_shot-k<K>.json",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help=(
            "per-seed staging dir, e.g. $WORK/logs/multiseed/amc/k10"
            " (must include k value to avoid collisions between concurrent few-shot jobs)"
        ),
    )
    parser.add_argument("--task", default="amc", help="task name (default: amc)")
    parser.add_argument(
        "--dataset", default="radioml_2016_10a", help="dataset name (default: radioml_2016_10a)"
    )
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.k_shot < 1:
        logger.error("k_shot must be >= 1, got %d", args.k_shot)
        return 2

    per_episode_values = run_episodes(
        args.model,
        args.k_shot,
        args.staging_dir,
        task_name=args.task,
        dataset_name=args.dataset,
        n_episodes=args.n_episodes,
        base_seed=args.base_seed,
    )
    values, uncertainty = aggregate_episode_values(
        per_episode_values, n_episodes=args.n_episodes, base_seed=args.base_seed
    )

    # Reuse the LAST episode's staged, schema-valid document as the structural template
    # (task/model/regime/split blocks verbatim); rewrite only metrics + fingerprint seed.
    last_seed = args.base_seed + args.n_episodes - 1
    template_path = args.staging_dir / f"{args.model}-seed{last_seed}.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    row = build_board_row(template, values, uncertainty, base_seed=args.base_seed)

    # Re-validate the aggregated row BEFORE it lands (pattern: backfill_uncertainty).
    _validate_or_raise(row)

    from rfbench.core.evaluate import _atomic_write_json

    _atomic_write_json(row, args.out)
    logger.info(
        "few-shot k=%d aggregated over %d episodes -> %s (primary %s=%.4f, spread +/-1 stdev)",
        args.k_shot,
        args.n_episodes,
        args.out,
        row["metrics"]["primary"],
        values[row["metrics"]["primary"]],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
