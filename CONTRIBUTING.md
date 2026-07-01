# Contributing to RF-Benchmark-Hub

This guide covers **how to add a score** (the two verification tiers) and, at later
milestones, **how to add a task or a model**. The evaluation rules themselves are normative and
live in [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md); the submission workflow is
summarised in [`docs/SUBMISSION.md`](docs/SUBMISSION.md). This file operationalises both.

Every score is a single `result.json` that validates against
[`schemas/result.schema.json`](schemas/result.schema.json), and every official submission ships a
reproducibility manifest that validates against
[`schemas/submission.schema.json`](schemas/submission.schema.json). **These schemas are frozen**
(Sprint 0) — see the frozen-contracts rule below.

---

## Adding a score

### Tier 1 — self-serve (for your paper)

Run the evaluation locally:

```bash
rfbench eval <task> --model <name> --regime <regime>   # e.g. amc / mcldnn / full_finetune
```

This emits a `result.json` validated against `schemas/result.schema.json`. The number is
**yours to cite**; on the board it is marked `self_reported`. **No PR is required.**

### Tier 2 — official (verified)

To earn the `verified` badge, a maintainer must reproduce your run. Steps:

1. **Open a PR** adding:
   - `leaderboard/results/<task>/<name>.json` — the result,
   - a reproducibility manifest validating against `schemas/submission.schema.json`
     (e.g. `<name>.submission.json`), referenced from the result via `submission_ref`
     (and by the manifest's `result_path`).

2. **Check it locally first** — this mirrors the `validate-submission.yml` CI job:

   ```bash
   rfbench submit --check leaderboard/results/<task>/<name>.json \
                  --manifest leaderboard/results/<task>/<name>.submission.json
   ```

   `submit --check` verifies: both schemas pass; the `regime` is declared; `metrics.primary`
   is a key of `metrics.values`; full-protocol conditions hold (AMC:
   `eval.conditions.full_snr_range == true`; SEI: one row per track — `closed_set`,
   `cross_receiver`, `cross_day`); the split `checksum` matches the versioned indices in
   `leaderboard/splits/<dataset>/`; and **no raw data** is staged.

3. **CI validates** the PR: schema conformance, reproducibility completeness (see checklist),
   and a split-checksum lint.

4. **A maintainer re-runs** on a multi-GPU station, in one of two modes:
   - `eval_only` (default) — weights + Docker provided, evaluation only;
   - `full_retrain` — for seed baselines, re-training from the declared seed.

5. **If reproduced within `tolerance`**, `rfbench verify` flips `verification.status` from
   `self_reported` to `verified` and stamps `verified_by`, `verified_date`,
   `verified_hardware`, and `method`. `build-leaderboard` then republishes the site.

---

## What makes a submission reproducible

A Tier-2 manifest is **1:1 with the required fields of `submission.schema.json`**. Missing any of
these means the result can only stay `self_reported`:

| Field | What it is |
|-------|-----------|
| `code_commit` | Exact git commit (`git@<sha>` or bare hex SHA). A dirty tree cannot be verified. |
| `command` | The exact shell command that reproduces the result. |
| `artifacts.weights_url` **and/or** `artifacts.docker_image` | Fetchable weights (enables `eval_only`) and/or a pinned, digest-addressed image that froze the environment. **At least one is required.** |
| `hardware` | Hardware the run was performed on, e.g. `1x NVIDIA A100 80GB`. |
| `expected_metrics` | Metric values the re-run must reproduce; the task's **primary** metric must be present. |
| `tolerance` | Match criterion (`absolute` and/or `relative`, optional `per_metric`) on the primary metric. |

The manifest also carries `schema_version`, `result_path`, `task`, and `regime` (which must match
the referenced result). Recommended extras: `weights_checksum`, `environment` (python/cuda/torch,
`pip_freeze_sha256`), `data_provenance` (`canonical_split_id`, `split_checksum`), and `contact`.

Copy [`schemas/examples/submission.valid.json`](schemas/examples/submission.valid.json) as a
starting template.

---

## Hard rules

> - **Never commit raw data.** `.h5 / .npy / .bin / .sigmf-data` (and friends) are git-ignored
>   **and** rejected by CI. Only split *indices* + *checksums* are versioned, under
>   `leaderboard/splits/`.
> - **Regime is always declared**, never inferred — one of
>   `from_scratch | full_finetune | linear_probe | few_shot`.
> - **AMC uses the full SNR range** (`eval.conditions.full_snr_range == true`); no cherry-picking.
> - **One `result.json` per `(task, model, regime, dataset, split, track)`.** SEI tracks
>   (`closed_set` / `cross_receiver` / `cross_day`) and detection tracks (`detection` /
>   `recognition`) are separate rows.
> - **The board never mixes two regimes in one column.**

---

## Adding a task or a model (later milestones)

- **Adding a model** follows a wrapper template under `rfbench/models/` and must **not** touch
  the frozen core contracts. A foundation-model wrapper exposes `embed()` so the
  `linear_probe` / `few_shot` regimes work unchanged. A dedicated guide ships with WP-61.
- **Adding a task** follows the WP-T* template: a `Task` + `Dataset` + `Metric` under
  `rfbench/tasks/<task>/` plus a Hydra config, and a normative entry in
  `docs/EVALUATION_PROTOCOL.md`. Changing an existing task's metric or split is a **breaking
  change** and bumps that task's `version`.

---

## Development conventions

- **Python ≥ 3.10, fully typed.** `ruff` + `black` with **line-length 100**; `mypy` strict.
  Docstrings on every public interface.
- **`ruff check . && pytest -q` must stay green.**
- **Small PRs — one per work package** (WP-xx from
  [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)).
- **Tests for every new contract or metric.**
- **Dependencies via `uv add`**, never `pip install` directly.

### Frozen contracts

The **core contracts** (`rfbench/core/`: `Task`, `Dataset`, `Metric`, `Model` + `Regime`,
`evaluate()`) and the **JSON schemas** (`schemas/*.schema.json`) are frozen at Sprint 0. Changing a
signature or a schema requires an explicit review **and** a version bump — it is never a drive-by
edit inside a feature PR. This is what keeps every historical `result.json` on the board readable.

### Schema examples

Under [`schemas/examples/`](schemas/examples/):

- `result.valid.json` — a valid result (copy this as your starting template),
- `result.invalid.json` — an intentionally invalid result (CI negative test),
- `submission.valid.json` — a valid reproducibility manifest.
