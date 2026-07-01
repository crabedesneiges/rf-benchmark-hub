# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Sprint 2 — Task adapters (M2)

- **WP-20 — AMC task.** `rfbench/tasks/amc/`: `AmcTask` (registered `amc`) + metrics
  `accuracy_overall` (primary, full SNR range), `accuracy_vs_snr` curve, `macro_f1` (pure-stdlib),
  dataset adapter, `configs/task/amc.yaml`.
- **WP-21 — SEI task.** `rfbench/tasks/sei/`: `SeiTask` (registered `sei`), tracks
  closed_set/cross_receiver/cross_day/open_set; `rank1_accuracy` (primary) + `auroc`/`eer` as
  separate metrics (pure-stdlib), `configs/task/sei.yaml`.
- **WP-22 — Wideband detection task.** `rfbench/tasks/wideband_detection/`: task + `mAP`/`mAR`/`IoU`
  (pure-stdlib IoU+AP path, lazy torchmetrics for production), detection vs recognition tracks kept
  distinct, `configs/task/wideband_detection.yaml`.
- **Cluster.** `slurm/setup_and_test_arm.sh`: CPU-only job that builds the `rfbench` venv on an ARM
  compute node and runs the full suite on aarch64 (proves the harness on the target arch).

### Added — Sprint 1 wave 2 — Data layer, leaderboard, CLI (M1/M2/M5, scaffolds)

Datasets are not redistributed and real `prepare` runs on the cluster (ARM venv, `rfbench[data]` /
`rfbench[detection]`); heavy deps are imported lazily and unit tests exercise the split path on
pure-stdlib synthetic fixtures.

- **WP-11 — AMC data (template).** `rfbench/data/prepare/{_common,amc}.py` + `download/amc_*.py`:
  RadioML → 80/10/10 stratified by (modulation × SNR); Sig53 → adopted official TorchSig split.
  `_common.py` (cache dir via `$RFBENCH_CACHE`, split+manifest helpers) is reused by SEI/detection.
- **WP-12 — SEI data.** `rfbench/data/prepare/sei.py` + `download/sei_*.py`: `closed_set`,
  `cross_receiver`, `cross_day` generated as separate grouped conditions with disjoint rx/day groups.
- **WP-13 — Detection data.** `rfbench/data/prepare/detection.py` + `download/detection_wbsig53.py`:
  split per policy + a T-F box annotations sidecar; detection vs recognition tracks kept distinct.
- **WP-50 — Leaderboard site.** `leaderboard/site/generate.py` (`build_site`): static HTML from
  `results/**.json`, sorted by primary metric, one table per regime (never mixed), verified/self_reported
  badges; seeded with sample result JSONs under `leaderboard/results/`.
- **WP-42 — CLI wiring.** `rfbench data prepare` / `eval` / `submit --check` / `leaderboard build` wired
  to the real implementations; heavy imports stay lazy so `import rfbench` and `rfbench --help` remain
  dependency-free.

### Added — Sprint 1 wave 1 — Splits & eval harness (M1/M4, partial)

- **Split policy (normative).** `docs/EVALUATION_PROTOCOL.md`: adopt an official/literature split when
  one exists (provenance recorded), otherwise a deterministic **80/10/10** stratified split, seed 42.
  Ratios + seed are part of `canonical_split_id`; this supersedes the earlier 60/20/20 AMC placeholder.
- **WP-10 — Deterministic splits.** `rfbench/core/splits.py`: `make_split` (pure-stdlib, seeded,
  stratified, default 80/10/10) → `SplitManifest`; `adopt_official_split` (pass-through), reproducible
  `write_split_index` + `split_checksum`. No generated indices committed (no data yet).
- **WP-40 — `evaluate()` + `result.json`.** `rfbench/core/evaluate.py`: the single canonical emitter;
  assembles a schema-valid `result.json` (regime declared verbatim, `verification.status=self_reported`),
  validated against `schemas/result.schema.json` via lazily-imported `jsonschema`.
- **WP-41 — Regime adapters.** `rfbench/regimes/` (`from_scratch`, `full_finetune`, `linear_probe`
  with a pure-Python nearest-centroid head, `few_shot(k)`) + `configs/config.yaml` and
  `configs/regime/*.yaml`. Dependency-free; real numerical heads deferred to M3/M6 behind extras.

### Added — Sprint 0 — Bootstrap & contracts (M0)

Scaffolds the repo, freezes the core contracts and JSON schemas, and lands the normative docs and
CI skeleton.

- **WP-00 — Repo & packaging.** `pyproject.toml` defining the `rfbench` package (at repo root,
  not `src/`) and the `rfbench` CLI entrypoint; pre-commit with `ruff` + `black` (line-length 100);
  Apache-2.0 `LICENSE`.
- **WP-01 — JSON schemas.** Frozen `schemas/result.schema.json` (one evaluation run = one
  leaderboard row; regime always declared; `verification.status` ∈ `{self_reported, verified}`)
  and `schemas/submission.schema.json` (reproducibility manifest), with valid/invalid examples
  under `schemas/examples/`.
- **WP-02 — Core contracts (ABCs).** `rfbench/core/{task,dataset,metric,model,registry,splits,`
  `evaluate,manifest}.py`: typed interfaces with docstrings and minimal bodies; ABCs are not
  instantiable.
- **WP-03 — Normative docs.** `docs/EVALUATION_PROTOCOL.md`, `docs/SUBMISSION.md`,
  `docs/ARCHITECTURE.md`, plus `README.md` and `CONTRIBUTING.md`. Each task has a defined
  metric + split + regimes; the two-tier submission workflow is described end to end.
- **WP-04 — CI skeleton.** `.github/workflows/ci.yml` running lint, unit tests, and schema checks.

### Notes

- **No raw data in git** (D3): only split indices + checksums are versioned under
  `leaderboard/splits/`; datasets are fetched via `rfbench data prepare` and honour `$RFBENCH_CACHE`.
- **Frozen contracts:** the core ABCs and JSON schemas are locked at Sprint 0; changing them
  requires an explicit review and a version bump.
- Scope is terrestrial RF only (D1); satellite RF is a separate repository.

[Unreleased]: https://github.com/rf-benchmark-hub/rf-benchmark-hub/commits/main
