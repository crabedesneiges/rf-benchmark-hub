# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — AMC board updated with the final-recipe retrain (MCLDNN, ResNet)

- Re-trained from scratch (RadioML 2016.10a, seed 42, 150 epochs) under the fixed recipe
  (val-accuracy best-checkpoint + gradient clipping + paper-exact archs): **MCLDNN 0.6008 → 0.6171**
  and **ResNet 0.5606 → 0.5661**. Both now exceed their prior board scores.
- **KNOWN ISSUE — CLDNN collapses to chance (0.0909) under the new recipe.** Gradient clipping ruled
  out the LSTM-explosion/NaN hypothesis (no divergence logged); the 3-LSTM CLDNN instead stays stuck
  at chance throughout training, even though the same architecture reached 0.5072 under the earlier
  50-epoch val-loss recipe. Under investigation (per-epoch val-accuracy/LR trajectory diagnostic
  pending). The board retains the last reproducible CLDNN figure until the fix lands.

### Added — Downstream-task prioritization mined from the FM bibliography

- `docs/DOWNSTREAM_TASKS.md`: mined the 9 foundation-model papers in `docs/BIBLIOGRAPHY.md` for the
  downstream tasks each one actually evaluates, normalized synonyms into a canonical taxonomy, and
  bucketed by FM coverage — **P1** (>= 2 FM papers): `amc`, `beam_prediction`, `sei`,
  `direction_finding`, `los_nlos`, `positioning`, `har`; **P2** (1 paper): `interference_id`,
  `protocol_tech_id`, `channel_estimation`, `snr_mobility_recognition`, `wideband_detection`;
  **P3** (defined benchmark track, 0 FM papers): `spectrum_sensing`. Each task carries its FM
  evaluators + datasets/metrics, current rfbench status, a recommended canonical dataset/protocol/
  metric, and scope-fit (IQ-signal vs a proposed separate CSI/6G-sensing track).

### Fixed — CLDNN training divergence (gradient clipping)

- The 150-epoch retrain collapsed CLDNN to chance (0.0909): its 3 stacked LSTMs explode at lr=1e-3
  over the longer schedule (the new recipe holds the LR higher for longer), and `argmax(NaN)`
  predicts a constant class. Added a global gradient-norm cap (`DEFAULT_GRAD_CLIP=5.0`, applied in
  `_train_one_epoch`) that stabilises the recurrent baselines without biting the CNN baselines, plus
  a NaN-loss guard that logs an ERROR and stops early, keeping the best checkpoint. MCLDNN/ResNet are
  unaffected (clip does not trigger for them). Baselines to be re-trained under the final recipe.

### Changed — AMC baseline paper-conformance + training-recipe fix (M3)

- **Regression root-caused** (4-way audit + adversarial verification): the `training.py` rewrite selected
  the best checkpoint on **validation loss**, whose minimum precedes the accuracy peak on RadioML, so it
  restored a suboptimal checkpoint and dragged every baseline down (CLDNN −8 pt). `train_baseline()` now
  selects/restores the best checkpoint on **validation accuracy** (same argmax/label convention as
  `core.evaluate`), keeps `ReduceLROnPlateau` on val loss, and early-stops on accuracy. Recipe loosened:
  patience 20→40, `min_delta` 1e-4→0, `lr_patience` 5→10, `min_lr` 1e-6→1e-7.
- **MCLDNN fusion made paper-exact**: element-wise add → channel-axis **concatenate** (`conv_fuse`
  in-channels 50→100, VALID padding → post-fusion length 124), matching the official `wzjialang/MCLDNN`.
- **ResNet depth adapted to the len-128 window**: `num_stacks` 6→3 (6 MaxPools over-pool 128→2; L=3 keeps
  the paper's ~16 final time steps, `flat_dim` 64→512), `alpha_dropout` 0.5→0.3.
- CLDNN left unchanged to isolate the recipe fix. Tests updated; suite green. Baselines to be re-trained
  from scratch (seed 42, 150 epochs) on the cluster to refresh the board.

### Changed — Leaderboard site redesign (generic, per-metric)

- **WP-50 rewrite.** `leaderboard/site/generate.py` is now fully data-driven: it renders **every** task
  (not just AMC), one `<task>.html` per task with results, and a **column or plot for every metric** —
  one table column per scalar metric (primary pinned first) and one inline `<svg>` line plot per curve
  metric (e.g. `accuracy_vs_snr`). Self-contained dark/light CSS, family chips, and
  `verified`/`self_reported` badges.
- **Protocol invariants enforced in markup.** One `<table data-regime data-track>` per distinct
  `(regime, track)` pair — two regimes never share a table, and same-regime different-track results split
  into separate tables. Rows sorted by the primary metric descending.
- `tests/test_site.py` rewritten (16 tests, mutation-checked non-trivial) against the new generic output;
  full suite green (342 passed, 29 skipped), `ruff`/`black`/`mypy` clean.

### Added — Real dataset loaders (M1, no generation)

Per the "use the datasets from the reference papers, do not generate" decision:

- **AMC.** RadioML 2016.10a (pickle, `opendata.deepsig.io`) + 2018.01a (HDF5) real loaders; **Sig53 is a
  reported blocker** (generation-only, no static release — not synthesised).
- **SEI.** Real loaders for WiSig (ManyTx), ORACLE (SigMF), LoRa RFFI (HDF5), each targeting the confirmed
  official source; credential-gated sources raise with manual-download instructions (no scraping).
- **Detection.** Adopted **RadDet** (ICASSP 2025, real published spectrogram + YOLO box annotations) as the
  wideband-detection dataset; **WBSig53 is a blocker** (generation-only). Protocol + task layer updated.
- Heavy deps (numpy/h5py/requests) stay lazy behind `rfbench[data]`; parsers tested on synthetic fixtures
  (real stdlib-pickle fixture for RadioML 2016; `importorskip` for HDF5). CLI wired to the real API.

### Added — Submission, publish, verify, FM wrappers (M5/M6)

- **WP-51/52 — Submission & publish CI.** `.github/workflows/validate-submission.yml`
  (`rfbench submit --check` + no-raw-data guard on PRs touching `leaderboard/results/**`) and
  `build-leaderboard.yml` (build the site → deploy to GitHub Pages on push to main); issue-form
  templates for submissions and task proposals.
- **WP-53 — Verification pipeline.** `rfbench/verify.py` + `rfbench verify`: checks manifest
  completeness and re-run metrics within tolerance, flips `verification.status → verified` with
  `verified_by/at/hardware`; `submit --check` strengthened to validate the submission manifest.
- **WP-60/61 — Foundation-model wrappers.** `rfbench/models/foundation/` generic wrapper (Model +
  `embed()`), a dependency-free example FM running in all four regimes via `evaluate()`, a copy-me
  template, and `docs/ADDING_A_MODEL.md`.
- **WP-50 fix — Leaderboard by track.** The board now separates by (task, track) as well as regime,
  so SEI (closed_set/cross_receiver/cross_day) and detection (detection/recognition) tracks are
  reported in separate tables.
- **Cluster recon.** `slurm/probe_torchsig*.sh`: `.[detection]` installs on aarch64 (torch 2.12+cu130,
  torchsig 2.1.1); torchsig 2.x replaced named WBSig53 with a config-driven wideband API — informs the
  real detection loader (still a lazy stub, pending a torchsig-1.x-vs-2.x decision).

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
