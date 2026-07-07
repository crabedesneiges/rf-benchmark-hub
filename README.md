# RF-Benchmark-Hub

**Reproducible benchmarks and a leaderboard for terrestrial RF machine learning.**
One frozen, canonical protocol per task; specialised baselines and fine-tuned foundation
models compared under the *same* rules, on the *same* splits, with the *same* metrics.

[![CI](https://github.com/crabedesneiges/rf-benchmark-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/crabedesneiges/rf-benchmark-hub/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/leaderboard-GitHub_Pages-brightgreen.svg)](https://crabedesneiges.github.io/rf-benchmark-hub/)

Scope is **terrestrial RF only** — automatic modulation classification (AMC),
specific emitter identification / RF fingerprinting (SEI), wideband detection, and
spectrum sensing. Satellite RF lives in a **separate repository** (design decision D1).

---

## Why this exists

The value of this hub is **not** to aggregate incomparable numbers scraped from papers.
It is to **freeze a canonical protocol per task** and to guarantee **reproducibility**
(see [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §2). Three invariants:

1. **A single contract.** Every task exposes a `Dataset`, a `Metric`, a canonical `split`,
   and an eval protocol; every model exposes a `Model` interface plus a **declared regime**.
2. **The official score is reproducible.** A verified submission ships code + weights + Docker
   + the exact command; a maintainer re-runs it and signs off.
3. **Comparability over exhaustiveness.** Full SNR range imposed, splits published, metrics
   defined formally. A smaller board of comparable numbers beats a large board of noise.

---

## Quickstart

```bash
pip install -e ".[dev]"                                          # editable install + dev tools
rfbench --help                                                   # data prepare / eval / submit / leaderboard

# Prepare the canonical, deterministic split (downloads from the official source; no data in git)
rfbench data prepare amc --dataset radioml_2016_10a --seed 42

# Evaluate a model under an explicitly declared regime -> emits result.json (self_reported)
rfbench eval amc --model mcldnn --regime full_finetune

ruff check . && pytest -q                                        # lint + tests (must stay green)
```

`rfbench eval` writes a `result.json` validated against
[`schemas/result.schema.json`](schemas/result.schema.json) before it can enter the board.

---

## Tasks and primary metrics

Mirrors [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) (the **normative** source).
Any change to a metric or split there is a breaking change and bumps the task `version`.

| Task | Datasets | Primary metric | Tracks / splits |
|------|----------|----------------|-----------------|
| **AMC** | RadioML 2016.10a, RadioML 2018.01a, Sig53 | `accuracy_overall` over the **full SNR range** | stratified `(modulation × SNR)`, seed 42 (`amc-strat-snr-seed42-v1`); also `accuracy_vs_snr`, `macro_f1` |
| **SEI** | WiSig, ORACLE, LoRa RFFI | closed-set `rank1_accuracy`; open-set `auroc`, `eer` | `closed_set`, `cross_receiver`, `cross_day` — **reported as separate rows** |
| **Wideband detection** | WBSig53 (impaired) | `mAP` | plus `mAR`, `IoU`; `detection` vs `recognition` tracks |
| **Spectrum sensing** (Wave B) | DeepSense (OTA 802.11 a/g + LTE-M) | `pd@pfa=0.1` | plus inference `latency_ms` (ROC curve reported) |

Full-protocol conditions (e.g. the AMC SNR range) are recorded in
`result.json.eval.conditions`. AMC forbids SNR cherry-picking.

---

## Regimes

Adaptation regime is one of four (D5), applied as an adapter around any `Model`:

| Regime | Meaning |
|--------|---------|
| `from_scratch` | trained from random init |
| `full_finetune` | all weights updated |
| `linear_probe` | frozen features + a linear head on `embed()` |
| `few_shot` | `k` labelled examples per class (`k_shot` declared) |

The regime is **always declared** in every `result.json` and **never inferred**. The
leaderboard **never mixes two regimes in one column** — an FM in `linear_probe` is never
ranked against a baseline in `full_finetune`.

---

## Leaderboard

`leaderboard/results/**.json` are the **source of truth** for every score; the static site
([GitHub Pages](https://crabedesneiges.github.io/rf-benchmark-hub/)) is generated from them.
The board shows **two tracks**:

- **self_reported** — anyone ran the eval locally and emitted a valid `result.json` (Tier 1).
- **verified** — a maintainer re-ran the submission within tolerance and signed it (Tier 2).

Confidence comes from the `verified` track, not from volume. See
[`docs/SUBMISSION.md`](docs/SUBMISSION.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Data policy

**No raw data ever in git** (D3 / [`CLAUDE.md`](CLAUDE.md)). Only split *indices* and
*checksums* are versioned, under `leaderboard/splits/<dataset>/*.idx.json`. Datasets are
downloaded from their official source via `rfbench data prepare`, which reconstructs the
canonical split and verifies checksums. The cache location honours the `$RFBENCH_CACHE`
environment variable — no hard-coded paths. Dataset licences remain those of their sources
(never redistributed); the **code** is Apache-2.0 (D7).

`.h5 / .npy / .bin / .sigmf-data` and similar are git-ignored **and** blocked in CI.

---

## Repository layout

```
rf-benchmark-hub/
  docs/            IMPLEMENTATION_PLAN · ARCHITECTURE · EVALUATION_PROTOCOL · SUBMISSION
  schemas/         result.schema.json · submission.schema.json · examples/
  rfbench/
    core/          Task · Dataset · Metric · Model+Regime · evaluate() (frozen contracts)
    tasks/         amc · sei · wideband_detection · spectrum_sensing
    models/        baselines/ (seed the board) · foundation/ (FM wrappers, embed())
    data/          download/ · prepare/ (no versioned data)
  configs/         Hydra: config.yaml · task/ · model/ · regime/
  leaderboard/     results/**.json (source of truth) · splits/ (indices+checksums) · site/
  tests/
```

---

## Documentation

- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) — metrics, splits, regimes per task (**normative**)
- [`docs/SUBMISSION.md`](docs/SUBMISSION.md) — submission workflow + two-tier verification
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to submit a score, add a task/model, dev conventions
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — engineering details of the harness
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — master spec, work packages, milestones

## License

Code is licensed under **Apache-2.0** (see `LICENSE`). Datasets retain their upstream licences
and are never redistributed by this repository.
