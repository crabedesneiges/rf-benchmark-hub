# RF-Benchmark-Hub

**Reproducible benchmarks and a leaderboard for terrestrial RF machine learning.**
One frozen, canonical protocol per task; specialised baselines and fine-tuned foundation
models compared under the *same* rules, on the *same* splits, with the *same* metrics.

[![CI](https://github.com/crabedesneiges/rf-benchmark-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/crabedesneiges/rf-benchmark-hub/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/leaderboard-GitHub_Pages-brightgreen.svg)](https://crabedesneiges.github.io/rf-benchmark-hub/)

Scope is **terrestrial RF only**. The board currently carries live results for automatic
modulation classification (AMC), specific emitter identification / RF fingerprinting (SEI),
SNR estimation, interference identification, and WiFi protocol/technology identification.
Wideband detection and spectrum sensing are declared tracks with a frozen protocol but no
committed baseline yet (work-in-progress). Satellite RF lives in a **separate repository**
(design decision D1).

---

## Why this exists

The value of this hub is **not** to aggregate incomparable numbers scraped from papers.
It is to **freeze a canonical protocol per task** and to guarantee **reproducibility**
(see [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §2). Three invariants:

1. **A single contract.** Every task exposes a `Dataset`, a `Metric`, a canonical `split`,
   and an eval protocol; every model exposes a `Model` interface plus a **declared regime**.
2. **The official score is reproducible.** A verified submission ships code + weights + a pinned
   environment spec + the exact command; a maintainer re-runs the recipe within the per-task
   tolerance (`rfbench verify`) and signs it off (`verification.status = "verified"`).
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

Mirrors [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) (the **normative** source)
and [`leaderboard/tasks.json`](leaderboard/tasks.json) (the task inventory the site renders).
Any change to a metric or split there is a breaking change and bumps the task `version`.

**Live tasks** — a committed canonical split and at least one board row:

| Task | Board datasets (one column each) | Primary metric | Tracks / splits |
|------|----------------------------------|----------------|-----------------|
| **AMC** | RadioML 2016.10a (11 cl.), RadioML 2018.01a (24 cl.) | `accuracy_overall` over the **full SNR range** | stratified `(modulation × SNR)`, seed 42 (`amc-radioml2016-strat-snr-8010-seed42-v1`, `amc-radioml2018-strat-snr-8010-seed42-v1`); also `accuracy_vs_snr`, `macro_f1` |
| **SEI** | WiSig (ManyTx), ORACLE (16-tx) | closed-set `rank1_accuracy` + `balanced_accuracy`; open-set `auroc`, `eer` | WiSig: `closed_set` / `cross_receiver` / `cross_day` / `open_set` — separate rows; ORACLE: `closed_set` only |
| **SNR estimation** | RadioML 2016.10a | `rmse_db` (**lower is better**, board ranks ascending) | single `all_snr` track over the full SNR range (`snr-radioml2016-strat-snr-8010-seed42-v1`, byte-identical to the AMC 2016 split); secondary `mae_db` |
| **Interference id.** | interf-gnss6 (GNSS jamming, 6 cl.) | `accuracy_overall` | `interf-gnss6-8010-seed42-v1`; secondary `macro_f1` (no SNR grid) |
| **Protocol / tech id.** | tprime-wifi4 (802.11 b/g/n/ax) | `accuracy_overall` | within-distribution (`proto-tprime-wifi4-8010-seed42-v1`) and leave-one-location-out cross-room (`proto-tprime-wifi4-crossroom-*`) — separate columns; secondary `macro_f1` |

**Declared, work-in-progress** — a frozen protocol in `EVALUATION_PROTOCOL.md`, no committed
split/baseline yet: **wideband detection** (RadDet, `mAP` / `mAR` / `IoU`) and **spectrum
sensing** (DeepSense, `pd@pfa=0.1`). CSI/channel-domain tasks (beam prediction, LoS/NLoS,
positioning, HAR, channel estimation) are **planned** and out of the current terrestrial-IQ
scope — a separate CSI track if the hub expands. See `leaderboard/tasks.json` for the full
inventory and per-task status.

Full-protocol conditions (e.g. the AMC SNR range) are recorded in
`result.json.eval.conditions`. AMC and SNR estimation forbid SNR cherry-picking; SEI tracks
are never blended into one column.

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
Every row carries a declared **evidence tier** (`result.json.verification.status`) so the reader
knows exactly how much a number is worth. Four tiers, strongest first:

- **`verified`** — a maintainer re-ran the committed recipe within the per-task tolerance
  (`docs/EVALUATION_PROTOCOL.md`) and signed it. Confidence comes from this tier, not volume.
- **`self_reported`** — someone ran the eval and emitted a valid `result.json` on a committed
  split, but it has not been re-run by a maintainer.
- **`from_paper`** — a literature number transcribed onto the board because it was measured on
  **the same dataset + protocol + our exact committed split** (Tier-3 doctrine). Not re-run here.
- **`from_paper_uncertain`** — a literature number where only the **dataset family** matches and
  the exact split/protocol is not confirmed. The weakest tier; read as an indicative reference.

Rows of different tiers are never silently merged and the board never mixes two **regimes**
(`from_scratch` / `full_finetune` / `linear_probe` / `few_shot`) in one column. `from_paper*`
rows are transcriptions, not reproductions — treat them as context, not as verified results.
See [`docs/SUBMISSION.md`](docs/SUBMISSION.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Data policy

**No raw data ever in git** (D3 / [`CLAUDE.md`](CLAUDE.md)). Only split *indices* and
*checksums* are versioned, under `leaderboard/splits/<dataset>/<canonical_split_id>.idx.json`
(the sample indices) alongside a `<canonical_split_id>.manifest.json` (source URL, seed,
`split_checksum`). Each `result.json` records the `canonical_split_id` + `checksum` it was scored
on, so a row is bound to an exact, versioned partition. Datasets are downloaded from their
official source via `rfbench data prepare`, which reconstructs the canonical split and verifies
checksums. The cache location honours the `$RFBENCH_CACHE` environment variable — no hard-coded
paths. Dataset licences remain those of their sources (never redistributed); the **code** is
Apache-2.0 (D7).

`.h5 / .npy / .bin / .sigmf-data` and similar are git-ignored **and** blocked in CI.

---

## Repository layout

```
rf-benchmark-hub/
  docs/            IMPLEMENTATION_PLAN · ARCHITECTURE · EVALUATION_PROTOCOL · SUBMISSION
  schemas/         result.schema.json · submission.schema.json · examples/
  rfbench/
    core/          Task · Dataset · Metric · Model+Regime · evaluate() (frozen contracts)
    tasks/         amc · sei · snr_estimation · interference_id · protocol_tech_id · wideband_detection
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

## Community & governance

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute code, tasks, or scores
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1; expected behaviour and enforcement
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability **privately** (do not open a public issue)
- [`CITATION.cff`](CITATION.cff) — how to cite this project (GitHub renders a "Cite this repository" button)

Bug reports, feature requests, and score submissions go through the structured
[issue templates](.github/ISSUE_TEMPLATE/).

## License

Code is licensed under **Apache-2.0** (see `LICENSE`). Datasets retain their upstream licences
and are never redistributed by this repository.
