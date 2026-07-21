# Downstream-Task Prioritization (FM-mined)

What downstream evaluation tasks do wireless/RF **foundation models** actually run, and which of them
should RF-Benchmark-Hub cover? This doc mines the FM bibliography (`docs/BIBLIOGRAPHY.md`) for the
tasks each FM paper *evaluates*, normalizes synonymous task names into a canonical vocabulary, and
buckets each canonical task by FM coverage.

## Method

- Source: the 9 FM papers tracked in `docs/BIBLIOGRAPHY.md` (Part A.5 / Part C.5).
- For each paper, we counted only tasks the paper **itself evaluates** downstream (frozen-encoder
  linear probe, LoRA, few-shot, or fine-tune) — not pretraining pretexts, not related-work mentions,
  not later model versions.
- Task names were normalized to canonical ids (synonyms merged; see taxonomy). Multiple datasets /
  regimes for the same canonical task inside one paper count **once** for that paper.
- Priority buckets:
  - **P1** = evaluated by **>= 2** distinct FM papers here.
  - **P2** = evaluated by **exactly 1** FM paper here.
  - **P3** = **0** FM papers here, but the task is a defined benchmark track in
    `docs/EVALUATION_PROTOCOL.md` (a coverage gap worth flagging).

FM papers mined (9):

| Model          | arXiv         | Domain / input                          |
|----------------|---------------|-----------------------------------------|
| WirelessJEPA   | 2601.20190    | IQ, OTA testbed; JEPA                    |
| IQFM           | 2506.06718    | IQ, OTA MIMO testbed; contrastive SSL   |
| RIS-MAE        | 2508.00274    | IQ, ViT-MAE                             |
| LWM-Spectro    | 2601.08780    | spectrogram, DeepMIMO synth; MSM + MoE  |
| LWM-base       | 2411.08872    | CSI/channel; MCM                        |
| WavesFM        | 2504.14100    | spectrogram + CSI; masked-waveform      |
| LatentWave     | 2606.06373    | spectrogram + CSI; JEPA                 |
| 6G-MSM         | 2411.09996    | spectrogram + CSI; MSM-ViT              |
| TorchSig-XCiT  | 2207.09918    | IQ, Sig53; supervised XCiT              |

## Normalized taxonomy

| Canonical id             | Definition                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| `amc`                    | Automatic modulation / signal(-type) classification: predict modulation scheme or signal-type label from IQ or spectrogram. Merges "modulation classification", "RF/signal classification", "signal recognition", and broad signal-type taxonomies (WiFi/FM/Cellular/BT). In-distribution vs OOD/cross-dataset are the same canonical task. |
| `sei`                    | Specific emitter identification / RF fingerprinting: identify the transmitting device/hardware from imperfection-driven features. Distinct from `amc` (identity, not modulation). |
| `direction_finding`      | Angle-of-arrival (AoA) / direction-of-arrival estimation from spatial phase across an antenna array (regression or discrete angle-bin classification). Distinct from `positioning` (bearing, not coordinates). |
| `beam_prediction`        | Predict the best beam / beam index from a predefined codebook (e.g. sub-6 GHz CSI -> strongest mmWave beam, or beam classification on a testbed). |
| `los_nlos`               | Line-of-sight vs non-line-of-sight classification from channel/CSI embeddings. |
| `positioning`            | Localization / geolocation: estimate transmitter/user 2D-3D coordinates (regression, mean distance error). Merges positioning = localization = geolocation. Distinct from `direction_finding`. |
| `channel_estimation`     | Estimate the channel response between user and base station (MIMO-OFDM), regression to MSE. Absorbs CSI feedback / compression / interpolation. |
| `har`                    | Human activity recognition / RF sensing: classify human motion/activity from CSI or RF measurements. |
| `interference_id`        | Interference / jamming detection and classification (e.g. multi-class GNSS jamming-condition recognition). Distinct from `spectrum_sensing` (occupancy) and `wideband_detection` (time-frequency localization). |
| `protocol_tech_id`       | Wireless technology / standard / protocol recognition (e.g. IEEE 802.11 ax/b/n/g variant). Distinct from `amc`: identifies the protocol/standard, not the modulation. |
| `snr_mobility_recognition` | Joint SNR-bin x mobility/Doppler-regime classification into a single combined (SNR, mobility) label space (classification, F1/accuracy — not a regression MAE). New id from LWM-Spectro Task 2; distinct from a pure SNR-estimation regression. |
| `snr_estimation`         | Per-window signal-to-noise-ratio **regression** (predict `snr_db` in dB from IQ), scored by `rmse_db` (primary, lower is better) / `mae_db`. Distinct from `snr_mobility_recognition` (a joint SNR-bin x mobility *classification*, not a scalar regression). Added 2026-07 as a board track from the RadioML 2016.10a `snr_db` field — not part of the FM mining. |
| `wideband_detection`     | Signal detection / localization in time-frequency (spectrogram): boxes or per-pixel semantic segmentation locating/identifying signals (e.g. noise/NR/LTE segmentation). Distinct from `spectrum_sensing`. |
| `spectrum_sensing`       | Per-subband spectrum occupancy / presence detection under a target false-alarm rate (pd@pfa). Distinct from `wideband_detection` (occupancy vs time-frequency localization). |
| `source_separation`      | Blind multi-source RF separation: reconstruct each component waveform from a single-channel mixture of 2+ unknown standard-compliant sources (permutation-invariant, per-source ground truth). Distinct from `spectrum_sensing`/`wideband_detection` (presence/localization, not reconstruction) and from interference *cancellation* (known desired signal). Added 2026-07 from RFSS (arXiv:2604.00398) — not part of the FM mining. |

## Coverage matrix (canonical task x 9 FM papers)

`X` = the paper evaluates this task downstream. Priority = FM-paper count bucket.

| Canonical task            | WirelessJEPA | IQFM | RIS-MAE | LWM-Spectro | LWM-base | WavesFM | LatentWave | 6G-MSM | TorchSig-XCiT | Priority   |
|---------------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-----------|
| `amc`                     | X   | X   | X   | X   |     | X   | X   |     | X   | P1 (7)    |
| `beam_prediction`         |     | X   |     |     | X   |     | X   |     |     | P1 (3)    |
| `sei`                     | X   | X   |     |     |     |     |     |     |     | P1 (2)    |
| `direction_finding`       | X   | X   |     |     |     |     |     |     |     | P1 (2)    |
| `los_nlos`                |     |     |     |     | X   |     | X   |     |     | P1 (2)    |
| `positioning`             |     |     |     |     |     | X   | X   |     |     | P1 (2)    |
| `har`                     |     |     |     |     |     | X   |     | X   |     | P1 (2)    |
| `interference_id`         | X   |     |     |     |     |     |     |     |     | P2 (1)    |
| `protocol_tech_id`        | X   |     |     |     |     |     |     |     |     | P2 (1)    |
| `channel_estimation`      |     |     |     |     |     | X   |     |     |     | P2 (1)    |
| `snr_mobility_recognition`|     |     |     | X   |     |     |     |     |     | P2 (1)    |
| `wideband_detection`      |     |     |     |     |     |     |     | X   |     | P2 (1)    |
| `spectrum_sensing`        |     |     |     |     |     |     |     |     |     | P3 (0)    |
| `source_separation`       |     |     |     |     |     |     |     |     |     | P3 (0)*   |
| `snr_estimation`          |     |     |     |     |     |     |     |     |     | P3 (0)†   |

\* `source_separation` added from RFSS (arXiv:2604.00398, not an FM paper); no FM evaluates it.
† `snr_estimation` is a board regression track on RadioML 2016.10a's `snr_db` field, not an
FM-mined task; no FM paper evaluates scalar SNR regression. It is nonetheless **live on the
board** (committed split + verified baseline) — see the P3 section below.

Per-paper canonical-task counts: WirelessJEPA 5, IQFM 4, LatentWave 4, WavesFM 4, LWM-base 2,
LWM-Spectro 2, 6G-MSM 2, RIS-MAE 1, TorchSig-XCiT 1.

---

## P1 — evaluated by >= 2 FM papers

### `amc` — automatic modulation / signal classification (7 papers)

- **FM evaluators & data/metrics**:
  - WirelessJEPA — OTA testbed (in-dist, 1/100-shot linear probe) + RML2016.10a OOD (11 mods,
    -20..+18 dB), 500-shot linear probe 74.8%.
  - IQFM — 7-class OTA testbed (99.67% @ 1 sample/class) + RML2016.10a OOD (11 mods), LoRA 50.0%
    @ 500/class.
  - RIS-MAE — RML2018.01a (24 cls, 48.41% OA @ 1% labels), HisarMod2019.1, CommData, Panoradio.
  - LWM-Spectro — DeepMIMO Phoenix 5-class spectrograms, few-shot macro-F1 (47.41 -> 92.01).
  - WavesFM — own 20-class signal-type set (WiFi/FM/Cellular/BT), 86.05% mean per-class acc.
  - LatentWave — CommRad RF, linear probe 80.9% (Region masking).
  - TorchSig-XCiT — Sig53 (53 cls) top-1 67.97-71.16%.
- **Status in rfbench**: **EXISTS.** `rfbench/tasks/amc/` is implemented; `EVALUATION_PROTOCOL.md`
  §AMC defines it on RadioML 2016.10a / 2018.01a / Sig53. This is the board's most mature track.
- **Recommended canonical dataset + protocol + metric**: keep RadioML 2016.10a (11 cls) as the
  primary board dataset, `amc-radioml2016-strat-snr-8010-seed42-v1`, 80/10/10 stratified by
  (modulation x SNR) seed 42, primary metric `accuracy_overall` over the full SNR range (add
  `accuracy_vs_snr` + `macro_f1`). RML2018.01a is the natural OOD/large track once unblocked.
- **Scope-fit**: core terrestrial-RF signal task; fully in scope. Caveat: WavesFM / LatentWave
  `amc` is a broad signal-TYPE taxonomy (not classic RadioML modulation) — closest normalized
  bucket but note it when comparing to RadioML rows.

### `beam_prediction` — best-beam / beam-index prediction (3 papers)

- **FM evaluators & data/metrics**:
  - IQFM — DeepBeam 5-Beam mmWave (58 GHz), LoRA 94.15% @ 500/class.
  - LWM-base — sub-6 GHz -> mmWave beam prediction on held-out DeepMIMO cities, F1-score.
  - LatentWave — DeepMIMO beam prediction, mean per-class acc (Freq. 63.1%, Supervised 88.9%).
- **Status in rfbench**: **ABSENT.** Not a board task; no dataset/protocol defined.
- **Recommended canonical dataset + protocol + metric**: DeepMIMO (a city scenario held out from
  pretraining), `beam-deepmimo-<scenario>-8010-seed42-v1`, 80/10/10 seed 42, primary metric
  `accuracy_overall` (top-1 beam), report `macro_f1`. Codebook size fixed per scenario manifest.
- **Scope-fit**: **CSI/channel-domain — out of the current terrestrial-signal scope.** If adopted,
  put under a separate **CSI / 6G-sensing track**, not the IQ-signal board.

### `sei` — specific emitter identification / RF fingerprinting (2 papers)

- **FM evaluators & data/metrics**:
  - WirelessJEPA — POWDER 4-device WiFi hardware, 500-shot linear probe 90.5% (vs IQFM 83.4%).
  - IQFM — POWDER RF Fingerprinting (4 devices), LoRA 96.05% @ 500/class.
- **Status in rfbench**: **EXISTS + built out (2026-07, feat/sei-complete).** `rfbench/tasks/sei/`
  implements WiSig / ORACLE / LoRa RFFI with `closed_set` / `cross_receiver` / `cross_day`. Added:
  the paper-exact **`wisig_cnn_paper`** 2-D CNN, **`oracle_cnn`** (Sankhe 2019), SOTA-leaning
  **`complex_cnn`** and **`resnet1d_sei`**, a **`balanced_accuracy`** secondary metric, a dedicated
  **`training_sei.py`** loop (shared AMC `training.py` untouched), and a **POWDER** dataset/track.
  WiSig board rows pending the cluster run.
- **Recommended canonical dataset + protocol + metric**: keep the board's WiSig (ManyTx) as
  primary, `sei-wisig-closedset-strat-tx-8010-seed42-v1` (+ `cross_receiver`, `cross_day` tracks),
  primary `rank1_accuracy` + secondary `balanced_accuracy` (open-set -> `auroc`, `eer`). **FM
  data-parity — ADDRESSED**: both FM SEI evals are on **POWDER** (Reus-Muns GLOBECOM 2020, the 4-BS
  WiFi set), so a **POWDER `closed_set` track** is now scaffolded for a like-for-like comparison
  (download blocked — manual, DRS anti-scrape). FM references are regime-separated: linear-probe
  (WirelessJEPA 90.5 / IQFM 83.4) vs LoRA (IQFM 96.05) — never one column. The fabricated `iqfm`
  WiSig rank1=0.7734 row was **removed** (`a689e86`) — IQFM never evaluates WiSig.
- **Scope-fit**: core terrestrial-RF signal task; in scope.

### `direction_finding` — angle-of-arrival estimation (2 papers)

- **FM evaluators & data/metrics**:
  - WirelessJEPA — AoA on OTA testbed, 1/100-shot linear probe (40.39% @ 1-shot antenna masking).
  - IQFM — AoA over 225 discrete 10-deg angle bins on a 4-rx MIMO testbed, 65.45% @ 1 sample/class,
    92.4% @ 10/class.
- **Status in rfbench**: **ABSENT.** No board task/dataset.
- **Recommended canonical dataset + protocol + metric**: no public canonical AoA dataset in the
  bibliography — both evals are on private OTA MIMO testbeds. If adopted, define as multi-antenna IQ
  angle-bin classification, `df-<testbed>-8010-seed42-v1`, primary `accuracy_overall` (report mean
  angular error in degrees if a regression head is used). Needs a public multi-rx dataset first.
- **Scope-fit**: multi-antenna IQ — plausibly in scope for a terrestrial-signal board, but blocked
  on a public dataset. Treat as **exploratory**.

### `los_nlos` — line-of-sight vs non-line-of-sight (2 papers)

- **FM evaluators & data/metrics**:
  - LWM-base — DeepMIMO (Denver, densified), F1-score, limited-label regime.
  - LatentWave — DeepMIMO, mean per-class acc (Freq. 93.4%, Supervised 95.9%).
- **Status in rfbench**: **ABSENT.**
- **Recommended canonical dataset + protocol + metric**: DeepMIMO held-out city,
  `losnlos-deepmimo-<scenario>-8010-seed42-v1`, 80/10/10 seed 42, primary `accuracy_overall`
  (report `macro_f1`; both papers use F1).
- **Scope-fit**: **CSI/channel-domain — out of current scope.** Separate CSI track if adopted.

### `positioning` — localization / geolocation (2 papers)

- **FM evaluators & data/metrics**:
  - WavesFM — 5G NR CSI at 476 locations, mean distance error 0.41 m.
  - LatentWave — 5G NR CSI (outdoor DeepMIMO), mean positioning error 2.32-2.54 m
    (Supervised 0.71 m).
- **Status in rfbench**: **ABSENT.**
- **Recommended canonical dataset + protocol + metric**: 5G NR CSI positioning set (DeepMIMO
  outdoor or the WavesFM real set if released), `pos-<dataset>-8010-seed42-v1`, 80/10/10 seed 42,
  primary metric **mean positioning error (meters, lower is better)** — a regression metric, so it
  needs a regression-metric entry in the protocol (currently the board has only classification /
  detection / sensing metrics).
- **Scope-fit**: **CSI/channel-domain — out of current scope.** Separate CSI track.

### `har` — human activity recognition / RF sensing (2 papers)

- **FM evaluators & data/metrics**:
  - WavesFM — WiFi CSI, 6 activity classes, 95.67% mean per-class acc.
  - 6G-MSM — CSI HAR (HSD, 6 classes), 93.9% acc (ViT-M).
- **Status in rfbench**: **ABSENT.**
- **Recommended canonical dataset + protocol + metric**: a public WiFi-CSI HAR set (6-class HSD-style),
  `har-<dataset>-8010-seed42-v1`, 80/10/10 seed 42, primary `accuracy_overall` (+ `macro_f1`).
- **Scope-fit**: **CSI-sensing domain — out of current signal scope.** Separate CSI / RF-sensing
  track if adopted.

---

## P2 — evaluated by exactly 1 FM paper

### `interference_id` — jamming / interference detection (WirelessJEPA)

- **Data/metric**: GNSS Jamming 6-class (Zenodo-synthesized), 500-shot linear probe 63.1%.
- **Status in rfbench**: **EXISTS** (implemented 2026-06: `rfbench/tasks/interference_id/`,
  dataset `interf_gnss6`, baseline `interf_cnn` — see CHANGELOG). Distinct from `spectrum_sensing`
  and `wideband_detection`.
- **Recommended**: GNSS jamming 6-class IQ set, `interf-gnss6-8010-seed42-v1`, 80/10/10 seed 42,
  primary `accuracy_overall` (+ `macro_f1`). **Scope-fit**: IQ-signal classification — in scope.

### `protocol_tech_id` — protocol / standard recognition (WirelessJEPA)

- **Data/metric**: OTA WiFi 802.11 ax/b/n/g (4 protocols), (2,1024) IQ, 500-shot linear probe 94.26%.
- **Status in rfbench**: **EXISTS** (implemented 2026-06: `rfbench/tasks/protocol_tech_id/`,
  dataset `tprime_wifi4`, baseline `tprime` — see CHANGELOG). Distinct from `amc` (standard, not
  modulation). **Candidate 2nd dataset**: RFSS `rfss_single.h5` (4k single-source GSM/UMTS/LTE/NR
  — cellular standards, BIBLIOGRAPHY A.6).
- **Recommended**: 802.11-variant IQ set, `proto-wifi4-8010-seed42-v1`, 80/10/10 seed 42, primary
  `accuracy_overall`. **Scope-fit**: IQ-signal classification — in scope.

### `channel_estimation` — MIMO-OFDM channel estimation (WavesFM)

- **Data/metric**: simulated uplink MIMO-OFDM (16-antenna BS, 3GPP UMi), MSE 0.329.
- **Status in rfbench**: ABSENT. Regression to MSE — needs a regression-metric protocol entry.
- **Recommended**: `chanest-<dataset>-8010-seed42-v1`, 80/10/10 seed 42, primary metric MSE
  (lower is better). **Scope-fit**: **CSI/channel-domain — out of current scope.** CSI track.

### `snr_mobility_recognition` — joint SNR x mobility classification (LWM-Spectro)

- **Data/metric**: DeepMIMO Phoenix spectrograms, joint (SNR bin x pedestrian/vehicular) classes,
  validation F1 / accuracy; multi-protocol MoE variant is the same Task 2.
- **Status in rfbench**: ABSENT. New canonical id (not in seed vocab). Classification, not an
  SNR-estimation regression.
- **Recommended**: `snrmob-deepmimo-<scenario>-8010-seed42-v1`, 80/10/10 seed 42, primary
  `accuracy_overall` (+ `macro_f1`). **Scope-fit**: spectrogram-domain; fits a spectrogram track.

### `wideband_detection` — time-frequency signal detection / segmentation (6G-MSM)

- **Data/metric**: 6G-MSM Spectrogram Segmentation (noise / NR / LTE, 3-class per-pixel), 97.6%
  mean segmentation accuracy (ViT-M).
- **Status in rfbench**: **SCAFFOLDED, WIP on the board.** The task package
  (`rfbench/tasks/wideband_detection/`) and a detection-flavored protocol exist
  (`EVALUATION_PROTOCOL.md` §Wideband uses **RadDet**, real/published, primary `mAP` + `mAR`, `IoU`),
  but there is **no committed split and no baseline row** yet — so `leaderboard/tasks.json` declares
  it `wip`, not `implemented`. Note the ONE FM evaluator here does **segmentation**, not box
  detection — a different sub-form of the same time-frequency-localization task.
- **Recommended**: keep RadDet + `mAP` as the board protocol (`detect-raddet-<track>-v1`); if a
  segmentation FM baseline is added, report per-pixel accuracy / mIoU as a separate segmentation
  sub-track rather than mixing it into the `mAP` column. **Scope-fit**: in scope (spectrogram
  detection track).

---

## P3 — 0 FM papers here, but a defined benchmark track

### `spectrum_sensing` — per-subband occupancy detection (pd@pfa)

- **FM coverage**: **NONE** of the 9 FM papers evaluate it. 6G-MSM's spectrogram segmentation is
  `wideband_detection` (time-frequency localization), **not** per-subband occupancy under a pfa
  constraint — so it does not count here.
- **Status in rfbench**: **EXISTS** as a defined track: `EVALUATION_PROTOCOL.md` §Spectrum sensing
  (Wave B) — **DeepSense** (OTA 802.11 a/g + LTE-M), primary metric `pd@pfa=0.1` + latency,
  `sensing-deepsense-<split>-v1`. No FM (or model) implemented against it yet.
- **Recommended**: keep DeepSense + `pd@pfa=0.1` as-is. **High-value gap**: an FM baseline on
  DeepSense would fill an existing board track that currently has **zero** FM coverage. **Scope-fit**:
  in scope (terrestrial OTA occupancy sensing).

### `source_separation` — blind multi-source RF separation (0 FM papers, candidate new track)

- **FM coverage**: **NONE** of the 9 FM papers evaluate it (and RFSS cites no FM paper — the task is
  outside the current FM-eval landscape, which is itself a signal: a hub track here would be ahead of
  the FM literature rather than chasing it).
- **Status in rfbench**: **ABSENT** — not in the taxonomy, the protocol, or the code before 2026-07.
- **Recommended**: **RFSS** (arXiv:2604.00398, BIBLIOGRAPHY A.6) as the canonical dataset —
  100k mixtures (2–4 sources, GSM/UMTS/LTE/5G NR, 3GPP TDL channels + 5 hardware impairments),
  official 70/15/15 index split (`sep-rfss-701515-official-v1`), primary metric **co-channel
  PI-SI-SINR** (higher is better; avoid the adjacent-channel evaluation-floor artifact, RFSS §VII).
  Baselines: Conv-TasNet (−12.34 dB co-channel 2-src) and DPRNN, checkpoints announced.
  **Blocked: dataset NOT released as of 2026-07-03** (HF release announced in the paper, nothing published yet). **Scope-fit**: raw-IQ
  terrestrial signals — in scope; synthetic-but-downloadable (allowed, like `interf_gnss6`).

### `snr_estimation` — per-window SNR regression (0 FM papers, live board track)

- **FM coverage**: **NONE** of the 9 FM papers evaluate scalar SNR regression. LWM-Spectro's Task 2
  is `snr_mobility_recognition` — a joint (SNR-bin x mobility) *classification*, not a regression to
  `snr_db` — so it does not count here. A board regression track here is orthogonal to the FM-eval
  landscape.
- **Status in rfbench**: **EXISTS + live on the board.** `rfbench/tasks/snr_estimation/` is
  implemented; `docs/EVALUATION_PROTOCOL.md` §"Regression metric (`snr_estimation`)" defines it on
  RadioML 2016.10a with primary `rmse_db` (lower is better) + secondary `mae_db`. The board carries a
  **verified** `snr_cnn` baseline (~5.73 dB RMSE) plus `mean_snr` and `snr_moment_ridge` references.
  Declared `implemented` in `leaderboard/tasks.json`.
- **Recommended**: keep RadioML 2016.10a on the single `all_snr` track over the **full SNR range**
  (−20…+18 dB, no cherry-picking), split `snr-radioml2016-strat-snr-8010-seed42-v1` (byte-identical
  indices to the AMC 2016 split `amc-radioml2016-strat-snr-8010-seed42-v1`, so SNR and AMC score the
  exact same held-out signals). The leaderboard ranks `snr_estimation` **ascending** and inverts the
  score bar (lower `rmse_db` is better). **Scope-fit**: raw-IQ terrestrial-signal regression — in
  scope; it is the board's first regression track and the reference for how a regression metric is
  represented (see `positioning` / `channel_estimation`, which will need the same treatment).

---

## Task inventory vs FM taxonomy (build status)

The buckets above are the **FM-coverage** priority (how many FM papers evaluate a task). Build status
in the repo is tracked separately in [`leaderboard/tasks.json`](../leaderboard/tasks.json) (the site
renders from it) and must stay in sync. Snapshot (2026-07):

| Canonical task | tasks.json status | On the board? |
|---|---|---|
| `amc` | implemented | yes — RadioML 2016.10a + 2018.01a columns |
| `sei` | implemented | yes — WiSig (closed/cross-rx/cross-day/open) + ORACLE (closed) |
| `snr_estimation` | implemented | yes — RadioML 2016.10a, verified `snr_cnn` |
| `interference_id` | implemented | yes — interf-gnss6 |
| `protocol_tech_id` | implemented | yes — tprime-wifi4 (within-dist + cross-room) |
| `wideband_detection` | wip | no — protocol frozen, no committed split/baseline |
| `spectrum_sensing` | wip | no — protocol frozen, no committed split/baseline |
| `snr_mobility_recognition` | wip | no — dataset + baseline pending |
| `beam_prediction` / `direction_finding` / `los_nlos` / `positioning` / `har` / `channel_estimation` | planned | no — CSI/channel-domain, out of current scope |

`source_separation` is in the taxonomy but not yet declared as a `tasks.json` entry (blocked on the
unreleased RFSS dataset).

---

## Recommended next tasks to implement (P1 then P2)

Ordered by FM-coverage priority and tied to the existing `rfbench/tasks/<task>/` +
`docs/EVALUATION_PROTOCOL.md` structure. In-scope (IQ terrestrial-signal) tasks first; CSI/channel
tasks flagged for a separate track.

1. **`amc` (P1, EXISTS)** — deepen, don't add: land the WirelessJEPA (74.78%) and IQFM (38.1%) FM
   rows on RadioML 2016.10a, and unblock RML2018.01a for the RIS-MAE 48.41% row.
2. **`sei` (P1, EXISTS)** — add a **POWDER** track so the two FM SEI evaluators (WirelessJEPA, IQFM)
   are comparable (the fabricated `iqfm` WiSig row is already removed, `a689e86`).
3. **`interference_id` (P2, in scope)** — new IQ classification task (GNSS jamming 6-class); small,
   self-contained, mirrors the `amc` task skeleton.
4. **`protocol_tech_id` (P2, in scope)** — new IQ classification task (802.11 ax/b/n/g); same
   skeleton as `amc`.
5. **`spectrum_sensing` (P3, EXISTS, 0 FM)** — implement a first model against the already-defined
   DeepSense track to close the empty board slot (not an FM task yet, but a defined gap).
6. **CSI / 6G-sensing track (P1/P2, out of current scope)** — `beam_prediction`, `los_nlos`,
   `positioning`, `har`, `channel_estimation` cluster heavily in FM papers but are CSI/channel-domain.
   Group them under a **separate CSI track** if the hub expands beyond terrestrial IQ signals; do not
   fold them into the IQ-signal board. `positioning` / `channel_estimation` additionally need a
   regression-metric entry in `EVALUATION_PROTOCOL.md` (MPE meters / MSE).
