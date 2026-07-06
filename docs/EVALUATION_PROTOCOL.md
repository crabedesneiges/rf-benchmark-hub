# Evaluation protocol (NORMATIVE, versioned)

Any change here that alters a metric or split is a **breaking change** → bump the task `version`.

## AMC — automatic modulation classification
- **Datasets**: RadioML 2016.10a (11 cl.), RadioML 2018.01a (24 cl.), Sig53 (53 cl.).
- **Split** (per split policy): **Sig53** adopts the official TorchSig split; **RadioML 2016.10a /
  2018.01a** have no canonical literature split → **80/10/10** stratified by (modulation × SNR), seed 42.
  Canonical ids per dataset, e.g. `amc-radioml2018-strat-snr-8010-seed42-v1`, `amc-sig53-official-v1`.
- **Metrics**: `accuracy_overall` (**primary**) over the **full SNR range** (no cherry-picking),
  `accuracy_vs_snr` curve, `macro_f1`.

## SEI — RF fingerprinting
- **Datasets**: WiSig (ManyTx, 150 tx / 18 rx / 4 days, non-equalized), ORACLE (16-tx), LoRa RFFI,
  and **POWDER** (4-BS WiFi; the dataset the two FM SEI evaluators use — see the POWDER track below).
- **Splits**: `closed_set`, **`cross_receiver`**, **`cross_day`** (WiSig) — reported separately, never
  blended. ORACLE / POWDER are `closed_set` only (single receiver). All are deterministic 80/10/10
  seed 42: `closed_set` stratified by transmitter; `cross_receiver` / `cross_day` **grouped** by
  receiver / day (whole receivers/days held out → no leakage across the boundary the condition
  guards). The **key WiSig result is the cross-receiver drop** vs closed_set; the full SNR-free
  protocol runs every transmitter (no cherry-picking).
- **Metrics**: closed-set → `rank1_accuracy` (**primary**, top-1 identification over the full test
  split) + `balanced_accuracy` (**secondary**, the unweighted mean of per-class recalls — the
  class-balanced accuracy the WiSig paper reports for the imbalanced ManyTx set, `p=0.9`; additive,
  does not change the ranking key); open-set → `auroc` (**primary**), `eer`.
- **Regimes**: `from_scratch` (the specialised baselines: `wisig_cnn_paper` — the paper-exact 2-D
  CNN; `oracle_cnn` — Sankhe 2019; `complex_cnn` and `resnet1d_sei` — SOTA-leaning baselines) +
  `linear_probe` / `full_finetune` via the standard adapters. `wisig_cnn` (compact 1-D) is retained
  as a board-seeding variant, not the paper reproduction.
- **WiSig recipe (baselines, `wisig_cnn_paper`, verbatim `d006_ManyTx_ntx.py`)**: input = first 256
  IQ samples → `(256, 2)`, **unit-average-power normalized** per signal, non-equalized; Adam
  `lr=5e-4`, categorical CE, **class_weight = max(count)/count**, batch 32, ≤100 epochs, early stop
  patience 5 on **val_loss**, best weights restored; **L2 λ=1e-4 on the three Dense layers only**.

### POWDER track (FM-comparable)
- **Dataset**: **POWDER RF Fingerprinting** (Reus-Muns, Jaisinghani, Sankhe, Chowdhury, "Trust in 5G
  Open RANs through Machine Learning: RF Fingerprinting on the POWDER PAWR Platform", *IEEE GLOBECOM
  2020*, pp. 1–6; GENESYS Lab, Northeastern) — 4 WiFi base stations (Tx, USRP X310) → 1 fixed
  receiver (USRP B210), SigMF captures, 5 MS/s @ 2.685 GHz, two capture days. Both public FM SEI
  evaluators use exactly this set. Split id base `sei-powder-wifi4-closedset-strat-dev-8010-seed42-v1`,
  `closed_set` (day-pooled), 256-sample frames (the FM convention; the origin paper used 512).
- **Availability (BLOCKED — manual download)**: publicly reachable **without** POWDER/Emulab
  credentials via the DRS Handle `hdl.handle.net/2047/D20385049` → record `neu:gm80mp276`
  ("POWDER-4BS-IQsample"), BUT the DRS host anti-scrapes programmatic clients (HTTP 403, not
  defeated by a browser User-Agent). So `rfbench data prepare --dataset powder` requires the SigMF
  captures to be placed under `$RFBENCH_CACHE/powder/` **by hand first** (see
  `rfbench.data.download.sei_powder` for the exact procedure). License: unspecified — cite the
  GLOBECOM 2020 paper; do not assert a CC license. We ship only split indices + checksums (never raw
  IQ, D3).
- **FM reference (comparability, separate regime columns — the board never mixes regimes)**: on this
  dataset WirelessJEPA (arXiv:2601.20190) reports **90.5%** and IQFM (arXiv:2506.06718, JEPA's
  reproduction) **83.4%** under a **500-shot linear probe**; IQFM's own paper reports **96.05% @ 500
  samples/class** under a **LoRA fine-tune** (vs 96.64% fully supervised). The 500-shot linear-probe
  column (90.5 / 83.4) and the LoRA column (96.05) are DIFFERENT regimes and must not share a column.
  Our from-scratch `*_cnn` baseline is a third, regime-separated reference on the same dataset.

## Wideband detection
- **Dataset**: **RadDet** (ICASSP 2025 — real published spectrogram images + YOLO time-frequency box
  annotations, 11 classes). *WBSig53 (TorchSig) is generation-only with no static published artifact, so
  it is excluded under the "use real published datasets, do not generate" policy.*
- **Split** (per split policy): adopt RadDet's official train/val/test split if provided; else **80/10/10**,
  seed 42 → `detect-raddet-<track>-v1`.
- **Metrics**: `mAP` (**primary**), `mAR`, `IoU`; report detection vs recognition tracks.

## interference_id — GNSS jamming classification
- **Dataset**: **interf-gnss6** (Swinney & Woods 2021, Zenodo record 4629685, DOI
  10.5281/zenodo.4629685, CC-BY-4.0) — a **raw-IQ 6-class** GNSS-jamming set: `DME`,
  `narrowband`, `single_am`, `single_chirp`, `single_fm`, `no_jamming`. The archive
  (`Raw_IQ_Dataset.zip`, ~1.9 GB) downloads without login.
  - **Honesty note**: the six signals are MATLAB-**synthesised** (a `signal_generation.m` script
    ships alongside), BUT the corpus is distributed as a **ready-to-download raw-IQ archive**, so
    it is treated as a public-download dataset (fetched, not regenerated) — *not* a generation-only
    blocker like Sig53/WBSig53.
- **Split** (per split policy): no canonical literature split adopted here → deterministic
  **80/10/10** **stratified by class**, seed 42 → `interf-gnss6-8010-seed42-v1`.
- **Regimes**: `from_scratch` (the `interf_cnn` baseline) + `linear_probe` / `full_finetune` via
  the standard adapters.
- **Metrics**: `accuracy_overall` (**primary**, top-1 over the whole test split), `macro_f1`
  (unweighted mean of per-class F1). No SNR curve (the set has no SNR grid).
- **Literature reference**: Morales-Ferre et al. 2019 (CNN/SVM on STFT images, 91.36/94.90%) and
  Swinney & Woods 2021 (CNN-feature + transfer learning on this exact raw-IQ set).

## protocol_tech_id — WiFi 802.11 standard recognition
- **Dataset**: **tprime-wifi4** (T-PRIME, Genesys Lab / Northeastern; paper arXiv:2401.04837,
  code github.com/genesys-neu/t-prime, data on the Northeastern DRS collection `neu:h989s847q`)
  — a **real over-the-air raw-IQ 4-class** WiFi-standard set: `802.11b`, `802.11g`, `802.11n`,
  `802.11ax`, captured at 20 MHz (802.11b upsampled 11→20 MHz for consistency). The archive
  (~66 GB) is stored as raw interleaved-IQ `.bin` captures. This recognises the WiFi **standard**,
  which is distinct from AMC (modulation scheme).
  - **License note**: the DRS collection is **openly downloadable** (no login), but the dataset's
    redistribution LICENSE is **unstated** on the landing page — treat the terms as unconfirmed.
    We only ship split indices + checksums (never raw IQ, D3), so redistribution is not at issue.
  - **Cluster-confirm TODO**: the on-disk `.bin` layout (float32 interleaved `[I0,Q0,I1,Q1,…]`
    assumed) and whether a capture is one fixed window or a long recording to tile must be
    confirmed on Lustre; the label + array loaders tile with the SAME window/stride to keep the
    committed split indices aligned.
- **Split** (per split policy): the T-PRIME repo ships **no** official train/test index lists →
  deterministic **80/10/10** **stratified by protocol class**, seed 42 →
  `proto-tprime-wifi4-8010-seed42-v1`.
- **Regimes**: `from_scratch` (the `tprime` baseline) + `linear_probe` / `full_finetune` via the
  standard adapters.
- **Metrics**: `accuracy_overall` (**primary**, top-1 over the whole test split), `macro_f1`
  (unweighted mean of per-class F1). No SNR curve (the set has no SNR grid).
- **Baseline**: `tprime` — the T-PRIME transformer over raw interleaved IQ with NO learned input
  embedding: a `(2, N)` window is sliced into `M` tokens of `(2, S)`, each flattened to a `1×2S`
  token fed straight to a 2-layer transformer encoder. Default = **SM** (`M=24`, `S=64`,
  `N=1536`, ~1.6M params); **LG** (`M=64`, `S=128`, `N=8192`, ~6.8M params) via `model.variant=LG`.

## Spectrum sensing (Wave B)
- **Dataset**: DeepSense (OTA 802.11 a/g + LTE-M).
- **Split** (per split policy): adopt the official DeepSense split if provided; else **80/10/10**,
  seed 42 → `sensing-deepsense-<split>-v1`.
- **Metrics**: ROC (`pd` at fixed `pfa`, **primary** = `pd@pfa=0.1`), inference latency (ms).

## Common rules
- **Split policy**: if the dataset ships a split used by the literature, adopt it verbatim and record
  its provenance in the manifest. Otherwise generate a **deterministic 80/10/10** train/val/test split,
  stratified by the task's label structure, **seed 42**. Ratios + seed are baked into
  `canonical_split_id`; changing either is a breaking change → bump the task `version`.
- Full protocol conditions are recorded in `result.json.eval.conditions`.
- The primary metric ranks the board; regimes are never mixed in one column.
