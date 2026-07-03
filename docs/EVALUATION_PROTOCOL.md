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
- **Datasets**: WiSig (ManyTx), ORACLE (16-tx), LoRa RFFI.
- **Splits**: `closed_set`, **`cross_receiver`**, **`cross_day`** (WiSig) — reported separately.
- **Metrics**: closed-set → `rank1_accuracy` (**primary**); open-set → `auroc`, `eer`.

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
