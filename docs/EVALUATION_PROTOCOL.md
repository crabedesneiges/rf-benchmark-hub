# Evaluation protocol (NORMATIVE, versioned)

Any change here that alters a metric or split is a **breaking change** → bump the task `version`.

## AMC — automatic modulation classification
- **Datasets**: RadioML 2016.10a (11 cl.), RadioML 2018.01a (24 cl.), Sig53 (53 cl.).
- **Split**: stratified by (modulation × SNR), seed 42, ratios 60/20/20 → `amc-strat-snr-seed42-v1`.
- **Metrics**: `accuracy_overall` (**primary**) over the **full SNR range** (no cherry-picking),
  `accuracy_vs_snr` curve, `macro_f1`.

## SEI — RF fingerprinting
- **Datasets**: WiSig (ManyTx), ORACLE (16-tx), LoRa RFFI.
- **Splits**: `closed_set`, **`cross_receiver`**, **`cross_day`** (WiSig) — reported separately.
- **Metrics**: closed-set → `rank1_accuracy` (**primary**); open-set → `auroc`, `eer`.

## Wideband detection
- **Dataset**: WBSig53 (impaired).
- **Metrics**: `mAP` (**primary**), `mAR`, `IoU`; report detection vs recognition tracks.

## Spectrum sensing (Wave B)
- **Dataset**: DeepSense (OTA 802.11 a/g + LTE-M).
- **Metrics**: ROC (`pd` at fixed `pfa`, **primary** = `pd@pfa=0.1`), inference latency (ms).

## Common rules
- Full protocol conditions are recorded in `result.json.eval.conditions`.
- The primary metric ranks the board; regimes are never mixed in one column.
