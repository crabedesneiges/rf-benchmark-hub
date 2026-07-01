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
- **Dataset**: WBSig53 (impaired).
- **Split** (per split policy): adopt the official WBSig53/TorchSig split if provided; else **80/10/10**,
  seed 42 → `detect-wbsig53-<split>-v1`.
- **Metrics**: `mAP` (**primary**), `mAR`, `IoU`; report detection vs recognition tracks.

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
