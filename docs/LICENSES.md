# Dataset license matrix

Living reference for the licensing terms of every dataset a benchmark task in this repo depends
on. Facts here are pulled from what is **already documented** in `docs/EVALUATION_PROTOCOL.md`,
`docs/BIBLIOGRAPHY.md`, and the download-script docstrings under `rfbench/data/download/` — nothing
below is a new claim; where a source states the license is unconfirmed, this table says so
explicitly rather than guessing.

**Design decision D3 applies to every row without exception**: rf-benchmark-hub **never commits
raw data** (`.h5/.npy/.bin/.sigmf-data/...` are git-ignored and CI-blocked). For every dataset we
only version deterministic **split indices + checksums** under `leaderboard/splits/`; the
`download`/`prepare` scripts fetch the dataset the user is themselves entitled to into
`$RFBENCH_CACHE` at run time. So the "what we redistribute" column is identical across the board
by construction — it is repeated per row for auditability, not because it varies.

| Dataset | Source / citation | Licence (as declared by the source) | What rf-benchmark-hub actually redistributes | Link / DOI |
|---|---|---|---|---|
| RadioML 2016.10a | O'Shea et al.; distributed by DeepSig | **CC BY-NC-SA 4.0** (per `rfbench/data/download/amc_radioml.py` docstring; DeepSig does not permit redistribution) | Nothing (D3) — only `amc-radioml2016-strat-snr-8010-seed42-v1` split indices + checksums | https://www.deepsig.ai/datasets/ (default fetch: Zenodo mirror, record 18397070) |
| RadioML 2018.01a | O'Shea, Roy, Clancy, *IEEE JSTSP* 2018, arXiv:1712.04578, DOI 10.1109/JSTSP.2018.2797022 | **CC BY-NC-SA 4.0** (same DeepSig terms as 2016.10a, per `rfbench/data/download/amc_radioml.py`) | Nothing (D3) — dataset now **prepared** on the cluster; only the `amc-radioml2018-strat-snr-8010-seed42-v1` split indices + checksums are versioned (`leaderboard/splits/radioml_2018_01a/`) | https://opendata.deepsig.io/datasets/2018.01/ |
| Sig53 | Boegner et al., "Large Scale RF Signal Classification," arXiv:2207.09918; TorchSig (MIT, v2.1.1) | Sig53 itself is **not statically distributed** — it is generated on demand by the TorchSig library (MIT-licensed code); no fixed CC/other data license applies because there is no static artifact | Nothing — **excluded** under the repo's "real published datasets only, no generation" policy (`rfbench/data/download/amc_sig53.py` raises a clear blocker error; no loader is wired) | https://github.com/TorchDSP/torchsig |
| WiSig (ManyTx) | Hanna, Karunaratne, Cabric, *IEEE Access* 2022, DOI 10.1109/ACCESS.2022.3154488/3154790, arXiv:2112.15363 | **CC BY-NC-SA 4.0** (per `rfbench/data/download/sei_wisig.py` docstring) | Nothing (D3) — only `closed_set` / `cross_receiver` / `cross_day` split indices + checksums | https://cores.ee.ucla.edu/downloads/datasets/wisig/ ; code github.com/WiSig-dataset/wisig-examples |
| ORACLE | Sankhe, Rajendran, Belgiovine, Chowdhury, Ioannidis, *IEEE INFOCOM 2019*, arXiv:1812.01124, DOI 10.1109/INFOCOM.2019.8737463 | **Unstated** — `rfbench/data/download/sei_oracle.py` documents provenance (Northeastern/Genesys Lab, SigMF captures) and the "never redistribute" rule but the source states no explicit redistribution licence; treat as unconfirmed (same handling as the other DRS/Genesys entries below) | Nothing (D3) — dataset now **prepared** on the cluster; only the `sei-oracle-closedset-strat-tx-8010-seed42-v1` split indices + checksums are versioned (`leaderboard/splits/oracle/`, capped at 1024 windows/capture to bound the index size) | https://genesys-lab.org/oracle |
| LoRa RFFI | Shen, Zhang, Marshall, Peng, Fu, *IEEE JSAC* 2021, DOI 10.1109/JSAC.2021.3087250 (JSAC-2021 headline numbers); actual wired loader reads the related **Shen et al. 2022** release ("Towards Scalable and Channel-Robust RFFI for LoRa," *IEEE TIFS* 2022, IEEE DataPort DOI 10.21227/qqt4-kz19) | **Not confirmed** — `rfbench/data/download/sei_lora.py` documents that IEEE DataPort gates the archive behind a free account login ("open-access" tier) but does not state a redistribution licence; treat as unconfirmed | Nothing (D3); gated behind IEEE DataPort login, **not yet fetched** — no split generated yet. Note: `docs/BIBLIOGRAPHY.md` §A.3 flags the wired loader reads the 2022 dataset (30-device, `gxhen/LoRa_RFFI` layout), not the JSAC-2021 corpus — the JSAC-2021 96.40% figure is not reproducible from the file this loader targets | https://ieee-dataport.org/open-access/lorarffidataset |
| RadDet | Huang, Denman, Pemasiri, Martin, Fookes, *ICASSP 2025*, arXiv:2501.10407 | **CC BY-NC 4.0** on the Kaggle listing (`abcxyzi/raddet-icassp-2025`, per `rfbench/data/download/detection_wbsig53.py` docstring); the GitHub code repo (`github.com/abcxyzi/RadDet`) is separately noted as **CC BY-NC-SA 4.0** in `docs/BIBLIOGRAPHY.md` §A.4 — code and dataset licences differ, both non-commercial | Nothing (D3); gated behind a Kaggle account/licence acceptance, **not yet fetched** — no split generated yet, though the annotation loader (`load_raddet_annotations`) and `wideband_detection` task wiring already exist | https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025 ; code github.com/abcxyzi/RadDet |
| interf-gnss6 | Swinney & Woods, 2021; Zenodo record 4629685 | **CC-BY-4.0**, confirmed (per `docs/EVALUATION_PROTOCOL.md` §interference_id and `rfbench/data/download/interference_gnss.py`) | Nothing (D3) — only `interf-gnss6-8010-seed42-v1` split indices + checksums (dataset already downloaded/prepared and wired to the `interference_id` task) | https://doi.org/10.5281/zenodo.4629685 (`Raw_IQ_Dataset.zip`, no login) |
| tprime-wifi4 (T-PRIME) | Belgiovine et al., arXiv:2401.04837 (extended IEEE INFOCOM 2024); Genesys Lab / Northeastern | **Non confirmé** — `docs/EVALUATION_PROTOCOL.md` §protocol_tech_id and `rfbench/data/download/protocol_tprime.py` both state the DRS landing page gives **no explicit redistribution licence**; openly downloadable, terms unstated | Nothing (D3) — only `proto-tprime-wifi4-8010-seed42-v1` split indices + checksums would be versioned once generated | Northeastern DRS, handle `hdl.handle.net/2047/D20621423` (item `neu:h989s8519`); code github.com/genesys-neu/t-prime |
| DeepSense | Uvaydov, D'Oro, Restuccia, Melodia, *IEEE INFOCOM 2021*, DOI 10.1109/INFOCOM42981.2021.9488764 | **Non confirmé** — no licence is stated anywhere in this repo's docs for DeepSense; not yet investigated | **Spec seulement, pas encore de loader implémenté** — no entry under `rfbench/data/download/` or `rfbench/data/prepare/` references DeepSense; the `spectrum_sensing` task (Wave B) is only described in `docs/EVALUATION_PROTOCOL.md` / `docs/IMPLEMENTATION_PLAN.md` (WP-14/WP-33), not yet coded | https://github.com/wineslab/deepsense-spectrum-sensing-datasets |

## Notes

- "Non confirmé" / "unstated" always means: the primary source (dataset landing page, DataPort,
  Kaggle, DRS) does not display an explicit redistribution licence, per the citations above — it
  is not this repo's own judgment call, and it does not change our behaviour, since D3 means we
  never redistribute raw data regardless of the upstream licence.
- Non-commercial licences (CC BY-NC-\*) on RadioML, WiSig, and RadDet's Kaggle listing are the
  reason `docs/IMPLEMENTATION_PLAN.md` (§Data policy) requires "jamais de données dans le repo ;
  uniquement scripts + indices," enforced by a CI guard on raw-data file extensions/sizes.
- Sig53 and DeepSense are the two entries in this table with **no wired loader at all** — Sig53 is
  an explicit, documented blocker (generation-only, excluded by policy); DeepSense simply has no
  code yet (Wave B, not started).
