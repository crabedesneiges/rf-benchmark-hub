# Bibliography & Reproduction Audit

Reference bibliography for RF-Benchmark-Hub baselines and foundation models, plus a per-model
**reproduction audit** of our implementations against the source papers.

Scope: terrestrial RF only. Our board's primary metrics: AMC `accuracy_overall` (full SNR range),
SEI `rank1_accuracy` (per-track), detection `mAP`, sensing `pd@pfa`.

Conventions:
- `(?)` = number/detail not confirmed against a primary source.
- "Our score" = the value we currently report on the board (`leaderboard/results/**`).
- All our AMC numbers are on **RadioML 2016.10a**, 11 classes, full SNR range (−20…+18 dB),
  80/10/10 stratified by (modulation × SNR), seed 42 — see `docs/EVALUATION_PROTOCOL.md`.
- Our training recipe (all from-scratch baselines, `rfbench/training.py`) was **fixed in 2026-06**:
  Adam lr=1e-3, CrossEntropy, **best-val-accuracy checkpoint, ReduceLROnPlateau (on val loss),
  early stopping on val accuracy (patience 40), gradient clipping 5.0, NaN guard**, batch 256,
  150 epochs, no augmentation. The Part B audit below was performed against the OLD recipe
  (fixed epochs, no schedule, no early stop) — its recipe-row verdicts are historical; the
  architecture-fidelity rows remain current unless noted.

---

## Part A — Bibliography by task and by foundation model

### A.1 AMC — RadioML 2016.10a (11 classes, −20…+18 dB, len=128)

Cross-paper overall-accuracy table. Reported = full-range overall accuracy unless noted; peak/high-SNR
figures (~90% @ +18 dB) are a different metric and are NOT used here.

| Model | Reported overall (source) | Independent repro | Our score | Reproduction status |
|---|---|---|---|---|
| **MCLDNN** (Xu 2020) | **61.01%** (TCN-GRU T3) | 61.52% (TLDNN T2) | **61.71%** | Reproduced, **+0.7 pt** (final recipe, 2026-06) |
| **CLDNN** (West & O'Shea 2017) | **60.56%** (TCN-GRU T3) / ~61% (orig. text) | — | **58.05%** | Reproduced, −2.5 pt (final recipe, 2026-07; paper-faithful 3-LSTM+skip arch + per-sample input norm — the collapse was an input-scale init fragility, fixed) |
| **ResNet** (O'Shea 2018) | **57.32%** (TLDNN T2) / 56.38% (TCN-GRU) | — | **56.61%** | Reproduced, −0.7 pt (final recipe; 3-stack len-128 adaptation) |
| VT-CNN2 / CNN2 (O'Shea 2016) | ~56.98% (TCN-GRU T3) | — | not run | Missing |
| LSTM2 (Rajendran 2018) | 61.02% (TLDNN T2) / 58.49% (TCN-GRU) | — | not run | Missing |
| GRU2 | 56.92% (TCN-GRU T3) | — | not run | Missing |
| PET-CGDNN (Zhang 2021) | ~60.1% (?) (secondary) | — | not run | Missing |
| MCformer | 60.54% (TLDNN T2) | — | not run | Missing |
| LSTM-DAE | 61.42% (TLDNN T2) | — | not run | Missing |
| TCN-GRU (Sensors 2024) | 61.56% (own T3) | — | not run | Missing |
| **TLDNN** (Qu 2024) — SOTA | **62.83%** (+SS 63.35%) | — | not run | Missing (target ceiling) |

Honest ceiling on 2016.10a is ~61–63% (11 classes). The 2026-06 recipe fix (val-accuracy checkpoint +
LR schedule + early stopping — Part B item 1, now addressed) closed the gap: MCLDNN now sits **above**
its paper target (61.71 vs 61.01) and ResNet within 0.7 pt. CLDNN's chance-collapse under the final
recipe was root-caused (2026-07) to an input-scale init fragility — RadioML's ~1e-2-RMS IQ fed to a
no-norm/no-BN 3-LSTM stack — and fixed by per-sample unit-variance input normalization (the ResNet
transform); the retrained 58.05% is the first honest figure for the paper-faithful arch (the prior
58.76% was the superseded 2-LSTM/no-skip arch).

Primary papers:
- **MCLDNN** — Xu, Luo, Chen, Luo, Wu, "A Spatiotemporal Multi-Channel Learning Framework for Automatic
  Modulation Recognition," *IEEE WCL* 9(10):1629–1632, 2020. DOI 10.1109/LWC.2020.2999453.
  Code: https://github.com/wzjialang/MCLDNN (Keras/TF). No arXiv.
- **CLDNN** — West & O'Shea, "Deep Architectures for Modulation Recognition," *IEEE DySPAN 2017*.
  arXiv:1703.09197. DOI 10.1109/DySPAN.2017.7920754. No canonical code.
- **ResNet** — O'Shea, Roy, Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification,"
  *IEEE JSTSP* 12(1):168–179, 2018. arXiv:1712.04578. DOI 10.1109/JSTSP.2018.2797022.
  (Original paper targets **2018.01a / 24-class**, peak ~95%; the 57.32% is the community 2016.10a number.)
- **PET-CGDNN** — Zhang, Luo, Xu, Luo, *IEEE Comm. Lett.* 25(10), 2021. arXiv:2110.04980.
  Code: https://github.com/Richardzhangxx/PET-CGDNN. Headline is param-efficiency (~71–75k params).
- **TLDNN** (SOTA + source of Table II baselines) — Qu, Lu, Zeng, Wang, Wang, "Enhancing Automatic
  Modulation Recognition through Robust Global Feature Extraction," *IEEE TVT* 2024. arXiv:2401.01056.
  Clean recipe worth adopting board-wide: **AdamW lr=1e-3, ReduceLROnPlateau (×0.1, patience 10),
  ≤150 epochs, batch 128, split 6:2:2, A/P input transform** (amplitude min-max→[0,1], phase→[−1,1] rad).
- **TCN-GRU** (source of second baseline table) — "Robust AMC via a Lightweight Temporal Hybrid Neural
  Network," *Sensors* 24(24):7908, 2024. DOI 10.3390/s24247908.

### A.2 AMC — RadioML 2018.01a (24 classes, −20…+30 dB, len=1024) — *dataset blocked on cluster*

| Model | Overall (full SNR) | Peak / high-SNR | Source |
|---|---|---|---|
| ResNet (O'Shea 2018) | 60.91% | ~95.7% (SNR>8 dB) | TLDNN T2 / orig. |
| MCLDNN | 61.92% | — | TLDNN T2 |
| LSTM2 | 62.52% | — | TLDNN T2 |
| FEA-T | 62.37% | — | TLDNN T2 |
| LSTM-DAE | 61.32% | — | TLDNN T2 |
| TLDNN (+SS) | 63.32% (63.42%) | — | TLDNN T2 |
| **RIS-MAE** (1% labels) | **48.41%** (κ 0.4616) | — | RIS-MAE T2 |

Note: papers most often quote peak ~95% on 2018.01a; full-range overall is only ~61–63% due to the deep
negative-SNR floor. Match our board metric to the full-range column.

### A.3 SEI / RF fingerprinting

Headline rank-1 numbers to reproduce (primary sources):

| Dataset | #cls | Model (paper) | Closed-set rank-1 | Cross-condition | Our loader / score |
|---|---|---|---|---|---|
| WiSig ManyTx | 150 | 2-D CNN (5 conv/3 dense) | ~53% (150 tx, non-eq, 50 sig) / ~80% (10 tx) | **cross-rx 99%→<33%**; cross-day 99%→drop | `wisig_cnn` (1-D CNN) — no board score (fabricated 0.9412 removed, `a689e86`) |
| WiSig ManySig | ≤6 | same | >99% same-day/rx | cross-day degradation | — |
| ORACLE | 16 | 2 conv + 2 FC, 2×128 raw IQ | **98.60%** | **cross-location 87.13%** | loader present (window=128); **no `oracle_cnn` model** |
| LoRa RFFI (JSAC'21) | 10 (study cites 25) | spectrogram-CNN (3 conv/1 FC) | **96.40%** (95.35% CFO-only) | 83.53% w/o CFO comp | loader reads WRONG dataset (see below); **no LoRa model** |

Board note (updated 2026-06): the fabricated SEI rows — `iqfm` rank1 0.7734 on WiSig cross_receiver
(IQFM's paper never evaluates WiSig) and the WiSig-CNN closed-set 0.9412 on a synthetic fixture split
with an unearned `verified` badge — were **removed from the board** in the pre-deploy cleanup
(commit `a689e86`). The board currently has **no SEI rows**; the analysis below stands as the
reproduction target for a real WiSig run.

Primary papers:
- **WiSig** — Hanna, Karunaratne, Cabric, "WiSig: A Large-Scale WiFi Signal Dataset...," *IEEE Access*
  10:22808–22818, 2022. DOI 10.1109/ACCESS.2022.3154488. arXiv:2112.15363.
  Code: github.com/WiSig-dataset/wisig-examples. Data: cores.ee.ucla.edu/downloads/datasets/wisig/.
  - Signal: **first 256 IQ samples** of the preamble → `(256, 2)`. **Unit average power normalized.**
    ManyTx headline (`d006`) uses **non-equalized** data (`equalized=0`).
  - Baseline CNN (`py/d006_ManyTx_ntx.py`): input `(256,2)`→`Reshape(256,2,1)`→5 conv blocks
    (filters **8/16/16/32/16**, kernels (3,2)/(3,1), pools (2,1)/(2,2))→Flatten→Dense(100)→Dense(80)
    →Dropout(0.5)→Dense(N, softmax). `padding='same'`, **L2 λ=1e-4**.
  - Recipe: **Adam lr=5e-4**, categorical CE, **100 ep + early stop patience=5 on val_loss** (best
    weights restored), **class_weight** for imbalance, **batch=32** (Keras default), split 80/10/10.
    Reports **balanced** test accuracy.
- **ORACLE** — Sankhe, Rajendran, Belgiovine, Chowdhury, Ioannidis, "ORACLE...," *IEEE INFOCOM 2019*.
  arXiv:1812.01124. DOI 10.1109/INFOCOM.2019.8737463. Data: genesys-lab.org/oracle.
  - 16 bit-identical USRP X310, single B210 rx, 802.11a, raw IQ **no equalization**, window **128** → `2×128`.
  - CNN (Fig. 4): **2 conv + 2 FC** — Conv1 50@1×7 ReLU, Conv2 50@2×7 ReLU, FC1 256, FC2 80, softmax;
    Dropout 0.5, L2 λ=1e-4.
  - Recipe: **Adam lr=1e-4**, categorical CE, **early stop patience 10** on val acc; 200K/10K/50K
    train/val/test windows per device (16-tx run). Static same-location → **98.60%**; different-location
    → **87.13%** (cross-channel drop). ORACLE has ONE rx → our cross_receiver/cross_day are N/A.
- **LoRa RFFI (JSAC'21)** — Shen, Zhang, Marshall, Peng, Fu, "RFFI for LoRa Using Deep Learning,"
  *IEEE JSAC* 39(8):2604–2616, 2021. DOI 10.1109/JSAC.2021.3087250.
  - 10 LoRa DUTs (study cites 25), USRP N210, cabled 40 dB attenuator (channel removed), SF7/BW125.
  - Best model: **spectrogram-CNN** (LeNet-style, **3 conv 8/16/32 3×3 + BN + 1 FC leaky-ReLU**),
    input **102×63 spectrogram** (STFT rect win 256, hop 128), **1,545,193 params**.
  - Recipe: Adam init lr=3e-4, **LR ×0.3 / 10 ep**, batch 32, L2 1e-4, **60 ep, no early stop**.
  - **CFO compensation** is the load-bearing knob: **83.53% (no CFO) → 95.35% (+CFO) → 96.40% (+hybrid)**.
  - **DATASET MISMATCH:** our loader (`rfbench/data/prepare/sei.py:434-471`) reads
    `dataset_training_aug.h5` (1-indexed labels, `(n, 2*n_samples)` layout) which is the **2022**
    `gxhen/LoRa_RFFI` release (30 devices, ResNet + augmentation), NOT the JSAC-2021 data. The
    JSAC-2021 96.40% is **not reproducible** from this file. Cite instead: **Shen et al. 2022, "Towards
    Scalable and Channel-Robust RFFI for LoRa," IEEE TIFS 2022**, IEEE DataPort DOI 10.21227/qqt4-kz19.

### A.4 Wideband detection & spectrum sensing

| Dataset | Best model | Score | Metric | Source |
|---|---|---|---|---|
| **RadDet** (NIST-CBRS real) | RT-DETR-L @512² | 95.31 mAP50 / 80.96 mAP50:95 | COCO-AP, SNR-avg | RadDet |
| RadDet-9T (dense synth) | YOLOv3-L @512² | 60.37 mAP50 / 53.97 | COCO-AP | RadDet |
| RadDet-1T (sparse synth) | YOLOv3-L @512² | 31.85 mAP50 / 25.41 | COCO-AP | RadDet |
| **WBSig53** detection (impaired) | DETR-B4-Nano | 86.98 mAP / 98.92 AP50 | COCO-AP | WBSig53 |
| WBSig53 recognition (6 fam) | DETR-B4-Nano | 80.65 mAP | COCO-AP | WBSig53 |
| **DeepSense** (WiFi in-the-wild) | lightweight CNN | Precision 98% / Recall 97% | pd/pr + latency | DeepSense |

Our board: detection track adopts **RadDet** (real, published). WBSig53 is a **blocker**
(generation-only). No detection/sensing model implemented yet.

Primary papers:
- **RadDet** — Huang, Denman, Pemasiri, Martin, Fookes, *ICASSP 2025*. arXiv:2501.10407.
  Code: github.com/abcxyzi/RadDet (CC BY-NC-SA 4.0). 40k frames, 11 radar classes, 6 SNR settings,
  3 resolutions (128²/256²/512²). Models: YOLOv3/v6-M/v9-M + RT-DETR-L. Recipe: AdamW, linear
  0.01→0.001, 3-ep warmup, 100 ep, input [0,1], heavy augmentation.
- **WBSig53** — Boegner et al., "Large Scale RF Wideband Signal Detection & Recognition,"
  arXiv:2211.10335 (2022). TorchSig: github.com/TorchDSP/torchsig. 550k examples / 53 classes / 6
  families. Input: 262,144 IQ → 512-pt FFT, no overlap, **Blackman-Harris** window → 512×512 →
  real/imag 2-ch → `(2,512,512)`. Models: YOLOv5 pico/nano/small, DETR-Nano (EffNet B0/B2/B4 + XCiT),
  PSPNet, Mask2Former. **DETR beats YOLO here** (opposite of RadDet). No augmentation.
- **DeepSense** — Uvaydov, D'Oro, Restuccia, Melodia, *IEEE INFOCOM 2021*, pp. 1–10.
  DOI 10.1109/INFOCOM42981.2021.9488764. Code: github.com/wineslab/deepsense-spectrum-sensing-datasets.
  Per-subband multi-label occupancy from a **32-IQ-sample** window; CNN (2×Conv16 k3 → pool → 2×Conv32
  k5 → pool → Dense64 → Dense K sigmoid), Adam lr=1e-3, batch 256, 150 ep, `binary_crossentropy`.
  **Precision 98% / Recall 97%**, latency **0.61 ms** (FPGA in-the-loop, not GPU).
  **pd@pfa=0.1 exact value not recovered from a primary source `(?)`** — our board's primary metric
  is `pd@pfa=0.1`, but the paper reports operating-point precision/recall, not a tabulated pd@pfa. Do
  not cite a specific pd@pfa=0.1 without the Xplore PDF.

Cross-dataset caveat: rankings do **not** transfer (DETR wins WBSig53, YOLOv3-L wins RadDet synthetic,
DeepSense is a different task). Keep the three eval protocols separate.

### A.5 Foundation models

Board-comparability warning: only **two** public FM results are on RadioML 2016.10a full SNR / 11-class
(our exact board setting): **WirelessJEPA 74.78%** and **IQFM 38.1%**. Everything else reports on other
datasets/protocols and must carry an asterisk.

Consolidated board-comparability table (AMC / RadioML only):

| Model | Weights | RadioML setting | Protocol | Reported | Our score | Board-comparable? |
|---|---|---|---|---|---|---|
| **WirelessJEPA** | ✗ (retrain) | 2016.10a, 11-cls, −20…+18 | linear probe, 500-shot, OOD | **74.78%** | wrapper implemented (`wireless-jepa`), in-repo retrain pending — NOT the paper's OOD 74.78% | ✅ beats our MCLDNN 61.71 |
| **IQFM** | ✗ (retrain) | 2016.10a, 11-cls, full SNR | linear probe, 50/cls, OOD | **38.1%** | wrapper implemented (`iqfm-base`), in-repo retrain pending — NOT the paper's OOD 38.1% | ✅ metric; ✗ data regime |
| **RIS-MAE** | ✗ (retrain) | 2018.01a, 24-cls | fine-tune, 1% labels | **48.41%** | not run | ✅ if 2018 unblocked |
| **LWM-Spectro** | ✅ HF (MIT declared, no LICENSE file) | **none** (DeepMIMO 5-cls) | few-shot F1, real linear/FT head | 47–95 F1 (own data) | **no row** (OOD; removed) | ❌ no RadioML in paper — own task reproduced (B.5) |
| **WavesFM** | ✗ `(?)` | none (own 20-cls) | fine-tune | 86.05% | not run | ❌ |
| **LatentWave** | ✗ | none (CommRad) | linear probe | 80.9% | not run | ❌ |
| **6G-MSM** | ✗ | none (CSI/seg) | fine-tune | 93.9 / 97.6% | not run | ❌ (no AMC) |
| **TorchSig XCiT** | ✗ (train from Sig53) | none (Sig53 53-cls) | supervised | Nano 67.97 / Tiny12 71.16 | not run (mislabeled row removed from board, `a689e86`) | ❌ (Sig53 excluded) |

Primary sources & key facts:
- **LWM-Spectro** (`wi-lab/lwm-spectro`, integrated + verified; **not** on the board) — Kim, Alikhani, Alkhateeb, arXiv:2601.08780
  (2026-01, cs.IT). **License: MIT is the only upstream signal — declared in `pyproject.toml`
  (`license = {text = "MIT"}` + OSI classifier) and README_model.md ("License: MIT"), but NO standalone
  LICENSE file ships (README frontmatter has `#license: mit` commented out; config.json has no license
  field). Effective status: MIT declared, no LICENSE file present — permissive (commercial use allowed,
  attribution via the paper citation); verify before a public leaderboard.** 12-layer
  Transformer d=128 h=8, 4×4 patches → seq 1024, 128×128 spectrogram input; MoE (WiFi/LTE/5G experts,
  top-1 router). Pretrained on **9.2M synthetic DeepMIMO spectrograms** — no real captures, **no
  RadioML**. Paper AMC = 5-class DeepMIMO spectrograms, few-shot **macro-F1**, real linear/FT head
  (LWM linear-probe 47.41→92.01 F1 over 5→400 shots). **There is no published LWM-Spectro RadioML
  number**, so we report **no** RadioML row for it; instead we reproduced the paper's own
  `snr_mobility` task at **93.9%** from the shipped ckpt + demo data (see B.5).
- **LWM base** — Alikhani, Charan, Alkhateeb, arXiv:2411.08872 (2024). CSI/channel model (12L, d=64,
  ~600k params), MCM pretraining. LWM-Spectro's lineage; **CSI-only, not a terrestrial-signal baseline**
  — exclude from AMC/SEI board.
- **TorchSig XCiT / Sig53** — Boegner et al., arXiv:2207.09918 (2022). MIT, v2.1.1. **Ships NO pretrained
  weights** (train-from-Sig53 only); v2.x dropped the named `WBSig53` dataset. XCiT-Nano ≈3.1M params,
  Tiny12 ≈6.7M. Input raw IQ len 4096 as 2-ch real, AdamW wd 0.04 lr 2.5e-4, 1M steps, batch 32.
  Impaired-val top-1: **Nano 67.97% (static)**, Tiny12 70.22 static / 71.16 online. Sig53 excluded from
  our board (generation-only).
- **WavesFM** — arXiv:2504.14100 (2025). **No public weights/code URL** — paper-only. ViT 12-block
  embed 512 (~38M), LoRA; MWM pretraining mask 75% on **real** RF spectrograms (3,332) + CSI. RF
  classification on own 20-class set: 86.05% (pretrained) vs 88.07% supervised. Not RadioML.
- **IQFM** — Mashaal, Abou-Zeid, arXiv:2506.06718v2 (2025). CC-BY 4.0, no weights. **ShuffleNetV2 0.5×,
  ~341k params**, contrastive SSL (SimCLR/InfoNCE), unit-max norm `iq/max(|iq|)`. OTA MIMO testbed.
  **OOD RML2016.10a: 38.1% @ 50 samples/class linear probe** (only IQFM RadioML figure).
  **Does not evaluate WiSig SEI** — our 0.7734 SEI row is fabricated.
  **Status (2026-07): board wrapper `iqfm-base` IMPLEMENTED** (`rfbench/models/foundation/iqfm.py` +
  reusable 1-D ShuffleNetV2-x0.5 backbone `shufflenet1d.py`, measured 335,096 params — the 1-D-vs-2-D
  delta from the paper's ~341k). Weights unpublished → we (re-)pre-train the recipe IN-REPO with SimCLR
  on RadioML-train delabelised (`scripts/pretrain/iqfm_simclr.py`, `slurm/pretrain_iqfm_arm.sh`), which
  is **in-distribution, NOT the paper's OOD OTA setting** — any resulting score is ours and must never
  be presented as the 38.1% figure. No `result.json` committed until a real cluster run lands.
- **WirelessJEPA** — arXiv:2601.20190 (2026). No weights. ShuffleNetV2-x0.5 (matched to IQFM), JEPA
  masked latent prediction, EMA teacher 0.996→1.0, no augmentation. Same OTA testbed as IQFM.
  **500-shot linear probe, OOD RML2016.10a (11 mods, −20…+18 dB): 74.78%** — the single most
  board-comparable public FM number, and it **beats our supervised MCLDNN (60.08%)**. Weights
  unreleased → would require retraining the JEPA recipe.
  **Status (2026-07): board wrapper `wireless-jepa` IMPLEMENTED**
  (`rfbench/models/foundation/wireless_jepa.py`), reusing IQFM's shared 1-D ShuffleNetV2-x0.5
  backbone `shufflenet1d.py` (the "matched to IQFM" contract — same `build_shufflenet1d`, 335,096
  params). Weights unpublished → we (re-)pre-train the JEPA recipe IN-REPO (masked-latent + EMA
  teacher 0.996→1.0, no augmentation) on RadioML-train delabelised
  (`scripts/pretrain/wireless_jepa.py`, `slurm/pretrain_wireless_jepa_arm.sh`), which is
  **in-distribution, NOT the paper's OOD OTA setting** — any resulting score is ours and must never
  be presented as 74.78%. No `result.json` committed until a real cluster run lands.
- **RIS-MAE** — Liu, Liu et al., arXiv:2508.00274 (2025). No weights. ViT-MAE encoder 12L d=768, 1D
  IQ patches of 8 (len 1024 → 128 patches), mask 75%. **2018.01a 24-cls, 1% labels, full SNR:
  48.41% OA / κ 0.4616** (beats MCLDNN 31.92 in that regime). Relevant only if 2018 unblocked.
- **LatentWave** — arXiv:2606.06373 (2026). No weights. ViT 8L/256d/6.4M, JEPA, EMA. Linear-probe RF
  classification **80.9% on CommRad** (not RadioML). Spectrogram-track FM baseline.
- **6G-MSM** ("Building 6G Radio FMs") — Aboulfotouh, Eshaghbeigi, Abou-Zeid, arXiv:2411.09996 (2024).
  Code pending. MAE-ViT (S/M/L), MSM mask 70–80% on OTA spectrograms. CSI HAR 93.9%, spectrogram
  segmentation 97.6%. **No AMC** — a pretraining template for a spectrogram/sensing track, not a baseline.

### A.6 RF source separation (candidate task — no board track yet)

Mined from **RFSS** (Chen, Jin, Tan, arXiv:2604.00398, 2026-04 — v2 of arXiv:2508.12106, cite the
2026 version). First public labeled corpus for blind multi-source RF separation; its related work
confirms our gap analysis: RadioML is single-signal (no mixtures, no per-source ground truth), DARPA
SC2 targets protocol research, and the audio corpora (WSJ0-2mix, WHAM!, MUSDB18) had no RF equivalent.

| Method | 2-src overall PI-SI-SINR | 2-src co-channel | Source |
|---|---|---|---|
| **Conv-TasNet** | **−21.18 dB** | **−12.34 dB** | RFSS T1/T2 |
| DPRNN | −21.53 dB | −12.51 dB (3-src −10.38) | RFSS |
| CNN-LSTM (regression) | −23.32 dB | −17.04 dB | RFSS |
| Frobenius-NMF | −26.07 dB | −16.19 dB | RFSS |
| FastICA | −34.91 dB | −28.04 dB | RFSS |

- **RFSS dataset**: 100k multi-source samples (2–4 sources), 4 standards (GSM/UMTS/LTE/5G NR),
  3GPP-exact waveform generation, per-source TDL-A..E channels + 5 hardware impairments (CFO, I/Q
  imbalance, phase noise, DC offset, Rapp PA), co-channel + adjacent-channel mixing, 30.72 MHz,
  122,880 IQ/sample (~4 ms). **103 GB HDF5** (`rfss_dataset.h5`), split 70/15/15 **by index range**
  (0–69,999 / 70,000–84,999 / 85,000–99,999 — official split to adopt). Plus `rfss_single.h5`
  (4,000 single-source, 1.3 GB) usable for **cellular-standard classification** (a candidate second
  `protocol_tech_id` dataset: 4 cellular standards vs our 4 WiFi standards).
- **Metric**: PI-SI-SINR (permutation-invariant SI-SNR, Le Roux 2019), *absolute* output (input
  SI-SINR is effectively −∞). **Co-channel is the recommended comparison metric** — adjacent-channel
  scores hit a ~−28 dB evaluation-floor artifact (baseband references vs frequency-shifted
  estimates), acknowledged in §VII.
- **Availability — NOT RELEASED (checked 2026-07-03)**: the paper announces a HuggingFace release
  "at submission time" (dataset, generation code, checkpoints, eval scripts) but **nothing is
  published yet**. Do NOT build a track until the release lands. Once released: synthetic but
  distributed as a static download → NOT a generation-only blocker (same category as `interf_gnss6`).
- **Related, absent from our biblio**: **RF Challenge** (Lancho et al., *IEEE OJ-COMS* 2025,
  arXiv:2409.08839, ICASSP 2024) — interference **cancellation** (1 known signal + 1 unknown
  interferer, BER metric), real OTA recordings; adjacent to `interference_id` but a different task
  (rejection, not classification, not blind separation). **RF Transformer for signal separation**
  (arXiv:2603.09201, 2026) — screen as a potential separation baseline `(?)` (not yet read).

---

## Part B — REPRODUCTION AUDIT

For each implemented model: OUR code vs the paper, per aspect. MATCH / MISMATCH / UNKNOWN.
"Gap driver?" flags the discrepancies most likely to explain the score gap.

### B.1 MCLDNN — `rfbench/models/baselines/mcldnn.py` — our **61.71%** vs paper 61.01% (repro 61.52%)

> **RESOLVED (2026-06).** The mismatches below (fusion concat 50→100, dropout-regularized 2-dense
> head, recipe) were fixed in the paper-conformance pass; retrained score 60.08 → **61.71** (above
> the paper target). Table kept as the historical audit record.

| Aspect | Paper (Xu 2020, official repo) | Our code | Verdict |
|---|---|---|---|
| Input branches | 3 branches: combined I/Q `(2,128,1)`, separate I, separate Q | 3 branches (conv_iq 2-D, conv_i/conv_q 1-D) — `mcldnn.py:95-107` | MATCH |
| Branch convs | Conv2D(50,(2,8)); Conv1D(50,8,causal); fuse Conv2D(50,(1,8)); Conv2D(100,(2,5),valid) | 50 filters, kernels (2,8)/(1,8)/(2,5); **fuse conv outputs 50 not 100** (`conv_fuse` out=conv_filters) `mcldnn.py:114-117`; **1-D convs use `padding="same"` not causal** | MISMATCH (minor: filter width 50 vs 100 on fusion; same vs causal) |
| LSTM | LSTM(128, return_seq) → LSTM(128) | 2-layer `nn.LSTM(128)` — `mcldnn.py:120-125` | MATCH |
| Dense head | Dense(128,selu)→**Dropout(0.5)**→Dense(128,selu)→**Dropout(0.5)**→Dense(11) | single Dense(128)+SELU, **NO dropout, only ONE dense** — `mcldnn.py:127-131` | **MISMATCH — gap driver** (regularization + capacity of head reduced) |
| Preprocessing | raw IQ `2×128`, no transform | raw IQ `2×128` | MATCH |
| Optimizer / LR | Adam (default 1e-3) | Adam lr=1e-3 (`training.py:205`) | MATCH |
| LR schedule | **ReduceLROnPlateau ×0.8 patience 5, min_lr 1e-7** | **none** (`training.py`) | **MISMATCH — gap driver** |
| Epochs / early stop | **max 10000, EarlyStopping patience 60**, best weights | **fixed epoch budget, no early stop, no best-val ckpt** | **MISMATCH — biggest gap driver** (under-trains LSTM) |
| Batch | 400 | 256 | MISMATCH (minor) |
| Augmentation | none | none | MATCH |
| Split | repo 50/50 random | 80/10/10 stratified seed 42 | MISMATCH (helps us — more train data) |
| Params | ~289k | ~289k (docstring), matches | MATCH |

Verdict: architecture close but the head lost its two dropout layers and its second dense layer;
the biggest levers for 60.08→61 are the **epoch/early-stopping budget** and the **LR schedule**, then
the **dropout-regularized 2-dense head**.

### B.2 CLDNN — `rfbench/models/baselines/cldnn.py` — our **58.05%** vs paper 60.56%

> **RESOLVED (2026-07).** The raw-waveform skip and 3rd LSTM below were added in the 2026-06
> paper-conformance pass; the resulting chance-collapse was root-caused to an input-scale init
> fragility (tiny ~1e-2-RMS IQ into a no-norm/no-BN deep LSTM) and fixed with per-sample
> unit-variance input normalization (`input_norm=True` default). Retrained under the final recipe:
> **58.05%** on the board. Table kept as the historical audit record.

| Aspect | Paper (West & O'Shea 2017) | Our code | Verdict |
|---|---|---|---|
| Conv stack | Conv(50,1×8) + 3 more Conv(50,1×8) layers | 3× Conv1d(64, k7) — `cldnn.py:99-105` | MISMATCH (filters 64 vs 50, kernel 7 vs 8, 3 vs 4 layers) |
| **Raw-waveform bypass/skip** | forward bypass **concatenating raw waveform with conv output** | **ABSENT** — no skip concat `cldnn.py:121-129` | **MISMATCH — gap driver** (costs ~1–2 pts) |
| LSTM | **three LSTM layers** | **2-layer** `nn.LSTM` — `cldnn.py:108-113` | **MISMATCH — gap driver** |
| Dropout | 50% dropout | **none** in head (`fc_embed` = Linear+SELU) `cldnn.py:115-118` | MISMATCH |
| Preprocessing | raw IQ `2×128` | raw IQ `2×128` | MATCH |
| Optimizer / LR | Adam default | Adam lr=1e-3 | MATCH |
| LR schedule | long convergence regime | none | MISMATCH — gap driver |
| Epochs / early stop | long training | fixed epochs, no early stop | MISMATCH — gap driver |
| Augmentation | none | none | MATCH |
| Split | paper split | 80/10/10 seed 42 | MISMATCH (helps us) |

Verdict: this is a **lighter re-implementation** than the paper's CLDNN. The two load-bearing paper
features — the **raw-waveform skip concatenation** and the **3 stacked LSTMs** — are both dropped; the
1.8-pt gap (largest of the three) is consistent with missing both, on top of the recipe gap.

### B.3 ResNet-AMC — `rfbench/models/baselines/resnet_amc.py` — our **56.61%** vs paper 57.32%

> **RESOLVED (2026-06).** Unit-variance input normalization, AlphaDropout + 2-dense SELU head, and
> the stack count (now **3** — the len-128 adaptation, not the paper's 6) were fixed; retrained
> 56.06 → **56.61** (−0.7 pt vs the community 2016.10a number). Historical audit record below.

| Aspect | Paper (O'Shea 2018) | Our code | Verdict |
|---|---|---|---|
| Residual stacks | **L=6** stacks (2018.01a); for 2016.10a fewer pooling stages | **4 stacks** (`DEFAULT_NUM_STACKS=4`) `resnet_amc.py:50,164-167` | **MISMATCH — gap driver** (128→8 after 4 pools; 6 would over-pool len-128, so this is a defensible adaptation but not the paper's L) |
| Filters | 32 | 32 (`DEFAULT_CONV_FILTERS=32`) | MATCH |
| Residual unit | conv+2 residual units per stack, kernel 3 | 1×1 mix conv + 2 ResidualUnit (2 conv-BN each) + maxpool `resnet_amc.py:94-119` | MATCH |
| Normalization (layer) | BatchNorm on conv | BatchNorm on conv + residual units | MATCH |
| FC head | **SELU Dense(128)→Dense(128)→softmax**, **AlphaDropout** | Linear(→128)+SELU (**one** dense), **NO AlphaDropout**, standard classifier `resnet_amc.py:176-180` | **MISMATCH — gap driver** (missing AlphaDropout + second dense) |
| **Input normalization** | **unit-variance input normalization** | **NONE** — raw IQ fed to `_iq_to_tensor` unmodified `resnet_amc.py:195-209` | **MISMATCH — gap driver** (systematic offset) |
| Optimizer / LR | Adam | Adam lr=1e-3 | MATCH |
| LR schedule | community: ~50–100 ep, early stop | none | MISMATCH — gap driver |
| Batch | community 256–1024 | 256 | MATCH (approx) |
| Augmentation | none | none | MATCH |
| Split | ~1M examples (2018.01a) | 2016.10a 80/10/10 seed 42 | context (different dataset scale) |

Verdict: three real gap drivers — **4 vs 6 residual stacks**, **no AlphaDropout + single dense head**,
and critically **no unit-variance input normalization** (the paper's explicit preprocessing; its absence
gives a systematic offset). BatchNorm partly compensates but not for the input scale.

### B.4 WiSig-CNN — `rfbench/models/baselines/sei_cnn.py` — no board score (fabricated 0.9412 removed) vs paper ~53–99%

> **BOARD CLEANED (2026-06).** The 0.9412 placeholder row (synthetic fixture split, unearned
> `verified` badge) was removed from the board in commit `a689e86`; the architecture mismatches
> below still stand — `wisig_cnn` remains a compact 1-D CNN, not the paper's 2-D CNN.

| Aspect | Paper (Hanna 2022, `d006_ManyTx_ntx.py`) | Our code | Verdict |
|---|---|---|---|
| Architecture | **2-D CNN** over `(256,2,1)`: 5 conv blocks 8/16/16/32/16, (3,2)/(3,1) kernels, (2,1)/(2,2) pools | **1-D CNN** over `(2,256)`: 3 conv blocks 32/64/128, k7, stride-2 pool + adaptive avg pool `sei_cnn.py:52,120-130` | **MISMATCH — architecture is different** (1-D compact vs paper 2-D) |
| Head | Dense(100)→Dense(80)→Dropout(0.5)→Dense(N) | Linear(128)+ReLU→Linear(N), **no dropout** `sei_cnn.py:126-130` | MISMATCH |
| Regularization | **L2 λ=1e-4** on all conv+dense | none | MISMATCH |
| Input | first 256 IQ samples, **unit-average-power normalized**, non-equalized | `(256,2)` non-eq (loader `equalized=0` ✓), **NO unit-power normalization** `sei.py:345-380` | MISMATCH (layout ✓, normalization ✗ — gap driver) |
| Optimizer / LR | **Adam lr=5e-4** | Adam lr=1e-3 (`training.py:205`) | MISMATCH |
| Early stop | **100 ep, patience 5 on val_loss**, best weights | fixed epochs, no early stop | MISMATCH |
| Class weighting | **class_weight** for imbalance | none | MISMATCH (matters in low-data ManyTx) |
| Batch | 32 (Keras default) | 256 (`training.py`) | MISMATCH |
| Metric | **balanced** test accuracy | plain `rank1_accuracy` | MISMATCH — cannot compare directly |
| Split | 80/10/10 | 80/10/10 stratified-by-tx seed 42 | MATCH (ratios) |
| Score provenance | ~53% (150 tx) / ~80% (10 tx) / >99% same-rx same-day | **0.9412 on a synthetic-fixture split**, badged `verified` | **MISMATCH — the 0.9412 is a placeholder, not a real WiSig run** |

Verdict: our `wisig_cnn` is a compact 1-D CNN and will **not** reproduce the paper's 2-D CNN numbers.
To match WiSig, add a `wisig_cnn_paper` variant (exact 2-D stack, lr=5e-4, batch 32, class_weight,
balanced accuracy, unit-power norm) and the **train-on-one-rx** protocol for the 99%→<33% cross-rx
headline. The current 0.9412 result carries an unearned `verified` badge.

There is also **no `oracle_cnn`** (paper 2-conv+2-FC, 2×128, lr=1e-4) and **no LoRa model** at all
(paper needs a spectrogram front-end + CFO compensation). `wisig_cnn`'s length-agnostic pooling lets it
run on ORACLE's 128-window, but it is the wrong architecture for both.

### B.5 LWM-Spectro — `rfbench/models/foundation/lwm_spectro.py` — integration VERIFIED; no AMC board row

Updated 2026-07 after a full ground-truth pass against the real weights + an on-cluster reproduction
of the paper's own task. The earlier "our linear_probe 22.74%" figure was produced by a **broken
encoder that loaded ZERO pretrained weights** (a fatal key mismatch) and has been **removed from the
board**. The corrected integration is now verified.

| Aspect | Paper (Kim 2026, HF `wi-lab/lwm-spectro`) | Our code | Verdict |
|---|---|---|---|
| Encoder | 12-layer Transformer d=128 h=8, custom `LayerNormalization` (`alpha/bias`), internal-residual MHA, ReLU FFN | reconstructed to load the real weights **bit-exact** (`missing=0`) | **MATCH — verified** |
| Weights source | — | `experts/{WiFi,LTE,5G}_expert.pth` (real 12-layer encoders, `module.`-prefixed). **NOT** `checkpoints/checkpoint.pth`, which is the `snr_mobility` MoE bundle (router+classifier, no encoder) | fixed after inspecting the real tensors |
| Token width | 4×4 patch of a **single-channel** 128×128 spectrogram → **16** | `ELEMENT_LENGTH=16` (proven by `embedding.proj`=`Linear(16,128)`, `decoder_bias`=`(16,)`) | **MATCH — verified** (earlier 32/interleaved was wrong) |
| Representation | mean-pool over sequence; `[CLS]`=0.2; per-sample `(x−mean)/std`; log-magnitude (dB) | mean-pool, `[CLS]`=0.2, joint per-sample norm, log-magnitude | **MATCH (token layout)** |
| Input / STFT | 128×128 spectrogram, `\|Y\|²` log-scaled — **hop/window/FFT unpublished; upstream ships NO IQ→spectrogram code** | best-effort `n_fft=512` STFT front-end | **UNVERIFIABLE — gap driver** (cannot be reproduced from public artifacts) |
| Eval dataset | DeepMIMO synthetic spectrograms | **RadioML 2016.10a IQ** (a hub choice) | **OOD by construction; paper reports no RadioML number** |

**Reproduction of the paper's own task (from the shipped ckpt + `demo_data_moe.pt`).** The released
MoE checkpoint targets **joint SNR/mobility recognition** (`task='snr_mobility'`, 14 classes = 7 SNR ×
{pedestrian, vehicular}). `demo_data_moe.pt` ships 10 500 **labelled** pre-computed 128×128
spectrograms + reference embeddings. Reconstructing their exact classifier head (`Res1DCNNHead` +
`LayerNorm`, loaded `missing/unexpected=[]`) and running it on the shipped `moe_embedding` reproduces
**93.9% accuracy** (pedestrian 96.6 / vehicular 91.2), matching the paper's Table II (94.4% @100-shot,
95.1% @400-shot); a logreg cross-check on the shipped embedding gives 92.6%. Our reconstructed encoder
fed the shipped spectrograms yields a **0.57 cosine** to the reference `moe_embedding` (vs ~0 random) —
substantively correct, not yet bit-exact (the precise embedding-extraction/MoE-combine recipe is the
remaining gap).

Verdict: **the integration and the model are VERIFIED** (real weights load bit-exact; the paper's own
snr_mobility task reproduces at 93.9% ≈ the paper). There is **no LWM-Spectro AMC/RadioML board row**,
and inventing one would be dishonest: (1) the paper defines **no** RadioML/AMC task, and (2) the
IQ→spectrogram preprocessing is **unpublished**, so any RadioML number is off-distribution (our
corrected linear_probe/few_shot land at ~chance, 16.6%/14.1%, and are **not** published). LWM-Spectro's
tasks are DeepMIMO-synthetic and fall **outside the hub's terrestrial-real-signal board scope**.

### Audit summary — status after the 2026-06 fixes

1. **Training recipe — FIXED.** `rfbench/training.py` now selects/restores the best checkpoint on
   **val accuracy**, runs ReduceLROnPlateau + early stopping (patience 40) + gradient clipping 5.0 +
   a NaN guard over 150 epochs. Result: MCLDNN 60.08 → **61.71** (above paper), ResNet 56.06 →
   **56.61**.
2. **Architecture fidelity — FIXED** (paper-conformance pass): CLDNN got its **raw-waveform skip +
   3rd LSTM**, ResNet its **unit-variance input norm + AlphaDropout + 2-dense head** (3 stacks as the
   len-128 adaptation), MCLDNN its **concat fusion (100 filters) + dropout-regularized 2-dense head**.
   The paper-exact CLDNN's chance-collapse was root-caused and fixed (2026-07, per-sample input
   normalization); board score **58.05%**.
3. **LWM-Spectro — VERIFIED, no board row** (2026-07): the encoder now loads the real expert weights
   **bit-exact** (`missing=0`) and the paper's own `snr_mobility` task reproduces at **93.9%** from the
   shipped ckpt + demo data. The broken 22.74% row (which had loaded **zero** weights) was **removed**.
   No AMC/RadioML row is reported — the paper defines no such task and the IQ→spectrogram preprocessing
   is unpublished (RadioML would be OOD; corrected probe lands at ~chance). See B.5.
4. **SEI fabrications — REMOVED from the board** (`a689e86`): the 0.9412 fixture-split row and the
   0.7734 iqfm row are gone. **STILL OPEN**: `wisig_cnn` remains the wrong architecture (1-D vs paper
   2-D) with the wrong lr/batch and no unit-power norm — a real WiSig run needs `wisig_cnn_paper`
   (C.2).

---

## Part C — MISSING FOR THE BENCHMARK

Papers/models/datasets to add (deduped).

### C.1 AMC baselines & datasets
- **PET-CGDNN** (Zhang 2021, arXiv:2110.04980, repo Richardzhangxx/PET-CGDNN) — param-efficient CNN+GRU
  with a phase-transform front-end, ~60% overall on 2016.10a at ~71–75k params (1/3 of MCLDNN). Cheap,
  strong baseline; the phase-transformation layer is the load-bearing novelty.
- **TLDNN** (Qu 2024, arXiv:2401.01056) — SOTA on 2016.10a (62.83%) and the source of our Table II
  numbers. Reproduce as the "best specialized baseline" target and **adopt its clean AdamW /
  ReduceLROnPlateau / 6:2:2 / A-P-transform recipe board-wide** (would also fix Part B item 1).
- **TCN-GRU** (Sensors 2024, DOI 10.3390/s24247908) — 61.56% overall at 253k params, beats MCLDNN;
  second independent source for CNN2/CLDNN/ResNet/GRU2 baseline numbers.
- **VT-CNN2 / CNN2** (O'Shea 2016, arXiv:1602.04105) — the original 2-layer CNN (~56–57%), the historical
  floor every AMC paper cites; currently absent.
- **LSTM2** (Rajendran 2018) — strong non-CNN baseline, 61.02% overall (TLDNN T2).
- **MCformer, LSTM-DAE** — transformer + denoising-AE baselines (60.54% / 61.42%, TLDNN T2) for a fuller
  mid-tier board.
- **RadioML 2018.01a full-range targets** (ResNet 60.91 / MCLDNN 61.92 / TLDNN 63.32) — usable once
  unblocked; keep peak ~95% vs full-range ~61% distinct on the board.

### C.2 SEI baselines, protocols & fixes
- **`wisig_cnn_paper`** — the exact paper 2-D CNN over `(256,2,1)` (5 conv blocks 8/16/16/32/16, (3,2)/
  (3,1) kernels, (2,1)/(2,2) pools, Dense 100/80/N, Dropout 0.5, L2 1e-4, Adam lr=5e-4, batch 32, 100 ep
  + patience-5 early stop, **unit-power normalization**). Our compact 1-D `wisig_cnn` will not reproduce
  the paper.
- **WiSig cross-receiver train-on-one-rx protocol** — the 99%→<33% headline needs train-on-single-rx /
  test-on-unseen-rx, distinct from our grouped 80/10/10 `cross_receiver`. Add as a reference protocol.
- **WiSig balanced-accuracy metric** — the paper reports class-balanced test accuracy (equal per-class
  resampling); add balanced rank-1 alongside `rank1_accuracy` for parity.
- **`oracle_cnn`** — no ORACLE-specific model exists (only `wisig_cnn`). Add 2 conv (50@1×7, 50@2×7) +
  FC(256) + FC(80) + softmax, input `2×128` raw IQ, Adam lr=1e-4, dropout 0.5, L2 1e-4, patience-10.
- **ORACLE cross-location track** — the 98.6%→87.13% is a different-location split (ORACLE has one rx, so
  our cross_receiver/cross_day don't apply). Add as an ORACLE reference track.
- **`lora_spectrogram_cnn`** — no LoRa model exists; the paper's best is a spectrogram (102×63) CNN (3
  conv 8/16/32 3×3 + BN + 1 FC), NOT raw-IQ. Add with an STFT (win 256/hop 128) front-end.
- **CFO-compensation preprocessing for LoRa** — required to move the spectrogram-CNN ~83.5%→~95%+.
  Absent from the SEI pipeline; without it any LoRa reproduction caps near 83%.
- **LoRa dataset provenance fix** — our loader reads `dataset_training_aug.h5` (30-device, ResNet-aug 2022
  release), NOT the JSAC-2021 spectrogram-CNN data. Either (a) cite **Shen et al. 2022, "Towards Scalable
  and Channel-Robust RFFI for LoRa," IEEE TIFS 2022** (30 dev, ResNet + aug, IEEE DataPort DOI
  10.21227/qqt4-kz19) as the matched baseline, or (b) obtain the JSAC-2021 10-device data to match 96.40%.

### C.3 Detection & sensing
- **DeepSweep** (arXiv:2401.04805, Uvaydov et al.) — parallel/scalable CNN spectrum sensing, direct
  successor to DeepSense with sub-1ms latency and a VGG16 baseline; strong second sensing baseline.
- **CSRD2025** (arXiv:2508.19552) — large-scale synthetic radio dataset for spectrum sensing; newer
  complement to WBSig53, worth screening as an FM fine-tuning target.
- **ParallelCNN** (Mei et al.) — dual-stream raw-IQ sensing derived from DeepSense; common DeepSense-family
  comparison baseline.
- **TorchSig NarrowBandSig53** (West & O'Shea) — companion recognition dataset to WBSig53; add for the
  53-class modulation-recognition track if the hub wants full-taxonomy classification.
- **RadDet baseline provenance** — confirm whether RT-DETR/YOLOv9 were trained from scratch or fine-tuned
  from COCO/ImageNet weights (not stated in the fetched text `(?)`); needed to reproduce RadDet exactly.
- **A third real-capture detection set** — RadDet + WBSig53 cover real + synthetic; a second real-capture
  detection set would strengthen generalization claims.

### C.4 Source separation (new candidate track — from RFSS, A.6)
- **`source_separation` task** — blind multi-source RF separation is absent from the hub taxonomy
  entirely. RFSS (arXiv:2604.00398) supplies exactly what a board track needs: a public static
  dataset (103 GB HDF5, official 70/15/15 index split — adopt as-is per split policy), a defined
  metric (**co-channel PI-SI-SINR**; avoid the adjacent-channel floor artifact), and 5 reproducible
  baselines with released checkpoints/eval code. **Blocked: NOT released as of 2026-07-03** (HF release announced, nothing published). Synthetic-but-downloadable → allowed (same category as `interf_gnss6`).
- **Conv-TasNet / DPRNN as baselines** — audio architectures transfer to RF unmodified (−12.34 /
  −12.51 dB co-channel 2-source); the RFSS training recipe (PIT + SI-SINR loss, Adam 1e-3, grad clip
  1.0, batch 8, cosine annealing 30 ep, crop 7,680 samples, seed 42) is fully specified.
- **`rfss_single.h5`** (4k single-source GSM/UMTS/LTE/NR) — candidate **second `protocol_tech_id`
  dataset** (cellular standards, complements our WiFi-standards T-PRIME track).
- **RF Challenge (arXiv:2409.08839)** — real-OTA interference *cancellation* (BER metric); adjacent
  to `interference_id`/`source_separation` but a distinct task. Add to the bibliography as context;
  screen its OTA recordings as a possible real-capture complement to RFSS's synthetic corpus.

### C.5 Foundation models
- **WirelessJEPA** (arXiv:2601.20190) — ADD as the top FM row: 74.78% linear-probe on RML2016.10a full-SNR
  11-class is the ONLY public FM number directly comparable to our board, and it beats supervised MCLDNN
  (60.08). Weights unreleased → retrain the JEPA recipe (ShuffleNetV2-x0.5, EMA teacher).
  **DONE (2026-07): wrapper `wireless-jepa` implemented** (shares IQFM's `shufflenet1d.py` backbone);
  JEPA retrain scripts landed (`scripts/pretrain/wireless_jepa.py`). Our in-repo retrain is
  in-distribution, NOT the paper's OOD — the 74.78% is never claimed as ours.
- **IQFM** (arXiv:2506.06718v2) — REPLACE the fabricated SEI/WiSig rank1=0.7734 row with the real number:
  RML2016.10a **38.1% @ 50 samples/class linear probe (OOD)**. Arch = ShuffleNetV2 0.5×, ~341k params,
  unit-max norm, SimCLR.
- **RIS-MAE** (arXiv:2508.00274) — ADD for the 2018.01a AMC track once unblocked: 48.41% OA / κ 0.4616 at
  1% labels, full SNR, 24-class (beats MCLDNN 31.92 in that regime).
- **LWM-Spectro self-protocol note** — the paper's tasks are on **DeepMIMO synthetic** spectrograms
  (5-class modulation few-shot macro-F1; `snr_mobility`; multi-protocol), NOT RadioML. We therefore
  report **no** RadioML row. We DID reproduce the paper's own `snr_mobility` task at **93.9%** from the
  shipped ckpt + `demo_data_moe.pt` (their exact `Res1DCNNHead`, `missing/unexpected=[]`). A RadioML
  number can't be made faithful (the IQ→spectrogram STFT is unpublished). Record "no published
  LWM-Spectro RadioML number exists; own snr_mobility task reproduced 93.9%."
- **LatentWave** (arXiv:2606.06373) — ADD as a JEPA-on-spectrograms FM: linear-probe 80.9% on CommRad (not
  RadioML). Useful spectrogram-track FM baseline (ViT 8L/256d/6.4M, EMA JEPA).
- **6G-MSM** (arXiv:2411.09996) — ADD as the reference MSM-ViT recipe for a spectrogram/segmentation track
  (ViT-S/M/L, mask 70–80%); no AMC, but the pretraining template for a sensing/detection FM. Code pending.
- **Fix `xcit-nano-sig53-linear_probe.json`** — the stored 0.7116 is **XCiT-Tiny12 online-impaired Sig53
  (supervised)**, not XCiT-Nano clean linear_probe; and the model name is "iqfm-base" (wrong). Nano static
  impaired = 67.97%. TorchSig ships NO pretrained weights, and Sig53 is excluded by our generation policy —
  this row is context, not a leaderboard entry.
- **LWM base channel model** (arXiv:2411.08872) — note in the bibliography as LWM-Spectro's lineage but
  **exclude from AMC/SEI** (CSI-only tasks).
- **License flag** — LWM-Spectro's only upstream license signal is **MIT** (`pyproject.toml`
  `license = {text = "MIT"}` + OSI classifier, and README_model.md "License: MIT"), but **no standalone
  LICENSE file ships** in the repo and the README frontmatter has `#license: mit` commented out
  (config.json has no license field; `/raw/main/LICENSE` 404s). Effective status: **MIT declared, no
  LICENSE file present** — permissive (commercial use allowed, attribution via the paper citation). Verify
  and record in the model registry before publishing a public leaderboard row (consistent with
  `docs/SOTA_REFERENCE.md` "license unstated → verify").
- **WavesFM** — keep paper-only: no public weights/code URL found; its 86.05% is on a private 20-class
  spectrogram set, not RadioML. Do not list weights as obtainable until a repo is confirmed.

---

## Sources

- MCLDNN: Xu et al., *IEEE WCL* 9(10):1629–1632, 2020, DOI 10.1109/LWC.2020.2999453 — repo
  github.com/wzjialang/MCLDNN.
- CLDNN: West & O'Shea, *DySPAN 2017*, arXiv:1703.09197, DOI 10.1109/DySPAN.2017.7920754.
- ResNet: O'Shea, Roy, Clancy, *IEEE JSTSP* 12(1):168–179, 2018, arXiv:1712.04578, DOI
  10.1109/JSTSP.2018.2797022.
- PET-CGDNN: Zhang et al., 2021, arXiv:2110.04980 — repo github.com/Richardzhangxx/PET-CGDNN.
- TLDNN (Table II baseline source): Qu et al., *IEEE TVT* 2024, arXiv:2401.01056.
- TCN-GRU (Table 3 baseline source): *Sensors* 24(24):7908, 2024, DOI 10.3390/s24247908.
- VT-CNN2/CNN2: O'Shea et al., 2016, arXiv:1602.04105.
- WiSig: Hanna et al., *IEEE Access* 10:22808–22818, 2022, DOI 10.1109/ACCESS.2022.3154488,
  arXiv:2112.15363 — repo github.com/WiSig-dataset/wisig-examples.
- ORACLE: Sankhe et al., *IEEE INFOCOM 2019*, arXiv:1812.01124, DOI 10.1109/INFOCOM.2019.8737463.
- LoRa RFFI (JSAC'21): Shen et al., *IEEE JSAC* 39(8):2604–2616, 2021, DOI 10.1109/JSAC.2021.3087250.
- LoRa RFFI (2022, our loader's actual source): Shen et al., "Towards Scalable and Channel-Robust RFFI
  for LoRa," *IEEE TIFS* 2022, IEEE DataPort DOI 10.21227/qqt4-kz19 — repo github.com/gxhen/LoRa_RFFI.
- RadDet: Huang et al., *ICASSP 2025*, arXiv:2501.10407 — repo github.com/abcxyzi/RadDet.
- WBSig53: Boegner et al., 2022, arXiv:2211.10335 — TorchSig github.com/TorchDSP/torchsig.
- DeepSense: Uvaydov et al., *IEEE INFOCOM 2021*, DOI 10.1109/INFOCOM42981.2021.9488764 — repo
  github.com/wineslab/deepsense-spectrum-sensing-datasets.
- DeepSweep: Uvaydov et al., arXiv:2401.04805.
- Sig53/TorchSig XCiT: Boegner et al., 2022, arXiv:2207.09918 — repo github.com/TorchDSP/torchsig.
- LWM-Spectro: Kim, Alikhani, Alkhateeb, 2026, arXiv:2601.08780 — HF wi-lab/lwm-spectro (MIT declared in
  pyproject.toml/README_model.md; no LICENSE file in repo).
- LWM (base): Alikhani, Charan, Alkhateeb, 2024, arXiv:2411.08872.
- WavesFM: 2025, arXiv:2504.14100 (no public weights `(?)`).
- IQFM: Mashaal, Abou-Zeid, 2025, arXiv:2506.06718v2.
- WirelessJEPA: 2026, arXiv:2601.20190.
- RIS-MAE: Liu, Liu et al., 2025, arXiv:2508.00274.
- LatentWave: 2026, arXiv:2606.06373.
- 6G-MSM: Aboulfotouh, Eshaghbeigi, Abou-Zeid, 2024, arXiv:2411.09996.
- RFSS: Chen, Jin, Tan, 2026, arXiv:2604.00398 (v2; supersedes arXiv:2508.12106 — cite the 2026
  version). Data/code: HF release announced, **not published as of 2026-07-03**.
- RF Challenge: Lancho, Weiss, Lee, Jayashankar, Kurien, Polyanskiy, Wornell, *IEEE OJ-COMS* 2025,
  arXiv:2409.08839 (ICASSP 2024 challenge).
- Conv-TasNet: Luo & Mesgarani, *IEEE/ACM TASLP* 27(8):1256–1266, 2019.
- DPRNN: Luo, Chen, Yoshioka, *ICASSP 2020*.
- SI-SNR / SI-SINR: Le Roux, Wisdom, Erdogan, Hershey, "SDR — half-baked or well done?," *ICASSP 2019*.
- RF Transformer (separation): 2026, arXiv:2603.09201 `(?)` (not yet screened).
