# SOTA reference — models, datasets, papers, published scores

Living reference for RF-Benchmark-Hub: what we have (datasets, models, results) and the numbers
the **reference papers** announce, so our reproductions can be compared apples-to-apples.

> **Metric caveat (read this first).** Papers on AMC usually report **peak / high-SNR accuracy**
> (e.g. accuracy at +18 dB, or averaged over SNR ≥ 0 dB). Our board's **primary metric is
> `accuracy_overall` over the FULL SNR range (−20…+18 dB), no cherry-picking** — a much lower
> number. We therefore track BOTH: our `accuracy_overall` *and* our high-SNR point (from the
> `accuracy_vs_snr` curve) so we can line up against the paper's reported figure.
> `?` = number not yet confirmed against the primary source; treat as indicative only.

---

## 1. Inventory — what we have

### Baselines (specialized, we train them)
| Model | Registry name | Impl | Trained | Tasks |
|---|---|---|---|---|
| **MCLDNN** (Xu et al. 2020) | `mcldnn` | ✅ | ✅ RadioML 2016.10a (self_reported) | AMC |
| **ResNet-AMC** (O'Shea et al. 2018) | `resnet_amc` | ✅ | ⏳ seed run in flight | AMC |
| **CLDNN** | `cldnn` | ✅ | ⏳ seed run in flight | AMC |

### Foundation models (public weights — we'd eval, not train from scratch)
| Model | Weights | Input | Embed | Tasks | Notes |
|---|---|---|---|---|---|
| **LWM-Spectro** (wi-lab) | ✅ HF `wi-lab/lwm-spectro`, no login | IQ→STFT 128×128 | 128 | AMC (demo'd), SEI? | Transformer MoE; license unstated → verify |
| **TorchSig XCiT** (Boegner et al.) | ✅ TorchSig repo script | raw IQ 2×4096 | ~192–384 | AMC (53 mod), SEI? | native raw-IQ; also a strong baseline |
| **TorchSig YOLO** | ✅ same | spectrogram | — | detection | only public detection checkpoint found |
| **WavesFM** | ⚠️ URL unconfirmed | raw IQ (2,C,T) | 256 | AMC+SEI+detect | best coverage *if* weights obtainable |
| **IQFM** (Mashaal, Abou-Zeid) | ✗ paper-only (CC-BY 4.0) | raw IQ (2,L), unit-max | 1024 | AMC | wrapper `iqfm-base` implemented; ShuffleNetV2-x0.5 1-D (335k params), SimCLR; retrain in-repo (NOT the paper's OOD 38.1%) |

Code/paper-only (no public weights): RIS-MAE, LatentWave, 6G-MSM, WirelessJEPA, most SSL-SEI.
IQFM (above): weights unpublished, but its recipe is now a board wrapper — we pre-train our own
ShuffleNetV2-x0.5 backbone with SimCLR on RadioML-train (`scripts/pretrain/iqfm_simclr.py`).

### Datasets (real, on the cluster)
| Dataset | Task | Status | Split indices |
|---|---|---|---|
| RadioML 2016.10a | AMC | ✅ downloaded (Zenodo mirror) + prepared | `amc-radioml2016-strat-snr-8010-seed42-v1` |
| WiSig ManyTx | SEI | ✅ downloaded (GDrive) + prepared | closed_set / cross_receiver / cross_day |
| RadioML 2018.01a | AMC | ⛔ blocked (Kaggle-gated / DeepSig cert expired) | — |
| ORACLE | SEI | ⏳ downloadable (Northeastern, valid cert), not fetched | — |
| RadDet | detection | 🔑 gated (Kaggle token) | — |
| LoRa RFFI | SEI | 🔑 gated (IEEE DataPort login) | — |
| Sig53 / WBSig53 | AMC / detection | ⛔ excluded (generation-only, per "no generation" policy) | — |

---

## 2. AMC — automatic modulation classification

**Primary metric:** `accuracy_overall` (full SNR range). Curve: `accuracy_vs_snr`. Also `macro_f1`.

### RadioML 2016.10a (11 classes, −20…+18 dB, 220k samples)
Published **overall** accuracy (avg over the full SNR range — the SAME metric as our board):

| Model | Paper | Published `accuracy_overall` | Our `accuracy_overall` | Our high-SNR pt |
|---|---|---|---|---|
| **MCLDNN** | Xu et al., *IEEE WCL* 2020 | **61.01%** (orig.) / 61.52% (repro in [2401.01056]) | **0.558** (15 ep) · **100-ep seed in flight** | 0.852 @ +6 dB |
| **ResNet-AMC** | O'Shea et al., *IEEE JSTSP* 2018 | **57.32%** (Table II, [2401.01056]) | ⏳ seed in flight | ⏳ |
| **CLDNN** | West & O'Shea 2017 | ~59% `?` | ⏳ seed in flight | ⏳ |
| *context — best on 2016.10a* | — | **TLDNN 62.83%**, LSTM-DAE 61.42%, MCformer 60.54%, LSTM2 61.02% | — | — |

> **Reading it:** the classic ~2016.10a ceiling is **~61–63% overall** (11 classes, full SNR range);
> MCLDNN sits at **61.0%**, ResNet at **57.3%**, TLDNN (2024, ≈ current SOTA) at **62.8%**. So the
> honest "hard" reference for our board is **~61% for MCLDNN, ~57% for ResNet**.
> Our MCLDNN 15-epoch validation run already reached **0.558 overall** (0.852 @ +6 dB, ≈chance 9% at
> ≤−16 dB — textbook S-curve); the **100-epoch seed run should land near the 0.61 paper figure**.

### RadioML 2018.01a (24 classes) — *dataset blocked, not yet prepared*
| Model | Paper | Paper score `?` |
|---|---|---|
| ResNet (deep) | O'Shea et al., *IEEE JSTSP* 2018 | high-SNR ~95% (24-class) `?` |

### Sig53 (53 classes) — *excluded (generation-only)*
| Model | Paper | Paper score |
|---|---|---|
| XCiT-Nano | Boegner et al. (TorchSig), arXiv:2207.09918 | **71.16%** (as cited in our plan) |

---

## 3. SEI — specific emitter identification / RF fingerprinting

**Primary metric:** `rank1_accuracy` (closed-set). Open-set: `auroc`, `eer`. Conditions reported
separately: `closed_set` / `cross_receiver` / `cross_day`.

### WiSig ManyTx (Hanna et al.) — *prepared, baseline pending*
| Condition | Paper (Hanna et al., *IEEE Access* 2022, DOI 10.1109/ACCESS.2022.3154790) | Ours |
|---|---|---|
| closed_set | high rank-1 accuracy `?` | ⏳ |
| cross_receiver | **large drop** vs closed-set (the paper's headline) `?` | ⏳ |
| cross_day | drop vs closed-set `?` | ⏳ |

### ORACLE (Sankhe et al., 2019) — *downloadable, not fetched*
| Condition | Paper | Ours |
|---|---|---|
| closed_set (same-day, 16 tx) | ~**99%** `?` | — |

---

## 4. Wideband detection

**Primary metric:** `mAP`. Also `mAR`, `IoU`. Tracks: detection vs recognition.

### RadDet (ICASSP 2025) — *real dataset adopted; gated download (Kaggle)*
| Model | Paper (arXiv:2501.10407) | Ours |
|---|---|---|
| YOLO / detector | mAP as reported `?` | — |

### WBSig53 — *excluded (generation-only)*
| Model | Paper (TorchSig, arXiv:2211.10335) | Score (our plan §8) |
|---|---|---|
| DETR-B4 | — | mAP ~86 `?` |
| YOLOv5-pico | — | mAP ~73 `?` |

---

## 5. Spectrum sensing (Wave B — not started)

**Primary metric:** `pd@pfa=0.1`. Also inference latency.

### DeepSense (OTA 802.11 a/g + LTE-M)
| Model | Paper | Score (our plan §8) |
|---|---|---|
| DeepSense CNN | wineslab | ~98% / 97% `?` |

---

## 6. How our numbers get produced & verified

- **self_reported**: `rfbench train … --regime from_scratch` (baselines) or `rfbench eval` (FMs)
  emits a schema-valid `result.json` under `leaderboard/results/<task>/`.
- **verified**: the maintainer re-runs on the multi-GPU station and flips `verification.status`
  within `tolerance` (see `docs/SUBMISSION.md`). Our current MCLDNN row is `self_reported`.
- The board ranks by the **primary metric only**, per (task, track, regime) — never mixing regimes
  or tracks in one column.

## Sources
- MCLDNN: Xu, Luo, et al., "A Spatiotemporal Multi-Channel Learning Framework for AMC," IEEE WCL 2020.
- O'Shea et al., "Over-the-Air Deep Learning Based Radio Signal Classification," IEEE JSTSP 2018.
- Sig53 / TorchSig: Boegner et al., arXiv:2207.09918 · WBSig53: arXiv:2211.10335.
- WiSig: Hanna, Karunaratne, Cabric, IEEE Access 2022, DOI 10.1109/ACCESS.2022.3154790.
- ORACLE: Sankhe et al., "ORACLE: Optimized Radio clAssification through Convolutional neuraL nEtworks," 2019.
- RadDet: ICASSP 2025, arXiv:2501.10407 · FM survey: see this session's FM-weights research report.
- Comparison table for 2016.10a overall accuracy (ResNet 57.32%, MCLDNN 61.52%, LSTM2 61.02%,
  LSTM-DAE 61.42%, MCformer 60.54%, TLDNN 62.83%): "Enhancing AMR through Robust Global Feature
  Extraction," arXiv:2401.01056, Table II. MCLDNN original 61.01%: Xu et al. IEEE WCL 2020.

> **TODO (confirm the `?` cells):** pull the exact reported figures from each primary paper before
> publishing comparative claims. This file is the working scoreboard; numbers marked `?` are
> indicative from memory/secondary sources and must be verified against the source PDF.
