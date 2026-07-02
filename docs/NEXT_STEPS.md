# Next steps & workflows

État (2026-07-02) : framework complet (M0/M1/M2/M4/M5/M6 code), 2 datasets réels préparés
(RadioML 2016.10a, WiSig), **1re colonne AMC réelle** : MCLDNN 60.08% + CLDNN 58.76% (vs papier
61.01% / ~59%), site généré. Réf. publiées dans `docs/SOTA_REFERENCE.md`.

Légende **Workflow ?** : `solo` = 1 agent / main-loop ; `WF` = workflow multi-agents (fan-out + revue) ;
`cluster` = job(s) SLURM sur GB200 ; `🔑` = bloqué sur credentials/décision.

---

## P0 — Consolider la colonne AMC (rapide)
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 1 | **Débugger ResNet-AMC** (reste à 9.09% = hasard) | solo + cluster | Lire `rfbench/models/baselines/resnet_amc.py` : suspects = BN/résiduel mal câblé, ReLU morts, tête déconnectée, ou LR inadapté. Fixer → re-train → viser ~57% (réf). |
| 2 | **Nettoyer le board** | solo | Retirer les `leaderboard/results/**/*.json` de démo (fake) maintenant qu'on a du réel, ou les isoler. Le board ne doit montrer que du réel. |
| 3 | **Recette d'entraînement propre** | solo + cluster | Ajouter LR-schedule (ReduceLROnPlateau) + early-stopping + best-val checkpoint dans `training.py`. Rend les scores seed reproductibles et « verified »-quality (grignote le dernier point vs papier). Puis re-run les 3 baselines. |
| 4 | **Dette perf** | solo | `mcldnn.py` `_iq_to_tensor` (eval-only) : `np.asarray` avant `torch.as_tensor` (warning). |

## P1 — Baseline SEI (WiSig déjà prêt)
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 5 | **Modèle + entraînement SEI** | **WF** | WiSig-CNN (+ ResNet1D) wrappé `Model`, registré. Training SEI diffère de l'AMC : métrique **rank1** (closed_set) + auroc/eer (open-set), 3 conditions (closed/cross-rx/cross-day), embeddings. WF = {modèle ∥ wiring training SEI} → revue → run cluster sur les 3 conditions. Montre la **chute cross-receiver/cross-day** (le point clé du papier WiSig). |

## P2 — Intégration Foundation Model (le cœur du benchmark, M6)
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 6 | **Wrapper FM + poids réels** | **WF** | Cible #1 = **LWM-Spectro** (HF `wi-lab/lwm-spectro`, poids publics, AMC démontré) ; alt = **TorchSig XCiT** (raw IQ). WF = {télécharger poids sur frontend · wrapper chargeant les vrais poids + `embed()` · adaptateur d'entrée IQ→STFT 128×128 (LWM) · intégration linear_probe/few_shot/finetune} → revue → **eval cluster** sur AMC (puis SEI). Produit les 1res lignes **FM vs baseline** du board — la thèse du hub. Vérifier la licence LWM avant de publier. |
| 7 | **Guide + 2e FM** | solo/WF | Une fois le 1er FM intégré, WavesFM (si poids obtenus) ou TorchSig, en suivant `docs/ADDING_A_MODEL.md`. |

## P3 — Data (I/O + tes décisions)
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 8 | **ORACLE** (SEI) | cluster | Cert Northeastern valide → download frontend + prepare ARM → 2e dataset SEI. |
| 9 | **RadioML 2018.01a** | 🔑 | Bloqué : Kaggle-gated (token) OU DeepSig cert expiré. Décision : token Kaggle / download insecure autorisé / dépôt manuel. |
| 10 | **RadDet (détection) + LoRa RFFI** | 🔑 | Kaggle token (RadDet) / compte IEEE (LoRa). Une fois là : tâche+baseline détection (WP-22/32, YOLO/DETR) = **WF**. |

## P4 — Vérification, Wave B, lancement
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 11 | **Démo tier `verified`** | cluster | `rfbench verify` : re-run MCLDNN sur station, flip `self_reported`→`verified` dans la tolérance. Prouve le WP-53 en vrai. |
| 12 | **Wave B — spectrum sensing** (M7) | **WF** | DeepSense : data + tâche (WP-14/23) + baseline (WP-33), copie du gabarit. |
| 13 | **Publication + v1** (WP-71) | solo | Activer `build-leaderboard.yml`→GitHub Pages, README final, CITATION.cff, tag v1.0. (nécessite un remote GitHub — repo local pour l'instant.) |

---

## Workflows à prévoir (récap)
1. **SEI baseline** (#5) — modèle WiSig-CNN + training SEI 3 conditions.
2. **FM integration** (#6) — wrapper LWM-Spectro/XCiT + adaptateur entrée + eval linear_probe/few_shot/finetune. *(le plus gros levier)*
3. **Détection** (#10, après data) — tâche+baseline RadDet.
4. **Wave B** (#12) — spectrum sensing bout-en-bout.

Ordre conseillé : **P0 (ResNet+recette) → #6 FM (thèse du hub) en //  de #5 SEI → data/#8 ORACLE → verified → Wave B → v1.**
