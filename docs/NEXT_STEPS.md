# Next steps & workflows

État (2026-07-06) : framework complet (M0/M1/M2/M4/M5/M6 code). **Colonne AMC réelle et
consolidée** sous la recette finale (val-acc checkpoint + ReduceLROnPlateau + early-stop, seed 42,
150 ep) : **MCLDNN 0.6171 · CLDNN 0.5805 · ResNet 0.5661**. **2e colonne réelle** :
`interference_id` = **interf_cnn 0.9987** (GNSS jamming 6-class, data Zenodo publique). Board honnête
(lignes fabriquées retirées, `a689e86`). **Site redessiné** (refonte mockup + nav Tasks/Guide/Submit)
et déployé sur GitHub Pages (⚠ le job *Deploy to Pages* échoue par intermittence — souci de config
Pages côté repo, pas de code). Réf. publiées dans `docs/SOTA_REFERENCE.md` / `docs/BIBLIOGRAPHY.md`.

Légende **Workflow ?** : `solo` = 1 agent / main-loop ; `WF` = workflow multi-agents (fan-out + revue) ;
`cluster` = job(s) SLURM sur GB200 ; `🔑` = bloqué sur credentials/décision.

---

## ✅ Fait depuis le 2026-07-02
- **P0 entièrement clos** : ResNet débuggé (n'était plus au hasard) → 0.5661 ; board nettoyé (fake
  retirés) ; recette d'entraînement propre (schedule + early-stop + best-val checkpoint) ; **collapse
  CLDNN root-causé et corrigé** (fragilité d'init sur IQ ~1e-2 RMS → normalisation unit-variance
  d'entrée) → 0.5805.
- **`interference_id`** : tâche + baseline `interf_cnn` + run cluster → **0.9987** au board (data
  publique, aucun blocage). Tâche promue `implemented`.
- **`protocol_tech_id`** : tâche + baseline `tprime` (transformer) codés ; dataset id corrigé
  (DS 3.0 `neu:h989s8519`, format `.bin` = complex128 natif). Bloqué au download (voir #10 ci-dessous).
- **FM LWM-Spectro (WP-62)** : wrapper rendu fidèle aux vrais poids (experts `*_expert.pth`, token
  width 16, LayerNormalization custom) — **poids chargés bit-exact** (`missing=0`). Ligne RadioML
  **retirée** (22.74% = ancien encodeur cassé chargeant 0 poids). **Tâche du papier `snr_mobility`
  reproduite à 93.9%** (leur classifieur exact sur `demo_data_moe.pt` ≈ Table II 94–95%). **Pas de
  ligne AMC** : le papier n'a aucune tâche RadioML, préproc IQ→spectrogramme non publiée (voir
  `docs/BIBLIOGRAPHY.md` B.5). Intégration **close** ; la thèse FM-vs-baseline passe par un FM à tâche
  terrestre réelle (voir #6).
- **RFSS** (arXiv:2604.00398) mined en biblio ; `source_separation` ajouté comme track candidat P3.

## En vol (sessions parallèles — ne pas toucher)
- **SEI** (`feat/sei-complete`, non mergé) : `wisig_cnn_paper` (2-D fidèle papier) + `oracle_cnn` +
  baselines SOTA + track POWDER + `training_sei.py` ; job `sei_train` en queue. Landera la colonne
  SEI (3 conditions closed/cross-rx/cross-day) dès scores obtenus.
- **FM** (`feat/fm-lwm-spectro`).

---

## P1 — Premières lignes vérifiées / FM-vs-baseline (le cœur du hub)
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 5 | **Lander la colonne SEI** | cluster | Attendre la fin du `sei_train`, merger `feat/sei-complete`, écrire les `result.json` des 3 conditions. Montre la **chute cross-receiver/cross-day** (point clé WiSig). |
| 6 | **1re ligne FM vs baseline** | **WF** | **LWM-Spectro ne peut PAS la fournir** : le papier n'a aucune tâche RadioML/terrestre et sa préproc IQ→spectrogramme est non publiée (intégration close + vérifiée, sa tâche `snr_mobility` reproduite à 93.9% — voir BIBLIOGRAPHY B.5). Utiliser un FM à **tâche RadioML réelle** : **WirelessJEPA** (74.78% linear-probe RML2016.10a, bat MCLDNN — poids à ré-entraîner) ou **IQFM** (38.1% @50/cl OOD). Head logreg sur features gelées, licence vérifiée avant publication. |
| 7 | **Démo tier `verified`** | cluster | `rfbench verify` : re-run MCLDNN depuis son checkpoint, flip `self_reported`→`verified` dans la tolérance. Prouve le WP-53 en vrai (1 job court, indépendant). |

## P2 — Compléter les colonnes existantes
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 8 | **`protocol_tech_id` : débloquer le download T-PRIME** | 🔑 | Le host DRS Northeastern sert une chaîne TLS incomplète (403/cert) → download navigateur manuel de DS 3.0 (`hdl.handle.net/2047/D20621423`) puis `rfbench data download tprime_wifi4 --manual-archive <zip>`. Ensuite prepare + train `tprime` → colonne réelle. |
| 9 | **2e dataset AMC — RadioML 2018.01a** | 🔑 | Kaggle-gated (token) OU DeepSig cert expiré. Décision : token Kaggle / download autorisé / dépôt manuel. Débloque 24-class + RIS-MAE. |
| 10 | **Détection — RadDet** | 🔑 | Kaggle token. Une fois là : baseline détection (WP-22/32, YOLO/DETR) sur `wideband_detection` = **WF**. |

## P3 — Wave B, nouveaux tracks, lancement
| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 11 | **Wave B — spectrum sensing** (M7) | **WF** | DeepSense : data + tâche (WP-14/23) + baseline (WP-33), copie du gabarit `interference_id`/`protocol_tech_id`. |
| 12 | **`source_separation` (RFSS)** | 🔑 | Bloqué : dataset RFSS pas encore released (HF annoncé). Une fois là : tâche + Conv-TasNet/DPRNN, métrique co-channel PI-SI-SINR. |
| 13 | **Fix déploiement GitHub Pages** | 🔑 | Le build réussit ; seul *Deploy to Pages* échoue (config/permissions du workflow ou réglages Pages). À régler dans Settings > Pages / Actions. |
| 14 | **Publication v1** (WP-71) | solo | README final, CITATION.cff, tag v1.0, une fois ≥1 ligne FM + SEI au board. |

---

## Workflows à prévoir (récap)
1. **FM reproduction propre** (#6) — le plus gros levier (thèse FM-vs-baseline). *(sous-agents bloqués
   tant que la limite de dépense mensuelle du compte n'est pas relevée.)*
2. **Détection RadDet** (#10, après data).
3. **Wave B spectrum sensing** (#11) — bout-en-bout.

Ordre conseillé : **#5 SEI (en vol) + #7 verified (court) → #6 FM (thèse) → #8 débloquer T-PRIME →
data #9/#10 → Wave B → v1.**
