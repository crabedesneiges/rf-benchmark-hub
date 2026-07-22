# Next steps & workflows

État (2026-07-21). Le framework est complet (contrats core, schémas, data-prep, métriques, site, CI).
Six tâches sont `implemented` avec au moins une colonne réelle au board :

- **`amc`** — RadioML 2016.10a : MCLDNN **0.6171** · CLDNN **0.5805** · ResNet **0.5661** (from_scratch,
  recette finale seed 42) + HOC-LR et lignes triviales (chance / majority). RadioML 2018.01a :
  **MCLDNN 0.606** (première baseline réelle 24-class, from_scratch). Lignes littérature `from_paper` :
  MoEformer (0.6374 sur 2016, 0.6422 sur 2018), TLDNN (0.6283 sur 2016, 0.6332 sur 2018).
- **`sei`** — WiSig ManyTx : `wisig_cnn_paper` / `complex_cnn` / `resnet1d_sei` sur les 4 pistes
  (closed_set / cross_receiver / cross_day / open_set). ORACLE : ligne `oracle-cnn-paper` (0.986,
  `from_paper_uncertain`) ; la repro in-repo `oracle_cnn` sur notre split est en cours.
- **`interference_id`** — `interf_cnn` **0.9987** (GNSS jamming 6-class, data Zenodo publique).
- **`snr_estimation`** — `snr_cnn` **5.73 dB** RMSE + 2 lignes triviales (mean / moment-ridge).
- **`protocol_tech_id`** — `tprime` (transformer) **0.995** closed_set (per-fenêtre, recette officielle)
  + **0.947** cross_room (scenario-split leave-one-location-out) ; lignes `from_paper*` T-PRIME et
  WirelessJEPA.

Le board est **groupé par (dataset, regime, k_shot, track)** ; il ne mélange jamais deux régimes ni
deux pistes dans une colonne. Site déployé sur GitHub Pages (page `foundation.html` dédiée aux FM).

Légende **Workflow ?** : `solo` = 1 agent / main-loop ; `WF` = workflow multi-agents (fan-out + revue) ;
`cluster` = job(s) SLURM sur GB200 ; `🔑` = bloqué sur credentials/décision.

---

## ✅ Landé récemment (juillet 2026)

- **`protocol_tech_id` complet** : T-PRIME téléchargé + préparé, `tprime` réimplémenté fidèle au code
  officiel (le 0.71 bugué de la première passe a été retiré). Deux pistes réelles au board :
  **closed_set 0.995** (reproduit le papier per-fenêtre) et **cross_room 0.947** (split
  leave-one-location-out RM_142 / RM_572C / RM_573C, seed 42).
- **RadioML 2018.01a préparé** : split canonique `amc-radioml2018-strat-snr-8010-seed42-v1` committé,
  première baseline réelle 24-class au board (**MCLDNN 0.606** full-SNR, from_scratch). Le dataset
  n'est plus un blocker.
- **ORACLE préparé** : split canonique `sei-oracle-closedset-8ft-strat-tx-8010-seed42-v1` committé (cap de
  1024 fenêtres/capture pour borner la taille de l'index, 243 Mo → ~5 Mo). Ligne `from_paper_uncertain`
  au board (0.986) ; repro in-repo en cours.
- **Bibliographie SOTA consolidée** (`docs/BIBLIOGRAPHY.md`, source de vérité unique — `SOTA_REFERENCE.md`
  n'est plus qu'un stub de redirection). Quatre lignes littérature `from_paper` au board (MoEformer +
  TLDNN sur 2016 et 2018, ORACLE, WirelessJEPA), taguées comme figures de papier jamais confondues avec
  une reproduction in-repo. Audit d'ajouts du 2026-07-20 intégré (MoEformer arXiv:2606.09085 = nouveau
  SOTA AMC 2016+2018 ; verdicts `not_addable` : CBADNN, Swinney-Woods, RadDet/DeepSense/Sig53).

---

## En attente de GPU (colonnes prêtes côté code, scores à produire)

Le code, les splits et les recettes sont en place ; il ne manque que des runs cluster pour flipper ces
lignes de « prête » à « chiffre réel au board ». Écrire les jobs SLURM (commiter avant `sbatch`) ; ne
pas lancer d'entraînement lourd en local.

| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 1 | **SEI : produire les scores WiSig réels** | cluster | Les baselines (`wisig_cnn_paper`, `complex_cnn`, `resnet1d_sei`) sont wirées sur les 4 pistes ; lancer le run et écrire les `result.json` closed/cross-rx/cross-day. Montre la **chute cross-receiver/cross-day** (point clé WiSig). |
| 2 | **SEI : repro ORACLE in-repo** | cluster | Données ORACLE sur le cluster + split committé ; lancer `oracle_cnn` (`from_scratch`) sur notre split → ligne `self_reported`, distincte du 0.986 `from_paper_uncertain`. |
| 3 | **AMC 2018 : compléter la colonne** | cluster | MCLDNN 0.606 landé ; ajouter CLDNN / ResNet / éventuellement une baseline mid-tier sur 2018.01a sous la même recette pour une colonne 24-class fournie. |
| 4 | **Démo tier `verified`** | cluster | `rfbench verify` : re-run d'une baseline (ex. MCLDNN) depuis son checkpoint, flip `self_reported`→`verified` dans la tolérance. Prouve le protocole de vérification en vrai (1 job court, indépendant). |

## FM vs baseline (le cœur du hub)

| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 5 | **1re ligne FM reproduite vs baseline** | **WF** | LWM-Spectro ne peut PAS la fournir (aucune tâche RadioML, préproc IQ→spectrogramme non publiée ; sa tâche `snr_mobility` reproduite à 93.9% — voir BIBLIOGRAPHY §B.5). Cibler un FM à **tâche RadioML réelle** : **WirelessJEPA** (74.78% linear-probe sur 2016.10a, bat MCLDNN — poids à ré-entraîner) ou **IQFM** (38.1% @50/cl OOD). Head logreg sur features gelées, licence vérifiée avant publication. Les deux ont déjà une ligne `from_paper` au board ; la repro in-repo sera un chiffre séparé, in-distribution, jamais confondu. |

## Nouveaux datasets / tracks (gated ou non released)

| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 6 | **Détection — RadDet** | 🔑 | Kaggle token. Une fois là : baseline détection (YOLO/DETR) sur `wideband_detection` = **WF**. Loader d'annotations + tâche déjà wirés. |
| 7 | **Wave B — spectrum sensing** (DeepSense) | **WF** | Track `spectrum_sensing` défini, pas encore de loader/modèle. Data + tâche + baseline CNN, en copiant le gabarit `interference_id` / `protocol_tech_id`. |
| 8 | **`source_separation` (RFSS)** | 🔑 | Bloqué : dataset RFSS pas encore released (HF annoncé). Une fois là : tâche + Conv-TasNet/DPRNN, métrique co-channel PI-SI-SINR. |

## Infra / publication

| # | Étape | Workflow ? | Détail |
|---|---|---|---|
| 9 | **Fix déploiement GitHub Pages** | 🔑 | Le build réussit ; seul *Deploy to Pages* échoue par intermittence (config/permissions du workflow ou réglages Pages). À régler dans Settings > Pages / Actions. |
| 10 | **Publication v1** | solo | README final, CITATION.cff, tag v1.0, une fois ≥1 ligne FM reproduite + colonne SEI réelle au board. |

---

## Ordre conseillé

**#1 SEI (scores WiSig) + #2 ORACLE repro** (colonnes SEI réelles, le gros trou actuel du board) →
**#4 démo `verified`** (court, indépendant) → **#5 FM reproduite** (thèse FM-vs-baseline) →
**#3 compléter AMC 2018** → datasets gated (#6/#7) → **v1** (#10).
