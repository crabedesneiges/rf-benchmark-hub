# RF-Benchmark-Hub — Plan d'implémentation

> Dépôt public de **benchmarks** et **leaderboard** pour les tâches downstream du ML sur signaux RF
> **terrestres**, permettant de comparer des **baselines spécialisées** et des **modèles de fondation
> fine-tunés** sous un protocole commun. Le satellite fera l'objet d'un dépôt séparé.

Ce document est la **spécification maîtresse** destinée à être découpée en lots (« work packages »,
WP) assignables à un swarm d'agents de code. Chaque WP a un livrable, des critères d'acceptation et des
dépendances explicites.

---

## 1. Décisions de conception (verrouillées)

| # | Décision | Choix |
|---|----------|-------|
| D1 | Périmètre | **Terrestre uniquement.** Satellite = dépôt séparé (2e GitHub). |
| D2 | Tâches v1 | **AMC**, **SEI/RF-fingerprinting**, **Détection large bande**, **Spectrum sensing** (phasées, cf. §4). |
| D3 | Données | **Aucune redistribution.** Scripts de `download` (source officielle) + `prepare` générant des **splits canoniques déterministes** (indices + graine). Seuls des *fichiers d'indices* et *checksums* sont versionnés. |
| D4 | Vérification | **Deux niveaux.** (1) *self-serve* : quiconque exécute l'éval en local et obtient son `result.json` pour son papier. (2) *officiel* : le mainteneur **re-exécute** (station multi-GPU) et passe le score `verified`. |
| D5 | Comparaison FM vs baseline | **Régime libre mais déclaré/taggé** : `from_scratch` \| `full_finetune` \| `linear_probe` \| `few_shot(k)`. Le leaderboard filtre/colonne par régime. |
| D6 | Stack (recommandation) | Package Python **`rfbench`** (PyTorch ; Lightning optionnel), configs **Hydra**, **Docker** pour l'éval reproductible, **résultats = JSON versionnés dans le repo** (source de vérité), **leaderboard = site statique GitHub Pages** généré par script + Action. HF Space en miroir optionnel plus tard. |
| D7 | Licence code | **Apache-2.0** (le code) ; les licences des *datasets* restent celles des sources (jamais redistribuées). |

---

## 2. Principe directeur

La valeur du dépôt n'est **pas** d'agréger des chiffres (souvent incomparables) mais de **figer un
protocole canonique par tâche** et d'en garantir la **reproductibilité**. Trois invariants :

1. **Un contrat unique** : toute tâche expose `Dataset`, `Metric`, un `split canonique`, et un
   `protocole d'éval`. Tout modèle expose une interface `Model` + un `régime` déclaré.
2. **Le score officiel est reproductible** : une soumission fournit code + poids + Docker + commande
   exacte ; le mainteneur rejoue et signe (`verified`).
3. **La comparabilité prime sur l'exhaustivité** : plage de SNR complète imposée, splits publiés,
   métriques définies formellement.

---

## 3. Architecture cible (arborescence)

```
rf-benchmark-hub/
  README.md                      # présentation + quickstart + lien leaderboard
  CONTRIBUTING.md                # comment soumettre (2 niveaux)
  LICENSE                        # Apache-2.0
  pyproject.toml                 # package rfbench + deps + entrypoint CLI
  docs/
    IMPLEMENTATION_PLAN.md       # ce document
    ARCHITECTURE.md              # détails d'ingénierie
    EVALUATION_PROTOCOL.md       # métriques/splits/régimes par tâche (source normative)
    SUBMISSION.md                # workflow de soumission + vérification
  schemas/
    result.schema.json           # entrée de leaderboard (une soumission = un JSON)
    submission.schema.json       # manifeste de repro (code, docker, poids, commande)
  rfbench/                       # le harnais (package importable + CLI `rfbench`)
    core/
      task.py                    # ABC Task (registre des tâches)
      dataset.py                 # ABC Dataset + DataLoader canonique
      metric.py                  # ABC Metric + implémentations communes
      model.py                   # ABC Model + Adapter (régimes)
      splits.py                  # génération déterministe des splits (seed)
      evaluate.py                # boucle d'éval -> result.json (validé par schema)
      registry.py                # enregistrement tâches/modèles/métriques
      manifest.py                # checksums datasets + provenance
    tasks/
      amc/                       # Wave A
      sei/                       # Wave A
      wideband_detection/        # Wave A
      spectrum_sensing/          # Wave B
    models/
      baselines/                 # ≥1-2 baselines reproductibles par tâche (seed du leaderboard)
      foundation/                # wrappers FM + têtes (linear_probe / finetune / few_shot)
    data/
      download/                  # scripts par dataset (téléchargement, PAS de données versionnées)
      prepare/                   # construction des splits canoniques + manifest
  configs/                       # Hydra
    config.yaml
    task/{amc,sei,wideband_detection,spectrum_sensing}.yaml
    model/*.yaml
    regime/{from_scratch,full_finetune,linear_probe,few_shot}.yaml
  leaderboard/
    results/<task>/*.json        # SOURCE DE VÉRITÉ des scores (versionnés)
    splits/<dataset>/*.idx.json  # indices canoniques + checksum (versionnés, pas les données)
    site/generate.py             # results/*.json -> site statique
    site/templates/
  .github/
    workflows/ci.yml                     # lint + tests unitaires
    workflows/validate-submission.yml    # PR: validation schema + complétude repro
    workflows/build-leaderboard.yml      # régénère le site -> GitHub Pages
    ISSUE_TEMPLATE/{submission.yml,task_proposal.yml}
    PULL_REQUEST_TEMPLATE.md
  tests/
```

---

## 4. Phasage des tâches

- **Wave A (cœur)** : AMC, SEI, Détection large bande. Datasets riches, métriques nettes, baselines
  reproductibles connues (TorchSig, MCLDNN, WiSig, WBSig53).
- **Wave B** : Spectrum sensing (DeepSense). Ajoutée une fois le harnais stabilisé sur Wave A.

Chaque tâche suit le **même gabarit** (WP-T*), donc Wave B ne réintroduit pas de dette : c'est un
copier-adapter du gabarit validé sur Wave A.

---

## 5. Contrats (interfaces normatives)

Ces signatures sont le **point de synchronisation** du swarm : elles sont figées en M0 et ne changent
plus sans revue. (Implémentations détaillées dans le scaffold `rfbench/core/`.)

- `Task` : `name`, `datasets()`, `metrics()`, `default_split()`, `build_targets(batch)`.
- `Dataset` : `download()`, `prepare()` (→ splits déterministes + manifest), `load(split)`,
  `canonical_split_id`, `checksum`.
- `Metric` : `update(pred, target, meta)`, `compute() -> dict`, `primary_key` (métrique de tri).
- `Model` : `forward(x)`, `embed(x)` (pour linear_probe/few_shot), `regime`, `n_params`.
- `evaluate(model, task, split, regime) -> result.json` conforme à `schemas/result.schema.json`.

**Régimes** (D5) implémentés comme *adaptateurs* autour d'un `Model` :
`from_scratch`, `full_finetune`, `linear_probe` (features gelées + tête linéaire),
`few_shot(k)` (k exemples/classe). Le régime est **déclaré** dans `result.json` et **jamais inféré**.

---

## 6. Modèle de vérification (D4) — le cœur du leaderboard

```
Soumissionnaire                          Dépôt / Mainteneur
---------------                          ------------------
1. rfbench eval ... --emit result.json   (local, self-serve : chiffre pour son papier)
2. ouvre une PR :                         3. CI valide :
   - leaderboard/results/<task>/x.json       - conformité result.schema.json
   - reproduce/ (submission.schema.json):     - complétude du manifeste repro
     commit code, image Docker, URL poids,     (code+docker+poids+commande présents)
     commande exacte, hardware                - lint des indices de split (checksum)
                                          4. Mainteneur (multi-GPU) rejoue :
                                             - eval-only (si poids fournis) OU
                                             - ré-entraînement complet (baselines)
                                          5. Si reproduit within tolérance :
                                             verified=true, verified_by/date/hardware
                                          6. build-leaderboard régénère le site
```

Le leaderboard affiche **deux pistes** : `self-reported` (tag) et `verified` (rejoué par le
mainteneur). La confiance vient de la piste `verified`, pas du volume.

---

## 7. Milestones

| M | Titre | Sortie |
|---|-------|--------|
| **M0** | Bootstrap & contrats | Repo, `pyproject`, ABCs figées, schémas JSON, CI squelette, docs normatives. |
| **M1** | Couche données | `download`+`prepare`+`splits` déterministes + manifest/checksums pour les 4 datasets. |
| **M2** | Tâches (Wave A) | AMC, SEI, Détection : `Dataset`+`Metric`+configs, tests. |
| **M3** | Baselines seed | ≥1-2 baselines reproductibles par tâche Wave A, scores `verified` de départ. |
| **M4** | Harnais d'éval + régimes | `evaluate()` + 4 régimes + CLI `rfbench eval` + émission `result.json`. |
| **M5** | Leaderboard + soumission | Site statique GH Pages + workflow PR + pipeline de vérification. |
| **M6** | Intégration FM | Wrappers FM + exemple : un FM évalué en linear_probe/finetune/few_shot sur les 3 tâches. |
| **M7** | Wave B + lancement | Spectrum sensing + polish docs + annonce v1. |

---

## 8. Work packages (assignables au swarm)

Format : **WP-id — titre** · _dépend de_ · **livrable** · **acceptation**.
Les WP d'un même milestone sont largement parallélisables sauf dépendance notée.

### M0 — Bootstrap & contrats
- **WP-00 — Repo & packaging** · _—_ · `pyproject.toml`, package `rfbench`, entrypoint CLI `rfbench`, pré-commit (ruff+black), LICENSE Apache-2.0. · *Accept.* : `pip install -e .` OK ; `rfbench --help` liste les sous-commandes.
- **WP-01 — Schémas JSON** · _—_ · `schemas/result.schema.json`, `schemas/submission.schema.json`. · *Accept.* : un exemple valide et un invalide testés (jsonschema) en CI.
- **WP-02 — Contrats core (ABCs)** · _WP-00_ · `rfbench/core/{task,dataset,metric,model,registry,splits,evaluate,manifest}.py` (interfaces + docstrings + types, corps minimal). · *Accept.* : `import rfbench` OK, mypy passe, ABCs non instanciables.
- **WP-03 — Docs normatives** · _—_ · `EVALUATION_PROTOCOL.md`, `SUBMISSION.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`. · *Accept.* : chaque tâche a métrique+split+régimes définis ; le workflow de soumission est décrit de bout en bout.
- **WP-04 — CI squelette** · _WP-00,01_ · `.github/workflows/ci.yml` (lint, tests, schema-check). · *Accept.* : CI verte sur un commit vide de features.

### M1 — Couche données (pas de redistribution)
- **WP-10 — Splits déterministes** · _WP-02_ · `core/splits.py` : génération reproductible (seed) → fichiers `leaderboard/splits/<dataset>/*.idx.json` + checksum. · *Accept.* : deux exécutions produisent des indices identiques (test).
- **WP-11 — AMC data (RadioML/Sig53)** · _WP-10_ · `data/download/amc_*.py` + `prepare` (stratifié par SNR), manifest. · *Accept.* : `rfbench data prepare amc` reconstruit les splits + vérifie les checksums, sans versionner de données.
- **WP-12 — SEI data (WiSig/ORACLE)** · _WP-10_ · idem, splits **cross-receiver** et **cross-day** explicites. · *Accept.* : splits séparés closed-set / cross-rx / cross-day générés.
- **WP-13 — Détection data (WBSig53)** · _WP-10_ · génération via TorchSig + annotations T-F. · *Accept.* : dataset détection chargé avec boîtes T-F.
- **WP-14 — Spectrum sensing data (DeepSense)** · _WP-10_ · download datasets libres + prepare. · *Accept.* : occupation labellisée chargée (Wave B).

### M2 — Tâches (Wave A)
- **WP-20 — Tâche AMC** · _WP-11_ · `tasks/amc/` : `Dataset`, `Metric` (accuracy globale + courbe acc-vs-SNR + macro-F1), `configs/task/amc.yaml`. · *Accept.* : métriques calculées sur un dummy model ; plage SNR complète imposée.
- **WP-21 — Tâche SEI** · _WP-12_ · `tasks/sei/` : rank-1 (closed-set) + AUROC/EER (open-set), conditions cross-rx/day. · *Accept.* : pistes closed/open-set distinctes.
- **WP-22 — Tâche Détection** · _WP-13_ · `tasks/wideband_detection/` : mAP/mAR (TorchMetrics), IoU. · *Accept.* : mAP calculé sur boîtes T-F.
- **WP-23 — Tâche Spectrum sensing** · _WP-14_ · `tasks/spectrum_sensing/` : Pd/Pfa (ROC), latence. · *Accept.* : courbe ROC produite (Wave B).

### M3 — Baselines seed
- **WP-30 — AMC baselines** · _WP-20_ · wrappers **TorchSig/XCiT** et **MCLDNN** (adaptateurs, poids reconstruits par entraînement documenté). · *Accept.* : reproduit ±tolérance les chiffres publiés (XCiT 71,16 % Sig53 ; MCLDNN sur 2016.10a) ; `verified` de départ.
- **WP-31 — SEI baseline** · _WP-21_ · wrapper WiSig (scripts officiels) ± ORACLE-CNN. · *Accept.* : reproduit l'accuracy de référence et la chute cross-rx/day.
- **WP-32 — Détection baseline** · _WP-22_ · wrapper WBSig53 (DETR/YOLOv5 via TorchSig). · *Accept.* : mAP ~86 (DETR-B4) / ~73 (YOLOv5-pico) reproduits ±tolérance.
- **WP-33 — Spectrum baseline** · _WP-23_ · wrapper DeepSense (repo wineslab). · *Accept.* : ~98 %/97 % reproduits (Wave B).

### M4 — Harnais d'éval + régimes
- **WP-40 — evaluate() + result.json** · _WP-02,01_ · boucle d'éval, agrégation métriques, émission `result.json` validé schema, empreinte hardware/seed/env. · *Accept.* : `rfbench eval` produit un JSON valide.
- **WP-41 — Régimes (adaptateurs)** · _WP-40_ · `linear_probe`, `full_finetune`, `few_shot(k)`, `from_scratch` + `configs/regime/*.yaml`. · *Accept.* : le même modèle tourne dans les 4 régimes, régime écrit dans le JSON.
- **WP-42 — CLI** · _WP-40_ · sous-commandes `data prepare`, `eval`, `submit --check`, `leaderboard build`. · *Accept.* : chaque sous-commande documentée et testée.

### M5 — Leaderboard + soumission
- **WP-50 — Générateur de site** · _WP-40_ · `leaderboard/site/generate.py` : lit `results/*.json` → site statique (tri par métrique primaire, filtre par régime, badge verified/self-reported). · *Accept.* : site généré localement à partir des JSON d'exemple.
- **WP-51 — Validation de soumission (CI)** · _WP-01,50_ · `validate-submission.yml` : schema + complétude du manifeste repro + checksum splits. · *Accept.* : une PR mal formée est rejetée avec message clair.
- **WP-52 — Publication (CI)** · _WP-50_ · `build-leaderboard.yml` → GitHub Pages ; templates d'issues + PR. · *Accept.* : merge sur `main` republie le site.
- **WP-53 — Pipeline de vérification (mainteneur)** · _WP-51_ · script `rfbench verify <pr>` qui rejoue l'éval (eval-only ou ré-entraînement) et flippe `verified`. · *Accept.* : un run de démonstration passe un score de self-reported à verified.

### M6 — Intégration modèles de fondation
- **WP-60 — Wrappers FM** · _WP-41_ · adaptateur générique `models/foundation/` (charge un backbone RF, expose `embed()` + têtes). · *Accept.* : un FM (ex. TorchSig/IQFM-like) évalué en linear_probe + finetune + few_shot sur AMC, SEI, Détection ; 3 lignes de leaderboard taggées par régime.
- **WP-61 — Guide « ajouter un modèle »** · _WP-60_ · doc + template. · *Accept.* : un contributeur ajoute un modèle en suivant le template sans toucher au core.

### M7 — Wave B + lancement
- **WP-70 — Intégration Wave B** · _WP-23,33_ · spectrum sensing bout-en-bout. · *Accept.* : tâche + baseline + leaderboard actifs.
- **WP-71 — Polish & annonce** · _tous_ · README final, page d'accueil leaderboard, CITATION.cff, versionnage v1.0. · *Accept.* : dépôt public prêt, quickstart reproductible en < 30 min.

---

## 9. Ordre de démarrage recommandé pour le swarm

1. **Sprint 0 (bloquant, séquentiel)** : WP-00 → WP-01 → WP-02 → WP-03/04. *Les contrats et schémas figés débloquent tout le reste.*
2. **Sprint 1 (parallèle)** : WP-10 puis {WP-11, WP-12, WP-13} en parallèle ; WP-40/41/42 (harnais) en parallèle de la data ; WP-50 (site) sur des JSON d'exemple.
3. **Sprint 2 (parallèle)** : {WP-20, WP-21, WP-22} ; puis {WP-30, WP-31, WP-32}.
4. **Sprint 3** : WP-51/52/53 (soumission+vérif) ; WP-60/61 (FM).
5. **Sprint 4** : Wave B (WP-14/23/33/70) + WP-71 (lancement).

---

## 10. Risques & garde-fous

- **Licences non-commerciales (RadioML, etc.)** → jamais de données dans le repo ; uniquement scripts + indices. Un test CI échoue si un fichier de données brut est commité (garde-fou taille/extension).
- **Reproductibilité fragile** (stacks TF1 anciennes : MCLDNN) → conteneur figé par baseline ; verdict `verified` conditionné au re-run.
- **Comparaisons trompeuses FM** → régime **obligatoire** dans `result.json` ; le site ne compare jamais deux régimes différents dans la même colonne.
- **Coût de vérification** → tier officiel en **eval-only** par défaut (poids fournis) ; ré-entraînement réservé aux baselines seed.
- **Dérive de barème** → `EVALUATION_PROTOCOL.md` est **normatif et versionné** ; tout changement de métrique = bump de version de tâche.
