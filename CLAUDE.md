# rf-benchmark-hub — Claude Code

> Benchmark hub public pour tâches RF **terrestres** (baselines spécialisées vs
> foundation models fine-tunés, sous protocole commun). Le satellite = dépôt séparé.
> Ce fichier est lu automatiquement par Claude Code à l'ouverture du dépôt.
>
> **Spec maîtresse = `docs/`** (source de vérité, prime sur ce fichier en cas de conflit) :
> - `docs/IMPLEMENTATION_PLAN.md` — plan + work packages (WP-xx) + milestones M0→M7.
> - `docs/EVALUATION_PROTOCOL.md` — **normatif et versionné** : métriques/splits par tâche.
> - `docs/ARCHITECTURE.md` — couches (contracts/data/tasks/models/leaderboard).
> - `docs/SUBMISSION.md` — workflow de soumission + vérification 2 niveaux.

## But du projet

Plateforme de benchmark reproductible pour tâches RF (AMC, SEI/RF-fingerprinting,
détection large bande, spectrum sensing) : préparation de données, entraînement/éval
de baselines et fine-tuning de FM, leaderboard avec soumissions **vérifiables**.
Le code s'écrit partout ; l'entraînement lourd tourne sur le cluster.
Licence code : **Apache-2.0**. Datasets : **jamais redistribués** (licences sources).

## Principes non négociables

- **Aucune donnée en git.** Datasets, checkpoints, résultats bruts vivent sur le
  storage du cluster via `RFBENCH_CACHE`. Le `.gitignore` protège déjà ; ne jamais
  forcer l'ajout (`git add -f`) de `data/`, `*.pt`, `*.ckpt`, `*.h5`, `*.npy`.
- **Contrats figés d'abord.** Sprint 0 (schémas + contrats core) atterrit sur
  `main` avant toute parallélisation. Tous les WP consomment ces interfaces.
- **Reproductibilité.** `seed: 42` par défaut dans les configs d'expérience.
  Toute soumission produit un `result.json` conforme au schéma.
- **Vérification à deux niveaux.** Niveau 1 = CI valide le format. Niveau 2 =
  re-run officiel de la soumission sur le cluster puis bascule du flag `verified`.

## Architecture (cible — cf. `docs/ARCHITECTURE.md` §3 du plan)

```
rf-benchmark-hub/
  pyproject.toml        # package rfbench + entrypoint CLI
  schemas/              # result.schema.json, submission.schema.json (JSON Schema)
  rfbench/              # package importable + CLI `rfbench` (racine, PAS src/)
    core/               # CONTRATS figés en M0 : task, dataset, metric, model,
                        #   registry, splits, evaluate, manifest
    tasks/{amc,sei,wideband_detection,spectrum_sensing}/   # une tâche = adapters+métriques+config
    models/{baselines,foundation}/   # foundation/ expose embed() pour linear_probe/few_shot
    data/{download,prepare}/         # scripts par dataset (jamais de données versionnées)
  configs/              # Hydra : config.yaml, task/, model/, regime/
  leaderboard/
    results/<task>/*.json   # SOURCE DE VÉRITÉ des scores (versionnés)
    splits/<dataset>/*.idx.json  # indices + checksum (versionnés, PAS les données)
    site/generate.py         # results/*.json -> site statique GitHub Pages
  docs/                 # spec maîtresse (voir en-tête)
  tests/
  .github/workflows/    # ci.yml, validate-submission.yml, build-leaderboard.yml
```

## CLI `rfbench` (contrat d'usage — WP-42)

- `rfbench data prepare <task>` — download source officielle + `prepare` : écrit les
  indices de split déterministes + checksums. Ne versionne jamais de données brutes.
- `rfbench eval <task> --model <name> --regime <regime>` — émet `result.json` validé
  vs `schemas/result.schema.json` (tier 1 self-serve → `self_reported`).
- `rfbench submit --check` — valide le manifeste de repro localement avant PR.
- `rfbench leaderboard build` — régénère le site statique depuis `results/*.json`.
- `rfbench verify <pr>` (mainteneur, WP-53) — rejoue l'éval (eval-only ou re-train)
  et flippe `verification.status → verified`.

## Vérification à deux niveaux (D4)

- **Tier 1 self-serve** : n'importe qui run l'éval en local → `self_reported`.
- **Tier 2 officiel** : le mainteneur **re-exécute** sur station multi-GPU et signe
  `verified` (avec `verified_by/date/hardware`) si repro dans la `tolerance`.
  Le leaderboard ne mélange jamais deux régimes dans une colonne.

## Environnement (cluster Dalia / IDRIS)

- Gestion Python : **uv** uniquement (`uv add`, `uv run python`) — jamais `pip`/`conda`.
- Frontale = Intel x86_64 : **ne jamais tester de code GPU / installer de deps ici.**
- Noeuds de calcul = **ARM aarch64**, GPU **GB200** : compiler/installer les deps
  sur un noeud interactif ARM.
- Scheduler **SLURM**. `RFBENCH_CACHE` pointe sur le storage cluster (`$WORK`).

## Workflow de dev (swarm sans collisions)

1. Sprint 0 (WP-00 bootstrap → WP-01 schémas → WP-02 contrats core → WP-03/04
   docs+CI) merge sur `main` **avant** toute parallélisation.
2. Ensuite : **un `git worktree` + une branche par WP** (`git worktree add
   ../wp-20-amc -b wp-20-amc`), un agent Claude Code par worktree.
3. N'assigner en parallèle que des WP **sans dépendance** entre eux (cf.
   `docs/IMPLEMENTATION_PLAN.md`). PR → CI → merge.

## Conventions de code

- Python : type hints partout, `snake_case`, modules `lowercase_with_underscores`.
- Pas de `print` de debug commité — utiliser `logging`.
- Vérifier `git status` avant tout commit ; commiter avant tout `sbatch`.
- Citer chemin + numéro de ligne quand on référence du code.
