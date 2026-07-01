# rf-benchmark-hub — Claude Code

> Benchmark hub pour modèles RF (foundation models et baselines).
> Ce fichier est lu automatiquement par Claude Code à l'ouverture du dépôt.
> **Statut : scaffold v0.** Les fichiers de spec réels (schémas, contrats, plan
> des work packages) sont fournis par Maxence et doivent être réconciliés avec
> ce document dès leur import — en cas de conflit, la spec importée fait foi.

## But du projet

Plateforme de benchmark reproductible pour tâches RF : préparation de données,
entraînement/éval de baselines et fine-tuning de FM, leaderboard avec
soumissions **vérifiables**. Le code s'écrit partout ; l'entraînement lourd
tourne sur le cluster.

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

## Architecture (cible)

```
rf-benchmark-hub/
  src/rfbench/          # package Python (CLI `rfbench` + lib)
    schemas/            # schémas pydantic : result.json, submission, task spec
    contracts/          # interfaces core (Task, Metric, Dataset, Model)
    tasks/              # une tâche = un plugin (AMC, SEI, detection, …)
    metrics/            # métriques par tâche
    data/               # préparation des splits (indices, pas les données)
    cli/                # `rfbench data prepare`, `rfbench verify`, …
  docs/
    IMPLEMENTATION_PLAN.md   # les work packages (WP-xx) + dépendances + critères
  configs/              # configs d'expérience (yaml), seed inclus
  slurm/                # scripts SLURM générés pour le cluster
  tests/
  .github/workflows/    # CI
```

## CLI `rfbench` (contrat d'usage)

- `rfbench data prepare <task>` — peuple le storage (via `RFBENCH_CACHE`), écrit les
  indices de split. Ne télécharge/écrit jamais dans le repo.
- `rfbench verify <submission>` — re-run officiel niveau 2, produit/valide
  `result.json`, positionne `verified`.

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
