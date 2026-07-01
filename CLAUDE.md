# CLAUDE.md — contexte projet pour Claude Code

Tu travailles sur **RF-Benchmark-Hub** : un dépôt public (privé au départ) de **benchmarks** et d'un
**leaderboard** pour les tâches downstream ML sur signaux RF **terrestres**, comparant baselines
spécialisées et modèles de fondation fine-tunés.

**Avant de coder, lis `docs/IMPLEMENTATION_PLAN.md`** (plan + work packages WP-xx) et
`docs/EVALUATION_PROTOCOL.md` (normatif). Travaille par work package.

## Règles non négociables
- **Jamais de données brutes dans git** (`.h5/.npy/.bin/.sigmf-data` sont git-ignorés ET bloqués en CI).
  Les datasets se téléchargent via `rfbench data prepare`; seuls les *indices de split* + checksums
  sont versionnés (`leaderboard/splits/`).
- **Contrats core figés** (`rfbench/core/`: Task, Dataset, Metric, Model+Regime, evaluate). Ne pas
  changer leurs signatures sans une revue explicite + bump de version.
- **Régime déclaré** dans chaque `result.json` (`from_scratch|full_finetune|linear_probe|few_shot`),
  jamais inféré. Le leaderboard ne mélange jamais deux régimes dans une colonne.
- **Protocole complet** : ex. AMC = plage SNR complète (pas de cherry-picking) ; SEI = pistes
  closed-set / cross-receiver / cross-day séparées.
- Toute soumission de score = un `result.json` valide contre `schemas/result.schema.json`.

## Où se trouve quoi
- `rfbench/core/` contrats · `rfbench/tasks/<task>/` tâches · `rfbench/models/{baselines,foundation}/`
- `rfbench/data/download/` scripts (pas de données) · `configs/` Hydra · `leaderboard/` résultats+site
- `schemas/` JSON · `.github/workflows/` CI

## Commandes

```bash
pip install -e ".[dev]"      # setup
ruff check . && pytest -q    # lint + tests (doit rester vert)
rfbench --help               # CLI: data prepare / eval / submit --check / leaderboard build
python leaderboard/site/generate.py --results leaderboard/results --out site_build
```

## Découpage compute (voir aussi le plan §)
- **Local / CPU** : contrats, schémas, data-prep (indices), métriques, site, CI. La majorité de M0–M2.
- **Cluster GPU** : entraînement des baselines (M3), fine-tuning FM (M6), et les **re-runs de
  vérification officielle**. Écris les jobs (SLURM ou script) ; ne lance pas d'entraînement lourd en local.
- Les datasets vivent sur le **storage du cluster** : respecte la variable `RFBENCH_CACHE`
  (chemin de cache datasets), ne code pas de chemins en dur.

## Style
- Python 3.10+, typé, `ruff`+`black`. Petites PR, une par work package. Tests pour tout nouveau contrat
  ou métrique. Docstrings sur les interfaces publiques.
