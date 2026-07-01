# Architecture

## Layers
1. **Contracts** (`rfbench/core/`): `Task`, `Dataset`, `Metric`, `Model`+`Regime`, `evaluate()`.
   Frozen in M0; changing them requires review + version bump.
2. **Data** (`rfbench/data/`): per-dataset `download`/`prepare`; deterministic splits via
   `core/splits.py`; only indices + checksums are versioned (`leaderboard/splits/`).
3. **Tasks** (`rfbench/tasks/<task>/`): dataset adapters + metrics + Hydra config.
4. **Models** (`rfbench/models/`): `baselines/` (seed the board) and `foundation/` (wrappers exposing
   `embed()` for `linear_probe`/`few_shot`).
5. **Leaderboard** (`leaderboard/`): `results/**.json` are the **source of truth**; `site/generate.py`
   renders a static site (GitHub Pages).

## Result flow
`rfbench eval` → `evaluate()` → `result.json` (validated vs `schemas/result.schema.json`) → PR →
CI validation → maintainer re-run → `verified` → `build-leaderboard` → Pages.

## Regimes
Adapters wrap any `Model`: `linear_probe` (freeze + linear head on `embed()`), `full_finetune`,
`few_shot(k)`, `from_scratch`. The regime is written into every `result.json` and used to filter the board.
