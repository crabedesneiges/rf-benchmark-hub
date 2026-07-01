<!-- Keep PRs small: one work package per PR (see docs/IMPLEMENTATION_PLAN.md §8). -->

## What & why

<!-- One or two sentences. Link the WP-id or issue. -->

## Checklist

- [ ] **Schema valid** — any `result.json` / manifest validates against `schemas/*.schema.json`.
- [ ] **Regime declared** — every result declares its regime (`from_scratch` | `full_finetune` | `linear_probe` | `few_shot`); never inferred, never mixed in a column.
- [ ] **No raw data** — only split indices + checksums are versioned; no datasets or weights committed (`python tools/check_no_raw_data.py` passes).
- [ ] **Tests green** — `ruff check .`, `black --check .`, `mypy rfbench`, and `pytest -q` all pass locally.

<!-- Tier-2 (verified) submissions must also include a reproduction manifest (submission.schema.json): code_commit, exact command, weights_url and/or docker_image, hardware, expected_metrics, tolerance. See docs/SUBMISSION.md. -->
