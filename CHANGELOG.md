# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Sprint 0 — Bootstrap & contracts (M0). Scaffolds the repo, freezes the core contracts and
JSON schemas, and lands the normative docs and CI skeleton.

### Added

- **WP-00 — Repo & packaging.** `pyproject.toml` defining the `rfbench` package (at repo root,
  not `src/`) and the `rfbench` CLI entrypoint; pre-commit with `ruff` + `black` (line-length 100);
  Apache-2.0 `LICENSE`.
- **WP-01 — JSON schemas.** Frozen `schemas/result.schema.json` (one evaluation run = one
  leaderboard row; regime always declared; `verification.status` ∈ `{self_reported, verified}`)
  and `schemas/submission.schema.json` (reproducibility manifest), with valid/invalid examples
  under `schemas/examples/`.
- **WP-02 — Core contracts (ABCs).** `rfbench/core/{task,dataset,metric,model,registry,splits,`
  `evaluate,manifest}.py`: typed interfaces with docstrings and minimal bodies; ABCs are not
  instantiable.
- **WP-03 — Normative docs.** `docs/EVALUATION_PROTOCOL.md`, `docs/SUBMISSION.md`,
  `docs/ARCHITECTURE.md`, plus `README.md` and `CONTRIBUTING.md`. Each task has a defined
  metric + split + regimes; the two-tier submission workflow is described end to end.
- **WP-04 — CI skeleton.** `.github/workflows/ci.yml` running lint, unit tests, and schema checks.

### Notes

- **No raw data in git** (D3): only split indices + checksums are versioned under
  `leaderboard/splits/`; datasets are fetched via `rfbench data prepare` and honour `$RFBENCH_CACHE`.
- **Frozen contracts:** the core ABCs and JSON schemas are locked at Sprint 0; changing them
  requires an explicit review and a version bump.
- Scope is terrestrial RF only (D1); satellite RF is a separate repository.

[Unreleased]: https://github.com/rf-benchmark-hub/rf-benchmark-hub/commits/main
