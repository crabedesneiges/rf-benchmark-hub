# Submission & verification

## Tier 1 — self-serve (for your paper)

```bash
rfbench eval <task> --model <name> --regime <regime>   # emits result.json
```

The number is yours to cite; it is marked `self_reported` on the board.

## Tier 2 — official (verified)
1. PR adds `leaderboard/results/<task>/<name>.json` (+ a `submission.schema.json` manifest).
2. **CI** checks: schema validity, regime declared, full-protocol conditions, no raw data committed.
3. **Maintainer** re-runs on a multi-GPU station:
   - *eval-only* if weights + Docker are provided (default), or
   - *full re-train* for seed baselines.
4. If the re-run matches within `tolerance`, `verification.status → verified` with
   `verified_by/date/hardware`.

## What makes a submission reproducible
`code_commit` (`git@sha`), exact `command`, `weights_url` and/or `docker_image`, `hardware`,
`expected_metrics`, `tolerance`. Without these, a result can only stay `self_reported`.
