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
`code_commit` (`git@sha`), exact `command`, an `artifacts` block, `hardware`, `expected_metrics`,
`tolerance`. Without these, a result can only stay `self_reported`.

The `artifacts` block says how the maintainer obtains the runnable environment — **one of**:
- `weights_url` (+ `weights_checksum`) — a fetchable checkpoint, for the default *eval_only* re-run;
- `docker_image` — a pinned, digest-addressed image (for fragile old stacks);
- `artifacts.source_only: true` (schema **1.1.0**) — the result is fully reproducible **from source
  alone**: `code_commit` + the exact `command` + the committed canonical split indices + the pinned
  `uv.lock`, with no external artifact to fetch. This is the honest form for a **deterministic
  `from_scratch` seed baseline** (e.g. `hoc_lr`, `mean_snr`): the maintainer re-runs the command at
  `code_commit` (`rfbench data prepare` re-downloads the dataset) and matches within `tolerance`.

## Tier 3 — literature reference (never re-run by us)

Schema 1.1.0 adds two `verification.status` values for numbers nobody ran through `rfbench`: they
are copied from a model's own paper, hand-curated into a `result.json`, and exist purely so the
board can cite a public FM figure without pretending we reproduced it. Neither can be produced by
`rfbench eval` — they are PR-only, and CI never promotes a row into or out of these two states.

- **`from_paper`** — the paper's own dataset **and** the board's exact canonical split/protocol
  match (e.g. IQFM/WirelessJEPA reporting on RadioML 2016.10a, 11-class, full SNR −20…+18 dB —
  our exact AMC setting). The paper's precise sample indices are still usually unpublished, so
  sample-level overlap with our `canonical_split_id`/`checksum` is not guaranteed — only the
  dataset and evaluation protocol are confirmed identical.
- **`from_paper_uncertain`** — only the dataset **family** matches (same task, same class count,
  plausibly the same source), but the split, exact preprocessing, or provenance of the paper's
  data was **not** confirmed against our canonical split. Use this tier rather than guessing —
  presenting an unconfirmed match as `from_paper` repeats the mistake fixed in `a689e86` (a
  fabricated SEI row on the wrong dataset carrying an unearned board score).

Both tiers REQUIRE a `verification.note` citing the source (arXiv id) and spelling out exactly
what is/isn't confirmed; they never carry `verified_by`/`verified_date`/`verified_hardware` (no
re-run happened). See `docs/BIBLIOGRAPHY.md` A.5 for the current `from_paper*` rows.
