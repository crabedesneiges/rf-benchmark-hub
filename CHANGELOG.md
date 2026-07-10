# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — tier `verified` (D4) : premières lignes officiellement re-vérifiées

Le board a ses **3 premières lignes `verified`** (jusqu'ici 25/25 `self_reported`). Le mainteneur a
re-joué les 3 baselines AMC déterministes et flippé leur statut via la machinerie `rfbench verify`
(WP-53), avec provenance estampillée :
- `hoc_lr`, `majority_class`, `chance` (amc, `from_scratch`) → **`verified`**,
  `verified_by=rf-bench-maintainers`, `method=full_retrain`, re-run **bit-identique** aux valeurs
  committées (|Δ|=0 ≤ tolérance amc ±0.005).

Pré-requis livrés :
- **`submission.schema.json` 1.1.0** (additif, 1.0.0 toujours valide) : nouvelle forme d'artefact
  `artifacts.source_only: true` — une baseline déterministe `from_scratch` est reproductible depuis
  les sources (`code_commit` + `command` + splits committés + `uv.lock`) sans poids ni image
  externes, la forme honnête pour un projet from-source (D3). `docs/SUBMISSION.md` documenté.
- **Manifests de vérification** `leaderboard/manifests/amc/{hoc_lr,majority_class,chance}.json`
  (`expected_metrics` = valeurs committées, `tolerance.absolute=0.005`, `rerun_mode=full_retrain`).
- Re-run officiel via `slurm/train_hoc_amc.sh` (CPU/defq), flip `rfbench verify --rerun --out`.

Le site affiche le badge `verified` sur ces 3 lignes. Les baselines déterministes SNR
(`mean_snr`/`snr_moment_ridge`) sont le lot suivant, bloqué par l'absence de `snr_estimation` dans
les enums de `submission.schema.json` (PR séparée) + une tolérance à fixer au protocole.

### Added — page « Methods » : explication cliquable et fidèle à l'implémentation par méthode

Chaque méthode **sans papier** du board a désormais un nom **cliquable** qui pointe vers une page
`methods.html` expliquant son fonctionnement, générée **verbatim depuis la docstring de la classe
du modèle** (donc fidèle à l'implémentation par construction) — les méthodes **à papier** gardent
leur lien externe (arXiv/DOI). Les explications sont extraites par **`ast`** (aucun import de
torch/numpy, donc le build de site reste dépendance-free), avec un lien vers le fichier source.

- `leaderboard/site/generate.py` : `_extract_method_docs()` (parse `rfbench/models/**/*.py`, résout
  `@register_model("x")` ET `@register_model(CONST)`), `render_methods_page()`, `_render_docstring()`
  (sous-ensemble reStructuredText sûr : paragraphes, listes à puces avec continuations, `` `code` ``,
  rôles `:class:`/`:meth:` → `<code>`, `**gras**`/`*emph*`, tout HTML-échappé), routage du nom
  (`model.url` externe sinon `methods.html#<nom>`), onglet **Methods** dans la nav.
- Couvre les 10 méthodes maison/reproductions (`mean_snr`, `snr_moment_ridge`, `hoc_lr`,
  `majority_class`, `chance`, `cldnn`, `interf_cnn`, `complex_cnn`, `resnet1d_sei`, `wisig_cnn_paper`).
- Tests : extraction, rendu docstring (puces/code/rôles, pas de fuite rst), linking end-to-end
  (sans-papier → `methods.html#`, papier → externe).

Enrichissement (archi + références) : chaque entrée Methods montre la **description d'architecture**
(la docstring **module** — qui porte la citation papier + le rationale d'archi — préposée à la class
docstring pour un fichier à modèle unique) et une ligne **« Paper / references »** avec les liens
**arXiv/DOI cliquables** (parsés du docstring + `result.json model.url`, dédupliqués) ; les arXiv/DOI
sont aussi rendus cliquables **inline**. Une méthode sans papier (non-deep : `hoc_lr`, `mean_snr`,
`snr_moment_ridge`, planchers) affiche une note explicite et sa description exhaustive. Les **8 liens
papier ont été vérifiés** (WebFetch : titre/auteurs/venue concordent — West&O'Shea/DySPAN,
GLOBECOM'19, WiSig Access'22, MCLDNN WCL'20, ORACLE INFOCOM'19, T-PRIME, etc.).

### Added — J4 complété : premières lignes de board `snr_estimation` (baselines de régression)

La tâche `snr_estimation` (livrée sans score) a désormais ses baselines seed et sa page de
leaderboard. Deux modèles (`rfbench/models/baselines/snr_regressors.py`), sur le split test
RadioML 2016.10a (plage SNR complète −20…+18 dB, from_scratch, seed 42, CPU) :
- **`mean_snr`** — plancher "zero-rule" (prédit la moyenne SNR du train, pure stdlib) :
  **rmse_db 11.53 / mae_db 10.00** (le RMSE = l'écart-type de la distribution SNR).
- **`snr_moment_ridge`** — réf DSP : 6 features d'enveloppe scale-invariantes (kurtosis M4/M2²,
  M6/M2³, PAPR, cov d'enveloppe, |C20|/M2, |C42|/M2²) + Ridge standardisé (analog de `hoc_lr`) :
  **rmse_db 7.64 / mae_db 5.83** — bat le plancher de ~3.9 dB RMSE.

lower-is-better ; déterministes single-seed. Job SLURM CPU `slurm/train_snr_baselines.sh`. Tests
pure-Python dont un **end-to-end `evaluate()` du plancher** (RMSE == std, `result.json` schéma-valide),
la couverture qui manquait aux baselines SEI. La page `snr_estimation` du site est enfin peuplée.

### Added — J2 baselines DSP AMC + J4 tâche régression `snr_estimation` (reprise 2026-07-09)

Reprise d'un WIP de session interrompue (jalons J2 + J4), câblé, vérifié (ruff/black/pytest verts)
et committé sur `claude/ecstatic-torvalds-a6ced8`.

- **J2 — baselines classiques AMC.** `hoc_lr` (cumulants d'ordre supérieur + `LogisticRegression`
  seed-42, même estimateur que la head logreg normative) comme référence DSP, et deux planchers
  triviaux `majority_class` / `chance` (stdlib pur). Enregistrés dans `MODELS` ; hors table CLI
  `_MODEL_MODULES` by-design (le job SLURM `slurm/train_hoc_amc.sh`, CPU/defq, les instancie
  explicitement). **Lignes de board produites** sur RadioML 2016.10a (plage SNR complète,
  from_scratch, seed 42) : `hoc_lr` 0.2629, `majority_class` 0.0909 (=1/11), `chance` 0.0895 —
  déterministes single-seed (IC bootstrap par seed conservé). Étage le board sous les baselines
  deep (resnet/cldnn/mcldnn 0.57–0.62).
- **J4 — tâche de régression `snr_estimation`** (raw-IQ → SNR dB) sur RadioML 2016.10a :
  `Task`/`Dataset`/métriques `rmse_db` (primary) + `mae_db` (**lower-is-better**), split
  `snr-radioml2016-strat-snr-8010-seed42-v1` (indices byte-identiques au split AMC, dérivés, id
  propre), config Hydra. **Extension additive des contrats figés** (précédent
  `interference_id`/`protocol_tech_id`) : `snr_estimation` ajouté à l'enum `task.name`
  (`result.schema.json`) et au `Literal TaskName` — additif, `task.version=v1`, `schema_version`
  inchangé. **CLI** câblée (`TASK_NAMES`/`_TASK_MODULES`/`_TASK_DEFAULTS`), `rfbench eval
  snr_estimation` sélectionnable ; le split SNR est dérivé du split AMC (pas de cible `data prepare`
  dédiée, documenté). **Site** : rendu lower-is-better (tri ascendant + barre inversée) + badge de
  contamination (`pretraining.overlap_with_eval`). Protocole normatif acté
  (`docs/EVALUATION_PROTOCOL.md` : track canonique `all_snr`, plage SNR complète, pas de
  cherry-picking). Tests : workaround `task.name='amc'` retiré + test end-to-end de rendu de page.

### Added — J3 : colonne SEI mergée sur la branche d'intégration + 9 lignes de board

`feat/sei-complete` mergée dans `claude/ecstatic-torvalds-a6ced8` (merge `--no-ff`, 2 conflits docs
résolus : CHANGELOG + BIBLIOGRAPHY ; 36 fichiers auto-mergés). Apporte l'implémentation SEI
paper-faithful (voir l'entrée « SEI benchmark column » ci-dessous) ET les **9 lignes de board**
post-fix Keras-fidelity (`wisig_cnn_paper`/`complex_cnn`/`resnet1d_sei` × closed_set/cross_receiver/
cross_day, validées 9/9 contre le schéma). Les splits WiSig sont byte-identiques entre les deux
branches (même `split_checksum`), donc les résultats attestent bien les splits d'intégration.

### Added — J3b : piste open-set WiSig (held-out Tx, AUROC/EER) de bout en bout

Le track `open_set` (jusque-là un stub non branché) est implémenté et scoré. **Design** (choisi
avec l'utilisateur) : détection d'émetteurs inconnus — ~80% des Tx forment la galerie connue, 20%
sont held-out comme impostors ; le modèle est fit en identifieur `|known|`-classes, le **score =
max-softmax probability (MSP)**, et l'**AUROC (primary) / EER** séparent genuine (in-gallery) des
probes novel. Split canonique `sei-wisig-openset-heldouttx-8010-seed42-v1` (511 515 records →
326 180/40 779/144 556) ; genuine/impostor **dérivé** par le dataset (`tx ∈ train`), non stocké.
Protocole normatif acté (`docs/EVALUATION_PROTOCOL.md` §SEI). **3 lignes de board** (from_scratch,
seed 42, GPU) : `resnet1d_sei` **0.822** AUROC (meilleure), `complex_cnn` 0.658, `wisig_cnn_paper`
**0.498** (~hasard, CI serré : bon en closed-set mais ne rejette pas les inconnus — un vrai constat).

Trois bugs révélés par le run cluster et corrigés (invisibles aux tests pure-Python, désormais
gardés en régression + un test end-to-end de l'éval open-set) : `match_score` appelait `.item()`
sur une ligne 1-D (tensor multi-éléments → lève) ; `eer()` était O(n²) (→ `bisect`, O(n log n)) ;
le bootstrap re-réduisait les 144 k lignes 1000× (→ hook `Metric.prepare_predictions`, réduction
1×). Éval open-set : ~minutes au lieu d'heures.

### Added — Phase 0 quality hardening: schema 1.2.0, protocol lock-in, bootstrap CI, repro ops

Six-block Phase 0 of the 2026-07 quality audit follow-up. Priority is repo quality, not public
launch (domain/governance/maintainers/paper work stays out of scope for now); no FM pretrain was
relaunched — schema and protocol are locked in spec-only.

- **Schema 1.2.0** (`schemas/result.schema.json`, `schemas/submission.schema.json`, additive,
  non-breaking): new optional fields `metrics.uncertainty` (per-metric CI, `method` ∈
  `{bootstrap_percentile, wilson_backfill, multi_seed_std}`), `pretraining`
  (`pretrain_datasets`/`overlap_with_eval`/`disclosure_note`), `transfer` (`source_dataset`,
  `source_domain`), `efficiency` (latency/throughput/FLOPs/memory/GPU-hours). Fixed both schemas'
  `$id` to the real org (`crabedesneiges/rf-benchmark-hub`, was `rf-benchmark-hub/rf-benchmark-hub`).
- **Protocol lock-in** (`docs/EVALUATION_PROTOCOL.md`): new normative "Statistical rigor &
  uncertainty" section — bootstrap percentile CI default (n=1000, confidence=0.95), Wilson backfill
  restricted to proportion metrics on non-`from_paper*` rows, few-shot k∈{1,10,100} with N≥10
  episodes (seeds 42..51), scikit-learn logistic regression as the normative probe head
  (nearest-centroid is a dependency-free fallback only, never for board numbers), COCO-style
  IoU-averaged mAP for `wideband_detection` (existing `task.py` still single-IoU @ 0.5 — not yet
  updated to match, tracked as follow-up), calibrated-on-val pd@pfa=0.1 for `spectrum_sensing` (no
  implementation yet), per-task tolerance table, contamination-disclosure rules.
- **Bootstrap CI + backfill** (`rfbench/core/evaluate.py`, `scripts/backfill_uncertainty.py`):
  `evaluate()` now accumulates per-chunk predictions and computes a stdlib-only percentile bootstrap
  CI (`compute_bootstrap_ci=True` by default, 1000 resamples, ~9.5s @ 22k samples) into
  `metrics.uncertainty`; `SCHEMA_VERSION` bumped to `1.2.0`. New `backfill_uncertainty.py` computes
  Wilson-interval CIs for 5 existing self_reported/verified board rows lacking raw predictions
  (`amc/{cldnn,mcldnn,resnet_amc,iqfm-base-linear_probe}`, `interference_id/interf_cnn}`) —
  `from_paper*` rows are structurally excluded (no `eval` block). Leaderboard site (`_sort_rows`)
  keeps its existing strict ordering (primary DESC → verified-first → name) unchanged — CI overlap
  is not transitive, so it is surfaced as a non-reordering `≈`-overlap annotation instead of
  re-ranking on statistical noise.
- **Regimes** (`rfbench/regimes/heads.py`, `few_shot.py`): new `LogisticRegressionHead` (lazy
  sklearn import, matches the new normative probe spec) wired into linear-probe/few-shot
  instantiation in `rfbench/models/foundation/base.py::run_regime` only (never for
  from_scratch/full_finetune), falling back to `NearestCentroidHead` with a warning if sklearn is
  absent. New `run_episodic()` helper for N≥10-episode few-shot runs (not yet wired into the CLI).
- **Repro ops** (`rfbench/training.py`, `slurm/*.sh`, `uv.lock`): checkpoints are now actually
  persisted to disk (`train_baseline(..., checkpoint_out=...)`, atomic write, `--out-checkpoint`
  CLI flag) — previously best-checkpoint restore only logged and never called `torch.save`, making
  bootstrap/re-scoring of `from_scratch` rows impossible without a full retrain. Added
  `cudnn.deterministic=True`/`cudnn.benchmark=False`. Generated `uv.lock` (was missing). Audited all
  14 `slurm/*.sh`: confirmed the audit report's suggested `--constraint=arm` is **wrong for this
  cluster** (single partition `defq*`, feature `location=local`, no `arm` feature exists) — documented
  instead of applied.
- **Licenses & URLs** (`docs/LICENSES.md`, `pyproject.toml`, `README.md`): new 10-dataset license
  matrix sourced from existing repo docs (RadDet's Kaggle-vs-GitHub CC BY-NC vs CC BY-NC-SA
  divergence flagged rather than smoothed over; ORACLE/LoRa RFFI/DeepSense marked unconfirmed).
  Fixed stray `rf-benchmark-hub/rf-benchmark-hub` GitHub URLs to `crabedesneiges/rf-benchmark-hub`.

Open follow-ups tracked for later milestones: `verification.tolerance` shape still differs between
`result.schema.json` (scalar) and `submission.schema.json` (structured object) — left as-is, not
reconciled; `wideband_detection` mAP implementation vs. new normative spec; `spectrum_sensing` has
no code yet; J1 (GPU multi-seed ×3 + real bootstrap on fresh runs + iqfm-base re-score under logreg
probe) is next.

### Changed — FM in-repo reproduction PAUSED; `iqfm-base` / `wireless-jepa` documented as homemade

Decision (2026-07): pause the in-repo *reproduction* of IQFM and WirelessJEPA. Both papers publish
**no weights** and pre-train on **proprietary OTA testbeds we do not have** (IQFM: OTA MIMO;
WirelessJEPA: multi-antenna MIMO testbed, 7 waveforms / 225 AoA / USRP X300), so an in-repo retrain
can only ever be a **homemade, in-distribution** model — not the paper's — which makes chasing the
papers' OOD numbers moot. The wrappers + shared `shufflenet1d.py` backbone are **kept** for future
use; further retraining is on hold.

- **Web-verified the WirelessJEPA architecture** (arXiv:2601.20190v1) and found the prior in-repo
  description was wrong: it is **multi-antenna** (2D antenna-time grid `2×256×256`, 4 antennas), a
  ShuffleNetV2-x0.5 **2-D** encoder, **spatio-temporal** mask geometries, a **depthwise-separable
  conv predictor**, and an **L2 loss over masked-region latents**; RML2016.10a is only a downstream
  **OOD eval** (74.78% @ 500-shot LP). Our `wireless-jepa` wrapper is a **single-antenna 1-D**
  JEPA-style model (pooled-latent target + homemade VICReg anti-collapse) — inspired by, not a
  reproduction of, the paper. Docstring updated to say so prominently; **no `wireless-jepa` board
  row committed** (the first cluster run collapsed to chance 9.09% — pooled-latent JEPA collapse —
  and the run is now paused, so the collapsed artifact was removed, not committed).
- **`iqfm-base`**: the committed board row (`leaderboard/results/amc/iqfm-base-linear_probe.json`,
  48.87% linear-probe, self_reported, PR-ready) is kept as an honest **homemade, in-distribution**
  number and is clearly separate from the paper's OOD **38.1%** (`from_paper` row). Docstring notes
  the pause. No further retraining.
- **Paper numbers stay in the bibliography, not as fabricated board rows.** Verified exact figures
  for the record (both papers, 500-shot linear-probe OOD unless noted): WirelessJEPA — RML2016.10a
  74.78% (kNN 68.01), POWDER fingerprinting 90.45% (kNN 87.82), GNSS jamming 63.15% (kNN 48.85),
  WiFi protocol 94.26% (kNN 84.89), 5G NR interference 76.27% (kNN 64.52), in-dist testbed mod
  99.98% / AoA 99.87% @ 100-shot; IQFM — RML2016.10a 38.1% @ 50-shot LP (also 50.0% @ 500-shot
  LoRA), POWDER fingerprinting 96.05% @ 500-shot LoRA, DeepBeam beam-prediction 94.1% @ 500-shot
  LoRA, testbed AoA 65.45% @ 1-shot. Only figures whose dataset matches a **committed split** get a
  board row (`amc` ×2, `interference_id` GNSS); the rest (POWDER, WiFi, DeepBeam, 5G-NR) have no
  committed dataset/split and remain bibliography-only — same no-dataset reason the retrain is
  paused.

### Added — literature-reference verification tiers (`from_paper` / `from_paper_uncertain`)

`result.schema.json` bumped **1.0.0 → 1.1.0** (additive, non-breaking; old rows and writers keep
validating unchanged — `schema_version` is now an enum of both) to add two `verification.status`
values for numbers copied from a model's own paper and hand-curated into a `result.json`, never
run through `rfbench`: **`from_paper`** (paper's dataset AND our exact canonical split/protocol
match) and **`from_paper_uncertain`** (only the dataset family matches; split/protocol overlap
with our canonical setting is unconfirmed). Documented in `docs/SUBMISSION.md` ("Tier 3").

- `leaderboard/site/generate.py`: new `badge-paper` / `badge-paper-uncertain` badge styles (light +
  dark), verified > self_reported > from_paper > from_paper_uncertain trust order for same-score
  tie-breaks, and an expanded Guide explanation of all four tiers.
- Three new board rows: `leaderboard/results/amc/{iqfm,wirelessjepa}_paper.json`
  (`from_paper` — IQFM 38.1% @ 50-shot, WirelessJEPA 74.78% @ 500-shot, both linear-probe OOD on
  RadioML 2016.10a full SNR, our exact AMC setting) and
  `leaderboard/results/interference_id/wirelessjepa_paper.json` (`from_paper_uncertain` — 63.1% on
  a GNSS-jamming 6-class set that only plausibly matches our `interf_gnss6` Zenodo source; split
  overlap unconfirmed). No `protocol_tech_id` row: WirelessJEPA's 94.26% figure is board-relevant
  but there is no canonical split committed under `leaderboard/splits/` yet to cite honestly.
- `docs/BIBLIOGRAPHY.md` A.5: IQFM and WirelessJEPA paragraphs now list, per canonical task, every
  figure each paper reports downstream (not just the board-comparable AMC number), with an
  explicit board-row/no-board-row verdict and why.

### Added — WirelessJEPA raw-IQ JEPA foundation-model wrapper (`wireless-jepa`)

Adds WirelessJEPA (arXiv:2601.20190) as an evaluable board FM **without touching the frozen core**
(`rfbench/core/`, `schemas/`, `rfbench/regimes/`, `evaluate()` all unchanged) — a pure new wrapper
per `docs/ADDING_A_MODEL.md`, **stacked on the IQFM wrapper** whose shared backbone it reuses.
WirelessJEPA's 74.78% (RML2016.10a, 11-cls, −20…+18 dB, 500-shot linear probe) is the single most
board-comparable public FM number and **beats our supervised MCLDNN (61.71%)**. Phase 1 (wrapper +
tests) is CPU-only and mergeable; Phase 2 (JEPA pre-training + eval) ships as cluster scripts, NOT
yet run.

- **Wrapper** `rfbench/models/foundation/wireless_jepa.py`: `WirelessJepa(FoundationModel)`,
  `@register_model("wireless-jepa")`, no positional args. **Reuses IQFM's shared
  `build_shufflenet1d()`** (the paper's "matched to IQFM" — same ShuffleNetV2-x0.5, 335,096 params)
  as the frozen JEPA context/target encoder; `embed()` = frozen `(B, 1024)` features with unit-max
  input norm (adopted as the matched-family convention; WJEPA's exact input norm is unpublished,
  flagged). No task head → inherits `forward()` (probing only). Same honesty guards as IQFM /
  LWM-Spectro: no checkpoint → random init + `pretrained=False` + warning; checkpoint present but
  keys don't match → RAISES. Loads the **EMA target encoder** (the representation the paper probes).
- **Registration** re-exported from `rfbench/models/foundation/__init__.py` (import stays
  dependency-free; torch loads only on first `embed`).
- **HONESTY / provenance.** WirelessJEPA's weights are **not published**; the 74.78% is an **OOD**
  figure (pre-trained on the authors' OTA MIMO testbed we do NOT have). We reproduce only the
  *recipe*: `scripts/pretrain/wireless_jepa.py` + `slurm/pretrain_wireless_jepa_arm.sh` (re-)pre-train
  the shared backbone with **JEPA** — masked-latent prediction + EMA teacher (0.996→1.0 cosine),
  **no data augmentation** (contiguous time-block masking is the only SSL signal) — on RadioML
  2016.10a **train** delabelised (seed 42), **in-distribution, NOT the paper's OOD setting**. The
  exact masking recipe is unpublished → documented approximation. Any resulting score is **ours**,
  labelled as such, and **never** presented as 74.78%. **No `result.json` committed** — the board
  row waits on a real cluster run.
- **Tests** `tests/test_wireless_jepa.py`: dep-free (package import registers `wireless-jepa`, cheap
  construction, cache-path helpers) + torch-gated (`embed` → `(B, 1024)`, param count = shared
  backbone, unit-max applied, missing-checkpoint flips `pretrained`, non-matching checkpoint raises,
  **shares-IQFM-backbone** round-trip). Verified on CPU torch: all 10 pass; the JEPA loop
  (mask → context/target encode → predictor → smooth-L1 → EMA update) runs, back-props to the
  context encoder only, and moves the target via EMA. Dep-free suite green; `ruff` + `mypy` (strict,
  73 files) clean.
- **Docs** `docs/BIBLIOGRAPHY.md` (§A.5 + §C.5 status → wrapper implemented) +
  `docs/SOTA_REFERENCE.md` (FM inventory row) updated.

### Added — IQFM raw-IQ SSL foundation-model wrapper (`iqfm-base`)

Adds IQFM (Mashaal & Abou-Zeid, arXiv:2506.06718v2, CC-BY 4.0) as an evaluable board FM **without
touching the frozen core** (`rfbench/core/`, `schemas/`, `rfbench/regimes/`, `evaluate()` all
unchanged) — a pure new wrapper per `docs/ADDING_A_MODEL.md`. This is the real-terrestrial-task FM
the LWM-Spectro follow-up (below) redirected the FM-vs-baseline thesis toward. Phase 1 (wrapper +
backbone + tests) is CPU-only and mergeable on its own; Phase 2 (SimCLR pre-training + eval) ships
as cluster scripts, NOT yet run.

- **Reusable 1-D backbone** `rfbench/models/foundation/shufflenet1d.py`: a faithful `Conv1d`
  transcription of ShuffleNetV2-x0.5 (Ma et al., ECCV 2018) over raw IQ `(2, L)` →
  1024-D mean-pooled embedding, **no classifier**. Measured **335,096 params** (the small delta
  from IQFM's reported ~341k is the expected 1-D-vs-2-D difference). Torch imported lazily via
  `require_torch()`; `build_shufflenet1d()` is the shared encoder the forthcoming WirelessJEPA
  wrapper (arXiv:2601.20190, "ShuffleNetV2-x0.5 matched to IQFM") will reuse.
- **Wrapper** `rfbench/models/foundation/iqfm.py`: `IqfmBase(FoundationModel)`,
  `@register_model("iqfm-base")`, constructible with no positional args. `embed()` = frozen
  `(B, 1024)` features with IQFM's **unit-max** input norm `iq/max(|iq|)` applied per sample;
  `n_params` from the loaded backbone; checkpoint loads lazily. No task head → inherits `forward()`
  (probing only). Same honesty guards as LWM-Spectro: **no checkpoint → random init +
  `pretrained=False` + loud warning**; **checkpoint present but keys don't match → RAISES** (never
  scores a partly-random encoder as pretrained).
- **Registration** re-exported from `rfbench/models/foundation/__init__.py`, so
  `import rfbench.models.foundation` registers `iqfm-base` (import stays dependency-free — torch
  loads only on first `embed`). `slurm/eval_fm_arm.sh` now also imports the package so the generic
  `[MODEL]` path reaches `iqfm-base`.
- **HONESTY / provenance.** IQFM's weights are **not published**; the paper's **38.1%** is a
  linear-probe/50-per-class/**OOD** figure (pre-trained on the authors' OTA MIMO testbed we do NOT
  have). We reproduce only the *recipe*: `scripts/pretrain/iqfm_simclr.py` +
  `slurm/pretrain_iqfm_arm.sh` (re-)pre-train the backbone with SimCLR/InfoNCE on the RadioML
  2016.10a **train** split delabelised (seed 42) — **in-distribution, NOT the paper's OOD setting**.
  Documented augmentations: circular time shift, additive Gaussian noise, global phase rotation.
  Any resulting score is **ours**, labelled as such, and **never** presented as the 38.1%. **No
  `result.json` is committed** — the board row waits on a real cluster run.
- **Tests** `tests/test_iqfm.py`: dep-free (package import registers `iqfm-base`, cheap
  construction, cache-path helpers) + torch-gated (backbone ~335k params, `embed` → `(B, 1024)`,
  unit-max applied, missing-checkpoint flips `pretrained`, non-matching checkpoint raises,
  round-trip checkpoint loads). Verified on CPU torch: all 12 pass; the SimCLR loop
  (augment → backbone+head → NT-Xent → backward) runs and back-props to the backbone. Dep-free
  suite stays green; `ruff check .` + `mypy` (strict, 72 files) clean.
- **Docs** `docs/BIBLIOGRAPHY.md` (IQFM status → wrapper implemented) + `docs/SOTA_REFERENCE.md`
  (FM inventory row) updated.

### Verified — LWM-Spectro integration validated on its own task; no AMC board row (WP-62)

On-cluster follow-up to the LWM-Spectro faithful-wrapper fix. Definitive outcome: the integration and
the model are **verified**, and — because the paper defines **no** RadioML/AMC task and ships **no**
IQ→spectrogram preprocessing — LWM-Spectro gets **no AMC/RadioML board row** (inventing one would be
dishonest). See `docs/BIBLIOGRAPHY.md` §B.5.

- **Real weights load bit-exact.** The encoder is loaded from `experts/{WiFi,LTE,5G}_expert.pth` (the
  real 12-layer LWM encoders; `checkpoints/checkpoint.pth` is the `snr_mobility` MoE bundle, not an
  encoder). `load_state_dict` reports `missing=0` — every one of the 201 encoder tensors matches by
  name and shape. The prior "22.74%" board row was produced by the earlier broken encoder that loaded
  **zero** weights and was already removed.
- **Paper's own task reproduced — 93.9%.** The released MoE ckpt targets joint SNR/mobility
  recognition (`snr_mobility`, 14 classes). Reconstructing their exact classifier head (`Res1DCNNHead`
  + `LayerNorm`, loaded `missing/unexpected=[]`) and running it on the shipped `demo_data_moe.pt`
  (10 500 labelled 128×128 spectrograms + reference embeddings) reproduces **93.9%** accuracy
  (pedestrian 96.6 / vehicular 91.2), matching the paper's Table II (94.4% @100-shot, 95.1% @400-shot);
  a logreg cross-check on the shipped embedding gives 92.6%.
- **Encoder correlation check.** Our reconstructed encoder fed the shipped spectrograms yields a
  **0.57 cosine** to the reference `moe_embedding` (vs ~0 random) — substantively correct; the residual
  gap to bit-exact is the precise embedding-extraction/MoE-combine recipe (out of scope).
- **RadioML AMC stays off the board.** The corrected encoder on RadioML 2016.10a lands at ~chance
  (linear_probe 16.6% / few_shot 14.1%) because the IQ→spectrogram front-end is unpublished and thus
  off-distribution; these are **provisional / UNVERIFIED and not published**. The FM-vs-baseline board
  thesis is redirected to an FM with a real terrestrial task (WirelessJEPA / IQFM — `docs/NEXT_STEPS.md`
  §6). Docs (`BIBLIOGRAPHY.md` §B.5 + audit summary, `NEXT_STEPS.md`) updated to this conclusion.

### Fixed — board/manifest/doc integrity audit (post-redesign)

Self-review pass after the (solo, unreviewed) site redesign — the intended multi-agent
adversarial audit could not run (subagents blocked on the account's monthly spend limit), so this
was done in the main loop and is flagged as such.

- **`interference_id` promoted `wip` → `implemented`** in `leaderboard/tasks.json`: it has a real
  board result (`interf_cnn` 0.9987) but was still declared work-in-progress, so the live site
  mislabeled a working leaderboard as WIP. Blurb/description updated (dropped "baseline pending
  cluster training"). Homepage "Implemented" stat now reads 4.
- **New honesty guard** `tests/test_committed_manifest_status_matches_committed_results`: any task
  with a committed `result.json` must be declared `implemented` (would have failed on the bug
  above; prevents the staleness class from recurring).
- **`docs/NEXT_STEPS.md` rewritten** to the real 2026-07-06 state: P0 marked done (ResNet fixed,
  board cleaned, recipe done, CLDNN collapse resolved 0.5805), `interference_id` 0.9987 landed,
  SEI/FM flagged in-flight, site redesign + Pages-deploy-flaky noted, roadmap re-prioritised.
- **`README.md` badge links fixed**: CI badge pointed at the stale template org
  (`rf-benchmark-hub/rf-benchmark-hub` → `crabedesneiges/rf-benchmark-hub`); the Leaderboard badge
  link was missing its `https://` scheme (rendered as a broken relative link on GitHub).
- Verified (no change needed): all 4 committed result files still schema-valid; HTML tag-balance
  clean on all 15 generated pages; the homepage filter JS reads attributes the HTML actually
  renders (`data-status`/`data-filter` sets match); `docs/BIBLIOGRAPHY.md` "Our score" already
  current (61.71/58.05/56.61); no stale "no runtime JS/CDN" doc claims remain.

### Added — SEI benchmark column: paper-faithful WiSig 2-D CNN, ORACLE + SOTA baselines, POWDER track

The SEI task and WiSig loader existed but the board had **no SEI rows** (fabricated lines removed,
`a689e86`) and the only model, `wisig_cnn`, was a compact 1-D CNN that does **not** reproduce the
paper. This lands the real SEI column. Preceded by a verbatim-code Phase-0 audit (the official WiSig
`master` branch, both FM papers, and the SOTA literature) — the REPO/primary source is authoritative,
and several `docs/BIBLIOGRAPHY.md` claims were **corrected** (below).

- **`wisig_cnn_paper` — byte-faithful WiSig ManyTx 2-D CNN** (`rfbench/models/baselines/wisig_cnn_paper.py`).
  Reconstructs `create_net` in `py/d006_ManyTx_ntx.py` exactly: `(256,2)`→`Reshape(256,2,1)`→conv
  8/16/16/32/16, kernels (3,2)×3 then (3,1)×2 `same`+ReLU, **only 4 max-pools** (the 5th conv is
  **unpooled**) →Flatten(256)→Dense(100)→Dense(80)→Dropout(0.5)→Dense(N). Keras **`same`** padding
  reproduced with the trailing-edge asymmetry (torch's `'same'` pads the leading edge); **L2 λ=1e-4 on
  the three Dense kernels ONLY** (via `l2_penalty()`, added to the loss — Keras-exact, not coupled
  `weight_decay`); per-signal **unit-average-power** normalisation folded into the model (scale-invariant
  logits, unit-tested). The compact 1-D `wisig_cnn` stays as a documented board-seeding variant.
- **`oracle_cnn`** (Sankhe et al. INFOCOM 2019, arXiv:1812.01124): Conv 50@(1×7) + Conv 50@(2×7) + FC
  256/80 + softmax, `2×128` raw IQ, Adam 1e-4, dropout 0.5, L2 1e-4, patience 10. (Default per-signal
  input norm on; the paper's exact scaling is under-specified — `input_norm=False` ablation provided.)
- **SOTA-leaning baselines (screened, 2 retained):** **`complex_cnn`** — faithful
  `network_20_modrelu_short` (Gopalakrishnan/Cekic/Madhow GLOBECOM 2019, arXiv:1905.09388; MIT repo
  `metehancekic/wireless-fingerprinting`): complex-multiply `ComplexConv1d` + Trabelsi **modReLU** →
  magnitude → GAP → Dense, the biggest inductive-bias contrast (phase-coupled) to the real-valued CNNs;
  and **`resnet1d_sei`** — a ResNet-18-1D over raw IQ (Jian et al. IoT-Mag 2020; He et al. 2016), the
  depth axis. Both raw-IQ, reproducible, registered + CLI-reachable. (Deferred with rationale:
  Al-Shawabka 2020 is a channel *study* not a packaged model; triplet/contrastive works lack runnable
  public code on WiSig/ORACLE — see `docs/BIBLIOGRAPHY.md` C.2.)
- **`balanced_accuracy` secondary metric** (mean per-class recall, pure-stdlib) alongside primary
  `rank1_accuracy` on the SEI closed-set tracks — the class-balanced accuracy the WiSig paper reports for
  the imbalanced ManyTx set. Additive (does not change the ranking key); no schema bump.
- **Dedicated SEI training loop** `rfbench/training_sei.py` (the shared AMC `rfbench/training.py` is
  **UNTOUCHED**, per constraint): class-weighted CE reproducing Keras' `class_weight=max(count)/count`
  semantics exactly (`Σ w·CE / N`), explicit L2 via the model's `l2_penalty()`, best checkpoint + early
  stop on **val_loss** (WiSig recipe, not the AMC loop's val-accuracy), and the SEI `(window,2)`
  time-major layout. A `rfbench sei-train --track {closed_set,cross_receiver,cross_day}` CLI subcommand
  threads the track into `evaluate` so the three conditions are scored as **SEPARATE** rows; a fixed
  `_InMemorySplit.__getitem__` makes the split map-style so the DataLoader works (also fixes the cluster
  path). Baselines added to the CLI `_MODEL_MODULES` dispatch (eval-reachable).
- **POWDER track (FM-comparable, download-blocked).** Identified the exact dataset both FM SEI evaluators
  use — **POWDER RF Fingerprinting** (Reus-Muns et al., *IEEE GLOBECOM 2020*; 4-BS WiFi), NOT
  Gaskin/Tractor. `rfbench/data/download/sei_powder.py` (+ `prepare`/loader/task wiring): the DRS record
  is public **without** POWDER/Emulab credentials (Handle `2047/D20385049` → `neu:gm80mp276`) but the host
  **anti-scrapes** programmatic clients (HTTP 403, not defeated by a browser UA), so the downloader raises
  a precise **manual-download** procedure and the split (`closed_set`, 256-frame, stratified by device) is
  built only once the SigMF captures are placed under `$RFBENCH_CACHE/powder/`. Indices/checksums only,
  never raw IQ (D3). FM references kept regime-separated (linear-probe 90.5/83.4 vs LoRA 96.05).
- **BIBLIOGRAPHY corrections (REPO is truth).** §A.3 WiSig: **L2 is on the 3 Dense layers only** (not
  conv); there are **4 pools** (5th conv unpooled); best weights via `ModelCheckpoint`+`load_weights` (no
  `restore_best_weights`); DOI is **10.1109/ACCESS.2022.3154790**; code repo is **BSD-3** (dataset CC
  BY-NC-SA); and the **99%→<33% cross-rx headline is a ManyRx/equalized/single-day experiment, NOT
  ManyTx** (which pools all rx/days for ~53%/~80%). §A.5 gains the POWDER FM SEI numbers; §B.4 rewritten as
  a per-baseline reproduction audit; §C.2 items marked done.
- **Tests** (all torch-gated tests SKIP cleanly in the dep-free venv; validated on CPU torch locally):
  `test_wisig_cnn_paper.py` (flatten-dim=256, L2-on-Dense-only, scale invariance), `test_oracle_cnn.py`,
  `test_complex_cnn.py` (ComplexConv1d complex-multiply + modReLU phase/threshold), `test_resnet1d_sei.py`,
  `test_training_sei.py` (end-to-end learns + emits a schema-valid track-tagged result; class-weight math;
  regime guard), plus POWDER prepare + `balanced_accuracy` dep-free tests. `ruff`/`black`/`mypy` green;
  dep-free `pytest -q` green. **Cluster runs pending** (WiSig `ManyTx.pkl` present; `slurm/train_sei_arm.sh`).

### Fixed — LWM-Spectro FM wrapper made faithful to the real weights (WP-62 verification)

Ground-truthed the committed LWM-Spectro integration against the real HF repo `wi-lab/lwm-spectro`
(`config.json`, `pretraining/pretrained_model.py`, `utils.py`) and fixed a chain of fidelity bugs
that made the encoder run **partially random**. The prior board row
(`leaderboard/results/amc/lwm-spectro-linear_probe.json`, `accuracy_overall=0.2274`) was produced by
that broken encoder and is **removed** — it must not stand as the hub's first FM-vs-baseline line.

- **Encoder source (found via on-cluster tensor inspection).** The wrapper loaded
  `checkpoints/checkpoint.pth` as the encoder, but that file is the `snr_mobility` **MoE bundle**
  (router + classifier + expert list, no encoder tensors). The real 12-layer LWM encoders are the
  per-protocol **expert** files `experts/{WiFi,LTE,5G}_expert.pth` (203-tensor state_dicts, keys
  `module.`-prefixed). The wrapper now loads one expert (default `WiFi`, `expert=` selectable); the
  load guard below is what surfaced the mismatch (all 201 keys missing).
- **Token width — 16, not 32.** The expert weights prove `embedding.proj` = `Linear(16, 128)` and
  `decoder_bias` = `(16,)`: the spectrogram is **single-channel** (a 4×4 patch = 16 values), NOT the
  real/imag-interleaved 32-wide layout previously assumed. `ELEMENT_LENGTH` 32→16, interleave removed.
- **FATAL — custom `LayerNormalization`.** Upstream every norm is a custom module storing
  `.alpha`/`.bias`, NOT `nn.LayerNorm`'s `.weight`/`.bias`. The reconstruction used `nn.LayerNorm`, so
  all 25 norm layers (50 tensors) silently failed to load and stayed at random init. Reimplemented
  `LayerNormalization(alpha, bias)`; the real checkpoint now loads by name.
- **Forward numerics.** MHA adds its residual internally (`residual + linear(attn)`); the FFN uses
  **ReLU** (was GELU); the block is post-norm `norm1(mha(x))` → `norm2(a + ffn(a))`.
- **Representation.** The frozen embedding is now the **mean over the sequence** of the raw encoder
  output (upstream "pooling mean"), taken BEFORE the top-level `norm`/`linear` (those run only in the
  masked-reconstruction branch; still defined so their keys load) — was `CLS[:,0]` through a spurious
  extra norm.
- **Tokenisation.** The adapter now builds a **single-channel log-magnitude (dB)** 128×128
  spectrogram (was a complex STFT with a magnitude `abs`/`polar`/`angle` normalisation that fed
  off-distribution tokens), 4×4-patched to `(1024, 16)`; `[CLS]` is the upstream constant **0.2**
  vector (was zeros); per-sample `(x-mean)/std` normalisation matches the upstream `mean_db`/`std_db`.
- **Load guard.** `_load_weights` now RAISES if any encoder key is missing when a checkpoint is
  present (refuses to score a partly-random encoder as "pretrained"); the exact bug class is now
  CI-catchable, and unexpected recon-head keys are INFO-logged.
- **Preprocessing honesty.** Upstream ships **no** IQ→spectrogram code (128×128 float16 spectrograms
  are pre-computed externally; the exact 512-FFT recipe is unpublished), so the IQ→STFT front-end is a
  best-effort approximation. `embed()` emits a loud one-time **UNVERIFIED** warning; any FM score is
  **provisional** until the upstream generation config is confirmed.
- **Regimes / SLURM.** `slurm/eval_fm_arm.sh` now handles `few_shot` (K as 3rd arg; the previous
  `RegimeSpec(regime)` crashed with no `k_shot`) and **refuses** `from_scratch`/`full_finetune` (a raw
  `forward` = untrained head ≈ chance; a real `full_finetune` needs a training loop — deferred).
  `linear_probe` (the validated chain) stays the default. `_download_lwm_spectro.py` now treats the
  `experts/*_expert.pth` files as the encoder backbone (the MoE bundle is fetched for completeness).
- **License.** Corrected `docs/BIBLIOGRAPHY.md` (4 mentions): LWM-Spectro is **MIT** (declared in
  `pyproject.toml`/`README_model.md`; no LICENSE file ships), NOT CC BY-NC-SA — publishing scores is
  permitted (we never redistribute weights). Consistent with `docs/SOTA_REFERENCE.md` "verify".
- **Tests.** New torch-gated regression guards in `tests/test_foundation_fm.py`: the encoder exposes
  the custom `.alpha`/`.bias` norm keys (not `.weight`/`.bias`); the adapter yields `(B, 1025, 16)`
  with a constant-0.2 CLS row; a non-matching checkpoint raises. Dep-free suite stays green;
  `ruff`/`black --line-length 100` clean.

### Fixed — CLDNN chance-collapse root-caused (CLDNN-scoped fix: per-sample input normalization)

- **Root cause (multi-agent workflow → per-epoch cluster diagnostic).** The earlier grad-clip "fix"
  only masked the NaN *symptom*; CLDNN still pinned at chance (0.0909). The real cause is a
  **CLDNN-specific input-conditioning fragility**: RadioML 2016.10a is ~unit-average-power, so raw
  per-sample IQ is tiny (RMS ~1e-2), and CLDNN has **no input normalization and no BatchNorm** —
  that near-zero-scale signal, fed through the conv front end AND (via the raw-waveform skip)
  straight into the 3-layer stacked LSTM, lets the deep recurrence collapse to a constant-class
  output **for some weight-init draws**. The diagnostic showed the un-normalized model **collapsed
  on the board's unseeded init yet learns on seed 42** — a fragility, not a deterministic bug.
  ResNet hit the *identical* exact-1/11 collapse earlier and was cured by exactly this normalization.
- **Fix — one change inside `rfbench/models/baselines/cldnn.py`** (cannot touch MCLDNN/ResNet;
  `training.py` recipe byte-for-byte unchanged): **per-sample unit-variance input normalization**
  (`_unit_variance_normalize`, the same transform `resnet_amc` uses) at the top of
  `CLDNNNet._conv_sequence`, before the conv **and** the raw skip, so both see ~unit-scale IQ. With
  a real input scale the LSTM cannot ignore the (tiny) input, so it learns robustly regardless of
  the init draw. Gated by `input_norm` (default **True**) → `MODELS.get("cldnn")()` builds the fixed
  model with no CLI change; `input_norm=False` reproduces the fragile config for ablation.
- **Diagnostic-driven (this is why we ran the short job first).** The per-epoch diagnostic
  (`slurm/diagnose_cldnn.py`, job 86194, seed 42, 20 ep) compared four variants and **overturned
  the workflow's proposed second half**: `broken` 0.5659 · **`norm` 0.5848** · `init` (forget-bias-1
  + orthogonal LSTM re-init, no norm) **0.0909 — collapsed** · `norm_init` 0.5848. So normalization
  is **necessary and sufficient**, and the LSTM re-init is **inert with norm and actively harmful
  without it** (the deep LSTM ignores the tiny input) — it was therefore **dropped** from the model.
  It logs, per epoch, val-accuracy · LR · pre-clip grad-norm · clip-bite · prediction entropy /
  top-class fraction · conv & LSTM activation std; a `--seed` sweep confirms `norm` is init-robust
  before the 150-epoch retrain (`slurm/retrain_cldnn_arm.sh`).
- **Follow-up flagged (not in this fix):** `training.py` val-accuracy checkpoint selection with
  `best_acc=-1.0` silently reports the untrained epoch-0 snapshot for a run that never beats chance —
  a robustness gap (not the CLDNN root cause) worth hardening separately.
- **Seed-robustness confirmed** before the long retrain (job 86196, 4 seeds × 12 ep): `norm` scores
  0.5631 / 0.5650 / 0.5690 / 0.5665 — tight and always ≫ 0.50 — while un-normalized `broken` swings
  0.4978–0.5400 and **collapses to 0.1275 on seed 123**, directly demonstrating the init fragility
  the normalization removes.
- **Board updated:** CLDNN re-trained from scratch (RadioML 2016.10a, seed 42, 150 epochs, final
  recipe) → **accuracy_overall 0.5805** (`leaderboard/results/amc/cldnn.json`, schema-valid +
  PR-ready; 440 907 params, 1× GB200), the first honest figure for the paper-faithful 3-LSTM+skip
  CLDNN under the final recipe (the prior 0.5876 was a superseded 2-LSTM/no-skip arch). MCLDNN
  (0.6171) / ResNet (0.5661) untouched. Tests (`tests/test_cldnn.py`: normalization applied on the
  default path, raw-skip identity under `input_norm=False`) + `ruff`/`black`/`mypy` green.

### Changed — BIBLIOGRAPHY.md refreshed to the current board (post-recipe-fix)

- **"Our score" values updated** to the live `leaderboard/results/**`: MCLDNN 60.08 → **61.71**
  (now above the paper's 61.01), ResNet 56.06 → **56.61**; CLDNN → **58.05** (final recipe,
  paper-faithful arch — the collapse noted at the time of this refresh was since root-caused and
  fixed, see the CLDNN entry above). Header convention
  block now describes the **fixed 2026-06 recipe** (val-accuracy checkpoint, ReduceLROnPlateau,
  early stop, grad clip 5.0) instead of the old fixed-epoch recipe.
- **Part B audit re-scoped as historical**: banners added to B.1–B.4 stating which mismatches were
  resolved by the 2026-06 paper-conformance pass (MCLDNN concat fusion + dropout head, CLDNN skip +
  3rd LSTM, ResNet unit-var norm + AlphaDropout + 2-dense head) and what stays open (CLDNN
  collapse; `wisig_cnn` still 1-D vs paper 2-D). Audit summary rewritten as a post-fix status.
- **Fabricated-row mentions updated**: the SEI 0.9412 / iqfm 0.7734 / mislabeled XCiT rows are
  recorded as **removed from the board** (`a689e86`) in A.3, A.5 and DOWNSTREAM_TASKS (the board
  currently has no SEI rows).

### Added — RFSS (arXiv:2604.00398) mined into the bibliography; `source_separation` candidate task

- **`docs/BIBLIOGRAPHY.md` §A.6 + §C.4**: RFSS (Chen/Jin/Tan, 2026-04 — v2 of arXiv:2508.12106,
  cite the 2026 id) — first public blind multi-source RF separation corpus (100k mixtures, 2–4
  sources, GSM/UMTS/LTE/5G NR, 3GPP TDL + 5 hardware impairments, 103 GB HDF5, official 70/15/15
  index split). Benchmarks table (Conv-TasNet best, −12.34 dB co-channel PI-SI-SINR 2-src);
  co-channel is the honest metric (adjacent-channel has a ~−28 dB evaluation-floor artifact).
  Availability: **not released as of 2026-07-03** (HF release announced in the paper only) — track blocked until it lands. Related refs added:
  RF Challenge (arXiv:2409.08839, interference cancellation, real OTA), Conv-TasNet, DPRNN,
  SI-SNR (Le Roux 2019), RF Transformer (arXiv:2603.09201, unscreened). Former §C.4 (FMs) → §C.5.
- **`docs/DOWNSTREAM_TASKS.md`**: new canonical id `source_separation` (taxonomy + coverage matrix
  + P3 section, RFSS as recommended dataset/protocol/metric); `interference_id` /
  `protocol_tech_id` statuses fixed ABSENT → EXISTS (implemented 2026-06); RFSS `rfss_single.h5`
  noted as a candidate 2nd `protocol_tech_id` dataset (cellular standards).

### Added — educational content on the leaderboard site (data-driven)

- **Enriched task manifest** (`leaderboard/tasks.json`): each task now merges optional
  educational fields alongside the existing `id/title/status/priority/blurb` —
  `description` (what/why), a `dataset` card object (`name`, `source`, `n_classes`,
  `modality`, `real_or_synthetic`, `conditions`, `license`, `split`), a `primary_metric`
  (`{name, definition}`) and a `secondary_metrics` list. All new fields are optional; the
  `$comment` documents the shape.
- **Per-task explanatory header** (`leaderboard/site/generate.py`): every task page — full
  leaderboard AND minimal WIP/planned page — is topped by a manifest-driven header
  (`_render_task_header`): the description, a compact dataset card and the primary +
  secondary metric definitions. Purely additive and generic (`DeclaredTask` extended with
  the optional fields, parsed in `load_manifest` via `_parse_metric_def`/`_parse_dataset`);
  a task missing any piece simply omits it, and an undeclared-but-has-results task renders
  no header — the existing generic per-metric/per-regime/WIP rendering is untouched.
- **Guide page** (`guide.html`, `render_guide`): renders the shared educational content
  (embedded `_GUIDE` constant) — a "What is I/Q?" section, the four evaluation regimes,
  verified-vs-self_reported, the data policy, the split policy, and a metrics glossary
  (name + definition + an up/down arrow for higher/lower-is-better). Linked as a **Guide**
  nav chip on every page (nav-chip mechanism extended; chip goes active on the Guide page).
- **Tests** (`tests/test_site.py`): asserts the Guide page is written with the I/Q section +
  metrics glossary (both arrow directions), a task page carries its dataset card + metric
  definitions above the tables, WIP pages still render the header, the header is omitted for
  undeclared tasks, and the manifest's educational fields load (and stay optional). All 16+
  existing site tests kept passing. `ruff`/`black --line-length 100`/`mypy` clean.

### Added — `protocol_tech_id` task (WiFi 802.11 standard recognition, P2)

- **New downstream task** `protocol_tech_id`: single-label closed-set classification of a raw-IQ
  window into 4 IEEE 802.11 standards (`802.11b`, `802.11g`, `802.11n`, `802.11ax`). Mirrors the
  AMC / interference_id skeleton exactly. `rfbench/tasks/protocol_tech_id/`
  (`ProtocolTechIdTask` registered `protocol_tech_id`), primary `accuracy_overall` + `macro_f1`
  (single-label classification metrics reused from AMC; the primary metric mirrors AMC's minus the
  SNR `eval_conditions` so `eval.conditions` stays clean). `configs/task/protocol_tech_id.yaml`.
  Distinct from `amc`: recognises the WiFi *standard*, not the modulation scheme.
- **Dataset** `tprime_wifi4` (split id base `proto-tprime-wifi4-8010-seed42-v1`): T-PRIME OTA WiFi
  set (Genesys Lab / Northeastern; paper arXiv:2401.04837, code github.com/genesys-neu/t-prime,
  data on Northeastern DRS collection `neu:h989s847q`) — **real over-the-air** raw interleaved-IQ
  `.bin` captures, 4 classes, ~66 GB, 20 MHz. No official split ships in the repo → 80/10/10
  stratified by class, seed 42. `rfbench/data/prepare/protocol.py` +
  `rfbench/data/download/protocol_tprime.py` (heavy deps lazy, `$RFBENCH_CACHE`, split indices +
  checksums only — never raw IQ, D3). **License**: DRS is openly downloadable but the dataset's
  redistribution terms are **unstated** (flagged in the dataset card). **Cluster-confirm TODOs**:
  the direct DRS artifact URL (item-specific, pass `source_url=`) and the exact `.bin` dtype /
  window tiling.
- **Baseline** `tprime` (`rfbench/models/baselines/tprime.py`): the T-PRIME transformer over raw
  interleaved IQ with NO learned input embedding — a `(2, N)` window sliced into `M` tokens of
  `(2, S)`, each flattened to a `1×2S` token fed to a 2-layer transformer encoder. Default **SM**
  (`M=24`, `S=64`, `N=1536`, ~1.6M params); **LG** (`M=64`, `S=128`, `N=8192`, ~6.8M) via
  `variant="LG"` / `model.variant=LG`. Registered + CLI-reachable
  (`--task protocol_tech_id --model tprime`). Cites T-PRIME (arXiv:2401.04837).
- **FROZEN-CONTRACT edit (reviewed)**: added `"protocol_tech_id"` to the `task.name` enum in
  BOTH `schemas/result.schema.json` and `schemas/submission.schema.json` (+ the `result_path`
  pattern), mirrored in `rfbench/core/types.py` `TaskName`. `schema_version` stays `1.0.0`; the
  new task owns `version: v1`.
- **Docs/site/CLI**: `docs/EVALUATION_PROTOCOL.md` §protocol_tech_id (normative), `TASK_TITLES`
  /`TASK_ORDER` in the site generator, CLI enum tables + prepare/download dispatch + `tprime`
  model module.
- **Tests**: `tests/test_task_protocol_tech_id.py` (dep-free metric/registry/end-to-end +
  numpy-guarded index-alignment regression) and `tests/test_tprime.py` (torch-gated). Both
  skip cleanly in the dep-free venv.

### Added — `interference_id` task (GNSS jamming classification, P2)

- **New downstream task** `interference_id`: single-label closed-set classification of a raw-IQ
  window into 6 GNSS-jamming classes (`DME`, `narrowband`, `single_am`, `single_chirp`,
  `single_fm`, `no_jamming`). Mirrors the AMC skeleton exactly. `rfbench/tasks/interference_id/`
  (`InterferenceIdTask` registered `interference_id`), primary `accuracy_overall` + `macro_f1`
  (single-label classification metrics reused from AMC; the primary metric mirrors AMC's minus the
  SNR `eval_conditions` so `eval.conditions` stays clean). `configs/task/interference_id.yaml`.
- **Dataset** `interf_gnss6` (split id base `interf-gnss6-8010-seed42-v1`): Swinney & Woods 2021
  raw-IQ set (Zenodo record 4629685, DOI 10.5281/zenodo.4629685, CC-BY-4.0,
  `Raw_IQ_Dataset.zip` ~1.9 GB, no login). 80/10/10 stratified by class, seed 42.
  `rfbench/data/prepare/interference.py` + `rfbench/data/download/interference_gnss.py` (heavy deps
  lazy, `$RFBENCH_CACHE`, split indices + checksums only — never raw data, D3). **Honesty**: the
  signals are MATLAB-synthesised but distributed as a downloadable raw-IQ archive, so this is a
  public-download dataset, not a generation-only blocker.
- **Baseline** `interf_cnn` (`rfbench/models/baselines/interf_cnn.py`): compact 1-D IQ CNN over
  `(2, L)` windows (conv-BN-ReLU blocks + global pool + linear head), registered + CLI-reachable
  (`--task interference_id --model interf_cnn`). Cites Morales-Ferre et al. 2019 and
  Swinney & Woods 2021 as literature SOTA.
- **FROZEN-CONTRACT edit (reviewed)**: added `"interference_id"` to the `task.name` enum in
  BOTH `schemas/result.schema.json` and `schemas/submission.schema.json` (+ the `result_path`
  pattern), mirrored in `rfbench/core/types.py` `TaskName`. `schema_version` stays `1.0.0`; the
  new task owns `version: v1`.
- **Docs/site/CLI**: `docs/EVALUATION_PROTOCOL.md` §interference_id (normative), `TASK_TITLES`
  /`TASK_ORDER` in the site generator, CLI enum tables + prepare/download dispatch.
- **Tests**: `tests/test_task_interference_id.py` (dep-free metric/registry/end-to-end +
  numpy-guarded index-alignment regression) and `tests/test_interf_cnn.py` (torch-gated). Both
  skip cleanly in the dep-free venv.

### Changed — AMC board updated with the final-recipe retrain (MCLDNN, ResNet)

- Re-trained from scratch (RadioML 2016.10a, seed 42, 150 epochs) under the fixed recipe
  (val-accuracy best-checkpoint + gradient clipping + paper-exact archs): **MCLDNN 0.6008 → 0.6171**
  and **ResNet 0.5606 → 0.5661**. Both now exceed their prior board scores.
- **~~KNOWN ISSUE — CLDNN collapses to chance (0.0909)~~ RESOLVED** (see the top "Fixed — CLDNN
  chance-collapse" entry): root-caused to a CLDNN input-conditioning fragility (tiny raw IQ + no
  input normalization into the 3-LSTM stack collapses for some init draws) and fixed with per-sample
  unit-variance input normalization inside `cldnn.py`. CLDNN re-trained from scratch under the same
  final recipe now scores **0.5805** (paper-faithful 3-LSTM+skip arch; the prior 0.5876 board figure
  was a superseded 2-LSTM/no-skip architecture). MCLDNN/ResNet unchanged.

### Added — Downstream-task prioritization mined from the FM bibliography

- `docs/DOWNSTREAM_TASKS.md`: mined the 9 foundation-model papers in `docs/BIBLIOGRAPHY.md` for the
  downstream tasks each one actually evaluates, normalized synonyms into a canonical taxonomy, and
  bucketed by FM coverage — **P1** (>= 2 FM papers): `amc`, `beam_prediction`, `sei`,
  `direction_finding`, `los_nlos`, `positioning`, `har`; **P2** (1 paper): `interference_id`,
  `protocol_tech_id`, `channel_estimation`, `snr_mobility_recognition`, `wideband_detection`;
  **P3** (defined benchmark track, 0 FM papers): `spectrum_sensing`. Each task carries its FM
  evaluators + datasets/metrics, current rfbench status, a recommended canonical dataset/protocol/
  metric, and scope-fit (IQ-signal vs a proposed separate CSI/6G-sensing track).

### Fixed — CLDNN training divergence (gradient clipping)

- The 150-epoch retrain collapsed CLDNN to chance (0.0909): its 3 stacked LSTMs explode at lr=1e-3
  over the longer schedule (the new recipe holds the LR higher for longer), and `argmax(NaN)`
  predicts a constant class. Added a global gradient-norm cap (`DEFAULT_GRAD_CLIP=5.0`, applied in
  `_train_one_epoch`) that stabilises the recurrent baselines without biting the CNN baselines, plus
  a NaN-loss guard that logs an ERROR and stops early, keeping the best checkpoint. MCLDNN/ResNet are
  unaffected (clip does not trigger for them). Baselines to be re-trained under the final recipe.

### Changed — AMC baseline paper-conformance + training-recipe fix (M3)

- **Regression root-caused** (4-way audit + adversarial verification): the `training.py` rewrite selected
  the best checkpoint on **validation loss**, whose minimum precedes the accuracy peak on RadioML, so it
  restored a suboptimal checkpoint and dragged every baseline down (CLDNN −8 pt). `train_baseline()` now
  selects/restores the best checkpoint on **validation accuracy** (same argmax/label convention as
  `core.evaluate`), keeps `ReduceLROnPlateau` on val loss, and early-stops on accuracy. Recipe loosened:
  patience 20→40, `min_delta` 1e-4→0, `lr_patience` 5→10, `min_lr` 1e-6→1e-7.
- **MCLDNN fusion made paper-exact**: element-wise add → channel-axis **concatenate** (`conv_fuse`
  in-channels 50→100, VALID padding → post-fusion length 124), matching the official `wzjialang/MCLDNN`.
- **ResNet depth adapted to the len-128 window**: `num_stacks` 6→3 (6 MaxPools over-pool 128→2; L=3 keeps
  the paper's ~16 final time steps, `flat_dim` 64→512), `alpha_dropout` 0.5→0.3.
- CLDNN left unchanged to isolate the recipe fix. Tests updated; suite green. Baselines to be re-trained
  from scratch (seed 42, 150 epochs) on the cluster to refresh the board.

### Changed — Leaderboard site redesign (generic, per-metric)

- **WP-50 rewrite.** `leaderboard/site/generate.py` is now fully data-driven: it renders **every** task
  (not just AMC), one `<task>.html` per task with results, and a **column or plot for every metric** —
  one table column per scalar metric (primary pinned first) and one inline `<svg>` line plot per curve
  metric (e.g. `accuracy_vs_snr`). Self-contained dark/light CSS, family chips, and
  `verified`/`self_reported` badges.
- **Protocol invariants enforced in markup.** One `<table data-regime data-track>` per distinct
  `(regime, track)` pair — two regimes never share a table, and same-regime different-track results split
  into separate tables. Rows sorted by the primary metric descending.
- `tests/test_site.py` rewritten (16 tests, mutation-checked non-trivial) against the new generic output;
  full suite green (342 passed, 29 skipped), `ruff`/`black`/`mypy` clean.

### Added — Real dataset loaders (M1, no generation)

Per the "use the datasets from the reference papers, do not generate" decision:

- **AMC.** RadioML 2016.10a (pickle, `opendata.deepsig.io`) + 2018.01a (HDF5) real loaders; **Sig53 is a
  reported blocker** (generation-only, no static release — not synthesised).
- **SEI.** Real loaders for WiSig (ManyTx), ORACLE (SigMF), LoRa RFFI (HDF5), each targeting the confirmed
  official source; credential-gated sources raise with manual-download instructions (no scraping).
- **Detection.** Adopted **RadDet** (ICASSP 2025, real published spectrogram + YOLO box annotations) as the
  wideband-detection dataset; **WBSig53 is a blocker** (generation-only). Protocol + task layer updated.
- Heavy deps (numpy/h5py/requests) stay lazy behind `rfbench[data]`; parsers tested on synthetic fixtures
  (real stdlib-pickle fixture for RadioML 2016; `importorskip` for HDF5). CLI wired to the real API.

### Added — Submission, publish, verify, FM wrappers (M5/M6)

- **WP-51/52 — Submission & publish CI.** `.github/workflows/validate-submission.yml`
  (`rfbench submit --check` + no-raw-data guard on PRs touching `leaderboard/results/**`) and
  `build-leaderboard.yml` (build the site → deploy to GitHub Pages on push to main); issue-form
  templates for submissions and task proposals.
- **WP-53 — Verification pipeline.** `rfbench/verify.py` + `rfbench verify`: checks manifest
  completeness and re-run metrics within tolerance, flips `verification.status → verified` with
  `verified_by/at/hardware`; `submit --check` strengthened to validate the submission manifest.
- **WP-60/61 — Foundation-model wrappers.** `rfbench/models/foundation/` generic wrapper (Model +
  `embed()`), a dependency-free example FM running in all four regimes via `evaluate()`, a copy-me
  template, and `docs/ADDING_A_MODEL.md`.
- **WP-50 fix — Leaderboard by track.** The board now separates by (task, track) as well as regime,
  so SEI (closed_set/cross_receiver/cross_day) and detection (detection/recognition) tracks are
  reported in separate tables.
- **Cluster recon.** `slurm/probe_torchsig*.sh`: `.[detection]` installs on aarch64 (torch 2.12+cu130,
  torchsig 2.1.1); torchsig 2.x replaced named WBSig53 with a config-driven wideband API — informs the
  real detection loader (still a lazy stub, pending a torchsig-1.x-vs-2.x decision).

### Added — Sprint 2 — Task adapters (M2)

- **WP-20 — AMC task.** `rfbench/tasks/amc/`: `AmcTask` (registered `amc`) + metrics
  `accuracy_overall` (primary, full SNR range), `accuracy_vs_snr` curve, `macro_f1` (pure-stdlib),
  dataset adapter, `configs/task/amc.yaml`.
- **WP-21 — SEI task.** `rfbench/tasks/sei/`: `SeiTask` (registered `sei`), tracks
  closed_set/cross_receiver/cross_day/open_set; `rank1_accuracy` (primary) + `auroc`/`eer` as
  separate metrics (pure-stdlib), `configs/task/sei.yaml`.
- **WP-22 — Wideband detection task.** `rfbench/tasks/wideband_detection/`: task + `mAP`/`mAR`/`IoU`
  (pure-stdlib IoU+AP path, lazy torchmetrics for production), detection vs recognition tracks kept
  distinct, `configs/task/wideband_detection.yaml`.
- **Cluster.** `slurm/setup_and_test_arm.sh`: CPU-only job that builds the `rfbench` venv on an ARM
  compute node and runs the full suite on aarch64 (proves the harness on the target arch).

### Added — Sprint 1 wave 2 — Data layer, leaderboard, CLI (M1/M2/M5, scaffolds)

Datasets are not redistributed and real `prepare` runs on the cluster (ARM venv, `rfbench[data]` /
`rfbench[detection]`); heavy deps are imported lazily and unit tests exercise the split path on
pure-stdlib synthetic fixtures.

- **WP-11 — AMC data (template).** `rfbench/data/prepare/{_common,amc}.py` + `download/amc_*.py`:
  RadioML → 80/10/10 stratified by (modulation × SNR); Sig53 → adopted official TorchSig split.
  `_common.py` (cache dir via `$RFBENCH_CACHE`, split+manifest helpers) is reused by SEI/detection.
- **WP-12 — SEI data.** `rfbench/data/prepare/sei.py` + `download/sei_*.py`: `closed_set`,
  `cross_receiver`, `cross_day` generated as separate grouped conditions with disjoint rx/day groups.
- **WP-13 — Detection data.** `rfbench/data/prepare/detection.py` + `download/detection_wbsig53.py`:
  split per policy + a T-F box annotations sidecar; detection vs recognition tracks kept distinct.
- **WP-50 — Leaderboard site.** `leaderboard/site/generate.py` (`build_site`): static HTML from
  `results/**.json`, sorted by primary metric, one table per regime (never mixed), verified/self_reported
  badges; seeded with sample result JSONs under `leaderboard/results/`.
- **WP-42 — CLI wiring.** `rfbench data prepare` / `eval` / `submit --check` / `leaderboard build` wired
  to the real implementations; heavy imports stay lazy so `import rfbench` and `rfbench --help` remain
  dependency-free.

### Added — Sprint 1 wave 1 — Splits & eval harness (M1/M4, partial)

- **Split policy (normative).** `docs/EVALUATION_PROTOCOL.md`: adopt an official/literature split when
  one exists (provenance recorded), otherwise a deterministic **80/10/10** stratified split, seed 42.
  Ratios + seed are part of `canonical_split_id`; this supersedes the earlier 60/20/20 AMC placeholder.
- **WP-10 — Deterministic splits.** `rfbench/core/splits.py`: `make_split` (pure-stdlib, seeded,
  stratified, default 80/10/10) → `SplitManifest`; `adopt_official_split` (pass-through), reproducible
  `write_split_index` + `split_checksum`. No generated indices committed (no data yet).
- **WP-40 — `evaluate()` + `result.json`.** `rfbench/core/evaluate.py`: the single canonical emitter;
  assembles a schema-valid `result.json` (regime declared verbatim, `verification.status=self_reported`),
  validated against `schemas/result.schema.json` via lazily-imported `jsonschema`.
- **WP-41 — Regime adapters.** `rfbench/regimes/` (`from_scratch`, `full_finetune`, `linear_probe`
  with a pure-Python nearest-centroid head, `few_shot(k)`) + `configs/config.yaml` and
  `configs/regime/*.yaml`. Dependency-free; real numerical heads deferred to M3/M6 behind extras.

### Added — Sprint 0 — Bootstrap & contracts (M0)

Scaffolds the repo, freezes the core contracts and JSON schemas, and lands the normative docs and
CI skeleton.

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
