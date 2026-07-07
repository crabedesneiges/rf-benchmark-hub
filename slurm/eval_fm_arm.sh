#!/bin/bash
# eval_fm_arm.sh — Evaluate a foundation model on AMC (RadioML 2016.10a, seed 42, FULL SNR range)
# on an ARM GB200 node, emitting a schema-valid result.json.
#
# ONLY the frozen-embedding regimes produce a meaningful FM row here:
#   * linear_probe (default) — fit a head on the frozen encoder embeddings (validated chain).
#   * few_shot K             — EPISODIC (EVALUATION_PROTOCOL.md: N=10 episodes, seeds 42..51):
#                              10 K-per-class draws, each evaluated on the full test split, then
#                              ONE aggregated row (mean + descriptive ±1 stdev, multi_seed_std).
#                              Delegated to slurm/eval_fm_episodic.py; K is the 3rd arg.
# from_scratch / full_finetune are REFUSED: they read model.forward(), which for lwm-spectro is a
# FRESH UNTRAINED head over the frozen encoder (~chance). A real full_finetune needs a separate
# training loop (not yet implemented) that fits + saves the head/encoder the wrapper then loads.
#
# PREREQUISITES (run once, not in this job):
#   1. Weights in $RFBENCH_CACHE (fetched with huggingface_hub — pure Python, OK on the frontend):
#        python -m rfbench.models.foundation._download_lwm_spectro   # needs rfbench[foundation]
#      -> $RFBENCH_CACHE/lwm-spectro/checkpoints/checkpoint.pth (+ moe + experts).
#   2. RadioML 2016.10a downloaded + prepared (split index committed):
#        sbatch slurm/download_prepare_arm.sh radioml_2016_10a
# The FM wrapper reads the cached checkpoint via torch.load (no huggingface_hub needed at eval time).
#
# NOTE: the IQ->STFT preprocessing is UNVERIFIED (upstream ships no IQ->spectrogram code; the exact
# 512-FFT recipe is unpublished). Any score from this job is PROVISIONAL and must not be published
# as a faithful LWM-Spectro figure until the upstream spectrogram-generation config is confirmed —
# embed() logs a loud warning to that effect.
#
# Usage: sbatch slurm/eval_fm_arm.sh [MODEL] [REGIME] [K_SHOT]
#   sbatch slurm/eval_fm_arm.sh                       # lwm-spectro linear_probe
#   sbatch slurm/eval_fm_arm.sh iqfm-base linear_probe   # rescore iqfm-base row (logreg head)
#   sbatch slurm/eval_fm_arm.sh iqfm-base few_shot 10    # episodic k=10 (seeds 42..51)
#SBATCH --job-name=rfbench_eval_fm
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_eval_fm_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_eval_fm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
# Derive REPO from the SLURM submit dir so the job ALWAYS imports the code of the worktree it
# was submitted from -- NOT whatever the ARM venv's editable .pth happens to point at (that
# .pth targets the canonical worktree, a pre-Phase-0 checkout). Falls back to $PWD when run
# outside SLURM (e.g. a manual `bash eval_fm_arm.sh` smoke test).
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
VENV="$WORK/envs/rfbench-arm-gpu"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
MODEL="${1:-lwm-spectro}"
REGIME="${2:-linear_probe}"
K_SHOT="${3:-}"
if [ "$REGIME" = "few_shot" ] && [ -n "$K_SHOT" ]; then
  OUT="$REPO/leaderboard/results/amc/${MODEL}-${REGIME}-k${K_SHOT}.json"
else
  OUT="$REPO/leaderboard/results/amc/${MODEL}-${REGIME}.json"
fi
# Per-episode few-shot staging (conventions): $WORK/logs/multiseed/<task>/k<K>/<model>-seed<seed>.json.
# Each k value gets its own sub-directory to avoid collisions when k=1/10/100 jobs run in parallel.
STAGING_BASE="$WORK/logs/multiseed/amc"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL regime=$REGIME k_shot=${K_SHOT:-n/a} date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
# PYTHONPATH must precede site-packages so `import rfbench` resolves to THIS repo, short-
# circuiting the editable-install .pth that points at the canonical (pre-Phase-0) worktree.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [ "$REGIME" = "few_shot" ]; then
  # Episodic few-shot (EVALUATION_PROTOCOL.md: N>=10 episodes, seeds 42..51): delegate to the
  # dedicated aggregator, which fits one FewShotAdapter per seed, evaluates each episode on the
  # FULL test split via evaluate() (the canonical result.json writer), stages a per-seed row
  # under $STAGING_DIR, then writes ONE aggregated board row (mean + descriptive +/-1 stdev,
  # method=multi_seed_std, n_episodes=10). linear_probe stays on the legacy heredoc path below.
  if [ -z "$K_SHOT" ]; then
    echo "[eval-fm] few_shot requires K_SHOT (3rd arg), e.g. \`sbatch ... few_shot 10\`"
    exit 2
  fi
  # Isolate staging files by k to prevent collisions when k=1/10/100 jobs run concurrently.
  STAGING_DIR="$STAGING_BASE/k${K_SHOT}"
  echo "[eval-fm] few_shot k=$K_SHOT: 10 episodes (seeds 42..51) -> aggregated row $OUT"
  echo "[eval-fm] staging dir: $STAGING_DIR"
  "$VENV/bin/python" "$REPO/slurm/eval_fm_episodic.py" \
    --model "$MODEL" \
    --k-shot "$K_SHOT" \
    --out "$OUT" \
    --staging-dir "$STAGING_DIR"
  rc=$?
else
  MODEL="$MODEL" REGIME="$REGIME" OUT="$OUT" "$VENV/bin/python" - <<'PY'
import os
import sys

import rfbench.models.foundation  # noqa: F401  (registers 'dummy-fm' + 'iqfm-base')
import rfbench.models.foundation.lwm_spectro  # noqa: F401  (registers 'lwm-spectro')
import rfbench.tasks.amc  # noqa: F401  (registers 'amc')
from rfbench.core.registry import MODELS, get_task
from rfbench.core.model import Regime, RegimeSpec
from rfbench.models.foundation.base import run_regime
from rfbench.core.evaluate import evaluate

model_name = os.environ["MODEL"]
regime = Regime(os.environ["REGIME"])
out = os.environ["OUT"]

if regime in (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE):
    sys.exit(
        f"[eval-fm] REFUSED: regime '{regime.value}' evaluates model.forward(), which for "
        "lwm-spectro is a FRESH UNTRAINED head over the frozen encoder (~chance). This job only "
        "produces meaningful rows for linear_probe / few_shot (frozen embeddings + fitted head). "
        "A real full_finetune needs a separate training loop that fits + saves the head/encoder."
    )
# few_shot is handled by slurm/eval_fm_episodic.py (episodic, N>=10) and never reaches here;
# this heredoc is the linear_probe path (single frozen-embedding head fit + evaluate).
spec = RegimeSpec(regime)

task = get_task("amc")
fm = MODELS.get(model_name)()
ds = next(d for d in task.datasets() if d.name == "radioml_2016_10a")
print(f"[eval-fm] {model_name} / {spec.name.value} on amc/{ds.name} — fitting head on train split...")
train = ds.load("train")
adapted = run_regime(fm, spec, train)
print("[eval-fm] evaluating on test split (full SNR range)...")
res = evaluate(adapted, task, "test", adapted.regime, dataset=ds.name, out_path=out)
vals = res["metrics"]["values"]
print(f"RESULT-FM acc_overall={vals.get('accuracy_overall')} macro_f1={vals.get('macro_f1')} -> {out}")
if model_name == "lwm-spectro":
    # Only lwm-spectro reconstructs an UNVERIFIED IQ->STFT front-end; iqfm/wireless-jepa are raw-IQ.
    print("[eval-fm] REMINDER: score is PROVISIONAL — IQ->STFT preprocessing is UNVERIFIED.")
PY
  rc=$?
fi
echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: SUCCESS — $MODEL $REGIME evaluated -> $OUT" || echo "RESULT: EVAL FAILED (rc=$rc)"
exit "$rc"
