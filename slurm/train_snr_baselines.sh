#!/bin/bash
# train_snr_baselines.sh — Fit + evaluate an SNR-estimation regression baseline on the REAL
# RadioML 2016.10a split and emit a schema-valid result.json under the multi-seed staging area (J4).
# CPU-ONLY: mean_snr is pure stdlib, snr_moment_ridge is numpy + scikit-learn — no GPU.
#
# Deterministic at seed 42 (mean_snr is a constant; ridge is closed-form), so ONE run per model is
# the canonical row — no multi-seed sweep. The orchestrator promotes the staging file to the board.
#
# Models (registry names): mean_snr | snr_moment_ridge
#
# The dataset must already be downloaded + prepared (SNR split committed, derived from the AMC one):
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a   # once (shared with AMC)
# then:
#   sbatch slurm/train_snr_baselines.sh [MODEL]
#     MODEL  default: snr_moment_ridge  — one of mean_snr | snr_moment_ridge
#
# Output (no write to leaderboard/results/ by this script):
#   result.json → $WORK/logs/multiseed/snr_estimation/<model>-seed42.json
#
#SBATCH --job-name=rfbench_snr
#SBATCH --output=logs/rfbench_snr_%j.out
#SBATCH --error=logs/rfbench_snr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
# cluster mono-partition (defq*, ARM); CPU-only regression baseline, no --gres gpu, no --constraint.
# NB: Dalia forbids --mem (RAM auto-allocated proportional to cores).

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK                Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_REPO        repo/worktree checkout to run       (default: $WORK/projets/rf-benchmark-hub[...]).
#   RFBENCH_VENV_CPU    CPU venv  .[dev,data]               (default: $WORK/envs/rfbench-arm).
#   RFBENCH_VENV_GPU    GPU venv  .[dev,data,tasks,torch]   (default: $WORK/envs/rfbench-arm-gpu).
#   RFBENCH_VENV_DETECTION  detection venv .[dev,detection] (default: $WORK/envs/rfbench-arm-detection).
#   RFBENCH_UV          uv binary for this arch             (default: $WORK/envs/uv-arm/uv).
#   RFBENCH_CACHE       dataset cache root                  (default: $WORK/data/rfbench_cache).
# SLURM logs go to logs/ relative to the submit dir: create it first (mkdir -p logs) or
# override with `sbatch --output=... --error=...`.
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
# Derive REPO from SLURM_SUBMIT_DIR so the submitting worktree's code is used (override with
# RFBENCH_REPO); strip a trailing /slurm if sbatch was run from the slurm/ subdirectory.
REPO="${RFBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO="${REPO%/slurm}"
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"          # .[dev,data,tasks,torch] — numpy + sklearn present
MODEL="${1:-snr_moment_ridge}"             # mean_snr | snr_moment_ridge; override as $1
DATASET="radioml_2016_10a"
SEED=42                                     # deterministic baselines — a single seed-42 run suffices

OUT_DIR="$WORK/logs/multiseed/snr_estimation"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="CPU (DSP/regression baseline)"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
mkdir -p "$OUT_DIR"

[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_snr_${MODEL}" 2>/dev/null || true

echo "=== rfbench fit+eval $MODEL on snr_estimation/$DATASET (from_scratch, seed $SEED, full SNR range) ==="
MODEL="$MODEL" OUT="$OUT" DATASET="$DATASET" "$VENV/bin/python" - <<'PY'
import os

# Explicit imports register the J4 SNR baselines (@register_model) and the snr_estimation task.
import rfbench.models.baselines.snr_regressors  # noqa: F401  (registers mean_snr + snr_moment_ridge)
import rfbench.tasks.snr_estimation  # noqa: F401  (registers 'snr_estimation')
from rfbench.core.evaluate import evaluate
from rfbench.core.registry import MODELS, get_task

model_name = os.environ["MODEL"]
out = os.environ["OUT"]
dataset = os.environ["DATASET"]

task = get_task("snr_estimation")
model = MODELS.get(model_name)()
ds = next(d for d in task.datasets() if d.name == dataset)

print(f"[snr] {model_name} on snr_estimation/{ds.name} — fitting on train split...")
model.fit(ds.load("train"))
print("[snr] evaluating on test split (full SNR range, bootstrap CI on)...")
res = evaluate(model, task, "test", model.regime, dataset=ds.name, track="all_snr",
               device="cpu", out_path=out)
vals = res["metrics"]["values"]
print(f"RESULT-SNR rmse_db={vals.get('rmse_db')} mae_db={vals.get('mae_db')} -> {out}")
PY
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL seed=$SEED fit+evaluated; result.json at $OUT"
else
  echo "RESULT: FIT/EVAL FAILED (rc=$rc)"
fi
exit "$rc"
