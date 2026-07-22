#!/bin/bash
# train_tldnn_arm.sh — Train the TLDNN AMC baseline from scratch on a REAL RadioML dataset and emit
# a schema-valid result.json under the multi-seed staging area (reimplementation target #1,
# docs/BIBLIOGRAPHY.md A.1). The dataset must already be downloaded + prepared (split indices
# committed):
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a   # once (and/or radioml_2018_01a)
# then:
#   sbatch slurm/train_tldnn_arm.sh [DATASET [EPOCHS [SEED [LR]]]]
#     DATASET default: radioml_2016_10a  — or radioml_2018_01a (24-class, window 1024, K=4)
#     EPOCHS  default: 150               — the paper's training budget
#     SEED    default: 42                — RNG seed (use 42/43/44 for multi-seed runs)
#     LR      default: 2e-4              — see the collapse note below (do NOT use 1e-3)
#
# lr note: TLDNN contains self-attention. The rfbench from-scratch loop is plain Adam with NO
# warmup, and a transformer-bearing model COLLAPSES at lr=1e-3 under it (documented lesson: tprime
# went 0.259->0.995 val). Train TLDNN at LR=2e-4, NOT the pure-CNN baselines' 1e-3.
#
# Output locations (no write to leaderboard/results/ by this script):
#   result.json  → $WORK/logs/multiseed/amc/tldnn-<dataset>-seed<seed>.json
#   checkpoint   → $WORK/checkpoints/multiseed/tldnn-<dataset>-seed<seed>.pt
#
#SBATCH --job-name=rfbench_tldnn
#SBATCH --output=logs/rfbench_train_tldnn_%j.out
#SBATCH --error=logs/rfbench_train_tldnn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise.

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK                Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_VENV_GPU    GPU venv  .[dev,data,tasks,torch]   (default: $WORK/envs/rfbench-arm-gpu).
#   RFBENCH_CACHE       dataset cache root                  (default: $WORK/data/rfbench_cache).
# SLURM logs go to logs/ relative to the submit dir: create it first (mkdir -p logs).
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
# Derive REPO from SLURM_SUBMIT_DIR so the correct worktree's code is used regardless of which
# rfbench editable install is registered in the venv's .pth.
REPO="${SLURM_SUBMIT_DIR:-$WORK/projets/rf-benchmark-hub}"
REPO="${REPO%/slurm}"
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"   # .[dev,data,tasks,torch] — torch + CUDA
DATASET="${1:-radioml_2016_10a}"
EPOCHS="${2:-150}"
SEED="${3:-42}"
LR="${4:-2e-4}"

# Multi-seed staging — no direct write to leaderboard/results/
OUT="$WORK/logs/multiseed/amc/tldnn-${DATASET}-seed${SEED}.json"
CKPT="$WORK/checkpoints/multiseed/tldnn-${DATASET}-seed${SEED}.pt"

# Ensure the submission repo's code precedes the editable-install .pth.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

# Exported for the Python driver (slurm/train_tldnn.py).
export DATASET EPOCHS SEED LR OUT CKPT

echo "=== node=$(hostname) arch=$(uname -m) dataset=$DATASET epochs=$EPOCHS seed=$SEED lr=$LR date=$(date -Is) ==="
echo "REPO=$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
echo "CKPT=$CKPT"

# Rename the SLURM log retroactively to include the dataset+seed (best-effort, non-blocking).
[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_tldnn_${DATASET}_s${SEED}" 2>/dev/null || true

echo "=== rfbench train tldnn on $DATASET (from_scratch, $EPOCHS epochs, seed $SEED, lr $LR) ==="
"$VENV/bin/python" "$REPO/slurm/train_tldnn.py"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — tldnn dataset=$DATASET seed=$SEED trained; result.json at $OUT ; checkpoint at $CKPT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
