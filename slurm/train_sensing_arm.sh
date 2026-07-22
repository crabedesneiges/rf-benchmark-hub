#!/bin/bash
# train_sensing_arm.sh — Train the DeepSense CNN spectrum-sensing baseline FROM SCRATCH on the REAL
# DeepSense LTE-M multi-label occupancy set, scored + emitted as a schema-valid result.json under
# leaderboard/results/spectrum_sensing/ (M3).
#
# Uses the sensing-specific MULTI-LABEL loop (rfbench.training_sensing via `rfbench sensing-train`):
# BCEWithLogitsLoss over the 16 LTE-M sub-bands, Adam, best checkpoint + early stop on validation
# MICRO-F1. The AMC/SEI/SNR loops are untouched. The DeepSense split indices
# (sensing-deepsense-official-v1) are already committed, so NO data-prepare step is needed here —
# only the extracted lte_m/*.h5 tree must be present under $RFBENCH_CACHE/deepsense/ (manual
# download; the dataset license is UNSTATED, do NOT re-host).
#
#   sbatch slurm/train_sensing_arm.sh [MODEL] [EPOCHS]
#     MODEL   default deepsense_cnn
#     EPOCHS  default 150 (paper). Use a SMALL value (e.g. 3) for a quick validation run first.
#   e.g. VALIDATION: sbatch slurm/train_sensing_arm.sh deepsense_cnn 3
#        FULL:       sbatch slurm/train_sensing_arm.sh deepsense_cnn 150
#
#SBATCH --job-name=sensing_train
#SBATCH --output=logs/rfbench_sensing_%j.out
#SBATCH --error=logs/rfbench_sensing_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK              Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_REPO      repo/worktree checkout to run   (default: $WORK/projets/rf-benchmark-hub).
#   RFBENCH_VENV_GPU  GPU venv .[dev,data,tasks,torch] (default: $WORK/envs/rfbench-arm-gpu).
#   RFBENCH_UV        uv binary for this arch          (default: $WORK/envs/uv-arm/uv).
#   RFBENCH_CACHE     dataset cache root               (default: $WORK/data/rfbench_cache).
# SLURM logs go to logs/ relative to the submit dir: create it first (mkdir -p logs).
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"   # .[dev,data,tasks,torch] — torch + CUDA present
UV="${RFBENCH_UV:-$WORK/envs/uv-arm/uv}"
MODEL="${1:-deepsense_cnn}"
EPOCHS="${2:-150}"

export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"           # reuse the prebuilt GPU venv (no re-sync)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
# The GPU venv is editable-installed against the MAIN repo, so force this worktree's rfbench onto
# the path (PYTHONPATH overrides the editable .pth) — otherwise the sensing code below never loads.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# DeepSense recipe (paper): Adam lr=1e-3, batch 256, 150 ep, BCE. Early stop on val micro-F1.
LR="1e-3"; BATCH=256; PATIENCE=15

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE  recipe: lr=$LR batch=$BATCH patience=$PATIENCE"

# Pre-flight: confirm THIS worktree's rfbench (with the DeepSense model + sensing-train) is loaded.
"$UV" run --no-sync python -c "
import rfbench, importlib
from rfbench.core.registry import MODELS
importlib.import_module('rfbench.models.baselines.$MODEL')
assert '$MODEL' in MODELS, 'model not registered: $MODEL'
import rfbench.training_sensing  # sensing loop must exist
print('rfbench (sensing worktree) =', rfbench.__file__)
" || { echo "PREFLIGHT FAILED: wrong rfbench or missing sensing code"; exit 5; }

OUT="$REPO/leaderboard/results/spectrum_sensing/${MODEL}.json"
mkdir -p "$(dirname "$OUT")"
echo ""
echo "=== rfbench sensing-train $MODEL on deepsense ($EPOCHS ep, seed 42) ==="
"$UV" run --no-sync rfbench sensing-train \
  --dataset deepsense \
  --model "$MODEL" \
  --regime from_scratch \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --patience "$PATIENCE" \
  --seed 42 \
  --device cuda \
  --out "$OUT"
rc=$?
echo "--- emitted ---"; [ -f "$OUT" ] && head -c 1200 "$OUT"; echo ""

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL trained; result.json under leaderboard/results/spectrum_sensing/"
  echo "NOTE: this row is Tier-1 self_reported (NOT committed automatically) — review before submit."
else
  echo "RESULT: FAILED (rc=$rc)"
fi
exit "$rc"
