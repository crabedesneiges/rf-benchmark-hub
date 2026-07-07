#!/bin/bash
# retrain_cldnn_arm.sh — Re-train the FIXED CLDNN AMC baseline from scratch on the REAL RadioML
# 2016.10a under the final recipe (150 epochs, seed 42, val-accuracy best-checkpoint, grad clip
# 5.0) and emit a schema-valid result.json under leaderboard/results/amc/cldnn.json (M3, WP-30).
#
# The CLDNN-scoped fix (per-sample input normalization + stabilised 3-LSTM init) is ON BY DEFAULT
# in rfbench/models/baselines/cldnn.py, so `rfbench train --model cldnn` picks it up with no CLI
# change. Launch ONLY after slurm/diagnose_cldnn_arm.sh confirms the fix lifts val-accuracy off
# chance on a short run. Target: CLDNN >= 0.50.
#
#   sbatch slurm/retrain_cldnn_arm.sh [EPOCHS]      # EPOCHS default 150
#
#SBATCH --job-name=cldnn_retrain
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_cldnn_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_cldnn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub-cldnn"   # the fix/cldnn-collapse worktree
VENV="$WORK/envs/rfbench-arm-gpu"             # .[dev,data,tasks,torch] — torch + CUDA present
UV="$WORK/envs/uv-arm/uv"
EPOCHS="${1:-150}"
DATASET="radioml_2016_10a"
REGIME="from_scratch"
OUT="$REPO/leaderboard/results/amc/cldnn.json"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"          # reuse the prebuilt GPU venv (no re-sync)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
# The GPU venv is editable-installed against the MAIN repo, so force this worktree's rfbench onto
# the path (PYTHONPATH overrides the editable .pth) -- otherwise the OLD, unfixed cldnn.py loads.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) model=cldnn epochs=$EPOCHS date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
mkdir -p "$(dirname "$OUT")"

# Pre-flight: confirm the FIXED worktree code (not the main-repo editable install) is loaded.
"$UV" run --no-sync python -c "import rfbench, inspect; \
from rfbench.models.baselines.cldnn import CLDNNNet; \
assert 'input_norm' in inspect.signature(CLDNNNet.__init__).parameters, 'FIX NOT LOADED: '+rfbench.__file__; \
print('rfbench (fixed) =', rfbench.__file__)" || { echo "PREFLIGHT FAILED: wrong rfbench"; exit 5; }

echo "=== rfbench train cldnn on $DATASET ($REGIME, $EPOCHS epochs, seed 42) ==="
"$UV" run --no-sync rfbench train \
  --task amc \
  --dataset "$DATASET" \
  --model cldnn \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --seed 42 \
  --device cuda \
  --out "$OUT"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — cldnn retrained; result.json at $OUT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
