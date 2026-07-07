#!/bin/bash
# diagnose_cldnn_arm.sh — SHORT per-epoch CLDNN collapse diagnostic (fix/cldnn-collapse).
# Runs slurm/diagnose_cldnn.py for the broken vs candidate-fix variants on the REAL RadioML
# 2016.10a, logging per-epoch (val-acc, LR, pre-clip grad-norm, pred-entropy, activation std) so
# a single short GB200 job adjudicates the root cause AND confirms the fix lifts val-accuracy off
# chance BEFORE the 150-epoch retrain. Does NOT touch the board.
#
#   sbatch slurm/diagnose_cldnn_arm.sh [EPOCHS] [VARIANTS] [SEEDS]
#     EPOCHS   default 20   (short trajectory; the collapse/recovery shows within ~10 epochs)
#     VARIANTS default broken,norm,init,norm_init
#     SEEDS    default 42   (comma-list; loops each seed -> confirm `norm` is INIT-ROBUST before the
#              retrain, e.g. `sbatch slurm/diagnose_cldnn_arm.sh 12 norm 0,7,42,123`)
#
#SBATCH --job-name=cldnn_diag
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/cldnn_diag_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/cldnn_diag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:15:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub-cldnn"   # the fix/cldnn-collapse worktree
VENV="$WORK/envs/rfbench-arm-gpu"             # .[dev,data,tasks,torch] — torch + CUDA present
UV="$WORK/envs/uv-arm/uv"
EPOCHS="${1:-20}"
VARIANTS="${2:-broken,norm,init,norm_init}"
SEEDS="${3:-42}"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export UV_PROJECT_ENVIRONMENT="$VENV"          # reuse the prebuilt GPU venv (no re-sync)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
# The GPU venv is editable-installed against the MAIN repo, so force this worktree's rfbench onto
# the path (PYTHONPATH overrides the editable .pth) -- otherwise the OLD, unfixed cldnn.py loads.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) epochs=$EPOCHS variants=$VARIANTS seeds=$SEEDS date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"

# Pre-flight: confirm the FIXED worktree code (not the main-repo editable install) is loaded.
"$UV" run --no-sync python -c "import rfbench, inspect; \
from rfbench.models.baselines.cldnn import CLDNNNet; \
assert 'input_norm' in inspect.signature(CLDNNNet.__init__).parameters, 'FIX NOT LOADED: '+rfbench.__file__; \
print('rfbench (fixed) =', rfbench.__file__)" || { echo "PREFLIGHT FAILED: wrong rfbench"; exit 5; }

rc=0
for SEED in ${SEEDS//,/ }; do
  OUT="$WORK/logs/cldnn_diag_${SLURM_JOB_ID}_seed${SEED}.json"
  echo "=== diagnostic: variants=$VARIANTS seed=$SEED epochs=$EPOCHS -> $OUT ==="
  "$UV" run --no-sync python slurm/diagnose_cldnn.py \
    --variants "$VARIANTS" \
    --epochs "$EPOCHS" \
    --lr 1e-3 \
    --batch-size 256 \
    --seed "$SEED" \
    --device cuda \
    --cache "$RFBENCH_CACHE" \
    --out "$OUT"
  rc=$((rc + $?))
done

echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: diagnostic ran for seeds [$SEEDS]; per-epoch JSON(s) in $WORK/logs/cldnn_diag_${SLURM_JOB_ID}_seed*.json"
  echo "  -> 'broken' is init-fragile; 'norm' must clear chance and reach ~0.5 for EVERY seed."
else
  echo "RESULT: DIAGNOSTIC FAILED (aggregate rc=$rc)"
fi
exit "$rc"
