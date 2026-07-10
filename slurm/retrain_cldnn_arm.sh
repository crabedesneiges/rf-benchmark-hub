#!/bin/bash
# retrain_cldnn_arm.sh — Re-train the FIXED CLDNN AMC baseline from scratch on the REAL RadioML
# 2016.10a under the final recipe (val-accuracy best-checkpoint, grad clip 5.0) and emit a
# schema-valid result.json under the multi-seed staging area (M3, WP-30).
#
# The CLDNN-scoped fix (per-sample input normalization + stabilised 3-LSTM init) is ON BY DEFAULT
# in rfbench/models/baselines/cldnn.py — `rfbench train --model cldnn` picks it up with no CLI
# change. Launch ONLY after slurm/diagnose_cldnn_arm.sh confirms the fix lifts val-accuracy off
# chance on a short run. Target: CLDNN >= 0.50.
#
#   sbatch slurm/retrain_cldnn_arm.sh [EPOCHS [SEED]]
#     EPOCHS default: 150 — training epochs
#     SEED   default: 42  — RNG seed (use 42/43/44 for multi-seed runs)
#
# Output locations (no direct write to leaderboard/results/ by this script):
#   result.json  → $WORK/logs/multiseed/amc/cldnn-seed<seed>.json
#   checkpoint   → $WORK/checkpoints/multiseed/cldnn-seed<seed>.pt
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
# Derive REPO from SLURM_SUBMIT_DIR so that the correct worktree's code is used regardless of
# which rfbench editable install is registered in the venv's .pth.
# NOTE: le worktree rf-benchmark-hub-cldnn n'est plus référencé ici — le fix est mergé dans la
#       branche principale, c'est le repo de soumission qui fait référence.
REPO="${SLURM_SUBMIT_DIR:-$WORK/projets/rf-benchmark-hub}"
# Strip trailing /slurm si sbatch lancé depuis le sous-répertoire slurm/.
REPO="${REPO%/slurm}"
VENV="$WORK/envs/rfbench-arm-gpu"             # .[dev,data,tasks,torch] — torch + CUDA present
UV="$WORK/envs/uv-arm/uv"
EPOCHS="${1:-150}"
SEED="${2:-42}"                               # RNG seed; override as $2 (use 42/43/44)
DATASET="radioml_2016_10a"
REGIME="from_scratch"
MODEL="cldnn"

# Multi-seed staging — no direct write to leaderboard/results/
OUT_DIR="$WORK/logs/multiseed/amc"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"
CKPT_DIR="$WORK/checkpoints/multiseed"
CKPT="$CKPT_DIR/${MODEL}-seed${SEED}.pt"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"          # reuse the prebuilt GPU venv (no re-sync)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
# Garantir que le code du repo de soumission précède le .pth de l'install editable.
# Cela court-circuite toute install editable pointant vers un autre worktree.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
echo "CKPT=$CKPT"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# Rename le job SLURM pour inclure le seed (best-effort, non bloquant).
[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="cldnn_retrain_s${SEED}" 2>/dev/null || true

# Pre-flight: confirm the correct (fixed) cldnn.py is loaded from this repo.
"$UV" run --no-sync python -c "import rfbench, inspect; \
from rfbench.models.baselines.cldnn import CLDNNNet; \
assert 'input_norm' in inspect.signature(CLDNNNet.__init__).parameters, 'FIX NOT LOADED: '+rfbench.__file__; \
print('rfbench (fixed) =', rfbench.__file__)" || { echo "PREFLIGHT FAILED: wrong rfbench"; exit 5; }

echo "=== rfbench train $MODEL on $DATASET ($REGIME, $EPOCHS epochs, seed $SEED) ==="
"$UV" run --no-sync rfbench train \
  --task amc \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --seed "$SEED" \
  --device cuda \
  --out-checkpoint "$CKPT" \
  --out "$OUT"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — cldnn seed=$SEED retrained; result.json at $OUT ; checkpoint at $CKPT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
