#!/bin/bash
# train_baseline_arm.sh — Train an AMC baseline from scratch on the REAL RadioML 2016.10a and emit
# a schema-valid result.json under the multi-seed staging area (M3, WP-30).
# The dataset must already be downloaded + prepared (split indices committed):
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a   # once
# then:
#   sbatch slurm/train_baseline_arm.sh [MODEL [EPOCHS [SEED]]]
#     MODEL  default: mcldnn  — baseline registry name (e.g. resnet_amc)
#     EPOCHS default: 50      — training epochs
#     SEED   default: 42      — RNG seed (use 42/43/44 for multi-seed runs)
#
# Output locations (no write to leaderboard/results/ by this script):
#   result.json  → $WORK/logs/multiseed/amc/<model>-seed<seed>.json
#   checkpoint   → $WORK/checkpoints/multiseed/<model>-seed<seed>.pt
#
#SBATCH --job-name=rfbench_train
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_mcldnn_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_mcldnn_%j.err
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
REPO="${SLURM_SUBMIT_DIR:-$WORK/projets/rf-benchmark-hub}"
# Strip trailing /slurm if sbatch was run from the slurm/ subdirectory.
REPO="${REPO%/slurm}"
VENV="$WORK/envs/rfbench-arm-gpu"          # .[dev,data,tasks,torch] — torch + CUDA present
MODEL="${1:-mcldnn}"                        # baseline registry name; override as $1
EPOCHS="${2:-50}"                          # small-but-real default; override as $2
SEED="${3:-42}"                            # RNG seed; override as $3 (use 42/43/44)
DATASET="radioml_2016_10a"
REGIME="from_scratch"

# Multi-seed staging — no direct write to leaderboard/results/
OUT_DIR="$WORK/logs/multiseed/amc"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"
CKPT_DIR="$WORK/checkpoints/multiseed"
CKPT="$CKPT_DIR/${MODEL}-seed${SEED}.pt"

# Garantir que le code du repo de soumission précède le .pth de l'install editable.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
echo "CKPT=$CKPT"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# Rename the SLURM log retroactivement pour inclure le seed (best-effort, non bloquant).
[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_train_${MODEL}_s${SEED}" 2>/dev/null || true

echo "=== rfbench train $MODEL on $DATASET ($REGIME, $EPOCHS epochs, seed $SEED) ==="
"$VENV/bin/rfbench" train \
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
  echo "RESULT: SUCCESS — $MODEL seed=$SEED trained; result.json at $OUT ; checkpoint at $CKPT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
