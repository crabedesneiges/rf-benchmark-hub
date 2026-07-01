#!/bin/bash
# train_baseline_arm.sh — Train the MCLDNN AMC baseline from scratch on the REAL RadioML
# 2016.10a and emit a schema-valid result.json under leaderboard/results/amc/ (M3, WP-30).
# Runs on an ARM compute node with 1 GB200 GPU, using the GPU venv built by setup_gpu_venv.sh.
#
# The dataset must already be downloaded + prepared (split indices committed):
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a   # once
# then:
#   sbatch slurm/train_baseline_arm.sh [EPOCHS]             # EPOCHS default: 50
#
#SBATCH --job-name=rfbench_train_mcldnn
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_mcldnn_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_train_mcldnn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm-gpu"          # .[dev,data,tasks,torch] — torch + CUDA present
EPOCHS="${1:-50}"                          # small-but-real default; override as $1
DATASET="radioml_2016_10a"
MODEL="mcldnn"
REGIME="from_scratch"
OUT="$REPO/leaderboard/results/amc/${MODEL}.json"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

echo "=== node=$(hostname) arch=$(uname -m) epochs=$EPOCHS date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
mkdir -p "$(dirname "$OUT")"

echo "=== rfbench train $MODEL on $DATASET ($REGIME, $EPOCHS epochs) ==="
"$VENV/bin/rfbench" train \
  --task amc \
  --dataset "$DATASET" \
  --model "$MODEL" \
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
  echo "RESULT: SUCCESS — $MODEL trained; result.json at $OUT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
