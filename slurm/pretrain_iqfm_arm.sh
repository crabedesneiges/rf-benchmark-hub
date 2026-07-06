#!/bin/bash
# pretrain_iqfm_arm.sh — SimCLR / InfoNCE pre-training of the IQFM ShuffleNetV2-x0.5 raw-IQ
# backbone on the RadioML 2016.10a TRAIN split (delabelised, seed 42), on an ARM GB200 node.
# Saves the backbone to $RFBENCH_CACHE/iqfm/iqfm_shufflenet1d_simclr.pth for the "iqfm-base"
# wrapper to load and probe (slurm/eval_fm_arm.sh iqfm-base linear_probe).
#
# HONESTY: this is IN-DISTRIBUTION pre-training on RadioML-train, NOT the paper's OOD OTA-testbed
# setting. IQFM does not publish weights; the resulting score is OURS, not the paper's 38.1%.
#
# PREREQUISITE (run once, not in this job): RadioML 2016.10a downloaded + prepared:
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a
#
# Usage: sbatch slurm/pretrain_iqfm_arm.sh [EPOCHS] [BATCH_SIZE]
#   sbatch slurm/pretrain_iqfm_arm.sh              # 100 epochs, batch 512
#   sbatch slurm/pretrain_iqfm_arm.sh 200 512
#SBATCH --job-name=rfbench_pretrain_iqfm
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_pretrain_iqfm_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_pretrain_iqfm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm-gpu"          # .[dev,data,tasks,torch] — torch + CUDA present
EPOCHS="${1:-100}"
BATCH_SIZE="${2:-512}"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"

echo "=== node=$(hostname) arch=$(uname -m) epochs=$EPOCHS batch=$BATCH_SIZE date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"

"$VENV/bin/python" scripts/pretrain/iqfm_simclr.py \
  --dataset radioml_2016_10a \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-3 \
  --temperature 0.2 \
  --seed 42 \
  --device cuda
rc=$?

CKPT="$RFBENCH_CACHE/iqfm/iqfm_shufflenet1d_simclr.pth"
echo "=================================================="
if [ "$rc" -eq 0 ] && [ -f "$CKPT" ]; then
  echo "RESULT: SUCCESS — IQFM backbone pre-trained -> $CKPT"
  echo "NEXT: sbatch slurm/eval_fm_arm.sh iqfm-base linear_probe   # emits leaderboard row"
else
  echo "RESULT: PRETRAIN FAILED (rc=$rc)"
fi
exit "$rc"
