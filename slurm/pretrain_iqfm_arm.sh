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
#SBATCH --output=logs/rfbench_pretrain_iqfm_%j.out
#SBATCH --error=logs/rfbench_pretrain_iqfm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

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
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"          # .[dev,data,tasks,torch] — torch + CUDA present
EPOCHS="${1:-100}"
BATCH_SIZE="${2:-512}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"

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
